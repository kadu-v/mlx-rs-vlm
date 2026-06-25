use std::path::Path;

use mlx_rs_vlm::config::{ProcessorConfig, Qwen35Configuration};
use mlx_rs_vlm::generate::{generate, GenerateParams};
use mlx_rs_vlm::hub::download_model;
use mlx_rs_vlm::model::Qwen35VLM;
use mlx_rs_vlm::weights::{load_weights, sanitize_weights};

const REPO_2B: &str = "mlx-community/Qwen3.5-2B-MLX-8bit";
const REPO_08B: &str = "mlx-community/Qwen3.5-0.8B-MLX-8bit";

fn resolve_model_dir(repo_id: &str) -> std::path::PathBuf {
    let env_key = match repo_id {
        REPO_2B => "MLX_RS_VLM_MODEL_DIR_2B",
        REPO_08B => "MLX_RS_VLM_MODEL_DIR_08B",
        _ => "MLX_RS_VLM_MODEL_DIR",
    };
    if let Ok(dir) = std::env::var(env_key) {
        std::path::PathBuf::from(dir)
    } else if let Ok(dir) = std::env::var("MLX_RS_VLM_MODEL_DIR") {
        std::path::PathBuf::from(dir)
    } else {
        download_model(repo_id, None).unwrap()
    }
}

fn load_config(model_dir: &Path) -> Qwen35Configuration {
    let config_json = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
    serde_json::from_str(&config_json).unwrap()
}

fn load_model(model_dir: &Path) -> (Qwen35VLM, Qwen35Configuration) {
    let config = load_config(model_dir);
    let loaded = load_weights(model_dir).unwrap();
    let result = sanitize_weights(loaded.weights, &config, loaded.is_mlx_format).unwrap();
    let mut model = Qwen35VLM::new(&config, result.quant_info.as_ref()).unwrap();
    model.load_weights(result.weights).unwrap();
    (model, config)
}

mod sanitize {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use mlx_rs::Array;
    use mlx_rs::module::ModuleParameters;

    #[test]
    fn test_sanitize_key_mapping_patterns() {
        let mut weights = HashMap::new();

        // HF-style language_model.model.* → language_model.*
        weights.insert("language_model.model.layers.0.self_attn.q_proj.weight".into(), Array::zeros::<u32>(&[4096, 512]).unwrap());
        weights.insert("language_model.model.layers.0.self_attn.q_proj.scales".into(), Array::zeros::<f32>(&[4096, 32]).unwrap());
        weights.insert("language_model.model.layers.0.self_attn.q_proj.biases".into(), Array::zeros::<f32>(&[4096, 32]).unwrap());
        weights.insert("language_model.model.layers.0.self_attn.k_proj.weight".into(), Array::zeros::<f32>(&[512, 2048]).unwrap());
        weights.insert("language_model.model.layers.0.mlp.gate_proj.weight".into(), Array::zeros::<u32>(&[6144, 256]).unwrap());
        weights.insert("language_model.model.layers.0.mlp.gate_proj.scales".into(), Array::zeros::<f32>(&[6144, 64]).unwrap());
        weights.insert("language_model.model.layers.0.mlp.gate_proj.biases".into(), Array::zeros::<f32>(&[6144, 64]).unwrap());

        // model.visual.* → vision_tower.*
        weights.insert("model.visual.blocks.0.attn.qkv.weight".into(), Array::zeros::<f32>(&[3072, 1024]).unwrap());

        // embed_tokens (tied → lm_head dropped)
        weights.insert("language_model.model.embed_tokens.weight".into(), Array::zeros::<f32>(&[248320, 2048]).unwrap());
        weights.insert("lm_head.weight".into(), Array::zeros::<f32>(&[248320, 2048]).unwrap());

        // A_log → a_log
        weights.insert("language_model.model.layers.0.linear_attn.A_log".into(), Array::zeros::<f32>(&[16]).unwrap());

        // mtp.* → filtered out
        weights.insert("mtp.layer.weight".into(), Array::from(1.0f32));

        // norm weights (HF: +1.0 adjustment)
        weights.insert("language_model.model.layers.0.input_layernorm.weight".into(), Array::zeros::<f32>(&[2048]).unwrap());
        weights.insert("language_model.model.norm.weight".into(), Array::zeros::<f32>(&[2048]).unwrap());

        let config: Qwen35Configuration = serde_json::from_str(r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": { "group_size": 64, "bits": 8 }
        }"#).unwrap();

        let result = sanitize_weights(weights, &config, false).unwrap();
        let keys: HashSet<String> = result.weights.keys().cloned().collect();

        // language_model.model.* → language_model.*
        assert!(keys.contains("language_model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(keys.contains("language_model.layers.0.self_attn.q_proj.scales"));
        assert!(keys.contains("language_model.layers.0.self_attn.k_proj.weight"));
        assert!(keys.contains("language_model.layers.0.mlp.gate_proj.inner.weight"));

        // model.visual.* → vision_tower.*
        assert!(keys.contains("vision_tower.blocks.0.attn.qkv.weight"));

        // embed_tokens kept, lm_head dropped (tie_word_embeddings)
        assert!(keys.contains("language_model.embed_tokens.weight"));
        assert!(!keys.iter().any(|k| k.contains("lm_head")));

        // A_log → a_log
        assert!(keys.contains("language_model.layers.0.linear_attn.a_log"));

        // mtp filtered
        assert!(!keys.iter().any(|k| k.starts_with("mtp.")));

        // quantization info
        let qi = result.quant_info.as_ref().unwrap();
        assert!(qi.for_path("language_model.layers.0.self_attn.q_proj").is_some());
        assert!(qi.for_path("language_model.layers.0.mlp.gate_proj").is_some());
        assert!(qi.for_path("language_model.layers.0.self_attn.k_proj").is_none());
    }

    #[test]
    fn test_model_construction_from_config() {
        let config: Qwen35Configuration = serde_json::from_str(r#"{
            "model_type": "qwen3_5_vl",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "intermediate_size": 6144,
                "tie_word_embeddings": true,
                "full_attention_interval": 4,
                "vocab_size": 248320
            },
            "vision_config": {},
            "quantization": { "group_size": 64, "bits": 8 }
        }"#).unwrap();

        let mut model = Qwen35VLM::new(&config, None).unwrap();
        let params = model.parameters().flatten();
        assert!(!params.is_empty(), "Model should have parameters");

        assert!(params.keys().any(|k| k.contains("language_model.layers.0.")));
        assert!(params.keys().any(|k| k.contains("language_model.layers.3.")));
        assert!(params.keys().any(|k| k.contains("embed_tokens")));
        assert!(params.keys().any(|k| k.contains("language_model.norm")));
        assert!(params.keys().any(|k| k.contains("vision_tower")));

        let mut dummy_weights: HashMap<String, Array> = HashMap::new();
        for (key, value) in params.iter() {
            let zeros = mlx_rs::ops::zeros_like_device(value, mlx_rs::StreamOrDevice::default()).unwrap();
            dummy_weights.insert(key.to_string(), zeros);
        }
        model.load_weights(dummy_weights).unwrap();
    }
}

mod weights {
    use super::*;
    use mlx_rs::module::ModuleParameters;

    /// Requires model download. Run with: cargo test --test test_qwen35 -- --ignored
    #[test]
    #[ignore]
    fn test_load_model_2b() {
        let model_dir = resolve_model_dir(REPO_2B);
        let (_model, _config) = load_model(&model_dir);
    }

    /// Requires model download. Run with: cargo test --test test_qwen35 -- --ignored
    #[test]
    #[ignore]
    fn test_weight_mapping_2b() {
        let model_dir = resolve_model_dir(REPO_2B);
        let config = load_config(&model_dir);
        let loaded = load_weights(&model_dir).unwrap();
        let result = sanitize_weights(loaded.weights, &config, loaded.is_mlx_format).unwrap();

        let model = Qwen35VLM::new(&config, result.quant_info.as_ref()).unwrap();
        let model_params = model.parameters().flatten();
        let model_keys: std::collections::HashSet<String> =
            model_params.keys().map(|k| k.to_string()).collect();
        let weight_keys: std::collections::HashSet<String> =
            result.weights.keys().cloned().collect();

        let mut extra: Vec<_> = weight_keys.difference(&model_keys).cloned().collect();
        extra.sort();
        let mut missing: Vec<_> = model_keys.difference(&weight_keys).cloned().collect();
        missing.sort();

        println!("\n=== Weights NOT in model ({}) ===", extra.len());
        for k in &extra {
            println!("  EXTRA: {}", k);
        }
        println!("\n=== Model params NOT in weights ({}) ===", missing.len());
        for k in &missing {
            println!("  MISSING: {}", k);
        }
        println!(
            "\nModel params: {}, Weight keys: {}",
            model_keys.len(),
            weight_keys.len()
        );

        assert!(
            extra.is_empty() && missing.is_empty(),
            "Weight mapping mismatch: {} extra, {} missing",
            extra.len(),
            missing.len()
        );
    }
}

mod text_generation {
    use super::*;

    /// Requires model download. Run with: cargo test --test test_qwen35 -- --ignored
    #[test]
    #[ignore]
    fn test_hello_2b() {
        let model_dir = resolve_model_dir(REPO_2B);
        let (mut model, config) = load_model(&model_dir);

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let input_array = mlx_rs::Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

        let params = GenerateParams {
            max_tokens: 32,
            temperature: 0.0,
            ..GenerateParams::with_config(&config)
        };
        let tokens = generate(&mut model, &input_array, None, None, &params).unwrap();
        assert!(!tokens.is_empty());

        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        let decoded = tokenizer.decode(&token_ids, true).unwrap();
        assert!(!decoded.is_empty(), "output should not be empty");
        println!("Output (2B): {}", decoded);
    }

    /// Requires model download. Run with: cargo test --test test_qwen35 -- --ignored
    #[test]
    #[ignore]
    fn test_hello_08b() {
        let model_dir = resolve_model_dir(REPO_08B);
        let (mut model, config) = load_model(&model_dir);

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let input_array = mlx_rs::Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

        let params = GenerateParams {
            max_tokens: 32,
            temperature: 0.0,
            ..GenerateParams::with_config(&config)
        };
        let tokens = generate(&mut model, &input_array, None, None, &params).unwrap();
        assert!(!tokens.is_empty());

        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        let decoded = tokenizer.decode(&token_ids, true).unwrap();
        assert!(!decoded.is_empty(), "output should not be empty");
        println!("Output (0.8B): {}", decoded);
    }
}

mod image_generation {
    use super::*;

    fn create_test_image() -> image::DynamicImage {
        let mut img = image::RgbImage::new(200, 100);
        for y in 0..100u32 {
            for x in 0..200u32 {
                let r = ((x * 255) / 200) as u8;
                let g = ((y * 255) / 100) as u8;
                let b = 128u8;
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        image::DynamicImage::ImageRgb8(img)
    }

    /// Requires model download. Run with: cargo test --test test_qwen35 -- --ignored
    #[test]
    #[ignore]
    fn test_image_describe_08b() {
        let model_dir = resolve_model_dir(REPO_08B);
        let (mut model, config) = load_model(&model_dir);

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

        let proc_config: ProcessorConfig = {
            let proc_path = model_dir.join("preprocessor_config.json");
            if proc_path.exists() {
                let json = std::fs::read_to_string(&proc_path).unwrap();
                serde_json::from_str(&json).unwrap()
            } else {
                ProcessorConfig::default()
            }
        };

        let img = create_test_image();
        let (pixels, thw) =
            mlx_rs_vlm::image::preprocess_image(&img, &config.vision_config, &proc_config)
                .unwrap();

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let mut input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

        let thw_list = [thw];
        input_ids = mlx_rs_vlm::image::expand_image_tokens(
            &input_ids,
            &thw_list,
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            config.vision_config.spatial_merge_size,
        );

        let input_array = mlx_rs::Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

        let params = GenerateParams {
            max_tokens: 64,
            temperature: 0.0,
            ..GenerateParams::with_config(&config)
        };
        let tokens = generate(
            &mut model,
            &input_array,
            Some(&pixels),
            Some(&thw_list),
            &params,
        )
        .unwrap();
        assert!(!tokens.is_empty());

        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        let decoded = tokenizer.decode(&token_ids, true).unwrap();
        assert!(!decoded.is_empty(), "output should not be empty");
        println!("Image output (0.8B): {}", decoded);
    }

    /// Requires model download. Run with: cargo test --test test_qwen35 -- --ignored
    #[test]
    #[ignore]
    fn test_image_describe_2b() {
        let model_dir = resolve_model_dir(REPO_2B);
        let (mut model, config) = load_model(&model_dir);

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

        let proc_config: ProcessorConfig = {
            let proc_path = model_dir.join("preprocessor_config.json");
            if proc_path.exists() {
                let json = std::fs::read_to_string(&proc_path).unwrap();
                serde_json::from_str(&json).unwrap()
            } else {
                ProcessorConfig::default()
            }
        };

        let img = create_test_image();
        let (pixels, thw) =
            mlx_rs_vlm::image::preprocess_image(&img, &config.vision_config, &proc_config)
                .unwrap();

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let mut input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

        let thw_list = [thw];
        input_ids = mlx_rs_vlm::image::expand_image_tokens(
            &input_ids,
            &thw_list,
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            config.vision_config.spatial_merge_size,
        );

        let input_array = mlx_rs::Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

        let params = GenerateParams {
            max_tokens: 64,
            temperature: 0.0,
            ..GenerateParams::with_config(&config)
        };
        let tokens = generate(
            &mut model,
            &input_array,
            Some(&pixels),
            Some(&thw_list),
            &params,
        )
        .unwrap();
        assert!(!tokens.is_empty());

        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        let decoded = tokenizer.decode(&token_ids, true).unwrap();
        assert!(!decoded.is_empty(), "output should not be empty");
        println!("Image output (2B): {}", decoded);
    }
}
