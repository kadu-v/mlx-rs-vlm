use std::path::Path;

use mlx_rs_vlm::config::{ProcessorConfig, Qwen35Configuration};
use mlx_rs_vlm::generate::{generate, GenerateParams};
use mlx_rs_vlm::hub::download_model;
use mlx_rs_vlm::model::Qwen35VLM;
use mlx_rs_vlm::weights::{load_weights, sanitize_weights};

const REPO_2B: &str = "mlx-community/Qwen3.5-2B-MLX-8bit";
const REPO_08B: &str = "mlx-community/Qwen3.5-0.8B-MLX-8bit";

fn load_config(model_dir: &Path) -> Qwen35Configuration {
    let config_json = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
    serde_json::from_str(&config_json).unwrap()
}

fn load_model(model_dir: &Path) -> (Qwen35VLM, Qwen35Configuration) {
    let config = load_config(model_dir);
    let loaded = load_weights(model_dir).unwrap();
    let weights = sanitize_weights(loaded.weights, &config, loaded.is_mlx_format);
    let mut model = Qwen35VLM::new(&config).unwrap();
    model.load_weights(weights).unwrap();
    (model, config)
}

mod weights {
    use super::*;
    use mlx_rs::module::ModuleParameters;

    #[test]
    #[ignore]
    fn test_load_model_2b() {
        let model_dir = download_model(REPO_2B, None).unwrap();
        let (_model, _config) = load_model(&model_dir);
    }

    #[test]
    #[ignore]
    fn test_weight_mapping_2b() {
        let model_dir = download_model(REPO_2B, None).unwrap();
        let config = load_config(&model_dir);
        let loaded = load_weights(&model_dir).unwrap();
        let weights = sanitize_weights(loaded.weights, &config, loaded.is_mlx_format);

        let model = Qwen35VLM::new(&config).unwrap();
        let model_params = model.parameters().flatten();
        let model_keys: std::collections::HashSet<String> =
            model_params.keys().map(|k| k.to_string()).collect();
        let weight_keys: std::collections::HashSet<String> = weights.keys().cloned().collect();

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

    #[test]
    #[ignore]
    fn test_hello_2b() {
        let model_dir = download_model(REPO_2B, None).unwrap();
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

    #[test]
    #[ignore]
    fn test_hello_08b() {
        let model_dir = download_model(REPO_08B, None).unwrap();
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

    #[test]
    #[ignore]
    fn test_image_describe_08b() {
        let model_dir = download_model(REPO_08B, None).unwrap();
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

    #[test]
    #[ignore]
    fn test_image_describe_2b() {
        let model_dir = download_model(REPO_2B, None).unwrap();
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
