use anyhow::{Context, Result};

use mlx_rs_vlm::config::{ProcessorConfig, Qwen35Configuration};
use mlx_rs_vlm::generate::{GenerateParams, generate};
use mlx_rs_vlm::hub::download_model;
use mlx_rs_vlm::image::{expand_image_tokens, preprocess_image};
use mlx_rs_vlm::model::Qwen35VLM;
use mlx_rs_vlm::weights::{load_weights, sanitize_weights};

fn main() -> Result<()> {
    let repo_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mlx-community/Qwen3.5-0.8B-MLX-8bit".to_string());
    let image_path = std::env::args().nth(2);

    println!("Downloading model: {}...", repo_id);
    let model_dir = download_model(&repo_id, None)?;
    let model_path = model_dir.as_path();

    println!("Loading config...");
    let config_json = std::fs::read_to_string(model_path.join("config.json"))
        .context("Failed to read config.json")?;
    let config: Qwen35Configuration =
        serde_json::from_str(&config_json).context("Failed to parse config.json")?;

    println!("Loading weights...");
    let loaded = load_weights(model_path)?;
    let result = sanitize_weights(loaded.weights, &config, loaded.is_mlx_format)?;
    println!(
        "Loaded {} weight tensors (mlx_format={})",
        result.weights.len(),
        loaded.is_mlx_format
    );

    println!("Building model...");
    let mut model = Qwen35VLM::new(&config, result.quant_info.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to build model: {}", e))?;

    println!("Loading weights into model...");
    model
        .load_weights(result.weights)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    println!("Weights loaded successfully");

    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let proc_config: ProcessorConfig = {
        let proc_path = model_path.join("preprocessor_config.json");
        if proc_path.exists() {
            let json = std::fs::read_to_string(&proc_path)
                .context("Failed to read preprocessor_config.json")?;
            serde_json::from_str(&json).context("Failed to parse preprocessor_config.json")?
        } else {
            ProcessorConfig::default()
        }
    };

    let (pixel_values, grid_thw) = if let Some(ref img_path) = image_path {
        println!("Processing image: {}", img_path);
        let img = image::open(img_path).context("Failed to open image")?;
        let (pixels, thw) = preprocess_image(&img, &config.vision_config, &proc_config)?;
        (Some(pixels), Some(vec![thw]))
    } else {
        (None, None)
    };

    let prompt = if image_path.is_some() {
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    } else {
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    };

    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

    if let Some(ref thw_list) = grid_thw {
        input_ids = expand_image_tokens(
            &input_ids,
            thw_list,
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            config.vision_config.spatial_merge_size,
        );
    }

    let input_array = mlx_rs::Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

    println!("Generating ({} input tokens)...", input_ids.len());
    let params = GenerateParams::with_config(&config);

    let start_time = std::time::Instant::now();
    let grid_thw_ref = grid_thw.as_deref();
    let tokens = generate(
        &mut model,
        &input_array,
        pixel_values.as_ref(),
        grid_thw_ref,
        &params,
    )?;

    let output_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let decoded = tokenizer
        .decode(&output_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

    let elapsed = start_time.elapsed();
    println!("Output: {}", decoded);
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        tokens.len(),
        elapsed.as_secs_f64(),
        tokens.len() as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
