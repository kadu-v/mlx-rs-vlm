use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};

use mlx_rs_vlm::chat::Message;
use mlx_rs_vlm::generate::generate_with_callback;
use mlx_rs_vlm::hub::download_model;
use mlx_rs_vlm::processor::Qwen35Processor;
use mlx_rs_vlm::weights::load_model;

fn main() -> Result<()> {
    let model_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mlx-community/Qwen3.5-0.8B-MLX-8bit".to_string());
    let image_path = std::env::args().nth(2);

    // ローカルディレクトリならそのまま使い、そうでなければ HF repo_id として扱う
    let model_dir = if PathBuf::from(&model_arg).is_dir() {
        PathBuf::from(&model_arg)
    } else {
        println!("Downloading model: {}...", model_arg);
        download_model(&model_arg, None)?
    };

    println!("Loading model...");
    let (mut model, _config) = load_model(&model_dir)?;
    let processor = Qwen35Processor::from_model_dir(&model_dir)?;

    let (messages, images) = match &image_path {
        Some(path) => {
            println!("Processing image: {}", path);
            let img = image::open(path).context("Failed to open image")?;
            (
                vec![
                    Message::system("You are a helpful assistant."),
                    Message::user_with_image("Describe this image."),
                ],
                vec![img],
            )
        }
        None => (
            vec![
                Message::system("You are a helpful assistant."),
                Message::user("Hello!"),
            ],
            vec![],
        ),
    };

    let input = processor.prepare(&messages, &images)?;
    let params = processor.generate_params();

    println!("Generating ({} input tokens)...", input.input_ids.shape()[1]);
    let start_time = std::time::Instant::now();

    let mut detok = processor.stream_decoder();
    let tokens = generate_with_callback(
        &mut model,
        &input.input_ids,
        input.pixel_values.as_ref(),
        input.grid_thw_slice(),
        &params,
        |token| {
            if let Some(chunk) = detok.push(token) {
                print!("{chunk}");
                std::io::stdout().flush().ok();
            }
            true
        },
    )?;
    if let Some(rest) = detok.finish() {
        print!("{rest}");
    }
    println!();

    let elapsed = start_time.elapsed();
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        tokens.len(),
        elapsed.as_secs_f64(),
        tokens.len() as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
