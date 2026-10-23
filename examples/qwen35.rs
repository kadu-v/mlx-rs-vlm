use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use mlx_rs_vlm as mlx_vlm;
use mlx_vlm::chat::{ContentPart, Message, Role, TemplateOptions};
use mlx_vlm::generate::generate_with_callback;
use mlx_vlm::hub::download_model;
use mlx_vlm::processor::Qwen35Processor;
use mlx_vlm::weights::load_model;

#[derive(Parser)]
#[command(about = "Qwen 3.5 VLM inference example")]
struct Args {
    /// プロンプト(省略時: 画像ありなら "Describe this image."、なしなら "Hello!")
    prompt: Option<String>,

    /// HF repo_id またはローカルモデルディレクトリ
    #[arg(short, long, default_value = "mlx-community/Qwen3.5-0.8B-MLX-8bit")]
    model: String,

    /// 画像パス(複数指定可: -i a.png -i b.png)
    #[arg(short, long)]
    image: Vec<PathBuf>,

    /// システムプロンプト
    #[arg(short, long, default_value = "You are a helpful assistant.")]
    system: String,

    /// 生成する最大トークン数
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// サンプリング温度(0.0 = greedy)
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,

    /// nucleus sampling の確率質量(1.0 = 無効)
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// 上位 k 語彙に制限(0 = 無効)
    #[arg(long, default_value_t = 0)]
    top_k: usize,

    /// thinking モードを有効化(<think> ブロック付きで生成)
    #[arg(long)]
    thinking: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // ローカルディレクトリならそのまま使い、そうでなければ HF repo_id として扱う
    let model_dir = if PathBuf::from(&args.model).is_dir() {
        PathBuf::from(&args.model)
    } else {
        println!("Downloading model: {}...", args.model);
        download_model(&args.model, None)?
    };

    println!("Loading model...");
    let (mut model, _config) = load_model(&model_dir)?;
    let processor = Qwen35Processor::from_model_dir(&model_dir)?;

    let images = args
        .image
        .iter()
        .map(|path| {
            println!("Processing image: {}", path.display());
            image::open(path).with_context(|| format!("Failed to open image: {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    let prompt = args.prompt.unwrap_or_else(|| {
        if images.is_empty() {
            "Hello!".to_string()
        } else {
            "Describe this image.".to_string()
        }
    });

    // 画像スロットを画像の枚数ぶん並べた user メッセージを構築する
    let mut content: Vec<ContentPart> = vec![ContentPart::Image; images.len()];
    content.push(ContentPart::Text(prompt));
    let messages = vec![
        Message::system(&args.system),
        Message {
            role: Role::User,
            content,
        },
    ];

    let template_opts = TemplateOptions {
        enable_thinking: args.thinking.then_some(true),
        ..Default::default()
    };
    let input = processor.prepare_with_options(&messages, &images, &template_opts)?;

    let mut params = processor.generate_params();
    params.max_tokens = args.max_tokens;
    params.temperature = args.temperature;
    params.top_p = args.top_p;
    params.top_k = args.top_k;

    println!(
        "Generating ({} input tokens)...",
        input.input_ids.shape()[1]
    );
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
