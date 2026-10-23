use std::path::Path;

use anyhow::{Result, bail};
use mlx::Array;
use mlx_rs as mlx;

use crate::chat::{Message, TemplateOptions};
use crate::config::{
    GenerationConfig, ProcessorConfig, Qwen35Configuration, THW, VisionConfiguration,
    load_processor_config,
};
use crate::detokenizer::StreamingDetokenizer;
use crate::generate::GenerateParams;
use crate::image::{expand_image_tokens, preprocess_image};
use crate::tokenizer::Tokenizer;

/// mlx-swift-lm の UserInputProcessor 相当。
/// メッセージ + 画像から、モデルに渡せるトークン列とピクセルテンソルを組み立てる。
pub struct Qwen35Processor {
    tokenizer: Box<dyn Tokenizer>,
    processor_config: ProcessorConfig,
    vision_config: VisionConfiguration,
    image_token_id: i32,
    vision_start_token_id: i32,
    vision_end_token_id: i32,
    eos_token_ids: Vec<i32>,
}

pub struct PreparedInput {
    /// [1, seq_len]
    pub input_ids: Array,
    pub pixel_values: Option<Array>,
    pub grid_thw: Vec<THW>,
}

impl PreparedInput {
    /// `generate` の `grid_thw` 引数にそのまま渡せる形。
    pub fn grid_thw_slice(&self) -> Option<&[THW]> {
        (!self.grid_thw.is_empty()).then_some(self.grid_thw.as_slice())
    }
}

impl Qwen35Processor {
    /// 任意の [`Tokenizer`] 実装を注入して構築する
    /// (mlx-swift-lm の TokenizerLoader 注入に相当)。
    pub fn with_tokenizer(
        model_dir: impl AsRef<Path>,
        tokenizer: Box<dyn Tokenizer>,
    ) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config = Qwen35Configuration::from_file(&dir.join("config.json"))?;
        let processor_config = load_processor_config(dir)?;
        let eos_token_ids = resolve_eos_token_ids(dir, &config, tokenizer.as_ref());

        Ok(Self {
            tokenizer,
            processor_config,
            vision_config: config.vision_config.clone(),
            image_token_id: config.image_token_id,
            vision_start_token_id: config.vision_start_token_id,
            vision_end_token_id: config.vision_end_token_id,
            eos_token_ids,
        })
    }

    /// モデルディレクトリから自己完結で構築する (tokenizer.json + chat_template.jinja +
    /// preprocessor_config.json / processor_config.json + generation_config.json)。
    #[cfg(feature = "tokenizer")]
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        let tokenizer = crate::tokenizer::HfTokenizer::from_model_dir(dir)?;
        Self::with_tokenizer(dir, Box::new(tokenizer))
    }

    pub fn apply_chat_template(
        &self,
        messages: &[Message],
        opts: &TemplateOptions,
    ) -> Result<String> {
        self.tokenizer.apply_chat_template(messages, opts)
    }

    /// chat template 適用 → encode → 画像トークン展開 → テンソル化。
    pub fn prepare(
        &self,
        messages: &[Message],
        images: &[image::DynamicImage],
    ) -> Result<PreparedInput> {
        self.prepare_with_options(messages, images, &TemplateOptions::default())
    }

    pub fn prepare_with_options(
        &self,
        messages: &[Message],
        images: &[image::DynamicImage],
        opts: &TemplateOptions,
    ) -> Result<PreparedInput> {
        let slot_count: usize = messages.iter().map(|m| m.image_count()).sum();
        if slot_count != images.len() {
            bail!(
                "Message image slots ({slot_count}) do not match provided images ({})",
                images.len()
            );
        }
        let prompt = self.tokenizer.apply_chat_template(messages, opts)?;
        self.prepare_prompt(&prompt, images)
    }

    /// chat template を通さない生プロンプト版。画像 1 枚につき
    /// `<|vision_start|><|image_pad|><|vision_end|>` を 1 組含めること。
    pub fn prepare_prompt(
        &self,
        prompt: &str,
        images: &[image::DynamicImage],
    ) -> Result<PreparedInput> {
        let mut pixel_arrays = Vec::with_capacity(images.len());
        let mut grid_thw = Vec::with_capacity(images.len());
        for img in images {
            let (pixels, thw) = preprocess_image(img, &self.vision_config, &self.processor_config)?;
            pixel_arrays.push(pixels);
            grid_thw.push(thw);
        }

        let mut input_ids = self.tokenizer.encode(prompt, false)?;
        if !grid_thw.is_empty() {
            let before = input_ids.len();
            input_ids = expand_image_tokens(
                &input_ids,
                &grid_thw,
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
                self.vision_config.spatial_merge_size,
            );
            if input_ids.len() == before {
                bail!(
                    "Prompt does not contain the expected \
                     <|vision_start|><|image_pad|><|vision_end|> markers for {} image(s)",
                    grid_thw.len()
                );
            }
        }

        let pixel_values = match pixel_arrays.len() {
            0 => None,
            1 => Some(pixel_arrays.into_iter().next().unwrap()),
            _ => {
                let refs: Vec<&Array> = pixel_arrays.iter().collect();
                Some(
                    mlx::ops::concatenate_axis(&refs, 0)
                        .map_err(|e| anyhow::anyhow!("Failed to concatenate pixel values: {e}"))?,
                )
            }
        };

        let input_ids = Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);
        Ok(PreparedInput {
            input_ids,
            pixel_values,
            grid_thw,
        })
    }

    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        self.tokenizer.decode(tokens, true)
    }

    pub fn stream_decoder(&self) -> StreamingDetokenizer<'_> {
        StreamingDetokenizer::new(self.tokenizer.as_ref())
    }

    pub fn eos_token_ids(&self) -> &[i32] {
        &self.eos_token_ids
    }

    /// eos 解決済みの GenerateParams。
    pub fn generate_params(&self) -> GenerateParams {
        GenerateParams {
            eos_token_ids: self.eos_token_ids.clone(),
            ..Default::default()
        }
    }
}

/// eos_token_ids の解決順位:
/// generation_config.json → config.json → tokenizer_config.json の eos_token 文字列
/// → 既存ハードコード値。
fn resolve_eos_token_ids(
    dir: &Path,
    config: &Qwen35Configuration,
    tokenizer: &dyn Tokenizer,
) -> Vec<i32> {
    if let Ok(Some(gen_config)) = GenerationConfig::from_model_dir(dir) {
        if let Some(ids) = gen_config.eos_token_ids() {
            return ids;
        }
    }
    if let Some(ids) = config.eos_token_ids() {
        return ids;
    }
    if let Some(ids) = eos_from_tokenizer_config(dir, tokenizer) {
        return ids;
    }
    GenerateParams::default().eos_token_ids
}

fn eos_from_tokenizer_config(dir: &Path, tokenizer: &dyn Tokenizer) -> Option<Vec<i32>> {
    let json = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&json).ok()?;
    let eos_token = value.get("eos_token")?.as_str()?;
    tokenizer.token_to_id(eos_token).map(|id| vec![id])
}
