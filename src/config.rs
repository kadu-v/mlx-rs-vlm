use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use mlx_rs as mlx;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum IntOrVec {
    Vec(Vec<i32>),
    Int(i32),
}

impl IntOrVec {
    fn into_vec(self) -> Vec<i32> {
        match self {
            IntOrVec::Vec(v) => v,
            IntOrVec::Int(i) => vec![i],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Configuration {
    pub model_type: String,
    pub text_config: TextConfiguration,
    pub vision_config: VisionConfiguration,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: i32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: i32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: i32,
    eos_token_id: Option<IntOrVec>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub image_token_index: Option<i32>,
    #[serde(default)]
    pub video_token_index: Option<i32>,
    #[serde(default)]
    pub vocab_size: Option<i32>,
    #[serde(default = "default_ignore_index")]
    pub ignore_index: i32,
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

impl Qwen35Configuration {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        serde_json::from_str(&json).with_context(|| format!("Failed to parse {}", path.display()))
    }

    pub fn image_token_index(&self) -> i32 {
        self.image_token_index.unwrap_or(self.image_token_id)
    }
    pub fn video_token_index(&self) -> i32 {
        self.video_token_index.unwrap_or(self.video_token_id)
    }
    pub fn vocab_size(&self) -> i32 {
        self.vocab_size.unwrap_or(self.text_config.vocab_size)
    }
    pub fn eos_token_ids(&self) -> Option<Vec<i32>> {
        self.eos_token_id.clone().map(|v| v.into_vec())
    }
    pub fn quantization_params(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
    }
}

fn default_ignore_index() -> i32 {
    -100
}
fn default_image_token_id() -> i32 {
    248056
}
fn default_video_token_id() -> i32 {
    248057
}
fn default_vision_start_token_id() -> i32 {
    248045
}
fn default_vision_end_token_id() -> i32 {
    248046
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfiguration {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    pub head_dim: Option<i32>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,
    pub rope_parameters: Option<RopeParameters>,
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: i32,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: i32,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: i32,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: i32,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub num_experts: Option<i32>,
}

fn default_hidden_size() -> i32 {
    4096
}
fn default_num_hidden_layers() -> i32 {
    32
}
fn default_num_attention_heads() -> i32 {
    32
}
fn default_num_key_value_heads() -> i32 {
    8
}
fn default_intermediate_size() -> i32 {
    14336
}
fn default_vocab_size() -> i32 {
    248320
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}
fn default_max_position_embeddings() -> i32 {
    131072
}
fn default_full_attention_interval() -> i32 {
    4
}
fn default_linear_conv_kernel_dim() -> i32 {
    4
}
fn default_linear_key_head_dim() -> i32 {
    192
}
fn default_linear_value_head_dim() -> i32 {
    128
}
fn default_linear_num_key_heads() -> i32 {
    16
}
fn default_linear_num_value_heads() -> i32 {
    64
}

impl TextConfiguration {
    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn mrope_section(&self) -> [i32; 3] {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.mrope_section.as_ref())
            .and_then(|s| {
                if s.len() >= 3 {
                    Some([s[0], s[1], s[2]])
                } else {
                    None
                }
            })
            .unwrap_or([11, 11, 10])
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.rope_theta)
            .unwrap_or(100_000.0)
    }

    pub fn is_linear_layer(&self, layer_idx: i32) -> bool {
        if let Some(ref layer_types) = self.layer_types {
            if let Some(lt) = layer_types.get(layer_idx as usize) {
                return lt == "linear_attention";
            }
        }
        (layer_idx + 1) % self.full_attention_interval != 0
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub mrope_interleaved: bool,
    pub mrope_section: Option<Vec<i32>>,
    pub partial_rotary_factor: Option<f32>,
    pub rope_theta: Option<f32>,
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfiguration {
    #[serde(default = "default_vision_depth")]
    pub depth: i32,
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_vision_num_heads")]
    pub num_heads: i32,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_vision_in_channels")]
    pub in_channels: i32,
    #[serde(default = "default_vision_spatial_merge_size")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_vision_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_vision_out_hidden_size")]
    pub out_hidden_size: i32,
    #[serde(default = "default_vision_num_position_embeddings")]
    pub num_position_embeddings: i32,
    pub hidden_act: Option<String>,
}

fn default_vision_depth() -> i32 {
    24
}
fn default_vision_hidden_size() -> i32 {
    1024
}
fn default_vision_num_heads() -> i32 {
    16
}
fn default_vision_intermediate_size() -> i32 {
    4096
}
fn default_vision_patch_size() -> i32 {
    16
}
fn default_vision_in_channels() -> i32 {
    3
}
fn default_vision_spatial_merge_size() -> i32 {
    2
}
fn default_vision_temporal_patch_size() -> i32 {
    2
}
fn default_vision_out_hidden_size() -> i32 {
    2048
}
fn default_vision_num_position_embeddings() -> i32 {
    2304
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
    #[serde(flatten)]
    pub per_layer: HashMap<String, serde_json::Value>,
}

pub const KNOWN_QUANT_KEYS: &[&str] = &["quant_method", "linear_class", "quantization_mode"];

fn default_quant_mode() -> String {
    "affine".to_string()
}

#[derive(Debug, Clone, Copy)]
pub struct THW {
    pub t: i32,
    pub h: i32,
    pub w: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorConfig {
    #[serde(default = "default_image_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_image_std")]
    pub image_std: Vec<f32>,
    #[serde(default = "default_min_pixels")]
    pub min_pixels: u32,
    #[serde(default = "default_max_pixels")]
    pub max_pixels: u32,
}

fn default_image_mean() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_image_std() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_min_pixels() -> u32 {
    4 * 28 * 28
}
fn default_max_pixels() -> u32 {
    16384 * 28 * 28
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            image_mean: default_image_mean(),
            image_std: default_image_std(),
            min_pixels: default_min_pixels(),
            max_pixels: default_max_pixels(),
        }
    }
}

/// preprocessor_config.json → processor_config.json → default の順で解決する。
pub fn load_processor_config(model_dir: &Path) -> anyhow::Result<ProcessorConfig> {
    for name in ["preprocessor_config.json", "processor_config.json"] {
        let path = model_dir.join(name);
        if path.exists() {
            let json = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            return serde_json::from_str(&json)
                .with_context(|| format!("Failed to parse {}", path.display()));
        }
    }
    Ok(ProcessorConfig::default())
}

/// generation_config.json (存在すれば)。eos_token_id / pad_token_id のみ使用。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerationConfig {
    eos_token_id: Option<IntOrVec>,
    #[serde(default)]
    pub pad_token_id: Option<i32>,
}

impl GenerationConfig {
    pub fn from_model_dir(model_dir: &Path) -> anyhow::Result<Option<Self>> {
        let path = model_dir.join("generation_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let config = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse {}", path.display()))?;
        Ok(Some(config))
    }

    pub fn eos_token_ids(&self) -> Option<Vec<i32>> {
        self.eos_token_id.clone().map(|v| v.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_config() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "intermediate_size": 6144,
                "vocab_size": 248320,
                "partial_rotary_factor": 0.25,
                "full_attention_interval": 4,
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 10000000.0
                },
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ]
            },
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "num_heads": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 2048
            },
            "image_token_id": 248056,
            "video_token_id": 248057,
            "quantization": { "group_size": 64, "bits": 8 }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();

        assert_eq!(config.model_type, "qwen3_5");
        assert_eq!(config.text_config.hidden_size, 2048);
        assert_eq!(config.text_config.num_hidden_layers, 24);
        assert_eq!(config.text_config.num_attention_heads, 8);
        assert_eq!(config.text_config.num_key_value_heads, 2);
        assert_eq!(config.text_config.head_dim(), 256);
        assert_eq!(config.text_config.intermediate_size, 6144);
        assert_eq!(config.text_config.vocab_size, 248320);
        assert_eq!(config.text_config.partial_rotary_factor, 0.25);
        assert_eq!(config.text_config.full_attention_interval, 4);
        assert_eq!(config.text_config.mrope_section(), [11, 11, 10]);
        assert_eq!(config.text_config.rope_theta(), 10_000_000.0);
        assert!(config.text_config.tie_word_embeddings);
        assert!(config.text_config.is_linear_layer(0));
        assert!(!config.text_config.is_linear_layer(3));

        assert_eq!(config.vision_config.depth, 24);
        assert_eq!(config.vision_config.hidden_size, 1024);
        assert_eq!(config.vision_config.num_heads, 16);
        assert_eq!(config.vision_config.patch_size, 16);
        assert_eq!(config.vision_config.spatial_merge_size, 2);
        assert_eq!(config.vision_config.out_hidden_size, 2048);

        assert_eq!(config.image_token_id, 248056);
        assert_eq!(config.video_token_id, 248057);
        assert!(config.quantization.is_some());
        let q = config.quantization.unwrap();
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
    }

    #[test]
    fn test_eos_token_id_single_int() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {},
            "vision_config": {},
            "eos_token_id": 128001
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_ids(), Some(vec![128001]));
    }

    #[test]
    fn test_eos_token_id_array() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {},
            "vision_config": {},
            "eos_token_id": [248046, 248044]
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_ids(), Some(vec![248046, 248044]));
    }

    #[test]
    fn test_eos_token_id_missing() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {},
            "vision_config": {}
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_ids(), None);
    }

    #[test]
    fn test_text_config_defaults() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {},
            "vision_config": {}
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert_eq!(config.text_config.hidden_size, 4096);
        assert_eq!(config.text_config.num_hidden_layers, 32);
        assert_eq!(config.text_config.num_attention_heads, 32);
        assert_eq!(config.text_config.num_key_value_heads, 8);
        assert_eq!(config.text_config.intermediate_size, 14336);
        assert_eq!(config.text_config.linear_key_head_dim, 192);
        assert_eq!(config.text_config.linear_num_value_heads, 64);
    }

    #[test]
    fn test_mrope_section_short_fallback() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "rope_parameters": { "mrope_section": [11] }
            },
            "vision_config": {}
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert_eq!(config.text_config.mrope_section(), [11, 11, 10]);
    }

    #[test]
    fn test_load_processor_config_fallback() {
        let dir = std::env::temp_dir().join("mlx_vlm_test_proc_config");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // ファイルなし → default
        let config = load_processor_config(&dir).unwrap();
        assert_eq!(config.min_pixels, default_min_pixels());

        // processor_config.json のみ (say2 モデルと同じ構成) → fallback で読まれ、
        // 欠落フィールドは default で埋まる
        std::fs::write(
            dir.join("processor_config.json"),
            r#"{"image_mean": [0.1, 0.1, 0.1], "image_std": [0.5, 0.5, 0.5]}"#,
        )
        .unwrap();
        let config = load_processor_config(&dir).unwrap();
        assert_eq!(config.image_mean, vec![0.1, 0.1, 0.1]);
        assert_eq!(config.max_pixels, default_max_pixels());

        // preprocessor_config.json があれば優先
        std::fs::write(
            dir.join("preprocessor_config.json"),
            r#"{"image_mean": [0.2, 0.2, 0.2]}"#,
        )
        .unwrap();
        let config = load_processor_config(&dir).unwrap();
        assert_eq!(config.image_mean, vec![0.2, 0.2, 0.2]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_generation_config() {
        let config: GenerationConfig =
            serde_json::from_str(r#"{"eos_token_id": [248046, 248044], "pad_token_id": 248055}"#)
                .unwrap();
        assert_eq!(config.eos_token_ids(), Some(vec![248046, 248044]));
        assert_eq!(config.pad_token_id, Some(248055));

        let config: GenerationConfig = serde_json::from_str(r#"{"use_cache": true}"#).unwrap();
        assert_eq!(config.eos_token_ids(), None);
    }

    #[test]
    fn test_mrope_section_empty_fallback() {
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "rope_parameters": { "mrope_section": [] }
            },
            "vision_config": {}
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert_eq!(config.text_config.mrope_section(), [11, 11, 10]);
    }
}
