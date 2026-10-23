use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx::Array;
use mlx_rs as mlx;

use crate::config::Qwen35Configuration;

#[derive(Debug)]
pub struct LoadedWeights {
    pub weights: HashMap<String, Array>,
    pub is_mlx_format: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Affine,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizationSpec {
    pub group_size: i32,
    pub bits: i32,
    pub mode: QuantizationMode,
}

#[derive(Debug, Clone)]
pub struct QuantizationInfo {
    pub layers: HashMap<String, QuantizationSpec>,
}

impl QuantizationInfo {
    pub fn for_path(&self, path: &str) -> Option<QuantizationSpec> {
        self.layers.get(path).copied()
    }
}

#[derive(Debug)]
pub struct SanitizeResult {
    pub weights: HashMap<String, Array>,
    pub quant_info: Option<QuantizationInfo>,
}

/// config パース → 重みロード → sanitize → モデル構築までをまとめた便宜関数。
pub fn load_model(
    model_dir: impl AsRef<Path>,
) -> Result<(crate::model::Qwen35VLM, Qwen35Configuration)> {
    let dir = model_dir.as_ref();
    let config = Qwen35Configuration::from_file(&dir.join("config.json"))?;
    let loaded = load_weights(dir)?;
    let result = sanitize_weights(loaded.weights, &config, loaded.is_mlx_format)?;
    let mut model = crate::model::Qwen35VLM::new(&config, result.quant_info.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to build model: {e}"))?;
    model
        .load_weights(result.weights)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {e}"))?;
    Ok((model, config))
}

fn collect_safetensors(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory {}", dir.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            files.extend(collect_safetensors(&path)?);
        } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }
    Ok(files)
}

pub fn load_weights(model_dir: &Path) -> Result<LoadedWeights> {
    let mut safetensor_files = collect_safetensors(model_dir)?;
    safetensor_files.sort();

    if safetensor_files.is_empty() {
        anyhow::bail!("No .safetensors files found in {}", model_dir.display());
    }

    let mut all_weights = HashMap::new();
    let mut is_mlx_format = false;

    for file_path in &safetensor_files {
        // MLX ネイティブの遅延ロード (mlx_load_safetensors)。各テンソルは
        // eval 時にファイルから最終バッファへ直接ストリーム読み込みされる
        // ため、ファイル全体をヒープへ読み込まずピークメモリが約 1x で済む
        // (Swift の loadArrays(url:) と同じ経路)。
        let (weights, metadata) = Array::load_safetensors_with_metadata(file_path)
            .with_context(|| format!("Failed to load {}", file_path.display()))?;

        // Swift reads metadata only from the first safetensor file.
        // We check all files for robustness — any file with format=mlx sets the flag.
        if metadata
            .get("format")
            .map(|f| f.eq_ignore_ascii_case("mlx"))
            == Some(true)
        {
            is_mlx_format = true;
        }

        all_weights.extend(weights);
    }

    Ok(LoadedWeights {
        weights: all_weights,
        is_mlx_format,
    })
}

/// Sanitize and remap weight keys to match our module parameter hierarchy.
///
/// Combines two stages:
/// 1. `sanitize_hf_to_mlx`: HF→MLX value transforms (skip for MLX format)
/// 2. `adapt_keys_to_rust`: Rust module path adaptation (always runs)
pub fn sanitize_weights(
    weights: HashMap<String, Array>,
    config: &Qwen35Configuration,
    is_mlx_format: bool,
) -> Result<SanitizeResult> {
    let (intermediate, all_scales_prefixes) = adapt_keys_to_rust(weights, config, is_mlx_format)?;

    let (adapted, quant_info) = if !all_scales_prefixes.is_empty() {
        let q = config.quantization_params().ok_or_else(|| {
            anyhow::anyhow!(
                "Weights contain {} quantized layers (.scales) but config has no quantization settings",
                all_scales_prefixes.len()
            )
        })?;
        if q.mode != "affine" {
            anyhow::bail!(
                "Unsupported quantization mode '{}': only 'affine' is supported",
                q.mode
            );
        }

        let normalized_per_layer: HashMap<String, &serde_json::Value> = q
            .per_layer
            .iter()
            .filter(|(k, _)| !crate::config::KNOWN_QUANT_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (rename_weight_key(k), v))
            .collect();

        let default_spec = QuantizationSpec {
            group_size: q.group_size,
            bits: q.bits,
            mode: QuantizationMode::Affine,
        };

        let mut layers = HashMap::new();
        for p in &all_scales_prefixes {
            match normalized_per_layer.get(p) {
                Some(serde_json::Value::Bool(false)) => {
                    anyhow::bail!(
                        "Layer '{}' is marked as skip in quantization config but has .scales in weights",
                        p
                    );
                }
                Some(serde_json::Value::Object(obj)) => {
                    let mode = obj.get("mode").and_then(|v| v.as_str()).unwrap_or("affine");
                    if mode != "affine" {
                        anyhow::bail!(
                            "Unsupported quantization mode '{}' for layer '{}': only 'affine' is supported",
                            mode,
                            p
                        );
                    }
                    let gs = obj
                        .get("group_size")
                        .and_then(|v| v.as_i64())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Per-layer quantization for '{}' is missing 'group_size'",
                                p
                            )
                        })? as i32;
                    let bits = obj.get("bits").and_then(|v| v.as_i64()).ok_or_else(|| {
                        anyhow::anyhow!("Per-layer quantization for '{}' is missing 'bits'", p)
                    })? as i32;
                    layers.insert(
                        p.clone(),
                        QuantizationSpec {
                            group_size: gs,
                            bits,
                            mode: QuantizationMode::Affine,
                        },
                    );
                }
                None | Some(serde_json::Value::Bool(true)) => {
                    layers.insert(p.clone(), default_spec);
                }
                Some(other) => {
                    anyhow::bail!(
                        "Invalid per-layer quantization value for '{}': expected object or false, got {}",
                        p,
                        value_type_name(other)
                    );
                }
            }
        }

        let effective_prefixes: Vec<String> = layers.keys().cloned().collect();
        let adapted = remap_quantized_keys(intermediate, &effective_prefixes);
        let qi = if layers.is_empty() {
            None
        } else {
            Some(QuantizationInfo { layers })
        };
        (adapted, qi)
    } else {
        (intermediate, None)
    };

    let adapted = sanitize_vision_weights(adapted, config.vision_config.in_channels)?;

    Ok(SanitizeResult {
        weights: adapted,
        quant_info,
    })
}

/// Vision weight sanitize: remove position_ids, transpose patch_embed.proj.weight.
/// Corresponds to Swift's `visionModel.sanitize(weights:)`.
fn sanitize_vision_weights(
    weights: HashMap<String, Array>,
    in_channels: i32,
) -> Result<HashMap<String, Array>> {
    let mut result = HashMap::with_capacity(weights.len());
    for (key, value) in weights {
        if key.contains("position_ids") {
            continue;
        } else if key.contains("patch_embed.proj.weight") {
            if value.ndim() == 5 && value.dim(-1) == in_channels {
                result.insert(key, value);
            } else if value.ndim() == 5 {
                let transposed = value
                    .transpose_axes(&[0, 2, 3, 4, 1])
                    .with_context(|| format!("Failed to transpose '{key}'"))?;
                result.insert(key, transposed);
            } else {
                anyhow::bail!(
                    "Unexpected shape for '{}': expected 5D tensor, got {}D",
                    key,
                    value.ndim()
                );
            }
        } else {
            result.insert(key, value);
        }
    }
    Ok(result)
}

/// HF→MLX value transforms. Corresponds to Swift's `sanitize(weights:)`.
/// Only called when `!is_mlx_format`.
fn sanitize_hf_to_mlx(key: &str, mut value: Array, norm_keys: &[&str]) -> Result<Array> {
    if key.contains("conv1d.weight") && value.ndim() == 3 {
        let last_dim = value.shape()[2];
        if last_dim != 1 {
            value = value
                .move_axis(2, 1)
                .with_context(|| format!("Failed to reorder conv1d weight axes for '{key}'"))?;
        }
    }

    if (norm_keys.iter().any(|nk| key.ends_with(nk)) || key == "language_model.norm.weight")
        && value.ndim() == 1
    {
        value = value
            .add(&Array::from(1.0f32))
            .with_context(|| format!("Failed to apply norm +1.0 adjustment for '{key}'"))?;
    }

    Ok(value)
}

/// Rust module path adaptation. No Swift counterpart — Rust-specific.
/// Handles key rename, mtp filter, lm_head drop, A_log fix.
/// Returns intermediate weights (before .inner remap) and all prefixes that have .scales.
fn adapt_keys_to_rust(
    weights: HashMap<String, Array>,
    config: &Qwen35Configuration,
    is_mlx_format: bool,
) -> Result<(HashMap<String, Array>, Vec<String>)> {
    let norm_keys = [
        "q_norm.weight",
        "k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ];

    let mut intermediate = HashMap::new();
    for (key, mut value) in weights {
        if key.starts_with("mtp.") {
            continue;
        }

        if config.text_config.tie_word_embeddings
            && (key == "lm_head.weight"
                || key == "language_model.lm_head.weight"
                || key.ends_with(".lm_head.weight"))
        {
            continue;
        }

        let mut key = rename_weight_key(&key);
        key = key.replace(".A_log", ".a_log");

        if !is_mlx_format {
            value = sanitize_hf_to_mlx(&key, value, &norm_keys)?;
        }

        intermediate.insert(key, value);
    }

    let quantized_prefixes: Vec<String> = intermediate
        .keys()
        .filter(|k| k.ends_with(".scales"))
        .map(|k| k.strip_suffix(".scales").unwrap().to_string())
        .collect();

    Ok((intermediate, quantized_prefixes))
}

fn remap_quantized_keys(
    intermediate: HashMap<String, Array>,
    quantized_prefixes: &[String],
) -> HashMap<String, Array> {
    let mut result = HashMap::new();
    for (key, value) in intermediate {
        let mut new_key = key.clone();

        for prefix in quantized_prefixes {
            if key == format!("{}.weight", prefix) {
                new_key = format!("{}.inner.weight", prefix);
                break;
            }
            if key == format!("{}.bias", prefix) {
                new_key = format!("{}.inner.bias", prefix);
                break;
            }
        }

        result.insert(new_key, value);
    }
    result
}

fn value_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Remap checkpoint weight keys to Rust module paths.
///
/// Swift ↔ Rust path mapping:
///   language_model.model.*  → language_model.*
///   model.language_model.*  → language_model.*
///   model.visual.*          → vision_tower.*
///   model.* (bare)          → language_model.*
///   lm_head.*               → language_model.lm_head.*
///
/// Swift's LanguageModel has a nested `model: Model` containing layers/norm/embed_tokens.
/// Rust flattens this: LanguageModel directly holds layers/norm/embed_tokens.
/// This function absorbs the `.model.` level difference.
fn rename_weight_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("language_model.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("language_model.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.visual.") {
        format!("vision_tower.{rest}")
    } else if let Some(rest) = key.strip_prefix("lm_head.") {
        format!("language_model.lm_head.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.") {
        format!("language_model.{rest}")
    } else {
        key.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_weights() {
        let mut weights = HashMap::new();

        weights.insert("mtp.layer.weight".to_string(), Array::from(1.0f32));
        weights.insert(
            "language_model.model.layers.0.self_attn.q_norm.weight".to_string(),
            Array::zeros::<f32>(&[256]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.linear_attn.conv1d.weight".to_string(),
            Array::zeros::<f32>(&[6144, 1, 4]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.biases".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.linear_attn.A_log".to_string(),
            Array::zeros::<f32>(&[16]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.input_layernorm.weight".to_string(),
            Array::zeros::<f32>(&[2048]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.post_attention_layernorm.weight".to_string(),
            Array::zeros::<f32>(&[2048]).unwrap(),
        );
        weights.insert(
            "language_model.model.norm.weight".to_string(),
            Array::zeros::<f32>(&[2048]).unwrap(),
        );

        let config: Qwen35Configuration = serde_json::from_str(
            r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": { "group_size": 64, "bits": 8 }
        }"#,
        )
        .unwrap();

        let result = sanitize_weights(weights, &config, false).unwrap();
        let sanitized = &result.weights;

        assert!(!sanitized.contains_key("mtp.layer.weight"));

        let norm_w = sanitized
            .get("language_model.layers.0.self_attn.q_norm.weight")
            .unwrap();
        let mean_val: f32 = norm_w.mean(None).unwrap().item();
        assert!((mean_val - 1.0).abs() < 1e-5);

        let conv_w = sanitized
            .get("language_model.layers.0.linear_attn.conv1d.weight")
            .unwrap();
        assert_eq!(conv_w.shape()[1], 4);
        assert_eq!(conv_w.shape()[2], 1);

        assert!(sanitized.contains_key("language_model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(sanitized.contains_key("language_model.layers.0.self_attn.q_proj.scales"));
        assert!(sanitized.contains_key("language_model.layers.0.self_attn.q_proj.biases"));

        assert!(sanitized.contains_key("language_model.layers.0.linear_attn.a_log"));

        let ln_w = sanitized
            .get("language_model.layers.0.input_layernorm.weight")
            .unwrap();
        let ln_mean: f32 = ln_w.mean(None).unwrap().item();
        assert!(
            (ln_mean - 1.0).abs() < 1e-5,
            "input_layernorm should be +1.0"
        );

        let pln_w = sanitized
            .get("language_model.layers.0.post_attention_layernorm.weight")
            .unwrap();
        let pln_mean: f32 = pln_w.mean(None).unwrap().item();
        assert!(
            (pln_mean - 1.0).abs() < 1e-5,
            "post_attention_layernorm should be +1.0"
        );

        let final_norm = sanitized.get("language_model.norm.weight").unwrap();
        let fn_mean: f32 = final_norm.mean(None).unwrap().item();
        assert!((fn_mean - 1.0).abs() < 1e-5, "final norm should be +1.0");

        let qi = result.quant_info.as_ref().unwrap();
        assert!(
            qi.layers
                .contains_key("language_model.layers.0.self_attn.q_proj")
        );
        assert_eq!(qi.layers.len(), 1);
    }

    #[test]
    fn test_partial_quantization() {
        let mut weights = HashMap::new();

        weights.insert(
            "language_model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "language_model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );
        weights.insert(
            "language_model.layers.0.self_attn.q_proj.biases".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );
        weights.insert(
            "language_model.layers.0.self_attn.k_proj.weight".to_string(),
            Array::zeros::<f32>(&[512, 2048]).unwrap(),
        );

        let config: Qwen35Configuration = serde_json::from_str(
            r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": { "group_size": 64, "bits": 8 }
        }"#,
        )
        .unwrap();

        let result = sanitize_weights(weights, &config, true).unwrap();
        let qi = result.quant_info.as_ref().unwrap();

        assert!(
            qi.layers
                .contains_key("language_model.layers.0.self_attn.q_proj")
        );
        assert!(
            !qi.layers
                .contains_key("language_model.layers.0.self_attn.k_proj")
        );

        assert!(
            result
                .weights
                .contains_key("language_model.layers.0.self_attn.q_proj.inner.weight")
        );
        assert!(
            result
                .weights
                .contains_key("language_model.layers.0.self_attn.k_proj.weight")
        );
    }

    #[test]
    fn test_quantization_config_fallback() {
        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization_config": { "group_size": 32, "bits": 4 }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        assert!(config.quantization.is_none());
        let params = config.quantization_params().unwrap();
        assert_eq!(params.group_size, 32);
        assert_eq!(params.bits, 4);
        assert_eq!(params.mode, "affine");
    }

    #[test]
    fn test_unsupported_quant_mode() {
        let mut weights = HashMap::new();
        weights.insert(
            "language_model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "language_model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );

        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": { "group_size": 64, "bits": 8, "mode": "symmetric" }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights, &config, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("symmetric"),
            "error should mention the mode: {err}"
        );
    }

    #[test]
    fn test_per_layer_quantization_skip_with_scales_is_error() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::zeros::<u32>(&[248320, 256]).unwrap(),
        );
        weights.insert(
            "model.embed_tokens.scales".to_string(),
            Array::zeros::<f32>(&[248320, 16]).unwrap(),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );

        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": false },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.embed_tokens": false
            }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights, &config, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("skip"), "error should mention skip: {err}");
        assert!(
            err.contains("embed_tokens"),
            "error should mention layer: {err}"
        );
    }

    #[test]
    fn test_per_layer_quantization_override() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Array::zeros::<u32>(&[6144, 256]).unwrap(),
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.scales".to_string(),
            Array::zeros::<f32>(&[6144, 64]).unwrap(),
        );

        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.layers.0.mlp.gate_proj": { "group_size": 32, "bits": 4 }
            }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights, &config, true).unwrap();
        let qi = result.quant_info.as_ref().unwrap();

        let q_proj = qi
            .layers
            .get("language_model.layers.0.self_attn.q_proj")
            .unwrap();
        assert_eq!(q_proj.group_size, 64);
        assert_eq!(q_proj.bits, 8);

        let gate = qi
            .layers
            .get("language_model.layers.0.mlp.gate_proj")
            .unwrap();
        assert_eq!(gate.group_size, 32);
        assert_eq!(gate.bits, 4);
    }

    #[test]
    fn test_per_layer_unsupported_mode() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );

        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.layers.0.self_attn.q_proj": { "group_size": 32, "bits": 4, "mode": "symmetric" }
            }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights, &config, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("symmetric"),
            "error should mention mode: {err}"
        );
        assert!(err.contains("q_proj"), "error should mention layer: {err}");
    }

    #[test]
    fn test_per_layer_skip_with_scales_is_error() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::zeros::<f32>(&[248320, 2048]).unwrap(),
        );
        weights.insert(
            "model.embed_tokens.scales".to_string(),
            Array::zeros::<f32>(&[248320, 16]).unwrap(),
        );

        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": false },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.embed_tokens": false
            }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights, &config, true);
        assert!(result.is_err(), "skip + .scales should be an error");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("skip"), "error should mention skip: {err}");
    }

    #[test]
    fn test_per_layer_missing_fields() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );

        // missing bits
        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.layers.0.self_attn.q_proj": { "group_size": 32 }
            }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights.clone(), &config, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("bits"),
            "error should mention missing field: {err}"
        );

        // missing group_size
        let json2 = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.layers.0.self_attn.q_proj": { "bits": 4 }
            }
        }"#;
        let config2: Qwen35Configuration = serde_json::from_str(json2).unwrap();
        let result2 = sanitize_weights(weights, &config2, true);
        assert!(result2.is_err());
        let err2 = result2.unwrap_err().to_string();
        assert!(
            err2.contains("group_size"),
            "error should mention missing field: {err2}"
        );
    }

    #[test]
    fn test_per_layer_invalid_type() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::zeros::<u32>(&[4096, 512]).unwrap(),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            Array::zeros::<f32>(&[4096, 32]).unwrap(),
        );

        let json = r#"{
            "model_type": "qwen3_5_vl",
            "text_config": { "tie_word_embeddings": true },
            "vision_config": {},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.layers.0.self_attn.q_proj": "keep"
            }
        }"#;
        let config: Qwen35Configuration = serde_json::from_str(json).unwrap();
        let result = sanitize_weights(weights, &config, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("string"),
            "error should mention the invalid type: {err}"
        );
    }

    #[test]
    fn test_vision_sanitize_removes_position_ids() {
        let mut weights = HashMap::new();
        weights.insert(
            "vision_tower.blocks.0.attn.qkv.weight".to_string(),
            Array::zeros::<f32>(&[3072, 1024]).unwrap(),
        );
        weights.insert(
            "vision_tower.position_ids".to_string(),
            Array::zeros::<i32>(&[1, 256]).unwrap(),
        );
        weights.insert(
            "vision_tower.blocks.0.position_ids".to_string(),
            Array::zeros::<i32>(&[1, 64]).unwrap(),
        );

        let result = sanitize_vision_weights(weights, 3).unwrap();
        assert!(result.contains_key("vision_tower.blocks.0.attn.qkv.weight"));
        assert!(!result.keys().any(|k| k.contains("position_ids")));
    }

    #[test]
    fn test_vision_sanitize_transposes_patch_embed() {
        // PyTorch format: [out, in, D, H, W] — needs transpose
        let weights_pytorch = HashMap::from([(
            "vision_tower.patch_embed.proj.weight".to_string(),
            Array::zeros::<f32>(&[1024, 3, 2, 16, 16]).unwrap(),
        )]);
        let result = sanitize_vision_weights(weights_pytorch, 3).unwrap();
        let w = result.get("vision_tower.patch_embed.proj.weight").unwrap();
        assert_eq!(w.shape(), &[1024, 2, 16, 16, 3]);

        // Already channels-last: [out, D, H, W, in] — no transpose
        let weights_mlx = HashMap::from([(
            "vision_tower.patch_embed.proj.weight".to_string(),
            Array::zeros::<f32>(&[1024, 2, 16, 16, 3]).unwrap(),
        )]);
        let result = sanitize_vision_weights(weights_mlx, 3).unwrap();
        let w = result.get("vision_tower.patch_embed.proj.weight").unwrap();
        assert_eq!(w.shape(), &[1024, 2, 16, 16, 3]);
    }

    #[test]
    fn test_vision_sanitize_unexpected_ndim() {
        let weights = HashMap::from([(
            "vision_tower.patch_embed.proj.weight".to_string(),
            Array::zeros::<f32>(&[1024, 3, 16, 16]).unwrap(),
        )]);
        let result = sanitize_vision_weights(weights, 3);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("4D"), "error should mention ndim: {err}");
    }

    /// Unique temp dir per test. Removed on drop.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "mlx-rs-vlm-test-{}-{}-{}",
                name,
                std::process::id(),
                id
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn save_fixture(
        path: &Path,
        arrays: &[(&str, Array)],
        metadata: Option<&HashMap<String, String>>,
    ) {
        let arrays: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (*k, v)).collect();
        Array::save_safetensors(arrays, metadata, path).unwrap();
    }

    #[test]
    fn test_load_weights_roundtrip_with_mlx_metadata() {
        let dir = TempDir::new("mlx-meta");
        let metadata = HashMap::from([("format".to_string(), "mlx".to_string())]);
        save_fixture(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "a.weight",
                    Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
                ),
                ("b.scales", Array::from_slice(&[5i32, 6], &[2])),
            ],
            Some(&metadata),
        );

        let loaded = load_weights(dir.path()).unwrap();
        assert!(loaded.is_mlx_format);
        assert_eq!(loaded.weights.len(), 2);

        let a = loaded.weights.get("a.weight").unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.dtype(), mlx::Dtype::Float32);
        assert_eq!(a.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);

        let b = loaded.weights.get("b.scales").unwrap();
        assert_eq!(b.dtype(), mlx::Dtype::Int32);
        assert_eq!(b.as_slice::<i32>(), &[5, 6]);
    }

    #[test]
    fn test_load_weights_without_metadata_is_not_mlx_format() {
        let dir = TempDir::new("no-meta");
        save_fixture(
            &dir.path().join("model.safetensors"),
            &[("a.weight", Array::from_slice(&[1.0f32], &[1]))],
            None,
        );

        let loaded = load_weights(dir.path()).unwrap();
        assert!(!loaded.is_mlx_format);
        assert_eq!(loaded.weights.len(), 1);
    }

    #[test]
    fn test_load_weights_merges_shards_and_any_mlx_flag_wins() {
        let dir = TempDir::new("shards");
        // 1 番目のファイルには metadata なし、2 番目に format=mlx。
        // どのファイルにあってもフラグが立つことを検証する。
        save_fixture(
            &dir.path().join("model-00001.safetensors"),
            &[("a.weight", Array::from_slice(&[1.0f32], &[1]))],
            None,
        );
        let metadata = HashMap::from([("format".to_string(), "MLX".to_string())]);
        save_fixture(
            &dir.path().join("model-00002.safetensors"),
            &[("b.weight", Array::from_slice(&[2.0f32], &[1]))],
            Some(&metadata),
        );

        let loaded = load_weights(dir.path()).unwrap();
        assert!(
            loaded.is_mlx_format,
            "format=MLX (case-insensitive) should set the flag"
        );
        assert_eq!(loaded.weights.len(), 2);
        assert!(loaded.weights.contains_key("a.weight"));
        assert!(loaded.weights.contains_key("b.weight"));
    }

    #[test]
    fn test_load_weights_empty_dir_is_error() {
        let dir = TempDir::new("empty");
        let result = load_weights(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("No .safetensors"),
            "error should mention missing files: {err}"
        );
    }
}
