use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::Array;
use safetensors::SafeTensors;

use crate::config::Qwen35Configuration;

pub struct LoadedWeights {
    pub weights: HashMap<String, Array>,
    pub is_mlx_format: bool,
}

pub fn load_weights(model_dir: &Path) -> Result<LoadedWeights> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut all_weights = HashMap::new();
    let mut is_mlx_format = false;

    if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path)
            .context("Failed to read model.safetensors.index.json")?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;

        let weight_map = index["weight_map"]
            .as_object()
            .context("Missing weight_map in index")?;

        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();

        for file in &files {
            let file_path = model_dir.join(file);
            let data = std::fs::read(&file_path)
                .with_context(|| format!("Failed to read {}", file_path.display()))?;
            let tensors = SafeTensors::deserialize(&data)
                .with_context(|| format!("Failed to parse {}", file))?;

            if !is_mlx_format {
                let (_, raw_meta) = SafeTensors::read_metadata(&data)
                    .with_context(|| format!("Failed to read metadata from {}", file))?;
                if let Some(meta) = raw_meta.metadata() {
                    if meta.get("format").map(|f| f.eq_ignore_ascii_case("mlx")) == Some(true) {
                        is_mlx_format = true;
                    }
                }
            }

            for (name, tensor_view) in tensors.tensors() {
                let shape: Vec<i32> = tensor_view.shape().iter().map(|&s| s as i32).collect();
                let data = tensor_view.data();
                let dtype = safetensors_dtype_to_mlx(tensor_view.dtype())?;
                let array = unsafe {
                    Array::from_raw_data(data.as_ptr() as *const std::ffi::c_void, &shape, dtype)
                };
                all_weights.insert(name.to_string(), array);
            }
        }
    } else {
        let single_path = model_dir.join("model.safetensors");
        let data = std::fs::read(&single_path)
            .context("Failed to read model.safetensors")?;
        let tensors = SafeTensors::deserialize(&data)?;

        let (_, raw_meta) = SafeTensors::read_metadata(&data)
            .context("Failed to read safetensors metadata")?;
        if let Some(meta) = raw_meta.metadata() {
            if meta.get("format").map(|f| f.eq_ignore_ascii_case("mlx")) == Some(true) {
                is_mlx_format = true;
            }
        }

        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<i32> = tensor_view.shape().iter().map(|&s| s as i32).collect();
            let data = tensor_view.data();
            let dtype = safetensors_dtype_to_mlx(tensor_view.dtype())?;
            let array = unsafe {
                Array::from_raw_data(data.as_ptr() as *const std::ffi::c_void, &shape, dtype)
            };
            all_weights.insert(name.to_string(), array);
        }
    }

    Ok(LoadedWeights {
        weights: all_weights,
        is_mlx_format,
    })
}

fn safetensors_dtype_to_mlx(dtype: safetensors::Dtype) -> Result<mlx_rs::Dtype> {
    match dtype {
        safetensors::Dtype::F32 => Ok(mlx_rs::Dtype::Float32),
        safetensors::Dtype::F16 => Ok(mlx_rs::Dtype::Float16),
        safetensors::Dtype::BF16 => Ok(mlx_rs::Dtype::Bfloat16),
        safetensors::Dtype::I32 => Ok(mlx_rs::Dtype::Int32),
        safetensors::Dtype::I64 => Ok(mlx_rs::Dtype::Int64),
        safetensors::Dtype::I16 => Ok(mlx_rs::Dtype::Int16),
        safetensors::Dtype::I8 => Ok(mlx_rs::Dtype::Int8),
        safetensors::Dtype::U8 => Ok(mlx_rs::Dtype::Uint8),
        safetensors::Dtype::U16 => Ok(mlx_rs::Dtype::Uint16),
        safetensors::Dtype::U32 => Ok(mlx_rs::Dtype::Uint32),
        safetensors::Dtype::BOOL => Ok(mlx_rs::Dtype::Bool),
        other => anyhow::bail!("unsupported safetensors dtype: {:?}", other),
    }
}

/// Sanitize and remap weight keys to match our module parameter hierarchy.
///
/// Remapping:
/// - `language_model.model.X` → `language_model.X` (strip extra `.model.`)
/// - `A_log` → `a_log` (case fix)
/// - conv1d.weight: reorder axes [D_out, 1, K] → [D_out, K, 1]
/// - q_norm/k_norm weights: +1.0
/// - Quantized layers: `.weight` → `.inner.weight` (detected by `.scales` sibling)
/// - mtp.* excluded
/// - tie_word_embeddings: lm_head excluded
pub fn sanitize_weights(
    weights: HashMap<String, Array>,
    config: &Qwen35Configuration,
    is_mlx_format: bool,
) -> HashMap<String, Array> {
    let norm_keys = [
        "q_norm.weight",
        "k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ];

    // First pass: collect all keys, apply basic renames, norms, conv fixes
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

        // Fix A_log casing
        key = key.replace(".A_log", ".a_log");

        if !is_mlx_format {
            // Conv1d weight axis reorder (HF [D,1,K] → MLX [D,K,1])
            if key.contains("conv1d.weight") && value.ndim() == 3 {
                let last_dim = value.shape()[2];
                if last_dim != 1 {
                    value = value.move_axis(2, 1).unwrap_or(value);
                }
            }

            // Norm weight +1.0 adjustment (HF stores deviation from 1)
            if (norm_keys.iter().any(|nk| key.ends_with(nk))
                || key == "language_model.norm.weight")
                && value.ndim() == 1
            {
                value = value.add(&Array::from(1.0f32)).unwrap_or(value);
            }
        }

        intermediate.insert(key, value);
    }

    // Second pass: add `inner.` prefix for quantized layer weights
    // A weight key belongs to a quantized layer if a corresponding `.scales` key exists
    let quantized_prefixes: Vec<String> = intermediate
        .keys()
        .filter(|k| k.ends_with(".scales"))
        .map(|k| k.strip_suffix(".scales").unwrap().to_string())
        .collect();

    let mut result = HashMap::new();
    for (key, value) in intermediate {
        let mut new_key = key.clone();

        for prefix in &quantized_prefixes {
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
        // Add quantized layer test data
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
        // A_log test
        weights.insert(
            "language_model.model.layers.0.linear_attn.A_log".to_string(),
            Array::zeros::<f32>(&[16]).unwrap(),
        );
        // Additional norm weights
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
            &std::fs::read_to_string("models/say2_q35_2B_mlx_8bit_20260522/config.json").unwrap(),
        )
        .unwrap();

        let sanitized = sanitize_weights(weights, &config, false);

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

        // Check quantized key remapping
        assert!(sanitized.contains_key("language_model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(sanitized.contains_key("language_model.layers.0.self_attn.q_proj.scales"));
        assert!(sanitized.contains_key("language_model.layers.0.self_attn.q_proj.biases"));

        // Check A_log → a_log
        assert!(sanitized.contains_key("language_model.layers.0.linear_attn.a_log"));

        // Check input_layernorm +1.0
        let ln_w = sanitized
            .get("language_model.layers.0.input_layernorm.weight")
            .unwrap();
        let ln_mean: f32 = ln_w.mean(None).unwrap().item();
        assert!((ln_mean - 1.0).abs() < 1e-5, "input_layernorm should be +1.0");

        // Check post_attention_layernorm +1.0
        let pln_w = sanitized
            .get("language_model.layers.0.post_attention_layernorm.weight")
            .unwrap();
        let pln_mean: f32 = pln_w.mean(None).unwrap().item();
        assert!((pln_mean - 1.0).abs() < 1e-5, "post_attention_layernorm should be +1.0");

        // Check language_model.norm.weight +1.0
        let final_norm = sanitized
            .get("language_model.norm.weight")
            .unwrap();
        let fn_mean: f32 = final_norm.mean(None).unwrap().item();
        assert!((fn_mean - 1.0).abs() < 1e-5, "final norm should be +1.0");
    }
}
