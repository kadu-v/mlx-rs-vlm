use mlx::Array;
use mlx::error::Exception;
use mlx::fast::{ScaledDotProductAttentionMask, scaled_dot_product_attention};
use mlx::macros::ModuleParameters;
use mlx::module::Module;
use mlx::ops::indexing::IndexOp;
use mlx_rs as mlx;

use crate::config::VisionConfiguration;

/// Rotate half of the hidden dims: [-x2, x1] where x = [x1, x2]
fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let half = x.dim(-1) / 2;
    let x1 = x.index((.., .., 0..half));
    let x2 = x.index((.., .., half..));
    let neg_x2 = mlx::ops::negative(&x2)?;
    mlx::ops::concatenate(&[&neg_x2, &x1], -1)
}

/// Apply rotary position embeddings to a tensor.
///
/// `tensor` shape: [seq_len, num_heads, head_dim]
/// `freqs` shape: [seq_len, head_dim/2]
///
/// The frequency tensor is expanded and tiled to match the head dimensions,
/// then applied via: x * cos(freqs) + rotate_half(x) * sin(freqs)
fn apply_rotary(tensor: &Array, freqs: &Array) -> Result<Array, Exception> {
    // freqs: [seq_len, dim/2]
    let mut cos_vals = mlx::ops::cos(freqs)?;
    let mut sin_vals = mlx::ops::sin(freqs)?;

    // Expand dims and tile to match tensor shape:
    // [seq_len, dim/2] -> [seq_len, 1, dim/2] -> [seq_len, 1, dim] -> [1, seq_len, 1, dim]
    cos_vals = mlx::ops::expand_dims(&cos_vals, 1)?;
    cos_vals = mlx::ops::tile(&cos_vals, &[1, 1, 2])?;
    cos_vals = mlx::ops::expand_dims(&cos_vals, 0)?;

    sin_vals = mlx::ops::expand_dims(&sin_vals, 1)?;
    sin_vals = mlx::ops::tile(&sin_vals, &[1, 1, 2])?;
    sin_vals = mlx::ops::expand_dims(&sin_vals, 0)?;

    // rotated = tensor * cos + rotate_half(tensor) * sin
    let rotated_half = rotate_half(tensor)?;
    let result = &(tensor * &cos_vals) + &(&rotated_half * &sin_vals);
    result.as_dtype(tensor.dtype())
}

/// Vision attention with rotary embeddings and block-diagonal attention.
///
/// Uses a combined QKV projection, applies rotary embeddings, and attends
/// each image (block defined by cumulative sequence lengths) independently
/// with the fused SDPA kernel — no dense mask is materialized.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx)]
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    /// Combined Q, K, V projection: hidden_size -> 3 * hidden_size
    #[param]
    pub qkv: mlx::nn::Linear,

    /// Output projection: hidden_size -> hidden_size
    #[param]
    pub proj: mlx::nn::Linear,
}

impl VisionAttention {
    /// Create a new VisionAttention from vision configuration.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        let dim = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        // QKV projection with bias
        let qkv = mlx::nn::Linear::new(dim, 3 * dim)?;
        // Output projection with bias
        let proj = mlx::nn::Linear::new(dim, dim)?;

        Ok(Self {
            num_heads,
            head_dim,
            scale,
            qkv,
            proj,
        })
    }

    /// Forward pass with rotary position embeddings and block-diagonal attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [seq_len, hidden_size]
    /// * `cu_seqlens` - Cumulative sequence lengths as an Array for block-diagonal masking
    /// * `rot_pos_emb` - Rotary position embeddings of shape [seq_len, head_dim/2]
    pub fn forward_with_rope(
        &mut self,
        x: &Array,
        cu_seqlens: &Array,
        rot_pos_emb: &Array,
    ) -> Result<Array, Exception> {
        let seq_len = x.dim(0);

        // Combined QKV projection
        let mut qkv_states = self.qkv.forward(x)?;

        // Reshape to [seq_len, 3, num_heads, head_dim] then transpose to [3, seq_len, num_heads, head_dim]
        qkv_states = qkv_states.reshape(&[seq_len, 3, self.num_heads, self.head_dim])?;
        qkv_states = qkv_states.transpose_axes(&[1, 0, 2, 3])?;

        // Split into Q, K, V: each [1, seq_len, num_heads, head_dim]
        let parts = mlx::ops::split_equal(&qkv_states, 3, Some(0))?;
        // Remove the leading dimension: [seq_len, num_heads, head_dim]
        let queries = parts[0].index(0);
        let keys = parts[1].index(0);
        let values = parts[2].index(0);

        // Apply rotary embeddings to queries and keys
        let queries = apply_rotary(&queries, rot_pos_emb)?;
        let keys = apply_rotary(&keys, rot_pos_emb)?;

        // Reshape for scaled_dot_product_attention: [1, num_heads, seq_len, head_dim]
        let queries = queries
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Attention is block-diagonal over the images in cu_seqlens, so each image is
        // attended independently instead of materializing a dense [L, L] additive mask —
        // an explicit mask forces MLX's SDPA onto the fallback path, which materializes
        // the [1, heads, L, L] score matrix. Mirrors Qwen3VL.swift / mlx-vlm.
        cu_seqlens.eval()?;
        let seqlens_slice = cu_seqlens.as_slice::<i32>();
        let attended = if seqlens_slice.len() <= 2 {
            // Single image: one block covering the whole sequence — no slicing needed
            fused_sdpa(&queries, &keys, &values, self.scale)?
        } else {
            let mut outputs = Vec::with_capacity(seqlens_slice.len() - 1);
            for window in seqlens_slice.windows(2) {
                let (start, end) = (window[0], window[1]);
                if start >= end {
                    continue;
                }
                outputs.push(fused_sdpa(
                    &queries.index((.., .., start..end, ..)),
                    &keys.index((.., .., start..end, ..)),
                    &values.index((.., .., start..end, ..)),
                    self.scale,
                )?);
            }
            let output_refs: Vec<&Array> = outputs.iter().collect();
            mlx::ops::concatenate(&output_refs, 2)?
        };

        // Transpose back and reshape: [1, num_heads, seq_len, head_dim] -> [seq_len, hidden_size]
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[seq_len, -1])?;

        // Output projection
        self.proj.forward(&attended)
    }
}

/// SDPA that always dispatches MLX's O(L)-memory fused Metal kernel
/// (equivalent of `ensureFusedSDPA` in Qwen3VL.swift).
///
/// The fused kernel supports head dims 64/80/128; anything else falls back to
/// composed ops that materialize the [L, L] score matrix per head. Zero-padding
/// Q/K/V up to the next supported dim preserves the math exactly (padded dims
/// contribute nothing to the dot products; `scale` is passed explicitly).
fn fused_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
) -> Result<Array, Exception> {
    const FUSED_DIMS: [i32; 3] = [64, 80, 128];
    let head_dim = queries.dim(-1);
    let target = FUSED_DIMS
        .iter()
        .copied()
        .find(|&d| head_dim <= d)
        .unwrap_or(head_dim);

    if target == head_dim {
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            None::<ScaledDotProductAttentionMask>,
            None::<&Array>,
        );
    }

    let widths = [(0, 0), (0, 0), (0, 0), (0, target - head_dim)];
    let attended = scaled_dot_product_attention(
        mlx::ops::pad(queries, &widths[..], None::<Array>, None)?,
        mlx::ops::pad(keys, &widths[..], None::<Array>, None)?,
        mlx::ops::pad(values, &widths[..], None::<Array>, None)?,
        scale,
        None::<ScaledDotProductAttentionMask>,
        None::<&Array>,
    )?;
    Ok(attended.index((.., .., .., 0..head_dim)))
}

impl Module<&Array> for VisionAttention {
    type Error = Exception;
    type Output = Array;

    /// Default forward without rotary embeddings (not typically used for vision).
    /// Use `forward_with_rope` for the full vision attention path.
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        // This path is here to satisfy the Module trait.
        // In practice, VisionAttention should be called via forward_with_rope.
        self.qkv.forward(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.qkv.training_mode(mode);
        self.proj.training_mode(mode);
    }
}
