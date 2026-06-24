use mlx_rs::error::Exception;
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::{
    concatenate_axis, cos, expand_dims, negative, ones, sin, split,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::config::VisionConfiguration;

/// Rotate half of the hidden dims: [-x2, x1] where x = [x1, x2]
fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let half = x.dim(-1) / 2;
    let x1 = x.index((.., .., 0..half));
    let x2 = x.index((.., .., half..));
    let neg_x2 = negative(&x2)?;
    concatenate_axis(&[&neg_x2, &x1], -1)
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
    let mut cos_vals = cos(freqs)?;
    let mut sin_vals = sin(freqs)?;

    // Expand dims and tile to match tensor shape:
    // [seq_len, dim/2] -> [seq_len, 1, dim/2] -> [seq_len, 1, dim] -> [1, seq_len, 1, dim]
    cos_vals = expand_dims(&cos_vals, 1)?;
    cos_vals = mlx_rs::ops::tile(&cos_vals, &[1, 1, 2])?;
    cos_vals = expand_dims(&cos_vals, 0)?;

    sin_vals = expand_dims(&sin_vals, 1)?;
    sin_vals = mlx_rs::ops::tile(&sin_vals, &[1, 1, 2])?;
    sin_vals = expand_dims(&sin_vals, 0)?;

    // rotated = tensor * cos + rotate_half(tensor) * sin
    let rotated_half = rotate_half(tensor)?;
    let result = &(tensor * &cos_vals) + &(&rotated_half * &sin_vals);
    result.as_dtype(tensor.dtype())
}

/// Vision attention with rotary embeddings and block-diagonal masking.
///
/// Uses a combined QKV projection, applies rotary embeddings,
/// and uses block-diagonal attention masks derived from cumulative
/// sequence lengths for variable-length sequences.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    /// Combined Q, K, V projection: hidden_size -> 3 * hidden_size
    #[param]
    pub qkv: Linear,

    /// Output projection: hidden_size -> hidden_size
    #[param]
    pub proj: Linear,
}

impl VisionAttention {
    /// Create a new VisionAttention from vision configuration.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        let dim = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        // QKV projection with bias
        let qkv = Linear::new(dim, 3 * dim)?;
        // Output projection with bias
        let proj = Linear::new(dim, dim)?;

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
        let parts = split(&qkv_states, 3, Some(0))?;
        // Remove the leading dimension: [seq_len, num_heads, head_dim]
        let queries = parts[0].index(0);
        let keys = parts[1].index(0);
        let values = parts[2].index(0);

        // Apply rotary embeddings to queries and keys
        let queries = apply_rotary(&queries, rot_pos_emb)?;
        let keys = apply_rotary(&keys, rot_pos_emb)?;

        // Reshape for scaled_dot_product_attention: [1, num_heads, seq_len, head_dim]
        let queries =
            queries
                .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Build block-diagonal attention mask from cumulative sequence lengths.
        // Initialize mask to -1e9 (large negative = masked out)
        let mask = &ones::<f32>(&[1, seq_len, seq_len])?
            * &Array::from_f32(-1e9);

        // Read cu_seqlens into a Vec and set block-diagonal entries to 0.
        // Each block [start..end, start..end] corresponds to one image/video sequence.
        let mask = build_block_diagonal_mask(mask, cu_seqlens, seq_len)?;

        // Scaled dot-product attention
        let attended = scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            self.scale,
            ScaledDotProductAttentionMask::Array(&mask),
            None::<&Array>,
        )?;

        // Transpose back and reshape: [1, num_heads, seq_len, head_dim] -> [seq_len, hidden_size]
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[seq_len, -1])?;

        // Output projection
        self.proj.forward(&attended)
    }
}

/// Build a block-diagonal attention mask from cumulative sequence lengths.
///
/// The mask starts as all -1e9, and for each segment [start..end] defined by
/// consecutive elements in cu_seqlens, the entries mask[start..end, start..end]
/// are set to 0 (unmasked).
fn build_block_diagonal_mask(
    mask: Array,
    cu_seqlens: &Array,
    seq_len: i32,
) -> Result<Array, Exception> {
    use mlx_rs::ops::indexing::IndexMutOp;

    let mut mask = mask;
    let n_seqs = cu_seqlens.dim(0);

    // Read cumulative sequence lengths as i32 values
    cu_seqlens.eval()?;
    let seqlens_slice = cu_seqlens.as_slice::<i32>();

    for idx in 1..n_seqs {
        let start = seqlens_slice[idx as usize - 1];
        let end = seqlens_slice[idx as usize];
        if start < end && end <= seq_len {
            let zero = Array::from_f32(0.0);
            mask.index_mut((.., start..end, start..end), zero);
        }
    }

    Ok(mask)
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
