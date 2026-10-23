use mlx::Array;
use mlx::error::Exception;
use mlx_rs as mlx;

/// Vision rotary embedding.
///
/// Generates rotary position embeddings for vision attention.
/// Computes the outer product of position indices and inverse frequencies.
#[derive(Debug, Clone)]
pub struct VisionRotaryEmbedding {
    /// Dimension of the embedding (half of head_dim)
    pub dimension: i32,
    /// Base theta for the frequency computation
    pub theta: f32,
}

impl VisionRotaryEmbedding {
    /// Create a new VisionRotaryEmbedding.
    ///
    /// # Arguments
    /// * `dimension` - Dimension of the rotary embedding (typically head_dim / 2)
    /// * `theta` - Base theta for frequency computation (default 10000.0)
    pub fn new(dimension: i32, theta: f32) -> Self {
        Self { dimension, theta }
    }

    /// Compute rotary position embeddings for a given sequence length.
    ///
    /// Returns the outer product of positions [0..seq_len] and inverse frequencies,
    /// with shape [seq_len, dimension].
    pub fn forward(&self, seq_len: i32) -> Result<Array, Exception> {
        // inv_freq = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
        let exp_range = mlx::ops::arange::<_, f32>(None, self.dimension, Some(2))?;
        let dim_f = Array::from_f32(self.dimension as f32);
        let exp_range = &exp_range / &dim_f;
        let theta_arr = Array::from_f32(self.theta);
        let inv_freq = &Array::from_f32(1.0) / mlx::ops::power(&theta_arr, &exp_range)?;

        // positions = [0, 1, ..., seq_len - 1]
        let seq = mlx::ops::arange::<_, f32>(None, seq_len, None)?;

        // outer product: [seq_len, dim/2]
        mlx::ops::outer(&seq, &inv_freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_rotary_embedding() {
        let rope = VisionRotaryEmbedding::new(32, 10000.0);
        let emb = rope.forward(16).unwrap();
        assert_eq!(emb.shape(), &[16, 16]);
        emb.eval().unwrap();
        let vals: Vec<f32> = emb.as_slice::<f32>().to_vec();
        // Position 0: 0 * inv_freq = 0 for all frequencies
        assert!(vals[..16].iter().all(|v| v.abs() < 1e-7));
    }
}
