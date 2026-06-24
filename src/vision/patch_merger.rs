use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn::{gelu, LayerNorm, Linear};
use mlx_rs::Array;

use crate::config::VisionConfiguration;

/// Patch merger - spatial merge and projection.
///
/// Merges spatial_merge_size x spatial_merge_size patches into single tokens
/// and projects them to the output hidden dimension.
///
/// Pipeline: norm -> reshape/merge -> linear1 -> gelu -> linear2
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct PatchMerger {
    #[param]
    pub norm: LayerNorm,

    #[param]
    pub linear_fc1: Linear,

    #[param]
    pub linear_fc2: Linear,

    pub merge_size: i32,
    pub hidden_size: i32,
    pub use_post_shuffle_norm: bool,
}

impl PatchMerger {
    /// Create a new PatchMerger from vision configuration.
    ///
    /// The merged dimension is hidden_size * merge_size^2.
    /// - norm: normalizes input features
    /// - linear1: merged_dim -> merged_dim (with bias)
    /// - linear2: merged_dim -> out_hidden_size (with bias)
    pub fn new(config: &VisionConfiguration, use_post_shuffle_norm: bool) -> Result<Self, Exception> {
        let merge_size = config.spatial_merge_size;
        let merged_dim = config.hidden_size * merge_size * merge_size;

        let norm_dim = if use_post_shuffle_norm {
            merged_dim
        } else {
            config.hidden_size
        };

        let norm = mlx_rs::nn::LayerNormBuilder::new(norm_dim)
            .eps(1e-6)
            .build()?;

        let linear_fc1 = Linear::new(merged_dim, merged_dim)?;
        let linear_fc2 = Linear::new(merged_dim, config.out_hidden_size)?;

        Ok(Self {
            norm,
            linear_fc1,
            linear_fc2,
            merge_size,
            hidden_size: merged_dim,
            use_post_shuffle_norm,
        })
    }
}

impl Module<&Array> for PatchMerger {
    type Error = Exception;
    type Output = Array;

    /// Forward pass.
    ///
    /// Input: [seq_len, hidden_size] where seq_len includes spatial merge groups.
    ///
    /// 1. Apply norm (pre or post shuffle depending on configuration)
    /// 2. Reshape to merge spatial patches: [-1, hidden_size * merge_size^2]
    /// 3. linear1 -> gelu -> linear2
    ///
    /// Output: [new_seq_len, out_hidden_size]
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut states = x.clone();

        if self.use_post_shuffle_norm {
            // Post-shuffle: reshape first, then normalize
            states = states.reshape(&[-1, self.hidden_size])?;
        }

        // Apply layer normalization
        states = self.norm.forward(&states)?;

        // Reshape to merge patches: [-1, merged_dim]
        states = states.reshape(&[-1, self.hidden_size])?;

        // MLP: linear1 -> gelu -> linear2
        states = self.linear_fc1.forward(&states)?;
        states = gelu(&states)?;
        states = self.linear_fc2.forward(&states)?;

        Ok(states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}
