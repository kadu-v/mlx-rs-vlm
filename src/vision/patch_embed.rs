use mlx::Array;
use mlx::builder::Builder;
use mlx::error::Exception;
use mlx::macros::ModuleParameters;
use mlx::module::Module;
use mlx::nn::Conv3d;
use mlx::utils::SingleOrTriple;
use mlx_rs as mlx;

use crate::config::VisionConfiguration;

/// Patch embedding using 3D convolution.
///
/// Extracts patches from image frames using a 3D convolution where the kernel
/// spans [temporal_patch_size, patch_size, patch_size] and stride equals the kernel size.
///
/// The Conv3d in mlx-rs expects channels-last input: [N, D, H, W, C].
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx)]
pub struct PatchEmbed {
    #[param]
    pub proj: Conv3d,

    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub hidden_size: i32,
}

impl PatchEmbed {
    /// Create a new PatchEmbed layer from vision configuration.
    ///
    /// Conv3d: in_channels -> hidden_size, kernel=[temporal_patch_size, patch_size, patch_size],
    /// stride=kernel_size, with bias.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        let kernel = SingleOrTriple::Triple(
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        );

        let proj = mlx::nn::Conv3dBuilder::new(config.in_channels, config.hidden_size, kernel)
            .stride(kernel)
            .bias(true)
            .build()?;

        Ok(Self {
            proj,
            patch_size: config.patch_size,
            temporal_patch_size: config.temporal_patch_size,
            in_channels: config.in_channels,
            hidden_size: config.hidden_size,
        })
    }
}

impl Module<&Array> for PatchEmbed {
    type Error = Exception;
    type Output = Array;

    /// Forward pass.
    ///
    /// Input: flattened pixel values [num_patches * in_channels * temporal * patch_h * patch_w]
    /// 1. Reshape to [batch, in_channels, temporal, patch_h, patch_w]
    /// 2. Move channels from axis 1 to axis 4 for channels-last: [batch, temporal, patch_h, patch_w, in_channels]
    /// 3. Apply Conv3d projection
    /// 4. Reshape output to [-1, hidden_size]
    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Self::Error> {
        // Reshape to [batch, in_channels, temporal_patch_size, patch_size, patch_size]
        let mut states = hidden_states.reshape(&[
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ])?;

        // Move channels from axis 1 to axis 4: [batch, T, H, W, C]
        states = states.move_axis(1, 4)?;

        // Apply conv3d projection
        states = self.proj.forward(&states)?;

        // Reshape to [-1, hidden_size]
        states = states.reshape(&[-1, self.hidden_size])?;

        Ok(states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}
