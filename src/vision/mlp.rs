use mlx::Array;
use mlx::error::Exception;
use mlx::macros::ModuleParameters;
use mlx::module::Module;
use mlx::nn::{Linear, gelu_fast_approximate};
use mlx_rs as mlx;

use crate::config::VisionConfiguration;

/// Vision MLP with GELU activation.
///
/// Two-layer MLP: hidden_size -> intermediate_size -> hidden_size
/// with GELU (fast approximation) activation between layers.
/// Both linear layers include bias.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx)]
pub struct VisionMlp {
    #[param]
    pub linear_fc1: Linear,

    #[param]
    pub linear_fc2: Linear,
}

impl VisionMlp {
    /// Create a new VisionMlp from vision configuration.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        // Both linear layers have bias (default true)
        let linear_fc1 = Linear::new(config.hidden_size, config.intermediate_size)?;
        let linear_fc2 = Linear::new(config.intermediate_size, config.hidden_size)?;

        Ok(Self {
            linear_fc1,
            linear_fc2,
        })
    }

    /// Create a new VisionMlp with explicit dimensions.
    pub fn new_with_dims(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let linear_fc1 = Linear::new(dim, hidden_dim)?;
        let linear_fc2 = Linear::new(hidden_dim, dim)?;

        Ok(Self {
            linear_fc1,
            linear_fc2,
        })
    }
}

impl Module<&Array> for VisionMlp {
    type Error = Exception;
    type Output = Array;

    /// Forward pass: linear2(gelu(linear1(x)))
    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let h = self.linear_fc1.forward(x)?;
        let h = gelu_fast_approximate(&h)?;
        self.linear_fc2.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}
