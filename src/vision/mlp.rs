use mlx::Array;
use mlx::error::Exception;
use mlx::macros::ModuleParameters;
use mlx::module::Module;
use mlx_rs as mlx;

use crate::config::VisionConfiguration;

/// Vision MLP with GELU activation.
///
/// Two-layer MLP: hidden_size -> intermediate_size -> hidden_size
/// with tanh-approximate GELU activation between layers.
/// Both linear layers include bias.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx)]
pub struct VisionMlp {
    #[param]
    pub linear_fc1: mlx::nn::Linear,

    #[param]
    pub linear_fc2: mlx::nn::Linear,
}

impl VisionMlp {
    /// Create a new VisionMlp from vision configuration.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        // Both linear layers have bias (default true)
        let linear_fc1 = mlx::nn::Linear::new(config.hidden_size, config.intermediate_size)?;
        let linear_fc2 = mlx::nn::Linear::new(config.intermediate_size, config.hidden_size)?;

        Ok(Self {
            linear_fc1,
            linear_fc2,
        })
    }

    /// Create a new VisionMlp with explicit dimensions.
    pub fn new_with_dims(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let linear_fc1 = mlx::nn::Linear::new(dim, hidden_dim)?;
        let linear_fc2 = mlx::nn::Linear::new(hidden_dim, dim)?;

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
        let h = mlx::nn::gelu_approximate(&h)?;
        self.linear_fc2.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear_fc1.training_mode(mode);
        self.linear_fc2.training_mode(mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_mlp_matches_gelu_pytorch_tanh() {
        let mut mlp = VisionMlp::new_with_dims(1, 1).unwrap();
        for linear in [&mut mlp.linear_fc1, &mut mlp.linear_fc2] {
            linear.weight.value = Array::from_slice(&[1.0f32], &[1, 1]);
            linear.bias.value = None;
        }

        let input = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5, 1]);
        let output = mlp.forward(&input).unwrap();
        let expected = [-0.0454023f32, -0.158808, 0.0, 0.841192, 1.9545977];
        for (&actual, expected) in output.as_slice::<f32>().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected tanh GELU {expected}, got {actual}"
            );
        }
    }
}
