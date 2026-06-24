use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn::{QuantizedLinear, QuantizedLinearBuilder};

use crate::config::TextConfiguration;

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Mlp {
    #[param]
    gate_proj: QuantizedLinear,
    #[param]
    up_proj: QuantizedLinear,
    #[param]
    down_proj: QuantizedLinear,
}

impl Mlp {
    pub fn new(config: &TextConfiguration, group_size: i32, bits: i32) -> Result<Self, Exception> {
        Self::with_dims(config.hidden_size, config.intermediate_size, group_size, bits)
    }

    pub fn with_dims(
        hidden_size: i32,
        intermediate_size: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        let gate_proj = QuantizedLinearBuilder::new(hidden_size, intermediate_size)
            .group_size(group_size)
            .bits(bits)
            .bias(false)
            .build()?;
        let up_proj = QuantizedLinearBuilder::new(hidden_size, intermediate_size)
            .group_size(group_size)
            .bits(bits)
            .bias(false)
            .build()?;
        let down_proj = QuantizedLinearBuilder::new(intermediate_size, hidden_size)
            .group_size(group_size)
            .bits(bits)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = mlx_rs::nn::silu(self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(up)?)
    }
}
