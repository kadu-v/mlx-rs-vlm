use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;

use crate::config::TextConfiguration;
use crate::weights::QuantizationInfo;

use super::linear_like::LinearLike;

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Mlp {
    #[param]
    gate_proj: LinearLike,
    #[param]
    up_proj: LinearLike,
    #[param]
    down_proj: LinearLike,
}

impl Mlp {
    pub fn new(
        config: &TextConfiguration,
        prefix: &str,
        quant_info: Option<&QuantizationInfo>,
    ) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = LinearLike::new(hidden_size, intermediate_size, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.gate_proj"))))?;
        let up_proj = LinearLike::new(hidden_size, intermediate_size, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.up_proj"))))?;
        let down_proj = LinearLike::new(intermediate_size, hidden_size, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.down_proj"))))?;

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
