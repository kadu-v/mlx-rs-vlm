use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn::{RmsNorm, RmsNormBuilder};

use crate::cache::Cache;
use crate::config::TextConfiguration;
use crate::weights::QuantizationInfo;

use super::attention::Attention;
use super::gated_delta_net::GatedDeltaNet;
use super::mlp::Mlp;

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DecoderLayer {
    is_linear: bool,
    #[param]
    self_attn: Option<Attention>,
    #[param]
    linear_attn: Option<GatedDeltaNet>,
    #[param]
    input_layernorm: RmsNorm,
    #[param]
    post_attention_layernorm: RmsNorm,
    #[param]
    mlp: Mlp,
}

impl DecoderLayer {
    pub fn new(
        config: &TextConfiguration,
        layer_idx: i32,
        prefix: &str,
        quant_info: Option<&QuantizationInfo>,
    ) -> Result<Self, Exception> {
        let is_linear = config.is_linear_layer(layer_idx);

        let self_attn = if !is_linear {
            Some(Attention::new(config, &format!("{prefix}.self_attn"), quant_info)?)
        } else {
            None
        };

        let linear_attn = if is_linear {
            Some(GatedDeltaNet::new(config, &format!("{prefix}.linear_attn"), quant_info)?)
        } else {
            None
        };

        let input_layernorm = RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        if config.num_experts.unwrap_or(0) > 0 {
            return Err(mlx_rs::error::Exception::custom(
                "MoE (num_experts > 0) is not supported",
            ));
        }
        let mlp = Mlp::new(config, &format!("{prefix}.mlp"), quant_info)?;

        Ok(Self {
            is_linear,
            self_attn,
            linear_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn is_linear(&self) -> bool {
        self.is_linear
    }

    pub fn forward(
        &mut self,
        x: &Array,
        attn_mask: Option<&Array>,
        ssm_mask: Option<&Array>,
        cache: &mut Cache,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;

        let r = if self.is_linear {
            let mamba_cache = cache
                .as_mamba_mut()
                .ok_or_else(|| Exception::custom("Linear layer requires MambaCache"))?;
            self.linear_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("Linear layer missing linear_attn module"))?
                .forward(&normed, ssm_mask, mamba_cache)?
        } else {
            let kv_cache = cache
                .as_kv_mut()
                .ok_or_else(|| Exception::custom("Full attention layer requires KVCacheSimple"))?;
            let pos_ids = position_ids
                .ok_or_else(|| Exception::custom("Full attention layer requires position_ids"))?;
            self.self_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("Full attention layer missing self_attn module"))?
                .forward(&normed, attn_mask, kv_cache, pos_ids)?
        };

        let h = x.add(&r)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}
