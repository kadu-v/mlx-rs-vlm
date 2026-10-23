pub mod attention;
pub mod decoder;
pub mod gated_delta_net;
pub mod linear_like;
pub mod mlp;
pub mod rope;
use mlx_rs as mlx;

use mlx::Array;
use mlx::builder::Builder;
use mlx::error::Exception;
use mlx::macros::ModuleParameters;
use mlx::module::Module;
use mlx::nn::{RmsNorm, RmsNormBuilder};

use crate::cache::{self, Cache, KVCacheSimple, MambaCache};
use crate::config::TextConfiguration;
use crate::weights::QuantizationInfo;

use self::decoder::DecoderLayer;
use self::linear_like::{EmbeddingLike, LinearLike};

#[derive(Debug, ModuleParameters)]
#[module(root = mlx)]
pub struct LanguageModel {
    #[param]
    pub embed_tokens: EmbeddingLike,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: RmsNorm,
    #[param]
    pub lm_head: Option<LinearLike>,
    pub config: TextConfiguration,
}

impl LanguageModel {
    pub fn new(
        config: &TextConfiguration,
        prefix: &str,
        quant_info: Option<&QuantizationInfo>,
    ) -> Result<Self, Exception> {
        let embed_tokens = EmbeddingLike::new(
            config.vocab_size,
            config.hidden_size,
            quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.embed_tokens"))),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{i}");
            layers.push(DecoderLayer::new(config, i, &layer_prefix, quant_info)?);
        }

        let norm = RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        let lm_head = if !config.tie_word_embeddings {
            let quant = quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.lm_head")));
            Some(LinearLike::new(
                config.hidden_size,
                config.vocab_size,
                false,
                quant,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
        })
    }

    pub fn forward(
        &mut self,
        inputs: Option<&Array>,
        inputs_embeds: Option<&Array>,
        cache: &mut [Cache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut h = match (inputs, inputs_embeds) {
            (Some(ids), None) => self.embed_tokens.forward(ids)?,
            (None, Some(embeds)) => embeds.clone(),
            _ => {
                return Err(Exception::custom(
                    "Exactly one of inputs or inputs_embeds must be provided",
                ));
            }
        };

        let t = h.dim(1);
        let fa_idx = self.first_full_attention_layer_idx();
        let fa_offset = if fa_idx < cache.len() {
            match &cache[fa_idx] {
                Cache::KV(kv) => kv.offset() as i32,
                _ => 0,
            }
        } else {
            0
        };
        let attn_mask = cache::create_causal_mask(t, fa_offset)?;

        let ssm_idx = self.first_linear_layer_idx();
        let ssm_mask = if ssm_idx < cache.len() {
            self.create_ssm_mask(&h, &cache[ssm_idx])?
        } else {
            None
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let (am, sm) = if layer.is_linear() {
                (None, ssm_mask.as_ref())
            } else {
                (attn_mask.as_ref(), None)
            };
            h = layer.forward(&h, am, sm, &mut cache[i], position_ids)?;
        }

        h = self.norm.forward(&h)?;

        match &mut self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    pub fn make_cache(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_linear() {
                    Cache::Mamba(MambaCache::new())
                } else {
                    Cache::KV(KVCacheSimple::new())
                }
            })
            .collect()
    }

    fn first_full_attention_layer_idx(&self) -> usize {
        self.layers.iter().position(|l| !l.is_linear()).unwrap_or(0)
    }

    fn first_linear_layer_idx(&self) -> usize {
        self.layers.iter().position(|l| l.is_linear()).unwrap_or(0)
    }

    fn create_ssm_mask(&self, h: &Array, cache: &Cache) -> Result<Option<Array>, Exception> {
        let n = h.dim(1);
        match cache {
            Cache::Mamba(mamba) => {
                if mamba.offset() > 0 || n > 1 {
                    Ok(Some(Array::ones::<bool>(&[h.dim(0), n])?))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}
