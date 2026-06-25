use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};

use crate::config::{Qwen35Configuration, THW};
use crate::model::Qwen35VLM;

pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub eos_token_ids: Vec<i32>,
}

impl GenerateParams {
    pub fn with_config(config: &Qwen35Configuration) -> Self {
        Self {
            eos_token_ids: config
                .eos_token_ids()
                .unwrap_or_else(|| vec![248046, 248044]),
            ..Default::default()
        }
    }
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            eos_token_ids: vec![248046, 248044],
        }
    }
}

pub fn generate(
    model: &mut Qwen35VLM,
    input_ids: &Array,
    pixel_values: Option<&Array>,
    grid_thw: Option<&[THW]>,
    params: &GenerateParams,
) -> Result<Vec<i32>, Exception> {
    let mut cache = model.make_cache();
    let mut generated_tokens = Vec::new();

    let logits = model.prepare(input_ids, pixel_values, grid_thw, &mut cache)?;

    let mut next_token = sample_token(&logits, params.temperature)?;
    generated_tokens.push(next_token);

    if params.eos_token_ids.contains(&next_token) {
        return Ok(generated_tokens);
    }

    for _ in 1..params.max_tokens {
        let token_array = Array::from_slice(&[next_token], &[1, 1]);
        let logits = model.forward_text(&token_array, &mut cache)?;
        next_token = sample_token(&logits, params.temperature)?;
        generated_tokens.push(next_token);

        if params.eos_token_ids.contains(&next_token) {
            break;
        }
    }

    Ok(generated_tokens)
}

fn sample_token(logits: &Array, temperature: f32) -> Result<i32, Exception> {
    let seq_len = logits.shape()[1];
    let last_logits = logits.index((.., seq_len - 1, ..));

    if temperature <= 0.0 || temperature < 1e-6 {
        let token = argmax_axis(&last_logits, -1, None)?;
        token.eval()?;
        let token: i32 = token.item();
        Ok(token)
    } else {
        let scaled = last_logits.multiply(&Array::from(1.0 / temperature))?;
        let probs = mlx_rs::ops::softmax_axis(&scaled, -1, None)?;
        let key = mlx_rs::random::key(0)?;
        let token = mlx_rs::random::categorical(&probs, None, None, &key)?;
        token.eval()?;
        let token: i32 = token.item();
        Ok(token)
    }
}
