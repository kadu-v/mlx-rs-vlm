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
    generate_with_callback(model, input_ids, pixel_values, grid_thw, params, |_| true)
}

/// 生成トークンごとに `on_token` を呼ぶ。`false` を返すと早期打ち切り。
/// EOS トークンも callback に渡り、戻り値にも含まれる。
pub fn generate_with_callback(
    model: &mut Qwen35VLM,
    input_ids: &Array,
    pixel_values: Option<&Array>,
    grid_thw: Option<&[THW]>,
    params: &GenerateParams,
    mut on_token: impl FnMut(i32) -> bool,
) -> Result<Vec<i32>, Exception> {
    let mut cache = model.make_cache();
    let mut generated_tokens = Vec::new();

    let logits = model.prepare(input_ids, pixel_values, grid_thw, &mut cache)?;

    let mut next_token = sample_token(&logits, params.temperature)?;
    generated_tokens.push(next_token);

    if !on_token(next_token) || params.eos_token_ids.contains(&next_token) {
        return Ok(generated_tokens);
    }

    for _ in 1..params.max_tokens {
        let token_array = Array::from_slice(&[next_token], &[1, 1]);
        let logits = model.forward_text(&token_array, &mut cache)?;
        next_token = sample_token(&logits, params.temperature)?;
        generated_tokens.push(next_token);

        if !on_token(next_token) || params.eos_token_ids.contains(&next_token) {
            break;
        }
    }

    Ok(generated_tokens)
}

fn sample_token(logits: &Array, temperature: f32) -> Result<i32, Exception> {
    let seq_len = logits.shape()[1];
    let last_logits = logits.index((.., seq_len - 1, ..));

    // 負値・ゼロ近傍に加え NaN も greedy に落とす (NaN はどの比較でも false になるため
    // この形でないと sampling 経路に流れて logits が全部 NaN になる)
    if !(temperature >= 1e-6) {
        let token = argmax_axis(&last_logits, -1, None)?;
        token.eval()?;
        let token: i32 = token.item();
        Ok(token)
    } else {
        // categorical は unnormalized logits を期待するため softmax は不要。
        // key は渡さず global stream を使う (呼び出し側の random::seed() が効き、
        // 呼び出しごとに状態が進んでステップ間で乱数が相関しない)。
        let scaled = last_logits.multiply(Array::from(1.0 / temperature))?;
        let token = mlx_rs::random::categorical(&scaled, None, None, None)?;
        token.eval()?;
        let token: i32 = token.item();
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_token_picks_dominant_logit_and_is_seed_reproducible() {
        // 圧倒的に大きい logit は argmax / categorical のどちらでも選ばれる
        let logits = Array::from_slice(&[0.0f32, 0.0, 100.0, 0.0], &[1, 1, 4]);
        assert_eq!(sample_token(&logits, 0.0).unwrap(), 2);
        assert_eq!(sample_token(&logits, 1.0).unwrap(), 2);

        // NaN 温度は sampling に流れず greedy に落ちる
        assert_eq!(sample_token(&logits, f32::NAN).unwrap(), 2);
        assert_eq!(sample_token(&logits, -1.0).unwrap(), 2);

        // 平坦な logits でも同一 seed なら同一系列になる (global stream を使用)
        let flat = Array::from_slice(&[0.0f32; 16], &[1, 1, 16]);
        mlx_rs::random::seed(42).unwrap();
        let first: Vec<i32> = (0..4).map(|_| sample_token(&flat, 1.0).unwrap()).collect();
        mlx_rs::random::seed(42).unwrap();
        let second: Vec<i32> = (0..4).map(|_| sample_token(&flat, 1.0).unwrap()).collect();
        assert_eq!(first, second);
    }
}
