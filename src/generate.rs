use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};

use crate::cache::Cache;
use crate::config::{Qwen35Configuration, THW};
use crate::model::Qwen35VLM;

pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    /// nucleus sampling。1.0 で無効 (mlx-lm と同じ規約)。
    pub top_p: f32,
    /// 上位 k 語彙に制限。0 で無効。
    pub top_k: usize,
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
            top_p: 1.0,
            top_k: 0,
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
/// 正常終了・callback 中断・エラーのいずれでも、先行投入した処理の完了を待って返す。
pub fn generate_with_callback(
    model: &mut Qwen35VLM,
    input_ids: &Array,
    pixel_values: Option<&Array>,
    grid_thw: Option<&[THW]>,
    params: &GenerateParams,
    on_token: impl FnMut(i32) -> bool,
) -> Result<Vec<i32>, Exception> {
    let stream = mlx_rs::Stream::thread_local_or_default();
    // cache を同期完了まで保持し、callback 中断後の先行評価も drain する。
    let mut cache = model.make_cache();
    let result = mlx_rs::with_stream(&stream, || {
        generate_inner(
            model,
            input_ids,
            pixel_values,
            grid_thw,
            params,
            &mut cache,
            on_token,
        )
    });
    let synchronized = stream.synchronize();
    // 生成自体が失敗した場合は、そのエラーを優先する。
    result.and_then(|tokens| synchronized.map(|()| tokens))
}

fn generate_inner(
    model: &mut Qwen35VLM,
    input_ids: &Array,
    pixel_values: Option<&Array>,
    grid_thw: Option<&[THW]>,
    params: &GenerateParams,
    cache: &mut [Cache],
    mut on_token: impl FnMut(i32) -> bool,
) -> Result<Vec<i32>, Exception> {
    let mut generated_tokens = Vec::new();

    let logits = model.prepare(input_ids, pixel_values, grid_thw, cache)?;
    let mut y = sample_token_array(&logits, params)?;
    mlx_rs::transforms::async_eval([&y])?;

    for n in 0..params.max_tokens {
        // 現トークンを item() で取り出す (= GPU 同期) 前に次ステップを投入しておき、
        // 取り出し・EOS 判定・callback の裏で GPU に次の forward を進めさせる
        // (mlx-lm の generate_step と同じ 1 ステップ先行パイプライン)。
        // EOS や callback 打ち切りで先行分 1 ステップが無駄になるのは速度との交換。
        let next_y = if n + 1 < params.max_tokens {
            let logits = model.forward_text(&y, cache)?;
            let next = sample_token_array(&logits, params)?;
            mlx_rs::transforms::async_eval([&next])?;
            Some(next)
        } else {
            None
        };

        y.eval()?;
        let token: i32 = y.item();
        generated_tokens.push(token);

        if !on_token(token) || params.eos_token_ids.contains(&token) {
            break;
        }
        match next_y {
            Some(next) => y = next,
            None => break,
        }
    }

    Ok(generated_tokens)
}

/// logits の最終位置から次トークンをサンプルし、[1, 1] の int32 Array として
/// **eval せずに** 返す。呼び出し側はこれをそのまま `forward_text` に渡すか、
/// `async_eval` で先行評価してから `item()` で取り出す。
fn sample_token_array(logits: &Array, params: &GenerateParams) -> Result<Array, Exception> {
    let seq_len = logits.shape()[1];
    let last_logits = logits.index((.., seq_len - 1, ..));

    // 負値・ゼロ近傍に加え NaN も greedy に落とす (NaN はどの比較でも false になるため
    // この形でないと sampling 経路に流れて logits が全部 NaN になる)
    let token = if !(params.temperature >= 1e-6) {
        argmax_axis(&last_logits, -1, None)?
    } else {
        // categorical は unnormalized logits を期待するため softmax は不要。
        // key は渡さず global stream を使う (呼び出し側の random::seed() が効き、
        // 呼び出しごとに状態が進んでステップ間で乱数が相関しない)。
        let mut scaled = last_logits.multiply(Array::from(1.0 / params.temperature))?;

        let vocab_size = scaled.shape()[1] as usize;
        if params.top_k > 0 && params.top_k < vocab_size {
            scaled = mask_below_top_k(&scaled, params.top_k)?;
        }

        if params.top_p < 1.0 {
            sample_top_p(&scaled, params.top_p)?
        } else {
            mlx_rs::random::categorical(&scaled, None, None, None)?
        }
    };
    // argmax / categorical は uint32 の [1] を返すので forward_text の入力形に揃える
    token.reshape(&[1, 1])?.as_type::<i32>()
}

#[cfg(test)]
fn sample_token(logits: &Array, params: &GenerateParams) -> Result<i32, Exception> {
    let token = sample_token_array(logits, params)?;
    token.eval()?;
    Ok(token.item())
}

/// 降順 k 番目の logit をしきい値として、それ未満を -inf にマスクする。
/// しきい値と同値のタイが k 個を超えて残ることは許容する (mlx-lm と同様の近似)。
fn mask_below_top_k(scaled: &Array, top_k: usize) -> Result<Array, Exception> {
    let neg = scaled.negative()?;
    let partitioned = mlx_rs::ops::partition_axis(&neg, top_k as i32 - 1, -1)?;
    let kth_value = partitioned
        .index((.., top_k as i32 - 1))
        .negative()?;
    let below = scaled.lt(&kth_value)?;
    mlx_rs::ops::which(&below, Array::from(f32::NEG_INFINITY), scaled)
}

/// nucleus (top-p) sampling。昇順ソート空間で累積確率が (1 - top_p) を超える
/// 上位確率側だけを残して categorical し、添字を語彙 ID に引き直す (mlx-lm と同形)。
fn sample_top_p(scaled: &Array, top_p: f32) -> Result<Array, Exception> {
    let sorted_idx = mlx_rs::ops::argsort_axis(scaled, -1)?;
    let sorted_logits = mlx_rs::ops::indexing::take_along_axis(scaled, &sorted_idx, -1)?;
    let probs = mlx_rs::ops::softmax_axis(&sorted_logits, -1, None)?;
    let cumulative = mlx_rs::ops::cumsum(&probs, -1, None, None)?;
    let keep = cumulative.gt(Array::from(1.0 - top_p))?;
    let masked = mlx_rs::ops::which(&keep, &sorted_logits, Array::from(f32::NEG_INFINITY))?;
    let sorted_token = mlx_rs::random::categorical(&masked, None, None, None)?;
    // categorical は末尾軸を潰した [1] を返すので、take_along_axis 用に [1, 1] へ戻す
    let sorted_token = sorted_token.reshape(&[1, 1])?;
    let token = mlx_rs::ops::indexing::take_along_axis(&sorted_idx, &sorted_token, -1)?;
    token.reshape(&[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> Qwen35VLM {
        let config = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_5_vl",
            "text_config": {
                "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 16, "intermediate_size": 64, "vocab_size": 16,
                "full_attention_interval": 1, "tie_word_embeddings": true,
                "rope_parameters": { "mrope_section": [1, 1, 0] }
            },
            "vision_config": {
                "depth": 1, "hidden_size": 32, "num_heads": 2,
                "intermediate_size": 64, "out_hidden_size": 32
            }
        }))
        .unwrap();
        Qwen35VLM::new(&config, None).unwrap()
    }

    #[test]
    fn generation_can_run_again_after_callback_cancellation_eos_and_error() {
        let stream = mlx_rs::Stream::gpu();
        mlx_rs::with_stream(&stream, || {
            let mut model = tiny_model();
            let input = Array::from_slice(&[1_i32, 2], &[1, 2]);
            let mut params = GenerateParams {
                max_tokens: 4,
                eos_token_ids: vec![],
                ..Default::default()
            };
            let mut callbacks = 0;
            let cancelled = generate_with_callback(&mut model, &input, None, None, &params, |_| {
                callbacks += 1;
                false
            })
            .unwrap();
            assert_eq!(callbacks, 1);
            assert_eq!(cancelled.len(), 1);

            params.eos_token_ids = (0..16).collect();
            assert_eq!(
                generate(&mut model, &input, None, None, &params)
                    .unwrap()
                    .len(),
                1
            );

            // Missing grid metadata fails during prepare; the same model/stream
            // must remain usable for the next request.
            let pixels = Array::from(0.0_f32);
            assert!(generate(&mut model, &input, Some(&pixels), None, &params).is_err());
            params.eos_token_ids.clear();
            assert_eq!(
                generate(&mut model, &input, None, None, &params)
                    .unwrap()
                    .len(),
                4
            );
        });
    }

    fn params(temperature: f32) -> GenerateParams {
        GenerateParams {
            temperature,
            ..Default::default()
        }
    }

    #[test]
    fn sample_token_picks_dominant_logit_and_is_seed_reproducible() {
        // 圧倒的に大きい logit は argmax / categorical のどちらでも選ばれる
        let logits = Array::from_slice(&[0.0f32, 0.0, 100.0, 0.0], &[1, 1, 4]);
        assert_eq!(sample_token(&logits, &params(0.0)).unwrap(), 2);
        assert_eq!(sample_token(&logits, &params(1.0)).unwrap(), 2);

        // NaN 温度は sampling に流れず greedy に落ちる
        assert_eq!(sample_token(&logits, &params(f32::NAN)).unwrap(), 2);
        assert_eq!(sample_token(&logits, &params(-1.0)).unwrap(), 2);

        // 平坦な logits でも同一 seed なら同一系列になる (global stream を使用)
        let flat = Array::from_slice(&[0.0f32; 16], &[1, 1, 16]);
        mlx_rs::random::seed(42).unwrap();
        let first: Vec<i32> =
            (0..4).map(|_| sample_token(&flat, &params(1.0)).unwrap()).collect();
        mlx_rs::random::seed(42).unwrap();
        let second: Vec<i32> =
            (0..4).map(|_| sample_token(&flat, &params(1.0)).unwrap()).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn top_k_one_equals_argmax_at_any_temperature() {
        let logits = Array::from_slice(&[1.0f32, 4.0, 2.0, 3.0], &[1, 1, 4]);
        let p = GenerateParams {
            temperature: 5.0,
            top_k: 1,
            ..Default::default()
        };
        for _ in 0..8 {
            assert_eq!(sample_token(&logits, &p).unwrap(), 1);
        }
    }

    #[test]
    fn small_top_p_keeps_only_the_head_of_the_distribution() {
        // softmax([1,2,3,4]) ≈ [0.03, 0.09, 0.23, 0.64] → top_p=0.5 の nucleus は {3} のみ
        let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let p = GenerateParams {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        for _ in 0..8 {
            assert_eq!(sample_token(&logits, &p).unwrap(), 3);
        }
    }

    #[test]
    fn top_k_masks_the_tail() {
        // top_k=2 で残るのは {1, 3} だけ。何回引いてもそれ以外は出ない
        let logits = Array::from_slice(&[1.0f32, 4.0, 2.0, 3.0], &[1, 1, 4]);
        let p = GenerateParams {
            temperature: 2.0,
            top_k: 2,
            ..Default::default()
        };
        for _ in 0..16 {
            let token = sample_token(&logits, &p).unwrap();
            assert!(token == 1 || token == 3, "unexpected token {token}");
        }
    }
}
