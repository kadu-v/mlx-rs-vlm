use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{Conv1d, Conv1dBuilder};
use mlx_rs::ops::indexing::{Ellipsis, IndexOp, NewAxis};

use crate::cache::MambaCache;
use crate::config::TextConfiguration;
use crate::weights::QuantizationInfo;

use super::linear_like::LinearLike;

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct RmsNormGated {
    #[param]
    weight: Param<Array>,
    eps: f32,
}

impl RmsNormGated {
    pub fn new(dimensions: i32, eps: f32) -> Result<Self, Exception> {
        let weight = Array::ones::<f32>(&[dimensions])?;
        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }

    pub fn forward(&mut self, x: &Array, gate: &Array) -> Result<Array, Exception> {
        let normed = mlx_rs::fast::rms_norm(x, self.weight.as_ref(), self.eps)?;
        let gate_act = mlx_rs::nn::silu(gate)?;
        normed.multiply(gate_act)
    }
}

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct GatedDeltaNet {
    num_v_heads: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    conv_dim: i32,
    conv_kernel_size: i32,

    #[param]
    in_proj_qkv: LinearLike,
    #[param]
    in_proj_z: LinearLike,
    #[param]
    in_proj_b: LinearLike,
    #[param]
    in_proj_a: LinearLike,
    #[param]
    conv1d: Conv1d,
    #[param]
    dt_bias: Param<Array>,
    #[param]
    a_log: Param<Array>,
    #[param]
    norm: RmsNormGated,
    #[param]
    out_proj: LinearLike,
}

impl GatedDeltaNet {
    pub fn new(
        config: &TextConfiguration,
        prefix: &str,
        quant_info: Option<&QuantizationInfo>,
    ) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;

        if num_v_heads % num_k_heads != 0 {
            return Err(Exception::custom(format!(
                "num_v_heads ({}) must be divisible by num_k_heads ({})",
                num_v_heads, num_k_heads
            )));
        }

        let in_proj_qkv = LinearLike::new(hidden_size, conv_dim, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.in_proj_qkv"))))?;
        let in_proj_z = LinearLike::new(hidden_size, value_dim, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.in_proj_z"))))?;
        let in_proj_b = LinearLike::new(hidden_size, num_v_heads, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.in_proj_b"))))?;
        let in_proj_a = LinearLike::new(hidden_size, num_v_heads, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.in_proj_a"))))?;

        let conv1d = Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
            .groups(conv_dim)
            .bias(false)
            .padding(0)
            .build()?;

        let dt_bias = Array::ones::<f32>(&[num_v_heads])?;
        let a_log = mlx_rs::random::uniform::<_, f32>(0.0, 16.0, &[num_v_heads], None)?.log()?;

        let norm = RmsNormGated::new(head_v_dim, config.rms_norm_eps)?;
        let out_proj = LinearLike::new(value_dim, hidden_size, false, quant_info.and_then(|qi| qi.for_path(&format!("{prefix}.out_proj"))))?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            conv_dim,
            conv_kernel_size,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            dt_bias: Param::new(dt_bias),
            a_log: Param::new(a_log),
            norm,
            out_proj,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut MambaCache,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let s = x.dim(1);

        let mut qkv = self.in_proj_qkv.forward(x)?;
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;
        let beta = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;

        let conv_state = match cache.conv_state() {
            Some(state) => state.clone(),
            None => {
                Array::zeros::<f32>(&[b, self.conv_kernel_size - 1, self.conv_dim])?
                    .as_dtype(x.dtype())?
            }
        };

        if let Some(m) = mask {
            let m_expanded = m.index((Ellipsis, NewAxis));
            qkv = mlx_rs::ops::r#where(&m_expanded, &qkv, &Array::from(0.0f32))?;
        }

        let conv_input = mlx_rs::ops::concatenate_axis(&[&conv_state, &qkv], 1)?;

        let new_conv_len = conv_input.dim(1);
        let keep_start = new_conv_len - (self.conv_kernel_size - 1);
        let new_conv_state = conv_input.index((.., keep_start.., ..));
        cache.set_conv_state(new_conv_state);

        let conv_out = mlx_rs::nn::silu(self.conv1d.forward(&conv_input)?)?;

        let split_indices = [self.key_dim, 2 * self.key_dim];
        let conv_split = conv_out.split_axis(&split_indices, Some(-1))?;
        let q = conv_split[0].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let k = conv_split[1].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let v = conv_split[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;

        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q_normed =
            rms_norm_no_weight(&q, 1e-6)?.multiply(Array::from(inv_scale * inv_scale))?;
        let k_normed = rms_norm_no_weight(&k, 1e-6)?.multiply(Array::from(inv_scale))?;

        let ssm_state = cache.ssm_state().cloned();
        let (mut out, new_ssm_state) = gated_delta_update(
            &q_normed,
            &k_normed,
            &v,
            &a,
            &beta,
            self.a_log.as_ref(),
            self.dt_bias.as_ref(),
            ssm_state.as_ref(),
            mask,
        )?;

        cache.set_ssm_state(new_ssm_state);
        cache.advance(s as usize);

        out = self.norm.forward(&out, &z)?;

        let out = out.reshape(&[b, s, -1])?;
        self.out_proj.forward(&out)
    }
}

fn rms_norm_no_weight(x: &Array, eps: f32) -> Result<Array, Exception> {
    let x_sq = x.square()?;
    let mean_sq = x_sq.mean_axes(&[-1], true)?;
    let rms = mean_sq.add(Array::from(eps))?.rsqrt()?;
    x.multiply(rms)
}

pub fn gated_delta_update(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let beta = mlx_rs::ops::sigmoid(b)?.as_dtype(mlx_rs::Dtype::Float32)?;

    let a_float = a.as_dtype(mlx_rs::Dtype::Float32)?;
    let a_log_float = a_log.as_dtype(mlx_rs::Dtype::Float32)?;
    let dt_bias_float = dt_bias.as_dtype(mlx_rs::Dtype::Float32)?;
    let sp = mlx_rs::nn::softplus(a_float.add(&dt_bias_float)?)?;
    let g = a_log_float.exp()?.multiply(sp)?.negative()?.exp()?;

    let b_dim = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);

    let mut state = match state {
        Some(s) => s.as_dtype(mlx_rs::Dtype::Float32)?,
        None => Array::zeros::<f32>(&[b_dim, hv, dv, dk])?,
    };

    let repeat_factor = hv / hk;
    let (q, k) = if repeat_factor > 1 {
        let q_expanded = q.expand_dims(3)?;
        let k_expanded = k.expand_dims(3)?;
        let q_rep = mlx_rs::ops::broadcast_to(&q_expanded, &[b_dim, t, hk, repeat_factor, dk])?;
        let k_rep = mlx_rs::ops::broadcast_to(&k_expanded, &[b_dim, t, hk, repeat_factor, dk])?;
        let q_rep = q_rep.reshape(&[b_dim, t, hv, dk])?;
        let k_rep = k_rep.reshape(&[b_dim, t, hv, dk])?;
        (q_rep, k_rep)
    } else {
        (q.clone(), k.clone())
    };

    let mut ys = Vec::with_capacity(t as usize);

    for t_idx in 0..t {
        let old_state = state.clone();

        let q_t = q.index((.., t_idx, .., ..));
        let k_t = k.index((.., t_idx, .., ..));
        let v_t = v.index((.., t_idx, .., ..));
        let g_t = g.index((.., t_idx, ..));
        let beta_t = beta.index((.., t_idx, ..));

        let g_t_expanded = g_t.expand_dims(-1)?.expand_dims(-1)?;
        state = state.multiply(&g_t_expanded)?;

        let k_t_expanded = k_t.expand_dims(-2)?;
        let kv_mem = state.multiply(&k_t_expanded)?.sum_axis(-1, false)?;

        let beta_t_expanded = beta_t.expand_dims(-1)?;
        let delta = v_t
            .as_dtype(mlx_rs::Dtype::Float32)?
            .subtract(&kv_mem)?
            .multiply(&beta_t_expanded)?;

        let delta_expanded = delta.expand_dims(-1)?;
        state = state.add(k_t_expanded.multiply(&delta_expanded)?)?;

        let q_t_expanded = q_t.expand_dims(-2)?;
        let y_t = state.multiply(&q_t_expanded)?.sum_axis(-1, false)?;

        if let Some(m) = mask {
            let m_t = m.index((.., t_idx));
            let m_expanded = m_t.expand_dims(-1)?.expand_dims(-1)?.expand_dims(-1)?;
            state = mlx_rs::ops::r#where(&m_expanded, &state, &old_state)?;
        }

        ys.push(y_t.as_dtype(q.dtype())?);
    }

    let y_refs: Vec<&Array> = ys.iter().collect();
    let output = mlx_rs::ops::stack_axis(&y_refs, 1)?;

    Ok((output, state))
}
