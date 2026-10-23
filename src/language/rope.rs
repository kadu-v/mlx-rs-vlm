use mlx::Array;
use mlx::error::Exception;
use mlx::ops::indexing::{Ellipsis, IndexOp};
use mlx_rs as mlx;

#[derive(Debug)]
pub struct MRoPE {
    inv_freq: Array,
    mrope_section: [i32; 3],
}

impl MRoPE {
    pub fn new(dim: i32, base: f32, mrope_section: [i32; 3]) -> Result<Self, Exception> {
        let half_dim = dim.max(1) / 2;
        let indices = Array::arange::<_, f32>(0, half_dim, None)?;
        let exponents = indices
            .multiply(Array::from(2.0f32))?
            .divide(Array::from(dim.max(1) as f32))?;
        let inv_freq = Array::from(1.0f32).divide(Array::from(base).power(&exponents)?)?;
        // This constant is outside ModuleParameters, so model.eval() does not
        // evaluate it. Materialize it before inference can run on another thread.
        inv_freq.eval()?;

        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }

    pub fn forward(
        &self,
        position_ids: &Array,
        target_dtype: mlx::Dtype,
    ) -> Result<(Array, Array), Exception> {
        let pos = position_ids.as_dtype(mlx::Dtype::Float32)?;
        let merged = self.interleave_axes(&pos)?;
        let emb = mlx::ops::concatenate(&[&merged, &merged], -1)?;
        let cos = emb.cos()?.as_dtype(target_dtype)?;
        let sin = emb.sin()?.as_dtype(target_dtype)?;
        Ok((cos, sin))
    }

    fn interleave_axes(&self, pos: &Array) -> Result<Array, Exception> {
        let half_dim = self.inv_freq.dim(0) as usize;
        let sections = self.mrope_section;

        let pos_t = pos.index(0);
        let pos_h = pos.index(1);
        let pos_w = pos.index(2);

        let mut t_mask_v = vec![0.0f32; half_dim];
        let mut h_mask_v = vec![0.0f32; half_dim];
        let mut w_mask_v = vec![0.0f32; half_dim];

        for idx in 0..half_dim {
            let mut axis = 0usize;
            for &(dim, offset) in &[(1usize, 1usize), (2usize, 2usize)] {
                let end = (sections[dim] * 3).min(half_dim as i32) as usize;
                if idx >= offset && idx < end && (idx - offset) % 3 == 0 {
                    axis = dim;
                    break;
                }
            }
            match axis {
                1 => h_mask_v[idx] = 1.0,
                2 => w_mask_v[idx] = 1.0,
                _ => t_mask_v[idx] = 1.0,
            }
        }

        let hd = half_dim as i32;
        let t_mask = Array::from_slice(&t_mask_v, &[1, 1, hd]);
        let h_mask = Array::from_slice(&h_mask_v, &[1, 1, hd]);
        let w_mask = Array::from_slice(&w_mask_v, &[1, 1, hd]);

        let pos_selected = pos_t
            .expand_dims(-1)?
            .multiply(&t_mask)?
            .add(pos_h.expand_dims(-1)?.multiply(&h_mask)?)?
            .add(pos_w.expand_dims(-1)?.multiply(&w_mask)?)?;

        let inv = self.inv_freq.reshape(&[1, 1, hd])?;
        pos_selected.multiply(&inv)
    }
}

fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let d = x.dim(-1);
    let half = d / 2;
    let x1 = x.index((Ellipsis, ..half));
    let x2 = x.index((Ellipsis, half..));
    let neg_x2 = x2.negative()?;
    mlx::ops::concatenate(&[&neg_x2, &x1], -1)
}

pub fn apply_multimodal_rotary_pos_emb(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, Array), Exception> {
    let rotary_dim = cos.dim(-1);

    let cos = cos.expand_dims(1)?;
    let sin = sin.expand_dims(1)?;

    let q_dim = q.dim(-1);
    let k_dim = k.dim(-1);

    let q_rot = q.index((Ellipsis, ..rotary_dim));
    let k_rot = k.index((Ellipsis, ..rotary_dim));

    let q_rotated = q_rot
        .multiply(&cos)?
        .add(rotate_half(&q_rot)?.multiply(&sin)?)?;
    let k_rotated = k_rot
        .multiply(&cos)?
        .add(rotate_half(&k_rot)?.multiply(&sin)?)?;

    let q_out = if rotary_dim < q_dim {
        let q_pass = q.index((Ellipsis, rotary_dim..));
        mlx::ops::concatenate(&[&q_rotated, &q_pass], -1)?
    } else {
        q_rotated
    };

    let k_out = if rotary_dim < k_dim {
        let k_pass = k.index((Ellipsis, rotary_dim..));
        mlx::ops::concatenate(&[&k_rotated, &k_pass], -1)?
    } else {
        k_rotated
    };

    Ok((q_out, k_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrope_forward_shape() {
        let mrope = MRoPE::new(64, 10000.0, [11, 11, 10]).unwrap();
        let pos_ids = Array::zeros::<i32>(&[3, 1, 8]).unwrap();
        let (cos, sin) = mrope.forward(&pos_ids, mlx::Dtype::Float32).unwrap();
        assert_eq!(cos.shape(), &[1, 8, 64]);
        assert_eq!(sin.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_mrope_interleave_values() {
        let mrope = MRoPE::new(64, 10000.0, [11, 11, 10]).unwrap();
        // All positions zero → all freqs = cos(0)=1
        let pos_ids = Array::zeros::<i32>(&[3, 1, 4]).unwrap();
        let (cos, _sin) = mrope.forward(&pos_ids, mlx::Dtype::Float32).unwrap();
        cos.eval().unwrap();
        let vals: Vec<f32> = cos.reshape(&[-1]).unwrap().as_slice::<f32>().to_vec();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "cos(0) should be 1.0 at index {}, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_mrope_axis_separation() {
        let mrope = MRoPE::new(64, 10000.0, [11, 11, 10]).unwrap();
        // T=1, H=0, W=0 — only T-axis positions contribute
        let pos_ids = Array::from_slice(&[1i32, 0, 0, 0, 0, 0], &[3, 1, 2]);
        let (cos_a, _) = mrope.forward(&pos_ids, mlx::Dtype::Float32).unwrap();
        // T=0, H=1, W=0 — only H-axis positions contribute
        let pos_ids2 = Array::from_slice(&[0i32, 0, 1, 0, 0, 0], &[3, 1, 2]);
        let (cos_b, _) = mrope.forward(&pos_ids2, mlx::Dtype::Float32).unwrap();
        cos_a.eval().unwrap();
        cos_b.eval().unwrap();
        // They should differ because T and H affect different output indices
        let diff = cos_a.subtract(&cos_b).unwrap();
        diff.eval().unwrap();
        let abs_diff = diff.abs().unwrap();
        let max_diff: f32 = abs_diff.max(None).unwrap().item_cast();
        assert!(
            max_diff > 1e-3,
            "T-only and H-only should produce different cos"
        );
    }
}
