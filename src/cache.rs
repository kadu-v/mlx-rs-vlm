use mlx_rs::Array;
use mlx_rs::error::Exception;

pub enum Cache {
    KV(KVCacheSimple),
    Mamba(MambaCache),
}

impl Cache {
    pub fn as_kv_mut(&mut self) -> Option<&mut KVCacheSimple> {
        match self {
            Cache::KV(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_mamba_mut(&mut self) -> Option<&mut MambaCache> {
        match self {
            Cache::Mamba(c) => Some(c),
            _ => None,
        }
    }
}

pub struct KVCacheSimple {
    keys: Option<Array>,
    values: Option<Array>,
    offset: usize,
}

impl KVCacheSimple {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array), Exception> {
        let seq_len = keys.shape()[2] as usize;

        let (new_keys, new_values) = match (&self.keys, &self.values) {
            (Some(prev_k), Some(prev_v)) => {
                let new_k = mlx_rs::ops::concatenate_axis(&[prev_k, keys], 2)?;
                let new_v = mlx_rs::ops::concatenate_axis(&[prev_v, values], 2)?;
                (new_k, new_v)
            }
            _ => (keys.clone(), values.clone()),
        };

        self.offset += seq_len;
        self.keys = Some(new_keys.clone());
        self.values = Some(new_values.clone());

        Ok((new_keys, new_values))
    }
}

pub struct MambaCache {
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    offset: usize,
}

impl MambaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn conv_state(&self) -> Option<&Array> {
        self.conv_state.as_ref()
    }

    pub fn ssm_state(&self) -> Option<&Array> {
        self.ssm_state.as_ref()
    }

    pub fn set_conv_state(&mut self, state: Array) {
        self.conv_state = Some(state);
    }

    pub fn set_ssm_state(&mut self, state: Array) {
        self.ssm_state = Some(state);
    }

    pub fn advance(&mut self, n: usize) {
        self.offset += n;
    }
}

pub fn create_causal_mask(n: i32, offset: i32) -> Result<Option<Array>, Exception> {
    if n <= 1 {
        return Ok(None);
    }

    let total = n + offset;
    let row_indices = mlx_rs::ops::expand_dims(
        &Array::arange::<_, i32>(None, n, None)?,
        1,
    )?;
    let col_indices = Array::arange::<_, i32>(None, total, None)?;

    let shifted = row_indices.add(&Array::from(offset))?;
    let valid = shifted.ge(&col_indices)?;

    let neg_inf = Array::from(f32::NEG_INFINITY);
    let zero = Array::from(0.0f32);
    let mask = mlx_rs::ops::r#where(&valid, &zero, &neg_inf)?;

    let mask = mask.reshape(&[1, 1, n, total])?;
    Ok(Some(mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_causal_mask() {
        assert!(create_causal_mask(1, 0).unwrap().is_none());

        let mask = create_causal_mask(3, 0).unwrap().unwrap();
        assert_eq!(mask.shape(), &[1, 1, 3, 3]);
        mask.eval().unwrap();
        let vals: Vec<f32> = mask.as_slice::<f32>().to_vec();
        // Lower-triangular: row i can see cols 0..=i
        assert!(vals[0].is_finite()); // [0,0] = 0.0
        assert!(vals[1].is_infinite()); // [0,1] = -inf
        assert!(vals[2].is_infinite()); // [0,2] = -inf
        assert!(vals[3].is_finite()); // [1,0] = 0.0
        assert!(vals[4].is_finite()); // [1,1] = 0.0
        assert!(vals[5].is_infinite()); // [1,2] = -inf

        let mask = create_causal_mask(2, 3).unwrap().unwrap();
        assert_eq!(mask.shape(), &[1, 1, 2, 5]);
        mask.eval().unwrap();
        let vals: Vec<f32> = mask.as_slice::<f32>().to_vec();
        // Row 0 (offset 3): can see cols 0..=3
        assert!(vals[0..=3].iter().all(|v| v.is_finite()));
        assert!(vals[4].is_infinite());
        // Row 1 (offset 4): can see cols 0..=4
        assert!(vals[5..=9].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KVCacheSimple::new();
        assert_eq!(cache.offset(), 0);

        let k1 = Array::zeros::<f32>(&[1, 2, 3, 64]).unwrap();
        let v1 = Array::zeros::<f32>(&[1, 2, 3, 64]).unwrap();
        let (keys, values) = cache.update(&k1, &v1).unwrap();
        assert_eq!(keys.shape(), &[1, 2, 3, 64]);
        assert_eq!(values.shape(), &[1, 2, 3, 64]);
        assert_eq!(cache.offset(), 3);

        let k2 = Array::zeros::<f32>(&[1, 2, 1, 64]).unwrap();
        let v2 = Array::zeros::<f32>(&[1, 2, 1, 64]).unwrap();
        let (keys, values) = cache.update(&k2, &v2).unwrap();
        assert_eq!(keys.shape(), &[1, 2, 4, 64]);
        assert_eq!(values.shape(), &[1, 2, 4, 64]);
        assert_eq!(cache.offset(), 4);
    }
}
