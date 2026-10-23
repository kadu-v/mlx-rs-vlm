use std::collections::HashMap;
use std::rc::Rc;

use mlx::Array;
use mlx::error::Exception;
use mlx::macros::ModuleParameters;
use mlx::module::{Module, ModuleParameters, ModuleParametersExt};
use mlx::ops::indexing::IndexOp;
use mlx_rs as mlx;

use crate::cache::Cache;
use crate::config::{Qwen35Configuration, THW};
use crate::language::LanguageModel;
use crate::vision::VisionModel;
use crate::weights::QuantizationInfo;

#[derive(Debug, ModuleParameters)]
#[module(root = mlx)]
pub struct Qwen35VLM {
    pub config: Qwen35Configuration,
    #[param]
    pub language_model: LanguageModel,
    #[param]
    pub vision_tower: VisionModel,
    rope_deltas: Vec<i32>,
}

impl Qwen35VLM {
    pub fn new(
        config: &Qwen35Configuration,
        quant_info: Option<&QuantizationInfo>,
    ) -> Result<Self, Exception> {
        let mut text_config = config.text_config.clone();
        text_config.vocab_size = config.vocab_size();
        let language_model = LanguageModel::new(&text_config, "language_model", quant_info)?;
        let vision_tower = VisionModel::new(&config.vision_config)?;

        Ok(Self {
            config: config.clone(),
            language_model,
            vision_tower,
            rope_deltas: Vec::new(),
        })
    }

    pub fn load_weights(&mut self, weights: HashMap<String, Array>) -> Result<(), Exception> {
        let model_params = self.parameters().flatten();
        let model_keys: std::collections::HashSet<&str> =
            model_params.keys().map(|k| k.as_ref()).collect();
        let weight_keys: std::collections::HashSet<&str> =
            weights.keys().map(|k| k.as_str()).collect();

        let mut missing: Vec<_> = model_keys.difference(&weight_keys).copied().collect();
        let mut extra: Vec<_> = weight_keys.difference(&model_keys).copied().collect();
        missing.sort();
        extra.sort();

        if !missing.is_empty() || !extra.is_empty() {
            let mut msg = format!(
                "Weight verification failed: {} missing, {} extra keys",
                missing.len(),
                extra.len()
            );
            if !missing.is_empty() {
                msg.push_str(&format!(
                    "\nMissing: {:?}",
                    &missing[..missing.len().min(10)]
                ));
            }
            if !extra.is_empty() {
                msg.push_str(&format!("\nExtra: {:?}", &extra[..extra.len().min(10)]));
            }
            return Err(Exception::custom(msg));
        }

        let remapped: HashMap<Rc<str>, Array> = weights
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.update_flattened(remapped);
        self.eval()?;
        Ok(())
    }

    pub fn make_cache(&self) -> Vec<Cache> {
        self.language_model.make_cache()
    }

    /// Run a forward pass: embed inputs, apply vision encoder if multimodal, and produce logits.
    ///
    /// For multimodal input (pixel_values + grid_thw), only batch_size=1 is supported.
    pub fn prepare(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        grid_thw: Option<&[THW]>,
        cache: &mut [Cache],
    ) -> Result<Array, Exception> {
        let input_ids_2d = if input_ids.ndim() == 1 {
            input_ids.expand_dims(0)?
        } else {
            input_ids.clone()
        };

        match (pixel_values.is_some(), grid_thw.is_some()) {
            (true, false) | (false, true) => {
                return Err(Exception::custom(
                    "pixel_values and grid_thw must both be provided or both be None",
                ));
            }
            _ => {}
        }

        if pixel_values.is_some() && input_ids_2d.dim(0) != 1 {
            return Err(Exception::custom(
                "batch > 1 not supported for multimodal input",
            ));
        }

        let (position_ids, rope_delta) = get_rope_index(
            &input_ids_2d,
            grid_thw,
            None,
            self.config.vision_config.spatial_merge_size,
            self.config.image_token_id,
            self.config.video_token_id,
            self.config.vision_start_token_id,
            None,
        )?;
        self.rope_deltas = rope_delta;

        let inputs_embeds = match (pixel_values, grid_thw) {
            (Some(pixels), Some(grids)) => {
                let text_embeds = self.language_model.embed_tokens.forward(&input_ids_2d)?;
                let (vision_hidden, _) = self.vision_tower.forward_with_grid(pixels, grids)?;
                let vision_features = vision_hidden.as_dtype(text_embeds.dtype())?;
                let merged = merge_input_ids_with_image_features(
                    &vision_features,
                    &text_embeds,
                    &input_ids_2d,
                    self.config.image_token_index(),
                    self.config.video_token_index(),
                )?;
                Some(merged)
            }
            _ => None,
        };

        match &inputs_embeds {
            Some(embeds) => {
                self.language_model
                    .forward(None, Some(embeds), cache, Some(&position_ids))
            }
            None => {
                self.language_model
                    .forward(Some(&input_ids_2d), None, cache, Some(&position_ids))
            }
        }
    }

    pub fn forward_text(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
    ) -> Result<Array, Exception> {
        let input_ids_2d = if input_ids.ndim() == 1 {
            input_ids.expand_dims(0)?
        } else {
            input_ids.clone()
        };

        let batch = input_ids_2d.dim(0);
        let seq_len = input_ids_2d.dim(1);
        let cache_offset = self.get_cache_offset(cache) as i32;

        let deltas = if self.rope_deltas.is_empty() {
            Array::zeros::<i32>(&[batch])?
        } else {
            let d = Array::from_slice(&self.rope_deltas, &[self.rope_deltas.len() as i32]);
            if (self.rope_deltas.len() as i32) < batch {
                let repeats =
                    (batch + self.rope_deltas.len() as i32 - 1) / self.rope_deltas.len() as i32;
                mlx::ops::tile(&d, &[repeats])?
                    .index(..batch)
                    .as_dtype(mlx::Dtype::Int32)?
            } else if (self.rope_deltas.len() as i32) > batch {
                d.index(..batch).as_dtype(mlx::Dtype::Int32)?
            } else {
                d
            }
        };

        let base = Array::arange::<_, i32>(None, seq_len, None)?;
        let base = mlx::ops::broadcast_to(&base.reshape(&[1, seq_len])?, &[batch, seq_len])?;
        let offset = deltas
            .add(Array::from(cache_offset))?
            .reshape(&[batch, 1])?;
        let positions = base.add(&offset)?;
        let position_ids =
            mlx::ops::broadcast_to(&positions.expand_dims(0)?, &[3, batch, seq_len])?;

        self.language_model
            .forward(Some(&input_ids_2d), None, cache, Some(&position_ids))
    }

    fn get_cache_offset(&self, cache: &[Cache]) -> usize {
        for c in cache {
            if let Cache::KV(kv) = c {
                return kv.offset();
            }
        }
        0
    }
}

fn merge_input_ids_with_image_features(
    image_features: &Array,
    input_embeds: &Array,
    input_ids: &Array,
    image_token_id: i32,
    video_token_id: i32,
) -> Result<Array, Exception> {
    let image_mask = input_ids.eq(&Array::from(image_token_id))?;
    let video_mask = input_ids.eq(&Array::from(video_token_id))?;
    let special_mask = image_mask.logical_or(&video_mask)?;

    let mask_int = special_mask.as_dtype(mlx::Dtype::Int32)?;

    let n_special: i32 = mask_int.sum(None)?.item_cast();
    let n_features = image_features.dim(0);
    if n_special != n_features {
        return Err(Exception::custom(format!(
            "Image token count ({}) != vision feature count ({})",
            n_special, n_features
        )));
    }
    let cum_mask = mlx::ops::cumsum(&mask_int, -1, None, None)?;
    let indices = mlx::ops::maximum(&(cum_mask - Array::from(1i32)), &Array::from(0i32))?;
    let indices_flat = indices.reshape(&[-1])?;

    let gathered = mlx::ops::indexing::take_axis(image_features, &indices_flat, 0)?;
    let gathered = gathered.reshape(input_embeds.shape())?;

    let special_expanded = special_mask.expand_dims(-1)?;
    mlx::ops::select(&special_expanded, &gathered, input_embeds)
}

/// Compute 3D M-RoPE position IDs for text and vision tokens.
///
/// Returns `(position_ids [3, batch, seq], mrope_position_deltas)`.
/// For multimodal input, only batch_size=1 is supported.
pub fn get_rope_index(
    input_ids: &Array,
    image_grid_thw: Option<&[THW]>,
    video_grid_thw: Option<&[THW]>,
    spatial_merge_size: i32,
    image_token_id: i32,
    video_token_id: i32,
    _vision_start_token_id: i32,
    attention_mask: Option<&Array>,
) -> Result<(Array, Vec<i32>), Exception> {
    let seq_length = input_ids.dim(1);
    let batch_size = input_ids.dim(0);
    let bs = batch_size as usize;

    if image_grid_thw.is_none() && video_grid_thw.is_none() {
        let pos = Array::arange::<_, i32>(None, seq_length, None)?;
        let pos = pos.reshape(&[1, seq_length])?;
        let pos = mlx::ops::broadcast_to(&pos, &[batch_size, seq_length])?;
        let pos_3d = pos.expand_dims(0)?;
        let pos_3d = mlx::ops::broadcast_to(&pos_3d, &[3, batch_size, seq_length])?;
        return Ok((pos_3d, vec![0i32; bs]));
    }

    if batch_size != 1 {
        return Err(Exception::custom(
            "get_rope_index: batch > 1 not supported for multimodal input",
        ));
    }

    let input_tokens: Vec<i32> = {
        input_ids.eval()?;
        let flat = input_ids.reshape(&[-1])?;
        flat.eval()?;
        flat.as_slice::<i32>().to_vec()
    };

    let mask_vals: Option<Vec<i32>> = if let Some(mask) = attention_mask {
        let flat = mask.reshape(&[-1])?.as_dtype(mlx::Dtype::Int32)?;
        flat.eval()?;
        Some(flat.as_slice::<i32>().to_vec())
    } else {
        None
    };

    let sl = seq_length as usize;
    let total = bs * sl;
    // Default position_ids = 1 for multimodal (matches Swift's ones(like: inputIds))
    let mut pos_ids = vec![1i32; 3 * total];
    let mut mrope_position_deltas: Vec<i32> = Vec::with_capacity(bs);

    let mut img_grid_idx = 0usize;
    let mut vid_grid_idx = 0usize;

    for b in 0..bs {
        let off = b * sl;
        let raw_tokens = &input_tokens[off..off + sl];

        // Apply mask: where(mask==1, tokens, 0) — matches Swift behavior
        let tokens: Vec<i32> = if let Some(ref mv) = mask_vals {
            let bm = &mv[off..off + sl];
            raw_tokens
                .iter()
                .zip(bm.iter())
                .map(|(&t, &m)| if m != 0 { t } else { 0 })
                .collect()
        } else {
            raw_tokens.to_vec()
        };

        let remain_images = tokens.iter().filter(|&&t| t == image_token_id).count();
        let remain_videos = tokens.iter().filter(|&&t| t == video_token_id).count();
        let num_visions = remain_images + remain_videos;

        let mut llm_pos_segments: Vec<Vec<[i32; 3]>> = Vec::new();
        let mut st = 0usize;
        let mut remain_img = remain_images;
        let mut remain_vid = remain_videos;

        for _ in 0..num_visions {
            let ed_image = if remain_img > 0 {
                tokens[st..]
                    .iter()
                    .position(|&t| t == image_token_id)
                    .map(|p| st + p)
                    .unwrap_or(sl + 1)
            } else {
                sl + 1
            };
            let ed_video = if remain_vid > 0 {
                tokens[st..]
                    .iter()
                    .position(|&t| t == video_token_id)
                    .map(|p| st + p)
                    .unwrap_or(sl + 1)
            } else {
                sl + 1
            };

            let (llm_t, llm_h, llm_w, ed);
            if ed_image < ed_video {
                let grid = image_grid_thw.and_then(|gs| gs.get(img_grid_idx));
                if grid.is_none() {
                    break;
                }
                let g = grid.unwrap();
                llm_t = g.t;
                llm_h = g.h / spatial_merge_size;
                llm_w = g.w / spatial_merge_size;
                img_grid_idx += 1;
                remain_img -= 1;
                ed = ed_image;
            } else {
                let grid = video_grid_thw.and_then(|gs| gs.get(vid_grid_idx));
                if grid.is_none() {
                    break;
                }
                let g = grid.unwrap();
                llm_t = g.t;
                llm_h = g.h / spatial_merge_size;
                llm_w = g.w / spatial_merge_size;
                vid_grid_idx += 1;
                remain_vid -= 1;
                ed = ed_video;
            }

            let st_idx = if let Some(last_seg) = llm_pos_segments.last() {
                let max_val = last_seg
                    .iter()
                    .flat_map(|p| p.iter())
                    .copied()
                    .max()
                    .unwrap_or(-1);
                max_val + 1
            } else {
                0
            };

            let text_len = (ed - st) as i32;
            if text_len > 0 {
                let mut text_seg = Vec::with_capacity(text_len as usize);
                for j in 0..text_len {
                    let p = st_idx + j;
                    text_seg.push([p, p, p]);
                }
                llm_pos_segments.push(text_seg);
            }

            let n_vis = (llm_t * llm_h * llm_w) as usize;
            let vis_offset = text_len + st_idx;
            let mut vis_seg = Vec::with_capacity(n_vis);
            for vi in 0..n_vis {
                let t_i = vi as i32 / (llm_h * llm_w);
                let rem = vi as i32 % (llm_h * llm_w);
                let h_i = rem / llm_w;
                let w_i = rem % llm_w;
                vis_seg.push([vis_offset + t_i, vis_offset + h_i, vis_offset + w_i]);
            }
            llm_pos_segments.push(vis_seg);

            st = ed + n_vis;
        }

        if st < sl {
            let st_idx = if let Some(last_seg) = llm_pos_segments.last() {
                let max_val = last_seg
                    .iter()
                    .flat_map(|p| p.iter())
                    .copied()
                    .max()
                    .unwrap_or(-1);
                max_val + 1
            } else {
                0
            };
            let text_len = sl - st;
            let mut text_seg = Vec::with_capacity(text_len);
            for j in 0..text_len {
                let p = st_idx + j as i32;
                text_seg.push([p, p, p]);
            }
            llm_pos_segments.push(text_seg);
        }

        if !llm_pos_segments.is_empty() {
            let mut flat_pos: Vec<[i32; 3]> = Vec::with_capacity(sl);
            for seg in &llm_pos_segments {
                flat_pos.extend_from_slice(seg);
            }

            let max_pos = flat_pos
                .iter()
                .flat_map(|p| p.iter())
                .copied()
                .max()
                .unwrap_or(-1);

            // Write computed positions
            for (j, pos) in flat_pos.iter().enumerate() {
                let idx = off + j;
                pos_ids[idx] = pos[0];
                pos_ids[total + idx] = pos[1];
                pos_ids[2 * total + idx] = pos[2];
            }

            // Apply mask: restore default (1) for masked-out positions
            if let Some(ref mv) = mask_vals {
                let bm = &mv[off..off + sl];
                for (j, &m) in bm.iter().enumerate() {
                    if m == 0 {
                        let idx = off + j;
                        pos_ids[idx] = 1;
                        pos_ids[total + idx] = 1;
                        pos_ids[2 * total + idx] = 1;
                    }
                }
            }

            mrope_position_deltas.push(max_pos + 1 - seq_length);
        }
    }

    if mrope_position_deltas.is_empty() {
        mrope_position_deltas = vec![0i32; bs];
    }

    let pos_array = Array::from_slice(&pos_ids, &[3, batch_size, seq_length]);
    Ok((pos_array, mrope_position_deltas))
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_TOKEN_ID: i32 = 248056;
    const VIDEO_TOKEN_ID: i32 = 248057;
    const VISION_START_ID: i32 = 248053;
    const VISION_END_ID: i32 = 248054;

    #[test]
    fn test_get_rope_index_text_only() {
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5]);
        let (result, rope_delta) = get_rope_index(
            &input_ids,
            None,
            None,
            2,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
            VISION_START_ID,
            None,
        )
        .unwrap();
        assert_eq!(result.shape(), &[3, 1, 5]);
        assert_eq!(rope_delta, vec![0]);
        let flat = result.reshape(&[15]).unwrap();
        flat.eval().unwrap();
        let vals: Vec<i32> = flat.as_slice::<i32>().to_vec();
        for axis in 0..3 {
            for i in 0..5 {
                assert_eq!(vals[axis * 5 + i], i as i32);
            }
        }
    }

    #[test]
    fn test_get_rope_index_with_image() {
        // [text, vision_start, img_pad×4, vision_end, text]
        // Swift scans for imageTokenId: ed=2, textLen=2 (text1 + vision_start)
        let tokens = vec![
            1i32,
            VISION_START_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            VISION_END_ID,
            99,
        ];
        let len = tokens.len();
        let input_ids = Array::from_slice(&tokens, &[1, len as i32]);
        let grid = [THW { t: 1, h: 4, w: 4 }]; // llm_grid 1×2×2, n_vis=4

        let (result, rope_delta) = get_rope_index(
            &input_ids,
            Some(&grid),
            None,
            2,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
            VISION_START_ID,
            None,
        )
        .unwrap();
        assert_eq!(result.shape(), &[3, 1, len as i32]);

        let flat = result.reshape(&[(3 * len) as i32]).unwrap();
        flat.eval().unwrap();
        let vals: Vec<i32> = flat.as_slice::<i32>().to_vec();

        // idx 0: text "1" → pos (0,0,0)
        assert_eq!(vals[0], 0);
        assert_eq!(vals[len], 0);
        assert_eq!(vals[2 * len], 0);

        // idx 1: vision_start (text segment) → pos (1,1,1)
        assert_eq!(vals[1], 1);
        assert_eq!(vals[len + 1], 1);
        assert_eq!(vals[2 * len + 1], 1);

        // idx 2-5: vision pad, base = textLen + stIdx = 2 + 0 = 2, grid 1×2×2
        // vi=0: t=0,h=0,w=0 → (2,2,2)
        assert_eq!(vals[2], 2);
        assert_eq!(vals[len + 2], 2);
        assert_eq!(vals[2 * len + 2], 2);
        // vi=1: t=0,h=0,w=1 → (2,2,3)
        assert_eq!(vals[3], 2);
        assert_eq!(vals[len + 3], 2);
        assert_eq!(vals[2 * len + 3], 3);
        // vi=2: t=0,h=1,w=0 → (2,3,2)
        assert_eq!(vals[4], 2);
        assert_eq!(vals[len + 4], 3);
        assert_eq!(vals[2 * len + 4], 2);
        // vi=3: t=0,h=1,w=1 → (2,3,3)
        assert_eq!(vals[5], 2);
        assert_eq!(vals[len + 5], 3);
        assert_eq!(vals[2 * len + 5], 3);

        // idx 6-7: vision_end + text (next text segment)
        // stIdx = max(2,3,3)+1 = 4
        // idx 6: vision_end → pos (4,4,4)
        assert_eq!(vals[6], 4);
        assert_eq!(vals[len + 6], 4);
        assert_eq!(vals[2 * len + 6], 4);
        // idx 7: text "99" → pos (5,5,5)
        assert_eq!(vals[7], 5);
        assert_eq!(vals[len + 7], 5);
        assert_eq!(vals[2 * len + 7], 5);

        // maxPosId=5, delta = 5+1-8 = -2
        assert_eq!(rope_delta, vec![-2]);
    }

    #[test]
    fn test_get_rope_index_with_mask() {
        // Left-padded sequence: [PAD, PAD, text1, text2, text3]
        // mask:                  [0,   0,   1,     1,     1  ]
        // Swift zeros out masked tokens, processes full seq, then restores default (1)
        let tokens = vec![0i32, 0, 10, 20, 30];
        let mask_vals = vec![0i32, 0, 1, 1, 1];
        let len = tokens.len();
        let input_ids = Array::from_slice(&tokens, &[1, len as i32]);
        let mask = Array::from_slice(&mask_vals, &[1, len as i32]);

        let (result, rope_delta) = get_rope_index(
            &input_ids,
            None,
            None,
            2,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
            VISION_START_ID,
            Some(&mask),
        )
        .unwrap();
        assert_eq!(result.shape(), &[3, 1, len as i32]);

        // Text-only with no vision tokens: returns simple arange, delta = 0
        assert_eq!(rope_delta, vec![0]);

        let flat = result.reshape(&[(3 * len) as i32]).unwrap();
        flat.eval().unwrap();
        let vals: Vec<i32> = flat.as_slice::<i32>().to_vec();

        // Text-only path doesn't use mask (returns early before mask processing)
        for axis in 0..3 {
            for i in 0..5 {
                assert_eq!(vals[axis * 5 + i], i as i32);
            }
        }
    }

    #[test]
    fn test_get_rope_index_masked_multimodal() {
        // Left-padded: [PAD, text, vision_start, img×4, vision_end, text]
        // mask:         [0,   1,    1,            1×4,   1,          1   ]
        let tokens = vec![
            0i32,
            1,
            VISION_START_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            VISION_END_ID,
            99,
        ];
        let mask_vals = vec![0i32, 1, 1, 1, 1, 1, 1, 1, 1];
        let len = tokens.len();
        let input_ids = Array::from_slice(&tokens, &[1, len as i32]);
        let mask = Array::from_slice(&mask_vals, &[1, len as i32]);
        let grid = [THW { t: 1, h: 4, w: 4 }];

        let (result, rope_delta) = get_rope_index(
            &input_ids,
            Some(&grid),
            None,
            2,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
            VISION_START_ID,
            Some(&mask),
        )
        .unwrap();
        assert_eq!(result.shape(), &[3, 1, len as i32]);

        let flat = result.reshape(&[(3 * len) as i32]).unwrap();
        flat.eval().unwrap();
        let vals: Vec<i32> = flat.as_slice::<i32>().to_vec();

        // idx 0: PAD (mask=0) → default position (1,1,1)
        assert_eq!(vals[0], 1, "masked position should be default 1");
        assert_eq!(vals[len], 1);
        assert_eq!(vals[2 * len], 1);

        // idx 1: text "1" (mask=1, token was zeroed → treated as text after mask)
        // Computed positions start from 0 for this batch
        // The masked token at idx 0 is zeroed (not image/video), so text segment includes it
        // Then after writing, mask=0 positions get default (1)

        // idx 8: last text token should have valid position
        assert!(vals[8] > 0, "unmasked text should have valid position");

        // delta is per-batch
        assert_eq!(rope_delta.len(), 1);
    }
}
