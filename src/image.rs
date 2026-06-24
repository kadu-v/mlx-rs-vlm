use anyhow::{bail, Result};
use mlx_rs::Array;

use crate::config::{ProcessorConfig, THW, VisionConfiguration};

pub fn target_size(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32)> {
    if height < factor || width < factor {
        bail!("Image dimensions must be >= factor ({factor})");
    }
    let ratio = width as f64 / height as f64;
    if ratio > 200.0 || ratio < 1.0 / 200.0 {
        bail!("Aspect ratio too extreme: {ratio}");
    }

    let mut h_bar = ((height as f64 / factor as f64).round() * factor as f64) as u32;
    let mut w_bar = ((width as f64 / factor as f64).round() * factor as f64) as u32;
    h_bar = h_bar.max(factor);
    w_bar = w_bar.max(factor);

    if h_bar * w_bar > max_pixels {
        let beta = ((height as f64 * width as f64) / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() * factor as f64) as u32;
        w_bar = ((width as f64 / beta / factor as f64).floor() * factor as f64) as u32;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() * factor as f64) as u32;
        w_bar = ((width as f64 * beta / factor as f64).ceil() * factor as f64) as u32;
    }

    h_bar = (h_bar / factor) * factor;
    w_bar = (w_bar / factor) * factor;

    if h_bar == 0 || w_bar == 0 {
        bail!("Invalid target dimensions: {w_bar} x {h_bar}");
    }

    Ok((h_bar, w_bar))
}

pub fn preprocess_image(
    img: &image::DynamicImage,
    config: &VisionConfiguration,
    proc_config: &ProcessorConfig,
) -> Result<(Array, THW)> {
    let factor = (config.patch_size * config.spatial_merge_size) as u32;

    let (orig_h, orig_w) = (img.height(), img.width());
    let (target_h, target_w) = target_size(
        orig_h,
        orig_w,
        factor,
        proc_config.min_pixels,
        proc_config.max_pixels,
    )?;

    let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb32f();

    let mean = &proc_config.image_mean;
    let std = &proc_config.image_std;

    let (h, w) = (rgb.height() as i32, rgb.width() as i32);
    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let [r, g, b] = p.0;
            [
                (r - mean[0]) / std[0],
                (g - mean[1]) / std[1],
                (b - mean[2]) / std[2],
            ]
        })
        .collect();

    let array = Array::from_slice(&pixels, &[1, h, w, 3]);

    let temporal_patch = config.temporal_patch_size;

    // Temporal padding: repeat frame to match temporal_patch_size
    let padded = if 1 < temporal_patch {
        mlx_rs::ops::tile(&array, &[temporal_patch, 1, 1, 1])?
    } else {
        array
    };

    let patches = patchify(&padded, config)?;

    let grid_t = temporal_patch / temporal_patch; // = 1
    let grid_h = h / config.patch_size;
    let grid_w = w / config.patch_size;

    let thw = THW {
        t: grid_t,
        h: grid_h,
        w: grid_w,
    };

    Ok((patches, thw))
}

pub fn patchify(
    frames: &Array,
    config: &VisionConfiguration,
) -> Result<Array> {
    let ps = config.patch_size;
    let tp = config.temporal_patch_size;
    let c = config.in_channels;
    let m = config.spatial_merge_size;

    let shape = frames.shape();
    let (n, h, w) = (shape[0], shape[1], shape[2]);

    let grid_t = n / tp;
    let grid_h_m = h / ps / m;
    let grid_w_m = w / ps / m;

    // [N, H, W, C] → [gT, tp, gHm, m, ps, gWm, m, ps, C]
    let patches = frames.reshape(&[grid_t, tp, grid_h_m, m, ps, grid_w_m, m, ps, c])?;

    // → [gT, gHm, gWm, m, m, C, tp, ps, ps] (merge-block order, matching Swift)
    let patches = patches.transpose_axes(&[0, 2, 5, 3, 6, 8, 1, 4, 7])?;

    // → [gT*gH*gW, C*tp*ps*ps]
    let patches = patches.reshape(&[-1, c * tp * ps * ps])?;

    Ok(patches)
}

pub fn expand_image_tokens(
    tokens: &[i32],
    thw_list: &[THW],
    image_pad_id: i32,
    vision_start_id: i32,
    vision_end_id: i32,
    merge_size: i32,
) -> Vec<i32> {
    let merge_length = merge_size * merge_size;
    let mut result = Vec::new();
    let mut thw_idx = 0;
    let mut i = 0;

    while i < tokens.len() {
        if i + 2 < tokens.len()
            && tokens[i] == vision_start_id
            && tokens[i + 1] == image_pad_id
            && tokens[i + 2] == vision_end_id
            && thw_idx < thw_list.len()
        {
            let thw = &thw_list[thw_idx];
            let pad_count = (thw.t * thw.h * thw.w) / merge_length;
            result.push(vision_start_id);
            for _ in 0..pad_count {
                result.push(image_pad_id);
            }
            result.push(vision_end_id);
            thw_idx += 1;
            i += 3;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchify_shape() {
        let frames = Array::zeros::<f32>(&[2, 32, 32, 3]).unwrap();
        let config = VisionConfiguration {
            depth: 24,
            hidden_size: 1024,
            num_heads: 16,
            intermediate_size: 4096,
            patch_size: 16,
            in_channels: 3,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 2048,
            num_position_embeddings: 2304,
            hidden_act: None,
        };
        let patches = patchify(&frames, &config).unwrap();
        // (2/2) * (32/16) * (32/16) = 4 patches, each 2*16*16*3 = 1536 dims
        assert_eq!(patches.shape(), &[4, 1536]);
    }

    #[test]
    fn test_target_size() {
        let min_px = 4 * 28 * 28;
        let max_px = 16384 * 28 * 28;

        let (h, w) = target_size(1024, 768, 32, min_px, max_px).unwrap();
        assert!(h % 32 == 0);
        assert!(w % 32 == 0);
        assert!(h * w <= max_px);
        assert!(h * w >= min_px);

        let (h, w) = target_size(100, 100, 32, min_px, max_px).unwrap();
        assert!(h % 32 == 0);
        assert!(w % 32 == 0);
        assert!(h * w >= min_px);

        // Swift parity: scale uses original h*w, not rounded
        let (h1, w1) = target_size(4000, 3000, 32, min_px, max_px).unwrap();
        assert!(h1 % 32 == 0);
        assert!(w1 % 32 == 0);
        assert!(h1 * w1 <= max_px);
    }
}
