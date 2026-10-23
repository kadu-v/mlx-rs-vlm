pub mod attention;
pub mod mlp;
pub mod patch_embed;
pub mod patch_merger;
pub mod rope;

use mlx::Array;
use mlx::builder::Builder;
use mlx::error::Exception;
use mlx::macros::ModuleParameters;
use mlx::module::Module;
use mlx::nn::{Embedding, LayerNorm};
use mlx::ops::indexing::IndexOp;
use mlx::ops::{
    LinspaceOptions, arange, broadcast_to, concatenate, cumsum, expand_dims, linspace, minimum,
    pad, stack, tile,
};
use mlx_rs as mlx;

use crate::config::{THW, VisionConfiguration};

use self::attention::VisionAttention;
use self::mlp::VisionMlp;
use self::patch_embed::PatchEmbed;
use self::patch_merger::PatchMerger;
use self::rope::VisionRotaryEmbedding;

/// A single vision transformer block.
///
/// Consists of pre-norm attention and pre-norm MLP, with residual connections:
/// x = x + attn(norm1(x))
/// x = x + mlp(norm2(x))
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx)]
pub struct VisionBlock {
    #[param]
    pub norm1: LayerNorm,

    #[param]
    pub attn: VisionAttention,

    #[param]
    pub norm2: LayerNorm,

    #[param]
    pub mlp: VisionMlp,
}

impl VisionBlock {
    /// Create a new VisionBlock from configuration.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        let norm1 = mlx::nn::LayerNormBuilder::new(config.hidden_size)
            .eps(1e-6_f32)
            .build()?;
        let norm2 = mlx::nn::LayerNormBuilder::new(config.hidden_size)
            .eps(1e-6_f32)
            .build()?;
        let attn = VisionAttention::new(config)?;
        let mlp = VisionMlp::new(config)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    /// Forward pass with rotary position embeddings and block-diagonal attention.
    ///
    /// # Arguments
    /// * `hidden_states` - [seq_len, hidden_size]
    /// * `cu_seqlens` - Cumulative sequence lengths for block-diagonal masking
    /// * `rot_pos_emb` - Rotary position embeddings [seq_len, head_dim/2]
    pub fn forward_with_rope(
        &mut self,
        hidden_states: &Array,
        cu_seqlens: &Array,
        rot_pos_emb: &Array,
    ) -> Result<Array, Exception> {
        // Pre-norm attention with residual
        let normed = self.norm1.forward(hidden_states)?;
        let attn_out = self
            .attn
            .forward_with_rope(&normed, cu_seqlens, rot_pos_emb)?;
        let states = hidden_states + &attn_out;

        // Pre-norm MLP with residual
        let normed = self.norm2.forward(&states)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let states = &states + &mlp_out;

        Ok(states)
    }
}

/// Complete vision model for Qwen 3.5 VLM.
///
/// Combines patch embedding, positional embeddings, transformer blocks,
/// rotary position embeddings, and a patch merger for spatial downsampling.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx)]
pub struct VisionModel {
    #[param]
    pub patch_embed: PatchEmbed,

    #[param]
    pub pos_embed: Embedding,

    #[param]
    pub blocks: Vec<VisionBlock>,

    #[param]
    pub merger: PatchMerger,

    pub rotary_embedding: VisionRotaryEmbedding,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub spatial_merge_size: i32,
    pub num_grid_per_side: i32,
}

impl VisionModel {
    /// Create a new VisionModel from configuration.
    pub fn new(config: &VisionConfiguration) -> Result<Self, Exception> {
        let patch_embed = PatchEmbed::new(config)?;

        let head_dim = config.hidden_size / config.num_heads;
        let rotary_embedding = VisionRotaryEmbedding::new(head_dim / 2, 10000.0);

        let pos_embed = Embedding::new(config.num_position_embeddings, config.hidden_size)?;

        let blocks = (0..config.depth)
            .map(|_| VisionBlock::new(config))
            .collect::<Result<Vec<_>, _>>()?;

        let merger = PatchMerger::new(config, false)?;

        let num_grid_per_side = (config.num_position_embeddings as f64).sqrt() as i32;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            rotary_embedding,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            spatial_merge_size: config.spatial_merge_size,
            num_grid_per_side,
        })
    }

    /// Compute rotary position embeddings from grid specifications.
    ///
    /// For each grid, computes spatial coordinates for the merged patches,
    /// looks up frequency embeddings, and concatenates height and width embeddings.
    fn rotary_position_embedding(&self, grids: &[THW]) -> Result<Array, Exception> {
        let max_hw = grids.iter().map(|g| g.h.max(g.w)).max().unwrap_or(0);
        if max_hw == 0 {
            return mlx::ops::zeros::<f32>(&[1, 1]);
        }

        let freq_table = self.rotary_embedding.forward(max_hw)?;
        let half_dim = freq_table.dim(-1);
        let merge = self.spatial_merge_size;

        let mut all_coords: Vec<Array> = Vec::new();

        for grid in grids {
            let merged_h = grid.h / merge;
            let merged_w = grid.w / merge;

            if merged_h <= 0 || merged_w <= 0 {
                continue;
            }

            // Generate block and intra-block indices
            let block_rows =
                arange::<_, i32>(None, merged_h, None)?.reshape(&[merged_h, 1, 1, 1])?;
            let block_cols =
                arange::<_, i32>(None, merged_w, None)?.reshape(&[1, merged_w, 1, 1])?;

            let intra = arange::<_, i32>(None, merge, None)?;
            let intra_row = intra.reshape(&[1, 1, merge, 1])?;
            let intra_col = intra.reshape(&[1, 1, 1, merge])?;

            let merge_scalar = Array::from_int(merge);

            // h_index = block_row * merge + intra_row
            // w_index = block_col * merge + intra_col
            let h_index = &(&block_rows * &merge_scalar) + &intra_row;
            let w_index = &(&block_cols * &merge_scalar) + &intra_col;

            let target_shape = &[merged_h, merged_w, merge, merge];
            let h_index = broadcast_to(&h_index, target_shape)?;
            let w_index = broadcast_to(&w_index, target_shape)?;

            let h_flat = h_index.flatten(None, None)?;
            let w_flat = w_index.flatten(None, None)?;

            // Stack coordinate pairs: [n_patches, 2]
            let mut coords = stack(&[&h_flat, &w_flat], -1)?;

            // Repeat for temporal frames
            if grid.t > 1 {
                coords = tile(&coords, &[grid.t, 1])?;
            }

            all_coords.push(coords);
        }

        if all_coords.is_empty() {
            return mlx::ops::zeros::<f32>(&[0, half_dim * 2]);
        }

        // Concatenate all coordinates: [total_patches, 2]
        let coord_refs: Vec<&Array> = all_coords.iter().collect();
        let all_coords_concat = concatenate(&coord_refs, 0)?;

        // Extract h and w indices
        let h_indices = all_coords_concat.index((.., 0));
        let w_indices = all_coords_concat.index((.., 1));

        // Look up embeddings
        let h_embeds = freq_table.index(&h_indices);
        let w_embeds = freq_table.index(&w_indices);

        // Concatenate height and width embeddings: [total_patches, dim]
        concatenate(&[&h_embeds, &w_embeds], -1)
    }

    /// Compute bilinearly interpolated positional embeddings for the given grids.
    ///
    /// Uses the learned position embedding table and bilinear interpolation
    /// to handle arbitrary spatial resolutions.
    fn positional_embeddings(&mut self, grids: &[THW]) -> Result<Array, Exception> {
        let hidden_size = self.hidden_size;
        let max_index = self.num_grid_per_side - 1;

        // Step 1: Collect all indices and weights from all grids
        let mut corner_indices: [Vec<Array>; 4] = [vec![], vec![], vec![], vec![]];
        let mut corner_weights: [Vec<Array>; 4] = [vec![], vec![], vec![], vec![]];
        let mut grid_sizes: Vec<i32> = Vec::new();

        for grid in grids {
            let h = grid.h;
            let w = grid.w;
            grid_sizes.push(h * w);

            // Create linspace indices
            let h_linspace = linspace::<f32, f32>(
                0.0,
                max_index as f32,
                LinspaceOptions {
                    count: h,
                    endpoint: true,
                },
            )?;
            let w_linspace = linspace::<f32, f32>(
                0.0,
                max_index as f32,
                LinspaceOptions {
                    count: w,
                    endpoint: true,
                },
            )?;

            // Floor and ceil
            let h_floor = h_linspace.as_type::<i32>()?;
            let h_ceil = minimum(
                &(&h_floor + &Array::from_int(1)),
                &Array::from_int(max_index),
            )?;
            let dh = &h_linspace - &h_floor.as_type::<f32>()?;

            let w_floor = w_linspace.as_type::<i32>()?;
            let w_ceil = minimum(
                &(&w_floor + &Array::from_int(1)),
                &Array::from_int(max_index),
            )?;
            let dw = &w_linspace - &w_floor.as_type::<f32>()?;

            // Broadcast to create meshgrid
            let h_floor_exp = expand_dims(&h_floor, 1)?; // (h, 1)
            let h_ceil_exp = expand_dims(&h_ceil, 1)?;
            let w_floor_exp = expand_dims(&w_floor, 0)?; // (1, w)
            let w_ceil_exp = expand_dims(&w_ceil, 0)?;

            let ngs = Array::from_int(self.num_grid_per_side);
            let base_h = &h_floor_exp * &ngs;
            let base_h_ceil = &h_ceil_exp * &ngs;

            // 4 corner indices
            corner_indices[0].push((&base_h + &w_floor_exp).flatten(None, None)?);
            corner_indices[1].push((&base_h + &w_ceil_exp).flatten(None, None)?);
            corner_indices[2].push((&base_h_ceil + &w_floor_exp).flatten(None, None)?);
            corner_indices[3].push((&base_h_ceil + &w_ceil_exp).flatten(None, None)?);

            // Bilinear weights
            let dh_exp = expand_dims(&dh, 1)?;
            let dw_exp = expand_dims(&dw, 0)?;
            let one = Array::from_f32(1.0);

            corner_weights[0].push(((&one - &dh_exp) * &(&one - &dw_exp)).flatten(None, None)?);
            corner_weights[1].push(((&one - &dh_exp) * &dw_exp).flatten(None, None)?);
            corner_weights[2].push((&dh_exp * &(&one - &dw_exp)).flatten(None, None)?);
            corner_weights[3].push((&dh_exp * &dw_exp).flatten(None, None)?);
        }

        if corner_indices[0].is_empty() {
            return mlx::ops::zeros::<f32>(&[0, hidden_size]);
        }

        // Step 2: Batch embedding lookup
        let indices_tensors: Vec<Array> = corner_indices
            .iter()
            .map(|idx_list| {
                let refs: Vec<&Array> = idx_list.iter().collect();
                let cat = concatenate(&refs, 0).unwrap();
                cat.as_type::<i32>().unwrap()
            })
            .collect();

        let weights_tensors: Vec<Array> = corner_weights
            .iter()
            .map(|w_list| {
                let refs: Vec<&Array> = w_list.iter().collect();
                concatenate(&refs, 0).unwrap()
            })
            .collect();

        let total_patches = indices_tensors[0].dim(0);
        let mut patch_pos_embeds = mlx::ops::zeros::<f32>(&[total_patches, hidden_size])?;

        for corner_idx in 0..4 {
            let corner_embeds = self.pos_embed.forward(&indices_tensors[corner_idx])?;
            let weighted = &corner_embeds * &expand_dims(&weights_tensors[corner_idx], -1)?;
            patch_pos_embeds = &patch_pos_embeds + &weighted;
        }

        // Step 3: Split by grid and process temporal / merge pattern
        let merge = self.spatial_merge_size;
        let mut result_embeds: Vec<Array> = Vec::new();
        let mut offset = 0;

        for grid in grids {
            let h = grid.h;
            let w = grid.w;
            let t = grid.t;
            let size = h * w;

            let pos_emb = patch_pos_embeds.index((offset..offset + size, ..));
            let feature_dim = pos_emb.dim(-1);
            offset += size;

            // Repeat for temporal dimension
            let mut temporal_embeds = tile(&pos_emb, &[t, 1])?;

            // Reshape for merge pattern:
            // [t*h*w, feat] -> [t, h, w, feat] -> [t, h/merge, merge, w/merge, merge, feat]
            // -> transpose(0, 1, 3, 2, 4, 5) -> flatten to [-1, feat]
            temporal_embeds = temporal_embeds.reshape(&[t, h, w, feature_dim])?;
            temporal_embeds =
                temporal_embeds.reshape(&[t, h / merge, merge, w / merge, merge, feature_dim])?;
            temporal_embeds = temporal_embeds.transpose_axes(&[0, 1, 3, 2, 4, 5])?;
            temporal_embeds = temporal_embeds.reshape(&[-1, feature_dim])?;

            result_embeds.push(temporal_embeds);
        }

        let result_refs: Vec<&Array> = result_embeds.iter().collect();
        concatenate(&result_refs, 0)
    }

    /// Compute cumulative sequence lengths for block-diagonal attention.
    ///
    /// For each grid, the per-frame patch count is h*w, repeated t times.
    /// Returns a 1D array of cumulative sums with a leading zero.
    fn cumulative_sequence_lengths(&self, grids: &[THW]) -> Result<Array, Exception> {
        let mut seq_lengths: Vec<Array> = Vec::new();

        for grid in grids {
            let per_frame = grid.h * grid.w;
            let per_frame_arr = Array::from_int(per_frame);
            let repeated = tile(&per_frame_arr, &[grid.t])?;
            seq_lengths.push(repeated);
        }

        if seq_lengths.is_empty() {
            return Ok(Array::from_int(0));
        }

        let len_refs: Vec<&Array> = seq_lengths.iter().collect();
        let concat_lengths = concatenate(&len_refs, 0)?.as_type::<i32>()?;
        let cum_sum = cumsum(&concat_lengths, None, None, None)?;

        // Pad with leading zero
        pad(&cum_sum, &[(1, 0)], None::<Array>, None)
    }

    /// Forward pass of the vision model.
    ///
    /// # Arguments
    /// * `pixel_values` - Flattened pixel values
    /// * `grid_thw` - Grid specifications (temporal, height, width) for each image/video
    ///
    /// # Returns
    /// Tuple of (merged hidden states, deepstack outputs)
    pub fn forward_with_grid(
        &mut self,
        pixel_values: &Array,
        grid_thw: &[THW],
    ) -> Result<(Array, Vec<Array>), Exception> {
        // 1. Patch embedding
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        // 2. Add positional embeddings
        let pos_embeds = self.positional_embeddings(grid_thw)?;
        hidden_states = &hidden_states + &pos_embeds;

        // 3. Compute rotary embeddings and cumulative sequence lengths
        let rotary_embeds = self.rotary_position_embedding(grid_thw)?;
        let cu_seqlens = self.cumulative_sequence_lengths(grid_thw)?;

        // 4. Transformer blocks
        let deepstack_outputs: Vec<Array> = Vec::new();
        for block in self.blocks.iter_mut() {
            hidden_states = block.forward_with_rope(&hidden_states, &cu_seqlens, &rotary_embeds)?;
        }

        // 5. Patch merger
        hidden_states = self.merger.forward(&hidden_states)?;

        Ok((hidden_states, deepstack_outputs))
    }

    /// Sanitize weight names for loading from different checkpoint formats.
    ///
    /// Handles transposition of patch_embed.proj.weight when it comes from
    /// PyTorch format (channels first -> channels last).
    pub fn sanitize(
        &self,
        weights: std::collections::HashMap<String, Array>,
    ) -> std::collections::HashMap<String, Array> {
        let in_channels = self.patch_embed.in_channels;
        let mut sanitized = std::collections::HashMap::new();

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue;
            } else if key.contains("patch_embed.proj.weight") {
                // Check if weight needs transposition for channels-last format
                if value.ndim() == 5 && value.dim(-1) == in_channels {
                    sanitized.insert(key, value);
                } else {
                    // Transpose from [out, in, D, H, W] to [out, D, H, W, in]
                    let transposed = value.transpose_axes(&[0, 2, 3, 4, 1]).unwrap();
                    sanitized.insert(key, transposed);
                }
            } else {
                sanitized.insert(key, value);
            }
        }

        sanitized
    }
}
