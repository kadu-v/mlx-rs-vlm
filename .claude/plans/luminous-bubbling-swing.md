# Vision Pipeline 修正計画

## Context

E2Eテスト `test_image_generation` が以下のエラーで失敗:
```
[broadcast_shapes] Shapes (1840,1024) and (0,1024) cannot be broadcast.
```
`vision/mod.rs:forward_with_grid` の `hidden_states + pos_embeds` で、patch_embedの出力 `(1840, 1024)` と positional_embeddings の出力 `(0, 1024)` が合わない。

## 根本原因

`src/image.rs` の `preprocess_image` に3つのバグがある:

### Bug 1: patchify未呼び出し
Swift実装では `QwenVL.patchify()` で画像をパッチ化してからvision modelに渡す。Rust版は生ピクセル `[1, H, W, 3]` をそのまま渡している。PatchEmbed.forward は `[-1, C, tp, ps, ps]` にreshapeするため、patchify済みデータを期待する。

### Bug 2: THW.t が 0
```rust
t: grid_t / temporal_patch,  // 1 / 2 = 0（整数除算で切り捨て）
```
Swiftでは temporal_patch_size 未満のフレーム数を padding（最終フレームの複製）してから除算するため `t = 2/2 = 1`。

### Bug 3: THW.h, THW.w が merge で割られている
```rust
h: grid_h / merge_size,  // (H/16) / 2 — 不正
w: grid_w / merge_size,  // (W/16) / 2 — 不正
```
Swift実装では `THW(gridT, gridH, gridW)` で返す（merge で割らない）。positional_embeddings や rotary_position_embedding が内部で merge 除算するため、THW は**パッチ数そのまま**を保持すべき。

### Bug 4: patchify の並び順が不正
現在の Rust `patchify` はラスタ順だが、Swift版は merge-block 順（merge隣接パッチが連続）。PatchMerger が `reshape([-1, merged_dim])` で連続 merge² パッチを結合するため、merge-block 順でなければならない。

## Swift リファレンス（QwenVL.patchify）

```swift
// 入力: [N, C, H, W]（channels-first）
// Temporal padding: N % temporalPatchSize != 0 の場合、最終フレーム複製
patches = patches.reshaped(gridT, tp, C, gH/m, m, ps, gW/m, m, ps)
patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
// → [gT, gH/m, gW/m, m, m, C, tp, ps, ps]
flattenedPatches = patches.reshaped(gT*gH*gW, C*tp*ps*ps)
return (flattenedPatches, THW(gridT, gridH, gridW))
```

## 修正内容

**変更ファイル: `src/image.rs` のみ**

### 1. `patchify` 関数の書き換え

channels-last入力 `[N, H, W, C]` に対して merge-block 順で patchify:

```rust
// [N, H, W, C] → [gT, tp, gHm, m, ps, gWm, m, ps, C]
reshape(&[grid_t, tp, grid_h/merge, merge, ps, grid_w/merge, merge, ps, C])
// transpose → [gT, gHm, gWm, m, m, C, tp, ps, ps]
transpose_axes(&[0, 2, 5, 3, 6, 8, 1, 4, 7])
// flatten → [gT*gH*gW, C*tp*ps*ps]
reshape(&[-1, C * tp * ps * ps])
```

### 2. `preprocess_image` 関数の修正

```
現状:   raw pixels [1, H, W, 3] + THW{t:0, h:(H/16)/2, w:(W/16)/2}
修正後: patchified [n_patches, 1536] + THW{t:1, h:H/16, w:W/16}
```

手順:
1. リサイズ・正規化（変更なし）
2. temporal padding: `[1, H, W, C]` → `tile([tp, 1, 1, 1])` → `[2, H, W, C]`
3. `patchify()` 呼び出し → `[n_patches, 1536]`
4. THW: `t = padded/tp = 1`, `h = H/ps`, `w = W/ps`（merge で割らない）

## 影響範囲の確認

- `vision/mod.rs`: 変更不要。positional_embeddings, rotary_position_embedding, cumulative_sequence_lengths は全て THW が正しければ正しく動く
- `model/qwen35.rs`: 変更不要。get_rope_index は内部で `grid.h / spatial_merge_size` する
- `vision/patch_embed.rs`: 変更不要。patchify済み入力を期待する設計
- `test_patchify_shape`: 出力shape `[4, 1536]` は変わらない（同じ総パッチ数）

## 検証

```bash
cargo test -- --test-threads=1                    # 既存11テスト全パス
cargo test -- --ignored --test-threads=1          # test_image_generation がパス
```
