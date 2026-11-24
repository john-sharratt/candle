//! Where an image's pieces go: its size on the patch grid, the order its
//! patches are read in, the taps that resample the learned position table, the
//! rotary positions inside the tower, and the multimodal positions the text
//! trunk gives its rows.
//!
//! All of it is a function of the image's grid alone, so it is built on the
//! host once per image and tested here against hand-computed values; nothing in
//! this file touches a tensor.
//!
//! Design: `docs/flash_next_vision.md` §1.3, §2, §4.4.

/// Pixels per patch side.
pub const PATCH: usize = 16;
/// Patches per merge-block side: the merger concatenates `MERGE × MERGE`
/// patches into one row of the text trunk.
pub const MERGE: usize = 2;
/// The resize factor: every resized side is a multiple of a merge block.
pub const FACTOR: usize = PATCH * MERGE;
/// The fewest pixels an image is scaled up to: 64 image tokens. The
/// checkpoint's own `preprocessor_config.json` value.
pub const MIN_PIXELS: usize = 65_536;
/// The most pixels an image is scaled down to: 4 096 image tokens. The
/// checkpoint allows 16 384; past 4 096 an image costs a turn more positions
/// than a coding conversation carries in a page of text, and the tower's
/// attention more than a 16 GB card lends a guest.
pub const MAX_PIXELS: usize = 4_194_304;
/// The widest aspect ratio accepted, as the reference processor refuses.
pub const MAX_ASPECT: f64 = 200.0;
/// The learned position table's side: 2 304 entries, a 48 × 48 grid.
pub const POS_SIDE: usize = 48;
/// Rotary frequencies of the tower's 2D RoPE per axis: the head is 72 wide,
/// so 36 pairs, half for the patch row and half for its column.
pub const VIT_ROPE_FREQS: usize = 18;
/// The tower's RoPE base.
pub const VIT_ROPE_THETA: f64 = 10_000.0;

/// An image's size after `smart_resize`, in pixels and in patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Resized height and width in pixels, each a multiple of [`FACTOR`].
    pub height: usize,
    pub width: usize,
}

impl Grid {
    /// Patch rows.
    pub fn rows(&self) -> usize {
        self.height / PATCH
    }

    /// Patch columns.
    pub fn cols(&self) -> usize {
        self.width / PATCH
    }

    /// Patches the tower reads.
    pub fn patches(&self) -> usize {
        self.rows() * self.cols()
    }

    /// Merged rows (`h'`): the image's extent on the trunk's H axis.
    pub fn merged_rows(&self) -> usize {
        self.rows() / MERGE
    }

    /// Merged columns (`w'`): its extent on the W axis.
    pub fn merged_cols(&self) -> usize {
        self.cols() / MERGE
    }

    /// Rows the image becomes in the trunk: one per `<|image_pad|>`.
    pub fn tokens(&self) -> usize {
        self.merged_rows() * self.merged_cols()
    }

    /// How far the image moves the trunk's position: `max(h', w')`.
    pub fn advance(&self) -> usize {
        self.merged_rows().max(self.merged_cols())
    }
}

/// `round(x / f) · f`, rounding a half to even, as Python's `round` does.
fn round_to(x: f64, f: usize) -> usize {
    ((x / f as f64).round_ties_even() as usize) * f
}

/// The reference processor's `smart_resize`: both sides to multiples of
/// [`FACTOR`], then scaled into `MIN_PIXELS ..= MAX_PIXELS` keeping the
/// aspect. Refuses an empty image and one past [`MAX_ASPECT`].
pub fn smart_resize(height: usize, width: usize) -> Result<Grid, String> {
    if height == 0 || width == 0 {
        return Err(format!("an image of {width}×{height} has no pixels"));
    }
    let (h, w) = (height as f64, width as f64);
    let aspect = h.max(w) / h.min(w);
    if aspect > MAX_ASPECT {
        return Err(format!(
            "an image of {width}×{height} has an aspect ratio of {aspect:.0}, past {MAX_ASPECT}"
        ));
    }
    let mut hb = round_to(h, FACTOR).max(FACTOR);
    let mut wb = round_to(w, FACTOR).max(FACTOR);
    if hb * wb > MAX_PIXELS {
        let beta = (h * w / MAX_PIXELS as f64).sqrt();
        hb = (((h / beta) / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
        wb = (((w / beta) / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
    } else if hb * wb < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        hb = ((h * beta) / FACTOR as f64).ceil() as usize * FACTOR;
        wb = ((w * beta) / FACTOR as f64).ceil() as usize * FACTOR;
    }
    Ok(Grid {
        height: hb,
        width: wb,
    })
}

/// Every patch's `(row, col)` in **merge-block order**: the 2 × 2 blocks
/// row-major over the grid, the four patches of a block `(0,0), (0,1), (1,0),
/// (1,1)` within it. The order the tower reads and the merger concatenates in.
pub fn patch_positions(grid: Grid) -> Vec<(usize, usize)> {
    let (bh, bw) = (grid.merged_rows(), grid.merged_cols());
    let mut out = Vec::with_capacity(grid.patches());
    for by in 0..bh {
        for bx in 0..bw {
            for dy in 0..MERGE {
                for dx in 0..MERGE {
                    out.push((by * MERGE + dy, bx * MERGE + dx));
                }
            }
        }
    }
    out
}

/// The two taps and their weights that resample a [`POS_SIDE`]-long axis to
/// `size` samples, bilinear with aligned corners and clamped borders: sample
/// `i` sits at `i · 47 / max(size − 1, 1)` of the table.
fn axis_taps(i: usize, size: usize) -> ([usize; 2], [f32; 2]) {
    let src = i as f64 * (POS_SIDE - 1) as f64 / (size.max(2) - 1) as f64;
    let lo = src.floor() as usize;
    let hi = (lo + 1).min(POS_SIDE - 1);
    let d = (src - lo as f64) as f32;
    ([lo, hi], [1.0 - d, d])
}

/// The position-table rows and weights each patch sums, in merge-block order:
/// four of each per patch, the outer product of its row's and its column's two
/// taps (`vision_utils.get_vision_interpolation_indices_and_weights`).
pub fn position_taps(grid: Grid) -> (Vec<u32>, Vec<f32>) {
    let (rows, cols) = (grid.rows(), grid.cols());
    let mut idx = Vec::with_capacity(grid.patches() * 4);
    let mut weight = Vec::with_capacity(grid.patches() * 4);
    for (r, c) in patch_positions(grid) {
        let (ri, rw) = axis_taps(r, rows);
        let (ci, cw) = axis_taps(c, cols);
        for a in 0..2 {
            for b in 0..2 {
                idx.push((ri[a] * POS_SIDE + ci[b]) as u32);
                weight.push(rw[a] * cw[b]);
            }
        }
    }
    (idx, weight)
}

/// The tower's rotary angles, `[patches, 36]` row-major in merge-block order:
/// pair `j < 18` turns by the patch's row at frequency `j`, pair `18 + j` by its
/// column at the same frequency. With NeoX pairing over the 72-wide head this is
/// the reference's `[row·f | col·f | row·f | col·f]` cos/sin layout.
pub fn vit_rope_angles(grid: Grid) -> Vec<f32> {
    let inv: Vec<f64> = (0..VIT_ROPE_FREQS)
        .map(|j| 1.0 / VIT_ROPE_THETA.powf(2.0 * j as f64 / (2 * VIT_ROPE_FREQS) as f64))
        .collect();
    let mut out = Vec::with_capacity(grid.patches() * 2 * VIT_ROPE_FREQS);
    for (r, c) in patch_positions(grid) {
        out.extend(inv.iter().map(|f| (r as f64 * f) as f32));
        out.extend(inv.iter().map(|f| (c as f64 * f) as f32));
    }
    out
}

/// The axis a trunk rotary pair turns on, under `mrope_interleaved` with
/// sections `[11, 11, 10]` over 32 pairs: `0` T, `1` H, `2` W.
pub fn trunk_axis(pair: usize) -> usize {
    match pair % 3 {
        1 => 1,
        2 => 2,
        _ => 0,
    }
}

/// How far the image row at index `i` (raster over the merged grid) of an
/// image of `grid` must be turned, per axis `(t, h, w)`, on top of the rotation
/// its slot gets: its multimodal position, anchored at the image's end, minus
/// its slot.
///
/// The image's `n` rows sit at slots `e − n .. e`, `e` the slot of
/// `<|vision_end|>`, and take positions `(e − m) + (0, r, c)` with
/// `m = max(h', w')`; so the text after the image relates to it exactly as the
/// reference's positions say, and only text before it sits further away
/// (`docs/flash_next_vision.md` §4.4).
pub fn image_delta(grid: Grid, i: usize) -> [i32; 3] {
    let n = grid.tokens() as i64;
    let m = grid.advance() as i64;
    let w = grid.merged_cols() as i64;
    let i = i as i64;
    let (r, c) = (i / w, i % w);
    [
        (n - m - i) as i32,
        (n - m + r - i) as i32,
        (n - m + c - i) as i32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference sizes, worked by hand from `smart_resize`.
    #[test]
    fn smart_resize_matches_the_reference_sizes() {
        let cases = [
            ((224, 224), (256, 256), 64, 8),
            ((480, 640), (480, 640), 300, 20),
            ((600, 800), (608, 800), 475, 25),
            ((720, 1280), (704, 1280), 880, 40),
            ((1080, 1920), (1088, 1920), 2040, 60),
            ((50, 100), (192, 384), 72, 12),
            ((1024, 1024), (1024, 1024), 1024, 32),
        ];
        for ((h, w), (rh, rw), tokens, advance) in cases {
            let g = smart_resize(h, w).unwrap();
            assert_eq!((g.height, g.width), (rh, rw), "{w}×{h}");
            assert_eq!(g.tokens(), tokens, "{w}×{h}");
            assert_eq!(g.advance(), advance, "{w}×{h}");
        }
    }

    /// Past the ceiling the image is scaled down by `floor` into 4 096 tokens.
    #[test]
    fn a_large_image_is_scaled_under_the_ceiling() {
        let g = smart_resize(2160, 3840).unwrap();
        // β = √(3840·2160/4194304) = 1.4062…; 2160/β/32 = 48.0 → 48·32,
        // 3840/β/32 = 85.33 → 85·32.
        assert_eq!((g.height, g.width), (1536, 2720));
        assert!(g.height * g.width <= MAX_PIXELS);
    }

    /// Half to even: 48 / 32 = 1.5 rounds to 2, 80 / 32 = 2.5 rounds to 2.
    #[test]
    fn rounding_is_half_to_even() {
        assert_eq!(round_to(48.0, 32), 64);
        assert_eq!(round_to(80.0, 32), 64);
        assert_eq!(round_to(112.0, 32), 128);
    }

    #[test]
    fn an_empty_or_too_narrow_image_is_refused() {
        assert!(smart_resize(0, 10).is_err());
        assert!(smart_resize(10, 2_100).is_err());
        assert!(smart_resize(100, 19_000).is_ok());
    }

    /// A 2 × 2-block grid: blocks row-major, patches within each block.
    #[test]
    fn patches_are_read_in_merge_block_order() {
        let g = Grid {
            height: 32,
            width: 64,
        };
        assert_eq!(
            patch_positions(g),
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
            ]
        );
    }

    /// The end samples land on the table's corners exactly, and the middle of
    /// a three-wide axis on its middle entry with no neighbour weight.
    #[test]
    fn the_axis_taps_align_the_corners() {
        assert_eq!(axis_taps(0, 32), ([0, 1], [1.0, 0.0]));
        assert_eq!(axis_taps(31, 32), ([47, 47], [1.0, 0.0]));
        assert_eq!(axis_taps(1, 3), ([23, 24], [0.5, 0.5]));
        // One sample: the corner.
        assert_eq!(axis_taps(0, 1), ([0, 1], [1.0, 0.0]));
        // 48 samples: the table itself.
        for i in 0..48 {
            let (t, w) = axis_taps(i, 48);
            assert_eq!(t[0], i);
            assert_eq!(w[0], 1.0);
        }
    }

    /// A 2 × 2 grid's four patches, each reading one corner of the table.
    #[test]
    fn position_taps_are_the_outer_product_of_two_axes() {
        let g = Grid {
            height: 32,
            width: 32,
        };
        let (idx, w) = position_taps(g);
        assert_eq!(idx.len(), 16);
        // Patch (0,0): row taps [0,1] w [1,0], col taps [0,1] w [1,0].
        assert_eq!(&idx[0..4], &[0, 1, 48, 49]);
        assert_eq!(&w[0..4], &[1.0, 0.0, 0.0, 0.0]);
        // Patch (0,1): column 1 of 2 → the last column, 47.
        assert_eq!(idx[4], 47);
        assert_eq!(w[4], 1.0);
        // Patch (1,1): the last corner.
        assert_eq!(idx[12], 47 * 48 + 47);
        assert_eq!(w[12], 1.0);
        for p in 0..4 {
            let s: f32 = w[p * 4..p * 4 + 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "patch {p} weights sum to {s}");
        }
    }

    /// The first patch turns by nothing; patch (1,0) — the third in block
    /// order — turns its row half by the frequencies and its column half not.
    #[test]
    fn the_vit_rope_turns_rows_then_columns() {
        let g = Grid {
            height: 32,
            width: 32,
        };
        let a = vit_rope_angles(g);
        assert_eq!(a.len(), 4 * 36);
        assert!(a[0..36].iter().all(|v| *v == 0.0));
        let third = &a[2 * 36..3 * 36];
        assert_eq!(third[0], 1.0);
        assert!((third[1] - (1.0 / 10_000f64.powf(2.0 / 36.0)) as f32).abs() < 1e-7);
        assert!(third[18..36].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn the_trunk_pairs_take_the_interleaved_axes() {
        let axes: Vec<usize> = (0..32).map(trunk_axis).collect();
        assert_eq!(axes.iter().filter(|a| **a == 0).count(), 11);
        assert_eq!(axes.iter().filter(|a| **a == 1).count(), 11);
        assert_eq!(axes.iter().filter(|a| **a == 2).count(), 10);
        assert_eq!(&axes[0..6], &[0, 1, 2, 0, 1, 2]);
        assert_eq!(axes[31], 1);
    }

    /// A 3 × 5 merged image (n 15, m 5): each row's slot plus its δ is its
    /// anchored position, `(e − m) + (0, r, c)` with the slot `e − n + i`.
    #[test]
    fn image_deltas_anchor_the_image_at_its_end() {
        let g = Grid {
            height: 3 * FACTOR,
            width: 5 * FACTOR,
        };
        assert_eq!((g.tokens(), g.advance()), (15, 5));
        let e: i64 = 100;
        for i in 0..15usize {
            let d = image_delta(g, i);
            let slot = e - 15 + i as i64;
            let (r, c) = (i as i64 / 5, i as i64 % 5);
            assert_eq!(slot + d[0] as i64, e - 5, "t of row {i}");
            assert_eq!(slot + d[1] as i64, e - 5 + r, "h of row {i}");
            assert_eq!(slot + d[2] as i64, e - 5 + c, "w of row {i}");
        }
        assert_eq!(image_delta(g, 0), [10, 10, 10]);
        assert_eq!(image_delta(g, 14), [-4, -2, 0]);
    }
}
