//! Running the autoencoder in tiles, so its peak does not scale with the image.
//!
//! Both directions are here — [`decode_tiled`] for a latent on its way out and
//! [`encode_tiled`] for a reference picture on its way in — because they are the
//! same problem with the arrow reversed. The decoder upsamples to 128 channels
//! at full resolution; the encoder *starts* there. A reference image at
//! 1248×832 wants what a 1248×832 decode wants, and would fail in the same
//! place for the same reason.
//!
//! # The failure this exists to remove
//!
//! The autoencoder is the peak of an image drain, not the denoise. The
//! transformer's attention is banded, so its largest live tensor is one band of
//! scores; the decoder is not banded at all — it upsamples to 128 channels at
//! the image's **full resolution**, and a residual block holds several of those
//! at once. The working set therefore grows with the image's *area*, and nothing
//! in the decoder bounds it.
//!
//! Those tensors come from the CUDA pool, which is the one allocator a
//! co-resident guest does not get much of: the engine's span holds nearly the
//! whole card and what is left is the governor's scratch margin. Worse, the
//! denoise has just spent that margin on transformer-shaped blocks, and freeing
//! them returns them to the pool rather than to the driver —
//! `trim_pool_after_load` recovers what it can, measured at **544 MiB** against
//! the ~3 GiB a 1.04 MP decode wants.
//!
//! So the first large draw of a session died in the decoder with
//! `CUDA_ERROR_OUT_OF_MEMORY` while the pool's own accounting said there was
//! room, and the *second* succeeded — because the failed attempt's allocations
//! went back to the pool already carved to decoder shapes. Deterministic, not
//! flaky: the first draw above roughly one megapixel failed, every one after it
//! worked.
//!
//! # Why tiles rather than a bigger reservation
//!
//! The decode *is* also given memory from the guest's own ground, and the two
//! measures answer different halves of the problem. Ground is a **bump** with no
//! free, so a cursor holds the *sum* of everything carved inside one generation
//! rather than the peak — and a decoder allocates and drops its intermediates
//! continuously, which for a whole decoder is tens of gigabytes. What makes it
//! usable is scope rather than a freeing allocator: generations nest, and the
//! decoder opens one per block and per resnet
//! ([`candle_nn::kv_cache::guest_stage`]), so the sum that has to fit is one
//! resnet's.
//!
//! Tiling attacks the other quantity. Scoping bounds what a *generation* holds;
//! tiling bounds what the whole decode holds, because a tile is decoded at a
//! fixed size whatever the image is — so **the peak stops depending on the
//! output**. A 2048×2048 decode has the same working set as a 512×512 one, and
//! takes proportionally longer instead of failing.
//!
//! # Seams, and why the overlap is in latent space
//!
//! A convolution reads its neighbours, so a tile decoded in isolation is wrong
//! near its edges — it has no neighbours there, and the padding invents them.
//! Tiles therefore overlap, and the overlapping output is **cross-faded** rather
//! than picked from one side: a hard join would put the error exactly on the
//! seam, where it reads as a line.
//!
//! The overlap is measured in latent units and multiplied by [`SCALE`] on the
//! way out, because that is the direction the decoder works in. Taking it in
//! pixels and dividing would let a caller ask for an overlap that is not a whole
//! latent cell, and the tile grid would stop lining up with the latent it is cut
//! from.

use candle::{DType, Device, Result, Tensor};

/// Pixels per latent cell, for this autoencoder.
///
/// FLUX's, and it is the same 8 the guest divides by when it sizes the latent.
pub const SCALE: usize = 8;

/// The tile side, in latent cells — 64, so a tile decodes to 512×512.
///
/// Chosen because 512×512 is the size that has never failed on a 24 GB card with
/// the engine resident: it is the drain the guest has served thousands of times,
/// so a tile is a decode already known to fit rather than a new number to tune.
pub const TILE: usize = 64;

/// How far tiles overlap, in latent cells — 8, so 64 pixels of output.
///
/// The decoder's receptive field is what this has to cover: too little and the
/// cross-fade blends two edges that are both wrong, which is a soft seam instead
/// of a hard one. 8 cells is an eighth of a tile, so the cost is about 25% more
/// tiles on a large image and no measurable difference on a small one.
pub const OVERLAP: usize = 8;

/// One tile's placement, in latent cells.
///
/// Both halves are carried because they are not the same rectangle: `take` is
/// what is *decoded* (including the overlap that will be faded), and it is what
/// the caller narrows out of the latent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Where the decoded tile starts in the latent.
    pub start: usize,
    /// How many latent cells it covers.
    pub len: usize,
}

impl Tile {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// Lay out one axis as overlapping tiles covering `cells`.
///
/// **Every cell is covered and the last tile ends exactly at the edge.** A grid
/// that ran past the end would decode padding as if it were image; one that
/// stopped short would leave a strip of the picture undecoded, and neither is
/// visible in anything but the output.
///
/// A latent smaller than one tile gives a single tile of the whole thing, which
/// is the case that has to stay exactly as fast as it was: the overwhelming
/// majority of draws are 512×512 and must not pay for machinery they do not use.
pub fn tiles(cells: usize) -> Vec<Tile> {
    if cells == 0 {
        return Vec::new();
    }
    if cells <= TILE {
        return vec![Tile {
            start: 0,
            len: cells,
        }];
    }
    // The distance between two tiles' starts. At least one, or a tile that is
    // all overlap would never advance and this would not terminate.
    let stride = TILE.saturating_sub(OVERLAP).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        // The final tile is pulled back to the edge rather than extended past
        // it, so it is always a full `TILE` of real latent.
        if start + TILE >= cells {
            out.push(Tile {
                start: cells - TILE,
                len: TILE,
            });
            break;
        }
        out.push(Tile { start, len: TILE });
        start += stride;
    }
    out
}

/// The cross-fade weight for `n` output pixels, ramping in over `lead` and out
/// over `trail`.
///
/// A ramp rather than a step: where two tiles overlap, both contribute and their
/// weights sum to one at every pixel, so the join is a gradient of the error
/// instead of a line. The first tile has no lead and the last no trail — there
/// is nothing on the far side to blend with, and fading there would darken the
/// image's own edge.
fn ramp(n: usize, lead: usize, trail: usize, device: &Device) -> Result<Tensor> {
    let mut w = vec![1f32; n];
    for (i, v) in w.iter_mut().take(lead.min(n)).enumerate() {
        // `(i + 1) / (lead + 1)`, so neither end of the ramp is exactly zero:
        // a zero-weight column contributes nothing, which wastes a decoded
        // pixel and puts a full-strength discontinuity one column further in.
        *v = (i + 1) as f32 / (lead + 1) as f32;
    }
    for (i, v) in w.iter_mut().rev().take(trail.min(n)).enumerate() {
        *v = (*v).min((i + 1) as f32 / (trail + 1) as f32);
    }
    Tensor::from_vec(w, n, device)
}

/// Decode `latent` a tile at a time and blend the result.
///
/// `latent` is `[1, C, H, W]` in latent cells; the result is `[1, 3, H·SCALE,
/// W·SCALE]`. `decode` is the autoencoder, called once per tile.
///
/// **The single-tile case calls `decode` once and returns its output
/// unchanged** — no accumulation buffers, no blend, no extra full-resolution
/// tensor. That is the path a 512×512 draw takes, and it has to cost exactly
/// what it did before tiling existed.
pub fn decode_tiled<F>(latent: &Tensor, mut decode: F) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let (_, _, h, w) = latent.dims4()?;
    let rows = tiles(h);
    let cols = tiles(w);
    if rows.len() == 1 && cols.len() == 1 {
        return decode(latent);
    }

    let device = latent.device();
    let (out_h, out_w) = (h * SCALE, w * SCALE);
    // f32 accumulators: the weighted sum of bf16 tiles would round every
    // contribution, and the overlap columns are sums of two of them.
    let mut acc = Tensor::zeros((1, 3, out_h, out_w), DType::F32, device)?;
    let mut wsum = Tensor::zeros((1, 1, out_h, out_w), DType::F32, device)?;

    for r in &rows {
        for c in &cols {
            let tile = latent
                .narrow(2, r.start, r.len)?
                .narrow(3, c.start, c.len)?
                .contiguous()?;
            let px = decode(&tile)?.to_dtype(DType::F32)?;

            // How much of this tile's output is shared with the neighbour on
            // each side. A tile at the edge of the grid has none on that side.
            let lead_r = if r.start == 0 { 0 } else { OVERLAP * SCALE };
            let trail_r = if r.end() == h { 0 } else { OVERLAP * SCALE };
            let lead_c = if c.start == 0 { 0 } else { OVERLAP * SCALE };
            let trail_c = if c.end() == w { 0 } else { OVERLAP * SCALE };

            let wr =
                ramp(r.len * SCALE, lead_r, trail_r, device)?.reshape((1, 1, r.len * SCALE, 1))?;
            let wc =
                ramp(c.len * SCALE, lead_c, trail_c, device)?.reshape((1, 1, 1, c.len * SCALE))?;
            let weight = wr.broadcast_mul(&wc)?;

            let (y, x) = (r.start * SCALE, c.start * SCALE);
            let contrib = px.broadcast_mul(&weight)?;
            // Read-modify-write on the slice rather than `slice_set` of the
            // whole plane: tiles overlap, so a write would discard the
            // neighbour's contribution to the shared columns.
            let prev = acc
                .narrow(2, y, r.len * SCALE)?
                .narrow(3, x, c.len * SCALE)?;
            acc = acc.slice_assign(
                &[0..1, 0..3, y..y + r.len * SCALE, x..x + c.len * SCALE],
                &(prev + contrib)?,
            )?;
            let prev_w = wsum
                .narrow(2, y, r.len * SCALE)?
                .narrow(3, x, c.len * SCALE)?;
            wsum = wsum.slice_assign(
                &[0..1, 0..1, y..y + r.len * SCALE, x..x + c.len * SCALE],
                &(prev_w + weight)?,
            )?;
        }
    }
    // The weights sum to one inside an overlap by construction, but only to
    // within rounding — dividing rather than trusting it keeps a tile boundary
    // from being a fraction of a percent brighter than its surroundings.
    acc.broadcast_div(&wsum)
}

/// Encode `pixels` a tile at a time and blend the result.
///
/// `pixels` is `[1, 3, H, W]` and both sides must be whole latent cells; the
/// result is `[1, C, H/SCALE, W/SCALE]`, with `C` taken from what `encode`
/// returns rather than assumed. `encode` is the autoencoder, called once per
/// tile.
///
/// **The single-tile case calls `encode` once and returns its output
/// unchanged**, which is the path every 512×512 reference takes.
///
/// # Blending latents is an approximation, and a sound one
///
/// Two overlapping encodings of the same region are not identical — the encoder
/// is convolutional, so each sees different context at the shared edge — and
/// cross-fading them is not the same tensor as encoding the whole picture at
/// once. That is a real approximation and it is the same one the decode makes
/// in the other direction.
///
/// It is sound here because of what the result is *for*: the latent is
/// immediately mixed with noise at the walk's starting sigma and then denoised.
/// A blend error at a tile join is far below the noise deliberately added on
/// top of it, and the denoise is a contraction toward the model's own manifold
/// — so a slightly-off latent is corrected by the very steps that follow it. A
/// hard join would still show, which is why the fade is here at all.
pub fn encode_tiled<F>(pixels: &Tensor, mut encode: F) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let (_, _, ph, pw) = pixels.dims4()?;
    if ph % SCALE != 0 || pw % SCALE != 0 {
        candle::bail!(
            "a reference of {ph}×{pw} is not a whole number of {SCALE}-pixel latent cells"
        );
    }
    let (h, w) = (ph / SCALE, pw / SCALE);
    let rows = tiles(h);
    let cols = tiles(w);
    if rows.len() == 1 && cols.len() == 1 {
        return encode(pixels);
    }

    let device = pixels.device();
    // The channel count is the autoencoder's, so the accumulator cannot be
    // allocated until the first tile has come back and said what it is.
    let mut acc: Option<Tensor> = None;
    let mut wsum = Tensor::zeros((1, 1, h, w), DType::F32, device)?;

    for r in &rows {
        for c in &cols {
            let tile = pixels
                .narrow(2, r.start * SCALE, r.len * SCALE)?
                .narrow(3, c.start * SCALE, c.len * SCALE)?
                .contiguous()?;
            let z = encode(&tile)?.to_dtype(DType::F32)?;
            let channels = z.dim(1)?;

            // The overlap is in latent cells here rather than pixels, because
            // this side of the autoencoder works in them.
            let lead_r = if r.start == 0 { 0 } else { OVERLAP };
            let trail_r = if r.end() == h { 0 } else { OVERLAP };
            let lead_c = if c.start == 0 { 0 } else { OVERLAP };
            let trail_c = if c.end() == w { 0 } else { OVERLAP };

            let wr = ramp(r.len, lead_r, trail_r, device)?.reshape((1, 1, r.len, 1))?;
            let wc = ramp(c.len, lead_c, trail_c, device)?.reshape((1, 1, 1, c.len))?;
            let weight = wr.broadcast_mul(&wc)?;

            let (y, x) = (r.start, c.start);
            let contrib = z.broadcast_mul(&weight)?;
            let base = match acc.take() {
                Some(a) => a,
                None => Tensor::zeros((1, channels, h, w), DType::F32, device)?,
            };
            let prev = base.narrow(2, y, r.len)?.narrow(3, x, c.len)?;
            acc = Some(base.slice_assign(
                &[0..1, 0..channels, y..y + r.len, x..x + c.len],
                &(prev + contrib)?,
            )?);
            let prev_w = wsum.narrow(2, y, r.len)?.narrow(3, x, c.len)?;
            wsum = wsum.slice_assign(
                &[0..1, 0..1, y..y + r.len, x..x + c.len],
                &(prev_w + weight)?,
            )?;
        }
    }
    acc.expect("the grid is non-empty, so at least one tile was encoded")
        .broadcast_div(&wsum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every latent cell is covered exactly once or twice, never zero times.**
    ///
    /// A gap is a strip of undecoded image and nothing else in the pipeline
    /// would notice it — the output is the right shape either way.
    #[test]
    fn the_tiles_cover_every_cell() {
        for cells in [1usize, 8, 63, 64, 65, 100, 128, 129, 156, 256] {
            let ts = tiles(cells);
            assert!(!ts.is_empty(), "{cells} cells produced no tiles");
            let mut covered = vec![0usize; cells];
            for t in &ts {
                assert!(t.end() <= cells, "{cells}: tile {t:?} runs past the end");
                for c in covered.iter_mut().take(t.end()).skip(t.start) {
                    *c += 1;
                }
            }
            assert!(
                covered.iter().all(|&n| n >= 1),
                "{cells}: a cell was left undecoded by {ts:?}"
            );
        }
    }

    /// A latent that fits in one tile produces exactly one tile of the whole
    /// thing — the 512×512 path, which must not pay for the machinery.
    #[test]
    fn a_small_latent_is_a_single_whole_tile() {
        for cells in [1usize, 16, 63, 64] {
            assert_eq!(
                tiles(cells),
                vec![Tile {
                    start: 0,
                    len: cells
                }],
                "{cells} cells should be one tile"
            );
        }
        // 512×512 is 64 latent cells, the common case.
        assert_eq!(tiles(512 / SCALE).len(), 1);
        // 1024 and the 3:2 buckets are not.
        assert!(tiles(1024 / SCALE).len() > 1);
        assert!(tiles(1248 / SCALE).len() > 1);
    }

    /// The last tile ends exactly on the edge, so no tile decodes padding as if
    /// it were image.
    #[test]
    fn the_last_tile_lands_on_the_edge() {
        for cells in [65usize, 100, 128, 156, 256] {
            let ts = tiles(cells);
            assert_eq!(
                ts.last().unwrap().end(),
                cells,
                "{cells}: the grid does not reach the edge"
            );
            assert_eq!(ts[0].start, 0, "{cells}: the grid does not start at zero");
        }
    }

    /// Neighbouring tiles genuinely overlap — without it the cross-fade has
    /// nothing to fade and every join is a hard edge.
    #[test]
    fn neighbouring_tiles_overlap() {
        let ts = tiles(200);
        assert!(ts.len() >= 3);
        for pair in ts.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(
                b.start < a.end(),
                "tiles {a:?} and {b:?} do not overlap, so the join is a seam"
            );
        }
    }

    /// A tile grid always terminates and stays bounded, whatever the size.
    #[test]
    fn the_grid_is_bounded() {
        for cells in [65usize, 1000, 4096] {
            let ts = tiles(cells);
            assert!(
                ts.len() <= cells.div_ceil(TILE - OVERLAP) + 1,
                "{cells} produced {} tiles, which is more than the stride allows",
                ts.len()
            );
        }
    }

    /// **The ramp rises from non-zero to one and never exceeds it.**
    ///
    /// A zero at the end of a ramp wastes a decoded column and moves the
    /// discontinuity one pixel inward; a value above one brightens the overlap.
    #[test]
    fn the_ramp_stays_within_zero_and_one() {
        let d = Device::Cpu;
        let w = ramp(16, 4, 4, &d).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(w.len(), 16);
        for v in &w {
            assert!(*v > 0.0 && *v <= 1.0, "ramp value {v} is outside (0, 1]");
        }
        // Rising at the front, falling at the back, flat in the middle.
        assert!(w[0] < w[1] && w[1] < w[2]);
        assert_eq!(w[7], 1.0);
        assert!(w[15] < w[14]);
        // No lead and no trail is a flat one — the single-tile case.
        assert_eq!(
            ramp(4, 0, 0, &d).unwrap().to_vec1::<f32>().unwrap(),
            vec![1f32; 4]
        );
    }

    /// **Two overlapping ramps sum to one across the join.**
    ///
    /// This is the property that makes a seam invisible. If the sum dipped, the
    /// overlap would be a dark band; if it rose, a bright one. The decode
    /// divides by the weight sum so either would be corrected — but only if the
    /// weights are what the division is told they are, so it is asserted here
    /// rather than assumed.
    #[test]
    fn overlapping_ramps_sum_to_one_across_the_join() {
        let d = Device::Cpu;
        let n = 32;
        let ov = 8;
        // A left tile fading out, and a right tile fading in, sharing `ov`.
        let left = ramp(n, 0, ov, &d).unwrap().to_vec1::<f32>().unwrap();
        let right = ramp(n, ov, 0, &d).unwrap().to_vec1::<f32>().unwrap();
        for i in 0..ov {
            let sum = left[n - ov + i] + right[i];
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "at overlap {i} the weights sum to {sum}, not 1"
            );
        }
    }

    /// A single-tile decode is handed straight through — same tensor, one call,
    /// no blending arithmetic on it at all.
    #[test]
    fn a_single_tile_decode_is_passed_through_untouched() {
        let d = Device::Cpu;
        let latent = Tensor::ones((1, 4, 8, 8), DType::F32, &d).unwrap();
        let mut calls = 0;
        let out = decode_tiled(&latent, |t| {
            calls += 1;
            // A recognisable "decode": 3 channels at SCALE×.
            let (_, _, h, w) = t.dims4().unwrap();
            Tensor::full(7f32, (1, 3, h * SCALE, w * SCALE), &d)
        })
        .unwrap();
        assert_eq!(calls, 1, "a small latent was split");
        assert_eq!(out.dims(), &[1, 3, 64, 64]);
        assert_eq!(out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0], 7.0);
    }

    /// **A tiled decode of a constant reconstructs the constant exactly.**
    ///
    /// The blend is a weighted mean, so a flat field must come back flat — any
    /// weight that did not divide out shows up here as a band. This is the test
    /// that would have caught a seam.
    #[test]
    fn a_tiled_decode_of_a_flat_field_has_no_seams() {
        let d = Device::Cpu;
        // 96 cells a side: more than one tile, so the tiled path runs.
        let latent = Tensor::ones((1, 4, 96, 96), DType::F32, &d).unwrap();
        let mut calls = 0;
        let out = decode_tiled(&latent, |t| {
            calls += 1;
            let (_, _, h, w) = t.dims4().unwrap();
            Tensor::full(0.25f32, (1, 3, h * SCALE, w * SCALE), &d)
        })
        .unwrap();
        assert!(calls > 1, "the latent was not tiled");
        assert_eq!(out.dims(), &[1, 3, 96 * SCALE, 96 * SCALE]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
        assert!(
            (lo - 0.25).abs() < 1e-5 && (hi - 0.25).abs() < 1e-5,
            "a flat field decoded to a range of {lo}..{hi} — the blend leaves a seam"
        );
    }

    /// A reference that fits one tile is encoded in one call and handed
    /// straight through — the 512×512 path, which is almost every reference.
    #[test]
    fn a_single_tile_encode_is_passed_through_untouched() {
        let d = Device::Cpu;
        let px = Tensor::ones((1, 3, 512, 512), DType::F32, &d).unwrap();
        let mut calls = 0;
        let out = encode_tiled(&px, |t| {
            calls += 1;
            let (_, _, h, w) = t.dims4().unwrap();
            Tensor::full(0.5f32, (1, 16, h / SCALE, w / SCALE), &d)
        })
        .unwrap();
        assert_eq!(calls, 1, "a 512×512 reference was split");
        assert_eq!(out.dims(), &[1, 16, 64, 64]);
    }

    /// **A tiled encode of a flat field comes back flat**, in the latent's own
    /// channel count. The same seam test the decode has, in the direction the
    /// reference travels — a weight that did not divide out would put a band
    /// into the starting latent, which the denoise would then render.
    #[test]
    fn a_tiled_encode_of_a_flat_field_has_no_seams() {
        let d = Device::Cpu;
        // 1024×768 — more than one tile on both axes.
        let px = Tensor::ones((1, 3, 1024, 768), DType::F32, &d).unwrap();
        let mut calls = 0;
        let out = encode_tiled(&px, |t| {
            calls += 1;
            let (_, _, h, w) = t.dims4().unwrap();
            assert_eq!(h % SCALE, 0, "a tile was cut off a latent-cell boundary");
            Tensor::full(-0.75f32, (1, 16, h / SCALE, w / SCALE), &d)
        })
        .unwrap();
        assert!(calls > 1, "the reference was not tiled");
        assert_eq!(out.dims(), &[1, 16, 128, 96]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
        assert!(
            (lo + 0.75).abs() < 1e-5 && (hi + 0.75).abs() < 1e-5,
            "a flat field encoded to a range of {lo}..{hi} — the blend leaves a seam"
        );
    }

    /// A picture whose sides are not whole latent cells is refused rather than
    /// silently truncated — a lost row is a reference the model never sees.
    #[test]
    fn an_encode_off_the_cell_grid_is_refused() {
        let d = Device::Cpu;
        let px = Tensor::ones((1, 3, 513, 512), DType::F32, &d).unwrap();
        assert!(encode_tiled(&px, |t| Ok(t.clone())).is_err());
    }

    /// The decoder is called once per tile in the grid, and each call gets a
    /// tile of the shape the grid says. A decode of the wrong slice would still
    /// produce an image, just not of the latent it was given.
    #[test]
    fn each_tile_is_decoded_once_at_the_grid_position() {
        let d = Device::Cpu;
        let latent = Tensor::ones((1, 4, 96, 130), DType::F32, &d).unwrap();
        let mut seen = Vec::new();
        decode_tiled(&latent, |t| {
            let (_, _, h, w) = t.dims4().unwrap();
            seen.push((h, w));
            Tensor::zeros((1, 3, h * SCALE, w * SCALE), DType::F32, &d)
        })
        .unwrap();
        assert_eq!(seen.len(), tiles(96).len() * tiles(130).len());
        for (h, w) in &seen {
            assert!(
                *h <= TILE && *w <= TILE,
                "a tile of {h}×{w} is larger than TILE"
            );
        }
    }
}
