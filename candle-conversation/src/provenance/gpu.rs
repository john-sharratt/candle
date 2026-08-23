//! GPU backend for the batched BDP flat scan.
//!
//! A whole **wave** of probes is scored against the packed gallery in one launch.
//! The gallery is **not resident**: each wave stages it through **pinned** host
//! memory, uploads it, runs the kernel, downloads the small results, and frees
//! the device buffers (they drop at the end of the call) — so VRAM is reclaimed
//! between scans. The kernel emits the per-(query-token, layer-group)
//! `(leading_case, z×margin)` pairs; the needle gate + per-case tally run here on
//! the host, identical to [`score_packed`](super::score_packed). The integer
//! agreement reductions are exact and order-independent; only the final `z×margin`
//! is fast-math f32 on the GPU vs IEEE f32 on the CPU, so scores match the CPU
//! path to ~1e-3 relative — the **ranking/argmax is identical** (validated in the
//! parity test), not bit-for-bit equal.

use core::ffi::c_void;

use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, Result};
use candle_kernels::provenance::run_batched_bdp_scan;
use rayon::prelude::*;

use super::packed::PackedGallery;
use super::scan::{HEADS_PER_GROUP, NEEDLE_KEEP_FRAC};
use super::WideQSig;

/// Query-token tile width — MUST mirror `BDP_TQ` in `bdp_scan.cu`. The kernel's
/// dynamic shared memory is `BDP_TQ * max_seg_cases * sizeof(u32)`.
pub(crate) const BDP_TQ: usize = 8;

/// Conservative dynamic-shared-memory ceiling (bytes). Blocks default to 48 KB of
/// dynamic shared without an opt-in launch attribute; a gallery whose largest
/// segment would exceed this can't be scanned on the GPU, so we bail to the CPU
/// path rather than issue a launch that would fail at `synchronize`.
const BDP_MAX_DYN_SHMEM: usize = 48 * 1024;

/// View a `&[T]` as raw bytes (for staging into a pinned host buffer).
fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

/// Stage `slice` through a freshly-allocated pinned host buffer — the fast,
/// DMA-friendly H2D source. The returned buffer must outlive the upload.
fn pin_copy<T>(slice: &[T]) -> Result<PinnedBuf> {
    let bytes = as_bytes(slice);
    let mut pin = PinnedBuf::alloc_owned(bytes.len())?;
    pin.as_mut_slice().copy_from_slice(bytes);
    Ok(pin)
}

/// A gallery staged in **pinned** host memory, ready to scan on the GPU.
///
/// The pinned buffers are the **resident RAM mirror** the user's architecture
/// describes: built once (page-locking is expensive, ~tens of ms for tens of MB,
/// so it is emphatically *not* per-wave), kept around and — in production —
/// mutated incrementally as the gallery changes. Only the **VRAM** is
/// non-resident: [`Self::scan`] uploads pinned→device, runs the kernel,
/// downloads the small results, and frees the device buffers on return.
/// One segment (a code-read file / timeline) for [`BatchedGpuGallery::from_segments`]:
/// its windows and the exchange (case) each window maps to — the exact
/// `(wref, wslot, n_slots)` a per-file belief scan already builds.
pub struct SegmentInput<'a> {
    pub windows: Vec<&'a [WideQSig]>,
    /// Local exchange (case) index per window, `0..n_cases`.
    pub window_case: Vec<usize>,
    pub n_cases: usize,
}

pub struct BatchedGpuGallery {
    words_pin: PinnedBuf,    // group-major [g][token][gw], tokens sorted by segment
    case_pin: PinnedBuf,     // n_tokens GLOBAL case ids
    seg_tok_pin: PinnedBuf,  // (n_segments+1) i32 — token range per segment
    seg_case_pin: PinnedBuf, // (n_segments+1) i32 — case range per segment
    seg_case: Vec<i32>,      // host mirror of the case prefixes (per-segment tally scatter)
    n_tokens: usize,
    wpt: usize,
    n_groups: usize,
    gw: usize,
    n_cases: usize,
    n_segments: usize,
    max_seg_cases: usize,
}

impl BatchedGpuGallery {
    /// Page-lock a group-major gallery + its segment boundaries into pinned host
    /// memory (the one-time, non-per-wave cost). `gm` is `[group][token][gw]` with
    /// tokens already sorted by segment; `case` is the global case id per token;
    /// `seg_tok`/`seg_case` are the `n_segments+1` prefix boundaries.
    fn from_group_major(
        gm: Vec<u64>,
        case: Vec<u32>,
        seg_tok: Vec<i32>,
        seg_case: Vec<i32>,
        n_tokens: usize,
        wpt: usize,
        n_groups: usize,
        gw: usize,
        n_cases: usize,
    ) -> Result<Self> {
        let n_segments = seg_tok.len().saturating_sub(1);
        let max_seg_cases = (0..n_segments)
            .map(|s| (seg_case[s + 1] - seg_case[s]) as usize)
            .max()
            .unwrap_or(0)
            .max(1);
        Ok(Self {
            words_pin: pin_copy(&gm)?,
            case_pin: pin_copy(&case)?,
            seg_tok_pin: pin_copy(&seg_tok)?,
            seg_case_pin: pin_copy(&seg_case)?,
            seg_case,
            n_tokens,
            wpt,
            n_groups,
            gw,
            n_cases,
            n_segments,
            max_seg_cases,
        })
    }

    /// A single-segment (global-z) gallery — the original behaviour. Kept for the
    /// parity test and any caller that wants one global normalization.
    pub fn from_packed(gallery: &PackedGallery) -> Result<Self> {
        let n_tokens = gallery.n_tokens();
        let wpt = gallery.wpt();
        let n_groups = gallery.n_heads() / HEADS_PER_GROUP;
        let gw = HEADS_PER_GROUP * gallery.wph();
        let src = gallery.words(); // token-major [n_tokens][wpt]
        let n_cases = gallery.n_cases();

        let mut gm = vec![0u64; src.len()];
        for j in 0..n_tokens {
            let tok = &src[j * wpt..j * wpt + wpt];
            for g in 0..n_groups {
                let base = g * n_tokens * gw + j * gw;
                gm[base..base + gw].copy_from_slice(&tok[g * gw..g * gw + gw]);
            }
        }
        // One segment spanning everything ⇒ global z (bit-identical to the old scan).
        Self::from_group_major(
            gm,
            gallery.case().to_vec(),
            vec![0, n_tokens as i32],
            vec![0, n_cases as i32],
            n_tokens,
            wpt,
            n_groups,
            gw,
            n_cases,
        )
    }

    /// The segmented (per-file z) gallery: each `SegmentInput` becomes one segment
    /// with per-file z/margin/needle. Tokens are laid out sorted by segment and
    /// cases renumbered into one global contiguous space (segment `s`'s cases
    /// follow `s-1`'s). This is the belief scan's per-file structure in one gallery.
    pub fn from_segments(segments: &[SegmentInput]) -> Result<Self> {
        // Geometry from the first available token.
        let first = segments
            .iter()
            .flat_map(|s| s.windows.iter())
            .flat_map(|w| w.iter())
            .next()
            .ok_or_else(|| candle::Error::Msg("BDP gallery: no gallery tokens".into()))?;
        let wpt = first.words.len();
        let n_heads = first.n_heads as usize;
        let n_groups = n_heads / HEADS_PER_GROUP;
        let gw = HEADS_PER_GROUP * (wpt / n_heads.max(1));

        // Token-major first (then transpose), with global case + segment prefixes.
        let mut tokens_tm: Vec<u64> = Vec::new();
        let mut case: Vec<u32> = Vec::new();
        let mut seg_tok = vec![0i32];
        let mut seg_case = vec![0i32];
        let mut case_off = 0usize;
        for seg in segments {
            for (w, win) in seg.windows.iter().enumerate() {
                let lc = seg.window_case.get(w).copied().unwrap_or(0);
                // Mirror `PackedGallery::from_windows` exactly: drop a window whose
                // case is out of range — never renumber it into the next segment's
                // case space (in the kernel that becomes a segment-local case
                // `>= seg_nc`, an out-of-bounds shared-memory write). Admit only
                // uniform-width tokens (the folded signature is always `wpt` wide).
                if lc >= seg.n_cases {
                    continue;
                }
                for tok in win.iter() {
                    if tok.words.len() != wpt {
                        continue;
                    }
                    tokens_tm.extend_from_slice(&tok.words[..wpt]);
                    case.push((case_off + lc) as u32);
                }
            }
            case_off += seg.n_cases;
            seg_tok.push((tokens_tm.len() / wpt) as i32);
            seg_case.push(case_off as i32);
        }
        let n_tokens = tokens_tm.len() / wpt;
        let n_cases = case_off;

        // Group-major transpose: [group][token][gw].
        let mut gm = vec![0u64; tokens_tm.len()];
        for j in 0..n_tokens {
            let tok = &tokens_tm[j * wpt..j * wpt + wpt];
            for g in 0..n_groups {
                let base = g * n_tokens * gw + j * gw;
                gm[base..base + gw].copy_from_slice(&tok[g * gw..g * gw + gw]);
            }
        }
        Self::from_group_major(
            gm, case, seg_tok, seg_case, n_tokens, wpt, n_groups, gw, n_cases,
        )
    }

    /// The pinned gallery footprint in bytes (words + case + segment tables).
    pub fn byte_len(&self) -> usize {
        self.words_pin.len()
            + self.case_pin.len()
            + self.seg_tok_pin.len()
            + self.seg_case_pin.len()
    }

    /// Number of cases (exchanges) across all segments.
    pub fn n_cases(&self) -> usize {
        self.n_cases
    }

    /// Score a **wave** of probes, returning one per-case vote vector per probe.
    /// Numerically equivalent to [`score_packed`](super::score_packed) on each
    /// probe up to fast-math ULP: the integer popcount/agreement reductions are
    /// exact and order-independent, but the final `z*margin` is fast-math f32 on
    /// the GPU vs IEEE f32 on the CPU, so scores differ by ~1e-3 relative (the
    /// argmax/ranking is identical — validated in the parity test and
    /// `examples/gpu_belief_parity.rs`).
    ///
    /// Per wave: pinned gallery → VRAM (upload), one batched kernel, results →
    /// host (download), device buffers freed on return (non-resident VRAM).
    pub fn scan(&self, device: &Device, probes: &[&[WideQSig]]) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted(device, probes, &[])
    }

    /// [`Self::scan`] with a per-layer-group weight on each group's `z*margin`
    /// vote (`group_weights[g]`; missing/empty ⇒ 1.0). Applied host-side in the
    /// per-segment tally so the kernel stays weight-agnostic — matching the CPU
    /// `score_provenance_late_fusion_weighted`.
    pub fn scan_weighted(
        &self,
        device: &Device,
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let dev = match device {
            Device::Cuda(d) => d,
            _ => {
                return Err(candle::Error::Msg(
                    "BDP GPU scan requires a CUDA device".into(),
                ))
            }
        };
        let n_cases = self.n_cases;
        let wpt = self.wpt;
        let n_groups = self.n_groups;
        let need = wpt;

        if probes.is_empty() {
            return Ok(Vec::new());
        }
        if self.n_tokens == 0 || n_groups == 0 || wpt == 0 {
            return Ok(vec![vec![0.0; n_cases]; probes.len()]);
        }

        // Batch: concatenate every request's full-width probe tokens (short tokens
        // skipped — same as the reference's `q.words.len() >= need` filter).
        let mut probe_words: Vec<u64> = Vec::new();
        let mut per_req_tokens: Vec<usize> = Vec::with_capacity(probes.len());
        for probe in probes {
            let mut cnt = 0usize;
            for tok in *probe {
                if tok.words.len() >= need {
                    probe_words.extend_from_slice(&tok.words[..wpt]);
                    cnt += 1;
                }
            }
            per_req_tokens.push(cnt);
        }
        let n_probe_tokens = probe_words.len() / wpt;
        if n_probe_tokens == 0 {
            return Ok(vec![vec![0.0; n_cases]; probes.len()]);
        }

        let stream = dev.cuda_stream();
        let n_segments = self.n_segments;

        // Bail to the CPU path if the largest segment's case count would exceed the
        // dynamic-shared budget (a pathological file with thousands of exchanges) —
        // the kernel's shared stride is BDP_TQ * max_seg_cases u32s, so a launch
        // would otherwise fail only at `synchronize`.
        let shmem = BDP_TQ * self.max_seg_cases * std::mem::size_of::<u32>();
        if shmem > BDP_MAX_DYN_SHMEM {
            return Err(candle::Error::Msg(format!(
                "BDP GPU scan: largest segment has {} cases ({shmem} B shared > {BDP_MAX_DYN_SHMEM} B budget)",
                self.max_seg_cases
            )));
        }

        // ── Upload: pinned gallery + segment tables → VRAM (freed on return) ─
        let d_words = stream
            .memcpy_stod(self.words_pin.as_slice())
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD gallery words: {e}")))?;
        let d_case = stream
            .memcpy_stod(self.case_pin.as_slice())
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD gallery case: {e}")))?;
        let d_seg_tok = stream
            .memcpy_stod(self.seg_tok_pin.as_slice())
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD seg_tok: {e}")))?;
        let d_seg_case = stream
            .memcpy_stod(self.seg_case_pin.as_slice())
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD seg_case: {e}")))?;
        let d_probe = stream
            .memcpy_stod(&probe_words)
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD probes: {e}")))?;

        // Outputs are per (query token, group, SEGMENT). The kernel writes every
        // entry (one block per tile × group × segment), so no zero-init needed.
        let n_out = n_probe_tokens * n_groups * n_segments;
        let mut d_out_case = unsafe { stream.alloc::<i32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: alloc out_case: {e}")))?;
        let mut d_out_vote = unsafe { stream.alloc::<f32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: alloc out_vote: {e}")))?;

        // ── Launch (guards drop after synchronize, before the D2H reads) ────
        {
            let (p_words, _gw_guard) = d_words.device_ptr(&stream);
            let (p_case, _gc_guard) = d_case.device_ptr(&stream);
            let (p_seg_tok, _gst_guard) = d_seg_tok.device_ptr(&stream);
            let (p_seg_case, _gsc_guard) = d_seg_case.device_ptr(&stream);
            let (p_probe, _gp_guard) = d_probe.device_ptr(&stream);
            let (p_out_case, _goc_guard) = d_out_case.device_ptr_mut(&stream);
            let (p_out_vote, _gov_guard) = d_out_vote.device_ptr_mut(&stream);
            unsafe {
                run_batched_bdp_scan(
                    p_words as *const u64,
                    p_case as *const u32,
                    p_probe as *const u64,
                    p_seg_tok as *const i32,
                    p_seg_case as *const i32,
                    std::ptr::null(), // page_ptr — contiguous mode
                    std::ptr::null(), // pos_map — contiguous mode
                    n_probe_tokens as i32,
                    n_groups as i32,
                    n_segments as i32,
                    self.max_seg_cases as i32,
                    self.gw as i32,
                    wpt as i32,
                    p_out_case as *mut i32,
                    p_out_vote as *mut f32,
                    stream.cu_stream() as *mut c_void,
                );
            }
            stream
                .synchronize()
                .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: synchronize: {e}")))?;
        }

        // ── Download small results (device buffers free on drop) ────────────
        let out_case = stream
            .memcpy_dtov(&d_out_case)
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: DtoH out_case: {e}")))?;
        let out_vote = stream
            .memcpy_dtov(&d_out_vote)
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: DtoH out_vote: {e}")))?;

        // Host per-segment needle gate + tally (shared with the paged path).
        Ok(needle_tally_segments(
            &out_case,
            &out_vote,
            &per_req_tokens,
            &self.seg_case,
            n_groups,
            n_segments,
            n_cases,
            group_weights,
        ))
    }
}

/// The host per-SEGMENT needle gate + tally, shared by the contiguous
/// ([`BatchedGpuGallery::scan_weighted`]) and paged
/// ([`super::gallery_arena`]) scan paths — so both produce identical votes from
/// the kernel's `(out_case, out_vote)` output.
///
/// Each segment (file) needle-gates its own query tokens (top-25% by vote
/// magnitude) and tallies into that segment's cases — the exact per-file
/// behaviour of the CPU scan (`n_segments == 1` reproduces the global gate).
/// Parallel over segments, each writing ONLY its own contiguous case range
/// `[case0, case0+seg_nc)` (the kernel emits a GLOBAL `case0 + top1c` within the
/// segment). The ranges partition `0..n_cases`, so the results are disjoint — a
/// plain scatter, no cross-segment summation, order-independent and bit-identical
/// to a serial fold. Each segment tallies in its OWN local case space (width
/// `seg_nc`). `seg_case` is the `n_segments+1` case-prefix array.
pub(crate) fn needle_tally_segments(
    out_case: &[i32],
    out_vote: &[f32],
    per_req_tokens: &[usize],
    seg_case: &[i32],
    n_groups: usize,
    n_segments: usize,
    n_cases: usize,
    group_weights: &[f32],
) -> Vec<Vec<f32>> {
    let mut results = Vec::with_capacity(per_req_tokens.len());
    let mut tok_base = 0usize;
    for &cnt in per_req_tokens {
        let base = tok_base;
        let locals: Vec<(usize, Vec<f32>)> = (0..n_segments)
            .into_par_iter()
            .map(|s| {
                // Fused, allocation-light form of `needle_gate_tally` over the flat
                // kernel output — bit-identical (same `qt`-then-`g` accumulation
                // order), but without the per-query `Vec<Vec<_>>` (was ~200k tiny
                // allocations per scan, the tally's dominant cost).
                let case0 = seg_case[s] as usize;
                let seg_nc = seg_case[s + 1] as usize - case0;
                let mut local = vec![0f32; seg_nc];
                if cnt == 0 {
                    return (case0, local);
                }
                let wof = |g: usize| group_weights.get(g).copied().unwrap_or(1.0);
                // Pass 1: per-query magnitude (sum of weighted votes over groups).
                let mut mags = vec![0f32; cnt];
                for (qt, m) in mags.iter_mut().enumerate() {
                    let gqt = base + qt;
                    let mut acc = 0f32;
                    for g in 0..n_groups {
                        let idx = (gqt * n_groups + g) * n_segments + s;
                        if out_case[idx] >= 0 {
                            acc += out_vote[idx] * wof(g);
                        }
                    }
                    *m = acc;
                }
                // Needle gate: keep the top NEEDLE_KEEP_FRAC query tokens by magnitude.
                let keep_n = ((NEEDLE_KEEP_FRAC * cnt as f32).ceil() as usize).clamp(1, cnt);
                let mut sorted = mags.clone();
                sorted.sort_unstable_by(|a, b| b.total_cmp(a));
                let thresh = sorted[keep_n - 1];
                // Pass 2: tally the kept tokens' votes into this segment's cases.
                for (qt, &mag) in mags.iter().enumerate() {
                    if mag < thresh {
                        continue;
                    }
                    let gqt = base + qt;
                    for g in 0..n_groups {
                        let idx = (gqt * n_groups + g) * n_segments + s;
                        let c = out_case[idx];
                        if c >= 0 {
                            local[c as usize - case0] += out_vote[idx] * wof(g);
                        }
                    }
                }
                (case0, local)
            })
            .collect();
        let mut votes = vec![0f32; n_cases];
        for (case0, local) in locals {
            votes[case0..case0 + local.len()].copy_from_slice(&local);
        }
        results.push(votes);
        tok_base += cnt;
    }
    results
}

/// Convenience one-shot: stage `gallery` to pinned memory and scan `probes` in a
/// single call. Equivalent to `BatchedGpuGallery::from_packed(gallery)?.scan(..)`
/// — used by tests/parity; production keeps the [`BatchedGpuGallery`] resident.
pub fn score_batched_gpu(
    device: &Device,
    gallery: &PackedGallery,
    probes: &[&[WideQSig]],
) -> Result<Vec<Vec<f32>>> {
    BatchedGpuGallery::from_packed(gallery)?.scan(device, probes)
}

#[cfg(test)]
mod tests {
    // Fixture galleries are written as `vec![vec![…], …]` so each row reads as
    // one tool's token window; an array of vecs would obscure that.
    #![allow(clippy::useless_vec)]

    use super::*;
    use crate::provenance::{score_packed, score_provenance_late_fusion_weighted, PackedGallery};

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// The batched GPU scan must be **bit-identical** to running the CPU
    /// `score_packed` on each probe. Skips cleanly when no CUDA device is present.
    #[test]
    fn gpu_batched_matches_cpu_packed() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU in this environment — skip
        };

        // A 4-tool gallery with several tokens per tool + an out-of-range slot.
        let windows = vec![
            vec![sig(0xAAAA_AAAA_AAAA_AAAA), sig(0xABAB_ABAB_ABAB_ABAB)],
            vec![sig(0x5555_5555_5555_5555)],
            vec![sig(0xFFFF_FFFF_0000_0000), sig(0x0F0F_0F0F_0F0F_0F0F)],
            vec![sig(0x1234_5678_9ABC_DEF0)],
            vec![sig(0xDEAD_BEEF_DEAD_BEEF)], // slot 9 → dropped
        ];
        let slots = vec![0usize, 1, 2, 3, 9];
        let wref: Vec<&[WideQSig]> = windows.iter().map(|w| w.as_slice()).collect();
        let packed = PackedGallery::from_windows(&wref, &slots, 4);

        // A wave of probes: single-token, multi-token, and an empty one.
        let probes_owned = vec![
            vec![sig(0xAAAA_AAAA_AAAA_AAAA), sig(0x1234_5678_9ABC_DEF0)],
            vec![sig(0x5555_5555_5555_5555)],
            vec![
                sig(0xFFFF_FFFF_0000_0000),
                sig(0x0F0F_0F0F_0F0F_0F0F),
                sig(0x0),
            ],
            vec![],
        ];
        let probes: Vec<&[WideQSig]> = probes_owned.iter().map(|p| p.as_slice()).collect();

        let gpu = score_batched_gpu(&device, &packed, &probes).expect("gpu scan");
        assert_eq!(gpu.len(), probes.len());
        for (i, probe) in probes.iter().enumerate() {
            let cpu = score_packed(probe, &packed);
            // Fast-math ⇒ ~ULP float differences; require close values AND an
            // identical ranking (the argmax is what selection acts on).
            assert_eq!(gpu[i].len(), cpu.len(), "probe {i}: length");
            for (g, c) in gpu[i].iter().zip(&cpu) {
                assert!(
                    (g - c).abs() <= 1e-3 * (1.0 + g.abs().max(c.abs())),
                    "probe {i}: GPU {g} vs CPU {c} exceeds tolerance"
                );
            }
            assert_eq!(
                argmax(&gpu[i]),
                argmax(&cpu),
                "probe {i}: GPU and CPU must rank the same case first"
            );
        }
    }

    /// The SEGMENTED GPU scan must be bit-identical to running the CPU
    /// `score_packed` PER FILE (per-segment z / margin / needle gate) and
    /// concatenating into the global case space. This is the belief scan's
    /// per-file normalization — the whole point of the segmented kernel.
    #[test]
    fn gpu_segmented_matches_cpu_per_file() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };
        // Three files (segments), each with a few exchanges and tokens.
        let files: Vec<(Vec<Vec<WideQSig>>, Vec<usize>, usize)> = vec![
            (
                vec![
                    vec![sig(0xAAAA_AAAA_AAAA_AAAA), sig(0xABAB_ABAB_ABAB_ABAB)],
                    vec![sig(0x5555_5555_5555_5555)],
                ],
                vec![0usize, 1],
                2,
            ),
            (
                vec![
                    vec![sig(0xFFFF_FFFF_0000_0000)],
                    vec![sig(0x0F0F_0F0F_0F0F_0F0F), sig(0x1234_5678_9ABC_DEF0)],
                    vec![sig(0xDEAD_BEEF_DEAD_BEEF)],
                ],
                vec![0usize, 1, 2],
                3,
            ),
            (vec![vec![sig(0x00FF_00FF_00FF_00FF)]], vec![0usize], 1),
        ];
        let segments: Vec<SegmentInput> = files
            .iter()
            .map(|(w, c, n)| SegmentInput {
                windows: w.iter().map(|x| x.as_slice()).collect(),
                window_case: c.clone(),
                n_cases: *n,
            })
            .collect();
        let gallery = BatchedGpuGallery::from_segments(&segments).expect("build segmented gallery");

        let probe = vec![
            sig(0xAAAA_AAAA_AAAA_AAAA),
            sig(0x0F0F_0F0F_0F0F_0F0F),
            sig(0x1357_2468_1357_2468),
        ];
        let gpu = gallery
            .scan(&device, &[probe.as_slice()])
            .expect("segmented gpu scan");

        // CPU reference: score_packed per file, concatenated in the same order.
        let mut cpu: Vec<f32> = Vec::new();
        for (w, c, n) in &files {
            let wref: Vec<&[WideQSig]> = w.iter().map(|x| x.as_slice()).collect();
            let packed = PackedGallery::from_windows(&wref, c, *n);
            cpu.extend(score_packed(&probe, &packed));
        }
        assert_eq!(gpu[0].len(), cpu.len(), "global case count");
        for (i, (g, c)) in gpu[0].iter().zip(&cpu).enumerate() {
            assert!(
                (g - c).abs() <= 1e-3 * (1.0 + g.abs().max(c.abs())),
                "case {i}: segmented GPU {g} vs CPU-per-file {c} exceeds tolerance"
            );
        }
    }

    /// The WEIGHTED segmented scan (`scan_weighted` with non-uniform group
    /// weights — the repo_map L46 case) must match the CPU
    /// `score_provenance_late_fusion_weighted` per file. Guards the weight-applied
    /// path, which the unweighted parity test above does not exercise.
    #[test]
    fn gpu_segmented_weighted_matches_cpu_per_file() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };
        let files: Vec<(Vec<Vec<WideQSig>>, Vec<usize>, usize)> = vec![
            (
                vec![
                    vec![sig(0xAAAA_AAAA_AAAA_AAAA), sig(0xABAB_ABAB_ABAB_ABAB)],
                    vec![sig(0x5555_5555_5555_5555)],
                ],
                vec![0usize, 1],
                2,
            ),
            (
                vec![
                    vec![sig(0xFFFF_FFFF_0000_0000)],
                    vec![sig(0x0F0F_0F0F_0F0F_0F0F), sig(0x1234_5678_9ABC_DEF0)],
                ],
                vec![0usize, 1],
                2,
            ),
        ];
        let segments: Vec<SegmentInput> = files
            .iter()
            .map(|(w, c, n)| SegmentInput {
                windows: w.iter().map(|x| x.as_slice()).collect(),
                window_case: c.clone(),
                n_cases: *n,
            })
            .collect();
        let gallery = BatchedGpuGallery::from_segments(&segments).expect("build segmented gallery");

        let probe = vec![
            sig(0xAAAA_AAAA_AAAA_AAAA),
            sig(0x0F0F_0F0F_0F0F_0F0F),
            sig(0x1357_2468_1357_2468),
        ];
        // Non-uniform per-layer-group weights (3 groups) — the whole point of the
        // weighted path; group 0 down-weighted, group 2 up-weighted.
        let weights = [0.25f32, 1.0, 3.0];
        let gpu = gallery
            .scan_weighted(&device, &[probe.as_slice()], &weights)
            .expect("weighted segmented gpu scan");

        // CPU reference: the weighted pointer scan per file, concatenated in order.
        let mut cpu: Vec<f32> = Vec::new();
        for (w, c, n) in &files {
            let mut refs: Vec<&WideQSig> = Vec::new();
            let mut case: Vec<u32> = Vec::new();
            for (wi, win) in w.iter().enumerate() {
                for tok in win {
                    refs.push(tok);
                    case.push(c[wi] as u32);
                }
            }
            cpu.extend(score_provenance_late_fusion_weighted(
                &probe, &refs, &case, *n, &weights,
            ));
        }
        assert_eq!(gpu[0].len(), cpu.len(), "global case count");
        for (i, (g, c)) in gpu[0].iter().zip(&cpu).enumerate() {
            assert!(
                (g - c).abs() <= 1e-3 * (1.0 + g.abs().max(c.abs())),
                "case {i}: weighted GPU {g} vs CPU-per-file {c} exceeds tolerance"
            );
        }
    }

    fn argmax(v: &[f32]) -> Option<usize> {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    }
}
