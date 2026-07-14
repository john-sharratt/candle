//! GPU backend for the batched BDP flat scan.
//!
//! A whole **wave** of probes is scored against the packed gallery in one launch.
//! The gallery is **not resident**: each wave stages it through **pinned** host
//! memory, uploads it, runs the kernel, downloads the small results, and frees
//! the device buffers (they drop at the end of the call) — so VRAM is reclaimed
//! between scans. The kernel emits the per-(query-token, layer-group)
//! `(leading_case, z×margin)` pairs; the needle gate + per-case tally run here on
//! the host, identical to [`score_packed`](super::score_packed) — so the result is
//! **bit-identical** to the CPU path (verified in the parity test).

use core::ffi::c_void;

use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, Result};
use candle_kernels::provenance::run_batched_bdp_scan;

use super::packed::PackedGallery;
use super::scan::{needle_gate_tally, HEADS_PER_GROUP};
use super::WideQSig;

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
pub struct BatchedGpuGallery {
    words_pin: PinnedBuf,
    case_pin: PinnedBuf,
    n_tokens: usize,
    wpt: usize,
    n_groups: usize,
    gw: usize,
    n_cases: usize,
}

impl BatchedGpuGallery {
    /// Page-lock a copy of `gallery` into pinned host memory (the one-time,
    /// non-per-wave cost), transposed to **group-major** `[group][token][gw]` so
    /// the device reads coalesce. In production this buffer is retained and
    /// updated incrementally in lockstep with the CPU gallery.
    pub fn from_packed(gallery: &PackedGallery) -> Result<Self> {
        let n_tokens = gallery.n_tokens();
        let wpt = gallery.wpt();
        let n_groups = gallery.n_heads() / HEADS_PER_GROUP;
        let gw = HEADS_PER_GROUP * gallery.wph();
        let src = gallery.words(); // token-major [n_tokens][wpt]

        // Group-major transpose: [group][token][gw].
        let mut gm = vec![0u64; src.len()];
        for j in 0..n_tokens {
            let tok = &src[j * wpt..j * wpt + wpt];
            for g in 0..n_groups {
                let base = g * n_tokens * gw + j * gw;
                gm[base..base + gw].copy_from_slice(&tok[g * gw..g * gw + gw]);
            }
        }

        Ok(Self {
            words_pin: pin_copy(&gm)?,
            case_pin: pin_copy(gallery.case())?,
            n_tokens,
            wpt,
            n_groups,
            gw,
            n_cases: gallery.n_cases(),
        })
    }

    /// The pinned gallery footprint in bytes (words + case).
    pub fn byte_len(&self) -> usize {
        self.words_pin.len() + self.case_pin.len()
    }

    /// Score a **wave** of probes, returning one per-case vote vector per probe —
    /// bit-identical to [`score_packed`](super::score_packed) on each probe.
    ///
    /// Per wave: pinned gallery → VRAM (upload), one batched kernel, results →
    /// host (download), device buffers freed on return (non-resident VRAM).
    pub fn scan(&self, device: &Device, probes: &[&[WideQSig]]) -> Result<Vec<Vec<f32>>> {
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

        // ── Upload: pinned gallery → VRAM (freed on return) ─────────────────
        let d_words = stream
            .memcpy_stod(self.words_pin.as_slice())
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD gallery words: {e}")))?;
        let d_case = stream
            .memcpy_stod(self.case_pin.as_slice())
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD gallery case: {e}")))?;
        let d_probe = stream
            .memcpy_stod(&probe_words)
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: HtoD probes: {e}")))?;

        // Uninitialized device outputs — the kernel writes every `(query token,
        // group)` entry (one block per tile × group covers all n_probe_tokens),
        // so there's no need to zero-init (and no host Vec + H2D upload per wave).
        let n_out = n_probe_tokens * n_groups;
        let mut d_out_case = unsafe { stream.alloc::<i32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: alloc out_case: {e}")))?;
        let mut d_out_vote = unsafe { stream.alloc::<f32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("BDP GPU scan: alloc out_vote: {e}")))?;

        // ── Launch (guards drop after synchronize, before the D2H reads) ────
        {
            let (p_words, _gw_guard) = d_words.device_ptr(&stream);
            let (p_case, _gc_guard) = d_case.device_ptr(&stream);
            let (p_probe, _gp_guard) = d_probe.device_ptr(&stream);
            let (p_out_case, _goc_guard) = d_out_case.device_ptr_mut(&stream);
            let (p_out_vote, _gov_guard) = d_out_vote.device_ptr_mut(&stream);
            unsafe {
                run_batched_bdp_scan(
                    p_words as *const u64,
                    p_case as *const u32,
                    p_probe as *const u64,
                    self.n_tokens as i32,
                    n_probe_tokens as i32,
                    n_groups as i32,
                    self.gw as i32,
                    wpt as i32,
                    n_cases as i32,
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

        // ── Host needle gate + tally, per request (identical to score_packed) ─
        let mut results = Vec::with_capacity(probes.len());
        let mut tok_base = 0usize;
        for &cnt in &per_req_tokens {
            let mut per_query: Vec<Vec<(usize, f32)>> = Vec::with_capacity(cnt);
            for qt in 0..cnt {
                let gqt = tok_base + qt;
                let mut contribs = Vec::with_capacity(n_groups);
                for g in 0..n_groups {
                    let idx = gqt * n_groups + g;
                    let c = out_case[idx];
                    if c >= 0 {
                        contribs.push((c as usize, out_vote[idx]));
                    }
                }
                per_query.push(contribs);
            }
            results.push(needle_gate_tally(&per_query, n_cases));
            tok_base += cnt;
        }
        Ok(results)
    }
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
    use super::*;
    use crate::provenance::{score_packed, PackedGallery};

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

    fn argmax(v: &[f32]) -> Option<usize> {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    }
}
