//! The paged belief scan: build the tiny per-scan index over the resident arena
//! and launch the paged BDP kernel.
//!
//! The gallery records stay resident (in the arena); a scan uploads only the
//! addressing — `page_ptr` (device address per page), `pos_map` (page + offset
//! per scanned token), `case`, the segment prefixes, and the probe — then runs
//! the same kernel as the contiguous path in its paged mode. The host tally is
//! shared with the contiguous path ([`needle_tally_segments`]). See
//! `docs/paged_gallery_arena.md` §7–8.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{Device, Result};
use candle_kernels::provenance::run_batched_bdp_scan;
use core::ffi::c_void;

use crate::persistence::streams::StreamId;

use super::super::gpu::{needle_tally_segments, BDP_TQ};
use super::super::WideQSig;
use super::pages::PAGE_TOKENS;
use super::GalleryArena;

/// A sub-window of a resident turn that contributes to a scan: `[start, end)`
/// tokens of the turn identified by `sid`/`fingerprint`, voting for `case`.
pub struct PagedWindow<'a> {
    pub sid: StreamId,
    pub fingerprint: u64,
    /// The WHOLE turn window (for residency); the scan reads `[start, end)`.
    pub turn: &'a [WideQSig],
    pub start: usize,
    pub end: usize,
    pub case: usize,
}

/// One segment (file): its windows and its exchange (case) count.
pub struct PagedSegment<'a> {
    pub windows: Vec<PagedWindow<'a>>,
    pub n_cases: usize,
}

/// The assembled per-scan index — all small arrays, built from the resident
/// arena. `page_ptr` holds absolute device addresses; the records never move.
pub(super) struct PagedIndex {
    page_ptr: Vec<u64>,
    pos_map: Vec<u32>,
    case: Vec<u32>,
    seg_tok: Vec<i32>,
    seg_case: Vec<i32>,
    n_cases: usize,
    n_segments: usize,
    max_seg_cases: usize,
    /// Turns referenced by this scan; pinned for its duration then unpinned.
    pub(super) pinned_sids: Vec<StreamId>,
}

/// A [`PagedIndex`] cached for reuse (keyed by segment fingerprint in the arena's
/// map), valid while `gen` matches the arena's residency generation.
pub(super) struct CachedIndex {
    pub(super) gen: u64,
    pub(super) idx: Arc<PagedIndex>,
}

/// Fingerprint the segment structure the built index depends on — file order,
/// each file's case count, and every window's `(sid, fingerprint, turn length,
/// start, end, case)`. Cheap (per-window, not per-token).
///
/// **Trust boundary:** two guarantees back reuse. (1) The residency *generation*
/// covers physical page moves — an evict/re-seal that relocates a turn's pages
/// bumps it, so the cache can't serve stale device addresses. (2) This 64-bit
/// SipHash covers logical changes to the scan; a caller MUST vary `w.fingerprint`
/// whenever the turn's content changes (the resolver's `fp_of` folds in the
/// decoded-sig `Arc` len + a content sample), and `turn.len()` is hashed directly
/// so a length change alone (which shifts `pos_map`) also invalidates. A 64-bit
/// hash collision under an unchanged generation could serve a wrong index — a
/// ~2⁻⁶⁴ event we accept, the standard content-address trade.
fn fingerprint_segments(segments: &[PagedSegment]) -> u64 {
    let mut h = DefaultHasher::new();
    segments.len().hash(&mut h);
    for seg in segments {
        seg.n_cases.hash(&mut h);
        seg.windows.len().hash(&mut h);
        for w in &seg.windows {
            w.sid.0.hash(&mut h);
            w.fingerprint.hash(&mut h);
            w.turn.len().hash(&mut h);
            w.start.hash(&mut h);
            w.end.hash(&mut h);
            w.case.hash(&mut h);
        }
    }
    h.finish()
}

impl GalleryArena {
    /// Ensure every referenced turn is resident and assemble the paged index.
    /// A turn is uploaded at most once per scan (deduped by `sid`); its pages'
    /// device addresses are appended to `page_ptr` in first-encounter order and
    /// every window into that turn resolves against the recorded base.
    fn build_index(&self, segments: &[PagedSegment]) -> Result<PagedIndex> {
        let wpt = self.wpt();
        let mut turn_base: HashMap<StreamId, usize> = HashMap::new();
        let mut page_ptr: Vec<u64> = Vec::new();
        let mut pos_map: Vec<u32> = Vec::new();
        let mut case: Vec<u32> = Vec::new();
        let mut seg_tok = vec![0i32];
        let mut seg_case = vec![0i32];
        let mut case_off = 0usize;
        let mut max_seg_cases = 1usize;

        // Turns pinned so far — released on error so a failed build never leaks pins.
        let mut pinned_sids: Vec<StreamId> = Vec::new();
        for seg in segments {
            for w in &seg.windows {
                // Mirror `PackedGallery::from_windows`: drop an out-of-range case.
                if w.case >= seg.n_cases {
                    continue;
                }
                let base = match turn_base.get(&w.sid) {
                    Some(&b) => b,
                    None => {
                        // Pin the turn atomically with residency so the governor
                        // can't free its pages before the launch reads them.
                        let addrs = match self.scan_ensure(w.sid, w.turn, w.fingerprint) {
                            Ok(a) => a,
                            Err(e) => {
                                for &s in &pinned_sids {
                                    self.unpin(s);
                                }
                                return Err(e);
                            }
                        };
                        pinned_sids.push(w.sid);
                        let b = page_ptr.len();
                        page_ptr.extend_from_slice(&addrs);
                        turn_base.insert(w.sid, b);
                        b
                    }
                };
                for t in w.start..w.end.min(w.turn.len()) {
                    // Uniform width only (mirror the reference's admission guard).
                    if w.turn[t].words.len() != wpt {
                        continue;
                    }
                    let page = base + t / PAGE_TOKENS;
                    let in_pg = t % PAGE_TOKENS;
                    pos_map.push(((page as u32) << 5) | (in_pg as u32));
                    case.push((case_off + w.case) as u32);
                }
            }
            case_off += seg.n_cases;
            seg_tok.push(pos_map.len() as i32);
            seg_case.push(case_off as i32);
            max_seg_cases = max_seg_cases.max(seg.n_cases);
        }

        Ok(PagedIndex {
            page_ptr,
            pos_map,
            case,
            seg_tok,
            seg_case,
            n_cases: case_off,
            n_segments: segments.len(),
            max_seg_cases,
            pinned_sids,
        })
    }

    /// Paged belief scan over the resident gallery. Ensures residency, builds the
    /// index, launches the paged kernel, and tallies — one per-case vote vector
    /// per probe. Numerically equivalent to the contiguous
    /// [`BatchedGpuGallery::scan_weighted`](super::super::gpu::BatchedGpuGallery)
    /// on the same windows (only the byte layout differs).
    pub fn scan_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        // Reuse the cached index if the same segment set is rescanned under an
        // unchanged residency generation (the common within-turn case) — this
        // pins the turns. Otherwise rebuild (which also pins) and cache it.
        let fp = fingerprint_segments(segments);
        let idx = match self.reuse_index(fp) {
            Some(idx) => idx,
            None => {
                let built = Arc::new(self.build_index(segments)?);
                let gen = self.residency_gen();
                self.store_index(fp, gen, built.clone());
                built
            }
        };
        let result = self.launch_paged(&idx, probes, group_weights);
        // Release the scan's pins whether or not the launch succeeded — the pages
        // are no longer being read once the launch has synchronized (or failed).
        for &sid in &idx.pinned_sids {
            self.unpin(sid);
        }
        result
    }

    fn launch_paged(
        &self,
        idx: &PagedIndex,
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let dev = match self.device() {
            Device::Cuda(d) => d,
            _ => return Err(candle::Error::Msg("paged scan requires CUDA".into())),
        };
        let wpt = self.wpt();
        let n_groups = self.n_groups();
        let gw = wpt / n_groups;
        let n_cases = idx.n_cases;
        let n_segments = idx.n_segments;

        if probes.is_empty() {
            return Ok(Vec::new());
        }
        if idx.pos_map.is_empty() || n_segments == 0 || n_cases == 0 {
            return Ok(vec![vec![0.0; n_cases]; probes.len()]);
        }

        // Shared-memory budget guard (mirror BatchedGpuGallery).
        let shmem = BDP_TQ * idx.max_seg_cases * std::mem::size_of::<u32>();
        if shmem > 48 * 1024 {
            return Err(candle::Error::Msg(format!(
                "paged scan: largest segment has {} cases ({shmem} B shared > 48 KiB)",
                idx.max_seg_cases
            )));
        }

        // Batch probes (token-major, full-width only — the reference's filter).
        let mut probe_words: Vec<u64> = Vec::new();
        let mut per_req_tokens: Vec<usize> = Vec::with_capacity(probes.len());
        for probe in probes {
            let mut cnt = 0usize;
            for tok in *probe {
                if tok.words.len() >= wpt {
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

        // Upload the tiny index arrays + probe (the records stay resident).
        let d_page_ptr = stream
            .memcpy_stod(&idx.page_ptr)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD page_ptr: {e}")))?;
        let d_pos_map = stream
            .memcpy_stod(&idx.pos_map)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD pos_map: {e}")))?;
        let d_case = stream
            .memcpy_stod(&idx.case)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD case: {e}")))?;
        let d_seg_tok = stream
            .memcpy_stod(&idx.seg_tok)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD seg_tok: {e}")))?;
        let d_seg_case = stream
            .memcpy_stod(&idx.seg_case)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD seg_case: {e}")))?;
        let d_probe = stream
            .memcpy_stod(&probe_words)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD probes: {e}")))?;

        let n_out = n_probe_tokens * n_groups * n_segments;
        let mut d_out_case = unsafe { stream.alloc::<i32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("paged scan: alloc out_case: {e}")))?;
        let mut d_out_vote = unsafe { stream.alloc::<f32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("paged scan: alloc out_vote: {e}")))?;

        {
            let (p_case, _g1) = d_case.device_ptr(&stream);
            let (p_probe, _g2) = d_probe.device_ptr(&stream);
            let (p_seg_tok, _g3) = d_seg_tok.device_ptr(&stream);
            let (p_seg_case, _g4) = d_seg_case.device_ptr(&stream);
            let (p_page_ptr, _g5) = d_page_ptr.device_ptr(&stream);
            let (p_pos_map, _g6) = d_pos_map.device_ptr(&stream);
            let (p_out_case, _g7) = d_out_case.device_ptr_mut(&stream);
            let (p_out_vote, _g8) = d_out_vote.device_ptr_mut(&stream);
            unsafe {
                run_batched_bdp_scan(
                    std::ptr::null(), // gallery_words — paged mode
                    p_case as *const u32,
                    p_probe as *const u64,
                    p_seg_tok as *const i32,
                    p_seg_case as *const i32,
                    p_page_ptr as *const u64,
                    p_pos_map as *const u32,
                    n_probe_tokens as i32,
                    n_groups as i32,
                    n_segments as i32,
                    idx.max_seg_cases as i32,
                    gw as i32,
                    wpt as i32,
                    p_out_case as *mut i32,
                    p_out_vote as *mut f32,
                    stream.cu_stream() as *mut c_void,
                );
            }
            stream
                .synchronize()
                .map_err(|e| candle::Error::Msg(format!("paged scan: synchronize: {e}")))?;
        }

        let out_case = stream
            .memcpy_dtov(&d_out_case)
            .map_err(|e| candle::Error::Msg(format!("paged scan: DtoH out_case: {e}")))?;
        let out_vote = stream
            .memcpy_dtov(&d_out_vote)
            .map_err(|e| candle::Error::Msg(format!("paged scan: DtoH out_vote: {e}")))?;

        Ok(needle_tally_segments(
            &out_case,
            &out_vote,
            &per_req_tokens,
            &idx.seg_case,
            n_groups,
            n_segments,
            n_cases,
            group_weights,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::turn_stream_id;
    use crate::provenance::gpu::{BatchedGpuGallery, SegmentInput};

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24)
                .map(|w| fill.wrapping_mul(0x9E37).wrapping_add(w))
                .collect(),
        }
    }

    /// The paged scan must be **bit-identical** to the contiguous
    /// `BatchedGpuGallery` scan over the same windows: identical bytes, identical
    /// kernel math — only the physical layout differs. Exercises multi-page turns,
    /// seam sub-windows (two windows of one turn into different cases), several
    /// files, and non-uniform group weights. Skips when no CUDA device is present.
    #[test]
    fn paged_matches_contiguous_bit_identical() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        // Turns (whole windows), owned so both paths can borrow them.
        let a0: Vec<WideQSig> = (0..3).map(|t| sig(0x1000 + t)).collect();
        let a1: Vec<WideQSig> = (0..2).map(|t| sig(0x2000 + t)).collect();
        let b0: Vec<WideQSig> = (0..40).map(|t| sig(0x3000 + t)).collect(); // 2 pages
        let c0: Vec<WideQSig> = (0..33).map(|t| sig(0x4000 + t)).collect(); // partial 2nd page

        // ── Paged segments (reference the WHOLE turn + [start,end)) ──
        let paged_segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(1, 0),
                        fingerprint: 10,
                        turn: &a0,
                        start: 0,
                        end: 3,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(1, 1),
                        fingerprint: 11,
                        turn: &a1,
                        start: 0,
                        end: 2,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                // Two sub-windows of ONE multi-page turn → different cases (a seam).
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(2, 0),
                        fingerprint: 20,
                        turn: &b0,
                        start: 0,
                        end: 20,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(2, 0),
                        fingerprint: 20,
                        turn: &b0,
                        start: 20,
                        end: 40,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(3, 0),
                    fingerprint: 30,
                    turn: &c0,
                    start: 0,
                    end: 33,
                    case: 0,
                }],
                n_cases: 1,
            },
        ];

        // ── Contiguous segments (sub-slices packed directly) ──
        let a0s = a0.as_slice();
        let a1s = a1.as_slice();
        let contig_segs = vec![
            SegmentInput {
                windows: vec![&a0s[0..3], &a1s[0..2]],
                window_case: vec![0, 1],
                n_cases: 2,
            },
            SegmentInput {
                windows: vec![&b0[0..20], &b0[20..40]],
                window_case: vec![0, 1],
                n_cases: 2,
            },
            SegmentInput {
                windows: vec![&c0[0..33]],
                window_case: vec![0],
                n_cases: 1,
            },
        ];

        let probe = vec![sig(0x1000), sig(0x3000 + 25), sig(0xBEEF), sig(0x4000 + 10)];
        let weights = [0.25f32, 1.0, 3.0];

        let contiguous = BatchedGpuGallery::from_segments(&contig_segs)
            .unwrap()
            .scan_weighted(&device, &[probe.as_slice()], &weights)
            .unwrap();
        let paged = arena
            .scan_weighted(&paged_segs, &[probe.as_slice()], &weights)
            .unwrap();

        assert_eq!(paged.len(), contiguous.len());
        assert_eq!(paged[0].len(), contiguous[0].len(), "global case count");
        for (i, (p, c)) in paged[0].iter().zip(&contiguous[0]).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "case {i}: paged {p} vs contiguous {c} must be bit-identical"
            );
        }

        // Unweighted must also match (weights = &[] → uniform).
        let contiguous_u = BatchedGpuGallery::from_segments(&contig_segs)
            .unwrap()
            .scan(&device, &[probe.as_slice()])
            .unwrap();
        let paged_u = arena
            .scan_weighted(&paged_segs, &[probe.as_slice()], &[])
            .unwrap();
        for (i, (p, c)) in paged_u[0].iter().zip(&contiguous_u[0]).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "unweighted case {i} must be bit-identical"
            );
        }
    }

    /// Governor relief path: a scan → evict EVERYTHING (as the cheap-rung relief
    /// closure does under VRAM pressure) → the next scan rebuilds residency from
    /// the same sigs and reproduces the result bit-identically. Skips without CUDA.
    #[test]
    fn eviction_then_rebuild_is_stable() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        let t0: Vec<WideQSig> = (0..50).map(|t| sig(0xA00 + t)).collect(); // 2 pages
        let t1: Vec<WideQSig> = (0..12).map(|t| sig(0xB00 + t)).collect();
        let t2: Vec<WideQSig> = (0..7).map(|t| sig(0xC00 + t)).collect();
        let segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(9, 0),
                        fingerprint: 1,
                        turn: &t0,
                        start: 0,
                        end: 50,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(9, 1),
                        fingerprint: 2,
                        turn: &t1,
                        start: 0,
                        end: 12,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(9, 2),
                    fingerprint: 3,
                    turn: &t2,
                    start: 0,
                    end: 7,
                    case: 0,
                }],
                n_cases: 1,
            },
        ];
        let probe = vec![sig(0xA00 + 5), sig(0xC00 + 2), sig(0x999)];
        let weights = [1.0f32, 0.5, 2.0];

        let first = arena
            .scan_weighted(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        let before = arena.resident_turns();
        assert_eq!(before, 3, "three turns resident after the first scan");
        assert!(arena.resident_bytes() > 0);

        // The scan unpinned its working set, so relief can evict everything.
        let freed = arena.evict_lru(u64::MAX);
        assert!(freed > 0, "eviction freed VRAM");
        assert_eq!(arena.resident_turns(), 0, "all turns evicted");

        // Re-scan → residency rebuilds → bit-identical result.
        let second = arena
            .scan_weighted(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        assert_eq!(arena.resident_turns(), before, "rebuilt the same residency");
        assert_eq!(first[0].len(), second[0].len());
        for (i, (a, b)) in first[0].iter().zip(&second[0]).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "case {i}: rebuild after eviction must reproduce the scan bit-identically"
            );
        }
    }

    /// The index cache is fingerprint-KEYED (not a single slot): two distinct
    /// segment sets scanned alternately both stay cached, so neither rebuilds
    /// after its first scan — the multi-belief-group reprojection pattern. Each
    /// still produces the correct result. Skips without CUDA.
    #[test]
    fn index_cache_keyed_by_fingerprint_no_thrash() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        // Two distinct "groups" — different turns/files → different fingerprints.
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xA00 + t)).collect();
        let b: Vec<WideQSig> = (0..15).map(|t| sig(0xB00 + t)).collect();
        let seg_a = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(7, 0),
                fingerprint: 1,
                turn: &a,
                start: 0,
                end: 20,
                case: 0,
            }],
            n_cases: 1,
        }];
        let seg_b = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(8, 0),
                fingerprint: 2,
                turn: &b,
                start: 0,
                end: 15,
                case: 0,
            }],
            n_cases: 1,
        }];
        let probe = vec![sig(0xA00 + 3), sig(0xB00 + 3)];

        // Prime both caches.
        let a1 = arena
            .scan_weighted(&seg_a, &[probe.as_slice()], &[])
            .unwrap();
        let b1 = arena
            .scan_weighted(&seg_b, &[probe.as_slice()], &[])
            .unwrap();
        // Interleave: with a single-slot cache these would each rebuild; keyed,
        // they both hit and reproduce their first result bit-identically. (No
        // residency mutation between scans, so the generation holds.)
        for _ in 0..3 {
            let a2 = arena
                .scan_weighted(&seg_a, &[probe.as_slice()], &[])
                .unwrap();
            let b2 = arena
                .scan_weighted(&seg_b, &[probe.as_slice()], &[])
                .unwrap();
            assert_eq!(a1, a2, "group A cached result must be stable");
            assert_eq!(b1, b2, "group B cached result must be stable");
        }
        assert_eq!(arena.resident_turns(), 2, "both turns resident, no churn");
    }
}
