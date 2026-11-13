//! The shadow scan's host side (`docs/hit_driven_projection.md` §6.3).
//!
//! The real selection ranks only the slot. At each boundary the shadow scan
//! asks, of every pooled round that is not loaded, what the indexer would have
//! selected had it been there: the stride's kept queries — written beside the
//! hit count, [`super::hits`] — are scored against the unloaded rows in
//! windows seated behind each query, and a row in its window's top `keep` and
//! above the query's cut is one hit for its round, in the same counters
//! (`qsa_shadow.cu`).
//!
//! A declaration names the unloaded rows once ([`ShadowRows`]): each pooled
//! round's resident page at the layer and the rows of it the round holds, and
//! a device map from row to counter. The page descriptors are read at the scan,
//! not at the declaration — a compaction between forwards moves a page, so an
//! address captured once is invalidated (AGENTS hot-path invariant 7) — and go
//! up with the span table in one staged upload.

use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

use candle::{DType, Device, Result, Tensor};
use candle_kernels::simple::qsa_shadow::{
    run_qsa_shadow, ShadowSpan, SHADOW_MAX_HEADS, SHADOW_MAX_Q_WORDS, SHADOW_PAGE_WORDS,
    SHADOW_SPAN_WORDS,
};

use super::index_keys::PAGE_BLOCKS;
use super::indexer::u32_ptr;
use super::resident_page::ResidentPage;
use crate::models::held_tensor::held_empty;
use crate::models::wave_buffers::upload_into;

/// One run of a pooled round's unloaded rows at one layer: the page, the rows
/// of it the round holds, and the round's counter.
#[derive(Debug, Clone)]
pub struct ShadowSegment {
    pub page: Arc<ResidentPage>,
    pub rows: Range<usize>,
    pub counter: u32,
}

/// A declaration's unloaded rows at one sampled layer, in pool order.
#[derive(Debug)]
pub struct ShadowRows {
    segments: Vec<ShadowSegment>,
    /// `[n_rows]` U32 on held ground: each row's counter.
    row_map: Tensor,
    n_rows: usize,
}

/// What the kernel's arithmetic depends on besides the rows and the queries.
#[derive(Debug, Clone, Copy)]
pub struct ShadowGeometry {
    pub heads: usize,
    pub head_dim: usize,
    /// Rotary pairs of the index keys.
    pub pairs: usize,
    pub ratio: usize,
    pub top_k: usize,
}

/// One due sequence at one sampled layer.
#[derive(Debug, Clone, Copy)]
pub struct ShadowJob<'a> {
    /// Device address of this layer's kept queries.
    pub keep: u64,
    /// The kept slots the stride wrote.
    pub mask: u64,
    pub rows: &'a ShadowRows,
    /// Device address of the sequence's counters.
    pub counters: u64,
    pub n_pieces: u32,
    /// Device address of the RoPE table of the sequence's rung.
    pub tab: u64,
}

impl ShadowRows {
    /// The rows `segments` name, in their order, with each row's counter
    /// uploaded to held ground once.
    pub fn new(segments: Vec<ShadowSegment>, device: &Device) -> Result<Self> {
        for s in &segments {
            if s.rows.end > s.page.rows() {
                candle::bail!(
                    "shadow rows: rows {:?} past a page of {}",
                    s.rows,
                    s.page.rows()
                );
            }
        }
        let map: Vec<u32> = segments
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.counter, s.rows.len()))
            .collect();
        let n_rows = map.len();
        // Never zero-sized; `n_rows` says none of it is read.
        let map = if map.is_empty() { vec![0] } else { map };
        let row_map = held_empty(map.len(), DType::U32, device)?;
        upload_into(&row_map, &map)?;
        Ok(Self {
            segments,
            row_map,
            n_rows,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The rows' page descriptors as they stand — `{keys, cstride, rstride, 0}`
    /// for every run of rows one chunk holds — and the exclusive prefix of rows
    /// over them.
    fn descriptors(&self) -> Result<(Vec<i64>, Vec<u32>)> {
        let mut desc: Vec<i64> = Vec::new();
        let mut first: Vec<u32> = vec![0];
        let mut at = 0u32;
        for s in &self.segments {
            for (c, chunk) in s.page.descriptors()?.iter().enumerate() {
                let lo = c * PAGE_BLOCKS;
                let hi = lo + chunk.rows;
                let (a, b) = (s.rows.start.max(lo), s.rows.end.min(hi));
                if a >= b {
                    continue;
                }
                // Row `a` of the page is row `a − lo` of the chunk: `row_stride`
                // float4 groups per row.
                let off = (a - lo) as u64 * chunk.row_stride as u64 * 16;
                let entry: [i64; SHADOW_PAGE_WORDS] = [
                    (chunk.ptr + off) as i64,
                    chunk.group_stride,
                    chunk.row_stride,
                    0,
                ];
                desc.extend(entry);
                at += (b - a) as u32;
                first.push(at);
            }
        }
        Ok((desc, first))
    }
}

/// Score every job's kept queries against its unloaded rows and count the
/// hits, in one launch. The span table, every job's descriptors and their row
/// prefixes go up in one staged upload.
pub fn launch(jobs: &[ShadowJob<'_>], geom: &ShadowGeometry, device: &Device) -> Result<()> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("the shadow scan runs on CUDA");
    };
    if geom.heads > SHADOW_MAX_HEADS || geom.heads * geom.head_dim > SHADOW_MAX_Q_WORDS {
        candle::bail!(
            "shadow scan: {} heads of {} past the kernel's query buffer of {SHADOW_MAX_Q_WORDS}",
            geom.heads,
            geom.head_dim
        );
    }
    let jobs: Vec<&ShadowJob<'_>> = jobs
        .iter()
        .filter(|j| j.mask != 0 && j.rows.n_rows > 0)
        .collect();
    if jobs.is_empty() {
        return Ok(());
    }
    let tables: Vec<(Vec<i64>, Vec<u32>)> = jobs
        .iter()
        .map(|j| j.rows.descriptors())
        .collect::<Result<_>>()?;
    // The spans, then every job's descriptors, then every job's prefixes — the
    // descriptors on even words, so the i64s are aligned.
    let span_words = jobs.len() * SHADOW_SPAN_WORDS;
    let desc_words: usize = tables.iter().map(|(d, _)| d.len() * 2).sum();
    let mut words: Vec<u32> = vec![0; span_words];
    let mut pages_at = span_words;
    let mut first_at = span_words + desc_words;
    let mut firsts: Vec<u32> = Vec::new();
    for (i, (job, (desc, first))) in jobs.iter().zip(&tables).enumerate() {
        let span = ShadowSpan {
            keep: job.keep as *const u32,
            row_map: u32_ptr(&job.rows.row_map)? as *const u32,
            counters: job.counters as *mut u32,
            tab: job.tab as *const f32,
            pages_off: pages_at as u32,
            first_off: first_at as u32,
            n_rows: job.rows.n_rows as u32,
            n_pages: (first.len() - 1) as u32,
            n_pieces: job.n_pieces,
            mask_lo: job.mask as u32,
            mask_hi: (job.mask >> 32) as u32,
            pad: 0,
        };
        words[i * SHADOW_SPAN_WORDS..(i + 1) * SHADOW_SPAN_WORDS].copy_from_slice(&span.words());
        for d in desc {
            words.push(*d as u64 as u32);
            words.push((*d as u64 >> 32) as u32);
        }
        pages_at += desc.len() * 2;
        first_at += first.len();
        firsts.extend_from_slice(first);
    }
    words.extend_from_slice(&firsts);
    let max_rows = jobs.iter().map(|j| j.rows.n_rows).max().unwrap_or(0);
    let stream = cuda.cuda_stream();
    cuda.with_staged_upload(&words, |table| {
        candle::set_kernel_breadcrumb("run_qsa_shadow", file!(), line!());
        // SAFETY: the table holds every span and the descriptors and prefixes
        // they name until the launch has read it; every pointer in it names a
        // live buffer — the kept queries, the row map and the counters are
        // held by their declaration, the pages by its segments — on this
        // stream.
        unsafe {
            run_qsa_shadow(
                table as *const c_void,
                jobs.len() as i32,
                max_rows as i32,
                geom.heads as i32,
                geom.head_dim as i32,
                geom.pairs as i32,
                geom.ratio as i32,
                geom.top_k as i32,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok(())
    })
}

/// The shadow scan's definition on the host, for one kept query: `q` its
/// rotated query (`heads × head_dim`), `cut`, `cand` and `tail` its cut
/// record, `rows` the unloaded rows' un-rotated keys in pool order and
/// `counter` each row's counter; `cos_sin(pos, pair)` the rung's rotation.
/// Adds the query's hits to `counters`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn reference(
    q: &[f32],
    cut: f32,
    cand: usize,
    tail: usize,
    rows: &[Vec<f32>],
    counter: &[u32],
    geom: &ShadowGeometry,
    cos_sin: impl Fn(usize, usize) -> (f32, f32),
    counters: &mut [u32],
) {
    let (h, d, pairs, ratio) = (geom.heads, geom.head_dim, geom.pairs, geom.ratio);
    let budget = geom.top_k + ratio - 1 - tail;
    let keep = budget.div_ceil(ratio);
    for lo in (0..rows.len()).step_by(cand) {
        let hi = (lo + cand).min(rows.len());
        let seat = cand - (hi - lo);
        let mut above: Vec<(f64, usize)> = Vec::new();
        for (j, k) in rows[lo..hi].iter().enumerate() {
            let g = lo + j;
            let pos = (seat + j) * ratio;
            let mut rot = k.clone();
            for c in 0..pairs {
                let (co, si) = cos_sin(pos, c);
                let (a, b) = (k[c] as f64, k[c + pairs] as f64);
                rot[c] = (a * co as f64 - b * si as f64) as f32;
                rot[c + pairs] = (b * co as f64 + a * si as f64) as f32;
            }
            let score: f64 = (0..h)
                .map(|hh| {
                    (0..d)
                        .map(|c| q[hh * d + c] as f64 * rot[c] as f64)
                        .sum::<f64>()
                        .max(0.0)
                })
                .sum();
            if score > cut as f64 {
                above.push((score, g));
            }
        }
        above.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, g) in above.into_iter().take(keep) {
            counters[counter[g] as usize] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial;
    use crate::models::rope_schedule::{plain_inv_freq, FactoredRope};
    use candle_kernels::simple::qsa_hits::KEEP_SLOTS;
    use candle_kernels::simple::qsa_topk::{CUT_WORDS, NO_CUT};

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// **The shadow scan counts what its definition counts, byte for byte**:
    /// three pooled rounds over two pages — one round's rows starting partway
    /// into a page — scored by four kept queries of differing depth, so the
    /// rows cut into windows of differing length with a short last window
    /// seated behind the query; budgets small enough that a window's rows above
    /// the cut outnumber its keep; a dense slot and an unkept slot that count
    /// nothing; the counters' shadow-query word one per kept query.
    #[test]
    fn the_shadow_scan_counts_what_its_definition_counts() -> Result<()> {
        let _gpu = gpu_serial();
        let Ok(device) = Device::cuda_if_available(0) else {
            return Ok(());
        };
        if !device.is_cuda() {
            return Ok(());
        }
        let geom = ShadowGeometry {
            heads: 2,
            head_dim: 16,
            pairs: 4,
            ratio: 4,
            top_k: 12,
        };
        let d = geom.head_dim;
        let rope = FactoredRope::new(&plain_inv_freq(2 * geom.pairs, 1e4), &device)?;
        let page_a: Vec<f32> = lcg(23 * d, 1);
        let page_b: Vec<f32> = lcg(30 * d, 2);
        let pages = ResidentPage::place_host(&[(&page_a, 4), (&page_b, 2)], d, &device)?;
        let seg = |p: usize, rows: Range<usize>, counter: u32| ShadowSegment {
            page: Arc::clone(&pages[p]),
            rows,
            counter,
        };
        // Rounds 2, 3, 4 of a declaration whose first two rounds are loaded.
        let segments = vec![seg(0, 0..23, 2), seg(1, 5..20, 3), seg(1, 20..30, 4)];
        let rows = ShadowRows::new(segments.clone(), &device)?;
        assert_eq!(rows.n_rows(), 48);
        let host_rows: Vec<Vec<f32>> = segments
            .iter()
            .flat_map(|s| {
                let src = if Arc::ptr_eq(&s.page, &pages[0]) {
                    &page_a
                } else {
                    &page_b
                };
                s.rows
                    .clone()
                    .map(move |r| src[r * d..(r + 1) * d].to_vec())
            })
            .collect();
        let counter: Vec<u32> = segments
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.counter, s.rows.len()))
            .collect();

        // Kept slots: (slot, cut, cand, tail). Slot 9 is dense; slot 12 is
        // written but not in the stride's mask.
        let q_words = geom.heads * d;
        let rec = q_words + CUT_WORDS;
        let kept: [(usize, f32, usize, usize); 5] = [
            (3, 0.4, 10, 3),
            (9, f32::NAN, 10, 1),
            (12, 0.1, 7, 2),
            (40, 0.9, 13, 1),
            (63, 0.2, 7, 4),
        ];
        let queries: Vec<Vec<f32>> = (0..kept.len())
            .map(|i| lcg(q_words, 100 + i as u64))
            .collect();
        let mut keep = vec![0u32; KEEP_SLOTS * rec];
        let mut mask = 0u64;
        for (i, &(slot, cut, cand, tail)) in kept.iter().enumerate() {
            let at = slot * rec;
            for (j, v) in queries[i].iter().enumerate() {
                keep[at + j] = v.to_bits();
            }
            keep[at + q_words] = if cut.is_nan() { NO_CUT } else { cut.to_bits() };
            keep[at + q_words + 1] = (slot + 64 * 9) as u32;
            keep[at + q_words + 2] = cand as u32;
            keep[at + q_words + 3] = tail as u32;
            if slot != 12 {
                mask |= 1 << slot;
            }
        }
        let keep_t = Tensor::from_vec(keep, KEEP_SLOTS * rec, &device)?;
        let n_pieces = 5usize;
        // Hits, queries, shadow queries, publish sequence; seeded to show the
        // scan adds.
        let counters = Tensor::from_vec(vec![7u32, 7, 1, 2, 3, 50, 60, 0], 8, &device)?;
        let tab = rope.table(0)?;
        let job = ShadowJob {
            keep: u32_ptr(&keep_t)?,
            mask,
            rows: &rows,
            counters: u32_ptr(&counters)?,
            n_pieces: n_pieces as u32,
            tab: super::super::indexer::tensor_ptr(&tab)?,
        };
        launch(&[job], &geom, &device)?;
        device.synchronize()?;

        let mut want = vec![7u32, 7, 1, 2, 3];
        let mut shadow_queries = 60u32;
        for (i, &(slot, cut, cand, tail)) in kept.iter().enumerate() {
            if slot == 12 || cut.is_nan() {
                continue;
            }
            shadow_queries += 1;
            reference(
                &queries[i],
                cut,
                cand,
                tail,
                &host_rows,
                &counter,
                &geom,
                |pos, k| rope.rungs().cos_sin(0, pos, k),
                &mut want,
            );
        }
        let got = counters.to_vec1::<u32>()?;
        assert_eq!(got[..n_pieces].to_vec(), want, "hits per round");
        assert!(
            want[2..].iter().zip([1u32, 2, 3]).any(|(&w, s)| w > s),
            "the scan found something to count"
        );
        assert_eq!(got[n_pieces], 50, "the selection's queries are untouched");
        assert_eq!(got[n_pieces + 1], shadow_queries, "one per kept query");
        Ok(())
    }
}
