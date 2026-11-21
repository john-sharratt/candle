//! Tests that the bytes priced for a scoring pass cover what the pass carves.
//!
//! The wave plan refuses a layer that carves past its price, so the bounds are
//! checked against the carves the GEMM and paged routes make at deep chunk shapes.
//! A narrow decode row must not widen the score buffer of a bulk run.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use crate::models::gpu_test_lock::gpu_serial;

    use crate::models::qwen4exp::select_bytes::{select_layer_bytes, CARVE_ALIGN};

    use candle_nn::kv_cache::{begin_wave, LayerPhase};

    use super::super::tests_support::tests::{cuda, lcg, Rig};

    /// **A fresh 128K turn's deepest chunk prices its cuBLAS scoring as one
    /// tile.** 8,192 rows over 32,000 live blocks is past the GEMM crossover:
    /// the tail's page table (1,000 B → 1,024), the rotated tail (16,384,000 B)
    /// and one 262-row tile of 134,144,000 B that all thirty-two tiles are
    /// written into — 150.5 MB, where a product per tile summed to 4.2 GB.
    #[test]
    fn a_fresh_deep_chunk_prices_one_score_tile() {
        let rows = ScoreRows {
            t: 8_192,
            cand_max: 32_000,
            heads: 4,
            d: 128,
        };
        assert_eq!(
            fresh_score_bound(32_000 + 8_192 / 4 + 1, rows, 256),
            150_529_024
        );
    }

    /// Under the crossover the fresh tail rides the paged scorer: four page
    /// descriptors (128 B → 256), five offsets (20 B → 256) and the row counts
    /// (512 B).
    #[test]
    fn a_fresh_shallow_chunk_prices_the_paged_tables() {
        let rows = ScoreRows {
            t: 128,
            cand_max: 1_000,
            heads: 4,
            d: 128,
        };
        assert_eq!(fresh_score_bound(1_001, rows, 256), 1_024);
    }

    /// **The stated bound covers what a layer's selection actually carves.**
    ///
    /// The wave plan prices a layer's selection at
    /// [`super::super::select_bytes::select_layer_bytes`], and an open phase
    /// refuses anything past its price — so a bound short of the real carves
    /// fails the wave. Measured here by the cursor: a marker carve before
    /// `select_layer` and one after, at a depth where the selection engages,
    /// with the block input on the phase as the forward puts it.
    #[test]
    fn the_selection_bound_covers_what_a_layer_carves() -> Result<()> {
        let _gpu = gpu_serial();
        let Some(device) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &device else {
            return Ok(());
        };
        let rig = Rig::new(device.clone(), 128)?;
        let (seq, past, rows) = (7usize, 96usize, 16usize);
        let mut cache = rig.build(past)?;
        cache.ensure_capacity(past + rows, rig.ratio)?;
        let mut idx_map: HashMap<usize, Vec<IndexCache>> = HashMap::new();
        idx_map.insert(seq, vec![cache]);
        let spans = [SeqSpan {
            seq,
            start: 0,
            len: rows,
        }];
        let offsets = [past];
        assert!(selection_engages(past + rows, &rig.cfg, rig.ratio));
        let bound = select_layer_bytes(0, rig.ratio, &spans, &offsets, &idx_map, &rig.cfg)?;

        let hidden = rig.w.q_proj.dim(1)?;
        let h = Tensor::from_vec(lcg(rows * hidden, 0x77, 1.0), (rows, hidden), &device)?;
        let wave = begin_wave(&cd.cuda_stream(), LayerPhase::Attention)?;
        let ticket = Some(wave.ticket());
        let h_wave = wave_empty_ticketed((rows, hidden), DType::F32, &device, ticket)?;
        h_wave.slice_set(&h, 0, 0)?;
        let before = wave.alloc(1, CARVE_ALIGN)?.ptr;
        let sel = select_layer(
            0,
            rig.ratio,
            &rig.w,
            &rig.rope,
            &h_wave,
            &spans,
            &offsets,
            &mut idx_map,
            None,
            rows,
            &rig.cfg,
            rig.eps,
            &device,
            &AtomicU64::new(0),
            ticket,
        )?;
        assert!(sel.is_some(), "the selection engaged");
        drop(sel);
        let after = wave.alloc(1, CARVE_ALIGN)?.ptr;
        let carved = (after - before) as usize - CARVE_ALIGN;
        assert!(
            carved <= bound,
            "select_layer carved {carved} B against a stated bound of {bound} B"
        );
        Ok(())
    }

    /// Runs follow the spans' packed order: a bulk span joins the bulk spans
    /// around it and a narrow one the narrow ones, and every span is in
    /// exactly one run.
    #[test]
    fn spans_are_scored_in_runs_of_bulk_and_narrow() {
        let span = |start: usize, len: usize| SeqSpan {
            seq: start,
            start,
            len,
        };
        let bulk = PREFILL_KV_STAGE_MIN_Q_LEN;
        let spans = [
            span(0, 1),
            span(1, 5),
            span(6, bulk),
            span(6 + bulk, 2 * bulk),
            span(6 + 3 * bulk, 1),
        ];
        assert_eq!(score_runs(&spans), vec![0..2, 2..4, 4..5]);
        assert_eq!(score_runs(&spans[..1]), vec![0..1]);
        assert!(score_runs(&[]).is_empty());
    }

    /// **A deep decode row does not widen a bulk chunk's score buffer.** The
    /// bound still covers what the layer carves, and it is the bulk rows at
    /// their own width plus the decode row at its — under the one-width figure
    /// that priced every row at the deepest row's candidates.
    #[test]
    fn a_deep_narrow_row_does_not_widen_a_bulk_runs_scores() -> Result<()> {
        let _gpu = gpu_serial();
        let Some(device) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &device else {
            return Ok(());
        };
        let (deep_past, bulk_past) = (4096usize, 96usize);
        let rig = Rig::new(device.clone(), deep_past + 1)?;
        let bulk_rows = PREFILL_KV_STAGE_MIN_Q_LEN;
        let mut deep = rig.build(deep_past)?;
        deep.ensure_capacity(deep_past + 1, rig.ratio)?;
        let mut shallow = rig.build(bulk_past)?;
        shallow.ensure_capacity(bulk_past + bulk_rows, rig.ratio)?;
        let mut idx_map: HashMap<usize, Vec<IndexCache>> = HashMap::new();
        idx_map.insert(1, vec![deep]);
        idx_map.insert(2, vec![shallow]);
        let spans = [
            SeqSpan {
                seq: 1,
                start: 0,
                len: 1,
            },
            SeqSpan {
                seq: 2,
                start: 1,
                len: bulk_rows,
            },
        ];
        let offsets = [deep_past, bulk_past];
        let rows = 1 + bulk_rows;
        let bound = select_layer_bytes(0, rig.ratio, &spans, &offsets, &idx_map, &rig.cfg)?;
        let widest = widest_candidates(&spans, &offsets, &idx_map, 0, rig.ratio);
        let one_width = rows * widest * 4;
        let bulk_width = widest_candidates(&spans[1..], &offsets[1..], &idx_map, 0, rig.ratio);
        assert!(
            bulk_rows * bulk_width * 4 + widest * 4 < one_width,
            "the bulk run is narrower than the deep row: {bulk_width} against {widest}"
        );

        let hidden = rig.w.q_proj.dim(1)?;
        let h = Tensor::from_vec(lcg(rows * hidden, 0x78, 1.0), (rows, hidden), &device)?;
        let wave = begin_wave(&cd.cuda_stream(), LayerPhase::Attention)?;
        let ticket = Some(wave.ticket());
        let h_wave = wave_empty_ticketed((rows, hidden), DType::F32, &device, ticket)?;
        h_wave.slice_set(&h, 0, 0)?;
        let before = wave.alloc(1, CARVE_ALIGN)?.ptr;
        let sel = select_layer(
            0,
            rig.ratio,
            &rig.w,
            &rig.rope,
            &h_wave,
            &spans,
            &offsets,
            &mut idx_map,
            None,
            rows,
            &rig.cfg,
            rig.eps,
            &device,
            &AtomicU64::new(0),
            ticket,
        )?;
        assert!(sel.is_some(), "the selection engaged");
        drop(sel);
        let after = wave.alloc(1, CARVE_ALIGN)?.ptr;
        let carved = (after - before) as usize - CARVE_ALIGN;
        assert!(
            carved <= bound,
            "select_layer carved {carved} B against a stated bound of {bound} B"
        );
        Ok(())
    }
}
