//! Tests the selection kernel against the shared definition, on scores the test
//! chooses.
//!
//! Ties are forced onto a coarse grid, so the rule that resolves them by
//! ascending block is exercised. The single pass, the split pass, pinned ranges,
//! several windows, and windows wider than the buffer are each compared with the
//! definition. A case at the deployed depth checks the geometry the released
//! budget uses.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use crate::models::qwen4exp::qsa_select::{entry_block, selection_entries, RowSelection};

    use super::super::tests_support::tests::{
        cuda, lcg, shallow_rows, strata, windowed_rows, KernelCase,
    };

    /// The three ceilings the selection kernel imposes, each refused by name,
    /// and the stride a run that fits is given.
    #[test]
    fn selection_stride_refuses_what_the_kernel_cannot_run() {
        let err = |r: Result<usize>| r.unwrap_err().to_string();
        assert!(err(selection_stride(0, 2048, 1000, &Strata::WHOLE)).contains("ratio 0"));
        assert!(
            err(selection_stride(MAX_RATIO + 1, 2048, 1000, &Strata::WHOLE)).contains("outside")
        );
        // 4096 positions at ratio 4 keep 1025 blocks, past the 768 buffer.
        assert!(err(selection_stride(4, 4096, 100_000, &Strata::WHOLE)).contains("survivors"));
        // One-block windows over 20,000 candidates gather 20,000 · 513.
        let tiny = Strata {
            window_blocks: 1,
            recent_blocks: 0,
            recent: Recent::Candidate,
            pin_blocks: 0,
        };
        assert!(err(selection_stride(4, 2048, 20_000, &tiny)).contains("gather"));
        assert_eq!(
            selection_stride(4, 2048, 73_728, &Strata::WHOLE).unwrap(),
            514
        );
    }

    /// The selection kernel against the shared definition, row by row, on
    /// scores chosen by the test — no model, no GEMM, so any disagreement is
    /// the kernel's.
    ///
    /// `quantize` collapses the scores onto a coarse grid, which is how the
    /// tie-breaking rule gets exercised: at 8 distinct values over a few
    /// hundred blocks the cut lands inside a run of equal scores on almost
    /// every row, and the reference resolves those by ascending block.
    pub(in super::super) fn kernel_matches(case: &KernelCase) -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let ratio = 4usize;
        let rows = case.qpos.len();
        let cand: Vec<u32> = case
            .qpos
            .iter()
            .map(|&p| ((p + 1) / ratio) as u32)
            .collect();
        let blocks = *cand.iter().max().unwrap() as usize;

        let mut host = lcg(rows * blocks, case.seed, 4.0);
        if let Some(step) = case.quantize {
            for v in host.iter_mut() {
                *v = (*v / step).floor() * step;
            }
        }
        let scores = Tensor::from_vec(host.clone(), (rows, blocks), &device)?;
        let pin_row = PinRow::new(&case.pins, usize::MAX);
        let pinned = pin_row.ranges();
        let strata = Strata {
            pin_blocks: pin_row.blocks(),
            ..case.strata
        };
        let mut table =
            SelectionTable::new(rows, ratio, case.top_k, blocks, &strata, &device, None)?;
        // Uniform blocks here, so the tail is what the kernel used to derive.
        let tail: Vec<u32> = case
            .qpos
            .iter()
            .zip(&cand)
            .map(|(&p, &c)| (p + 1 - c as usize * ratio) as u32)
            .collect();
        let prompt: Vec<u32> = (0..rows as u32)
            .map(|r| case.prompt_blocks + r * case.prompt_spread)
            .collect();
        let pins = vec![pin_row; rows];
        table.fill_rows(
            &scores, &cand, &case.qpos, &tail, &prompt, &pins, ratio, case.top_k, 0,
        )?;

        let mut want = Vec::new();
        for (r, &p) in case.qpos.iter().enumerate() {
            let row_scores = &host[r * blocks..r * blocks + blocks];
            let sel = selection_entries(
                row_scores,
                p,
                ratio,
                case.top_k,
                &strata,
                prompt[r] as usize,
                &pinned,
                &mut want,
            );
            let got = table.row_to_host(r)?;
            let (cut, geometry) = table.cut_to_host(r)?;
            assert_eq!(
                geometry,
                [p as u32, cand[r], tail[r]],
                "row {r}'s cut record carries its geometry"
            );
            match sel {
                RowSelection::Dense => {
                    assert!(got.is_none(), "row {r} should be dense");
                    assert_eq!(cut, None, "a dense row has no cut");
                }
                RowSelection::Entries(_) => {
                    let got = got.unwrap_or_else(|| panic!("row {r} came back dense"));
                    assert_eq!(got, want, "row {r} (qpos {p}) selection differs");
                    // The cut is the lowest score any window ranked in: every
                    // selected block but the pinned ones, the forced span and
                    // the query's own tail block.
                    let c = cand[r] as usize;
                    let forced_lo = match strata.recent {
                        Recent::Forced => c.saturating_sub(strata.recent_blocks),
                        _ => c,
                    };
                    let want_cut = want
                        .iter()
                        .map(|&e| entry_block(e))
                        .filter(|&b| b < forced_lo && !pinned.iter().any(|g| g.contains(&b)))
                        .map(|b| row_scores[b])
                        .reduce(f32::min);
                    assert_eq!(
                        cut.map(f32::to_bits),
                        want_cut.map(f32::to_bits),
                        "row {r} (qpos {p}) cut differs"
                    );
                }
            }
        }
        Ok(())
    }

    /// **Pinned ranges on the device match the definition**: one window
    /// (the bitmap path) and several (the sorted union), on a tie grid, with
    /// pins at the prompt, in the middle, and reaching into a forced span.
    #[test]
    fn pinned_selection_kernel_matches_the_definition() -> Result<()> {
        let cases = [
            (Strata::WHOLE, shallow_rows(), vec![0..3, 6..8]),
            (Strata::WHOLE, shallow_rows(), vec![1..2]),
            (
                strata(32, 0, Recent::Candidate),
                windowed_rows(),
                vec![0..12, 40..44],
            ),
            (
                strata(32, 8, Recent::Forced),
                windowed_rows(),
                vec![2..6, 150..400],
            ),
            (
                strata(0, 0, Recent::Candidate),
                (0..4).map(|i| 16000 + i * 3).collect(),
                vec![0..30, 900..905, 2000..2001, 3000..3010],
            ),
            // Past `SPLIT_BLOCKS` rows, so the single pass runs them: one window
            // (the bitmap) and several (the sorted union).
            (
                Strata::WHOLE,
                (0..512).map(|i| 40 + (i % 16) * 7).collect(),
                vec![0..3, 6..8],
            ),
            (
                strata(32, 8, Recent::Forced),
                (0..512).map(|i| 200 + (i % 16) * 37).collect(),
                vec![2..6, 150..400],
            ),
        ];
        for (n, (strata, qpos, pins)) in cases.into_iter().enumerate() {
            for (quantize, seed) in [(None, 0x91 + n as u64), (Some(0.5), 0xA1 + n as u64)] {
                kernel_matches(&KernelCase {
                    qpos: qpos.clone(),
                    top_k: if n == 4 { 64 } else { 8 },
                    quantize,
                    seed,
                    strata,
                    prompt_blocks: 5,
                    prompt_spread: 0,
                    pins: pins.clone(),
                })?;
            }
        }
        Ok(())
    }

    #[test]
    fn selection_kernel_matches_the_definition() -> Result<()> {
        kernel_matches(&KernelCase {
            qpos: shallow_rows(),
            top_k: 8,
            quantize: None,
            seed: 0x51,
            strata: Strata::WHOLE,
            prompt_blocks: 0,
            prompt_spread: 0,
            pins: Vec::new(),
        })
    }

    #[test]
    fn selection_kernel_breaks_ties_by_ascending_block() -> Result<()> {
        // A coarse grid forces long runs of equal scores through the cut: at 8
        // distinct values the cut lands inside a run on almost every row, and
        // the reference resolves those by ascending block.
        kernel_matches(&KernelCase {
            qpos: shallow_rows(),
            top_k: 8,
            quantize: Some(0.5),
            seed: 0x52,
            strata: Strata::WHOLE,
            prompt_blocks: 0,
            prompt_spread: 0,
            pins: Vec::new(),
        })
    }

    #[test]
    fn selection_kernel_survives_more_blocks_than_the_buffer_holds() -> Result<()> {
        // Past one streaming trim: ~4000 candidate blocks against a 1024-slot
        // buffer, so the threshold path runs many times.
        kernel_matches(&KernelCase {
            qpos: (0..4).map(|i| 16000 + i * 3).collect(),
            top_k: 64,
            quantize: None,
            seed: 0x53,
            strata: Strata::WHOLE,
            prompt_blocks: 0,
            prompt_spread: 0,
            pins: Vec::new(),
        })
    }

    /// Several windows, each spending the budget, with the prompt ranked in
    /// every one — and on a tie grid, so a prompt block two windows both choose
    /// arrives twice and must leave once, at its widest cut.
    #[test]
    fn stratified_selection_kernel_matches_the_definition() -> Result<()> {
        for (quantize, seed) in [(None, 0x61), (Some(0.5), 0x62)] {
            kernel_matches(&KernelCase {
                qpos: windowed_rows(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(32, 0, Recent::Candidate),
                prompt_blocks: 5,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    #[test]
    fn stratified_selection_kernel_ranks_a_recent_candidate_in_every_window() -> Result<()> {
        for (quantize, seed) in [(None, 0x63), (Some(0.5), 0x64)] {
            kernel_matches(&KernelCase {
                qpos: windowed_rows(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(32, 12, Recent::Candidate),
                prompt_blocks: 5,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    #[test]
    fn stratified_selection_kernel_attends_a_forced_span_whole() -> Result<()> {
        for (quantize, seed) in [(None, 0x65), (Some(0.5), 0x66)] {
            kernel_matches(&KernelCase {
                qpos: windowed_rows(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(32, 12, Recent::Forced),
                prompt_blocks: 5,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    /// Windows wider than the buffer, so every window streams through several
    /// trims of its own, with a forced span past one chunk's width.
    #[test]
    fn stratified_selection_kernel_survives_windows_wider_than_the_buffer() -> Result<()> {
        for recent in [Recent::Candidate, Recent::Forced] {
            kernel_matches(&KernelCase {
                qpos: (0..4).map(|i| 16000 + i * 3).collect(),
                top_k: 64,
                quantize: None,
                seed: 0x67,
                strata: strata(1024, 300, recent),
                prompt_blocks: 100,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    /// Which pass a launch runs: a launch of a few rows splits every row across
    /// blocks, and one of `SPLIT_BLOCKS` rows or more never splits — so the
    /// wide cases below are the single pass's, on any device.
    pub(in super::super) fn split_parts(
        rows: usize,
        cand_max: usize,
        strata: &Strata,
        top_k: usize,
    ) -> i32 {
        unsafe {
            qsa_topk_split_parts(
                rows as i32,
                cand_max as i32,
                strata.window_blocks as i32,
                top_k as i32,
                4,
            )
        }
    }

    #[test]
    fn a_narrow_launch_splits_and_a_wide_one_does_not() -> Result<()> {
        let Some(_device) = cuda() else { return Ok(()) };
        assert!(split_parts(4, 188, &strata(32, 12, Recent::Forced), 8) > 0);
        assert!(split_parts(4, 4000, &Strata::WHOLE, 64) > 1);
        assert_eq!(split_parts(512, 188, &strata(32, 12, Recent::Forced), 8), 0);
        assert_eq!(split_parts(512, 4000, &Strata::WHOLE, 64), 0);
        Ok(())
    }

    /// The prompt overlapping everything a pool is cut from. Row r's prompt is
    /// `3r` blocks against 15–30 candidates, so across the rows it ends inside
    /// a window, inside the recent span (the R∩W∩P term), at or past the forced
    /// span (R2's clip), and past the row itself (the clamp to the row).
    #[test]
    fn a_prompt_overlapping_the_recent_and_forced_spans_matches_the_definition() -> Result<()> {
        let narrow: Vec<usize> = (0..16).map(|i| 60 + i * 4).collect();
        // The same rows repeated past `SPLIT_BLOCKS`, so the single pass runs
        // them too.
        let wide: Vec<usize> = (0..512).map(|i| 60 + (i % 16) * 4).collect();
        for (qpos, spread) in [(narrow, 3), (wide, 0)] {
            for recent in [Recent::Candidate, Recent::Forced] {
                for (quantize, seed) in [(None, 0x81), (Some(0.5), 0x82)] {
                    kernel_matches(&KernelCase {
                        qpos: qpos.clone(),
                        top_k: 8,
                        quantize,
                        seed,
                        strata: strata(8, 12, recent),
                        prompt_blocks: if spread == 0 { 20 } else { 0 },
                        prompt_spread: spread,
                        pins: Vec::new(),
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Many narrow windows on a few rows: each window runs as one block of its
    /// own, unsplit, and the merge only gathers them.
    #[test]
    fn windows_run_side_by_side_and_merge_to_the_definition() -> Result<()> {
        let few: Vec<usize> = (0..4).map(|i| 200 + i * 37).collect();
        let narrow = strata(4, 6, Recent::Candidate);
        let Some(_device) = cuda() else { return Ok(()) };
        // The deepest row, qpos 311, has 78 candidates.
        assert_eq!(split_parts(4, 78, &narrow, 8), 1);
        for (recent, quantize, seed) in [
            (Recent::Candidate, None, 0x77),
            (Recent::Candidate, Some(0.5), 0x78),
            (Recent::Forced, Some(0.5), 0x79),
        ] {
            kernel_matches(&KernelCase {
                qpos: few.clone(),
                top_k: 8,
                quantize,
                seed,
                strata: strata(4, 6, recent),
                prompt_blocks: 5,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    #[test]
    fn the_single_pass_matches_the_definition_whole_and_stratified() -> Result<()> {
        let wide: Vec<usize> = (0..512).map(|i| 200 + i).collect();
        for (strata, quantize, seed) in [
            (Strata::WHOLE, None, 0x71),
            (Strata::WHOLE, Some(0.5), 0x72),
            (strata(32, 12, Recent::Candidate), None, 0x73),
            (strata(32, 12, Recent::Candidate), Some(0.5), 0x74),
            (strata(32, 12, Recent::Forced), Some(0.5), 0x75),
        ] {
            kernel_matches(&KernelCase {
                qpos: wide.clone(),
                top_k: 8,
                quantize,
                seed,
                strata,
                prompt_blocks: 5,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    /// The single pass at the released budget — 513 survivors a window, past
    /// half the buffer, so every chunk past the first few trims.
    #[test]
    fn the_single_pass_matches_at_the_released_budget() -> Result<()> {
        let wide: Vec<usize> = (0..512).map(|i| 65_536 + i * 3).collect();
        for strata in [Strata::WHOLE, strata(4096, 512, Recent::Forced)] {
            kernel_matches(&KernelCase {
                qpos: wide.clone(),
                top_k: 2048,
                quantize: None,
                seed: 0x76,
                strata,
                prompt_blocks: 500,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }

    /// The deployed geometry at the depth that motivated it: the released
    /// budget of 2048 positions at ratio 4, ~294K positions deep, cut into
    /// 128K-position windows with an 8K recent span and a 2K-block prompt.
    #[test]
    fn stratified_selection_kernel_matches_at_the_deployed_depth() -> Result<()> {
        for recent in [Recent::Candidate, Recent::Forced] {
            kernel_matches(&KernelCase {
                qpos: (0..4).map(|i| 294_000 + i * 5).collect(),
                top_k: 2048,
                quantize: None,
                seed: 0x68,
                strata: strata(32_768, 2048, recent),
                prompt_blocks: 2000,
                prompt_spread: 0,
                pins: Vec::new(),
            })?;
        }
        Ok(())
    }
}
