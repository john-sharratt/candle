//! Tests that the streaming compressor matches the prefill compressor.
//!
//! Streamed groups are numerically equal to the prefill forward over the same
//! prefix, for overlapping and non-overlapping compressors. Batched emission is
//! bit-identical to streaming the same rows, including a call that splits a group.
//! A snapshot, a rejected draft tail and a restore leave the compressor exactly as
//! if the tail had never been absorbed. Decode's batched pool across sessions is
//! held to a tolerance, since the per-row and pooled softmaxes reduce in different
//! orders, while the batch width itself changes nothing.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use candle::{Device, IndexOp};

    /// The streaming (decode) compressor emits, group-by-group, entries numerically equal to
    /// the prefill `forward` over the full prefix — the mandatory prefill/decode equivalence
    /// (docs/deepseek/deepseek_v4_flash.md §2.2). Exercises both the overlapping (`ratio == 4`) and
    /// non-overlapping compressors, across several complete groups plus trailing tokens.
    pub(in super::super) fn incremental_matches_prefill_case(
        ratio: usize,
        d: usize,
        rd: usize,
        s: usize,
    ) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;

        // Oracle: prefill over the whole prefix → [1, groups, d].
        let prefill = c.forward(&x, &rope)?.unwrap();
        let groups = s / ratio;
        assert_eq!(prefill.dim(1)?, groups);

        // Stream one token at a time; collect the emitted per-group entries.
        let mut inc = c.incremental();
        let mut emitted: Vec<Tensor> = Vec::new();
        for t in 0..s {
            let row = x.i((0, t))?; // [dim]
            if let Some(entry) = inc.push(&row, &rope)? {
                emitted.push(entry); // [1, 1, d]
            }
        }
        assert_eq!(emitted.len(), groups, "entry count (ratio={ratio})");
        let streamed = Tensor::cat(&emitted, 1)?; // [1, groups, d]

        let a = prefill.flatten_all()?.to_vec1::<f32>()?;
        let b = streamed.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "prefill vs streamed diverge (ratio={ratio}): max|Δ| = {max_abs}"
        );
        Ok(())
    }

    #[test]
    fn incremental_matches_prefill_overlap() -> Result<()> {
        // ratio 4 (overlapping): 3 complete groups + 2 trailing window tokens.
        incremental_matches_prefill_case(4, 6, 4, 14)
    }

    #[test]
    fn incremental_matches_prefill_nonoverlap() -> Result<()> {
        // ratio 3 (non-overlapping): 4 complete groups + 1 trailing token.
        incremental_matches_prefill_case(3, 5, 2, 13)
    }

    /// [`IncrementalCompressor::emit_groups_projected`] (batched, carried-state
    /// aware) must be bit-identical to streaming the same pre-projected rows one
    /// at a time through `push_projected` (raw) / `push_projected_roped` (roped) —
    /// including across a call boundary that splits a group (the resume-mid-group
    /// case a prefill at `base > 0` hits), for both the overlapping and
    /// non-overlapping compressors.
    pub(in super::super) fn emit_groups_batched_matches_streamed_case(
        ratio: usize,
        d: usize,
        rd: usize,
        s: usize,
        split: usize,
    ) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        let x = Tensor::randn(0f32, 1.0, (s, dim), &dev)?; // [s, dim]
        let (kv_all, score_all) = c.project_rows(&x)?; // [s, cd]

        let chunks = [(0usize, split), (split, s - split)];

        // Streamed vs batched, over both the raw (pre-RoPE) and roped forms.
        for roped in [false, true] {
            let mut inc_s = c.incremental();
            let mut ent_s: Vec<Tensor> = Vec::new();
            let mut pos_s: Vec<u32> = Vec::new();
            for t in 0..s {
                let kv = kv_all.narrow(0, t, 1)?;
                let sc = score_all.narrow(0, t, 1)?;
                if roped {
                    if let Some(e) = inc_s.push_projected_roped(&kv, &sc, &rope)? {
                        ent_s.push(e.reshape((1, d))?);
                    }
                } else if let Some((e, p)) = inc_s.push_projected(&kv, &sc)? {
                    ent_s.push(e.reshape((1, d))?);
                    pos_s.push(p);
                }
            }

            let mut inc_b = c.incremental();
            let mut ent_b: Vec<Tensor> = Vec::new();
            let mut pos_b: Vec<u32> = Vec::new();
            for &(a, len) in &chunks {
                if len == 0 {
                    continue;
                }
                let kv = kv_all.narrow(0, a, len)?;
                let sc = score_all.narrow(0, a, len)?;
                let rope_arg = if roped { Some(&rope) } else { None };
                if let Some((e, p)) = inc_b.emit_groups_projected(&kv, &sc, rope_arg)? {
                    ent_b.push(e);
                    pos_b.extend(p);
                }
            }

            if !roped {
                assert_eq!(pos_s, pos_b, "positions (ratio={ratio}, split={split})");
            }
            let a = Tensor::cat(&ent_s, 0)?.flatten_all()?.to_vec1::<f32>()?;
            let b = Tensor::cat(&ent_b, 0)?.flatten_all()?.to_vec1::<f32>()?;
            let max_abs = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 1e-6,
                "batched vs streamed diverge (ratio={ratio}, split={split}, roped={roped}): max|Δ| = {max_abs}"
            );
        }
        Ok(())
    }

    #[test]
    fn emit_groups_batched_matches_streamed() -> Result<()> {
        // Overlapping (ratio 4): single-shot, split mid-group, split off-boundary,
        // and a first chunk that completes NO group (all buffered → None).
        emit_groups_batched_matches_streamed_case(4, 6, 4, 14, 14)?;
        emit_groups_batched_matches_streamed_case(4, 6, 4, 14, 6)?;
        emit_groups_batched_matches_streamed_case(4, 6, 4, 23, 7)?;
        emit_groups_batched_matches_streamed_case(4, 6, 4, 14, 2)?;
        // Non-overlapping (ratio 3).
        emit_groups_batched_matches_streamed_case(3, 5, 2, 13, 13)?;
        emit_groups_batched_matches_streamed_case(3, 5, 2, 13, 5)?;
        Ok(())
    }

    /// The speculative-verify rollback contract: snapshot → absorb a whole verify
    /// block (accepted prefix + rejected draft tail, crossing a group boundary
    /// INSIDE the block — the corrupting case) → `state_restore` +
    /// `emit_groups_projected` over ONLY the accepted prefix must leave the
    /// compressor in exactly the state of never having seen the rejected tail:
    /// the replay's emissions AND every later emission (positions and bytes)
    /// match a reference stream fed the accepted prefix alone.
    pub(in super::super) fn snapshot_restore_replay_case(
        ratio: usize,
        d: usize,
        rd: usize,
        accepted: usize,
    ) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        // Pre-rows: one full group behind (prev-group overlap populated) + a
        // partial buffer, so the 6-row block crosses the next boundary mid-block.
        let n_pre = 2 * ratio - 1;
        let block = 6usize;
        let tail = 2 * ratio;
        let pre_x = Tensor::randn(0f32, 1.0, (n_pre, dim), &dev)?;
        let blk_x = Tensor::randn(0f32, 1.0, (block, dim), &dev)?;
        let cont_x = Tensor::randn(0f32, 1.0, (tail, dim), &dev)?;
        let (kv_pre, sc_pre) = c.project_rows(&pre_x)?;
        let (kv_blk, sc_blk) = c.project_rows(&blk_x)?;
        let (kv_cont, sc_cont) = c.project_rows(&cont_x)?;

        for roped in [false, true] {
            let rope_arg = if roped { Some(&rope) } else { None };
            // Per-row feed helper (matches the live decode/prefill streaming path).
            let feed = |inc: &mut IncrementalCompressor,
                        kv: &Tensor,
                        sc: &Tensor,
                        range: std::ops::Range<usize>,
                        ent: &mut Vec<Tensor>,
                        pos: &mut Vec<u32>|
             -> Result<()> {
                for t in range {
                    let k = kv.narrow(0, t, 1)?;
                    let s = sc.narrow(0, t, 1)?;
                    if roped {
                        if let Some(e) = inc.push_projected_roped(&k, &s, &rope)? {
                            ent.push(e.reshape((1, d))?);
                        }
                    } else if let Some((e, p)) = inc.push_projected(&k, &s)? {
                        ent.push(e.reshape((1, d))?);
                        pos.push(p);
                    }
                }
                Ok(())
            };

            // Reference: pre + accepted prefix + continuation — the rejected tail
            // never existed. Emissions collected from the accepted prefix onward.
            let mut r = c.incremental();
            let (mut r_ent, mut r_pos) = (Vec::new(), Vec::new());
            feed(
                &mut r,
                &kv_pre,
                &sc_pre,
                0..n_pre,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            feed(
                &mut r,
                &kv_blk,
                &sc_blk,
                0..accepted,
                &mut r_ent,
                &mut r_pos,
            )?;
            feed(&mut r, &kv_cont, &sc_cont, 0..tail, &mut r_ent, &mut r_pos)?;

            // Test: pre → SNAPSHOT → the whole block (draft tail absorbed) →
            // RESTORE → replay the accepted prefix (the rollback's exact call) →
            // continuation.
            let mut t_ = c.incremental();
            let (mut t_ent, mut t_pos) = (Vec::new(), Vec::new());
            feed(
                &mut t_,
                &kv_pre,
                &sc_pre,
                0..n_pre,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            let snap = t_.state_snapshot();
            feed(
                &mut t_,
                &kv_blk,
                &sc_blk,
                0..block,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            t_.state_restore(snap);
            if accepted > 0 {
                if let Some((e, p)) = t_.emit_groups_projected(
                    &kv_blk.narrow(0, 0, accepted)?,
                    &sc_blk.narrow(0, 0, accepted)?,
                    rope_arg,
                )? {
                    let g = e.dim(0)?;
                    for gi in 0..g {
                        t_ent.push(e.narrow(0, gi, 1)?);
                    }
                    if !roped {
                        t_pos.extend(p);
                    }
                }
            }
            feed(&mut t_, &kv_cont, &sc_cont, 0..tail, &mut t_ent, &mut t_pos)?;

            assert_eq!(
                r_ent.len(),
                t_ent.len(),
                "emission count (ratio={ratio}, accepted={accepted}, roped={roped})"
            );
            if !roped {
                assert_eq!(
                    r_pos, t_pos,
                    "group positions (ratio={ratio}, accepted={accepted}) — a mismatch means \
                     group_idx was not rolled back"
                );
            }
            if !r_ent.is_empty() {
                let a = Tensor::cat(&r_ent, 0)?.flatten_all()?.to_vec1::<f32>()?;
                let b = Tensor::cat(&t_ent, 0)?.flatten_all()?.to_vec1::<f32>()?;
                let max_abs = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_abs < 1e-6,
                    "restore+replay diverges from clean absorb \
                     (ratio={ratio}, accepted={accepted}, roped={roped}): max|Δ| = {max_abs}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn snapshot_restore_replay_matches_clean_absorb() -> Result<()> {
        for &ratio in &[4usize, 3] {
            let (d, rd) = if ratio == 4 { (6, 4) } else { (5, 2) };
            // accepted = 0 (all drafts rejected), 1 (boundary completes during
            // replay for ratio 4), mid, and block-1 (max partial).
            for accepted in [0usize, 1, 3, 5] {
                snapshot_restore_replay_case(ratio, d, rd, accepted)?;
            }
        }
        Ok(())
    }

    /// The DECODE wave's cross-session emit batch: every session absorbs ONE row
    /// per step through [`IncrementalCompressor::assemble_groups`] and DEFERS its
    /// pool, then the sessions that completed a group on that step are pooled
    /// TOGETHER in a single [`Compressor::pool_and_norm`] — reproducing, per
    /// session per step, the per-row `push_projected` / `push_projected_roped`
    /// emit it replaces.
    ///
    /// Sessions are staggered so their group boundaries land on different steps —
    /// the ragged case the batch must handle, where only some sessions contribute
    /// a pool on any given step (and, at ratio 4, some steps contribute none).
    /// Running many steps also proves the retained streaming state advanced
    /// identically: a divergence in the carried buffer, the overlap `prev_*` or
    /// `group_idx` would surface as a mismatched entry on a later step.
    ///
    /// Tolerance, not bit-exactness, and only for the reason
    /// `emit_groups_batched_matches_streamed` already records: the per-row emit
    /// softmaxes dim 0 of a `[P, d]` group while the pooled path softmaxes dim 1
    /// of `[G, P, d]`, so the two reductions may differ in the last ULP. That the
    /// BATCH WIDTH itself changes nothing — session `s`'s entry is identical
    /// whether it pools alone or beside 15 others, so a wave's session mix cannot
    /// perturb any session's corpus — is the bit-exact half, held by
    /// `pool_batched_across_seqs_matches_per_seq`.
    #[test]
    fn decode_pool_batched_across_sessions_matches_per_row() -> Result<()> {
        let dev = Device::Cpu;
        let (d, rd, dim) = (6usize, 4usize, 8usize);
        let n_sess = 3usize;
        for &ratio in &[4usize, 3usize] {
            let coff = if ratio == 4 { 2 } else { 1 };
            let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
            let c = Compressor::new(
                Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
                Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
                Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
                Tensor::randn(0f32, 1.0, d, &dev)?,
                ratio,
                d,
                rd,
                1e-6,
            );
            for roped in [false, true] {
                let rope_arg = if roped { Some(&rope) } else { None };
                let scope = desc::scope(&dev)?;
                // Stagger: session `s` enters the loop having already absorbed `s`
                // rows, so the sessions' group boundaries fall on different steps.
                let mut refs: Vec<IncrementalCompressor> = Vec::with_capacity(n_sess);
                let mut bats: Vec<IncrementalCompressor> = Vec::with_capacity(n_sess);
                for s in 0..n_sess {
                    let mut inc = c.incremental();
                    if s > 0 {
                        let x = Tensor::randn(0f32, 1.0, (s, dim), &dev)?;
                        let (kv, sc) = c.project_rows(&x)?;
                        inc.emit_groups_projected(&kv, &sc, rope_arg)?;
                    }
                    let mut b = c.incremental();
                    b.state_restore(inc.state_snapshot());
                    bats.push(b);
                    refs.push(inc);
                }

                let mut want: Vec<Vec<Tensor>> = vec![Vec::new(); n_sess];
                let mut got: Vec<Vec<Tensor>> = vec![Vec::new(); n_sess];
                for _ in 0..(3 * ratio + 2) {
                    // One wave-batched projection, one row per session.
                    let x = Tensor::randn(0f32, 1.0, (n_sess, dim), &dev)?;
                    let (kv, sc) = c.project_rows(&x)?; // [n_sess, cd]

                    // Reference: each row pools inside its own push.
                    for (s, inc) in refs.iter_mut().enumerate() {
                        let (k, v) = (kv.narrow(0, s, 1)?, sc.narrow(0, s, 1)?);
                        if roped {
                            if let Some(e) = inc.push_projected_roped(&k, &v, &rope)? {
                                want[s].push(e.reshape((1, d))?);
                            }
                        } else if let Some((e, _)) = inc.push_projected(&k, &v)? {
                            want[s].push(e.reshape((1, d))?);
                        }
                    }

                    // Batched: assemble every session, then ONE pool over those
                    // that completed a group — the wave's pass 1a / pass 1b.
                    let mut pools: Vec<Option<GroupPool>> = Vec::with_capacity(n_sess);
                    for (s, inc) in bats.iter_mut().enumerate() {
                        pools.push(inc.assemble_row(&kv.narrow(0, s, 1)?, &sc.narrow(0, s, 1)?)?);
                    }
                    let live: Vec<(usize, &GroupPool)> = pools
                        .iter()
                        .enumerate()
                        .filter_map(|(s, g)| g.as_ref().map(|g| (s, g)))
                        .collect();
                    if live.is_empty() {
                        continue;
                    }
                    let refs: Vec<&GroupPool> = live.iter().map(|(_, g)| *g).collect();
                    let pooled = c.pool_and_norm(&refs, rope_arg, &scope)?;
                    let mut off = 0usize;
                    for (s, g) in &live {
                        let n = g.groups();
                        got[*s].push(pooled.narrow(0, off, n)?);
                        off += n;
                    }
                }

                for s in 0..n_sess {
                    assert_eq!(
                        want[s].len(),
                        got[s].len(),
                        "session {s} emitted a different number of groups \
                         (ratio={ratio}, roped={roped})"
                    );
                    assert!(
                        !want[s].is_empty(),
                        "session {s} emitted nothing — the case proves nothing \
                         (ratio={ratio}, roped={roped})"
                    );
                    let a = Tensor::cat(&want[s], 0)?.flatten_all()?.to_vec1::<f32>()?;
                    let b = Tensor::cat(&got[s], 0)?.flatten_all()?.to_vec1::<f32>()?;
                    let max_abs = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        max_abs < 1e-6,
                        "session {s}: batched decode pool vs per-row emit diverge \
                         (ratio={ratio}, roped={roped}): max|Δ| = {max_abs}"
                    );
                }
            }
        }
        Ok(())
    }
}
