//! Tests that the batched assemble and the cross-sequence pool match the
//! per-sequence paths.
//!
//! The batched assemble must produce the same pooling rows and the same retained
//! state for every sequence, including staggered carried state and a second round
//! that continues from the first. A pool over every sequence's groups in one call
//! equals pooling each sequence on its own, split back bit for bit, with ragged
//! lengths and both the overlapping and non-overlapping shapes.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use candle::Device;

    /// [`assemble_groups_batched`] over a staggered fleet is byte-identical,
    /// per sequence, to [`IncrementalCompressor::assemble_groups`] called per
    /// seq — pools, positions, AND the retained state (proven behaviorally: a
    /// second per-seq round through both paths must also match, so the carried
    /// buffer / overlap prev / group counter advanced identically).
    pub(in super::super) fn assemble_batched_matches_per_seq_case(
        ratio: usize,
        d: usize,
    ) -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (coff * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, coff * d), &dev)?,
            Tensor::randn(0f32, 1.0, d, &dev)?,
            ratio,
            d,
            2,
            1e-6,
        );
        // Staggered carried state per seq: 0 rows (fresh), a partial group, and
        // past a completed group (carried prev + partial buffer).
        let seeds = [0usize, ratio - 1, ratio + 2];
        // Round-1 row counts: complete ≥1 group, complete none, land mid-group.
        let lens1 = [2 * ratio + 1, 1, ratio];
        let lens2 = [ratio, 2 * ratio - 1, ratio + 1];

        let mut inc_a: Vec<IncrementalCompressor> = Vec::new(); // per-seq path
        let mut inc_b: Vec<IncrementalCompressor> = Vec::new(); // batched path
        for &p in &seeds {
            let mut i = c.incremental();
            if p > 0 {
                let x = Tensor::randn(0f32, 1.0, (p, dim), &dev)?;
                let (kv, sc) = c.project_rows(&x)?;
                i.emit_groups_projected(&kv, &sc, None)?;
            }
            inc_b.push(c.incremental());
            inc_b.last_mut().unwrap().state_restore(i.state_snapshot());
            inc_a.push(i);
        }

        // The two paths describe the same rows with DIFFERENT segment shapes —
        // the streamer one segment per row, the batched form two wide ones — so
        // the comparison is over the pooling rows themselves (`materialize`),
        // which is exactly the block both used to concatenate.
        let assert_pools = |a: &Option<GroupPool>, b: &Option<GroupPool>, tag: &str| match (a, b) {
            (None, None) => {}
            (Some(ga), Some(gb)) => {
                assert_eq!(ga.positions, gb.positions, "{tag}: positions");
                let ma = ga.materialize().unwrap();
                let mb = gb.materialize().unwrap();
                assert_eq!(ma.len(), mb.len(), "{tag}: group count");
                for (g, ((ka, sa), (kb, sb))) in ma.iter().zip(mb.iter()).enumerate() {
                    let f = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert_eq!(f(ka), f(kb), "{tag}: group {g} pool_kv bytes");
                    assert_eq!(f(sa), f(sb), "{tag}: group {g} pool_score bytes");
                }
            }
            _ => panic!("{tag}: one path emitted a pool, the other did not"),
        };

        for (round, lens) in [(1, &lens1), (2, &lens2)] {
            let rows: Vec<(Tensor, Tensor)> = lens
                .iter()
                .map(|&n| {
                    let x = Tensor::randn(0f32, 1.0, (n, dim), &dev)?;
                    c.project_rows(&x)
                })
                .collect::<Result<_>>()?;
            // Per-seq reference.
            let gps_a: Vec<Option<GroupPool>> = inc_a
                .iter_mut()
                .zip(&rows)
                .map(|(i, (kv, sc))| i.assemble_groups(kv, sc))
                .collect::<Result<_>>()?;
            // Batched.
            let inputs: Vec<SeqAssemble> = inc_b
                .iter()
                .zip(&rows)
                .map(|(i, (kv, sc))| i.assemble_input(kv, sc))
                .collect::<Result<_>>()?;
            let outs = assemble_groups_batched(&inc_b[0], &inputs)?;
            for (i, (gp_b, upd)) in outs.into_iter().enumerate() {
                assert_pools(
                    &gps_a[i],
                    &gp_b,
                    &format!("ratio={ratio} round={round} seq={i}"),
                );
                inc_b[i].assemble_apply(upd);
            }
        }
        Ok(())
    }

    #[test]
    fn assemble_batched_matches_per_seq() -> Result<()> {
        assemble_batched_matches_per_seq_case(4, 6)?; // overlapping
        assemble_batched_matches_per_seq_case(3, 5)?; // non-overlapping
        Ok(())
    }

    /// The prefill wave's cross-sequence emit batch: assembling each sequence's
    /// groups independently, then concatenating their [`GroupPool`]s on the group
    /// axis and calling [`Compressor::pool_and_norm`] ONCE, is BIT-IDENTICAL to
    /// pooling each sequence separately (the pool is per-group-independent). Covers
    /// overlap + non-overlap, raw + roped, and ragged per-sequence lengths.
    #[test]
    fn pool_batched_across_seqs_matches_per_seq() -> Result<()> {
        let dev = Device::Cpu;
        let (d, rd, dim) = (6usize, 4usize, 8usize);
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
                // Ragged: two sequences of different lengths, each completing ≥1 group.
                let lens = [ratio * 3 + 2, ratio * 2 + 1];
                let mut per_seq: Vec<Tensor> = Vec::new();
                let mut pools: Vec<GroupPool> = Vec::new();
                for &len in &lens {
                    let x = Tensor::randn(0f32, 1.0, (len, dim), &dev)?;
                    let (kv, sc) = c.project_rows(&x)?;
                    let gp = c
                        .incremental()
                        .assemble_groups(&kv, &sc)?
                        .expect("completes a group");
                    per_seq.push(c.pool_and_norm(&[&gp], rope_arg, &scope)?);
                    pools.push(gp);
                }
                // Batched: every sequence's pool in ONE call — a wider descriptor
                // table, not a concatenation of their rows.
                let refs: Vec<&GroupPool> = pools.iter().collect();
                let batched = c.pool_and_norm(&refs, rope_arg, &scope)?;
                // Split back per sequence and compare bit-for-bit.
                let mut off = 0usize;
                for (i, gp) in pools.iter().enumerate() {
                    let g = gp.positions.len();
                    let got = batched.narrow(0, off, g)?.flatten_all()?.to_vec1::<f32>()?;
                    off += g;
                    let want = per_seq[i].flatten_all()?.to_vec1::<f32>()?;
                    for (j, (x, y)) in want.iter().zip(&got).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "seq {i} elem {j} (ratio={ratio}, roped={roped})"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
