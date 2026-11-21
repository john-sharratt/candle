//! Tests that two-stage selection recovers the full Indexer top-k and holds its
//! bounds as the corpus grows.
//!
//! With the shortlist covering the whole corpus, two-stage selection is the full
//! top-k, and with a modest shortlist over query-correlated data it still recovers
//! that set exactly. The causal and batched causal selects are bit-identical to
//! selecting on a gallery holding only the visible prefix. The attended set stays at
//! the top-k while the corpus grows two hundred times, and a spilled corpus still
//! selects exactly.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the BDP sign-pack kernel: the dim loops carry the kernel's
    // own bit positions, so they stay index-based.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    pub(in super::super) const HD: usize = 512;

    pub(in super::super) const IH: usize = 128;

    pub(in super::super) fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Full 32-bit draw: `>> 33` would keep only 31 bits and map every
        // value into [-1, 0) — all-negative data zeroes every sign bit and
        // silently blinds the sign-agreement recall.
        (((*seed >> 32) as u32 as f64) / (u32::MAX as f64) * 2.0 - 1.0) as f32
    }

    pub(in super::super) fn rows(n: usize, d: usize, seed: &mut u64) -> Vec<f32> {
        (0..n * d).map(|_| lcg(seed)).collect()
    }

    /// Gate (c): with the shortlist covering everything, the two-stage
    /// selection IS the full Indexer top-k; with a modest shortlist over
    /// structured (query-correlated) data it still recovers it exactly.
    #[test]
    #[ignore]
    fn two_stage_equals_full_indexer() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 5u64;
        let g_total = 500usize;
        let n_heads = 4usize;
        let top_k = 8usize;

        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        // A trained Indexer space makes relevance ≈ direction and its heads
        // are largely redundant on relevant content: model the query as a
        // shared direction plus per-head jitter, and plant relevant entries
        // along that shared direction. (Planting along the mean of fully
        // independent heads caps sign agreement at ~2/3 — not what a learned
        // relevance space looks like.)
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals.clone(), (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        let mut keys = rows(g_total, IH, &mut s);
        for (j, &e) in [7usize, 42, 137, 260, 401, 444, 471, 490]
            .iter()
            .enumerate()
        {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * (2.0 + j as f32 * 0.1) + lcg(&mut s) * 0.1;
            }
        }
        let attn = rows(g_total, HD, &mut s);
        let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();
        gal.append_batch(
            &Tensor::from_vec(attn, (g_total, HD), &dev)?,
            &Tensor::from_vec(keys, (g_total, IH), &dev)?,
            &pos,
        )?;

        let full = gal.full_indexer_top_k(&q, &weights, top_k)?;

        // Diagnostic: planted-entry agreement counts vs the random field.
        {
            let q_signs = sign_pack(&q)?;
            let counts = bdp_recall(&q_signs, &gal.packed_signs()?, IH)?.to_vec1::<u32>()?;
            let planted = [7usize, 42, 137, 260, 401, 444, 471, 490];
            let pc: Vec<u32> = planted.iter().map(|&e| counts[e]).collect();
            let mut rest: Vec<u32> = counts
                .iter()
                .enumerate()
                .filter(|(i, _)| !planted.contains(i))
                .map(|(_, &c)| c)
                .collect();
            rest.sort_unstable_by(|a, b| b.cmp(a));
            eprintln!("[diag] planted counts = {pc:?}");
            eprintln!("[diag] top random counts = {:?}", &rest[..8]);
        }

        // M = everything → identical by construction.
        let (sel_t, k) = gal.two_stage_select(&q, &weights, g_total, top_k)?;
        assert_eq!(k, top_k);
        let sel = sel_t.to_vec1::<u32>()?;
        assert_eq!(sel, full, "M=len must equal full top-k (two-stage {sel:?})");

        // Modest shortlist on structured data → still exact.
        let (sel_t, k) = gal.two_stage_select(&q, &weights, 64, top_k)?;
        assert_eq!(k, top_k);
        assert_eq!(
            sel_t.to_vec1::<u32>()?,
            full,
            "M=64 on structured corpus must recover the full top-k"
        );
        Ok(())
    }

    /// `two_stage_select_causal(.., n_causal = K)` over an N-entry gallery must be
    /// bit-identical to `two_stage_select` over a gallery holding only the first
    /// K of the same entries — the causal-prefix bound the batched prefill relies
    /// on (each entry's sign-pack / key is independent of the ones after it).
    #[test]
    #[ignore]
    fn two_stage_select_causal_matches_prefix_gallery() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 17u64;
        let n_full = 300usize;
        let k_causal = 173usize; // arbitrary non-boundary prefix
        let n_heads = 4usize;
        let top_k = 8usize;

        // Shared-direction query with per-head jitter (as in two_stage_equals_full).
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals, (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        // Plant a few relevant entries INSIDE the causal prefix so the shortlist
        // is non-trivial and exact.
        let mut keys = rows(n_full, IH, &mut s);
        for (j, &e) in [3usize, 40, 88, 150, 170].iter().enumerate() {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * (2.0 + j as f32 * 0.1) + lcg(&mut s) * 0.1;
            }
        }
        let attn = rows(n_full, HD, &mut s);
        let pos: Vec<u32> = (0..n_full as u32).map(|i| i * 4).collect();

        let mut gal_full = FloatGallery::new(&dev, HD, IH, 8)?;
        gal_full.append_batch(
            &Tensor::from_vec(attn.clone(), (n_full, HD), &dev)?,
            &Tensor::from_vec(keys.clone(), (n_full, IH), &dev)?,
            &pos,
        )?;

        // Prefix gallery: only the first K entries (identical bytes).
        let mut gal_prefix = FloatGallery::new(&dev, HD, IH, 8)?;
        gal_prefix.append_batch(
            &Tensor::from_vec(attn[..k_causal * HD].to_vec(), (k_causal, HD), &dev)?,
            &Tensor::from_vec(keys[..k_causal * IH].to_vec(), (k_causal, IH), &dev)?,
            &pos[..k_causal],
        )?;

        for &m in &[64usize, 1024] {
            let (a_t, ak) = gal_full.two_stage_select_causal(&q, &weights, m, top_k, k_causal)?;
            let (b_t, bk) = gal_prefix.two_stage_select(&q, &weights, m, top_k)?;
            assert_eq!(ak, bk, "k mismatch (m={m})");
            assert_eq!(
                a_t.to_vec1::<u32>()?,
                b_t.to_vec1::<u32>()?,
                "causal(K) over N-gallery must equal full over K-gallery (m={m})"
            );
        }
        Ok(())
    }

    /// Whole-prompt [`FloatGallery::batched_causal_select`] must be bit-identical
    /// to the per-token [`FloatGallery::two_stage_select_causal`] loop it replaces
    /// in the in-regime case (widest window ≤ shortlist, so two-stage == full
    /// top-k), across per-token causal bounds that grow with position.
    #[test]
    #[ignore]
    fn batched_causal_select_matches_per_token() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 23u64;
        let n_corpus = 200usize; // ≤ 1024 → in-regime (exact) path
        let n_tok = 40usize;
        let n_heads = 4usize;
        let top_k = 8usize;

        // Corpus keys, a few planted along a shared direction so top-k is
        // non-trivial; per-token queries share that direction with jitter.
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let mut keys = rows(n_corpus, IH, &mut s);
        for (j, &e) in [3usize, 30, 77, 120, 160].iter().enumerate() {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * (2.0 + j as f32 * 0.1) + lcg(&mut s) * 0.1;
            }
        }
        let attn = rows(n_corpus, HD, &mut s);
        let pos: Vec<u32> = (0..n_corpus as u32).map(|i| i * 4).collect();
        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        gal.append_batch(
            &Tensor::from_vec(attn, (n_corpus, HD), &dev)?,
            &Tensor::from_vec(keys, (n_corpus, IH), &dev)?,
            &pos,
        )?;

        let mut qv = Vec::with_capacity(n_tok * n_heads * IH);
        for _ in 0..n_tok {
            for _ in 0..n_heads {
                for d in 0..IH {
                    qv.push(shared[d] + lcg(&mut s) * 0.5);
                }
            }
        }
        let q_idx = Tensor::from_vec(qv, (n_tok, n_heads, IH), &dev)?;
        let wv: Vec<f32> = (0..n_tok * n_heads).map(|_| 0.5 + lcg(&mut s)).collect();
        let weights = Tensor::from_vec(wv, (n_tok, n_heads), &dev)?;

        // Causal bound grows with position (groups completing), capped at n_corpus.
        let n_visible: Vec<usize> = (0..n_tok).map(|t| ((t + 1) * 5).min(n_corpus)).collect();

        let (comp_idx, comp_cnt, _n) =
            gal.batched_causal_select_device(&q_idx, &weights, &n_visible, top_k)?;
        let comp_idx = comp_idx.to_vec2::<u32>()?;
        let comp_cnt = comp_cnt.to_vec1::<u32>()?;

        for t in 0..n_tok {
            let qt = q_idx.narrow(0, t, 1)?.reshape((n_heads, IH))?;
            let wt = weights.narrow(0, t, 1)?.reshape(n_heads)?;
            // top_m = n_corpus ≥ n_visible[t] → the m≥n (full top-k) branch.
            let (gids_t, k) =
                gal.two_stage_select_causal(&qt, &wt, n_corpus, top_k, n_visible[t])?;
            let expect: Vec<u32> = if k == 0 {
                Vec::new()
            } else {
                gids_t.to_vec1::<u32>()?
            };
            // The device select returns absolute gids ascending, MAX-padded past
            // `comp_cnt`; the first `comp_cnt[t]` must equal the per-token oracle.
            let got: Vec<u32> = comp_idx[t][..comp_cnt[t] as usize].to_vec();
            assert_eq!(comp_cnt[t] as usize, expect.len(), "token {t} count");
            assert_eq!(got, expect, "token {t} (n_visible={})", n_visible[t]);
        }
        Ok(())
    }

    /// Recall sweep on the structured corpus: the sign top-M contains the
    /// float top-k with recall → 1 as M grows (printed for inspection).
    #[test]
    #[ignore]
    fn recall_sweep_synthetic() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 9u64;
        let g_total = 2000usize;
        let n_heads = 4usize;
        let top_k = 8usize;

        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        // Shared-direction query with per-head jitter (see
        // `two_stage_equals_full_indexer`) — relevant entries planted along
        // it with moderate noise, so the recall curve has a realistic shape.
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals.clone(), (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;
        let mut keys = rows(g_total, IH, &mut s);
        for e in (0..g_total).step_by(97) {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * 1.5 + lcg(&mut s) * 0.5;
            }
        }
        let attn = rows(g_total, HD, &mut s);
        let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();
        gal.append_batch(
            &Tensor::from_vec(attn, (g_total, HD), &dev)?,
            &Tensor::from_vec(keys, (g_total, IH), &dev)?,
            &pos,
        )?;

        let full = gal.full_indexer_top_k(&q, &weights, top_k)?;
        let q_signs = sign_pack(&q)?;
        let counts = bdp_recall(&q_signs, &gal.packed_signs()?, IH)?;
        let bins = n_heads * IH + 1;
        let mut last_recall = 0.0;
        for m in [16usize, 32, 64, 128, 256, 512, 1024, 2000] {
            let ids = topm_select(&counts, m, bins)?.to_vec1::<u32>()?;
            let short: std::collections::HashSet<u32> = ids.into_iter().collect();
            let hit = full.iter().filter(|g| short.contains(g)).count();
            last_recall = hit as f32 / full.len() as f32;
            eprintln!("[recall] M={m:5}: {hit}/{} = {last_recall:.3}", full.len());
        }
        assert!(
            (last_recall - 1.0).abs() < f32::EPSILON,
            "M=len must have recall 1.0"
        );
        Ok(())
    }

    /// §L footprint-flat (attended set, runtime): as the corpus grows from 1k
    /// to 200k entries, what each query ACTUALLY ATTENDS stays bounded — the
    /// two-stage selector rescores at most `top_m ≤ 1024` shortlist keys and
    /// returns exactly `top_k` entries, regardless of corpus size. That fixed
    /// attended set — not the growing corpus — is the O(1)-error budget. The
    /// resident BDP-scan bytes (`packed_signs`) are `sign_words·4 B` per entry
    /// (index_head_dim/32 words), a tiny constant-per-entry that stays the
    /// minority of the corpus (vs `head_dim` attended + `index_head_dim`
    /// scoring floats) — the part that must remain resident is bounded small,
    /// the rest (`attn`/`keys`) is spillable and gathered per query at the same
    /// fixed shortlist/top-k width at any depth.
    #[test]
    #[ignore]
    fn attended_set_bounded_as_corpus_grows() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 21u64;
        let n_heads = 4usize;
        let top_k = 8usize;
        let top_m = 512usize;

        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals, (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        let n_sign_words = sign_words(IH);
        for &g_total in &[1_000usize, 50_000, 200_000] {
            let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
            let keys = rows(g_total, IH, &mut s);
            let attn = rows(g_total, HD, &mut s);
            let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();
            gal.append_batch(
                &Tensor::from_vec(attn, (g_total, HD), &dev)?,
                &Tensor::from_vec(keys, (g_total, IH), &dev)?,
                &pos,
            )?;

            let (sel_t, k) = gal.two_stage_select(&q, &weights, top_m, top_k)?;
            // The ATTENDED set is exactly top_k — flat across a 200× corpus.
            assert_eq!(k, top_k, "selected count must stay top_k at N={g_total}");
            assert_eq!(
                sel_t.dim(0)?,
                top_k,
                "selection tensor width must stay top_k at N={g_total}"
            );

            // Resident BDP bytes grow linearly but at the tiny sign rate; the
            // spillable float pair dwarfs it — so the must-stay-resident share
            // is a bounded-small fraction that shrinks as head_dim dominates.
            let resident_sign_bytes = g_total * n_sign_words * 4;
            let spillable_float_bytes = g_total * (HD + IH) * 4;
            assert!(
                resident_sign_bytes * 20 < spillable_float_bytes,
                "resident sign bytes ({resident_sign_bytes}) must stay a small \
                 fraction of the spillable float pair ({spillable_float_bytes}) at N={g_total}"
            );
        }
        Ok(())
    }

    /// §L(b) resident-corpus spill: past `HOT_ENTRY_CAP` the float pair moves
    /// to CPU RAM (the sign/pos index stays on the GPU), and selection is still
    /// EXACT — the two-stage top-k over a spilled corpus reproduces the full
    /// Indexer oracle, proving the on-demand gather feeds the kernel the same
    /// entries it would have found fully-resident.
    #[test]
    #[ignore]
    fn spilled_corpus_selects_exactly() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 33u64;
        let g_total = HOT_ENTRY_CAP + 4_000; // straddle the spill threshold
        let n_heads = 4usize;
        let top_k = 8usize;

        // Shared-direction query; plant relevant entries along it at known ids.
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals, (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        // Plant exactly top_k STRONGLY-aligned entries (some before, some after
        // the spill boundary) so the top-k is unambiguous — dominating the
        // random field — and both the oracle and the two-stage path must return
        // precisely this set. Strong alignment also guarantees BDP recall.
        let planted = [
            11usize,
            900,
            4_242,
            7_000,
            HOT_ENTRY_CAP + 7,
            HOT_ENTRY_CAP + 500,
            HOT_ENTRY_CAP + 1_500,
            HOT_ENTRY_CAP + 3_900,
        ];
        assert_eq!(planted.len(), top_k);
        let mut keys = rows(g_total, IH, &mut s);
        for &e in &planted {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * 6.0 + lcg(&mut s) * 0.05;
            }
        }
        let attn = rows(g_total, HD, &mut s);
        let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();

        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        // Append in two batches straddling the threshold so the spill fires
        // mid-stream, exactly as a growing conversation would trip it.
        let half = g_total / 2;
        gal.append_batch(
            &Tensor::from_vec(attn[..half * HD].to_vec(), (half, HD), &dev)?,
            &Tensor::from_vec(keys[..half * IH].to_vec(), (half, IH), &dev)?,
            &pos[..half],
        )?;
        gal.append_batch(
            &Tensor::from_vec(attn[half * HD..].to_vec(), (g_total - half, HD), &dev)?,
            &Tensor::from_vec(keys[half * IH..].to_vec(), (g_total - half, IH), &dev)?,
            &pos[half..],
        )?;

        // The corpus spilled; the index stayed resident.
        assert!(
            gal.is_spilled(),
            "corpus past HOT_ENTRY_CAP must have spilled"
        );
        assert!(
            !gal.keys.device().is_cuda(),
            "keys must be on CPU when spilled"
        );
        assert!(
            !gal.nope_i8.device().is_cuda()
                && !gal.nope_scale.device().is_cuda()
                && !gal.rope_bf.device().is_cuda(),
            "the two-region cache must be on CPU when spilled (bounded hot VRAM)"
        );
        assert!(gal.signs.device().is_cuda(), "signs must stay GPU-resident");
        assert!(gal.pos.device().is_cuda(), "pos must stay GPU-resident");

        // Selection is EXACT over the spilled corpus: the dominant planted set
        // is what both the oracle and the two-stage path return.
        let mut expected: Vec<u32> = planted.iter().map(|&e| e as u32).collect();
        expected.sort_unstable();
        let full = gal.full_indexer_top_k(&q, &weights, top_k)?;
        assert_eq!(full, expected, "oracle must return the planted top-k");
        let (sel_t, k) = gal.two_stage_select(&q, &weights, 1024, top_k)?;
        assert_eq!(k, top_k);
        assert_eq!(
            sel_t.to_vec1::<u32>()?,
            expected,
            "spilled two-stage top-k must equal the planted set (== oracle)"
        );

        // gather_selected feeds the reference path a compacted GPU pair of the
        // dense f32, reconstructed from the (spilled) two-region cache.
        let (comp, comp_pos) = gal.gather_selected(&sel_t)?;
        assert_eq!(comp.dims(), &[top_k, HD]);
        assert_eq!(comp_pos.dims(), &[top_k]);
        assert!(comp.device().is_cuda() && comp_pos.device().is_cuda());

        // gather_corpus re-heats the bounded two-region selection from the
        // spilled RAM tier back onto the GPU (the hot decode/prefill path).
        let (ni8, nsc, rbf, cpos) = gal.gather_corpus(&sel_t)?;
        assert_eq!(ni8.dims(), &[top_k, NOPE_DIM]);
        assert_eq!(nsc.dims(), &[top_k, NOPE_BANDS]);
        assert_eq!(rbf.dims(), &[top_k, ROPE_DIM]);
        assert_eq!(cpos.dims(), &[top_k]);
        assert!(
            ni8.device().is_cuda()
                && nsc.device().is_cuda()
                && rbf.device().is_cuda()
                && cpos.device().is_cuda(),
            "gather_corpus must return GPU tensors even when spilled"
        );
        Ok(())
    }
}
