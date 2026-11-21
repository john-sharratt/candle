//! Tests for the recall kernels against host reference packing and popcount.
//!
//! The device sign pack and the XNOR-popcount recall are compared bit for bit with
//! host references, including a chain where one kernel's output feeds the next with
//! no host read between. The pack is checked after gallery activity that forces a
//! growth, and a dense view that starts part-way into its storage must pack its own
//! rows, not the first rows of the storage. The argsort probe records a known limit:
//! a single-block argsort above 1024 entries is broken, and the gallery never relies
//! on it past that width.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the BDP sign-pack kernel: the dim loops carry the kernel's
    // own bit positions, so they stay index-based.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use candle::IndexOp;

    use super::super::tests_selection::tests::{rows, HD, IH};

    /// Diagnostic: candle CUDA argsort sanity across widths (stream + width
    /// limits) — the two-stage selector leans on it.
    #[test]
    #[ignore]
    fn argsort_cuda_probe() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for n in [64usize, 500, 1024, 1500, 2000, 4096] {
            let vals: Vec<f32> = (0..n).map(|i| ((i * 7919) % n) as f32).collect();
            let t = Tensor::from_vec(vals.clone(), (1, n), &dev)?;
            let order = t.arg_sort_last_dim(false)?.squeeze(0)?.to_vec1::<u32>()?;
            let max = order.iter().max().copied().unwrap_or(0);
            let uniq: std::collections::HashSet<_> = order.iter().collect();
            eprintln!(
                "[argsort] n={n}: max_idx={max} unique={} first4={:?}",
                uniq.len(),
                &order[..4]
            );
            if n <= 1024 {
                assert!(
                    (max as usize) < n && uniq.len() == n,
                    "argsort broken at n={n} (within its supported width)"
                );
            }
            // n > 1024: known-broken (single-block bitonic) — printed above for
            // visibility; the gallery never relies on it past 1024.
        }
        Ok(())
    }

    /// The recall kernel in isolation: hand-uploaded bit patterns vs a host
    /// XNOR+popcount reference.
    #[test]
    #[ignore]
    fn bdp_recall_matches_host() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 21u64;
        let (h, g, words, dim) = (4usize, 33usize, 4usize, 128usize);
        let qs: Vec<u32> = (0..h * words)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 24) as u32
            })
            .collect();
        let es: Vec<u32> = (0..g * words)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 24) as u32
            })
            .collect();
        let q_t = Tensor::from_vec(qs.clone(), (h, words), &dev)?;
        let e_t = Tensor::from_vec(es.clone(), (g, words), &dev)?;
        let counts = bdp_recall(&q_t, &e_t, dim)?.to_vec1::<u32>()?;
        for e in 0..g {
            let mut expect = 0u32;
            for hh in 0..h {
                for w in 0..words {
                    expect += (!(qs[hh * words + w] ^ es[e * words + w])).count_ones();
                }
            }
            assert_eq!(counts[e], expect, "entry {e}: {} vs {expect}", counts[e]);
        }
        Ok(())
    }

    /// Bisection: does prior gallery activity (allocations, slice_set,
    /// growth, candle matmul) corrupt a subsequent direct sign_pack?
    #[test]
    #[ignore]
    fn pack_after_gallery_activity() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 41u64;
        let check = |label: &str, dev: &Device, s: &mut u64| -> Result<()> {
            let qv = rows(2, IH, s);
            let t = Tensor::from_vec(qv.clone(), (2, IH), dev)?;
            let packed = sign_pack(&t)?.to_vec2::<u32>()?;
            let mut expect0 = [0u32; 4];
            for d in 0..IH {
                if qv[d] >= 0.0 {
                    expect0[d / 32] |= 1 << (d % 32);
                }
            }
            assert_eq!(
                packed[0],
                expect0.to_vec(),
                "sign_pack broken after: {label}"
            );
            Ok(())
        };
        check("nothing", &dev, &mut s)?;

        let mut gal = FloatGallery::new(&dev, HD, IH, 4)?;
        check("gallery alloc", &dev, &mut s)?;

        let attn = rows(3, HD, &mut s);
        let keys = rows(3, IH, &mut s);
        gal.append_batch(
            &Tensor::from_vec(attn, (3, HD), &dev)?,
            &Tensor::from_vec(keys, (3, IH), &dev)?,
            &[0, 4, 8],
        )?;
        check("first append", &dev, &mut s)?;

        // Force growth (4 → 16).
        let attn = rows(9, HD, &mut s);
        let keys = rows(9, IH, &mut s);
        gal.append_batch(
            &Tensor::from_vec(attn, (9, HD), &dev)?,
            &Tensor::from_vec(keys, (9, IH), &dev)?,
            &[12, 16, 20, 24, 28, 32, 36, 40, 44],
        )?;
        check("growth append", &dev, &mut s)?;

        let q = Tensor::from_vec(rows(4, IH, &mut s), (4, IH), &dev)?;
        let w = Tensor::from_vec(vec![1.0f32; 4], 4, &dev)?;
        let _ = gal.full_indexer_top_k(&q, &w, 4)?;
        check("full_indexer (matmul)", &dev, &mut s)?;

        // The failing scale: one big append (500 rows → growth to 512).
        let n = 500usize;
        let attn = rows(n, HD, &mut s);
        let keys = rows(n, IH, &mut s);
        let pos: Vec<u32> = (0..n as u32).map(|i| 48 + i * 4).collect();
        gal.append_batch(
            &Tensor::from_vec(attn, (n, HD), &dev)?,
            &Tensor::from_vec(keys.clone(), (n, IH), &dev)?,
            &pos,
        )?;
        check("big append (500 rows)", &dev, &mut s)?;
        let _ = gal.full_indexer_top_k(&q, &w, 8)?;
        check("full_indexer big", &dev, &mut s)?;

        // And the gallery's own stored signs at scale: entry 100's packed
        // signs must match the host packing of what was appended.
        let got = gal.packed_signs()?.i(12 + 100)?.to_vec1::<u32>()?;
        let mut expect = [0u32; 4];
        for d in 0..IH {
            if keys[100 * IH + d] >= 0.0 {
                expect[d / 32] |= 1 << (d % 32);
            }
        }
        assert_eq!(got, expect.to_vec(), "stored signs at scale");

        // Pack the SAME tensor that already flowed through the matmul (the
        // failing tests reuse `q`; fresh-tensor checks can't see this).
        let qv_host = q.flatten_all()?.to_vec1::<f32>()?;
        let packed_q = sign_pack(&q)?.to_vec2::<u32>()?;
        let mut expect_q0 = [0u32; 4];
        for d in 0..IH {
            if qv_host[d] >= 0.0 {
                expect_q0[d / 32] |= 1 << (d % 32);
            }
        }
        assert_eq!(packed_q[0], expect_q0.to_vec(), "sign_pack of the reused q");
        Ok(())
    }

    /// The pack→recall CHAIN (kernel output feeding the next kernel with no
    /// host read-back in between) vs host reference.
    #[test]
    #[ignore]
    fn pack_recall_chain_matches_host() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 31u64;
        let (h, g) = (4usize, 55usize);
        let qv = rows(h, IH, &mut s);
        let ev = rows(g, IH, &mut s);
        let q_t = Tensor::from_vec(qv.clone(), (h, IH), &dev)?;
        let e_t = Tensor::from_vec(ev.clone(), (g, IH), &dev)?;
        let q_signs = sign_pack(&q_t)?;
        let e_signs = sign_pack(&e_t)?;
        let counts = bdp_recall(&q_signs, &e_signs, IH)?.to_vec1::<u32>()?;
        let pack_host = |v: &[f32], r: usize| -> [u32; 4] {
            let mut out = [0u32; 4];
            for d in 0..IH {
                if v[r * IH + d] >= 0.0 {
                    out[d / 32] |= 1 << (d % 32);
                }
            }
            out
        };
        for e in 0..g {
            let es = pack_host(&ev, e);
            let mut expect = 0u32;
            for hh in 0..h {
                let qs = pack_host(&qv, hh);
                for w in 0..4 {
                    expect += (!(qs[w] ^ es[w])).count_ones();
                }
            }
            assert_eq!(counts[e], expect, "entry {e}");
        }
        Ok(())
    }

    /// The device sign-pack matches host bit packing exactly.
    #[test]
    #[ignore]
    fn sign_pack_matches_host() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 11u64;
        let n = 37usize;
        let vals = rows(n, IH, &mut s);
        let t = Tensor::from_vec(vals.clone(), (n, IH), &dev)?;
        let packed = sign_pack(&t)?.to_vec2::<u32>()?;
        for (r, row) in packed.iter().enumerate() {
            for (w, &word) in row.iter().enumerate() {
                let mut expect = 0u32;
                for b in 0..32 {
                    let d = w * 32 + b;
                    if d < IH && vals[r * IH + d] >= 0.0 {
                        expect |= 1 << b;
                    }
                }
                assert_eq!(word, expect, "row {r} word {w}");
            }
        }
        Ok(())
    }

    /// A dense row view that starts part-way into its storage packs ITS rows,
    /// not the storage's first rows: the launch addresses the view's first
    /// element.
    #[test]
    #[ignore]
    fn sign_pack_of_an_offset_view_packs_the_view() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 29u64;
        let n = 9usize;
        let vals = rows(n, IH, &mut s);
        let all = Tensor::from_vec(vals.clone(), (n, IH), &dev)?;
        let view = all.narrow(0, 4, 3)?;
        assert!(view.is_contiguous() && view.layout().start_offset() == 4 * IH);
        let got = sign_pack(&view)?.to_vec2::<u32>()?;
        let want = sign_pack(&Tensor::from_vec(
            vals[4 * IH..7 * IH].to_vec(),
            (3, IH),
            &dev,
        )?)?
        .to_vec2::<u32>()?;
        assert_eq!(got, want);
        Ok(())
    }
}
