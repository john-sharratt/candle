//! Tests that the fused and batched gathers reproduce the per-region gather.
//!
//! A fused gather of all four regions must equal the per-region `index_select` path
//! bit for bit, at a non-zero destination offset. The single launch over several
//! galleries must place each session's rows at its own offset. A round trip must
//! read the keys and positions back bit for bit across a growth, and the attended
//! rows must equal the exact dequantisation of the gallery's own stored bytes, with
//! no tolerance.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the BDP sign-pack kernel: the dim loops carry the kernel's
    // own bit positions, so they stay index-based.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use candle::IndexOp;

    use super::super::tests_selection::tests::{rows, HD, IH};

    /// The fused `gather_corpus_into` (one kernel launch per session, writing
    /// straight into a shared block at a row offset) must be BIT-IDENTICAL to the
    /// per-region `index_select` path (`gather_corpus`), for every one of the
    /// four hot-cache regions and at a non-zero destination offset.
    #[test]
    #[ignore]
    fn gather_corpus_into_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut seed = 0xC0FFEEu64;
        let n = 200usize;
        let mut g = FloatGallery::new(&dev, HD, IH, 128)?;
        let attn = Tensor::from_vec(rows(n, HD, &mut seed), (n, HD), &dev)?;
        let keys = Tensor::from_vec(rows(n, IH, &mut seed), (n, IH), &dev)?;
        let positions: Vec<u32> = (0..n as u32).map(|i| i * 4).collect();
        g.append_batch(&attn, &keys, &positions)?;

        let gid_v: Vec<u32> = vec![5, 17, 42, 3, 199, 0, 128, 63];
        let k = gid_v.len();
        let gids = Tensor::from_vec(gid_v, k, &dev)?;

        // Reference: the per-region gather.
        let (rn, rs, rr, rp) = g.gather_corpus(&gids)?;

        // Fused: gather into a shared block at offset 2 (test the placement).
        let off = 2usize;
        let total = k + off + 1; // trailing untouched row
        let on = Tensor::zeros((total, NOPE_DIM), DType::U8, &dev)?;
        let os = Tensor::zeros((total, NOPE_BANDS), DType::F32, &dev)?;
        let or_ = Tensor::zeros((total, ROPE_DIM), DType::BF16, &dev)?;
        let op = Tensor::zeros(total, DType::U32, &dev)?;
        let scope = desc::scope(&dev)?;
        g.gather_corpus_into(&gids, &on, &os, &or_, &op, off, &scope)?;

        let got_n = on.narrow(0, off, k)?;
        let got_s = os.narrow(0, off, k)?;
        let got_r = or_.narrow(0, off, k)?;
        let got_p = op.narrow(0, off, k)?;

        let bits_u8 = |t: &Tensor| -> Result<Vec<u8>> { t.flatten_all()?.to_vec1::<u8>() };
        let bits_u32 = |t: &Tensor| -> Result<Vec<u32>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        assert_eq!(bits_u8(&got_n)?, bits_u8(&rn)?, "nope_i8 mismatch");
        assert_eq!(bits_u32(&got_s)?, bits_u32(&rs)?, "nope_scale mismatch");
        assert_eq!(bits_u32(&got_r)?, bits_u32(&rr)?, "rope_bf mismatch");
        assert_eq!(
            got_p.to_vec1::<u32>()?,
            rp.to_vec1::<u32>()?,
            "pos mismatch"
        );
        Ok(())
    }

    /// The single-launch `gather_corpus_batched` across several galleries must be
    /// BIT-IDENTICAL, per session, to the per-session `gather_corpus` — the
    /// device pointer table must address each gallery's rows correctly and each
    /// session's block land at its own output offset.
    #[test]
    #[ignore]
    fn gather_corpus_batched_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut seed = 0xBA7C4EDu64;
        let ks = [7usize, 3, 12, 1, 9];
        let mut galleries: Vec<FloatGallery> = Vec::new();
        let mut gids: Vec<Tensor> = Vec::new();
        let mut refs: Vec<(Tensor, Tensor, Tensor, Tensor)> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut cum = 0u32;
        for (s, &k) in ks.iter().enumerate() {
            let entries = 40 + s * 30;
            let mut g = FloatGallery::new(&dev, HD, IH, 64)?;
            let attn = Tensor::from_vec(rows(entries, HD, &mut seed), (entries, HD), &dev)?;
            let keys = Tensor::from_vec(rows(entries, IH, &mut seed), (entries, IH), &dev)?;
            let positions: Vec<u32> = (0..entries as u32).map(|i| i * 4).collect();
            g.append_batch(&attn, &keys, &positions)?;
            let gv: Vec<u32> = (0..k).map(|j| ((j * 3 + s) % entries) as u32).collect();
            let gt = Tensor::from_vec(gv, k, &dev)?;
            refs.push(g.gather_corpus(&gt)?);
            gids.push(gt);
            offsets.push(cum);
            cum += k as u32;
            galleries.push(g);
        }
        let total = cum as usize;
        let on = Tensor::zeros((total, NOPE_DIM), DType::U8, &dev)?;
        let os = Tensor::zeros((total, NOPE_BANDS), DType::F32, &dev)?;
        let or_ = Tensor::zeros((total, ROPE_DIM), DType::BF16, &dev)?;
        let op = Tensor::zeros(total, DType::U32, &dev)?;
        let grefs: Vec<&FloatGallery> = galleries.iter().collect();
        let scope = desc::scope(&dev)?;
        gather_corpus_batched(&grefs, &gids, &offsets, &on, &os, &or_, &op, &scope)?;

        let bits_u8 = |t: &Tensor| -> Result<Vec<u8>> { t.flatten_all()?.to_vec1::<u8>() };
        let bits_u32 = |t: &Tensor| -> Result<Vec<u32>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        for (s, &k) in ks.iter().enumerate() {
            let o = offsets[s] as usize;
            let (rn, rs, rr, rp) = &refs[s];
            assert_eq!(
                bits_u8(&on.narrow(0, o, k)?)?,
                bits_u8(rn)?,
                "sess {s} nope"
            );
            assert_eq!(
                bits_u32(&os.narrow(0, o, k)?)?,
                bits_u32(rs)?,
                "sess {s} scale"
            );
            assert_eq!(
                bits_u32(&or_.narrow(0, o, k)?)?,
                bits_u32(rr)?,
                "sess {s} rope"
            );
            assert_eq!(
                op.narrow(0, o, k)?.to_vec1::<u32>()?,
                rp.to_vec1::<u32>()?,
                "sess {s} pos"
            );
        }
        Ok(())
    }

    /// Gate (b): `keys`/`pos` read back bit-for-bit across the grow-triggering
    /// second append (lossless tiers), and `attn_entries` reconstructs the dense
    /// f32 EXACTLY as the deterministic two-region dequant of the input (int8
    /// nope · per-band amax ‖ bf16 rope) — the f32 archive is gone, so the
    /// reconstruction equals what the kernels read, asserted against the
    /// recomputed quantized bytes (no tolerance).
    #[test]
    #[ignore]
    fn gallery_round_trip_raw_bytes() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut g = FloatGallery::new(&dev, HD, IH, 4)?;
        let mut s = 3u64;
        // Two appends that force a growth (4 → 16 cap).
        for (n, base) in [(3usize, 0u32), (9, 12)] {
            let attn = rows(n, HD, &mut s);
            let keys = rows(n, IH, &mut s);
            let pos: Vec<u32> = (0..n as u32).map(|i| base + i * 4).collect();
            let attn_t = Tensor::from_vec(attn.clone(), (n, HD), &dev)?;
            let keys_t = Tensor::from_vec(keys.clone(), (n, IH), &dev)?;
            let before = g.len();
            g.append_batch(&attn_t, &keys_t, &pos)?;
            assert_eq!(g.len(), before + n);
            let got_attn = g
                .attn_entries()?
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let got_keys = g
                .scoring_keys()?
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // `attn_entries` reconstructs the dense f32 from the lossy two-region,
            // so it is NOT the input. Its contract is to be the EXACT dequant of
            // the gallery's OWN stored bytes: nope band `= int8 · per-band scale`,
            // rope tail `= bf16 → f32`. Recompute that from the stored tensors and
            // assert bit-for-bit (the accessor reads the right rows post-growth and
            // applies the scale correctly). The rope tail is additionally a true
            // bf16 passthrough of the input. The input→int8 quantization itself is
            // gated bit-exact by the `build_corpus_cache` mirror test.
            let ni8: Vec<u8> = g
                .nope_i8
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<u8>()?;
            let nsc: Vec<f32> = g
                .nope_scale
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for i in 0..n {
                for band in 0..NOPE_BANDS {
                    let scale = nsc[i * NOPE_BANDS + band];
                    for j in 0..SUB_DIM {
                        let d = band * SUB_DIM + j;
                        let expected = (ni8[i * NOPE_DIM + d] as i8 as f32) * scale;
                        assert_eq!(
                            got_attn[i * HD + d].to_bits(),
                            expected.to_bits(),
                            "nope attn[{i}][{d}]"
                        );
                    }
                }
                for d in NOPE_DIM..HD {
                    let expected = half::bf16::from_f32(attn[i * HD + d]).to_f32();
                    assert_eq!(
                        got_attn[i * HD + d].to_bits(),
                        expected.to_bits(),
                        "rope attn[{i}][{d}] not a bf16 passthrough of the input"
                    );
                }
            }
            for (i, (&a, &b)) in keys.iter().zip(&got_keys).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "keys[{i}]");
            }
            let got_pos = g.positions()?.to_vec1::<u32>()?;
            assert_eq!(&got_pos[before..before + n], &pos[..]);
        }
        Ok(())
    }
}
