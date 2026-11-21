//! Tests for the gallery snapshot codec and its restore.
//!
//! The codec's header is checked as raw little-endian bytes, and a foreign magic, a
//! geometry mismatch and a truncated payload are each refused. A live gallery that
//! is snapshotted and restored at a new base is byte-identical on its durable set,
//! while its positions are reconstructed at the new base rather than taken from the
//! snapshot.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the BDP sign-pack kernel: the dim loops carry the kernel's
    // own bit positions, so they stay index-based.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use super::super::tests_selection::tests::{rows, HD, IH};

    /// `CorpusSnapshot` byte codec: encode → decode is the identity, and the
    /// header bytes are exactly the documented little-endian layout (magic +
    /// 5×u32 geometry). Pure host logic — asserts raw bytes, not tolerances.
    #[test]
    fn corpus_snapshot_codec_round_trip() -> Result<()> {
        let ih = IH; // 128
        let len = 3usize;
        let snap = CorpusSnapshot {
            index_head_dim: ih,
            len,
            nope_i8: (0..len * NOPE_DIM).map(|i| (i % 251) as u8).collect(),
            nope_scale: (0..len * NOPE_BANDS)
                .map(|i| i as f32 * 0.5 - 3.0)
                .collect(),
            rope_bf: (0..len * ROPE_DIM).map(|i| (i * 7 + 1) as u16).collect(),
            keys: (0..len * ih).map(|i| (i as f32).sin()).collect(),
        };
        let bytes = snap.encode();

        // Exact header: "DSC2" · ih · len · NOPE_DIM · NOPE_BANDS · ROPE_DIM.
        assert_eq!(&bytes[0..4], b"DSC2");
        assert_eq!(&bytes[4..8], &(ih as u32).to_le_bytes());
        assert_eq!(&bytes[8..12], &(len as u32).to_le_bytes());
        assert_eq!(&bytes[12..16], &(NOPE_DIM as u32).to_le_bytes());
        assert_eq!(&bytes[16..20], &(NOPE_BANDS as u32).to_le_bytes());
        assert_eq!(&bytes[20..24], &(ROPE_DIM as u32).to_le_bytes());
        // Total length: header + each region's exact byte count (NO positions).
        let expect_len = 24 + len * (NOPE_DIM + NOPE_BANDS * 4 + ROPE_DIM * 2 + ih * 4);
        assert_eq!(bytes.len(), expect_len);

        let back = CorpusSnapshot::decode(&bytes).expect("decode");
        assert_eq!(back, snap);

        // Foreign magic / geometry mismatch → None.
        assert!(CorpusSnapshot::decode(b"XXXX....................").is_none());
        let mut bad = bytes.clone();
        bad[12] ^= 0xff; // corrupt NOPE_DIM
        assert!(CorpusSnapshot::decode(&bad).is_none());
        // Truncated payload → None.
        assert!(CorpusSnapshot::decode(&bytes[..bytes.len() - 1]).is_none());

        // Empty corpus round-trips too.
        let empty = CorpusSnapshot {
            index_head_dim: ih,
            len: 0,
            nope_i8: Vec::new(),
            nope_scale: Vec::new(),
            rope_bf: Vec::new(),
            keys: Vec::new(),
        };
        assert_eq!(CorpusSnapshot::decode(&empty.encode()).unwrap(), empty);
        Ok(())
    }

    /// Gate (Artifact C): a live gallery snapshotted and restored is
    /// byte-identical on the native durable set — the two-region cache, keys,
    /// AND the repacked sign index — independent of positions, and the injected
    /// positions are RECONSTRUCTED at a fresh base (`base + i·ratio`), NOT taken
    /// from the snapshot (which stores none). Straddles a growth (4 → 16).
    #[test]
    #[ignore]
    fn gallery_snapshot_restore_round_trip() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let ratio = 4u32; // CSA
        let mut g = FloatGallery::new(&dev, HD, IH, 4)?;
        let mut s = 77u64;
        // Live append uses the true group-start positions i·ratio.
        let mut appended = 0u32;
        for n in [3usize, 9] {
            let attn = rows(n, HD, &mut s);
            let keys = rows(n, IH, &mut s);
            let pos: Vec<u32> = (0..n as u32).map(|i| (appended + i) * ratio).collect();
            appended += n as u32;
            g.append_batch(
                &Tensor::from_vec(attn, (n, HD), &dev)?,
                &Tensor::from_vec(keys, (n, IH), &dev)?,
                &pos,
            )?;
        }
        let snap = g.snapshot()?;
        assert_eq!(snap.len, g.len());

        // Byte-exact codec on real data, then restore from the decoded blob at a
        // NEW absolute base (the resume frame) — positions are reconstructed.
        let blob = snap.encode();
        let snap2 = CorpusSnapshot::decode(&blob).expect("decode");
        assert_eq!(snap2, snap);
        let new_base = 10_000u32;
        let recon_pos: Vec<u32> = (0..snap2.len as u32)
            .map(|i| new_base + i * ratio)
            .collect();
        let r = FloatGallery::from_snapshot(&dev, &snap2, &recon_pos)?;
        assert_eq!(r.len(), g.len());

        // Durable buffers must match bit-for-bit (independent of positions).
        let cmp_u8 = |a: &Tensor, b: &Tensor, label: &str| -> Result<()> {
            let av = a.flatten_all()?.to_vec1::<u8>()?;
            let bv = b.flatten_all()?.to_vec1::<u8>()?;
            assert_eq!(av, bv, "{label}");
            Ok(())
        };
        let live = g.len();
        cmp_u8(
            &g.nope_i8.narrow(0, 0, live)?.to_device(&Device::Cpu)?,
            &r.nope_i8.narrow(0, 0, live)?.to_device(&Device::Cpu)?,
            "nope_i8",
        )?;
        let f_ns_g = g
            .nope_scale
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let f_ns_r = r
            .nope_scale
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (&a, &b)) in f_ns_g.iter().zip(&f_ns_r).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "nope_scale[{i}]");
        }
        let rope_g = g
            .rope_bf
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        let rope_r = r
            .rope_bf
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        for (i, (&a, &b)) in rope_g.iter().zip(&rope_r).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "rope_bf[{i}]");
        }
        let keys_g = g.scoring_keys()?.flatten_all()?.to_vec1::<f32>()?;
        let keys_r = r.scoring_keys()?.flatten_all()?.to_vec1::<f32>()?;
        for (i, (&a, &b)) in keys_g.iter().zip(&keys_r).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "keys[{i}]");
        }
        // Positions are the reconstructed ones (shifted to the new base), NOT
        // the originals.
        assert_eq!(
            r.positions()?.to_vec1::<u32>()?,
            recon_pos,
            "reconstructed pos"
        );
        assert_eq!(
            g.packed_signs()?.flatten_all()?.to_vec1::<u32>()?,
            r.packed_signs()?.flatten_all()?.to_vec1::<u32>()?,
            "signs (repacked from keys)"
        );
        Ok(())
    }
}
