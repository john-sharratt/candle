//! The durable form of a gallery, saved as bytes and rebuilt on resume.
//!
//! A snapshot stores the latent and the scoring keys but no positions, because a
//! group's start position is a function of its order and the layer's ratio, and it
//! is reconstructed where the turn lands. The packed sign index is rebuilt from the
//! keys, giving the same bytes the live append produced. Decoding refuses a foreign
//! magic, a geometry mismatch or a truncated payload, and returns nothing rather
//! than a partial gallery. A restored gallery honours the hot and warm bound exactly
//! as a live one does.

use super::*;

/// A turn's compressed corpus in **native durable form** (Artifact C of
/// docs/deepseek/deepseek_turn_seal_persistence.md): the two-region attended cache
/// (`nope_i8`/`nope_scale`/`rope_bf`) and the Indexer scoring `keys`, all
/// host-resident and self-describing. This is what the seal persists (no
/// re-quant — these already are the QAT storage precision) and what resume
/// injects back into a [`FloatGallery`].
///
/// **No positions are stored.** Group-start positions are a pure function of
/// entry order and the layer's compression ratio (`pos[i] = i · ratio`,
/// turn-relative), so they are *reconstructed* at inject time against the
/// position the turn lands at in the reconstructed context — exactly as the
/// chunked KV cache derives positions from cumulative layout rather than
/// persisting an absolute per token. Persisting the original absolute `pos`
/// would be a stale second source of truth that provenance re-layout
/// invalidates. The packed `signs` index is likewise rebuilt from `keys`
/// (`sign_pack`). There is no f32 copy of the rows to store: the gallery holds
/// only the two-region cache, and `gather_selected` reconstructs f32 rows from
/// it for the reference callers.
#[derive(Clone, Debug, PartialEq)]
pub struct CorpusSnapshot {
    pub index_head_dim: usize,
    pub len: usize,
    pub nope_i8: Vec<u8>,     // len * NOPE_DIM
    pub nope_scale: Vec<f32>, // len * NOPE_BANDS
    pub rope_bf: Vec<u16>,    // len * ROPE_DIM  (bf16 bit patterns)
    pub keys: Vec<f32>,       // len * index_head_dim
}

impl CorpusSnapshot {
    pub(super) const MAGIC: &'static [u8; 4] = b"DSC2";

    /// Serialize to a self-contained little-endian blob. Layout: magic(4) ·
    /// index_head_dim/len/nope_dim/nope_bands/rope_dim (5×u32) · nope_i8 bytes ·
    /// nope_scale f32 · rope_bf u16 · keys f32. Positions are NOT stored (they
    /// are reconstructed at inject time — see the type docs).
    pub fn encode(&self) -> Vec<u8> {
        let ih = self.index_head_dim;
        let len = self.len;
        let mut out =
            Vec::with_capacity(24 + len * (NOPE_DIM + NOPE_BANDS * 4 + ROPE_DIM * 2 + ih * 4));
        out.extend_from_slice(Self::MAGIC);
        for v in [
            ih as u32,
            len as u32,
            NOPE_DIM as u32,
            NOPE_BANDS as u32,
            ROPE_DIM as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.nope_i8);
        for &v in &self.nope_scale {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.rope_bf {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.keys {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::encode`]. Returns `None` on foreign magic, geometry
    /// mismatch (a different single-latent shape), or a truncated payload.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 || &bytes[0..4] != Self::MAGIC {
            return None;
        }
        let rd_u32 =
            |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        let ih = rd_u32(4);
        let len = rd_u32(8);
        if rd_u32(12) != NOPE_DIM || rd_u32(16) != NOPE_BANDS || rd_u32(20) != ROPE_DIM {
            return None;
        }
        let mut off = 24;
        let take = |off: &mut usize, n: usize| -> Option<&[u8]> {
            let s = bytes.get(*off..*off + n)?;
            *off += n;
            Some(s)
        };
        let nope_i8 = take(&mut off, len * NOPE_DIM)?.to_vec();
        let nope_scale: Vec<f32> = take(&mut off, len * NOPE_BANDS * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let rope_bf: Vec<u16> = take(&mut off, len * ROPE_DIM * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        let keys: Vec<f32> = take(&mut off, len * ih * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Some(Self {
            index_head_dim: ih,
            len,
            nope_i8,
            nope_scale,
            rope_bf,
            keys,
        })
    }
}

impl FloatGallery {
    /// Snapshot the live corpus into its native durable form
    /// ([`CorpusSnapshot`]) — the seal artifact. Reads the `len` live rows of
    /// the two-region cache + keys + pos back to host (tier-aware: the buffers
    /// may be on CPU once spilled). The signs index is omitted: restore
    /// rebuilds it from the keys.
    #[cfg(feature = "cuda")]
    pub fn snapshot(&self) -> Result<CorpusSnapshot> {
        let len = self.len;
        if len == 0 {
            return Ok(CorpusSnapshot {
                index_head_dim: self.index_head_dim,
                len: 0,
                nope_i8: Vec::new(),
                nope_scale: Vec::new(),
                rope_bf: Vec::new(),
                keys: Vec::new(),
            });
        }
        let nope_i8 = self
            .nope_i8
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let nope_scale = self
            .nope_scale
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rope_bf: Vec<u16> = self
            .rope_bf
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(|b| b.to_bits())
            .collect();
        let keys = self
            .keys
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(CorpusSnapshot {
            index_head_dim: self.index_head_dim,
            len,
            nope_i8,
            nope_scale,
            rope_bf,
            keys,
        })
    }

    /// Rebuild a gallery from a [`CorpusSnapshot`] — the resume injection. The
    /// two-region cache + keys are stamped directly (no re-quant); the sign
    /// index is repacked from `keys` (bit-identical to the live append, which
    /// packs the same keys).
    ///
    /// `positions` (length `snap.len`) are the RECONSTRUCTED group-start
    /// positions the injected entries take in the resumed context — the caller
    /// computes them from where the turn lands in the reconstruction and the
    /// layer's compression ratio (`base + i·ratio`), NOT from any stored value.
    /// Given identical positions, the restored gallery is byte-identical to the
    /// live one on the durable set (`nope_i8`/`nope_scale`/`rope_bf`/`keys`/
    /// `pos`/`signs`).
    #[cfg(feature = "cuda")]
    pub fn from_snapshot(
        device: &Device,
        snap: &CorpusSnapshot,
        positions: &[u32],
    ) -> Result<Self> {
        let ih = snap.index_head_dim;
        if snap.len == 0 {
            return Self::new(device, NOPE_DIM + ROPE_DIM, ih, 1);
        }
        let len = snap.len;
        if positions.len() != len {
            candle::bail!(
                "from_snapshot: {} positions for {} entries",
                positions.len(),
                len
            );
        }
        let cap = len;
        let nope_i8 = Tensor::from_vec(snap.nope_i8.clone(), (len, NOPE_DIM), device)?;
        let nope_scale = Tensor::from_vec(snap.nope_scale.clone(), (len, NOPE_BANDS), device)?;
        let rope_bf = Tensor::from_vec(
            snap.rope_bf
                .iter()
                .map(|&b| half::bf16::from_bits(b))
                .collect::<Vec<_>>(),
            (len, ROPE_DIM),
            device,
        )?;
        let keys = Tensor::from_vec(snap.keys.clone(), (len, ih), device)?;
        let pos = Tensor::from_vec(positions.to_vec(), len, device)?;
        // Signs are a deterministic function of the keys — repack on the GPU
        // (identical to the live append's `sign_pack`).
        let signs = sign_pack(&keys)?;
        let mut g = Self {
            keys,
            nope_i8,
            nope_scale,
            rope_bf,
            signs,
            pos,
            len,
            cap,
            head_dim: NOPE_DIM + ROPE_DIM,
            index_head_dim: ih,
            device: device.clone(),
            spilled: false,
            region_ptr_cache: std::cell::Cell::new(None),
        };
        // Honor the hot/warm tier bound exactly as a live gallery would: past
        // HOT_ENTRY_CAP the float pair lives in CPU RAM (signs/pos stay GPU).
        g.maybe_spill(len)?;
        Ok(g)
    }
}
