//! Holds a checked snapshot of a sequence's recurrent state until a store is
//! wanted, and fingerprints the layout so a foreign snapshot is refused.
//!
//! A snapshot's rows return to the host pool when its last holder drops it. The
//! schedule hash is a dependency-free identity check over the layer schedule and
//! the DeltaNet dimensions, so a snapshot taken under one model is never scattered
//! into another. States cross as little-endian floats.

use super::*;

/// A checked snapshot held for a sequence that has no store yet — installed
/// into one when the sequence's state is first wanted. The layers are shared:
/// every slot opened on one prompt branch holds the same rows.
#[derive(Debug, Clone)]
pub struct StagedSnapshot {
    pub schedule_hash: u64,
    pub layers: Arc<[ExportedLayerState]>,
}

impl StagedSnapshot {
    /// A store of `layer_kinds`/`dims` holding this snapshot, marked seeded so
    /// its first wave keeps it.
    pub fn build(
        &self,
        layer_kinds: &[LayerKind],
        dims: &DeltaNetDims,
        device: &Device,
    ) -> Result<RecurrentStateStore> {
        let mut store = RecurrentStateStore::new(layer_kinds, dims, device)?;
        store.import(self.schedule_hash, &self.layers)?;
        Ok(store)
    }
}

impl Drop for StagedSnapshot {
    /// The last holder of the rows returns them to [`host_rows`], for the next
    /// decode to fill, instead of freeing them where the snapshot is dropped —
    /// inside a reprojection's swap or a store's build, on the scheduler thread.
    fn drop(&mut self) {
        if let Some(layers) = Arc::get_mut(&mut self.layers) {
            for l in layers.iter_mut() {
                host_rows::give(std::mem::take(&mut l.state));
                host_rows::give(std::mem::take(&mut l.conv_tail));
            }
        }
    }
}

/// Fingerprint of a model's recurrent layout: the layer schedule plus the
/// DeltaNet dims. A snapshot taken under one hash must never be scattered
/// into a store built under another — resume recomputes instead.
pub fn schedule_hash(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> u64 {
    // FNV-1a: stable, dependency-free, and this is an identity check, not
    // crypto.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |b: u64| {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(layer_kinds.len() as u64);
    for (i, k) in layer_kinds.iter().enumerate() {
        mix(i as u64);
        mix(match k {
            LayerKind::DeltaNet => 1,
            LayerKind::Attention => 2,
        });
    }
    mix(dims.head_dim as u64);
    mix(dims.n_k_heads as u64);
    mix(dims.n_v_heads as u64);
    mix(dims.conv_kernel as u64);
    h
}

/// `v` as little-endian bytes — the snapshot's wire form of a state tensor.
///
/// Written into a pre-sized buffer a fixed four bytes at a time, which the compiler
/// turns into a straight copy. The `flat_map(to_le_bytes).collect()` this replaced
/// could neither pre-size nor vectorise, and it ran over every float of every layer
/// on every turn seal — ~16 M floats on the 35B-A3B, on the scheduler's loop thread.
pub(super) fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 4];
    for (dst, f) in out.chunks_exact_mut(4).zip(v) {
        dst.copy_from_slice(&f.to_le_bytes());
    }
    out
}
