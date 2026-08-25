//! Where the token-embedding table lives, and what fetching a row costs.
//!
//! The table is the worst VRAM-per-access ratio in the model — `vocab × hidden`
//! resident, one row read per token — so this lineage has always kept it off the
//! card. The question this type answers is *how*, because the obvious way is not
//! free either.
//!
//! [`EmbeddingTable::Host`] is that obvious way: dequantize once into an F32
//! tensor on the CPU and `index_select` it. It costs `vocab × hidden × 4` bytes
//! of host RAM (2.4 GiB on the 9B's 151936 × 4096) and, worse, a **readback**:
//! the ids are a device tensor by the time the embedding is wanted, so fetching
//! a row means draining the pipeline at the one point the forward cannot proceed
//! without the answer.
//!
//! [`EmbeddingTable::HostMapped`] keeps the rows *quantized* in a
//! `cuMemHostAlloc(DEVICEMAP)` buffer and lets the GPU gather them across PCIe
//! from device-side ids. It is better on both axes at once:
//!
//! * **RAM** — the quantized table, not the F32 one: 600 MiB against 2.4 GiB.
//! * **Bus** — what crosses PCIe is a quantized row (3360 B at Q6_K × 4096
//!   against 16 KiB of F32), and the widening happens on the far side because
//!   the dequantize emits the residual stream's dtype directly.
//! * **Sync** — none. The ids are read where they already are.
//!
//! So `Host` is not a mode to choose; it is what is left when the allocation
//! cannot be made — a CPU device, or a host too short of pinnable memory.
//!
//! The drafter is what made the sync matter enough to fix. A NextN block needs
//! `embed(argmax(...))` to take its next step, so under `Host` every drafted
//! token pays a full device→host→device barrier against roughly a millisecond of
//! actual block arithmetic. Under `HostMapped` the whole draft block runs on the
//! device timeline and reads back once, at the end, when the caller needs the
//! ids as tokens.

use candle::cuda_backend::Backing;
use candle::quantized::gguf_file::Content;
use candle::{DType, Device, Result, Tensor};

use crate::models::host_embedding::HostEmbedding;
use crate::models::operand_guard::expect_dtype;

/// A token-embedding table, and the residency that decides how a row is read.
pub enum EmbeddingTable {
    /// Quantized rows in host-mapped memory, gathered by the GPU over PCIe.
    HostMapped(HostEmbedding),
    /// Dequantized `[vocab, hidden]` F32 on the host, gathered by the CPU.
    Host(Tensor),
}

impl EmbeddingTable {
    /// Bind `token_embd.weight` to host-mapped memory, or `None` if it cannot be
    /// pinned.
    ///
    /// There is no capacity gate here, unlike the dense-model policy in
    /// [`crate::models::host_embedding::should_serve_from_host`]. That policy
    /// weighs host residency against keeping the table in VRAM, and the trade
    /// turns on how tight the card is. Here the alternative is not VRAM — it is
    /// [`Self::Host`], which this beats on RAM, on bus bytes, and on
    /// synchronisation simultaneously. A choice with no losing axis is not a
    /// choice.
    pub fn host_mapped(content: &Content, mmap: &memmap2::Mmap) -> Option<HostEmbedding> {
        let info = content.tensor_infos.get("token_embd.weight")?;
        // `[vocab, hidden]`: a row is one token's embedding, so `rows` is the
        // vocabulary and `cols` the hidden size. Inverting them makes `cols` the
        // vocabulary, which fails the whole-blocks check rather than gathering
        // garbage — but fails it at load, for the wrong reason.
        let (rows, cols) = match info.shape.dims() {
            [r, c] => (*r, *c),
            _ => return None,
        };
        let byte_offset = content.tensor_data_offset as usize + info.offset as usize;
        match HostEmbedding::new(mmap, byte_offset, info.ggml_dtype, rows, cols) {
            Ok(h) => {
                tracing::info!(
                    target: "candle_transformers::qwen35",
                    vocab = rows,
                    hidden = cols,
                    dtype = ?info.ggml_dtype,
                    pinned_mib = (rows * h.layout().row_bytes) >> 20,
                    "embedding host-mapped; the forward's embed no longer synchronises"
                );
                Some(h)
            }
            // Not GPU-reachable — a CPU device, or `cuMemHostAlloc` refused the
            // size. Fall back to the F32 host table rather than failing the load.
            Err(e) => {
                tracing::warn!(
                    target: "candle_transformers::qwen35",
                    "embedding could not be host-mapped ({e}); gathering it on the CPU"
                );
                None
            }
        }
    }

    /// Hidden size — the table's column count.
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::HostMapped(h) => h.layout().ncols,
            Self::Host(t) => t.dims().last().copied().unwrap_or(0),
        }
    }

    /// Gather the rows named by `ids` as `[n, hidden]` of `dtype` on `device`.
    ///
    /// `ids` may be of any integer dtype, any shape, and either side of the bus;
    /// it is read flat, and moved to whichever side its residency reads from.
    /// Both directions are cheap in the way the naive version is not, because
    /// what moves is **4 bytes per token** rather than a row per token: the
    /// wave packs its ids on the host, so `HostMapped` uploads 3.6 KiB for a
    /// 905-token prefill and the GPU then pulls 2.9 MiB of quantized rows out of
    /// pinned memory itself.
    ///
    /// The upload is not a synchronisation — nothing waits on a result — which
    /// is the whole difference from `Host`'s `to_device(Cpu)` below.
    ///
    /// `staging` is where the host-mapped path carves the gathered quantized
    /// bytes from — a wave span when one is open, since those bytes are written
    /// by the gather and read by the dequantize on the next line and are dead
    /// immediately after.
    ///
    /// The result is `dtype` with no conversion after the fact: the dequantize
    /// emits it (CLAUDE.md invariant 1), and under `Host` the CPU-side cast
    /// happens before the upload so the narrower type is what crosses the bus.
    pub fn rows(
        &self,
        ids: &Tensor,
        device: &Device,
        staging: Backing,
        dtype: DType,
    ) -> Result<Tensor> {
        // Validated, not converted. Token ids are U32 everywhere they are
        // produced — the scheduler's `input_ids`, the draft's argmax — so a cast
        // here would be a full pass over the ids, per gather, guarding against a
        // producer that does not exist.
        expect_dtype(ids, DType::U32, "embedding ids")?;
        let flat = ids.flatten_all()?;
        match self {
            Self::HostMapped(h) => h.embed(&flat.to_device(device)?, device, staging, dtype),
            Self::Host(table) => {
                // The readback this residency cannot avoid: the CPU gather needs
                // the ids where the CPU can see them, and if they are on the
                // device that means draining the pipeline to fetch them.
                let host_ids = flat.to_device(&Device::Cpu)?;
                table
                    .index_select(&host_ids, 0)?
                    .to_dtype(dtype)?
                    .to_device(device)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(vocab: usize, hidden: usize) -> Result<EmbeddingTable> {
        let n = vocab * hidden;
        let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        Ok(EmbeddingTable::Host(Tensor::from_vec(
            vals,
            (vocab, hidden),
            &Device::Cpu,
        )?))
    }

    /// The width a caller reshapes by comes from the table, so it must be the
    /// column count under either residency — not the row count, which is the
    /// vocabulary and the classic way to get this transposed.
    #[test]
    fn hidden_size_is_the_column_count() -> Result<()> {
        assert_eq!(table(17, 8)?.hidden_size(), 8);
        Ok(())
    }

    /// Repeats and a non-monotonic order, so a gather that ignored the ids and
    /// copied a contiguous span cannot pass.
    #[test]
    fn rows_follow_the_ids() -> Result<()> {
        let t = table(6, 4)?;
        let ids = Tensor::new([3u32, 0, 3, 5].as_slice(), &Device::Cpu)?;
        let got = t
            .rows(&ids, &Device::Cpu, Backing::Owned, DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], vec![3.0, 3.5, 4.0, 4.5]);
        assert_eq!(got[1], vec![-3.0, -2.5, -2.0, -1.5]);
        assert_eq!(got[2], got[0]);
        assert_eq!(got[3], vec![7.0, 7.5, 8.0, 8.5]);
        Ok(())
    }

    /// **Shape is not part of the contract — only the order is.** The wave
    /// packs `[1, total]`, the drafter's argmax is `[n]`, and both are read
    /// flat.
    #[test]
    fn ids_are_read_flat_whatever_their_shape() -> Result<()> {
        let t = table(6, 4)?;
        let ids = Tensor::from_vec(vec![3u32, 0, 3, 5], (2, 2), &Device::Cpu)?;
        let got = t.rows(&ids, &Device::Cpu, Backing::Owned, DType::F32)?;
        assert_eq!(got.dims(), &[4, 4]);
        assert_eq!(got.to_vec2::<f32>()?[3], vec![7.0, 7.5, 8.0, 8.5]);
        Ok(())
    }

    /// **The dtype IS part of the contract, and a wrong one is refused.**
    ///
    /// This test used to assert the opposite — that `i64` ids were quietly
    /// retyped — which made the gather carry a full pass over the ids on every
    /// wave to serve a producer that does not exist: every token source in this
    /// engine holds `u32` (the scheduler's `&[u32]`, the drafter's argmax).
    /// Validating states the requirement where a cast only hid it
    /// (CLAUDE.md invariant 1b).
    #[test]
    fn ids_of_the_wrong_type_are_refused_rather_than_retyped() -> Result<()> {
        let t = table(6, 4)?;
        let ids = Tensor::from_vec(vec![3i64, 0], (2,), &Device::Cpu)?;
        let err = t
            .rows(&ids, &Device::Cpu, Backing::Owned, DType::F32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("I64") && err.contains("U32"), "{err}");
        Ok(())
    }

    /// Ids reach this from wherever the caller had them — the wave packs its
    /// own on the host, the drafter's are a device argmax — so `rows` places
    /// them, and a caller that has to know which side to be on would have to
    /// know the residency too.
    #[test]
    fn ids_are_placed_for_the_residency() -> Result<()> {
        let t = table(6, 4)?;
        let ids = Tensor::new([5u32, 1].as_slice(), &Device::Cpu)?;
        let got = t.rows(&ids, &Device::Cpu, Backing::Owned, DType::F32)?;
        assert_eq!(got.to_vec2::<f32>()?[0], vec![7.0, 7.5, 8.0, 8.5]);
        Ok(())
    }

    /// The requested dtype is what comes back — the property that lets every
    /// caller drop its own `to_dtype` pass.
    #[test]
    fn rows_emit_the_requested_dtype() -> Result<()> {
        let t = table(6, 4)?;
        let ids = Tensor::new([2u32].as_slice(), &Device::Cpu)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let got = t.rows(&ids, &Device::Cpu, Backing::Owned, dtype)?;
            assert_eq!(got.dtype(), dtype);
        }
        Ok(())
    }
}
