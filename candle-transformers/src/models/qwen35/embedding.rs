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
use candle::quantized::gguf_file::{Content, TensorInfo};
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
    /// The widest `token_embd.weight` among the files this load has open.
    ///
    /// **The embedding is the one tensor where taking the wider copy is nearly free.** It is
    /// host-resident by construction — gathered from a `cuMemHostAlloc` mapping, never made
    /// resident — so its width costs host RAM and no VRAM at all. The trunk descends a quant
    /// ladder to fit the card; the MTP sidecar is published at a single quant and does not. Below
    /// the ladder's upper rungs the sidecar's copy is therefore the better one, and it is already
    /// on disk because the drafter needs the file regardless.
    ///
    /// It pays in two places that are easy to miss. The draft head steps by
    /// `embed(argmax(...))`, so embedding error feeds the proposals whose acceptance rate *is*
    /// the throughput — this is a speed change, not only a quality one. And a 248,320-row
    /// vocabulary has a long rare-token tail whose rows saw the fewest updates and are the
    /// worst-conditioned in the table, which is exactly where the narrow quant hurts.
    ///
    /// `sources[0]` is the checkpoint and decides the table's **shape**; a later source is
    /// considered only if its table is the same `[vocab, hidden]`, because a sidecar converted
    /// against a different vocabulary would gather the wrong rows rather than fail. Ties keep the
    /// earlier source, so the checkpoint wins where the two are equal and is displaced only by a
    /// genuine improvement.
    pub fn widest_host_mapped(sources: &[(&Content, &memmap2::Mmap)]) -> Option<HostEmbedding> {
        let contents: Vec<&Content> = sources.iter().map(|(c, _)| *c).collect();
        let (content, mmap) = sources[widest_embedding(&contents)?];
        Self::host_mapped(content, mmap)
    }

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

/// Which of `sources` carries the best `token_embd.weight`, by index.
///
/// Split out from [`EmbeddingTable::widest_host_mapped`] because the choice is header arithmetic
/// and the mapping is not: this runs anywhere, the pinning it feeds needs a CUDA device and a
/// file on disk.
///
/// `sources[0]` is the checkpoint and fixes the table's **shape**. A later source is considered
/// only if its table is the same `[vocab, hidden]` — a sidecar converted against a different
/// vocabulary would gather the wrong rows rather than fail, which is the one way this can be
/// silently wrong. Ties keep the earlier source, so the checkpoint is displaced only by a genuine
/// improvement.
fn widest_embedding(sources: &[&Content]) -> Option<usize> {
    fn info(c: &Content) -> Option<&TensorInfo> {
        c.tensor_infos.get("token_embd.weight")
    }
    let first = info(sources.first()?)?;
    let shape = first.shape.dims();
    let mut best = 0usize;
    let mut best_bpw = first.ggml_dtype.bits_per_weight();
    for (i, c) in sources.iter().enumerate().skip(1) {
        let Some(t) = info(c) else { continue };
        if t.shape.dims() != shape {
            continue;
        }
        let bpw = t.ggml_dtype.bits_per_weight();
        if bpw > best_bpw {
            best = i;
            best_bpw = bpw;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::VersionedMagic;
    use candle::quantized::GgmlDType;
    use candle::Shape;
    use std::collections::HashMap;

    /// A header carrying nothing but a `token_embd.weight` of the given form.
    fn header(dtype: GgmlDType, dims: &[usize]) -> Content {
        let mut tensor_infos = HashMap::new();
        tensor_infos.insert(
            "token_embd.weight".to_string(),
            TensorInfo {
                ggml_dtype: dtype,
                shape: Shape::from_dims(dims),
                offset: 0,
            },
        );
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos,
            tensor_data_offset: 0,
        }
    }

    const V: &[usize] = &[248320, 5120];

    /// The 27B's actual case: a Q3_K_M trunk beside the sidecar's Q4_0 copy.
    #[test]
    fn a_wider_sidecar_displaces_the_checkpoint() {
        let trunk = header(GgmlDType::Q3_K, V);
        let side = header(GgmlDType::Q4_0, V);
        assert_eq!(widest_embedding(&[&trunk, &side]), Some(1));
    }

    /// Upper rungs of the same ladder: the trunk is already wider, so the sidecar is ignored.
    #[test]
    fn a_narrower_sidecar_is_ignored() {
        let trunk = header(GgmlDType::Q6_K, V);
        let side = header(GgmlDType::Q4_0, V);
        assert_eq!(widest_embedding(&[&trunk, &side]), Some(0));
    }

    /// Equal width keeps the checkpoint — the two formats are not interchangeable at equal bits
    /// (Q4_K's per-sub-block scales beat Q4_0's single one), and the checkpoint is the copy the
    /// rest of the model was converted with.
    #[test]
    fn an_equal_width_sidecar_does_not_displace_the_checkpoint() {
        let trunk = header(GgmlDType::Q4_K, V);
        let side = header(GgmlDType::Q4_0, V);
        assert_eq!(GgmlDType::Q4_K.bits_per_weight(), 4.5);
        assert_eq!(GgmlDType::Q4_0.bits_per_weight(), 4.5);
        assert_eq!(widest_embedding(&[&trunk, &side]), Some(0));
    }

    /// **A different vocabulary is refused, not preferred.** This is the only way the choice can
    /// be silently wrong: a wider table of the wrong shape would gather the wrong rows for every
    /// token, and nothing downstream checks a row index against a vocabulary it did not choose.
    #[test]
    fn a_sidecar_with_another_vocabulary_is_never_taken() {
        let trunk = header(GgmlDType::Q3_K, V);
        for dims in [&[151936, 5120][..], &[248320, 4096][..], &[248320][..]] {
            let side = header(GgmlDType::Q8_0, dims);
            assert_eq!(widest_embedding(&[&trunk, &side]), Some(0), "{dims:?}");
        }
    }

    /// A sidecar without the tensor at all, and a checkpoint without it.
    #[test]
    fn a_source_with_no_table_is_skipped() {
        let trunk = header(GgmlDType::Q3_K, V);
        let empty = Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        };
        assert_eq!(widest_embedding(&[&trunk, &empty]), Some(0));
        // No table in the checkpoint is no table at all: the shape it would be checked against
        // does not exist, so there is nothing to prefer.
        assert_eq!(widest_embedding(&[&empty, &trunk]), None);
        assert_eq!(widest_embedding(&[]), None);
    }

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
