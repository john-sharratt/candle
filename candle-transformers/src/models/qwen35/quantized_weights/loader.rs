//! The per-tensor reader that turns checkpoint tensors into resident or streamed
//! projections.
//!
//! Each read goes to the source its tensor needs: a donor for a repaired
//! recurrent gate, an override's own file, the checkpoint's mapping read a band
//! at a time, or the device whole. A projection's KO twin is only ever narrowed
//! to the schedule's width, so a schedule cannot inflate a block the checkpoint
//! kept small. Each read is logged before it happens, so an out-of-memory load
//! names the tensor it was reading.

use super::*;

/// `pub` so the layer-streaming loader can drive the same reader the trunk loop
/// does — one transcription of a tensor's name, not two. Its fields stay
/// private, so the only things a caller can do with one are the reads
/// [`load_layer`] and [`load_residue`] expose.
pub struct Loader<'a, R: Read + Seek> {
    pub(super) content: &'a gguf_file::Content,
    pub(super) reader: &'a mut R,
    /// The checkpoint's mapped bytes, when the caller has them — see
    /// [`LoadInputs::map`]. `None` falls back to reading each tensor onto the device whole.
    pub(super) map: Option<&'a [u8]>,
    pub(super) device: Device,
    pub(super) mode: Int8Mode,
    /// Where this load's repacked projections live.
    ///
    /// `Span` for a load whose weights stay — the trunk, the MTP head, the
    /// residues. `Pool` for the pack build, which materialises a layer only to
    /// read it back and drop it, and would otherwise claim dense-block ground
    /// for all 64 layers of a model that by definition does not fit.
    pub(super) residency: WeightResidency,
    /// Checkpoint bytes pulled onto the device so far.
    ///
    /// Summed from the tensors themselves rather than bracketed as a driver
    /// delta: the governor's balloon releases into the CUDA pool, the weights
    /// then allocate out of those cached blocks, and driver-used barely moves —
    /// a delta reads near zero for a model that is really there. Qwen3-MoE
    /// learned this twice, at 3936 MiB and at 733 MiB against a true ~990.
    pub(super) device_bytes: usize,
    /// Narrow this weight's KO twin to [`NARROW_TWIN`], for as long as it is set.
    ///
    /// Set around the two loads that produce **permanently resident** weights on a card that
    /// streams: the output head and the MTP draft block. Everything else — the trunk's layers —
    /// is a slot tenant whose width is already the ladder's business, so narrowing it here
    /// would be deciding the same thing twice.
    ///
    /// State on the loader rather than an argument because those two loads reach `proj` through
    /// a dozen call sites (`attention`, `dense_ffn`, the mixer), and threading a parameter
    /// through all of them to say one thing about two tensors is how a parameter ends up
    /// ignored at the one site that mattered.
    pub(super) narrow: Option<GgmlDType>,
    /// Apply [`streaming_twin`] to the trunk's layers, carrying `num_layers`.
    ///
    /// Separate from [`Self::narrow`] because they answer different questions: `narrow` forces
    /// one width over whatever is being loaded right now (the resident weights), while this is
    /// a per-tensor *schedule* over the streamed trunk. `narrow` wins where both apply, so the
    /// draft block keeps its own answer even though its tensors are named like a layer's.
    pub(super) stream_narrow: Option<usize>,
    /// The donor checkpoint's header and bytes, when this load is repairing a recurrent path.
    ///
    /// **The source, not a pre-read map, because a load has more than one `Loader`.** The
    /// trunk's large projections are built by the layer store, which constructs its own
    /// loaders (`layer_loader`), and a map consumed by whichever ran first would leave the
    /// others reading the checkpoint's quantized copies while the log said otherwise. Holding
    /// the source instead makes the substitution a property of *reading a tensor* rather than
    /// of one loader's bookkeeping, so every path that reads a gate gets the repaired one.
    ///
    /// `None` for every checkpoint that stores its gates at F32, which is every stock
    /// conversion. See [`LoadInputs::gate_src`].
    pub(super) gate_src: Option<(&'a gguf_file::Content, &'a [u8])>,
    /// Tensors this load reads from another checkpoint — see [`LoadInputs::overrides`].
    ///
    /// Consulted by every read, ahead of the donor and the mapping, so an overridden tensor
    /// comes from its own file whichever route builds it.
    pub(super) overrides: Option<&'a TensorOverrides<'a>>,
}

impl<'a, R: Read + Seek> Loader<'a, R> {
    /// A reader over `content`, for a caller outside [`load_quantized_model`].
    ///
    /// The layer-streaming pack build drives the same helpers the trunk loop
    /// does — which is the point, since a second transcription of a tensor's
    /// name is how a pack comes to describe a layer the loader does not build.
    pub(crate) fn new(
        content: &'a gguf_file::Content,
        reader: &'a mut R,
        device: &Device,
        mode: Int8Mode,
        residency: WeightResidency,
    ) -> Self {
        Self {
            content,
            reader,
            // A caller that reaches for `new` has a reader; the mapping arrives only through
            // `LoadInputs`, so this route keeps the upload-then-repack path.
            map: None,
            device: device.clone(),
            mode,
            residency,
            device_bytes: 0,
            narrow: None,
            stream_narrow: None,
            // A caller reaching for `new` is reading one checkpoint and repairing nothing; the
            // donor arrives through `LoadInputs` and only when the primary needs one.
            gate_src: None,
            // Overrides arrive through `LoadInputs` and reach only the trunk's loader; every
            // other loader reading one of them is caught by `TensorOverrides::ensure_all_read`.
            overrides: None,
        }
    }

    /// Give this loader the checkpoint's mapped bytes — see [`LoadInputs::map`].
    pub(crate) fn with_map(mut self, map: Option<&'a [u8]>) -> Self {
        self.map = map;
        self
    }

    /// Give this loader a donor to read the recurrent path from — see [`LoadInputs::gate_src`].
    pub(crate) fn with_gate_src(
        mut self,
        gate_src: Option<(&'a gguf_file::Content, &'a [u8])>,
    ) -> Self {
        self.gate_src = gate_src;
        self
    }

    /// Give this loader the tensors to read from another checkpoint — see
    /// [`LoadInputs::overrides`].
    pub(crate) fn with_overrides(mut self, overrides: Option<&'a TensorOverrides<'a>>) -> Self {
        self.overrides = overrides;
        self
    }

    /// This tensor as the donor stores it, if the donor should supply it.
    ///
    /// `None` — the overwhelmingly common answer — means "read it from the checkpoint as
    /// usual": there is no donor, this is not a recurrent-path tensor, or the two files
    /// already agree on its format and there is nothing to repair.
    ///
    /// **The format comparison is what keeps this from being a blanket override.** A fine-tune
    /// that stored its recurrent path exactly as the base did has changed only the values
    /// there, and those values are its own; a differing format is what marks the tensor as one
    /// this conversion re-quantized, which is the defect being repaired.
    pub(super) fn donated(&mut self, name: &str) -> Option<Result<QTensor>> {
        // A tensor named by an override comes from that file, repaired or not: the override
        // is the caller saying exactly which copy to run.
        if self.overrides.is_some_and(|o| o.names(name)) {
            return None;
        }
        let (donor, bytes) = self.gate_src?;
        if !RECURRENT_PATH.iter().any(|r| name.ends_with(r)) {
            return None;
        }
        let mine = self.content.tensor_infos.get(name)?;
        let theirs = donor.tensor_infos.get(name)?;
        if theirs.ggml_dtype == mine.ggml_dtype {
            return None;
        }
        if theirs.shape.dims() != mine.shape.dims() {
            return Some(Err(candle::Error::Msg(format!(
                "the gate donor's `{name}` is {:?} where this checkpoint's is {:?} — these are \
                 different models, not a checkpoint and its base",
                theirs.shape.dims(),
                mine.shape.dims()
            ))));
        }
        let mut cursor = std::io::Cursor::new(bytes);
        Some(donor.tensor(&mut cursor, name, &self.device))
    }

    /// Narrow every projection loaded until this is cleared. See [`Loader::narrow`].
    pub(crate) fn set_narrow(&mut self, narrow: Option<GgmlDType>) {
        self.narrow = narrow;
    }

    /// Apply the streamed trunk's narrowing schedule for a stack of `num_layers`.
    /// See [`Loader::stream_narrow`] and [`streaming_twin`].
    pub(crate) fn set_stream_narrow(&mut self, num_layers: Option<usize>) {
        self.stream_narrow = num_layers;
    }
}

impl<R: Read + Seek> Loader<'_, R> {
    /// A tensor read to the load's device, counted toward the model's footprint.
    pub(super) fn raw(&mut self, name: &str) -> Result<QTensor> {
        let device = self.device.clone();
        let t = self.read_to(name, &device)?;
        self.device_bytes += t.storage_size_in_bytes();
        Ok(t)
    }

    /// A tensor read to the host, never touching the card: for a weight the
    /// model keeps on the host, which read to the device would stand there whole
    /// for nothing — and the load headroom, which the span never takes back,
    /// would have to hold it (`dense_span::peak_load_pool_bytes`).
    pub(super) fn host(&mut self, name: &str) -> Result<QTensor> {
        self.read_to(name, &Device::Cpu)
    }

    pub(super) fn read_to(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        // **Which tensor, and how big, at the moment it is read.**
        //
        // A load that runs out of VRAM reports `CUDA_ERROR_OUT_OF_MEMORY` and nothing else —
        // no tensor, no size, no indication whether the card is small or one tensor is
        // unusual. That cost a bisect against a known-good checkpoint to find out that a
        // `…NEO-IMATRIX-MAX…` conversion carries its `output.weight` at F16, 1,940 MiB where
        // the stock file has 834. Logged before the read so the *failing* tensor is the last
        // line, not the one before it.
        let source = self.overrides.and_then(|o| o.take(name));
        let content = source.map_or(self.content, |(c, _)| c);
        if let Some(info) = content.tensor_infos.get(name) {
            let bytes = info.shape.elem_count() / info.ggml_dtype.block_size()
                * info.ggml_dtype.type_size();
            tracing::debug!(
                target: "candle_transformers::qwen35",
                tensor = name,
                dtype = ?info.ggml_dtype,
                shape = ?info.shape.dims(),
                mib = bytes >> 20,
                overridden = source.is_some(),
                "reading"
            );
        }
        match source {
            Some((c, bytes)) => c.tensor(&mut std::io::Cursor::new(bytes), name, device),
            None => self.content.tensor(self.reader, name, device),
        }
    }

    /// The twin this tensor should take: the scoped override if one is set, else the streaming
    /// schedule if this load is narrowing, else whatever the mode picks.
    pub(super) fn twin_for(&self, name: &str) -> Option<GgmlDType> {
        self.narrow
            .or_else(|| self.stream_narrow.and_then(|n| streaming_twin(name, n)))
    }

    /// A projection: quantized, repacked for the numeric mode.
    pub(super) fn proj(&mut self, name: &str) -> Result<QMatMul> {
        let want = self.twin_for(name);
        // **A donated gate is answered before any route that reads `content`.** The
        // host-banded path reads the checkpoint's own mapping by offset, which for these
        // tensors holds exactly the quantized copies being repaired — so consulting the map
        // second would silently ignore the donor for every mapped load, which is every
        // production one. See `Loader::gates`.
        if let Some(qt) = self.donated(name) {
            let qt = qt?;
            self.device_bytes += qt.storage_size_in_bytes();
            return QMatMul::from_qtensor_in(qt, self.mode, self.residency);
        }
        if let Some(m) = self.host_banded(name, want)? {
            return Ok(m);
        }
        // A source read whole only to be repacked into a twin in the dense block is a
        // transient of the load, served from its scratch at the span's top — under the
        // rule `host_banded` takes a tensor by, since it is the same tensor this load
        // could not band for want of a mapping.
        let twin_in_dense = self.residency == WeightResidency::Span
            && self
                .overrides
                .and_then(|o| o.effective_info(self.content, name))
                .or_else(|| self.content.tensor_infos.get(name))
                .is_some_and(|info| dense_span::bands_from_host(info, self.mode));
        if twin_in_dense {
            dense_span::transient(|| self.read_whole(name, want))
        } else {
            self.read_whole(name, want)
        }
    }

    /// A projection read to the device whole and built there, its twin narrowed to `want`
    /// where that shrinks it.
    pub(super) fn read_whole(&mut self, name: &str, want: Option<GgmlDType>) -> Result<QMatMul> {
        let qt = self.raw(name)?;
        match want {
            None => QMatMul::from_qtensor_in(qt, self.mode, self.residency),
            // A source already at or below the narrowing width has nothing to give up, and
            // asking for a *wider* twin than the mode picked is refused downstream — so pass
            // the request only when it can actually shrink this tensor. That is what lets the
            // schedule name one target per role and leave the checkpoint's own carve-outs
            // alone: a block already at or below the target simply keeps what it has.
            Some(n) => {
                let picked = qt.dtype().to_ko(self.mode).ok();
                if picked.is_some_and(|p| n.bits_per_weight() < p.bits_per_weight()) {
                    QMatMul::from_qtensor_narrowed(qt, self.mode, self.residency, n)
                } else {
                    QMatMul::from_qtensor_in(qt, self.mode, self.residency)
                }
            }
        }
    }

    /// Repack a projection **straight from the mapping**, never putting the source on the card.
    ///
    /// `Ok(None)` means this weight is not eligible and the caller should take the ordinary
    /// route — that is the common answer and not a failure.
    ///
    /// # Why the eligibility test is these five things and not a size threshold
    ///
    /// Each one is a condition `QMatMul::build` would apply anyway, asked earlier because the
    /// decision has to be made before the source is read rather than after:
    ///
    /// * a mapping to read from;
    /// * an int8 mode — at `Off` there is no KO twin to build;
    /// * `Span` residency — the pack build materialises a layer only to copy it out, and its
    ///   `Pool` twins are not the ones competing for the load budget;
    /// * a 2-D shape the KO matmul can tile, which is `build`'s own gate;
    /// * a source `repack_ko_into` can read directly.
    ///
    /// A tensor failing any of them takes the older path and costs what it always did. There
    /// is deliberately no "only if it is big": a rule that applies to some weights is a rule
    /// with an untested branch, and the banded route is not slower for a small tensor — it is
    /// one band either way.
    #[cfg(feature = "cuda")]
    pub(super) fn host_banded(
        &mut self,
        name: &str,
        narrow: Option<GgmlDType>,
    ) -> Result<Option<QMatMul>> {
        use candle::quantized::cuda::repack_ko_from_host;
        use candle::quantized::{QStorage, QTensor};

        // An overridden tensor is banded out of its own file's mapping, under the same rules.
        let (content, map) = match self.overrides.and_then(|o| o.take(name)) {
            Some((c, bytes)) => (c, Some(bytes)),
            None => (self.content, self.map),
        };
        let (Some(map), Device::Cuda(cuda)) = (map, &self.device) else {
            return Ok(None);
        };
        if self.residency != WeightResidency::Span {
            return Ok(None);
        }
        let Some(info) = content.tensor_infos.get(name) else {
            return Ok(None);
        };
        // The int8 mode, the tiling and the source: the rule the load headroom
        // leaves a banded head out by (`head_bands_from_host`).
        if !dense_span::bands_from_host(info, self.mode) {
            return Ok(None);
        }
        let [nrows, ncols] = info.shape.dims() else {
            return Ok(None);
        };
        let (nrows, ncols) = (*nrows, *ncols);
        // The twin this weight would get anyway: the narrowing target when one is set and it
        // can actually shrink this source, else whatever the mode picks. Same rule as `proj`'s
        // own — a wider request leaves the tensor alone.
        let picked = info.ggml_dtype.to_ko(self.mode)?;
        let ko_dtype = match narrow {
            Some(n) if n.bits_per_weight() < picked.bits_per_weight() => n,
            _ => picked,
        };

        // The tensor's bytes where GGUF put them: whole blocks, row-major, contiguous.
        let start = (content.tensor_data_offset + info.offset) as usize;
        let len = nrows * ncols / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let Some(bytes) = map.get(start..start + len) else {
            // A mapping that does not cover the tensor is a truncated file, not a reason to
            // silently take a different route — the ordinary path would fail on the same
            // bytes, and saying so here names the tensor.
            candle::bail!(
                "{name}: the checkpoint mapping is {} bytes and this tensor needs \
                 {start}..{}",
                map.len(),
                start + len
            );
        };

        let shape = candle::Shape::from((nrows, ncols));
        let dst =
            crate::models::quantized_matmul::dense_destination_for(&self.device, &shape, ko_dtype);
        let twin = repack_ko_from_host(cuda, bytes, &shape, info.ggml_dtype, ko_dtype, dst)?;
        // The twin is what stays resident, so it is the twin that counts toward the model's
        // device footprint — the source never had any.
        self.device_bytes += twin.storage_size_in_bytes();
        let qt = QTensor::new(QStorage::Cuda(twin), (nrows, ncols))?;
        Ok(Some(QMatMul::from_qtensor_view(qt, self.mode)?))
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn host_banded(&mut self, _: &str, _: Option<GgmlDType>) -> Result<Option<QMatMul>> {
        Ok(None)
    }

    /// An elementwise constant: dequantized to F32 once, at load.
    pub(super) fn f32(&mut self, name: &str) -> Result<Tensor> {
        self.raw(name)?.dequantize(&self.device)
    }

    /// A residual-stream norm, retained quantized so it can re-materialise
    /// itself in whatever dtype the session's activations arrive in.
    pub(super) fn norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let qt = self.raw(name)?;
        RmsNorm::from_qtensor(qt, eps)
    }

    pub(super) fn has(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    /// A gated-attention block's projections and Q/K norms, at tensor prefix
    /// `p`. Shared by the trunk's attention layers and the MTP draft head,
    /// which carries exactly the same tensor set.
    pub(super) fn attention(&mut self, p: &str, eps: f64) -> Result<QuantAttentionWeights> {
        // Three weights, for the same reason the DeltaNet loader keeps four:
        // `streaming_twin` narrows `attn_q` alone.
        let wq = self.proj(&format!("{p}.attn_q.weight"))?;
        let wk = self.proj(&format!("{p}.attn_k.weight"))?;
        let rows = |w: &QMatMul, what: &str| -> Result<usize> {
            match w.weight_dims().as_slice() {
                [rows, _] => Ok(*rows),
                other => candle::bail!("{what} is rank {}, expected 2", other.len()),
            }
        };
        let q_rows = rows(&wq, "attn_q")?;
        let kv_rows = rows(&wk, "attn_k")?;
        Ok(QuantAttentionWeights {
            wqkv: vec![wq, wk, self.proj(&format!("{p}.attn_v.weight"))?],
            q_rows,
            kv_rows,
            wo: self.proj(&format!("{p}.attn_output.weight"))?,
            q_norm: self.norm(&format!("{p}.attn_q_norm.weight"), eps)?,
            k_norm: self.norm(&format!("{p}.attn_k_norm.weight"), eps)?,
        })
    }

    /// A dense gated MLP at tensor prefix `p`.
    pub(super) fn dense_ffn(&mut self, p: &str) -> Result<QuantFfnWeights> {
        // **One target per projection, each asked for by name.**
        //
        // This used to derive a single width from `ffn_down` and hand it to all
        // three, on the reasoning that gate and up are already at or below it so
        // naming it leaves them untouched. That is a property of some
        // checkpoints, not of the schedule: narrowing is a floor that *shrinks*,
        // so a gate or up whose twin is wider than the down-projection's target
        // — a Q4_K gate at `Int8Mode::Performance` against a `Q3_KO` down — is
        // narrowed too. `streaming_twin` names no target for those roles and
        // `an_unnamed_role_is_untouched` asserts they are untouched, so the
        // loader was implementing a different schedule from the one its tests
        // describe. Asking per tensor is the schedule.
        let narrow = |t: &str| {
            self.narrow
                .or_else(|| self.twin_for(&format!("{p}.{t}.weight")))
        };
        let narrow = (narrow("ffn_gate"), narrow("ffn_up"), narrow("ffn_down"));
        let (mode, residency) = (self.mode, self.residency);
        QuantFfnWeights::from_weights_in(
            self.raw(&format!("{p}.ffn_gate.weight"))?,
            self.raw(&format!("{p}.ffn_up.weight"))?,
            self.raw(&format!("{p}.ffn_down.weight"))?,
            mode,
            residency,
            narrow,
        )
    }
}
