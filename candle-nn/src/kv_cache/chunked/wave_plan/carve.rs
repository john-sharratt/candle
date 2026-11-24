//! Gives each buffer its shape for a wave of a given width, which is what the
//! plan and the call sites both size from.
//!
//! Each buffer names the unit it scales with: a decode-chain buffer by the
//! decode group, a prefill-chain one by the prefill group, and the head by its
//! sequence count. That is why shape takes a `WaveWidth` rather than a row
//! count. Every variant states its shape here and takes its byte count from
//! `BufferShape::bytes`, so no call site writes its own arithmetic. Buffers a
//! stack never allocates price at zero, and a one-row wave copies nothing out
//! of its fused projection row, because a narrow along the last dimension is
//! contiguous when every leading dimension is one.

use super::*;

impl WaveBuffer {
    /// This buffer's shape for a wave of width `w` — the declaration the call
    /// site and the admission gate both size from.
    ///
    /// **Each buffer names the unit it scales with**, which is the whole point
    /// of taking a [`WaveWidth`] rather than a row count: a decode-chain buffer
    /// is sized by the decode group, a prefill-chain one by the prefill group,
    /// and the head by the sequence count.
    ///
    /// Also exhaustive: a new variant must state its shape here, and takes its
    /// byte count from [`BufferShape::bytes`] rather than writing arithmetic of
    /// its own.
    pub fn shape(&self, g: &ModelGeometry, w: WaveWidth) -> BufferShape {
        let dense = |rows, cols, dtype| BufferShape {
            rows,
            cols,
            encoding: Encoding::Dense(dtype),
        };
        let q8 = |rows, cols| BufferShape {
            rows,
            cols,
            encoding: Encoding::Q8a128,
        };
        // Every row of the wave, which is what a buffer both chains allocate is
        // sized by. The two groups' own counts are named where they are used.
        let rows = w.rows();
        let er = g.expert_rows(rows);
        // Zero widths where there is no shared expert, which the guard arm below
        // prices at nothing before any of these are read.
        let shared = g.shared_expert.unwrap_or(SharedExpertWidths {
            intermediate: 0,
            gate_cols: 0,
        });
        // A layer norm's output encoding is the session's, not the buffer's —
        // the same fork for the attention norm, the FFN norm and the dense
        // FFN's norm.
        let norm = |rows: usize| {
            if g.packed_norm {
                q8(rows, g.hidden)
            } else {
                dense(rows, g.hidden, g.act_dtype)
            }
        };
        // The pre-mix's widths, zero where the stack has none — which prices
        // both preludes at nothing and leaves every other stack's spans alone.
        let hyper = g.hyper.unwrap_or(HyperWidths {
            streams: 0,
            low_rank: 0,
            draft_head: false,
            ple: false,
            mixer_dtype: g.act_dtype,
        });
        let hyper_rows = if g.hyper.is_some() { rows } else { 0 };
        let hyper_scored = if g.hyper.is_some() { w.scored_rows } else { 0 };
        // The KO projections' padded widths (`HcWeightsKo`): the rank to the
        // int8 K tile, the down projection's rows to the matmul's N tile. The
        // two q8a128 operands exist only where the matmul is int8.
        let gate_cols = hyper.low_rank.div_ceil(Q8A128_TILE_ELEMS) * Q8A128_TILE_ELEMS;
        let down_cols = |inject: usize| (gate_cols + inject).div_ceil(KO_N_TILE) * KO_N_TILE;
        let operand_rows = |r: usize| if g.packed_norm { r } else { 0 };
        let hc_dim = hyper.streams * g.hidden;
        match self {
            Self::HyperAttnNorm
            | Self::HyperAttnGateRaw
            | Self::HyperFfnNorm
            | Self::HyperFfnGateRaw => dense(hyper_rows, hc_dim, DType::F32),
            Self::HyperAttnNormOperand | Self::HyperFfnNormOperand => {
                q8(operand_rows(hyper_rows), hc_dim)
            }
            Self::HyperAttnLowRank | Self::HyperFfnLowRank => {
                dense(hyper_rows, down_cols(hyper.streams), DType::F32)
            }
            Self::HyperAttnGateAct | Self::HyperFfnGateAct => {
                dense(hyper_rows, gate_cols, DType::F32)
            }
            Self::HyperAttnGateOperand | Self::HyperFfnGateOperand => {
                q8(operand_rows(hyper_rows), gate_cols)
            }
            Self::HyperAttnMixed | Self::HyperFfnMixed => dense(hyper_rows, g.hidden, DType::F32),
            // The head's mix, over the scored rows, with no inject columns.
            Self::HyperHeadNorm | Self::HyperHeadGateRaw => dense(hyper_scored, hc_dim, DType::F32),
            Self::HyperHeadNormOperand => q8(operand_rows(hyper_scored), hc_dim),
            Self::HyperHeadLowRank => dense(hyper_scored, down_cols(0), DType::F32),
            Self::HyperHeadGateAct => dense(hyper_scored, gate_cols, DType::F32),
            Self::HyperHeadGateOperand => q8(operand_rows(hyper_scored), gate_cols),
            Self::HyperHeadMixed => dense(hyper_scored, g.hidden, DType::F32),
            // Under the pre-mix a float head's "norm" is a wrap of the mix
            // output and carves nothing; a packed one still quantizes it.
            Self::HeadNorm | Self::HeadNormF32 if g.hyper.is_some() && !g.packed_head => {
                dense(0, 0, g.act_dtype)
            }
            // A float session's attention "norm" under the pre-mix is a wrap of
            // the mix output, which carves nothing; an int8 one still quantizes
            // it into q8a128 on the span.
            Self::AttnNorm if g.hyper.is_some() && !g.packed_norm => dense(0, 0, g.act_dtype),
            Self::AttnNorm => norm(rows),
            Self::QkvProjection => dense(rows, g.qkv_cols(), g.act_dtype),
            // A bias add reads the narrowed view directly and writes its own
            // contiguous result, so where there are biases the adds are the Q,
            // K and V buffers and no split copy happens at any width.
            Self::QSplit | Self::KSplit | Self::VContiguous if g.qkv_bias => {
                dense(0, 0, g.act_dtype)
            }
            Self::QBias if g.qkv_bias => dense(rows, g.attn_cols(), g.act_dtype),
            Self::KBias | Self::VBias if g.qkv_bias => dense(rows, g.kv_cols(), g.act_dtype),
            Self::QBias | Self::KBias | Self::VBias => dense(0, 0, g.act_dtype),
            // **A one-row wave copies nothing out of the fused row.** The narrow
            // is along the last dimension, so it is strided only while some
            // leading dimension is wider than one. With a single row every
            // leading dimension is 1, `Shape::is_contiguous` skips unit
            // dimensions, and `contiguous()` hands the view straight back.
            // Measured on Qwen3-30B-A3B: a one-row decode priced 40,704 B of
            // attention and carved 30,464 — these three, exactly.
            //
            // Not Q on a gated lineage: its `[q | gate]` is interleaved per
            // head, so the narrow strides across heads however few rows there
            // are.
            Self::QSplit if rows == 1 && !g.gated_qkv => dense(0, 0, g.act_dtype),
            Self::KSplit | Self::VContiguous if rows == 1 => dense(0, 0, g.act_dtype),
            // Q is narrowed out of a wider buffer whenever there *is* one: a
            // fused projection's `qkv_cols` row, or a gated lineage's
            // interleaved `[q | gate]`. With neither, `wq`'s output is already
            // the tensor Q wants.
            // A hyper-connection stack loads its q/k/v stacked into one
            // projection, so `split_group` scatters the whole group into one
            // carve rather than narrowing Q alone out of it.
            Self::QSplit if g.hyper.is_some() => dense(rows, g.qkv_cols(), g.act_dtype),
            Self::QSplit if g.fused_qkv || g.gated_qkv => dense(rows, g.attn_cols(), g.act_dtype),
            Self::QSplit => dense(0, 0, g.act_dtype),
            Self::QNormOut if g.head_qk_norm => dense(rows, g.attn_cols(), g.act_dtype),
            Self::QNormOut => dense(0, 0, g.act_dtype),
            // **Prefill's rows, not the wave's.** The reshapes exist because a
            // `[batch, seq, ..]` view cannot be transposed contiguously — which
            // `seq == 1` makes moot, so the decode chain carves none of them.
            // The decode census confirms it: nine carves, and these four are
            // not among them.
            // Q is a strided narrow out of the interleaved `[q | gate]` rows the
            // stacked split wrote, so flattening it for the norm copies it, on
            // every row.
            Self::QNormIn if g.hyper.is_some() && g.head_qk_norm && g.gated_qkv => {
                dense(rows, g.attn_cols(), g.act_dtype)
            }
            Self::QNormIn | Self::QHeadsPacked if g.head_qk_norm && g.head_norm_reshapes => {
                dense(w.prefill_rows, g.attn_cols(), g.act_dtype)
            }
            Self::QNormIn | Self::QHeadsPacked => dense(0, 0, g.act_dtype),
            // The split happens on both chains — the gate is a strided narrow
            // out of the interleaved `[q | gate]` however the context is
            // computed.
            Self::GateSplit if g.gated_qkv => dense(rows, g.attn_cols(), g.act_dtype),
            // **The sigmoid and the apply are prefill's, on an int8 session.**
            // The fused q8 decode context folds the gate into the kernel, so a
            // decode row carves neither; the FP decode path computes them as
            // ordinary ops and does. This was the union the plan used to charge
            // at the whole wave's width — measured on the 9B at 20 decode rows,
            // exactly the 327,680 B its attention span went unused by.
            Self::GateSigmoid | Self::GatedContext if g.gated_qkv && g.packed_norm => {
                dense(w.prefill_rows, g.attn_cols(), g.act_dtype)
            }
            Self::GateSigmoid | Self::GatedContext if g.gated_qkv => {
                dense(rows, g.attn_cols(), g.act_dtype)
            }
            Self::GateSplit | Self::GateSigmoid | Self::GatedContext => dense(0, 0, g.act_dtype),
            Self::QRotaryPermute if g.partial_rotary => dense(rows, g.attn_cols(), g.act_dtype),
            Self::KRotaryPermute if g.partial_rotary => dense(rows, g.kv_cols(), g.act_dtype),
            Self::QRotaryPermute | Self::KRotaryPermute => dense(0, 0, g.act_dtype),
            // K and V are copied out of the fused row; separate projections
            // write them contiguous and neither copy exists.
            Self::KSplit | Self::VContiguous if g.fused_qkv => {
                dense(rows, g.kv_cols(), g.act_dtype)
            }
            Self::KSplit | Self::VContiguous => dense(0, 0, g.act_dtype),
            Self::KNormIn | Self::KHeadsPacked if g.head_qk_norm && g.head_norm_reshapes => {
                dense(w.prefill_rows, g.kv_cols(), g.act_dtype)
            }
            Self::KNormIn | Self::KHeadsPacked => dense(0, 0, g.act_dtype),
            Self::KNormOut if g.head_qk_norm => dense(rows, g.kv_cols(), g.act_dtype),
            Self::KNormOut => dense(0, 0, g.act_dtype),
            // The dense context is what the paged *prefill* kernel writes; the
            // decode kernel emits q8a1024 into `DecodeContext` instead. Two
            // buffers because a mixed wave carves both into one generation —
            // each at its own group's width, which is what a single row count
            // could not say.
            // A decode row's context is dense too where the decode kernel does
            // not emit q8 (`decode_q8_context` off — Qwen2's head dim 64).
            // One header per prefill or glue sequence — at least one where any
            // prefill row runs — in two carves when a wave runs both groups.
            Self::PrefillSlotHeaders if w.prefill_rows > 0 => dense(
                w.prefill_spans.max(1) * SLOT_HEADER_BYTES + BUMP_ALIGNMENT,
                1,
                DType::U8,
            ),
            Self::PrefillSlotHeaders => dense(0, 0, DType::U8),
            Self::AttnOutput if g.decode_q8_context => {
                dense(w.prefill_rows, g.attn_cols(), g.act_dtype)
            }
            Self::AttnOutput => dense(w.rows(), g.attn_cols(), g.act_dtype),
            Self::PrefillKvStage => dense(w.kv_stage.bytes(), 1, DType::U8),
            // Only where the decode kernel emits it — see
            // `ModelGeometry::decode_q8_context`.
            Self::DecodeContext if g.decode_q8_context => q8(w.decode_rows, g.attn_cols()),
            Self::DecodeContext => dense(0, 0, g.act_dtype),
            // A decode row's context is already packed where the decode kernel
            // emits q8 (`DecodeContext`); elsewhere it is quantized here too.
            Self::OProjOperand if g.packed_norm && g.decode_q8_context => {
                q8(w.prefill_rows, g.attn_cols())
            }
            Self::OProjOperand if g.packed_norm => q8(w.rows(), g.attn_cols()),
            Self::OProjOperand => dense(0, 0, g.act_dtype),
            Self::OProjOutput if g.packed_norm => {
                dense(w.decode_rows + w.prefill_rows, g.hidden, g.act_dtype)
            }
            Self::OProjOutput => dense(w.decode_rows, g.hidden, g.act_dtype),
            // The DeltaNet chain prices zero on a stack with no DeltaNet
            // layers, which makes `Chain::DeltaNet` sum to zero and drop out of
            // the phase's max — no all-attention model's span moves.
            // Under the pre-mix the mixer's input is the mix output, already
            // carved by the prelude — no norm of its own.
            Self::DeltaNetNorm => {
                let cols = if g.delta_net.is_some() && g.hyper.is_none() {
                    g.hidden
                } else {
                    0
                };
                dense(rows, cols, g.act_dtype)
            }
            // The upcast to F32 is a no-op on an operand that is already F32,
            // and carves nothing.
            Self::DeltaNetBetaOperand | Self::DeltaNetAlphaOperand => {
                let cols = if g.delta_net.is_some() && g.act_dtype != DType::F32 {
                    g.hidden
                } else {
                    0
                };
                dense(rows, cols, DType::F32)
            }
            // Under the pre-mix β and α ride the stacked projection instead.
            Self::DeltaNetBetaProj | Self::DeltaNetAlphaProj if g.hyper.is_some() => {
                dense(0, 0, DType::F32)
            }
            Self::DeltaNetBetaProj | Self::DeltaNetAlphaProj => {
                dense(rows, g.delta_net.map_or(0, |d| d.n_v_heads), DType::F32)
            }
            // The single-stream stack's separate `[q|k|v]` and `z` projections;
            // the pre-mix stack stacks them into `DeltaNetLiveProjection`.
            Self::DeltaNetQkvOperand
            | Self::DeltaNetQkvProj
            | Self::DeltaNetZOperand
            | Self::DeltaNetZProj
                if g.hyper.is_some() || g.delta_net.is_none() =>
            {
                dense(0, 0, DType::F32)
            }
            Self::DeltaNetQkvOperand | Self::DeltaNetZOperand if g.packed_norm => {
                q8(rows, g.hidden)
            }
            Self::DeltaNetQkvOperand | Self::DeltaNetZOperand => dense(0, 0, DType::F32),
            Self::DeltaNetQkvProj => dense(
                rows,
                g.delta_net.expect("guarded above").conv_dim,
                DType::F32,
            ),
            Self::DeltaNetZProj => dense(
                rows,
                g.delta_net.expect("guarded above").value_dim,
                DType::F32,
            ),
            // The pre-mix stack's stacked projection: only there.
            Self::DeltaNetLiveOperand | Self::DeltaNetLiveProjection | Self::DeltaNetLiveSplit
                if g.hyper.is_none() || g.delta_net.is_none() =>
            {
                dense(0, 0, DType::F32)
            }
            // The rest of the live chain, on either stack with DeltaNet layers.
            Self::DeltaNetLiveSpanPtrs
            | Self::DeltaNetLiveSpanExtents
            | Self::DeltaNetLiveConv
            | Self::DeltaNetLiveScanOut
            | Self::DeltaNetLiveScanU
            | Self::DeltaNetLiveScanW
            | Self::DeltaNetLiveScanKq
            | Self::DeltaNetLiveDecay
            | Self::DeltaNetLiveNormGate
            | Self::DeltaNetLiveOutOperand
            | Self::DeltaNetLiveOut
                if g.delta_net.is_none() =>
            {
                dense(0, 0, DType::F32)
            }
            Self::DeltaNetLiveOperand => q8(rows, g.hidden),
            Self::DeltaNetLiveProjection | Self::DeltaNetLiveSplit => {
                let d = g.delta_net.expect("guarded above");
                dense(rows, d.conv_dim + d.value_dim + 2 * d.n_v_heads, DType::F32)
            }
            // One entry per multi-row span; a decode row has none.
            Self::DeltaNetLiveSpanPtrs => dense(w.prefill_spans, 4, DType::I64),
            Self::DeltaNetLiveSpanExtents => dense(w.prefill_spans, 2, DType::U32),
            Self::DeltaNetLiveConv => dense(
                rows,
                g.delta_net.expect("guarded above").conv_dim,
                DType::F32,
            ),
            Self::DeltaNetLiveScanOut | Self::DeltaNetLiveNormGate => dense(
                rows,
                g.delta_net.expect("guarded above").value_dim,
                DType::F32,
            ),
            // The prefill scan's four transients exist only when a multi-row
            // span runs it, and are then sized by the WHOLE wave: each span
            // writes its own rows of one wave-wide allocation.
            Self::DeltaNetLiveScanU | Self::DeltaNetLiveScanW => {
                let scan_rows = if w.prefill_spans > 0 { rows } else { 0 };
                dense(
                    scan_rows,
                    g.delta_net.expect("guarded above").value_dim,
                    DType::F32,
                )
            }
            Self::DeltaNetLiveScanKq => {
                let d = g.delta_net.expect("guarded above");
                let scan_rows = if w.prefill_spans > 0 { rows } else { 0 };
                dense(scan_rows, d.n_v_heads * DELTA_NET_SCAN_CHUNK, DType::F32)
            }
            Self::DeltaNetLiveDecay => {
                let scan_rows = if w.prefill_spans > 0 { rows } else { 0 };
                dense(
                    scan_rows,
                    g.delta_net.expect("guarded above").n_v_heads,
                    DType::F32,
                )
            }
            Self::DeltaNetLiveOutOperand => q8(rows, g.delta_net.expect("guarded above").value_dim),
            Self::DeltaNetLiveOut => dense(rows, g.hidden, g.act_dtype),
            // The shared expert's half of the chain, on a stack whose MoE has
            // one — and nothing at all otherwise.
            Self::SharedGateUp
            | Self::SharedAct
            | Self::SharedGated
            | Self::SharedDownOperand
            | Self::SharedDown
            | Self::SharedGateLogits
            | Self::SharedGateSigmoid
            | Self::SharedGatedOut
            | Self::MoeSharedSum
                if !g.is_moe() || g.shared_expert.is_none() =>
            {
                dense(0, 0, g.act_dtype)
            }
            Self::SharedGateUp => dense(rows, 2 * shared.intermediate, g.work_dtype()),
            Self::SharedAct | Self::SharedGated => dense(rows, shared.intermediate, g.work_dtype()),
            Self::SharedDownOperand => q8(rows, shared.intermediate),
            Self::SharedDown | Self::SharedGatedOut => dense(rows, g.hidden, g.work_dtype()),
            Self::SharedGateLogits => dense(rows, shared.gate_cols, g.work_dtype()),
            Self::SharedGateSigmoid => dense(rows, 1, g.work_dtype()),
            Self::MoeSharedSum => dense(rows, g.hidden, g.work_dtype()),
            // The two FFN chains are alternatives a layer dispatches between,
            // so each prices zero on the geometry that does not run it and the
            // phase's `max` takes whichever is live.
            Self::FfnNorm
            | Self::MoeCombine
            | Self::MoeResultCast
            | Self::FfnNormOperand
            | Self::DownGemm
            | Self::MoeGather
            | Self::RouterLogits
            | Self::LookAheadLogits
            | Self::RouteWeights
            | Self::RouteIndices
            | Self::RoutingTokenIds
            | Self::RoutingWeightIds
            | Self::RoutingPermutation
            | Self::RoutingTokenStarts
            | Self::GateGemm
            | Self::UpGemm
            | Self::SwigluAct
                if !g.is_moe() =>
            {
                dense(0, 0, g.act_dtype)
            }
            // Under the pre-mix the FFN's "norm" is the mix output quantized
            // once for every consumer — a carve on an int8 session, a wrap on a
            // float one.
            Self::FfnNorm if g.hyper.is_some() && !g.packed_norm => dense(0, 0, g.act_dtype),
            Self::FfnNorm => norm(rows),
            Self::RouterLogits => dense(rows, g.n_experts, g.act_dtype),
            Self::LookAheadLogits => dense(rows, g.n_experts * w.look_ahead_hops, DType::F32),
            Self::RouteWeights => dense(rows, g.experts_per_tok, DType::F32),
            Self::RouteIndices => dense(rows, g.experts_per_tok, DType::U32),
            Self::RoutingTokenIds | Self::RoutingWeightIds | Self::RoutingPermutation => {
                dense(er, 1, DType::U32)
            }
            // Prefix offsets: one per token plus the end, so none at all for a
            // phase no token reaches.
            Self::RoutingTokenStarts if rows == 0 => dense(0, 0, DType::U32),
            Self::RoutingTokenStarts => dense(rows + 1, 1, DType::U32),
            Self::MoeGather => q8(er, g.hidden),
            Self::GateGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::UpGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::SwigluAct => q8(er, g.intermediate),
            Self::DownGemm => dense(er, g.hidden, g.accum_dtype),
            Self::MoeCombine => dense(rows, g.hidden, g.act_dtype),
            Self::MoeResultCast if g.work_dtype() != g.act_dtype => {
                dense(rows, g.hidden, g.act_dtype)
            }
            Self::MoeResultCast => dense(0, 0, g.act_dtype),
            // The F16 stability cast's operand, which does not exist without
            // it: the experts run in `work_dtype`, which equals `act_dtype`
            // unless the session is F16. A packed operand skips it outright.
            Self::FfnNormOperand if g.work_dtype() != g.act_dtype && !g.packed_norm => {
                dense(rows, g.hidden, g.work_dtype())
            }
            Self::FfnNormOperand => dense(0, 0, g.act_dtype),
            Self::DenseFfnNorm | Self::DenseGateUp | Self::DenseSilu | Self::DenseSwiglu
                if g.is_moe() =>
            {
                dense(0, 0, g.act_dtype)
            }
            Self::DenseFfnNorm => norm(rows),
            Self::DenseGateUp => dense(rows, 2 * g.intermediate, g.act_dtype),
            Self::DenseSilu | Self::DenseSwiglu => dense(rows, g.intermediate, g.act_dtype),
            // One row per sequence, both of them — the head scores the last
            // token of each.
            // All three follow the **head's** mode, not a layer's — see
            // `packed_head`. A packed head carves a q8a128 norm and its logits;
            // a float one carves a dense norm and an F32 working copy, and its
            // logits leave the span.
            Self::HeadNorm if g.packed_head => q8(w.scored_rows, g.hidden),
            Self::HeadNorm => dense(w.scored_rows, g.hidden, g.act_dtype),
            Self::HeadNormF32 if !g.packed_head => dense(w.scored_rows, g.hidden, DType::F32),
            Self::HeadLogits => dense(w.scored_rows, g.vocab, g.act_dtype),
            Self::HeadLogitsF32 if !g.packed_head && g.act_dtype != DType::F32 => {
                dense(w.scored_rows, g.vocab, DType::F32)
            }
            Self::HeadLogitsF32 => dense(0, 0, DType::F32),
            Self::AcceptRows => dense(w.accept_rows, g.vocab, g.act_dtype),
            Self::StencilBitmaps => dense(w.stencil_rows, g.vocab.div_ceil(32), DType::U32),
            Self::HeadNormF32 => dense(0, 0, g.act_dtype),
            // Every row of the wave, fresh or resumed, whether or not the head
            // runs: a window that stops short hands its residual on from here.
            Self::HyperResidual => dense(hyper_rows, hc_dim, DType::F32),
            // The rows' embeddings, where the head runs.
            Self::HeadRowEmbeds if w.scored_rows > 0 && hyper.draft_head => {
                dense(rows, g.hidden, DType::F32)
            }
            Self::HeadRowEmbeds => dense(0, 0, DType::F32),
            Self::HeadScoredResidual if g.hyper.is_some() => {
                dense(hyper_scored, hc_dim, DType::F32)
            }
            // A single-stream stack's residual is on this span too
            // (`ForwardResidual`), so the head's gather of its scored rows
            // lands here in the compute dtype.
            Self::HeadScoredResidual => dense(w.scored_rows, g.hidden, g.act_dtype),
            Self::HeadScoredRows => dense(w.scored_rows, 1, DType::U32),
            Self::QsaSelection => dense(w.qsa_bytes, 1, DType::U8),
            // The forward's setup tables: only the multi-stream stack builds
            // them on its forward span — every other stack's are its own.
            // A wave of no tokens is not a forward and builds no setup.
            Self::PrefillCuSeqlens
            | Self::PrefillQLens
            | Self::PrefillKvLens
            | Self::GlueWriteTables
            | Self::RowTokenIds
                if rows == 0 =>
            {
                dense(0, 0, DType::U32)
            }
            // A single-stream stack builds two ragged sets — the prefill group's
            // and the glue group's — so each table is two carves, one bump
            // alignment apart, over the spans of both.
            Self::PrefillCuSeqlens if g.hyper.is_none() => {
                dense(w.prefill_spans + 2 + BUMP_ALIGNMENT / 4, 1, DType::U32)
            }
            Self::PrefillQLens | Self::PrefillKvLens if g.hyper.is_none() => {
                dense(w.prefill_spans + BUMP_ALIGNMENT / 4, 1, DType::U32)
            }
            // Three per-row tables; the glue rows are among the prefill rows.
            Self::GlueWriteTables if g.hyper.is_none() => {
                dense(3 * w.prefill_rows + 2 * (BUMP_ALIGNMENT / 4), 1, DType::U32)
            }
            Self::GlueWriteTables => dense(0, 0, DType::U32),
            Self::ForwardResidual if g.hyper.is_none() => dense(rows, g.hidden, g.act_dtype),
            Self::ForwardResidual => dense(0, 0, g.act_dtype),
            Self::DeltaNetDecodeTable => match g.delta_net {
                Some(dn) if w.decode_rows > 0 => dense(
                    dn.layers * 3 * w.decode_rows * 8 + BUMP_ALIGNMENT + w.decode_rows * 4,
                    1,
                    DType::U8,
                ),
                _ => dense(0, 0, DType::U8),
            },
            // One carve on a single-stream stack: the ids laid on the span for
            // the embedding, nothing read back.
            Self::RowTokenIds if g.hyper.is_none() => dense(rows, 1, DType::U32),
            Self::PrefillCuSeqlens => dense(w.prefill_spans + 1, 1, DType::U32),
            Self::PrefillQLens | Self::PrefillKvLens => dense(w.prefill_spans, 1, DType::U32),
            // Two carves at most, so one bump alignment between them as well.
            Self::RowTokenIds => dense(2 * rows + BUMP_ALIGNMENT / 4, 1, DType::U32),
            Self::ImageRows => dense(w.image_rows, g.hidden, DType::F32),
            // A row id and `2 · pairs ≤ head_dim` table values a row, each a
            // word, in three carves one bump alignment apart.
            Self::ImageTurns if w.image_rows == 0 => dense(0, 0, DType::F32),
            Self::ImageTurns => dense(
                w.image_rows * (1 + g.head_dim) + 3 * (BUMP_ALIGNMENT / 4),
                1,
                DType::F32,
            ),
            // The draft head's input assembly, over every row of the wave.
            Self::HeadShift
            | Self::HeadHiddenNorm
            | Self::HeadEmbedNorm
            | Self::HeadEmbedOperand
            | Self::HeadEmbedProj
            | Self::HeadHiddenOperand
            | Self::HeadHiddenProj
            | Self::HeadInput
                if !hyper.draft_head =>
            {
                dense(0, 0, DType::F32)
            }
            Self::HeadShift | Self::HeadHiddenNorm | Self::HeadInput => {
                dense(rows, hc_dim, DType::F32)
            }
            // The embedding half, only on a wave that reaches the head.
            Self::HeadEmbedNorm | Self::HeadEmbedOperand | Self::HeadEmbedProj
                if w.scored_rows == 0 =>
            {
                dense(0, 0, DType::F32)
            }
            Self::HeadEmbedNorm | Self::HeadEmbedProj => dense(rows, g.hidden, DType::F32),
            Self::HeadEmbedOperand => q8(operand_rows(rows), g.hidden),
            Self::HeadHiddenOperand => q8(operand_rows(rows * hyper.streams), g.hidden),
            Self::HeadHiddenProj => dense(rows * hyper.streams, g.hidden, DType::F32),
            // A single-stream head's pass, over every row of a wave that
            // reaches it.
            Self::MtpRowEmbeds
            | Self::MtpTrunkNorm
            | Self::MtpShift
            | Self::MtpEmbedNorm
            | Self::MtpHiddenNorm
            | Self::MtpInputCat
            | Self::MtpInputOperand
            | Self::MtpInput
                if !g.mtp_head || w.scored_rows == 0 =>
            {
                dense(0, 0, g.act_dtype)
            }
            Self::MtpInputCat => dense(rows, 2 * g.hidden, g.act_dtype),
            // `eh_proj` follows the session's int8 mode, as every projection
            // the norms feed does; a float `eh_proj` reads the concatenation as
            // it stands.
            Self::MtpInputOperand => q8(operand_rows(rows), 2 * g.hidden),
            Self::MtpRowEmbeds
            | Self::MtpTrunkNorm
            | Self::MtpShift
            | Self::MtpEmbedNorm
            | Self::MtpHiddenNorm
            | Self::MtpInput => dense(rows, g.hidden, g.act_dtype),
            // The PLE block, over every row of the wave.
            Self::PleRows | Self::PleKeyValue | Self::PleNormalized | Self::PleSpanTable
                if !hyper.ple =>
            {
                dense(0, 0, DType::F32)
            }
            Self::PleRows => dense(rows, g.hidden, DType::F32),
            Self::PleKeyValue => dense(rows, hc_dim + g.hidden, DType::F32),
            Self::PleNormalized => dense(rows, hc_dim, DType::F32),
            Self::PleSpanTable => dense(w.scored_rows, 4, DType::I64),
        }
    }

    /// Bytes this buffer needs for a wave of width `w`.
    pub fn bytes(&self, g: &ModelGeometry, w: WaveWidth) -> usize {
        match g.hyper {
            Some(hyper) if self.is_mixer_carve() => self
                .shape(
                    &ModelGeometry {
                        act_dtype: hyper.mixer_dtype,
                        ..*g
                    },
                    w,
                )
                .bytes(),
            _ => self.shape(g, w).bytes(),
        }
    }

    /// Whether a hyper-connection stack carves this buffer in its
    /// [`HyperWidths::mixer_dtype`] rather than the F32 its residual carries:
    /// every attention-chain buffer but `o_proj`'s output, which is the
    /// mixer's contribution to that residual and is written in F32.
    pub fn is_mixer_carve(&self) -> bool {
        self.chain() == Chain::Attention && *self != Self::OProjOutput
    }
}
