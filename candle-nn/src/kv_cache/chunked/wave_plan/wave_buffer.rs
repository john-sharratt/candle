//! Names every buffer a layer allocates from the transient tier, one variant per
//! allocation site.
//!
//! Each variant is read off the census taken on Qwen3-30B-A3B, and its doc says
//! which chain allocates it, so the list can be checked against the code that
//! makes it. The variants are grouped by the stage that carves them: the Gated
//! Residual pre-mix, attention, the DeltaNet mixer, the FFN and its dense variant,
//! the draft head's pass, and the head after the last layer. The plan's inventory
//! is derived from these variants, so a new allocation is one new variant and
//! nothing else.

use super::*;

/// Every buffer a layer allocates from the transient tier.
///
/// One variant per allocation site, and the list is the **union** over the
/// chains a phase can run — see the module header for why a union rather than a
/// maximum. Every one of these was read off [`super::wave_census`] on
/// Qwen3-30B-A3B; the comment on each says which chain allocates it, so a
/// reader can check the list against the code that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum WaveBuffer {
    // ── The Gated Residual pre-mix, ahead of the attention phase's mixer ────
    // Five carves, in carve order, all F32 and all at the wave's full width —
    // the pre-mix runs once over every row before the groups split. The FFN
    // phase opens with the same five (`HyperFfn*`). Priced zero on a stack
    // without hyper-connections.
    /// The grouped norm over every stream: `[rows, streams · hidden]`.
    HyperAttnNorm,
    /// The norm quantized to q8a128 for the KO down projection — int8
    /// sessions only.
    HyperAttnNormOperand,
    /// The stacked down projection — the read gate's columns (the rank padded
    /// to the int8 K tile) with the `streams` inject columns beneath them, the
    /// whole padded to the matmul's N tile. The inject is read from here as a
    /// strided view by the combine; it carves nothing of its own.
    HyperAttnLowRank,
    /// `silu` over the (padded) gate columns, which is also their compaction
    /// for the up projection.
    HyperAttnGateAct,
    /// That SiLU quantized for the KO up projection — int8 sessions only.
    HyperAttnGateOperand,
    /// The up projection's raw gate, `[rows, streams · hidden]`.
    HyperAttnGateRaw,
    /// The gated mean over streams — the block input, `hidden` wide.
    HyperAttnMixed,
    /// Attention RMSNorm, in the encoding the QKV matmul consumes — q8a128 on
    /// an int8 session, `act_dtype` on a float one, per
    /// [`ModelGeometry::packed_norm`]. Both chains.
    AttnNorm,
    /// Q/K/V projection output, in the compute dtype.
    ///
    /// `qkv_cols` wide, which prices either dispatch: one segmented launch over
    /// the three KO weights narrowed afterwards, or three separate
    /// `forward_dynamic` calls whose widths sum to the same number.
    ///
    /// **There is no accumulate-dtype round trip to charge on either side of
    /// it.** The int8 path's MMA converts on the store out of registers, and the
    /// float path (`dense_qmatmul_float`) converts its activation only for a
    /// dtype outside `{F16, BF16, F32}` and otherwise stores at the activation's
    /// own width — so no session upcasts a compute-dtype operand to F32 and
    /// casts back. A pair of operand/accum charges modelling that round trip
    /// used to stand here and on `o_proj`, priced at `3 × hidden` and
    /// `qkv_cols` in F32: 90.2 MiB of a 219.9 MiB attention span on the 0.8B,
    /// for four buffers the census shows are never carved. What the census
    /// actually caught was a *DeltaNet* layer's F32 projections — a different
    /// layer kind sharing this arena, priced below at its own widths.
    QkvProjection,
    /// Q, copied out of the fused QKV buffer.
    ///
    /// The narrow is a strided view over a `qkv_cols`-wide row, so reshaping it
    /// to `[batch, seq, heads, dim]` cannot alias and copies. Both chains.
    QSplit,
    /// Q flattened to `[batch · heads · seq, dim]` for the head-wise RMSNorm.
    ///
    /// Free on the decode chain, where `seq == 1` leaves the transposed view
    /// contiguous; a real copy on prefill.
    QNormIn,
    /// Q RMSNorm output. Both chains.
    QNormOut,
    /// Q returned to `[batch, seq, heads · dim]` after the norm — a transpose
    /// then a reshape, so again a copy on prefill and free on decode.
    QHeadsPacked,
    /// The gate, split out of the interleaved `[q | gate]` projection. The
    /// narrow is strided over the `[.., heads, 2, dim]` view, so flattening it
    /// copies. Gated lineages only.
    GateSplit,
    /// `sigmoid(gate)`, materialised before the context multiply. Gated
    /// lineages only; the fused q8 decode context folds it into the kernel,
    /// but prefill and the FP decode path allocate it, and the plan prices
    /// the union.
    GateSigmoid,
    /// `context ⊙ sigmoid(gate)` — the gated context handed to `o_proj`.
    /// Gated lineages only, same union rule as [`Self::GateSigmoid`].
    GatedContext,
    /// Q re-ordered into the kernels' full-width rotary pairing — a gather
    /// copy (`RotaryLayout`). Partial-rotary models only.
    QRotaryPermute,
    /// K's half of the same re-ordering. Partial-rotary models only.
    KRotaryPermute,
    /// K, copied out of the fused QKV buffer. Both chains.
    KSplit,
    /// K flattened for the head-wise RMSNorm. Prefill only, as [`Self::QNormIn`].
    KNormIn,
    /// K RMSNorm output. Both chains.
    KNormOut,
    /// K returned to `[batch, seq, kv_heads · dim]`. Prefill only.
    KHeadsPacked,
    /// V, made contiguous out of the fused QKV buffer for the cache write.
    VContiguous,
    /// `q + bq`, on a stack with Q/K/V biases — see
    /// [`ModelGeometry::qkv_bias`]. Both chains, every width.
    QBias,
    /// `k + bk`, as [`Self::QBias`].
    KBias,
    /// `v + bv`, as [`Self::QBias`].
    VBias,
    /// The `SlotHeader`s a multi-token attention launch reads first, one per
    /// sequence, built on the host and uploaded beside the launch. Prefill and
    /// glue each build their own set, so a wave carrying both carves twice.
    PrefillSlotHeaders,
    /// The attention context in its **dense** form, which prefill and glue
    /// write — and decode too, on a stack whose decode kernel emits no q8
    /// context ([`ModelGeometry::decode_q8_context`]).
    AttnOutput,
    /// The prefill launch's carve beside its context: the pre-staged K/V of up
    /// to one key chunk per bulk sequence, the online-softmax carry and the
    /// resume table ([`PrefillKvStageLayout::bytes`] of
    /// [`WaveWidth::kv_stage`]). One per attention layer, so the phase's reset
    /// hands every layer the same ground.
    PrefillKvStage,
    /// The attention context in its **packed** form.
    ///
    /// The int8 decode kernel emits q8a1024 directly (`PagedDecode::q8_byte_size`)
    /// so `o_proj` needs no standalone quantize. Declared separately from
    /// [`Self::AttnOutput`] rather than folded into it because a mixed wave
    /// allocates both — the decode group's packed context and the prefill
    /// group's dense one — into the same generation.
    DecodeContext,
    /// The context quantized for an int8 `o_proj`, q8a128 over the attention
    /// columns.
    ///
    /// Prefill's `o_proj` takes a `Float` context, and the int8 matmul
    /// quantizes it at the call. The quantize inherits the context's arena, so
    /// the operand — and the projection after it — land on this span. A decode
    /// row's too, on a stack whose decode kernel emits a float context
    /// (`decode_q8_context` off — Qwen2's head dim 64). Int8 sessions only: a
    /// float `o_proj` reads the context as it stands.
    OProjOperand,
    /// `o_proj`'s result, in the compute dtype.
    ///
    /// The decode group's always; the prefill group's too on an int8 session,
    /// whose quantized operand ([`Self::OProjOperand`]) carries the context's
    /// arena on to the projection.
    OProjOutput,

    // ── The DeltaNet mixer ──────────────────────────────────────────────────
    // The other thing a layer can open the attention generation for, on a
    // hybrid stack. Priced as its own chain and compared against attention's
    // with a `max`, not summed into it — see `Chain`.
    //
    // **The whole mixer is on the span.** The layer norm seeds the chain on the
    // phase, the int8 projections' activation quantize inherits the norm's
    // arena, and every carve after it follows. In carve order on a
    // single-stream stack — the census of the 0.8B at 134 prefill rows: the
    // norm, the `[q|k|v]` projection's q8a128 operand and its F32 result, the
    // `z` projection's operand and result, the F32 upcast and result of each of
    // `beta` and `alpha` (whose weights are too small to KO-repack), and then
    // the live chain's span tables, conv, recurrence and its transients,
    // norm-gate and `w_out` (`DeltaNetLive*` below).
    /// The DeltaNet layer's input norm — `ln1` over the packed buffer, which is
    /// where this chain is seeded on the span.
    DeltaNetNorm,
    /// The `[q|k|v]` projection's q8a128 operand: the norm quantized for the
    /// int8 matmul, on a single-stream int8 stack.
    DeltaNetQkvOperand,
    /// The `[q|k|v]` projection's result, `conv_dim` wide, F32.
    DeltaNetQkvProj,
    /// The `z` projection's q8a128 operand — the norm quantized again for it.
    DeltaNetZOperand,
    /// The `z` projection's result, `value_dim` wide, F32.
    DeltaNetZProj,
    /// `beta`'s operand, upcast to F32 by the float arm of `forward_live_as`.
    DeltaNetBetaOperand,
    /// `beta` itself: one F32 per V head per row.
    DeltaNetBetaProj,
    /// `alpha`'s operand upcast, a separate carve from [`Self::DeltaNetBetaOperand`]
    /// because the two projections each upcast the norm output for themselves.
    DeltaNetAlphaOperand,
    /// `alpha`, the same shape as [`Self::DeltaNetBetaProj`].
    DeltaNetAlphaProj,

    // ── The live DeltaNet mixer on the Gated Residual stack ─────────────────
    // Under the pre-mix the mixer's input is F32 on the span and the
    // projections' activation quantize inherits it, so the whole mixer lands on
    // the span — nothing breaks provenance. In carve order: the operand, the
    // stacked `[qkv | z | β | α]` projection and its split, the two span
    // tables, the conv, the recurrence output, the prefill scan's four
    // transients (`u`, `w`, `kq`, `g_cs`), then the norm-gate and `w_out`.
    // Measured on Qwen3.8-Flash-Next at 2100 rows over four spans. From the
    // span tables on, the single-stream stack carves the same chain behind its
    // separate projections; the operand, stacked projection and split are the
    // pre-mix stack's alone.
    /// The four projections' q8a128 operand: the mix output quantized once.
    DeltaNetLiveOperand,
    /// The stacked projection, `conv_dim + value_dim + 2 · n_v_heads` wide, F32.
    DeltaNetLiveProjection,
    /// Its ragged split into the four parts — one carve the same width.
    DeltaNetLiveSplit,
    /// The per-span pointer tables the mixer builds beside its operands — the
    /// short-span table's three device pointers and the prefill table's four,
    /// one upload for both, priced at four for each multi-row span
    /// ([`WaveWidth::prefill_spans`]).
    DeltaNetLiveSpanPtrs,
    /// The per-span extents: two `u32` for each multi-row span, one upload for
    /// both tables.
    DeltaNetLiveSpanExtents,
    /// The causal conv's output, `conv_dim` wide.
    DeltaNetLiveConv,
    /// The recurrence's output, `value_dim` wide.
    DeltaNetLiveScanOut,
    /// The prefill scan's `u` transient, `[n_v_heads, rows, head_dim]` —
    /// `value_dim` a row. This and the three below are carved only when a
    /// multi-row span runs the scan, and then over every row of the wave.
    DeltaNetLiveScanU,
    /// The prefill scan's `w` transient, the same shape as `u`.
    DeltaNetLiveScanW,
    /// The prefill scan's intra-chunk `kq`, `[n_v_heads, rows,
    /// DELTA_NET_SCAN_CHUNK]`.
    DeltaNetLiveScanKq,
    /// The scan's cumulative decay `g_cs`, one F32 per V head per row.
    DeltaNetLiveDecay,
    /// `norm(out) ⊙ gate(z)`, `value_dim` wide — what `w_out` reads.
    DeltaNetLiveNormGate,
    /// `w_out`'s q8a128 operand.
    DeltaNetLiveOutOperand,
    /// `w_out`'s result, `hidden` wide in the activation dtype.
    DeltaNetLiveOut,

    /// The FFN phase's Gated Residual pre-mix — the same five carves as
    /// [`Self::HyperAttnNorm`] and its siblings, in the same order.
    HyperFfnNorm,
    /// See [`Self::HyperAttnNormOperand`].
    HyperFfnNormOperand,
    /// See [`Self::HyperAttnLowRank`].
    HyperFfnLowRank,
    /// See [`Self::HyperAttnGateAct`].
    HyperFfnGateAct,
    /// See [`Self::HyperAttnGateOperand`].
    HyperFfnGateOperand,
    /// See [`Self::HyperAttnGateRaw`].
    HyperFfnGateRaw,
    /// See [`Self::HyperAttnMixed`].
    HyperFfnMixed,
    /// FFN RMSNorm, in whatever encoding the expert GEMMs consume. Both
    /// dispatch paths, and priced dense for the reason given on
    /// [`Self::AttnNorm`].
    FfnNorm,
    /// The shared expert's fused `[gate | up]` projection, in `work_dtype`. It
    /// runs before the router — the routed half consumes the activation, so the
    /// shared half reads it first — and every shared buffer below is carved in
    /// that order. See [`SharedExpertWidths`] for which of its steps land here.
    SharedGateUp,
    /// `silu(gate)` over the shared expert's gate half.
    SharedAct,
    /// `silu(gate) ⊙ up` — what the shared down projection consumes.
    SharedGated,
    /// The shared down projection's operand: [`Self::SharedGated`] quantized to
    /// q8a128, in the generation the product was carved in.
    SharedDownOperand,
    /// The shared expert's down projection, `hidden` wide in `work_dtype` — the
    /// width of the routed combine it is summed into.
    SharedDown,
    /// The shared expert's gate projection, a KO tile wide; only column 0 is
    /// the gate, the rest are the padding rows' zeros.
    SharedGateLogits,
    /// `sigmoid` of that column: one scalar per token, and the one shared
    /// buffer whose length is rarely a multiple of the alignment.
    SharedGateSigmoid,
    /// `down · sigmoid(gate)` — the gated shared contribution
    /// [`Self::MoeSharedSum`] adds to the routed combine.
    SharedGatedOut,
    /// The router's per-token logits over every expert. Both paths.
    RouterLogits,
    /// The look-ahead routers' logits — later rows' routers applied to this
    /// layer's input in launches of their own, F32, `n_experts` columns a hop
    /// ([`WaveWidth::look_ahead_hops`]) — that the expert cache's votes are
    /// counted from.
    LookAheadLogits,
    /// Top-k routing weights, in F32. Both paths.
    RouteWeights,
    /// Top-k expert ids, as u32. Both paths.
    RouteIndices,
    /// The threaded pipeline's three assignment-indexed uploads, one `u32` per
    /// (token, expert) pair each: the gathered token ids, the weight ids in
    /// token-major order, and the permutation the scatter reads `down_out`
    /// through. Threaded pipeline only: the GPU-native path bucketizes into the
    /// dispatch tables' own workspace, off the span.
    ///
    /// **Three carves, not one table**, and the plan has to say so: each pays
    /// its own alignment pad when the assignment count is not a multiple of 64,
    /// where a single three-wide table would pay it once — 256 B under what the
    /// 35B carved at 2,100 rows, which is an overrun rather than a rounding.
    ///
    /// One upload per layer, because the pipeline computes every routed expert
    /// in one canonically-ordered call; splitting hits from misses made a
    /// token's k terms sum in a residency-dependent grouping. Measured at 9,880
    /// rows × 8 experts: three carves of 316,160 B. The plan used to charge
    /// eight tables per assignment, on the reasoning that batching made the
    /// count unknowable — 1.5 MiB a phase of ground conceded to tables that did
    /// not exist.
    RoutingTokenIds,
    /// See [`Self::RoutingTokenIds`].
    RoutingWeightIds,
    /// See [`Self::RoutingTokenIds`].
    RoutingPermutation,
    /// The per-token prefix offsets into the permutation, `rows + 1` of them.
    /// Threaded pipeline only.
    ///
    /// The one routing table whose length is not a multiple of the alignment,
    /// so the next carve pays a pad after it — the 156 B the census reports
    /// lost at 9,880 rows.
    RoutingTokenStarts,
    /// Tokens gathered into expert-major order for the grouped GEMMs.
    MoeGather,
    /// Gate projection over the gathered tokens.
    GateGemm,
    /// Up projection over the gathered tokens.
    UpGemm,
    /// Fused SwiGLU output, requantized to feed the down GEMM.
    SwigluAct,
    /// Down projection, in the accumulate dtype — which the scatter reads as it
    /// is. Both paths: `grouped_qmatmul_dev_q8a128` inherits the gathered
    /// operand's arena exactly as the pipeline's grouped GEMM does.
    ///
    /// **There is no cast after it.** The plan used to charge one, `experts ×
    /// hidden` in the compute dtype, that neither path makes —
    /// `fused_deterministic_scatter` validates the F32 operand rather than
    /// converting it. At 9,880 rows that was 323,747,840 B of a 1,816,655,360 B
    /// FFN plan that the census never saw carved.
    DownGemm,
    /// The MoE combine target the scatter accumulates into. Both paths.
    MoeCombine,
    /// `routed + gated shared` — the layer's output on a stack with a shared
    /// expert. It carves beside the routed combine because that is its first
    /// operand; the gated shared half it adds is [`Self::SharedGatedOut`].
    MoeSharedSum,

    // ── The dense FFN ───────────────────────────────────────────────────────
    // Four carves, measured on the 0.8B at 2100 rows: 62,630,400 B. No router,
    // no gather, no expert replication, and every intermediate in the compute
    // dtype — the down projection's own result is downstream of a quantize and
    // lands off the span, like the DeltaNet projections.
    /// The dense FFN's input norm.
    DenseFfnNorm,
    /// The fused `[gate | up]` projection: one GEMM, `2 × intermediate` wide.
    DenseGateUp,
    /// `silu(gate)`.
    DenseSilu,
    /// `silu(gate) ⊙ up` — the SwiGLU result the down projection consumes.
    DenseSwiglu,

    // ── The head, after the last layer ──────────────────────────────────────
    // The forward phase's whole content, and the reason it is sized in
    // sequences. Measured on the 0.8B: two carves, 497,920 B **per sequence**
    // — so `WAVE_FORWARD_BYTES`, at 16 MiB, covered 33 of them. This engine
    // composes waves of up to 64, where the phase needs 31.9 MiB and the
    // reservation would have been overrun by a forward that had already
    // launched every layer.
    /// The head's Gated Residual pre-mix — the output mix that IS the final
    /// norm on a multi-stream stack — over the **scored** rows only: they are
    /// selected before it, since the mix is row-wise. The same five carves as
    /// [`Self::HyperAttnNorm`] and its siblings, except that the head's module
    /// injects nothing, so its down projection is `low_rank` wide.
    HyperHeadNorm,
    /// See [`Self::HyperHeadNorm`].
    HyperHeadNormOperand,
    /// See [`Self::HyperHeadNorm`].
    HyperHeadLowRank,
    /// See [`Self::HyperHeadNorm`].
    HyperHeadGateAct,
    /// See [`Self::HyperHeadNorm`].
    HyperHeadGateOperand,
    /// See [`Self::HyperHeadNorm`].
    HyperHeadGateRaw,
    /// See [`Self::HyperHeadNorm`].
    HyperHeadMixed,
    /// The final norm, in the encoding the head's matmul consumes.
    HeadNorm,
    /// The float head-norm's F32 working copy, beside its `act_dtype` result.
    ///
    /// **Float sessions only.** An int8 session's `RmsNorm::forward_dynamic`
    /// emits q8a128 from one fused kernel and carves nothing else; the float arm
    /// goes through `forward_with_ticket`, whose RMSNorm materialises the row in
    /// F32 as well. Measured on Qwen2 at 60 scored rows, hidden 896: two carves,
    /// `60 × 896 × 2` and `60 × 896 × 4`, summing to the 322,560 B the arena
    /// peaked at.
    HeadNormF32,
    /// The head's logits: one row per scored row, `vocab` wide, in the compute
    /// dtype.
    ///
    /// **Both head modes.** The int8 matmul carves its output from the
    /// operand's arena and converts on the store; the float arm reaches the
    /// dequantized-weight path through `to_owned_tensor`, whose copy inherits
    /// the operand's arena as well. Measured: Qwen3.5-0.8B (int8) carves
    /// `4 × 248,320 × 2` here, and Qwen2's float head ran out of span exactly
    /// one `151,936 × 4` row into its logits once the float path stopped
    /// dropping them on the pool.
    ///
    /// Being *returned* from the forward is not what decides it — the head's
    /// span is reset per forward precisely so the logits may outlive the layer
    /// guards.
    HeadLogits,
    /// A float head's logits as its matmul writes them, F32, before they are
    /// narrowed to the compute dtype — `QMatMul::forward_live_as` off the int8
    /// path. Only where the compute dtype is narrower: an F32 session's matmul
    /// result *is* its logits.
    HeadLogitsF32,
    /// The accept walk's row selections: at each block position, the rows of
    /// the sequences still standing, gathered out of [`Self::HeadLogits`] for
    /// the chooser.
    ///
    /// The walk reads the head's logits **in place**, behind the forward's
    /// guard, rather than copying them off the span first — so what it gathers
    /// lands on the span beside them. A row is selected at most once across the
    /// walk (each sequence's row at each position), so the selections together
    /// are at most the scored rows: this is the logits' second buffer, the same
    /// width, in either head mode.
    AcceptRows,
    /// The sampler's grammar bitmaps, `[stencil_rows, vocab / 32]` `u32`: one bit
    /// per token for each row held to a set too wide to name. Written from the
    /// host beside the logits they mask and read by the sampling kernel, so a
    /// wider forward simply has more of them and the span's reset takes them all.
    StencilBitmaps,
    /// The wide residual, `[rows, streams · hidden]` F32 — on a multi-stream
    /// stack, for every forward.
    ///
    /// One buffer for the whole sweep, combined into in place by every layer,
    /// so it lives on the forward span rather than a layer's: it crosses every
    /// layer phase's reset. A resumed forward gathers its input into it, and a
    /// forward that stops short of the head hands it on from here
    /// (`docs/wave_step.md`), so it is charged whether or not the head runs.
    HyperResidual,
    /// The rows' bare embeddings, `[rows, hidden]` F32: the draft head's
    /// `embed(t)` half, gathered with the residual and read after the trunk.
    /// Only on a stack with a draft head, and only where the head runs.
    HeadRowEmbeds,
    /// The scored rows of the wide residual, gathered beside it ahead of the
    /// head's mix, `[scored, streams · hidden]` F32 — on a multi-stream stack;
    /// `[scored, hidden]` in the compute dtype on a single-stream one, whose
    /// residual is on the span as well ([`Self::ForwardResidual`]). A
    /// selection that is one run of rows is a view and carves nothing; the
    /// plan charges the gather whenever the head runs, since only the rows
    /// decide which.
    HeadScoredResidual,
    /// That gather's row index, one `u32` per scored row.
    HeadScoredRows,
    /// The ragged prefill's cumulative query offsets, one `u32` per prefill
    /// span plus one — the first of three tables the multi-stream stack builds
    /// once per forward on its forward span, whether or not the wave reaches
    /// the head: every layer reads them.
    PrefillCuSeqlens,
    /// Each prefill span's query length, one `u32` per span.
    PrefillQLens,
    /// Each prefill span's KV length after the wave, one `u32` per span.
    ///
    /// A single-stream stack builds the three tables as well — twice, for its
    /// prefill and its glue group — on its forward span, before the first
    /// layer reads them.
    PrefillKvLens,
    /// A reprojection-glue forward's three per-row scatter tables — write
    /// slice, offset within the block, forward-ahead count — on the forward
    /// span every layer's glue scatter reads them from. Single-stream stacks,
    /// which are the ones that carry glue.
    GlueWriteTables,
    /// The residual stream a single-stream stack sweeps in, `[rows, hidden]`
    /// in the compute dtype: the resident table's gather inherits the arena of
    /// the ids it reads, which are on this span, and a resumed forward gathers
    /// its input into one here. A window that stops short hands it on from
    /// here, so it is charged for every forward. A multi-stream stack's is
    /// [`Self::HyperResidual`].
    ForwardResidual,
    /// A hybrid stack's DeltaNet **decode table**: three addresses (the conv
    /// tail, the state and the timeline the row writes) per decode row per
    /// DeltaNet layer, `i64`, and each decode row's position in the wave,
    /// `u32` — two carves, built once per forward and read by every recurrent
    /// layer.
    DeltaNetDecodeTable,
    /// The wave's token ids, one `u32` per row, laid on the forward span for
    /// the embedding gather to read. On a multi-stream stack it is also the
    /// buffer the readback lays the device inputs into — and, for a wave with
    /// an input already on the host, a second upload of every row: two words a
    /// row there, the most either takes. A single-stream stack reads nothing
    /// back and takes one.
    RowTokenIds,
    /// One attention layer's sparse selection, as the model states it — see
    /// [`WaveWidth::qsa_bytes`] for why the width cannot say it alone. Carved
    /// on the attention phase, after the pre-mix and ahead of the projections.
    QsaSelection,

    /// The MoE result **narrowed to the residual's dtype**, when the experts ran
    /// in a different one. Both paths.
    ///
    /// `ffn_residual` hands the experts `work_dtype` — BF16 for an F16 session,
    /// the F16-overflow stability cast — and then calls
    /// `to_dtype_mut(out_dtype)` on the combine. That returns early when the
    /// dtypes agree and **allocates** a fresh `rows × hidden` buffer when they do
    /// not: the census shows the `to_dtype_mut` carve on every F16 session and on
    /// no BF16 one.
    ///
    /// It used to be charged unconditionally, as a second combine target "for
    /// the threaded pipeline". Either path has exactly one combine target
    /// ([`Self::MoeCombine`]); the buffer beside it was always this cast, so a
    /// BF16 session paid `rows × hidden × 2` for it on every forward — while an
    /// F16 one was told the cast was free and the charge was for something else.
    MoeResultCast,
    /// The FFN norm's operand widened to `work_dtype` — the other half of the
    /// same F16 stability cast, on the MoE arm's **float** path.
    ///
    /// A packed operand is range-safe and skips it (`q8a128` carries its own
    /// scales), and a session whose `act_dtype` is already the work dtype has
    /// nothing to widen. So this is the one buffer that needs *both* a float
    /// session and an F16 activation.
    FfnNormOperand,

    // ── The draft head's pass, after the trunk ──────────────────────────────
    // The head runs one attention block over every row of the wave, in the
    // attention phase it opens for itself. Its input is assembled first, in
    // this order: the trunk residual shifted one row right, its grouped norm,
    // the rows' embedding norm, the two halves of `eh_proj` with their q8a128
    // operands, and the sum. Then the block runs its pre-mix and its attention
    // as a trunk attention layer does — see [`Chain::HeadPass`]. Priced zero on
    // a stack without a head.
    //
    // **The embedding half is the head's attention phase's too.** The rows'
    // embeddings live on the forward span ([`Self::HeadRowEmbeds`]), but the
    // assembly hands the embedding norm the attention phase's ticket as its
    // root, so the norm, its operand and its projection carve there: measured
    // on Qwen3.8-Flash-Next at BF16 ×4, all three in the head pass's attention
    // generation and none in the forward one.
    /// The shifted residual, `[rows, streams · hidden]` F32.
    HeadShift,
    /// Its grouped norm, the same width.
    HeadHiddenNorm,
    /// The rows' embeddings normed, `[rows, hidden]` F32.
    HeadEmbedNorm,
    /// That norm quantized for `eh_proj`'s embedding half — int8 sessions only.
    HeadEmbedOperand,
    /// The embedding half's projection, `[rows, hidden]` F32.
    HeadEmbedProj,
    /// The hidden norm quantized for `eh_proj`'s hidden half, one row per
    /// stream — int8 sessions only.
    HeadHiddenOperand,
    /// The hidden half's projection, `[rows · streams, hidden]` F32.
    HeadHiddenProj,
    /// Their broadcast sum: the block's wide input, `[rows, streams · hidden]`.
    HeadInput,

    // ── A single-stream draft head's pass (the Qwen3.5 lineage's NextN) ─────
    // The same pass on a stack with ordinary layer norms: the trunk's final
    // norm over every row, that shifted one row right, the two input norms,
    // their concatenation, its q8a128 operand on an int8 session, and
    // `eh_proj` — then the head's attention block, as a trunk attention layer
    // runs it. Every buffer is `[rows, ·]`, in the compute dtype but for the
    // operand. Priced zero without a head ([`ModelGeometry::mtp_head`]) and on
    // a wave that stops short of it.
    /// The rows' embeddings, gathered for the head on the **forward** phase —
    /// beside the ids they are read with, and outside the head's attention
    /// generation, which the gather's staging would otherwise share.
    MtpRowEmbeds,
    /// The trunk's final norm over every row, `[rows, hidden]`.
    MtpTrunkNorm,
    /// That norm shifted one row right, `[rows, hidden]`.
    MtpShift,
    /// The embeddings normed (`enorm`), `[rows, hidden]`.
    MtpEmbedNorm,
    /// The shifted hidden normed (`hnorm`), `[rows, hidden]`.
    MtpHiddenNorm,
    /// The two norms side by side, `[rows, 2 · hidden]`.
    MtpInputCat,
    /// That concatenation quantized to q8a128 for `eh_proj` — int8 sessions
    /// only. `MtpInput::forward` hands `eh_proj` the float concatenation, and
    /// the int8 matmul quantizes it at the call into the concatenation's arena,
    /// so it lands on the head's attention phase beside it: `[rows, 2 · hidden]`
    /// of q8a128, the multi-stream head's [`Self::HeadHiddenOperand`] by
    /// another name.
    ///
    /// Undeclared, it was the head pass's one carve the plan did not know
    /// about, and on the 35B-A3B (`2 · hidden` = `attn_cols` = 4096) exactly the
    /// size of the [`Self::OProjOperand`] the pass carves second-to-last — so
    /// wherever the head pass set the attention phase's price the span ran out
    /// on that operand (`4,704,768 B asked`, 1,021 prefill rows).
    MtpInputOperand,
    /// `eh_proj`'s output: the head block's input, `[rows, hidden]`.
    MtpInput,

    // ── The PLE block, ahead of its layer's mixer ───────────────────────────
    // Its own attention generation, opened and closed before the mixer's: the
    // gathered table rows, the stacked key|value projection, the conv's normed
    // input and the span table. Everything else the block computes is written
    // into the residual in place. Priced zero on a stack without one.
    /// The gathered table rows, `[rows, hidden]` F32.
    PleRows,
    /// The key and value projections stacked, `[rows, streams · hidden + hidden]`.
    PleKeyValue,
    /// The conv's normed input, `[rows, streams · hidden]` F32.
    PleNormalized,
    /// Four 64-bit words per sequence — bounded by the scored rows, which are
    /// at least one per sequence.
    PleSpanTable,
}
