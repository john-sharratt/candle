//! Provides the model geometries and wave shapes the plan tests are written
//! against.
//!
//! Each fixture is a real checkpoint's shape, chosen to be unlike the others so a
//! rule that holds for one cannot pass by accident: a Qwen3 MoE, a gated
//! partial-rotary 0.8B, a Llama-shaped dense stack with no per-head norms, and
//! a multi-stream stack with a draft head. The per-row chains that census-pinned
//! tests compare against are kept beside them, with each itemised carve named.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    /// What a prefill's slot headers add to its attention chain over `spans`
    /// sequences: a header each and the glue group's room beside them, and the
    /// alignment the carve after them starts on — one sequence's 288 bytes walk
    /// as 512.
    pub(in super::super) fn prefill_headers(spans: usize) -> usize {
        (spans * SLOT_HEADER_BYTES + BUMP_ALIGNMENT).div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT
    }

    /// A prefill wave of `rows` tokens over one sequence.
    ///
    /// The shape most of these cases want: they vary one number and ask how the
    /// cost moves, and the prefill chain is the wider of the two. The cases that
    /// are *about* the split — the decode chain, the head — build their widths
    /// explicitly.
    pub(in super::super) fn w(rows: usize) -> WaveWidth {
        WaveWidth::prefill(rows, 1)
    }

    /// An empty wave, for the bounds that ask "how much prefill fits at all".
    /// The one sequence is the one the prefill rows will belong to.
    pub(in super::super) fn empty_head() -> WaveWidth {
        WaveWidth::prefill(0, 1)
    }

    /// Qwen3.8-Flash-Next's pre-mix geometry: 4 streams over `hidden` 2560
    /// through a rank-320 gate, on the MoE geometry, with F32 activations.
    pub(in super::super) fn hyper_moe() -> ModelGeometry {
        ModelGeometry {
            hidden: 2560,
            act_dtype: DType::F32,
            accum_dtype: DType::F32,
            hyper: Some(HyperWidths {
                streams: 4,
                low_rank: 320,
                draft_head: false,
                ple: false,
                mixer_dtype: DType::F32,
            }),
            ..moe()
        }
    }

    /// [`hyper_moe`] with a draft head and a PLE block, as Flash-Next runs.
    pub(in super::super) fn hyper_moe_full() -> ModelGeometry {
        ModelGeometry {
            hyper: Some(HyperWidths {
                streams: 4,
                low_rank: 320,
                draft_head: true,
                ple: true,
                mixer_dtype: DType::F32,
            }),
            ..hyper_moe()
        }
    }

    /// Qwen3.6-35B-A3B's shapes as an int8 BF16 session with its NextN head:
    /// hidden 2048, 16 query and 2 KV heads at 256, a 3:1 hybrid whose 30
    /// DeltaNet layers run a `conv_dim` 8192 / `value_dim` 4096 mixer over 32
    /// V heads, and a 256-expert MoE with a shared expert.
    pub(in super::super) fn qwen36_35b_a3b_with_head() -> ModelGeometry {
        ModelGeometry {
            hidden: 2048,
            vocab: 248_320,
            intermediate: 512,
            n_head: 16,
            n_kv_head: 2,
            head_dim: 256,
            experts_per_tok: 8,
            n_experts: 256,
            delta_net: Some(DeltaNetWidths {
                conv_dim: 8192,
                value_dim: 4096,
                n_v_heads: 32,
                layers: 30,
            }),
            shared_expert: Some(SharedExpertWidths {
                intermediate: 512,
                gate_cols: 32,
            }),
            mtp_head: true,
            ..gated_partial_rotary()
        }
    }

    /// Flash-Next's DeltaNet geometry on the pre-mix stack: 16 K heads and 48 V
    /// heads of 128, so `conv_dim` 10240 and `value_dim` 6144.
    pub(in super::super) fn hyper_hybrid() -> ModelGeometry {
        ModelGeometry {
            delta_net: Some(DeltaNetWidths {
                conv_dim: 10240,
                value_dim: 6144,
                n_v_heads: 48,
                layers: 36,
            }),
            ..hyper_moe()
        }
    }

    /// Qwen3-30B-A3B's real shapes.
    pub(in super::super) fn moe() -> ModelGeometry {
        ModelGeometry {
            hidden: 2048,
            vocab: 151_936,
            intermediate: 768,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            experts_per_tok: 8,
            n_experts: 128,
            // Every layer attends, so there is no mixer chain to compare
            // against and the attention chain sizes the phase alone.
            delta_net: None,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Every census these fixtures are pinned against was taken on an
            // int8 session, where the norm's fused epilogue emits q8a128.
            packed_norm: true,
            packed_head: true,
            gated_qkv: false,
            partial_rotary: false,
            hyper: None,
            shared_expert: None,
            qkv_bias: false,
            decode_q8_context: true,
            // The fused segmented projection and a `[batch, seq, ..]` wave —
            // the twelve carves `MEASURED_ATTN_PREFILL_PER_ROW` itemises.
            fused_qkv: true,
            head_qk_norm: true,
            head_norm_reshapes: true,
            mtp_head: false,
        }
    }

    /// Qwen3.5-0.8B's real shapes: gated `[q | gate]` projection and 64-of-256
    /// partial rotary. The head count is 8, read off the census (the q matmul
    /// emits `2 · 8 · 256` columns; an earlier fixture said 16 ungated — which
    /// priced the same `qkv_cols` by accident and hid the gate's whole
    /// downstream chain). The FFN width is the real one, 3584; the DeltaNet
    /// half of this hybrid is priced by its own widths, not folded in here.
    pub(in super::super) fn gated_partial_rotary() -> ModelGeometry {
        ModelGeometry {
            hidden: 1024,
            // The 0.8B's real vocabulary, read off the loaded checkpoint.
            vocab: 248_320,
            intermediate: 3584,
            n_head: 8,
            n_kv_head: 2,
            head_dim: 256,
            experts_per_tok: 1,
            n_experts: 1,
            // Three layers in four are DeltaNet on this lineage.
            delta_net: Some(DeltaNetWidths {
                conv_dim: 6144,
                value_dim: 2048,
                n_v_heads: 16,
                layers: 18,
            }),
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Every census these fixtures are pinned against was taken on an
            // int8 session, where the norm's fused epilogue emits q8a128.
            packed_norm: true,
            packed_head: true,
            gated_qkv: true,
            partial_rotary: true,
            hyper: None,
            shared_expert: None,
            qkv_bias: false,
            decode_q8_context: true,
            // Three separate projections, and a wave already packed `[rows, ..]`
            // — so neither the K/V narrows nor the head-norm reshapes exist.
            fused_qkv: false,
            head_qk_norm: true,
            head_norm_reshapes: false,
            // The 0.8B carries no NextN head.
            mtp_head: false,
        }
    }

    /// **Llama-3-8B's shapes, and the no-per-head-norm case.**
    ///
    /// Kept deliberately unlike the two above: Llama and Qwen2 have no
    /// `attn_q_norm`/`attn_k_norm` weight at all, so six attention buffers must
    /// price zero here. Charging them was six buffers those stacks never
    /// allocate, two of them `attn_cols` wide.
    pub(in super::super) fn dense() -> ModelGeometry {
        ModelGeometry {
            hidden: 4096,
            vocab: 128_256,
            intermediate: 12288,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            experts_per_tok: 1,
            n_experts: 1,
            delta_net: None,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Every census these fixtures are pinned against was taken on an
            // int8 session, where the norm's fused epilogue emits q8a128.
            packed_norm: true,
            packed_head: true,
            gated_qkv: false,
            partial_rotary: false,
            hyper: None,
            shared_expert: None,
            qkv_bias: false,
            decode_q8_context: true,
            fused_qkv: true,
            head_qk_norm: false,
            head_norm_reshapes: true,
            mtp_head: false,
        }
    }

    pub(in super::super) const MIB: usize = 1 << 20;

    /// Per-row cost of the attention chain a **prefill** group runs on
    /// Qwen3-30B-A3B: the twelve carves a `wave-census` build measured, every
    /// one of them a whole number of bytes per row, and the int8 `o_proj`'s
    /// two after them. The census predates the quantize inheriting the
    /// context's arena; Qwen2's gate carved the operand on the span and ran
    /// out of it, 34,560 B into an unpriced `o_proj` quantize.
    ///
    /// ```text
    /// [ 0]    2304  AttnNorm       q8a128 over hidden
    /// [ 1]   10240  QkvProjection  the fused segmented launch
    /// [ 2]    8192  QSplit         narrow out of the fused buffer
    /// [ 3]    8192  QNormIn        flatten for the head-wise norm
    /// [ 4]    8192  QNormOut       the norm itself
    /// [ 5]    8192  QHeadsPacked   transpose back
    /// [ 6]    1024  KSplit
    /// [ 7]    1024  KNormIn
    /// [ 8]    1024  KNormOut
    /// [ 9]    1024  KHeadsPacked
    /// [10]    1024  VContiguous
    /// [11]    8192  AttnOutput     the paged prefill kernel's context
    /// [12]    4608  OProjOperand   q8a128 over the attention columns
    /// [13]    4096  OProjOutput
    /// ```
    pub(in super::super) const MEASURED_ATTN_PREFILL_PER_ROW: usize = 67328;

    /// The same for a **decode** group: nine carves, and a different set — the
    /// `seq == 1` reshapes are free, and the context comes back already packed,
    /// so `o_proj` runs off it and lands on the span.
    ///
    /// ```text
    /// [0]  2304  AttnNorm
    /// [1] 10240  QkvProjection
    /// [2]  8192  QSplit
    /// [3]  8192  QNormOut
    /// [4]  1024  KSplit
    /// [5]  1024  KNormOut
    /// [6]  1024  VContiguous
    /// [7]  4608  DecodeContext   q8a1024, emitted by the decode kernel
    /// [8]  4096  OProjOutput
    /// ```
    pub(in super::super) const MEASURED_ATTN_DECODE_PER_ROW: usize = 40704;

    /// The same decode chain at **one row**, where `QSplit`, `KSplit` and
    /// `VContiguous` are not carved: the narrows out of a single fused row are
    /// already contiguous. Read off the per-forward pair on the Qwen3-30B-A3B
    /// gate — planned 40,704 B, used 30,464 B at `0 prefill + 1 decode rows`.
    pub(in super::super) const MEASURED_ATTN_DECODE_ONE_ROW: usize = 30464;

    /// The FFN chain per row on a **BF16** int8 session, excluding the routing
    /// tables (which do not scale by the row alone):
    ///
    /// ```text
    ///  2304  FfnNorm        q8a128 over hidden
    ///   256  RouterLogits
    ///    64  RouteWeights + RouteIndices
    ///  4096  MoeCombine
    /// 18432  MoeGather      8 × q8a128 over hidden
    /// 49152  Gate + Up      8 × 768 × F32, twice
    ///  6912  SwigluAct      8 × q8a128 over 768
    /// 65536  DownGemm       8 × 2048 × F32
    /// ```
    pub(in super::super) const MEASURED_FFN_PIPELINE_PER_ROW: usize = 146752;

    /// Two whole FFN generations off the census, to the byte — the plan must
    /// price each exactly, routing tables and alignment pad included.
    ///
    /// * BF16, 4,960 rows: 14 carves, no result cast, 124 B lost to alignment.
    /// * F16, 9,880 rows: 15 carves, the `to_dtype_mut` result cast among them,
    ///   156 B lost to alignment.
    pub(in super::super) const MEASURED_FFN_BF16_4960: usize = 728_386_048;

    pub(in super::super) const MEASURED_FFN_F16_9880: usize = 1_491_366_400;

    /// The same generation on Qwen3.5-35B-A3B, whose MoE adds a shared expert:
    /// nine more carves (gate/up, SiLU, product, the down projection's q8a128
    /// operand and its result, gate tile, sigmoid, the gated product, the final
    /// sum) and the alignment pads the sigmoid, the down operand and the routing
    /// tables leave.
    ///
    /// * BF16, 2,100 rows: 23 carves, 836 B lost to alignment.
    /// * F16, 1,070 rows: 24 carves — the result cast again — 808 B lost.
    pub(in super::super) const MEASURED_FFN_35B_BF16_2100: usize = 305_437_440;

    pub(in super::super) const MEASURED_FFN_35B_F16_1070: usize = 160_010_752;

    /// Qwen3.5-35B-A3B's FFN shapes. The FFN chain takes nothing from the
    /// attention side, so only the fields it reads are the checkpoint's.
    pub(in super::super) fn moe_35b() -> ModelGeometry {
        ModelGeometry {
            intermediate: 512,
            experts_per_tok: 8,
            n_experts: 256,
            shared_expert: Some(SharedExpertWidths {
                intermediate: 512,
                gate_cols: 32,
            }),
            ..moe()
        }
    }
}
