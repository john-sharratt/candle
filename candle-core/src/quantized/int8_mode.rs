//! The numeric mode of the int8 tensor-core matmul path, and the rule that picks
//! a weight twin and a mode for a given card.
//!
//! `Off` keeps the FP16 reference. `Performance` pairs int8 activations with the
//! same-width weight twin, and `Precision` pairs them with the stepped-up twin,
//! whose extra bit absorbs the re-quantisation error. Auto-selection judges the
//! model's size against the card's total VRAM rather than what is free, so the
//! same model pack is named on every load. The sum-scale choice is a property of
//! the stored bytes, and the normalised form keeps activations whose block sums
//! would overflow f16 finite.

/// Inference numeric mode for the q8a128 int8 tensor-core matmul path.
///
/// A single knob selecting both halves of the int8 pairing (weight repack via
/// [`QMatMul::repack_for_optimization`] and activation conversion via `cuda::to_dynamic`):
///
/// - [`Int8Mode::Off`] — FP16 GEMM. Weights repack to the gemx float layout, activations stay
///   float. The numeric reference path; no int8 anywhere.
/// - [`Int8Mode::Performance`] — int8 with the *same-width* KO weight twin (`Q4_K`→`Q4_KO`,
///   `Q5_K`→`Q5_KO`, …). Fastest and smallest; takes the per-128 granularity hit on the weight.
/// - [`Int8Mode::Precision`] — int8 with the *stepped-up* KO weight twin (`Q4_K`→`Q5_KO`,
///   `Q5_K`→`Q6_KO`, …). The extra weight bit absorbs the per-128 re-quant error, leaving int8
///   near-lossless versus the source quant. Activations (q8a128) are identical to Performance —
///   only the weight twin differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Int8Mode {
    /// FP16 GEMM — no int8. The numeric reference.
    #[default]
    Off,
    /// int8 with the same-width KO weight twin. Fastest, lossier.
    Performance,
    /// int8 with the stepped-up KO weight twin. Near-lossless versus the source quant.
    Precision,
}

/// How a q8a128 block stores the per-128 `Σx` in its `ds[0].y` header field.
///
/// **A property of the bytes, not of the kernel.** It is chosen by whichever
/// producer wrote the block and carried on `cuda::Q8a128Operand`, so the matmul
/// never has to guess which convention it is reading. Both halves cost the same:
/// the producer stores one value either way, and the matmul rebuilds `Σx` with
/// one `fmaf` whose two coefficients are hoisted from this choice — so it is a
/// runtime argument, never a template parameter, and adds no kernel variants.
///
/// [`SumScale::Raw`] is the default and is what every language model here uses;
/// its bytes and its arithmetic are bit-identical to the unparameterised path
/// that preceded this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SumScale {
    /// `ds[0].y = Σx` — the raw per-128 sum.
    #[default]
    Raw,
    /// `ds[0].y = Σx / amax`.
    ///
    /// **For activations whose block sums overflow f16.** Both header fields are
    /// f16 and `|Σx|` can reach `128 · amax`, so any activation whose per-128
    /// sums pass 65504 stores `+inf` and every dot product touching that block
    /// becomes NaN. An LLM's stay under 10³ and never come near it; Z-Image's
    /// SwiGLU intermediate reaches ≈2×10⁵, which produced a black image with no
    /// other symptom.
    ///
    /// Normalising bounds the field at `|Σx/amax| ≤ 128` whatever the
    /// activation's magnitude, and costs no precision — f16's relative precision
    /// is scale-free, so the normalised value carries the same ~2⁻¹¹ the raw one
    /// did, and rather more at the bottom, where a small raw `Σx` went
    /// subnormal. `amax == 0` stores 0, which reconstructs to 0.
    ///
    /// It is opt-in rather than universal because it changes the stored bytes:
    /// a model whose thresholds were derived against raw sums would have them
    /// invalidated by a silent format change.
    ByAmax,
}

impl SumScale {
    /// The two coefficients that rebuild `Σx` from the stored field, as
    /// `sum = ds.y · fmaf(ds.x, a, b)`.
    ///
    /// `ds.x` is `amax/127`, so `(127, 0)` gives `ds.y · amax` = `Σx` for
    /// [`Self::ByAmax`], and `(0, 1)` gives `ds.y · 1` — exactly the raw field,
    /// with no rounding introduced — for [`Self::Raw`].
    pub fn reconstruct_coeffs(self) -> (f32, f32) {
        match self {
            Self::Raw => (0.0, 1.0),
            Self::ByAmax => (127.0, 0.0),
        }
    }

    /// The flag as the kernels take it: `0` raw, `1` normalised.
    pub fn as_code(self) -> i32 {
        match self {
            Self::Raw => 0,
            Self::ByAmax => 1,
        }
    }
}

impl Int8Mode {
    /// True when the int8 tensor-core path is active (any mode other than [`Int8Mode::Off`]).
    pub fn is_int8(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Auto-select the numeric mode for `device`: [`Int8Mode::Precision`] when the device can run
    /// the int8 `m16n8k32` tensor-core MMA (CUDA, compute capability >= 8.0 / Ampere+), otherwise
    /// [`Int8Mode::Off`] (the FP16 reference).
    ///
    /// **Precision, and the margin is model-dependent — do not generalise it from one model.**
    /// The two modes differ only in the weight twin (the q8a128 activation is identical), so the
    /// choice is a throughput/accuracy dial. Measured on an RTX PRO 5000 Blackwell (sm_120):
    ///
    /// * Qwen3.5-9B is **insensitive** — every C-ladder rung valid in both, identical compression
    ///   ratios, throughput inside the run-to-run band (C8 5823 vs 5831 bulk).
    /// * Llama-3 is **not** — the same swap takes its ladder from green to C6 0/1, C7 0/1 and
    ///   C8 9/10. Precision's stepped-up twin is doing real work there.
    ///
    /// So a model that shows no difference says nothing about the next one, and the default holds
    /// the accurate twin. `Performance` stays available where a gate has measured it safe (the
    /// DeepSeek-V4 engine pins it) or where the larger twin does not fit — see
    /// [`Int8Mode::auto_sized`].
    pub fn auto(device: &crate::Device) -> Self {
        match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(d) if d.supports_int8_mma() => Self::Precision,
            _ => Self::Off,
        }
    }

    /// VRAM-aware [`Int8Mode::auto`]: on an int8-MMA-capable CUDA device, picks
    /// [`Int8Mode::Precision`] (the accurate, but *larger* stepped-up weight
    /// twin) only when the weights leave comfortable headroom — the model is at
    /// most ~70% of the card's VRAM, so the KV cache, activations, and (MoE) hot
    /// experts still fit — and otherwise drops to [`Int8Mode::Performance`] (the
    /// smaller same-width twin) so a tight model still fits. `Off` (FP16) on CPU
    /// / non-int8 devices, or [`Int8Mode::Performance`] if VRAM can't be queried.
    ///
    /// The fallback is a **capability** judgement, not a quality one: dropping to
    /// the smaller twin costs accuracy on some models (see [`Int8Mode::auto`]),
    /// and is worth it only when the accurate twin would not fit at all.
    ///
    /// **Judged against the card's total, not what is free right now.** The
    /// answer names a model pack (`{stem}.{mode}.pack.gguf`), so it has to be the
    /// same on every load on the same card: judged against free VRAM, another
    /// process holding a few GiB turned a Precision card into a Performance one,
    /// and the load built a second pack — downloading its sources again — beside
    /// the one that already served it.
    ///
    /// `model_bytes` is the on-disk quantized weight size (e.g. the GGUF length).
    // `model_bytes` is only weighed against the card's VRAM, which is a CUDA
    // query; every other device answers `Off` without consulting it.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn auto_sized(device: &crate::Device, model_bytes: usize) -> Self {
        match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(d) if d.supports_int8_mma() => match device.mem_get_info() {
                Ok((_, total)) => Self::sized_for_card(model_bytes, total),
                Err(_) => Self::Performance,
            },
            _ => Self::Off,
        }
    }

    /// [`Int8Mode::auto_sized`]'s choice on an int8-capable card of
    /// `card_bytes`: Precision when `model / card ≤ 7/10`, Performance otherwise.
    pub fn sized_for_card(model_bytes: usize, card_bytes: usize) -> Self {
        if model_bytes.saturating_mul(10) <= card_bytes.saturating_mul(7) {
            Self::Precision
        } else {
            Self::Performance
        }
    }
}
