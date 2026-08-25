//! Zero-cost operand guards — the engine-wide alternative to a defensive
//! `to_dtype` / `contiguous` on a value that should already be right.
//!
//! **Validate, don't convert.** Wherever a value reaches a consumer that has a
//! fixed requirement — a hand-written kernel wrapper, a buffer another pass
//! fills, a cache row a later step reads back — state the requirement here
//! rather than papering over it at the call site.
//!
//! A kernel that indexes `base + row * row_len` has two hard requirements on
//! every operand: the element type it was compiled against, and a dense layout.
//! The tempting way to satisfy them is `t.to_dtype(WANT)?.contiguous()?`, and
//! that is wrong twice over.
//!
//! * It violates hot-path invariant 1 (no `to_dtype` in the loop) and invariant
//!   2 (no allocate-plus-copy to materialise a layout). When the operand does
//!   NOT already match, each call is a full-tensor memory pass, per layer, per
//!   step — and it is invisible, because the code reads as a cast rather than as
//!   a copy.
//! * When the operand DOES already match — which is the case everywhere on the
//!   engine path — both calls are dead. They cost nothing, and they also protect
//!   nothing: they silently absorb a caller that starts handing over the wrong
//!   type, converting a loud bug into a quiet slowdown.
//!
//! These guards state the requirement instead of papering over it. They read
//! layout metadata only — no allocation, no launch, no device work — so they are
//! free to call on the hot path, and a caller that violates the contract gets an
//! error naming the operand rather than a hidden per-layer copy.
//!
//! The rule this encodes: **a consumer VALIDATES its operands; it does not
//! CONVERT them.** If a genuinely different type or layout has to be supported,
//! the fix is to teach the consumer to read it (a template parameter, a stride
//! argument, a producer that emits the right type), not to rewrite the tensor at
//! the call site.
//!
//! `to_dtype` is for a conversion the design actually calls for — an F32
//! accumulator deliberately narrowed for storage, a table built once at load.
//! Reach for these guards for the other case, where the two types are *supposed*
//! to agree and the cast is there in case they do not.

use candle::{DType, LiveTensor, Result, Tensor};

/// Require `t` to already be `want`.
///
/// Takes a `LiveTensor<'_>` rather than a `Tensor` so it reaches **wave-scoped**
/// operands too — `Tensor` is `LiveTensor<'static>`, so every owned-tensor
/// caller still passes unchanged, but the residual stream and the per-layer
/// intermediates that live on a wave arena are exactly where this guard earns
/// its keep.
pub fn expect_dtype(t: &LiveTensor<'_>, want: DType, what: &str) -> Result<()> {
    if t.dtype() != want {
        candle::bail!(
            "{what}: kernel operand is {:?}, expected {want:?} — the wrapper validates \
             operands rather than converting them (hot-path invariant 1); widen the kernel \
             or fix the producer",
            t.dtype()
        );
    }
    Ok(())
}

/// Require `t` to be densely packed **and to start at its storage base**, so
/// `base + row * row_len` addresses it.
///
/// The offset check is not pedantry. The kernel wrappers take their pointer with
/// `cuda_f32_ptr!`, which reads the storage's base pointer and **discards the
/// layout** — so a view that is stride-contiguous but offset into its storage is
/// silently read from the wrong place. The dangerous shape is exactly the one
/// the wave produces: on a `[1, rows, hc, d]` residual stream, `narrow(1, k, n)`
/// leaves every stride intact (the leading dim is 1, so nothing is skipped) and
/// `is_contiguous()` returns TRUE, while `start_offset` is `k · hc · d`. The
/// kernel would read from row 0 and return plausible, wrong numbers.
///
/// A caller that hits this wants either the full tensor or a wrapper that
/// forwards the offset — not a copy.
pub fn expect_dense(t: &Tensor, what: &str) -> Result<()> {
    if !t.is_contiguous() {
        candle::bail!(
            "{what}: kernel operand has layout {:?} stride {:?}, which is not dense — the \
             wrapper validates operands rather than copying them (hot-path invariant 2); \
             give the kernel a stride argument or produce the layout directly",
            t.dims(),
            t.stride()
        );
    }
    if t.layout().start_offset() != 0 {
        candle::bail!(
            "{what}: kernel operand is dense but starts at element {} of its storage, and the \
             launch path takes the storage BASE pointer — the kernel would read from offset 0. \
             Pass the whole tensor, or thread the offset through to the launch",
            t.layout().start_offset()
        );
    }
    Ok(())
}

/// Both checks, which is what a kernel operand almost always needs.
pub fn expect_dense_dtype(t: &Tensor, want: DType, what: &str) -> Result<()> {
    expect_dtype(t, want, what)?;
    expect_dense(t, what)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn accepts_a_dense_operand_of_the_right_type() -> Result<()> {
        let t = Tensor::zeros((4, 8), DType::F32, &Device::Cpu)?;
        expect_dense_dtype(&t, DType::F32, "probe")?;
        Ok(())
    }

    /// The whole point: a wrong dtype is an ERROR, not a silent conversion.
    #[test]
    fn rejects_the_wrong_dtype() -> Result<()> {
        let t = Tensor::zeros((4, 8), DType::BF16, &Device::Cpu)?;
        let e = expect_dtype(&t, DType::F32, "probe")
            .unwrap_err()
            .to_string();
        assert!(e.contains("BF16") && e.contains("F32"), "{e}");
        Ok(())
    }

    /// A transpose is the layout that would have the kernel read garbage.
    #[test]
    fn rejects_a_non_dense_layout() -> Result<()> {
        let t = Tensor::zeros((4, 8), DType::F32, &Device::Cpu)?.t()?;
        assert!(expect_dense(&t, "probe").is_err());
        Ok(())
    }

    /// **An offset view is contiguous and still wrong**, because the launch path
    /// takes the storage base pointer and drops the layout.
    ///
    /// This test previously asserted the opposite — it accepted `narrow(0, 1, 2)`
    /// as "a dim-0 narrow stays dense" — which was true about the strides and
    /// irrelevant to what the kernel actually reads.
    #[test]
    fn rejects_a_dense_view_that_is_offset_into_its_storage() -> Result<()> {
        let n = Tensor::zeros((4, 8), DType::F32, &Device::Cpu)?.narrow(0, 1, 2)?;
        assert!(n.is_contiguous(), "precondition: strides say dense");
        let e = expect_dense(&n, "probe").unwrap_err().to_string();
        assert!(e.contains("starts at element 8"), "{e}");

        // The wave's real shape: a token-window narrow of `[1, rows, hc, d]` keeps
        // EVERY stride (leading dim 1), so `is_contiguous()` is true while the
        // data begins `k·hc·d` elements in.
        let stream = Tensor::zeros((1, 6, 2, 4), DType::F32, &Device::Cpu)?;
        let win = stream.narrow(1, 2, 3)?;
        assert!(win.is_contiguous(), "precondition: strides say dense");
        assert!(
            expect_dense(&win, "probe").is_err(),
            "offset window must be rejected"
        );

        // Offset zero is the accepted case, narrowed or not.
        expect_dense(&stream.narrow(1, 0, 3)?, "probe")?;
        expect_dense(&stream, "probe")?;
        Ok(())
    }
}
