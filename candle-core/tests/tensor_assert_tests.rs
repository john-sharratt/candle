//! Raw-value tests for the `tensor-assert` kernel.
//!
//! Every assertion here is an exact count or an exact value, never a tolerance:
//! the whole reason this instrument exists is to be believed about where a NaN
//! came from, and a probe that is approximately right about how many it saw is
//! not evidence. The counts are chosen so the expected number is arithmetic on
//! the input, not a property of the kernel's tiling.
//!
//! Slot names are unique per test because the slot table is process-global and
//! the tests run in parallel — sharing a name would let one test's tensor
//! accumulate into another's report.

#![cfg(all(feature = "cuda", feature = "tensor-assert"))]

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::tensor_assert::find;
use candle_core::{DType, Device, Result, Tensor};

fn dev() -> Result<Device> {
    Device::new_cuda(0)
}

#[test]
fn counts_nan_and_inf_exactly_and_reports_the_finite_range() -> Result<()> {
    let d = dev()?;
    // 3 NaN, 2 Inf (one of each sign), 5 finite spanning [-4.0, 6.0].
    let vals: Vec<f32> = vec![
        f32::NAN,
        1.0,
        f32::INFINITY,
        -4.0,
        f32::NAN,
        6.0,
        f32::NEG_INFINITY,
        0.5,
        f32::NAN,
        2.0,
    ];
    let t = Tensor::from_vec(vals, (10,), &d)?;
    t.assert("test::mixed");

    let f = find(&d, "test::mixed")?.expect("the site must be registered");
    assert_eq!(f.nan, 3, "NaN count");
    assert_eq!(f.inf, 2, "Inf count (both signs)");
    assert_eq!(f.elems, 10, "every element must be examined");
    assert_eq!(f.min, Some(-4.0), "min over the FINITE values only");
    assert_eq!(f.max, Some(6.0), "max over the FINITE values only");
    assert!(f.is_bad());
    assert!(f.seq.is_some(), "a bad slot must carry an order ticket");
    Ok(())
}

#[test]
fn a_clean_tensor_is_not_bad_and_carries_no_ticket() -> Result<()> {
    let d = dev()?;
    let t = Tensor::from_vec(vec![-1.0f32, 0.0, 3.5, 2.0], (4,), &d)?;
    t.assert("test::clean");

    let f = find(&d, "test::clean")?.expect("registered");
    assert_eq!((f.nan, f.inf), (0, 0));
    assert_eq!(f.elems, 4);
    assert_eq!(f.min, Some(-1.0));
    assert_eq!(f.max, Some(3.5));
    assert!(!f.is_bad());
    assert_eq!(f.seq, None, "a clean slot must never be stamped");
    Ok(())
}

#[test]
fn a_strided_view_examines_only_the_view_not_the_backing_buffer() -> Result<()> {
    let d = dev()?;
    // Row 1 holds the only NaN; the view selects rows 0 and 2, so a kernel that
    // walked the whole allocation instead of the view would report nan == 1.
    let vals: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        f32::NAN, 9.0, 9.0, 9.0, // row 1
        -2.0, 5.0, 5.0, 5.0, // row 2
    ];
    let t = Tensor::from_vec(vals, (3, 4), &d)?;

    let row0 = t.narrow(0, 0, 1)?;
    row0.assert("test::view_row0");
    let f = find(&d, "test::view_row0")?.expect("registered");
    assert_eq!(f.nan, 0, "row 0 holds no NaN");
    assert_eq!(f.elems, 4, "exactly the view's element count");
    assert_eq!(f.min, Some(1.0));
    assert_eq!(f.max, Some(4.0));

    // A genuinely non-contiguous view: every row's first column.
    let col0 = t.narrow(1, 0, 1)?;
    assert!(!col0.is_contiguous(), "the test needs the strided path");
    col0.assert("test::view_col0");
    let f = find(&d, "test::view_col0")?.expect("registered");
    assert_eq!(f.nan, 1, "column 0 holds exactly the one NaN");
    assert_eq!(f.elems, 3, "one element per row, not the whole buffer");
    assert_eq!(f.min, Some(-2.0));
    assert_eq!(f.max, Some(1.0));
    Ok(())
}

#[test]
fn repeated_asserts_accumulate_into_the_same_slot() -> Result<()> {
    let d = dev()?;
    let t = Tensor::from_vec(vec![f32::NAN, 1.0, 2.0], (3,), &d)?;
    t.assert("test::accumulate");
    t.assert("test::accumulate");
    t.assert("test::accumulate");

    let f = find(&d, "test::accumulate")?.expect("registered");
    assert_eq!(f.nan, 3, "one NaN per pass, three passes");
    assert_eq!(f.elems, 9, "three elements per pass, three passes");
    assert_eq!(f.min, Some(1.0));
    assert_eq!(f.max, Some(2.0));
    Ok(())
}

#[test]
fn assert_once_folds_exactly_one_pass_however_often_it_is_called() -> Result<()> {
    let d = dev()?;
    let t = Tensor::from_vec(vec![f32::NAN, 4.0], (2,), &d)?;
    for _ in 0..5 {
        t.assert_once("test::once");
    }
    let f = find(&d, "test::once")?.expect("registered");
    assert_eq!(f.nan, 1, "five calls, one pass");
    assert_eq!(f.elems, 2);
    Ok(())
}

#[test]
fn half_precision_inputs_are_examined_in_their_own_width() -> Result<()> {
    let d = dev()?;
    for (dt, name) in [
        (DType::F16, "test::f16"),
        (DType::BF16, "test::bf16"),
    ] {
        let t = Tensor::from_vec(vec![f32::NAN, -2.0f32, 8.0, f32::INFINITY], (4,), &d)?
            .to_dtype(dt)?;
        t.assert(name);
        let f = find(&d, name)?.expect("registered");
        assert_eq!(f.nan, 1, "{dt:?} NaN count");
        assert_eq!(f.inf, 1, "{dt:?} Inf count");
        assert_eq!(f.elems, 4, "{dt:?} element count");
        // -2.0 and 8.0 are exact in both half formats, so this stays a raw
        // equality rather than a tolerance.
        assert_eq!(f.min, Some(-2.0), "{dt:?} min");
        assert_eq!(f.max, Some(8.0), "{dt:?} max");
    }
    Ok(())
}

#[test]
fn an_integer_tensor_reports_its_range_and_is_never_bad() -> Result<()> {
    let d = dev()?;
    let t = Tensor::from_vec(vec![7u32, 1, 900, 42], (4,), &d)?;
    t.assert("test::u32");
    let f = find(&d, "test::u32")?.expect("registered");
    assert_eq!((f.nan, f.inf), (0, 0), "integers cannot be non-finite");
    assert_eq!(f.elems, 4);
    assert_eq!(f.min, Some(1.0));
    assert_eq!(f.max, Some(900.0));
    Ok(())
}

#[test]
fn a_large_tensor_is_counted_exactly_past_the_grid_stride_cap() -> Result<()> {
    let d = dev()?;
    // Deliberately larger than ASSERT_MAX_BLOCKS * ASSERT_BLOCK (4096 * 256 =
    // 1,048,576), so every thread takes several grid-stride iterations and a
    // kernel that dropped the tail would under-count.
    let n = 3_000_000usize;
    let mut vals = vec![1.0f32; n];
    vals[0] = f32::NAN;
    vals[n - 1] = f32::NEG_INFINITY;
    vals[n / 2] = -17.0;
    let t = Tensor::from_vec(vals, (n,), &d)?;
    t.assert("test::large");

    let f = find(&d, "test::large")?.expect("registered");
    assert_eq!(f.elems as usize, n, "every element must be visited once");
    assert_eq!(f.nan, 1);
    assert_eq!(f.inf, 1);
    assert_eq!(f.min, Some(-17.0));
    assert_eq!(f.max, Some(1.0));
    Ok(())
}

#[test]
fn a_quantized_weight_is_examined_through_its_dequantized_values() -> Result<()> {
    let d = dev()?;
    // Zero quantizes exactly in Q8_0 (scale 0, quants 0), so the dequantized
    // values are exactly zero and the expected report is arithmetic, not a
    // codec tolerance.
    let n = 256usize;
    let t = Tensor::from_vec(vec![0.0f32; n], (1, n), &d)?;
    let q = QTensor::quantize(&t, GgmlDType::Q8_0)?;
    q.assert("test::q_zeros");

    let f = find(&d, "test::q_zeros")?.expect("registered");
    assert_eq!((f.nan, f.inf), (0, 0), "zeros dequantize finite");
    assert_eq!(f.elems as usize, n);
    assert_eq!(f.min, Some(0.0));
    assert_eq!(f.max, Some(0.0));
    Ok(())
}

#[test]
fn an_inf_poisons_exactly_its_own_quant_block_and_no_other() -> Result<()> {
    let d = dev()?;
    // Q8_0 derives one scale per 32-element block from the block's `amax`, so
    // an Inf drives `d = inf`, `id = 1/d = 0`, every quant to 0, and every
    // dequantized value in that block to `0 * inf = NaN`. The neighbouring
    // blocks are untouched, which makes the expected count exactly 32.
    let n = 128usize;
    let mut vals = vec![1.0f32; n];
    vals[40] = f32::INFINITY; // block 1 (elements 32..64)
    let t = Tensor::from_vec(vals, (1, n), &d)?;
    let q = QTensor::quantize(&t, GgmlDType::Q8_0)?;
    q.assert("test::q_inf_block");

    let f = find(&d, "test::q_inf_block")?.expect("registered");
    assert_eq!(f.elems as usize, n);
    assert_eq!(
        f.nan, 32,
        "exactly the poisoned block's 32 elements, not the whole tensor"
    );
    assert_eq!(f.inf, 0, "the Inf becomes NaN through the scale, not an Inf");
    assert!(f.is_bad());
    Ok(())
}

#[test]
fn quantization_erases_a_nan_rather_than_propagating_it() -> Result<()> {
    let d = dev()?;
    // `amax = amax.max(x.abs())` — and `f32::max` returns the NON-NaN operand,
    // so a NaN never reaches the scale. The scale stays finite, `round(NaN) as
    // i8` saturates to 0, and the block dequantizes entirely finite.
    //
    // This is load-bearing for reading any assert downstream of a quantize
    // step: a NaN that crosses one is DESTROYED, not carried. A clean report
    // after an int8 quantization therefore does not clear the values that went
    // into it, and a NaN observed after one cannot have come from before it.
    let n = 64usize;
    let mut vals = vec![1.0f32; n];
    vals[10] = f32::NAN;
    let t = Tensor::from_vec(vals, (1, n), &d)?;
    let q = QTensor::quantize(&t, GgmlDType::Q8_0)?;
    q.assert("test::q_nan_erased");

    let f = find(&d, "test::q_nan_erased")?.expect("registered");
    assert_eq!(f.elems as usize, n);
    assert_eq!((f.nan, f.inf), (0, 0), "the NaN is gone after quantization");
    assert!(!f.is_bad());
    Ok(())
}

#[test]
fn the_order_ticket_names_which_site_went_bad_first() -> Result<()> {
    let d = dev()?;
    let clean = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &d)?;
    let bad = Tensor::from_vec(vec![f32::NAN, 2.0], (2,), &d)?;

    // Order matters: the ticket comes from a device counter, so the site whose
    // kernel observes a NaN first must carry the lower number. Each assert is
    // drained before the next so the launches cannot be reordered against each
    // other.
    bad.assert("test::order_first");
    let first = find(&d, "test::order_first")?.expect("registered");
    clean.assert("test::order_clean");
    bad.assert("test::order_second");
    let second = find(&d, "test::order_second")?.expect("registered");
    let clean_f = find(&d, "test::order_clean")?.expect("registered");

    let (a, b) = (first.seq.expect("stamped"), second.seq.expect("stamped"));
    assert!(a < b, "first bad site must carry the lower ticket ({a} vs {b})");
    assert_eq!(clean_f.seq, None, "a clean site is never stamped");
    Ok(())
}
