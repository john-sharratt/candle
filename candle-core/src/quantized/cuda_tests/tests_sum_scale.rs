//! The q8a128 block sum against f16's ceiling.
//!
//! An activation block's Σx is stored in an f16 beside its scale, and an
//! asymmetric weight's min multiplies it into every output (`d_w·s_a·C +
//! m_w·Σx`). Stored raw, a block whose sum passes 65504 stores `+inf`; stored
//! by amax (`Σx/amax`, never above 128 in magnitude) it cannot.

use super::*;
use crate::quantized::{QMatMul, QTensor};
use crate::Tensor;

/// One channel large enough that its block's raw sum passes f16, beside a row
/// that is ordinary.
const OUTLIER: f32 = 7.0e4;

/// **A block sum past f16's ceiling is NaN stored raw and finite stored by
/// amax.** One channel at 7×10⁴ keeps `amax/127` inside f16 while Σx is not:
/// through an asymmetric weight's min term the raw sum turns its whole row
/// non-finite, which is how a Qwen expert call on WhiteWorld came out NaN in
/// every element of its gate and up outputs from a finite input. The ordinary
/// row is finite either way, so the difference is the sum's storage alone.
#[test]
pub(super) fn a_block_sum_past_f16_is_finite_only_stored_by_amax() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (nrows, ncols) = (256usize, 512usize);
    let mut rng = rand::rng();
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let ordinary: Vec<f32> = (0..ncols).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let mut outlier = ordinary.clone();
    outlier[5] = OUTLIER;
    let act = Tensor::from_vec([ordinary, outlier].concat(), (2, ncols), &device)?;
    let weight = QMatMul::from_qtensor(QTensor::quantize(
        &Tensor::from_vec(w, (nrows, ncols), &device)?,
        GgmlDType::Q4_K,
    )?)?
    .repack_for_optimization(Int8Mode::Performance)?;

    let rows_finite = |sum: SumScale| -> Result<[bool; 2]> {
        let y = weight
            .forward_via_int8(&act, Int8Mode::Performance, DType::F32, sum)?
            .to_vec2::<f32>()?;
        Ok([
            y[0].iter().all(|v| v.is_finite()),
            y[1].iter().all(|v| v.is_finite()),
        ])
    };
    assert_eq!(
        rows_finite(SumScale::Raw)?,
        [true, false],
        "stored raw, the outlier's block sum overflows f16 and its row goes non-finite"
    );
    assert_eq!(
        rows_finite(SumScale::ByAmax)?,
        [true, true],
        "stored by amax, the same rows are finite"
    );
    Ok(())
}

/// **The MoE gather keeps its source's sum convention.** The gather copies
/// blocks header and all, so its rows store `Σx` however the FFN input did; an
/// operand that relabelled them would have every expert GEMM misread each
/// block's sum — which is how a by-amax FFN input drafted garbage on
/// BlackWorld while the gather labelled its result raw. Under each convention
/// the gathered operand must carry that convention, and a matmul over it must
/// match, bit for bit, the same matmul over the float rows gathered first and
/// quantized under that convention.
#[test]
pub(super) fn the_moe_gather_keeps_its_sources_sum_convention() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (tokens, cols, nrows) = (5usize, 512usize, 256usize);
    let ids: Vec<u32> = vec![3, 0, 4, 4, 1, 2, 0, 3];
    let ids_dev = dev.memcpy_stod(&ids)?;
    let xf: Vec<f32> = (0..tokens * cols)
        .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) * 0.003)
        .collect();
    let xs = Tensor::from_vec(xf, (tokens, cols), &device)?;
    let w: Vec<f32> = (0..nrows * cols)
        .map(|i| ((i * 104729 % 4001) as f32 - 2000.0) * 5.0e-5)
        .collect();
    let weight = QMatMul::from_qtensor(QTensor::quantize(
        &Tensor::from_vec(w, (nrows, cols), &device)?,
        GgmlDType::Q4_K,
    )?)?
    .repack_for_optimization(Int8Mode::Performance)?;

    for sum in [SumScale::Raw, SumScale::ByAmax] {
        let quantize = |t: &Tensor| -> Result<Q8a128Operand<'static>> {
            match to_dynamic(t, Int8Mode::Performance, &dev, sum)? {
                DynamicActs::Int8(op) => Ok(op),
                DynamicActs::Float(_) => unreachable!("Performance mode yields Int8"),
            }
        };
        let gathered =
            fused_moe_gather_q8a128(&quantize(&xs)?, &ids_dev, ids.len(), &dev, Backing::Owned)?;
        assert_eq!(
            gathered.sum_scale, sum,
            "the gather relabelled {sum:?} rows"
        );
        let want = quantize(&fused_moe_gather(&xs, &ids_dev, ids.len(), &dev)?)?;
        let matmul = |op: Q8a128Operand<'static>| -> Result<Vec<u32>> {
            let y = weight.forward_dynamic(DynamicActs::Int8(op).as_dynamic(), DType::F32)?;
            Ok(y.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        assert_eq!(
            matmul(gathered)?,
            matmul(want)?,
            "{sum:?}: a matmul over the gathered operand differs from one over the rows \
             gathered as float and quantized"
        );
    }
    Ok(())
}
