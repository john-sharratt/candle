//! Routed experts requantized to a KO format, one expert at a time, on the GPU.
//!
//! An expert tensor is `[n_expert, rows, cols]` with each expert's rows a
//! contiguous run of whole blocks. The KO layout orders an expert's chunks
//! k-block-major (`k_blk · row_groups + g`), so experts cannot be stacked into
//! one taller matrix and quantized together: each expert is its own
//! `[rows, cols]` matrix and its KO image is the concatenation unit.
//! `repack_ko_from_host` does the dequantize-then-quantize in bounded device
//! bands, so the peak is a few tens of MiB whatever the tensor.

use candle::quantized::cuda::repack_ko_from_host;
use candle::quantized::GgmlDType;
use candle::{Device, Result, Shape};

/// Requantize every expert of a raw `[n_expert, rows, cols]` tensor from
/// `src_dtype` to `target` and return the concatenated KO images.
pub fn requant_experts(
    src: &[u8],
    dims: &[usize],
    src_dtype: GgmlDType,
    target: GgmlDType,
    device: &Device,
) -> Result<Vec<u8>> {
    let Device::Cuda(dev) = device else {
        candle::bail!("requant_experts: the KO quantizer runs on a CUDA device");
    };
    let [n_expert, rows, cols] = dims[..] else {
        candle::bail!("requant_experts: {dims:?} is not an [n_expert, rows, cols] tensor");
    };
    let per_src = rows * cols / src_dtype.block_size() * src_dtype.type_size();
    if src.len() != n_expert * per_src {
        candle::bail!(
            "requant_experts: {} source bytes for {n_expert} × [{rows}, {cols}] of {src_dtype:?}",
            src.len()
        );
    }
    let per_out = rows * cols / target.block_size() * target.type_size();
    let shape = Shape::from((rows, cols));
    let mut out = Vec::with_capacity(n_expert * per_out);
    for e in 0..n_expert {
        let ko = repack_ko_from_host(
            dev,
            &src[e * per_src..(e + 1) * per_src],
            &shape,
            src_dtype,
            target,
            None,
        )?;
        let bytes = ko.data()?;
        if bytes.len() != per_out {
            candle::bail!(
                "requant_experts: expert {e} produced {} bytes, {target:?} needs {per_out}",
                bytes.len()
            );
        }
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::ko_quant::quantize_ko;
    use candle::quantized::QTensor;
    use candle::Tensor;

    /// Deterministic weights in roughly the range real ones take.
    fn weights(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    /// The GPU path must produce exactly what the CPU reference does from the
    /// same `Q8_0` source: dequantize each expert, `quantize_ko` it, concatenate.
    #[test]
    fn gpu_requant_matches_the_cpu_reference_per_expert() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (n_expert, rows, cols) = (3usize, 32usize, 256usize);
        let w = weights(n_expert * rows * cols);
        let q8 = QTensor::quantize(
            &Tensor::from_vec(w, (n_expert * rows, cols), &Device::Cpu)?,
            GgmlDType::Q8_0,
        )?;
        let src = q8.data()?.to_vec();
        let dense: Vec<f32> = q8.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1()?;
        for target in [GgmlDType::Q2_KO, GgmlDType::Q3_KO, GgmlDType::Q4_KO] {
            let gpu = requant_experts(
                &src,
                &[n_expert, rows, cols],
                GgmlDType::Q8_0,
                target,
                &device,
            )?;
            let mut cpu = Vec::new();
            for e in 0..n_expert {
                cpu.extend(quantize_ko(
                    &dense[e * rows * cols..(e + 1) * rows * cols],
                    rows,
                    cols,
                    target,
                ));
            }
            assert_eq!(gpu.len(), n_expert * rows * cols / 128 * target.type_size());
            assert!(
                gpu == cpu,
                "{target:?}: GPU requant diverges from the CPU reference"
            );
        }
        Ok(())
    }

    #[test]
    fn a_short_source_is_refused() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let err = requant_experts(
            &[0u8; 10],
            &[1, 32, 256],
            GgmlDType::Q8_0,
            GgmlDType::Q2_KO,
            &device,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("10 source bytes"), "{err}");
        Ok(())
    }
}
