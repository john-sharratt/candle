//! A wave's token ids on the host, read with one readback.
//!
//! The models whose forward needs its ids on the CPU — a host-side embedding
//! gather, the PLE n-gram hash — are handed one id tensor per sequence, and the
//! scheduler builds those on the device. Reading each back on its own is one
//! synchronous device→host transfer per sequence per wave. The embedding's id
//! readback is a transfer hot-path invariant 3 sanctions, but as one transfer:
//! this reads every device-side input in a single concatenation and a single
//! readback, and the per-sequence uploads `upload_plan_rows` makes sit back to
//! back, so the concatenation is one copy per upload as well.

use candle::{DType, Result, Tensor};

use crate::models::operand_guard::expect_dtype;

/// Each input's ids, in order, as host vectors. Inputs already on the host are
/// read in place; every device input together costs one concatenation and one
/// readback.
///
/// The ids are validated as U32 rather than converted: every producer of a
/// wave input holds `u32` tokens, so a cast here would only hide one that did
/// not (invariant 1b).
pub fn host_token_ids<'a>(inputs: impl IntoIterator<Item = &'a Tensor>) -> Result<Vec<Vec<u32>>> {
    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut on_device: Vec<(usize, Tensor)> = Vec::new();
    for (i, t) in inputs.into_iter().enumerate() {
        expect_dtype(t, DType::U32, "wave token ids")?;
        let flat = t.flatten_all()?;
        if flat.device().is_cuda() {
            on_device.push((i, flat));
            out.push(Vec::new());
        } else {
            out.push(flat.to_vec1::<u32>()?);
        }
    }
    if on_device.is_empty() {
        return Ok(out);
    }
    let flats: Vec<&Tensor> = on_device.iter().map(|(_, t)| t).collect();
    let all = Tensor::cat(&flats, 0)?.to_vec1::<u32>()?;
    let mut start = 0;
    for (i, t) in &on_device {
        let n = t.elem_count();
        out[*i] = all[start..start + n].to_vec();
        start += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Order follows the inputs whatever their shape, a host input is read in
    /// place beside device ones, and an empty wave is empty.
    #[test]
    fn ids_come_back_in_input_order() -> Result<()> {
        let cpu = Device::Cpu;
        let a = Tensor::new(&[[3u32, 4, 5]], &cpu)?;
        let b = Tensor::new(&[9u32], &cpu)?;
        let c = Tensor::new(&[[7u32], [8]], &cpu)?;
        let got = host_token_ids([&a, &b, &c])?;
        assert_eq!(got, vec![vec![3, 4, 5], vec![9], vec![7, 8]]);
        assert!(host_token_ids(std::iter::empty::<&Tensor>())?.is_empty());
        Ok(())
    }

    /// Ids that are not U32 are refused, not retyped.
    #[test]
    fn wrong_dtype_is_refused() {
        let t = Tensor::new(&[1i64, 2], &Device::Cpu).unwrap();
        let err = host_token_ids([&t]).unwrap_err().to_string();
        assert!(err.contains("I64") && err.contains("U32"), "{err}");
    }

    /// Device inputs — views of one upload, as `upload_plan_rows` makes them,
    /// beside a separate tensor and a host one — come back in input order.
    #[test]
    fn device_and_host_inputs_interleave_in_order() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let up = Tensor::new(&[[11u32], [12], [13]], &dev)?;
        let rows: Vec<Tensor> = (0..3).map(|i| up.narrow(0, i, 1)).collect::<Result<_>>()?;
        let other = Tensor::new(&[[21u32, 22]], &dev)?;
        let host = Tensor::new(&[31u32], &Device::Cpu)?;
        let got = host_token_ids([&rows[0], &host, &rows[1], &other, &rows[2]])?;
        assert_eq!(
            got,
            vec![vec![11], vec![31], vec![12], vec![21, 22], vec![13]]
        );
        Ok(())
    }
}
