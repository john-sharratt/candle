//! A wave's token ids, read to the host once.
//!
//! The models whose forward needs its ids on the CPU — a host-side embedding
//! gather, the PLE n-gram hash — are handed one id tensor per sequence, and the
//! scheduler builds those on the device. Reading each back on its own is one
//! synchronous device→host transfer per sequence per wave. The embedding's id
//! readback is a transfer hot-path invariant 3 sanctions, but as one transfer:
//! this lays every device-side input into one buffer and reads that back once.
//!
//! The buffer is the caller's to keep: on the ticket's arena when one is given,
//! it is also every row's id **on the device**, in wave order — which is what
//! the embedding gather reads, so a caller holding it uploads nothing more.

use candle::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};

use crate::models::operand_guard::expect_dtype;
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::{wave_empty_ticketed, wave_from_vec_ticketed};

/// A wave's ids: per input on the host, and — when every input was on the
/// device — all of them in one device buffer, in input order.
#[derive(Debug)]
pub struct WaveTokenIds {
    pub per_input: Vec<Vec<u32>>,
    pub device: Option<Tensor>,
}

/// Each input's ids, in order, as host vectors. Inputs already on the host are
/// read in place; every device input is copied into one buffer on `ticket`'s
/// arena (the pool without one), which is read back once.
///
/// The ids are validated as U32 rather than converted: every producer of a
/// wave input holds `u32` tokens, so a cast here would only hide one that did
/// not (invariant 1b).
pub fn host_token_ids<'a>(
    inputs: impl IntoIterator<Item = &'a Tensor>,
    ticket: Option<WaveTicket>,
) -> Result<WaveTokenIds> {
    let mut per_input: Vec<Vec<u32>> = Vec::new();
    let mut on_device: Vec<(usize, Tensor)> = Vec::new();
    for (i, t) in inputs.into_iter().enumerate() {
        expect_dtype(t, DType::U32, "wave token ids")?;
        let flat = t.flatten_all()?;
        if flat.device().is_cuda() {
            on_device.push((i, flat));
            per_input.push(Vec::new());
        } else {
            per_input.push(flat.to_vec1::<u32>()?);
        }
    }
    if on_device.is_empty() {
        return Ok(WaveTokenIds {
            per_input,
            device: None,
        });
    }
    let total: usize = on_device.iter().map(|(_, t)| t.elem_count()).sum();
    let dev = on_device[0].1.device().clone();
    // Fully written by the copies below (invariant 6).
    #[cfg(feature = "cuda")]
    let all = wave_empty_ticketed((total,), DType::U32, &dev, ticket)?;
    #[cfg(not(feature = "cuda"))]
    let all = {
        let _ = ticket;
        Tensor::empty((total,), DType::U32, &dev)?
    };
    let mut start = 0;
    for (_, t) in &on_device {
        let n = t.elem_count();
        all.narrow(0, start, n)?.slice_set(t, 0, 0)?;
        start += n;
    }
    let host = all.to_vec1::<u32>()?;
    let mut start = 0;
    for (i, t) in &on_device {
        let n = t.elem_count();
        per_input[*i] = host[start..start + n].to_vec();
        start += n;
    }
    let every_input_on_device = on_device.len() == per_input.len();
    Ok(WaveTokenIds {
        per_input,
        device: every_input_on_device.then_some(all),
    })
}

/// Every input's ids in one **device** buffer, in input order, on `ticket`'s
/// arena — what a device-side embedding gather reads. Nothing is read back.
///
/// The scheduler hands its prompt pieces over on the host, so a wave's ids
/// reach the device here, in **one** upload onto the forward span, rather than
/// as one pool allocation per sequence ahead of the forward. Ids already on the
/// device (a sampled decode token) are copied into their place beside them.
///
/// Always copied onto the span when a ticket is given, even a lone device
/// input that is already one buffer: the gather's output inherits the ids'
/// arena, and that output is the forward's residual. Without a ticket a lone
/// device input is returned as it stands.
///
/// Validated as U32, never converted (invariant 1b).
pub fn device_token_ids(
    inputs: &[&Tensor],
    device: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let mut flat: Vec<Tensor> = Vec::with_capacity(inputs.len());
    for t in inputs {
        expect_dtype(t, DType::U32, "wave token ids")?;
        flat.push(t.flatten_all()?);
    }
    if let ([only], None) = (flat.as_slice(), ticket) {
        if only.device().same_device(device) {
            return Ok(only.clone());
        }
    }
    let total: usize = flat.iter().map(|t| t.elem_count()).sum();
    let mut host: Vec<u32> = Vec::with_capacity(total);
    let mut on_device: Vec<(usize, &Tensor)> = Vec::new();
    for t in &flat {
        if t.device().same_device(device) {
            on_device.push((host.len(), t));
            // A placeholder the device copy below overwrites.
            host.resize(host.len() + t.elem_count(), 0);
        } else {
            host.extend(t.to_vec1::<u32>()?);
        }
    }
    #[cfg(feature = "cuda")]
    let all = if on_device.len() == flat.len() {
        // Fully written by the copies below (invariant 6).
        wave_empty_ticketed((total,), DType::U32, device, ticket)?
    } else {
        wave_from_vec_ticketed(host, (total,), device, ticket)?
    };
    #[cfg(not(feature = "cuda"))]
    let all = {
        let _ = ticket;
        Tensor::from_vec(host, (total,), device)?
    };
    for (start, t) in on_device {
        all.narrow(0, start, t.elem_count())?.slice_set(t, 0, 0)?;
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Host and device inputs land in one buffer in input order; a lone input
    /// on the target device is handed back as it stands.
    #[test]
    fn device_ids_pack_every_input_in_order() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let a = Tensor::new(&[[3u32, 4]], &Device::Cpu)?;
        let b = Tensor::new(&[[5u32]], &dev)?;
        let c = Tensor::new(&[[6u32, 7, 8]], &Device::Cpu)?;
        let got = device_token_ids(&[&a, &b, &c], &dev, None)?;
        assert!(got.device().is_cuda());
        assert_eq!(got.to_vec1::<u32>()?, vec![3, 4, 5, 6, 7, 8]);
        let lone = device_token_ids(&[&b], &dev, None)?;
        assert_eq!(lone.to_vec1::<u32>()?, vec![5]);
        let host_only = device_token_ids(&[&a], &dev, None)?;
        assert!(host_only.device().is_cuda());
        assert_eq!(host_only.to_vec1::<u32>()?, vec![3, 4]);
        Ok(())
    }

    /// Order follows the inputs whatever their shape, a host input is read in
    /// place beside device ones, and an empty wave is empty.
    #[test]
    fn ids_come_back_in_input_order() -> Result<()> {
        let cpu = Device::Cpu;
        let a = Tensor::new(&[[3u32, 4, 5]], &cpu)?;
        let b = Tensor::new(&[9u32], &cpu)?;
        let c = Tensor::new(&[[7u32], [8]], &cpu)?;
        let got = host_token_ids([&a, &b, &c], None)?;
        assert_eq!(got.per_input, vec![vec![3, 4, 5], vec![9], vec![7, 8]]);
        assert!(got.device.is_none(), "host inputs make no device buffer");
        assert!(host_token_ids(std::iter::empty::<&Tensor>(), None)?
            .per_input
            .is_empty());
        Ok(())
    }

    /// Ids that are not U32 are refused, not retyped.
    #[test]
    fn wrong_dtype_is_refused() {
        let t = Tensor::new(&[1i64, 2], &Device::Cpu).unwrap();
        let err = host_token_ids([&t], None).unwrap_err().to_string();
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
        let got = host_token_ids([&rows[0], &host, &rows[1], &other, &rows[2]], None)?;
        assert_eq!(
            got.per_input,
            vec![vec![11], vec![31], vec![12], vec![21, 22], vec![13]]
        );
        assert!(
            got.device.is_none(),
            "a host input leaves no whole-wave buffer"
        );
        Ok(())
    }

    /// Every input on the device: the buffer read back is every row's id in
    /// input order, kept for the caller.
    #[test]
    fn an_all_device_wave_keeps_its_ids_on_the_device() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let a = Tensor::new(&[[5u32, 6]], &dev)?;
        let b = Tensor::new(&[[7u32]], &dev)?;
        let got = host_token_ids([&a, &b], None)?;
        assert_eq!(got.per_input, vec![vec![5, 6], vec![7]]);
        let device = got.device.expect("every input was on the device");
        assert_eq!(device.to_vec1::<u32>()?, vec![5, 6, 7]);
        Ok(())
    }
}
