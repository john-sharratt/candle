//! Per-sequence carried buffers in the reservation's recurrent-state slots.
//!
//! The PLE block's two conv histories and the draft head's two seeds are
//! state a sequence carries for its whole life, beside the DeltaNet states its
//! recurrent store already holds in this tenant. Claimed here — at admission,
//! before the forward opens, as the store's are — they are ground the
//! partition counts rather than pool memory outside it, and the forward never
//! allocates them.

#[cfg(feature = "cuda")]
use std::sync::Arc;

use candle::{DType, Device, Result, Shape, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{claim_arena_slots, SlotTenant};

#[cfg(feature = "cuda")]
use crate::models::wave_buffers::upload_into;

/// `n` F32 buffers of `shape`, each in its own recurrent-state slot on CUDA —
/// zeroed when `zeroed`, uninitialised otherwise — and ordinary tensors on any
/// other device.
///
/// **Between forwards.** A claim that needs a new arena takes the arena window,
/// which refuses inside a forward.
pub fn state_buffers(
    device: &Device,
    shape: impl Into<Shape>,
    n: usize,
    zeroed: bool,
) -> Result<Vec<Tensor>> {
    let shape = shape.into();
    let bytes = shape.elem_count() * DType::F32.size_in_bytes();
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return claim_arena_slots(device, SlotTenant::RecurrentState, bytes, n)?
            .into_iter()
            .map(|slot| {
                if zeroed {
                    slot.zero(bytes, device)?;
                }
                Arc::new(slot).tensor(0, DType::F32, shape.clone(), device)
            })
            .collect();
    }
    let _ = bytes;
    (0..n)
        .map(|_| {
            if zeroed {
                Tensor::zeros(shape.clone(), DType::F32, device)
            } else {
                Tensor::empty(shape.clone(), DType::F32, device)
            }
        })
        .collect()
}

/// One buffer of [`state_buffers`].
pub fn state_buffer(device: &Device, shape: impl Into<Shape>, zeroed: bool) -> Result<Tensor> {
    let mut one = state_buffers(device, shape, 1, zeroed)?;
    Ok(one.pop().expect("one buffer claimed"))
}

/// One buffer of [`state_buffers`] holding `values` — a restored state, written
/// straight from the host into its slot.
///
/// `Tensor::from_vec` would put the same bytes in a driver allocation outside
/// the reservation, for as long as the sequence lives.
pub fn state_buffer_from(
    device: &Device,
    shape: impl Into<Shape>,
    values: &[f32],
) -> Result<Tensor> {
    let shape = shape.into();
    if shape.elem_count() != values.len() {
        candle::bail!(
            "state buffer: {} values for a shape of {shape:?}",
            values.len()
        );
    }
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        let buffer = state_buffer(device, shape, false)?;
        upload_into(&buffer, values)?;
        return Ok(buffer);
    }
    Tensor::from_vec(values.to_vec(), shape, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off the device a state buffer is an ordinary tensor of the shape asked,
    /// zeroed when asked.
    #[test]
    fn a_host_state_buffer_is_its_shape_and_zeroed_on_request() -> Result<()> {
        let b = state_buffer(&Device::Cpu, (3, 4), true)?;
        assert_eq!(b.dims(), &[3, 4]);
        assert_eq!(b.to_vec2::<f32>()?, vec![vec![0f32; 4]; 3]);
        assert_eq!(state_buffers(&Device::Cpu, (2, 2), 3, false)?.len(), 3);
        Ok(())
    }

    /// A restored buffer holds exactly the values it was given, in its shape,
    /// and a count that does not fill the shape is refused.
    #[test]
    fn a_restored_state_buffer_holds_its_values() -> Result<()> {
        let b = state_buffer_from(&Device::Cpu, (2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        assert_eq!(
            b.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
        assert!(state_buffer_from(&Device::Cpu, (2, 3), &[1.0]).is_err());
        Ok(())
    }
}
