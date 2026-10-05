//! The residual a forward hands back when it stops short of the head.
//!
//! A forward that reaches the head keeps its residual on the forward span: it
//! dies with the forward. A window that stops short hands it back for a later
//! forward to resume from — past the span's reset — so it has to live
//! somewhere the span does not. Allocating it per window put a driver
//! allocation in every co-batched creep wave, and that wave carries decode.
//!
//! These buffers are held instead and handed out as views. A buffer is free
//! again once every view of it has been dropped, which
//! [`Tensor::is_sole_owner`] answers on the pool's own handle: the residual a
//! scheduler is still holding for its next window keeps its buffer busy, and
//! the next window's residual goes into another one. How many are busy at
//! once is the caller's business — two in the co-batched creep (the held
//! creep part, and the window being written) — and the pool grows to that and
//! no further.

use std::sync::Mutex;

use candle::{DType, Device, Result, Shape, Tensor};

/// Held residual buffers, each flat, each handed out as one view at a time.
#[derive(Default)]
pub struct WindowResiduals {
    held: Mutex<Vec<Tensor>>,
}

impl WindowResiduals {
    /// A `dims` buffer of `dtype` that nothing else holds, uninitialised.
    ///
    /// A free held buffer wide enough is reused; a free one too narrow is
    /// replaced at twice its width (or `dims`, if wider), so a ramp of widths
    /// settles in a few steps; with none free, one more is held.
    pub fn take(&self, dims: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        let shape: Shape = dims.into();
        let need = shape.elem_count();
        let mut held = self
            .held
            .lock()
            .map_err(|_| candle::Error::Msg("window residual pool poisoned".into()))?;
        let free =
            |t: &Tensor| t.is_sole_owner() && t.dtype() == dtype && t.device().same_device(device);
        let slot = match held.iter().position(|t| free(t) && t.elem_count() >= need) {
            Some(i) => i,
            None => {
                let grown = match held.iter().position(free) {
                    Some(i) => {
                        let width = (held[i].elem_count() * 2).max(need);
                        held.swap_remove(i);
                        width
                    }
                    None => need,
                };
                // Fully written by the forward that takes it before anything
                // reads it (invariant 6).
                held.push(Tensor::empty(grown, dtype, device)?);
                held.len() - 1
            }
        };
        held[slot].narrow(0, 0, need)?.reshape(shape)
    }

    /// Buffers held, busy or free.
    pub fn held(&self) -> usize {
        self.held.lock().map_or(0, |h| h.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dropped_residual_gives_its_buffer_to_the_next_window() -> Result<()> {
        let pool = WindowResiduals::default();
        let first = pool.take((1, 4, 3), DType::F32, &Device::Cpu)?;
        let keep = first.clone();
        drop(first);
        let busy = pool.take((1, 4, 3), DType::F32, &Device::Cpu)?;
        assert!(!busy.same_storage(&keep));
        drop(keep);
        let again = pool.take((1, 2, 3), DType::F32, &Device::Cpu)?;
        assert!(!again.same_storage(&busy));
        assert_eq!(again.dims(), &[1, 2, 3]);
        assert_eq!(pool.held(), 2);
        Ok(())
    }

    #[test]
    fn a_free_buffer_reused_while_another_is_busy() -> Result<()> {
        let pool = WindowResiduals::default();
        let a = pool.take(12, DType::F32, &Device::Cpu)?;
        let b = pool.take(12, DType::F32, &Device::Cpu)?;
        drop(a);
        let c = pool.take(8, DType::F32, &Device::Cpu)?;
        assert!(!c.same_storage(&b));
        assert_eq!(pool.held(), 2);
        Ok(())
    }

    #[test]
    fn a_free_buffer_too_narrow_is_replaced_at_twice_its_width() -> Result<()> {
        let pool = WindowResiduals::default();
        drop(pool.take(10, DType::F32, &Device::Cpu)?);
        let wide = pool.take(12, DType::F32, &Device::Cpu)?;
        assert_eq!(pool.held(), 1);
        drop(wide);
        // The replacement was 20 wide, so 20 fits without another.
        drop(pool.take(20, DType::F32, &Device::Cpu)?);
        assert_eq!(pool.held(), 1);
        let held = pool.take(21, DType::F32, &Device::Cpu)?;
        assert_eq!(held.elem_count(), 21);
        assert_eq!(pool.held(), 1);
        Ok(())
    }

    #[test]
    fn a_buffer_of_another_dtype_is_not_handed_out() -> Result<()> {
        let pool = WindowResiduals::default();
        drop(pool.take(8, DType::F32, &Device::Cpu)?);
        let bf = pool.take(8, DType::BF16, &Device::Cpu)?;
        assert_eq!(bf.dtype(), DType::BF16);
        assert_eq!(pool.held(), 2);
        Ok(())
    }
}
