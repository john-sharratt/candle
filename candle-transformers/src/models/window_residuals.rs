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
//!
//! # Where they live, and in what sizes
//!
//! On the reservation's **held ground** ([`held_empty`]), which a model that
//! needs these widens by `qwen4exp::loader::window_residual_reserve_bytes`. In
//! the CUDA pool they were memory outside the span, which WDDM demotes first.
//!
//! A buffer is taken in whole [`WINDOW_RESIDUAL_STEP`]s. Sized exactly, every
//! window a few rows wider than the last replaced a buffer — 320, then 321
//! MiB — and each replacement was a fresh allocation of the whole residual on
//! the wave that carries decode. In steps, a buffer serves every width up to
//! its step, and the reserve is priced in the same steps
//! ([`window_residual_buffer_bytes`]), so no buffer is ever wider than planned.

use std::sync::Mutex;

use candle::{DType, Device, Result, Shape, Tensor};

use super::held_tensor::held_empty;

/// The granule a window residual buffer is sized in.
pub const WINDOW_RESIDUAL_STEP: usize = 64 * 1024 * 1024;

/// Bytes a buffer holding `bytes` of residual takes: whole steps.
pub fn window_residual_buffer_bytes(bytes: usize) -> usize {
    bytes.next_multiple_of(WINDOW_RESIDUAL_STEP)
}

/// Held residual buffers, each flat, each handed out as one view at a time.
#[derive(Default)]
pub struct WindowResiduals {
    held: Mutex<Vec<Tensor>>,
}

impl WindowResiduals {
    /// A `dims` buffer of `dtype` that nothing else holds, uninitialised.
    ///
    /// A free held buffer wide enough is reused; a free one too narrow is
    /// replaced; with none free, one more is held. A new buffer is
    /// [`window_residual_buffer_bytes`] of the residual — whole steps, never
    /// doubled: doubling once put two 404 MiB buffers under a 202 MiB residual,
    /// past anything the reserve priced.
    #[track_caller]
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
                if let Some(i) = held.iter().position(free) {
                    held.swap_remove(i);
                }
                let size = dtype.size_in_bytes();
                let grown = window_residual_buffer_bytes(need * size) / size;
                // Fully written by the forward that takes it before anything
                // reads it (invariant 6).
                held.push(held_empty(grown, dtype, device)?);
                let held_bytes: usize = held
                    .iter()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .sum();
                tracing::debug!(
                    target: "candle_transformers::window_residuals",
                    buffer_mib = (grown * size) >> 20,
                    buffers = held.len(),
                    held_mib = held_bytes >> 20,
                    "window residual pool grew"
                );
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

    /// One step's worth of `f32` elements.
    const STEP_F32: usize = WINDOW_RESIDUAL_STEP / 4;

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

    /// **A window a little wider than the last reuses its buffer.** The case
    /// that churned: 320 MiB, then 321.
    #[test]
    fn a_slightly_wider_window_reuses_the_buffer() -> Result<()> {
        let pool = WindowResiduals::default();
        let first = pool.take(STEP_F32 - 300, DType::F32, &Device::Cpu)?;
        drop(first);
        let wider = pool.take(STEP_F32 - 10, DType::F32, &Device::Cpu)?;
        assert_eq!(pool.held(), 1);
        let backing: usize = pool
            .held
            .lock()
            .unwrap()
            .iter()
            .map(|t| t.elem_count())
            .sum();
        assert_eq!(backing, STEP_F32, "one buffer of one whole step");
        assert_eq!(wider.elem_count(), STEP_F32 - 10);
        Ok(())
    }

    /// A window past its buffer's step replaces it with one step-rounded to the
    /// new width — never doubled.
    #[test]
    fn a_free_buffer_too_narrow_is_replaced_in_whole_steps() -> Result<()> {
        let pool = WindowResiduals::default();
        drop(pool.take(10, DType::F32, &Device::Cpu)?);
        let wide = pool.take(STEP_F32 + 1, DType::F32, &Device::Cpu)?;
        assert_eq!(pool.held(), 1);
        assert_eq!(wide.elem_count(), STEP_F32 + 1);
        let backing: usize = pool
            .held
            .lock()
            .unwrap()
            .iter()
            .map(|t| t.elem_count())
            .sum();
        assert_eq!(backing, 2 * STEP_F32, "two steps, not double the first");
        Ok(())
    }

    #[test]
    fn a_buffer_is_whole_steps() {
        assert_eq!(window_residual_buffer_bytes(1), WINDOW_RESIDUAL_STEP);
        assert_eq!(
            window_residual_buffer_bytes(WINDOW_RESIDUAL_STEP),
            WINDOW_RESIDUAL_STEP
        );
        assert_eq!(
            window_residual_buffer_bytes(320 * 1024 * 1024 + 1),
            384 * 1024 * 1024
        );
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
