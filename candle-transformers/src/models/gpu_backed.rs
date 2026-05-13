//! GPU-backed host buffer with automatic sync-on-drop.
//!
//! [`GpuBacked<T>`] wraps a host-side value `T` that implements [`GpuSerialize`]
//! and maintains a lazily-synced GPU tensor mirror. Mutations go through a scoped
//! guard ([`as_mut()`](GpuBacked::as_mut)) that automatically serialises and uploads
//! to the GPU when the guard is dropped.
//!
//! ```ignore
//! let mut buf: GpuBacked<MyData> = GpuBacked::new(data, &device);
//!
//! // Mutate — GPU upload happens automatically when guard drops.
//! {
//!     let mut guard = buf.as_mut();
//!     guard.field = 42;
//! } // <- serialise + H2D copy here
//!
//! // Read the GPU tensor (guaranteed fresh after guard drop).
//! let tensor = buf.gpu()?;
//! ```

use candle::{DType, Device, Result, Tensor};
use std::ops::{Deref, DerefMut};

/// Trait for types that can be serialised into a flat byte buffer for GPU upload.
pub trait GpuSerialize {
    /// Serialise `self` into raw bytes matching the expected CUDA struct layout.
    fn gpu_serialize(&self) -> Vec<u8>;
}

/// A host value with an automatically-synced GPU tensor mirror.
///
/// The GPU copy is updated when a [`GpuMutGuard`] (from [`as_mut`](Self::as_mut))
/// is dropped, or explicitly via [`force_sync`](Self::force_sync). Read-only access
/// via [`Deref`] never triggers a sync.
pub struct GpuBacked<T: GpuSerialize> {
    host: T,
    gpu: Option<Tensor>,
    device: Device,
    /// Stored error from a guard-drop sync (Drop can't return Result).
    last_error: Option<candle::Error>,
}

impl<T: GpuSerialize> GpuBacked<T> {
    pub fn new(host: T, device: &Device) -> Self {
        Self {
            host,
            gpu: None,
            device: device.clone(),
            last_error: None,
        }
    }

    /// Get a scoped mutable reference. When the returned guard is dropped,
    /// the host data is serialised and uploaded to the GPU automatically.
    pub fn as_mut(&mut self) -> GpuMutGuard<'_, T> {
        GpuMutGuard { inner: self }
    }

    /// Mutate the host value without triggering a GPU sync.
    ///
    /// Use for changes the GPU already knows about (e.g. shadow counters
    /// that the kernel self-increments on the device side).
    pub fn mutate_silent(&mut self, f: impl FnOnce(&mut T)) {
        f(&mut self.host);
    }

    /// Get the GPU tensor, returning an error if the last sync failed.
    ///
    /// If no sync has ever occurred, performs one now.
    pub fn gpu(&mut self) -> Result<&Tensor> {
        if let Some(err) = self.last_error.take() {
            return Err(err);
        }
        if self.gpu.is_none() {
            self.do_sync()?;
        }
        Ok(self.gpu.as_ref().unwrap())
    }

    /// Get the cached GPU tensor without checking or syncing.
    /// Returns `None` if never synced.
    pub fn gpu_cached(&self) -> Option<&Tensor> {
        self.gpu.as_ref()
    }

    /// Force a sync now, replacing the GPU tensor.
    pub fn force_sync(&mut self) -> Result<&Tensor> {
        self.do_sync()?;
        Ok(self.gpu.as_ref().unwrap())
    }

    fn do_sync(&mut self) -> Result<()> {
        let bytes = self.host.gpu_serialize();
        let tensor = if bytes.is_empty() {
            Tensor::zeros(1, DType::U8, &self.device)?
        } else {
            Tensor::from_slice(&bytes, bytes.len(), &self.device)?
        };
        self.gpu = Some(tensor);
        Ok(())
    }
}

impl<T: GpuSerialize> Deref for GpuBacked<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.host
    }
}

/// Scoped mutable guard for [`GpuBacked`].
///
/// Provides `&mut T` via [`Deref`]/[`DerefMut`]. When dropped, automatically
/// serialises the host data and uploads it to the GPU. If the upload fails,
/// the error is stored and surfaced on the next [`GpuBacked::gpu`] call.
pub struct GpuMutGuard<'a, T: GpuSerialize> {
    inner: &'a mut GpuBacked<T>,
}

impl<T: GpuSerialize> Deref for GpuMutGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner.host
    }
}

impl<T: GpuSerialize> DerefMut for GpuMutGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner.host
    }
}

impl<T: GpuSerialize> Drop for GpuMutGuard<'_, T> {
    fn drop(&mut self) {
        if let Err(e) = self.inner.do_sync() {
            self.inner.last_error = Some(e);
        }
    }
}
