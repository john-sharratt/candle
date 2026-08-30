#![allow(unused)]
use super::GgmlDType;
use crate::{CudaDevice, CudaStorage, Error, Result};

/// Mirrors the CUDA repack's band size so the load-time pool arithmetic
/// (`dense_span::peak_load_pool_bytes`) is one expression on both builds. Nothing here repacks,
/// so nothing here reads it for its own sake.
pub const REPACK_BAND_BYTES: usize = 48 * 1024 * 1024;

pub fn set_force_dmmv(_f: bool) {}

/// Dummy implementation for non-CUDA builds
pub fn get_dispatch_info(_batch_size: i32, _weight_bytes: usize) -> String {
    "cpu".to_string()
}

#[derive(Clone)]
pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(_: &CudaDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_bf16(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn data_range(&self, _range: std::ops::Range<usize>) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn fwd_into_dtype(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}

pub fn load_repacked(
    _device: &CudaDevice,
    _repacked_data: &[u8],
    _dtype: super::GgmlDType,
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}
