//! Applies sparse in-place updates (add, subtract, multiply, divide and the
//! repeat penalty) to a storage at a list of indices.
//!
//! Each in-place operation uploads its index list once and mutates the existing
//! buffer through a direct FFI call, so no copy of the tensor is made. Empty
//! index lists return early. The variant that takes a value per index keeps its
//! uploaded buffers alive until the kernel has been launched. The repeat penalty
//! divides or multiplies by sign in one kernel, and a penalty of 1.0 is a no-op.
//! The non-`_mut` forms clone the storage first and then call the in-place one.

use super::*;

impl CudaStorage {
    /// In-place sparse addition - mutates the tensor directly without cloning.
    /// This is 20x+ faster than add_at_indices for large tensors with sparse updates.
    pub fn add_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get the CUDA stream for pointer access
        let stream = device.cuda_stream();

        // Mutate in-place based on dtype using direct FFI calls
        match &mut self.slice {
            CudaStorageSlice::F32(dst) => {
                use cudarc::driver::DevicePtrMut;
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_f32(
                        dst_ptr as *mut f32,
                        indices_ptr as *const u32,
                        num_indices,
                        value,
                        1usize,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            CudaStorageSlice::F16(dst) => {
                use cudarc::driver::DevicePtrMut;
                let value_f16 = half::f16::from_f32(value);
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_f16(
                        dst_ptr as *mut half::f16,
                        indices_ptr as *const u32,
                        num_indices,
                        value_f16,
                        1usize,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            CudaStorageSlice::BF16(dst) => {
                use cudarc::driver::DevicePtrMut;
                let value_bf16 = half::bf16::from_f32(value);
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_bf16(
                        dst_ptr as *mut half::bf16,
                        indices_ptr as *const u32,
                        num_indices,
                        value_bf16,
                        1usize,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            CudaStorageSlice::F64(dst) => {
                use cudarc::driver::DevicePtrMut;
                let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
                let (indices_ptr, _guard2) = indices_dev.device_ptr(&stream);
                unsafe {
                    kernels::simple::add_at_indices::add_at_indices_f64(
                        dst_ptr as *mut f64,
                        indices_ptr as *const u32,
                        num_indices,
                        value as f64,
                        1usize,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            _ => crate::bail!(
                "add_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        }

        Ok(())
    }

    /// In-place sparse subtraction - mutates the tensor directly without cloning.
    /// This is 20x+ faster than sub_at_indices for large tensors with sparse updates.
    pub fn sub_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "sub_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::scatter_op::run_scatter_op_at_indices(
                kernels::simple::scatter_op::ScatterOp::Sub as i32,
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                value,
                value as f64,
                1, // stride
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        Ok(())
    }

    /// In-place sparse subtraction with per-index values - mutates the tensor directly without cloning.
    /// Each index gets its own value: data[indices[i]] -= values[i]
    /// This is 20x+ faster than sub_at_indices for large tensors with sparse updates.
    pub fn sub_at_indices_mut_with_values(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        if indices.len() != values.len() {
            crate::bail!(
                "indices and values must have the same length, got {} and {}",
                indices.len(),
                values.len()
            );
        }

        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices and values to GPU
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "sub_at_indices_with_values is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        // For F64, we need to convert values to f64
        // Keep the allocated GPU buffers alive until after the kernel call
        let values_dev_f64: Option<CudaSlice<f64>>;
        let values_dev_f32: Option<CudaSlice<f32>>;

        let values_ptr: *const std::ffi::c_void = if self.dtype() == DType::F64 {
            let values_f64_vec: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            values_dev_f64 = Some(device.memcpy_stod(&values_f64_vec)?);
            values_dev_f32 = None;
            let (ptr, _g) = values_dev_f64.as_ref().unwrap().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else {
            values_dev_f64 = None;
            values_dev_f32 = Some(device.memcpy_stod(values)?);
            let (ptr, _g) = values_dev_f32.as_ref().unwrap().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        };

        // Keep both buffers alive through the unsafe block
        let _keep_alive_f64 = &values_dev_f64;
        let _keep_alive_f32 = &values_dev_f32;

        unsafe {
            kernels::simple::scatter_op::run_sub_at_indices_with_values(
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                values_ptr,
                num_indices,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        Ok(())
    }

    pub fn sub_at_indices(&self, _layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        forbidden_alloc::record("CudaStorage::sub_at_indices", self.slice.byte_len());
        let device = self.device().clone();

        // Early return for empty indices
        if indices.is_empty() {
            let slice = match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            };
            return Ok(Self {
                slice,
                device,
                backing: Backing::Owned,
                anchor: None,
            });
        }

        // Clone and then mutate in-place
        let mut result = Self {
            slice: match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            },
            device,
            backing: Backing::Owned,
            anchor: None,
        };

        // Use in-place mutation method
        result.sub_at_indices_mut(_layout, indices, value)?;
        Ok(result)
    }

    /// In-place sparse division - mutates the tensor directly without cloning.
    /// This is 20x+ faster than div_at_indices for large tensors with sparse updates.
    pub fn div_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "div_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::scatter_op::run_scatter_op_at_indices(
                kernels::simple::scatter_op::ScatterOp::Div as i32,
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                value,
                value as f64,
                1, // stride
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        Ok(())
    }

    /// In-place sparse multiplication - mutates the tensor directly without cloning.
    /// This is 20x+ faster than mul_at_indices for large tensors with sparse updates.
    pub fn mul_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        // Early return for empty indices
        if indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_scatter_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "mul_at_indices is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::scatter_op::run_scatter_op_at_indices(
                kernels::simple::scatter_op::ScatterOp::Mul as i32,
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                value,
                value as f64,
                1, // stride
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        Ok(())
    }

    /// In-place repeat penalty - applies penalty based on logit sign in a single GPU kernel pass.
    /// For positive logits: divides by penalty (reduces probability)
    /// For negative/zero logits: multiplies by penalty (reduces probability)
    /// This is extremely efficient as it combines the logic from both div_at_indices and mul_at_indices.
    pub fn repeat_penalty_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        penalty: f32,
    ) -> Result<()> {
        // Early return for empty indices or penalty of 1.0 (no-op)
        if indices.is_empty() || penalty == 1.0 {
            return Ok(());
        }

        let device = &self.device;
        let stream = device.cuda_stream();

        // Upload indices once
        let indices_dev = device.memcpy_stod(indices)?;
        let num_indices = indices.len();

        // Get dtype and pointers for FFI
        let dtype = match dtype_to_repeat_penalty_dtype(self.dtype()) {
            Some(d) => d,
            None => crate::bail!(
                "repeat_penalty is only supported for float types (f16, bf16, f32, f64)"
            ),
        };

        let dst_ptr = self.slice.device_ptr_mut(&stream)?;
        let (indices_ptr, _guard) = indices_dev.device_ptr(&stream);

        unsafe {
            kernels::simple::repeat_penalty::run_repeat_penalty(
                dtype,
                dst_ptr,
                indices_ptr as *const u32,
                num_indices,
                penalty as f64,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        Ok(())
    }

    pub fn div_at_indices(&self, _layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        forbidden_alloc::record("CudaStorage::div_at_indices", self.slice.byte_len());
        let device = self.device().clone();

        // Early return for empty indices
        if indices.is_empty() {
            let slice = match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            };
            return Ok(Self {
                slice,
                device,
                backing: Backing::Owned,
                anchor: None,
            });
        }

        // Clone and then mutate in-place
        let mut result = Self {
            slice: match &self.slice {
                CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.try_clone().w()?),
                CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.try_clone().w()?),
                CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.try_clone().w()?),
                CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.try_clone().w()?),
                CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.try_clone().w()?),
                CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.try_clone().w()?),
                CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.try_clone().w()?),
                CudaStorageSlice::F8E4M3(s) => CudaStorageSlice::F8E4M3(s.try_clone().w()?),
                CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
            },
            device,
            backing: Backing::Owned,
            anchor: None,
        };

        // Use in-place mutation method
        result.div_at_indices_mut(_layout, indices, value)?;
        Ok(result)
    }
}
