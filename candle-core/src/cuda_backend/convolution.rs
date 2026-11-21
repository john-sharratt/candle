//! Launches the convolution, pooling and upsampling kernels over a storage's
//! values.
//!
//! Each operation is a small unit struct implementing `Map1` or `Map2`, which
//! resolves its dtype for the FFI dispatcher and then launches. Destination
//! buffers are allocated uninitialised because each launch writes every
//! element: one thread per destination element, and the upsampling launch
//! covers the whole destination. The cuDNN-backed convolution takes its output
//! from the arena its input came from, since the convolution's result is the
//! largest allocation in a convolutional layer.

use super::*;

#[allow(unused)]
pub(super) struct Im2Col1D {
    pub(super) l_k: usize,
    pub(super) stride: usize,
    pub(super) dilation: usize,
    pub(super) padding: usize,
}

impl Im2Col1D {
    #[allow(unused)]
    pub(super) fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_im2col1d");
        let shape = layout.shape();
        let dims = shape.dims();
        let l_out = self.l_out(dims[2]);
        let dst_numel = dims[0] * l_out * dims[1] * self.l_k;
        let ds = dev.info_table(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("im2col1d not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        // SAFETY: Set later by running the kernel.
        let (dst, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_numel, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_im2col1d(
                    dtype,
                    // One thread per destination element: every one is
                    // written, so the uninitialised allocation is fully set.
                    dst_numel,
                    l_out,
                    self.l_k,
                    self.stride,
                    self.padding,
                    self.dilation,
                    ds_ptr as *const usize,
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((dst, out_backing))
    }
}

#[allow(unused)]
pub(super) struct Im2Col {
    pub(super) h_k: usize,
    pub(super) w_k: usize,
    pub(super) stride: usize,
    pub(super) dilation: usize,
    pub(super) padding: usize,
}

impl Im2Col {
    #[allow(unused)]
    pub(super) fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_im2col");
        let shape = layout.shape();
        let dims = shape.dims();
        let (h_out, w_out) = self.hw_out(dims[2], dims[3]);
        let dst_el = dims[0] * h_out * w_out * dims[1] * self.h_k * self.w_k;
        let ds = dev.info_table(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("im2col not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        // SAFETY: Set later by running the kernel.
        let (dst, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_im2col(
                    dtype,
                    dst_el,
                    h_out,
                    w_out,
                    self.h_k,
                    self.w_k,
                    self.stride,
                    self.padding,
                    self.dilation,
                    ds_ptr as *const usize,
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((dst, out_backing))
    }
}

pub(super) struct Conv1D<'a>(pub(super) &'a crate::conv::ParamsConv1D);

impl Map2 for Conv1D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv1d");
        // Kernel shape: (c_out, c_in_k, k_size)
        // Input shape: (b_size, c_in, l_in) or (c_in, l_in)
        let p = &self.0;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv1d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else if dims.len() == 2 {
            [&[1], dims, &[1], inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv1d(
                    dtype,
                    dst_el,
                    el,
                    l_out,
                    p.stride,
                    p.padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

pub(super) struct Conv2D<'a>(pub(super) &'a crate::conv::ParamsConv2D);

impl Map2 for Conv2D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv2d");
        // Kernel shape: (c_out, c_in_k, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv2d(
                    dtype,
                    dst_el,
                    el,
                    out_w,
                    out_h,
                    p.stride,
                    p.padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

pub(super) struct Col2Im1D {
    pub(super) stride: usize,
}

impl Map1 for Col2Im1D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        col: &CudaSlice<T>,
        dev: &CudaDevice,
        l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_col2im1d");
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let dst_el = b_size * c_out * l_out;
        let (im, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("col2im1d not supported for dtype {:?}", T::DTYPE),
        };

        let stream = dev.cuda_stream();
        {
            let (col_ptr, _col_guard) = col.device_ptr(&stream);
            let (im_ptr, _im_guard) = im.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_col2im1d(
                    dtype,
                    dst_el,
                    l_out,
                    l_in,
                    c_out,
                    k_size,
                    stride,
                    col_ptr as *const std::ffi::c_void,
                    im_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((im, out_backing))
    }
}

pub(super) struct ConvTranspose1D<'a>(pub(super) &'a crate::conv::ParamsConvTranspose1D);

impl Map2 for ConvTranspose1D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv_transpose1d");
        // Kernel shape: (c_in_k, c_out, l_k)
        // Input shape: (b_size, c_in, l_in)
        let p = &self.0;
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv_transpose1d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose1d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv_transpose1d(
                    dtype,
                    dst_el,
                    el,
                    l_out,
                    p.stride,
                    p.padding,
                    p.output_padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

pub(super) struct ConvTranspose2D<'a>(pub(super) &'a crate::conv::ParamsConvTranspose2D);

impl Map2 for ConvTranspose2D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        inp_l: &Layout,
        k: &CudaSlice<T>,
        k_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_conv_transpose2d");
        // Kernel shape: (c_in_k, c_out, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("conv_transpose2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}")
        };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_conv_transpose2d(
                    dtype,
                    dst_el,
                    el,
                    out_w,
                    out_h,
                    p.stride,
                    p.padding,
                    p.output_padding,
                    p.dilation,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}

pub(super) enum PoolOp {
    Max,
    Avg,
}

pub(super) struct Pool2D {
    pub(super) w_k: usize,
    pub(super) h_k: usize,
    pub(super) w_stride: usize,
    pub(super) h_stride: usize,
    pub(super) op: PoolOp,
}

impl Map1 for Pool2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        dev: &CudaDevice,
        inp_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_pool2d");
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for pool {dims:?}")
        };
        let el = shape.elem_count();
        let out_w = (dims[2] - self.w_k) / self.w_stride + 1;
        let out_h = (dims[3] - self.h_k) / self.h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("pool2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = dev.info_table(&ds)?;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                match self.op {
                    PoolOp::Max => {
                        kernels::simple::conv::run_max_pool2d(
                            dtype,
                            el,
                            self.w_k,
                            self.h_k,
                            self.w_stride,
                            self.h_stride,
                            ds_ptr as *const usize,
                            inp_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                    PoolOp::Avg => {
                        kernels::simple::conv::run_avg_pool2d(
                            dtype,
                            el,
                            self.w_k,
                            self.h_k,
                            self.w_stride,
                            self.h_stride,
                            ds_ptr as *const usize,
                            inp_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
            }
        }
        Ok((out, out_backing))
    }
}

pub(super) struct UpsampleNearest2D(pub(super) usize, pub(super) usize);

impl Map1 for UpsampleNearest2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        dev: &CudaDevice,
        inp_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_upsample_nearest2d");
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        };
        let (out_w, out_h) = (self.0, self.1);
        let dst_el = out_w * out_h * dims[0] * dims[1];

        // Get dtype for FFI dispatcher
        let dtype = match dtype_to_conv_dtype(T::DTYPE) {
            Some(d) => d,
            None => crate::bail!("upsample_nearest2d not supported for dtype {:?}", T::DTYPE),
        };

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        let ds = dev.info_table(&ds)?;
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;

        let stream = dev.cuda_stream();
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            unsafe {
                kernels::simple::conv::run_upsample_nearest2d(
                    dtype,
                    // The whole destination. The dispatcher used to size its own
                    // launch from `out_w * out_h`, which is one channel of one
                    // batch item — everything past that came back as whatever
                    // the allocation held.
                    dst_el,
                    out_w,
                    out_h,
                    scale_w,
                    scale_h,
                    ds_ptr as *const usize,
                    inp_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((out, out_backing))
    }
}
