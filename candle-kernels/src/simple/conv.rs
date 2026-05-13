//! FFI bindings for convolution operation dispatchers
//!
//! Provides unified interfaces to dispatch convolution operations based on
//! data type enums.
//!
//! Operations:
//! - conv1d: 1D convolution
//! - conv2d: 2D convolution
//! - conv_transpose1d: 1D transposed convolution
//! - conv_transpose2d: 2D transposed convolution
//! - im2col: Image to column transformation (2D)
//! - im2col1d: Image to column transformation (1D)
//! - col2im1d: Column to image transformation (1D)
//! - avg_pool2d: 2D average pooling
//! - max_pool2d: 2D max pooling
//! - upsample_nearest2d: 2D nearest neighbor upsampling

use core::ffi::c_void;

/// Data type enum for convolution operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    U8 = 4,
    U32 = 5,
}

impl ConvDType {
    /// Returns true if this dtype is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            ConvDType::F32 | ConvDType::F64 | ConvDType::F16 | ConvDType::BF16
        )
    }

    /// Returns true if this dtype is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self, ConvDType::U8 | ConvDType::U32)
    }
}

extern "C" {
    /// Dispatcher for 1D convolution.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `dst_numel`: Total number of output elements (for grid sizing)
    /// - `src_numel`: Total number of source elements
    /// - `l_out`: Output length
    /// - `stride`: Convolution stride
    /// - `padding`: Convolution padding
    /// - `dilation`: Convolution dilation
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `kernel`: Convolution kernel
    /// - `dst`: Destination tensor
    pub fn run_conv1d(
        dtype: i32,
        dst_numel: usize,
        src_numel: usize,
        l_out: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 2D convolution.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `dst_numel`: Total number of output elements (for grid sizing)
    /// - `src_numel`: Total number of source elements
    /// - `w_out`: Output width
    /// - `h_out`: Output height
    /// - `stride`: Convolution stride
    /// - `padding`: Convolution padding
    /// - `dilation`: Convolution dilation
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `kernel`: Convolution kernel
    /// - `dst`: Destination tensor
    pub fn run_conv2d(
        dtype: i32,
        dst_numel: usize,
        src_numel: usize,
        w_out: usize,
        h_out: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 1D transposed convolution.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `src_numel`: Total number of source elements
    /// - `l_out`: Output length
    /// - `stride`: Convolution stride
    /// - `padding`: Convolution padding
    /// - `out_padding`: Output padding
    /// - `dilation`: Convolution dilation
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `kernel`: Convolution kernel
    /// - `dst`: Destination tensor
    pub fn run_conv_transpose1d(
        dtype: i32,
        dst_numel: usize,
        src_numel: usize,
        l_out: usize,
        stride: usize,
        padding: usize,
        out_padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 2D transposed convolution.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `dst_numel`: Total number of output elements (for grid sizing)
    /// - `src_numel`: Total number of source elements
    /// - `w_out`: Output width
    /// - `h_out`: Output height
    /// - `stride`: Convolution stride
    /// - `padding`: Convolution padding
    /// - `out_padding`: Output padding
    /// - `dilation`: Convolution dilation
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `kernel`: Convolution kernel
    /// - `dst`: Destination tensor
    pub fn run_conv_transpose2d(
        dtype: i32,
        dst_numel: usize,
        src_numel: usize,
        w_out: usize,
        h_out: usize,
        stride: usize,
        padding: usize,
        out_padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 2D im2col transformation.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `dst_numel`: Total number of destination elements
    /// - `h_out`: Output height
    /// - `w_out`: Output width
    /// - `h_k`: Kernel height
    /// - `w_k`: Kernel width
    /// - `stride`: Convolution stride
    /// - `padding`: Convolution padding
    /// - `dilation`: Convolution dilation
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    pub fn run_im2col(
        dtype: i32,
        dst_numel: usize,
        h_out: usize,
        w_out: usize,
        h_k: usize,
        w_k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 1D im2col transformation.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `dst_numel`: Total number of destination elements
    /// - `l_out`: Output length
    /// - `l_k`: Kernel length
    /// - `stride`: Convolution stride
    /// - `padding`: Convolution padding
    /// - `dilation`: Convolution dilation
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    pub fn run_im2col1d(
        dtype: i32,
        dst_numel: usize,
        l_out: usize,
        l_k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 1D col2im transformation.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `dst_el`: Total number of destination elements
    /// - `l_out`: Output length
    /// - `l_in`: Input length
    /// - `c_out`: Output channels
    /// - `k_size`: Kernel size
    /// - `stride`: Convolution stride
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    pub fn run_col2im1d(
        dtype: i32,
        dst_el: usize,
        l_out: usize,
        l_in: usize,
        c_out: usize,
        k_size: usize,
        stride: usize,
        src: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 2D average pooling.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `src_numel`: Total number of source elements
    /// - `w_k`: Kernel width
    /// - `h_k`: Kernel height
    /// - `w_stride`: Width stride
    /// - `h_stride`: Height stride
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    pub fn run_avg_pool2d(
        dtype: i32,
        src_numel: usize,
        w_k: usize,
        h_k: usize,
        w_stride: usize,
        h_stride: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 2D max pooling.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `src_numel`: Total number of source elements
    /// - `w_k`: Kernel width
    /// - `h_k`: Kernel height
    /// - `w_stride`: Width stride
    /// - `h_stride`: Height stride
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    pub fn run_max_pool2d(
        dtype: i32,
        src_numel: usize,
        w_k: usize,
        h_k: usize,
        w_stride: usize,
        h_stride: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    );

    /// Dispatcher for 2D nearest neighbor upsampling.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see ConvDType enum values)
    /// - `w_out`: Output width
    /// - `h_out`: Output height
    /// - `w_scale`: Width scale factor
    /// - `h_scale`: Height scale factor
    /// - `info`: Pointer to dims and strides info
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    pub fn run_upsample_nearest2d(
        dtype: i32,
        w_out: usize,
        h_out: usize,
        w_scale: f64,
        h_scale: f64,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    );
}

/// Safe wrapper for convolution operations dispatcher
pub struct ConvDispatcher;

impl ConvDispatcher {
    /// Perform 1D convolution
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn conv1d(
        dtype: ConvDType,
        dst_numel: usize,
        src_numel: usize,
        num_dims: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    ) {
        run_conv1d(
            dtype as i32,
            dst_numel,
            src_numel,
            num_dims,
            stride,
            padding,
            dilation,
            info,
            src,
            kernel,
            dst,
        )
    }

    /// Perform 2D convolution
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn conv2d(
        dtype: ConvDType,
        dst_numel: usize,
        src_numel: usize,
        w_out: usize,
        h_out: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    ) {
        run_conv2d(
            dtype as i32,
            dst_numel,
            src_numel,
            w_out,
            h_out,
            stride,
            padding,
            dilation,
            info,
            src,
            kernel,
            dst,
        )
    }

    /// Perform 1D transposed convolution
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn conv_transpose1d(
        dtype: ConvDType,
        dst_numel: usize,
        src_numel: usize,
        l_out: usize,
        stride: usize,
        padding: usize,
        out_padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    ) {
        run_conv_transpose1d(
            dtype as i32,
            dst_numel,
            src_numel,
            l_out,
            stride,
            padding,
            out_padding,
            dilation,
            info,
            src,
            kernel,
            dst,
        )
    }

    /// Perform 2D transposed convolution
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn conv_transpose2d(
        dtype: ConvDType,
        dst_numel: usize,
        src_numel: usize,
        w_out: usize,
        h_out: usize,
        stride: usize,
        padding: usize,
        out_padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        kernel: *const c_void,
        dst: *mut c_void,
    ) {
        run_conv_transpose2d(
            dtype as i32,
            dst_numel,
            src_numel,
            w_out,
            h_out,
            stride,
            padding,
            out_padding,
            dilation,
            info,
            src,
            kernel,
            dst,
        )
    }

    /// Perform 2D im2col transformation
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn im2col(
        dtype: ConvDType,
        dst_numel: usize,
        h_out: usize,
        w_out: usize,
        h_k: usize,
        w_k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    ) {
        run_im2col(
            dtype as i32,
            dst_numel,
            h_out,
            w_out,
            h_k,
            w_k,
            stride,
            padding,
            dilation,
            info,
            src,
            dst,
        )
    }

    /// Perform 1D im2col transformation
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn im2col1d(
        dtype: ConvDType,
        dst_numel: usize,
        l_out: usize,
        l_k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    ) {
        run_im2col1d(
            dtype as i32,
            dst_numel,
            l_out,
            l_k,
            stride,
            padding,
            dilation,
            info,
            src,
            dst,
        )
    }

    /// Perform 1D col2im transformation
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    pub unsafe fn col2im1d(
        dtype: ConvDType,
        dst_el: usize,
        l_out: usize,
        l_in: usize,
        c_out: usize,
        k_size: usize,
        stride: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) {
        run_col2im1d(
            dtype as i32,
            dst_el,
            l_out,
            l_in,
            c_out,
            k_size,
            stride,
            src,
            dst,
        )
    }

    /// Perform 2D average pooling
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn avg_pool2d(
        dtype: ConvDType,
        src_numel: usize,
        w_k: usize,
        h_k: usize,
        w_stride: usize,
        h_stride: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    ) {
        run_avg_pool2d(
            dtype as i32,
            src_numel,
            w_k,
            h_k,
            w_stride,
            h_stride,
            info,
            src,
            dst,
        )
    }

    /// Perform 2D max pooling
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn max_pool2d(
        dtype: ConvDType,
        src_numel: usize,
        w_k: usize,
        h_k: usize,
        w_stride: usize,
        h_stride: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    ) {
        run_max_pool2d(
            dtype as i32,
            src_numel,
            w_k,
            h_k,
            w_stride,
            h_stride,
            info,
            src,
            dst,
        )
    }

    /// Perform 2D nearest neighbor upsampling
    ///
    /// # Safety
    /// - All pointers must be valid for the specified data type
    /// - `info` must point to a valid dims/strides array
    pub unsafe fn upsample_nearest2d(
        dtype: ConvDType,
        w_out: usize,
        h_out: usize,
        w_scale: f64,
        h_scale: f64,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    ) {
        run_upsample_nearest2d(dtype as i32, w_out, h_out, w_scale, h_scale, info, src, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_enum_values() {
        assert_eq!(ConvDType::F32 as i32, 0);
        assert_eq!(ConvDType::F64 as i32, 1);
        assert_eq!(ConvDType::F16 as i32, 2);
        assert_eq!(ConvDType::BF16 as i32, 3);
        assert_eq!(ConvDType::U8 as i32, 4);
        assert_eq!(ConvDType::U32 as i32, 5);
    }

    #[test]
    fn test_dtype_classification() {
        assert!(ConvDType::F32.is_float());
        assert!(ConvDType::F64.is_float());
        assert!(ConvDType::F16.is_float());
        assert!(ConvDType::BF16.is_float());
        assert!(!ConvDType::U8.is_float());
        assert!(!ConvDType::U32.is_float());

        assert!(!ConvDType::F32.is_integer());
        assert!(!ConvDType::F64.is_integer());
        assert!(ConvDType::U8.is_integer());
        assert!(ConvDType::U32.is_integer());
    }
}
