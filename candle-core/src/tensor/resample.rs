//! Resizes and pools spatial tensors.
//!
//! `interpolate1d` and `interpolate2d` resample to a target size by taking the
//! nearest element, and `upsample_nearest1d` and `upsample_nearest2d` are
//! aliases for them. `avg_pool2d` and `max_pool2d` take windowed averages and
//! maxima, and their `_with_stride` forms let the stride differ from the kernel
//! size.

use super::*;

/// The tensor API, generic over how long the storage is valid.
///
/// Unlike [`crate::quantized::LiveQTensor`], this block is *not* split into
/// `'static` constructors and `'w` accessors, because here the `-> Result<Self>`
/// methods are overwhelmingly **views** — `reshape`, `narrow`, `t`,
/// `broadcast_as` — and a view of a lease has to stay a lease. Returning `Self`
/// is what makes that true by construction. The handful of methods that
/// allocate leave `'w` free, which is sound because owned memory outlives every
/// choice of `'w`; the discipline that matters is applied at the *lease*
/// constructors instead, where `'w` is tied to a borrow of whatever owns the
/// bytes.
impl<'w> LiveTensor<'w> {
    /// Interpolate the input tensor to the `target_size` size, taking the value of the nearest element.
    ///
    /// The input tensor should have three dimensions, `(batch, channels, l)`, the returned
    /// tensor also has three dimensions, `(batch, channels, target_size)`.
    pub fn interpolate1d(&self, target_size: usize) -> Result<Self> {
        let (n, c, _l) = self.dims3()?;
        let op = BackpropOp::new1(self, |arg| Op::UpsampleNearest1D { arg, target_size });
        let storage = self
            .storage()
            .upsample_nearest1d(self.layout(), target_size)?;
        Ok(from_storage(storage, (n, c, target_size), op, false))
    }

    /// Alias for `interpolate1d`.
    pub fn upsample_nearest1d(&self, target_size: usize) -> Result<Self> {
        self.interpolate1d(target_size)
    }

    /// Interpolate the input tensor to the `(target_h, target_w)` size, taking the value of the
    /// nearest element.
    ///
    /// The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
    /// tensor also has four dimensions, `(batch, channels, target_h, target_w)`.
    pub fn interpolate2d(&self, target_h: usize, target_w: usize) -> Result<Self> {
        let (n, c, _h, _w) = self.dims4()?;
        let op = BackpropOp::new1(self, |arg| Op::UpsampleNearest2D {
            arg,
            target_h,
            target_w,
        });
        let storage = self
            .storage()
            .upsample_nearest2d(self.layout(), target_h, target_w)?;
        Ok(from_storage(storage, (n, c, target_h, target_w), op, false))
    }

    /// Alias for `interpolate2d`.
    pub fn upsample_nearest2d(&self, target_h: usize, target_w: usize) -> Result<Self> {
        self.interpolate2d(target_h, target_w)
    }

    /// 2D average pooling over an input tensor with multiple channels.
    ///
    /// The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
    /// tensor also has four dimensions, `(batch, channels, h', w')`. The pooling is performed on
    /// the two last dimensions using a kernel of size `sz`. The returned element is the average
    /// value over the kernel window.
    pub fn avg_pool2d<T: crate::ToUsize2>(&self, sz: T) -> Result<Self> {
        let sz = sz.to_usize2();
        self.avg_pool2d_with_stride(sz, sz)
    }

    /// Same as `avg_pool2d` but with a `stride` that can be set to a value different from the
    /// kernel size.
    pub fn avg_pool2d_with_stride<T: crate::ToUsize2>(
        &self,
        kernel_size: T,
        stride: T,
    ) -> Result<Self> {
        let kernel_size = kernel_size.to_usize2();
        let stride = stride.to_usize2();
        let (n, c, h, w) = self.dims4()?;
        if h < kernel_size.0 || w < kernel_size.1 {
            bail!("kernel-size {kernel_size:?} is larger than the input size {h},{w}")
        }
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d
        let h_out = (h - kernel_size.0) / stride.0 + 1;
        let w_out = (w - kernel_size.1) / stride.1 + 1;
        let op = BackpropOp::new1(self, |arg| Op::AvgPool2D {
            arg,
            kernel_size,
            stride,
        });
        let storage = self
            .storage()
            .avg_pool2d(self.layout(), kernel_size, stride)?;
        Ok(from_storage(storage, (n, c, h_out, w_out), op, false))
    }

    /// 2D max pooling over an input tensor with multiple channels.
    ///
    /// The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
    /// tensor also has four dimensions, `(batch, channels, h', w')`. The pooling is performed on
    /// the two last dimensions using a kernel of size `sz`, the returned element is the maximum
    /// value over the kernel window.
    pub fn max_pool2d<T: crate::ToUsize2>(&self, sz: T) -> Result<Self> {
        let sz = sz.to_usize2();
        self.max_pool2d_with_stride(sz, sz)
    }

    /// Same as `max_pool2d` but with a `stride` that can be set to a value different from the
    /// kernel size.
    pub fn max_pool2d_with_stride<T: crate::ToUsize2>(
        &self,
        kernel_size: T,
        stride: T,
    ) -> Result<Self> {
        let kernel_size = kernel_size.to_usize2();
        let stride = stride.to_usize2();
        let (n, c, h, w) = self.dims4()?;
        if h < kernel_size.0 || w < kernel_size.1 {
            bail!("kernel-size {kernel_size:?} is larger than the input size {h},{w}")
        }
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        let h_out = (h - kernel_size.0) / stride.0 + 1;
        let w_out = (w - kernel_size.1) / stride.1 + 1;
        let op = BackpropOp::new1(self, |arg| Op::MaxPool2D {
            arg,
            kernel_size,
            stride,
        });
        let storage = self
            .storage()
            .max_pool2d(self.layout(), kernel_size, stride)?;
        Ok(from_storage(storage, (n, c, h_out, w_out), op, false))
    }
}
