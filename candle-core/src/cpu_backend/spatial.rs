//! The CPU spatial operations: pooling, nearest-neighbour upsampling, convolutions
//! and their transposes.
//!
//! Each operation is a map over one input, and its comment gives the output shape,
//! such as `[batch, channels, length]` for a 1-D convolution. Whether a convolution
//! goes through im2col is set by the constants in `mod.rs`. In the transposed
//! forms each output position has a unique destination index, so the parallel tasks,
//! one per output channel, never write the same location.

use super::*;

pub(super) struct AvgPool2D(pub(super) (usize, usize), pub(super) (usize, usize));

impl Map1 for AvgPool2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_index = layout.start_offset();
        let mut dst = vec![T::zero(); b_sz * c * h_out * w_out];
        let scale = 1f64 / (k_h * k_w) as f64;
        let scale = T::from_f64(scale);
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * h_out * w_out..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * h_out * w_out..];
                let src_index = src_index + c_idx * stride[1];
                for h_idx in 0..h_out {
                    for w_idx in 0..w_out {
                        let mut sum = T::zero();
                        for m in 0..k_h {
                            for n in 0..k_w {
                                let m = s_h * h_idx + m;
                                let n = s_w * w_idx + n;
                                sum += src[src_index + m * stride_h + n * stride_w]
                            }
                        }
                        dst[h_idx * w_out + w_idx] = sum * scale;
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct MaxPool2D(pub(super) (usize, usize), pub(super) (usize, usize));

impl Map1 for MaxPool2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_index = layout.start_offset();
        let mut dst = vec![T::zero(); b_sz * c * h_out * w_out];
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * h_out * w_out..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * h_out * w_out..];
                let src_index = src_index + c_idx * stride[1];
                for h_idx in 0..h_out {
                    for w_idx in 0..w_out {
                        let mut largest =
                            src[src_index + s_h * h_idx * stride_h + s_w * w_idx * stride_w];
                        for m in 0..k_h {
                            for n in 0..k_w {
                                let m = s_h * h_idx + m;
                                let n = s_w * w_idx + n;
                                if largest < src[src_index + m * stride_h + n * stride_w] {
                                    largest = src[src_index + m * stride_h + n * stride_w]
                                }
                            }
                        }
                        dst[h_idx * w_out + w_idx] = largest;
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct UpsampleNearest1D(pub(super) usize);

impl Map1 for UpsampleNearest1D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: Specialized implementation for the case 2*sz?
        let dst_sz = self.0;
        let (b_sz, c, src_sz) = layout.shape().dims3()?;
        let stride = layout.stride();
        let stride_sz = stride[2];
        let src_index = layout.start_offset();
        let scale_sz = src_sz as f64 / dst_sz as f64;
        let mut dst = vec![T::zero(); b_sz * c * dst_sz];
        let src_idxs = (0..dst_sz)
            .map(|idx| usize::min(src_sz - 1, (idx as f64 * scale_sz) as usize))
            .collect::<Vec<_>>();
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * dst_sz..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * dst_sz..];
                let src_index = src_index + c_idx * stride[1];
                for (idx, src_idx) in src_idxs.iter().enumerate() {
                    dst[idx] = src[src_index + src_idx * stride_sz]
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct UpsampleNearest2D(pub(super) usize, pub(super) usize);

impl Map1 for UpsampleNearest2D {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: Specialized implementation for the case 2*h, 2*w?
        let (dst_h, dst_w) = (self.0, self.1);
        let (b_sz, c, src_h, src_w) = layout.shape().dims4()?;
        let stride = layout.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let src_index = layout.start_offset();
        let scale_h = src_h as f64 / dst_h as f64;
        let scale_w = src_w as f64 / dst_w as f64;
        let mut dst = vec![T::zero(); b_sz * c * dst_h * dst_w];
        let src_h_idxs = (0..dst_h)
            .map(|h_idx| usize::min(src_h - 1, (h_idx as f64 * scale_h) as usize))
            .collect::<Vec<_>>();
        let src_w_idxs = (0..dst_w)
            .map(|w_idx| usize::min(src_w - 1, (w_idx as f64 * scale_w) as usize))
            .collect::<Vec<_>>();
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * dst_h * dst_w..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * dst_h * dst_w..];
                let src_index = src_index + c_idx * stride[1];
                for (h_idx, src_h_idx) in src_h_idxs.iter().enumerate() {
                    for (w_idx, src_w_idx) in src_w_idxs.iter().enumerate() {
                        let src_index = src_index + src_h_idx * stride_h + src_w_idx * stride_w;
                        dst[h_idx * dst_w + w_idx] = src[src_index]
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct Conv1D<'a>(pub(super) &'a crate::conv::ParamsConv1D);

impl Map2 for Conv1D<'_> {
    const OP: &'static str = "conv1d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = crate::shape::dims3(inp_l.stride())?;
        let (k_s0, k_s1, k_s2) = crate::shape::dims3(k_l.stride())?;
        let l_out = p.l_out();
        let dst_elems = p.c_out * l_out * p.b_size;
        // The output shape is [b_size, c_out, l_out]
        let dst = vec![T::zero(); dst_elems];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        for b_idx in 0..p.b_size {
            for src_l in 0..p.l_in {
                for src_c_idx in 0..p.c_in {
                    let inp_idx = b_idx * inp_s0 + src_c_idx * inp_s1 + src_l * inp_s2;
                    inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in + src_c_idx] = inp[inp_idx]
                }
            }
        }

        for offset in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let dst_idx = dst_c_idx * l_out;
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[dst_c_idx * k_s0 + c_in_idx * k_s1 + offset * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    let dst_idx = dst_idx + b_idx * p.c_out * l_out;
                    for dst_l in 0..l_out {
                        let dst_idx = dst_idx + dst_l;
                        let src_l = p.stride * dst_l + offset * p.dilation;
                        if src_l < p.padding || src_l >= p.padding + p.l_in {
                            continue;
                        }
                        let src_l = src_l - p.padding;
                        let inp_cont = &inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in..];
                        assert!(inp_cont.len() >= p.c_in);
                        assert!(k_cont.len() >= p.c_in);
                        let mut d = T::zero();
                        unsafe { T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in) }
                        let dst_p = dst.as_ptr();
                        // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                        // the different tasks so no two threads can try to write at the same
                        // location.
                        unsafe {
                            let ptr = dst_p.add(dst_idx) as *mut T;
                            *ptr += d
                        }
                    }
                }
            })
        }
        Ok(dst)
    }
}

pub(super) struct Im2Col1D {
    pub(super) l_k: usize,
    pub(super) stride: usize,
    pub(super) dilation: usize,
    pub(super) padding: usize,
}

impl Im2Col1D {
    pub(super) fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let &Self {
            l_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, l) = layout.shape().dims3()?;
        let l_out = self.l_out(l);
        let src = &vs[layout.start_offset()..];
        let mut dst = vec![T::zero(); b * l_out * c * l_k];
        let (src_s0, src_s1, src_s2) = {
            let s = layout.stride();
            (s[0], s[1], s[2])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - l_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * l_out * c * l_k;
            for l_idx in 0..l_out {
                let dst_idx = dst_idx + l_idx * c * l_k;
                for c_idx in 0..c {
                    let dst_idx = dst_idx + c_idx * l_k;
                    let src_idx = c_idx * src_s1 + src_idx;
                    for l_k_idx in 0..l_k {
                        let src_l = l_idx * stride + l_k_idx * dilation;
                        if padding != 0 && (src_l < padding || src_l >= l + padding) {
                            continue;
                        }
                        let src_l = src_l - padding;
                        let src_idx = src_idx + src_l * src_s2;
                        let dst_idx = dst_idx + l_k_idx;
                        dst[dst_idx] = src[src_idx]
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct Im2Col {
    pub(super) h_k: usize,
    pub(super) w_k: usize,
    pub(super) stride: usize,
    pub(super) dilation: usize,
    pub(super) padding: usize,
}

impl Im2Col {
    pub(super) fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let &Self {
            h_k,
            w_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, h, w) = layout.shape().dims4()?;
        let (h_out, w_out) = self.hw_out(h, w);
        let src = &vs[layout.start_offset()..];
        let mut dst = vec![T::zero(); b * h_out * w_out * c * h_k * w_k];
        let (src_s0, src_s1, src_s2, src_s3) = {
            let s = layout.stride();
            (s[0], s[1], s[2], s[3])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - h_k = w_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * h_out * w_out * c * h_k * w_k;
            for h_idx in 0..h_out {
                let dst_idx = dst_idx + h_idx * w_out * c * h_k * w_k;
                for w_idx in 0..w_out {
                    let dst_idx = dst_idx + w_idx * c * h_k * w_k;
                    for c_idx in 0..c {
                        let dst_idx = dst_idx + c_idx * h_k * w_k;
                        let src_idx = c_idx * src_s1 + src_idx;
                        for h_k_idx in 0..h_k {
                            let src_h = h_idx * stride + h_k_idx * dilation;
                            if padding != 0 && (src_h < padding || src_h >= h + padding) {
                                continue;
                            }
                            let src_h = src_h - padding;
                            let src_idx = src_idx + src_h * src_s2;
                            let dst_idx = dst_idx + h_k_idx * w_k;
                            for w_k_idx in 0..w_k {
                                let src_w = w_idx * stride + w_k_idx * dilation;
                                if padding != 0 && (src_w < padding || src_w >= w + padding) {
                                    continue;
                                }
                                let src_w = src_w - padding;
                                let src_idx = src_idx + src_w * src_s3;
                                let dst_idx = dst_idx + w_k_idx;
                                dst[dst_idx] = src[src_idx]
                            }
                        }
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct Col2Im1D {
    pub(super) stride: usize,
}

impl Map1 for Col2Im1D {
    fn f<T: WithDType>(&self, col: &[T], l: &Layout) -> Result<Vec<T>> {
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let mut im = vec![T::zero(); b_size * c_out * l_out];
        let (dst_s0, dst_s1) = (c_out * l_out, l_out);
        let (src_s0, src_s1, src_s2) = (c_out * k_size * l_in, c_out * k_size, k_size);
        for l_in_i in 0..l_in {
            for k_i in 0..k_size {
                let l_out_i = l_in_i * stride + k_i;
                for b_i in 0..b_size {
                    for c_i in 0..c_out {
                        let dst_idx = b_i * dst_s0 + c_i * dst_s1 + l_out_i;
                        let src_idx = b_i * src_s0 + l_in_i * src_s1 + c_i * src_s2 + k_i;
                        im[dst_idx] += col[src_idx]
                    }
                }
            }
        }
        Ok(im)
    }
}

pub(super) struct ConvTranspose1D<'a>(pub(super) &'a crate::conv::ParamsConvTranspose1D);

impl Map2 for ConvTranspose1D<'_> {
    const OP: &'static str = "conv_transpose1d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = crate::shape::dims3(inp_l.stride())?;
        let (k_s0, k_s1, k_s2) = crate::shape::dims3(k_l.stride())?;
        let l_out = p.l_out();

        // Output shape: [b_size, c_out, l_out].
        let dst_elems = p.c_out * l_out * p.b_size;
        let dst = vec![T::zero(); dst_elems];
        let dst_s0 = p.c_out * l_out;
        let dst_s1 = l_out;
        let dst_s2 = 1;

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        let cont_s0 = p.l_in * p.c_in;
        let cont_s1 = p.c_in;
        for b_idx in 0..p.b_size {
            for l_idx in 0..p.l_in {
                for c_idx in 0..p.c_in {
                    let src_idx = b_idx * inp_s0 + c_idx * inp_s1 + l_idx * inp_s2;
                    let dst_idx = b_idx * cont_s0 + l_idx * cont_s1 + c_idx;
                    inp_cont[dst_idx] = inp[src_idx]
                }
            }
        }

        for k_idx in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[c_in_idx * k_s0 + dst_c_idx * k_s1 + k_idx * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    for l_idx in 0..p.l_in {
                        let out_idx = l_idx * p.stride + k_idx * p.dilation;
                        if out_idx < p.padding {
                            continue;
                        }
                        let out_idx = out_idx - p.padding;
                        if out_idx < l_out {
                            let inp_cont = &inp_cont[b_idx * cont_s0 + l_idx * cont_s1..];
                            let dst_idx = b_idx * dst_s0 + out_idx * dst_s2 + dst_c_idx * dst_s1;
                            let mut d = T::zero();
                            unsafe {
                                T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                            }
                            let dst_p = dst.as_ptr();
                            // Safety: dst_idx are uniques per dst_c_idx which is used to
                            // parallelise the different tasks so no two threads can try to
                            // write at the same location.
                            unsafe {
                                let ptr = dst_p.add(dst_idx) as *mut T;
                                *ptr += d
                            }
                        }
                    }
                }
            })
        }
        Ok(dst)
    }
}

pub(super) struct Conv2D<'a>(pub(super) &'a crate::conv::ParamsConv2D);

impl Map2 for Conv2D<'_> {
    const OP: &'static str = "conv2d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }

        for offset_h in 0..p.k_h {
            for offset_w in 0..p.k_w {
                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let dst_idx = dst_c_idx * out_w * out_h;
                    let k_cont = (0..p.c_in)
                        .map(|c_in_idx| {
                            k[dst_c_idx * k_s0
                                + c_in_idx * k_s1
                                + offset_h * k_s2
                                + offset_w * k_s3]
                        })
                        .collect::<Vec<_>>();
                    for b_idx in 0..p.b_size {
                        let dst_idx = dst_idx + b_idx * p.c_out * out_h * out_w;
                        for dst_h in 0..out_h {
                            let dst_idx = dst_idx + dst_h * out_w;
                            let src_h = p.stride * dst_h + offset_h * p.dilation;
                            if src_h < p.padding || src_h >= p.i_h + p.padding {
                                continue;
                            }
                            let src_h = src_h - p.padding;
                            for dst_w in 0..out_w {
                                let dst_idx = dst_idx + dst_w;
                                let src_w = p.stride * dst_w + offset_w * p.dilation;
                                if src_w < p.padding || src_w >= p.i_w + p.padding {
                                    continue;
                                }
                                let src_w = src_w - p.padding;
                                let inp_cont = &inp_cont
                                    [b_idx * cont_s0 + src_h * cont_s1 + src_w * cont_s2..];
                                assert!(inp_cont.len() >= p.c_in);
                                assert!(k_cont.len() >= p.c_in);
                                let mut d = T::zero();
                                unsafe {
                                    T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                                }
                                let dst_p = dst.as_ptr();
                                // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                                // the different tasks so no two threads can try to write at the same
                                // location.
                                unsafe {
                                    let ptr = dst_p.add(dst_idx) as *mut T;
                                    *ptr += d
                                }
                            }
                        }
                    }
                });
            }
        }

        Ok(dst)
    }
}

pub(super) struct ConvTranspose2D<'a>(pub(super) &'a crate::conv::ParamsConvTranspose2D);

impl Map2 for ConvTranspose2D<'_> {
    const OP: &'static str = "conv_transpose2d";
    fn f<T: WithDType>(&self, inp: &[T], inp_l: &Layout, k: &[T], k_l: &Layout) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3) = crate::shape::dims4(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3) = crate::shape::dims4(k_l.stride())?;
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];
        let dst_s0 = p.c_out * out_h * out_w;
        let dst_s1 = out_h * out_w;
        let dst_s2 = out_w;
        let dst_s3 = 1;

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }

        for k_y in 0..p.k_h {
            for k_x in 0..p.k_w {
                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let k_cont = (0..p.c_in)
                        .map(|c_in_idx| {
                            k[c_in_idx * k_s0 + dst_c_idx * k_s1 + k_y * k_s2 + k_x * k_s3]
                        })
                        .collect::<Vec<_>>();
                    for b_idx in 0..p.b_size {
                        for inp_y in 0..p.i_h {
                            for inp_x in 0..p.i_w {
                                let out_x = inp_x * p.stride + k_x * p.dilation;
                                let out_y = inp_y * p.stride + k_y * p.dilation;
                                if out_x < p.padding || out_y < p.padding {
                                    continue;
                                }
                                let out_x = out_x - p.padding;
                                let out_y = out_y - p.padding;
                                if out_x < out_w && out_y < out_h {
                                    let inp_cont = &inp_cont
                                        [b_idx * cont_s0 + inp_y * cont_s1 + inp_x * cont_s2..];
                                    let dst_idx = b_idx * dst_s0
                                        + out_y * dst_s2
                                        + out_x * dst_s3
                                        + dst_c_idx * dst_s1;
                                    let mut d = T::zero();
                                    unsafe {
                                        T::vec_dot(
                                            inp_cont.as_ptr(),
                                            k_cont.as_ptr(),
                                            &mut d,
                                            p.c_in,
                                        )
                                    }
                                    let dst_p = dst.as_ptr();
                                    // Safety: dst_idx are uniques per dst_c_idx which is used to
                                    // parallelise the different tasks so no two threads can try to
                                    // write at the same location.
                                    unsafe {
                                        let ptr = dst_p.add(dst_idx) as *mut T;
                                        *ptr += d
                                    }
                                }
                            }
                        }
                    }
                })
            }
        }
        Ok(dst)
    }
}
