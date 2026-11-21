//! The CPU reductions: sums over dimensions and index reductions.
//!
//! `ReduceSum` has a fast path for reductions over trailing dimensions of a
//! contiguous layout, which it detects by checking that the reduced dimensions run
//! from the first to the last. Otherwise it zeroes the reduced indices and uses a
//! divmod to walk the layout. `ReduceIndex` replaces its accumulator whenever the
//! comparison selects the new value, and so returns the position of the selected
//! value.

use super::*;

pub(super) struct ReduceIndex {
    pub(super) reduce_dim_index: usize,
    pub(super) use_min: bool,
    pub(super) return_index: bool,
}

impl ReduceIndex {
    // The value gets replaced if f(s[current_acc], s[i]) returns true.
    #[inline(always)]
    pub(super) fn fold_impl<T, U, F, G>(
        &self,
        src: &[T],
        src_l: &Layout,
        f: F,
        g: G,
    ) -> Result<Vec<U>>
    where
        T: Clone + Copy,
        U: Clone + Copy,
        F: Fn(T, T) -> bool,
        G: Fn(T, usize) -> U,
    {
        let reduce_dim_size = src_l.dims()[self.reduce_dim_index];
        let reduce_dim_stride = src_l.stride()[self.reduce_dim_index];
        let dst_len = src_l.shape().elem_count() / reduce_dim_size;
        let mut dst: Vec<U> = Vec::with_capacity(dst_len);
        let dst_to_set = dst.spare_capacity_mut();
        let dst_to_set =
            unsafe { std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(dst_to_set) };
        match src_l.contiguous_offsets() {
            Some((o1, o2)) => {
                let src = &src[o1..o2];
                if reduce_dim_stride == 1 {
                    for (start_src_i, dst_v) in dst_to_set.iter_mut().enumerate() {
                        let start_src_i = start_src_i * reduce_dim_size;
                        let src = &src[start_src_i..start_src_i + reduce_dim_size];
                        let mut acc = 0;
                        let mut val = src[0];
                        for (src_i, &s) in src.iter().enumerate() {
                            if f(val, s) {
                                acc = src_i;
                                val = s
                            }
                        }
                        *dst_v = g(val, acc)
                    }
                } else {
                    for (start_src_i, dst_v) in dst_to_set.iter_mut().enumerate() {
                        let (p, q) = (
                            start_src_i / reduce_dim_stride,
                            start_src_i % reduce_dim_stride,
                        );
                        // start_src_i = p * reduce_dim_stride + q
                        let start_src_i = p * reduce_dim_stride * reduce_dim_size + q;
                        let src = &src[start_src_i..];
                        let mut acc = 0;
                        let mut val = src[0];
                        for src_i in 0..reduce_dim_size {
                            let s = src[src_i * reduce_dim_stride];
                            if f(val, s) {
                                acc = src_i;
                                val = s
                            }
                        }
                        *dst_v = g(val, acc)
                    }
                }
            }
            None => {
                let l = src_l.narrow(self.reduce_dim_index, 0, 1)?;
                for (unstr_index, src_index) in l.strided_index().enumerate() {
                    let src = &src[src_index..];
                    let mut acc = 0;
                    let mut val = src[0];
                    for src_i in 0..reduce_dim_size {
                        let s = src[src_i * reduce_dim_stride];
                        if f(val, s) {
                            acc = src_i;
                            val = s
                        }
                    }
                    dst_to_set[unstr_index] = g(val, acc)
                }
            }
        }
        unsafe { dst.set_len(dst_len) };
        Ok(dst)
    }
}

impl Map1Any for ReduceIndex {
    #[inline(always)]
    fn f<T: WithDType, W: Fn(Vec<T>) -> CpuStorage>(
        &self,
        src: &[T],
        src_l: &Layout,
        wrap: W,
    ) -> Result<CpuStorage> {
        if src_l.shape().elem_count() == 0 {
            Err(Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dst = match (self.return_index, self.use_min) {
            (false, true) => wrap(self.fold_impl(src, src_l, |x, y| x > y, |v, _i| v)?),
            (false, false) => wrap(self.fold_impl(src, src_l, |x, y| x < y, |v, _i| v)?),
            (true, true) => {
                CpuStorage::U32(self.fold_impl(src, src_l, |x, y| x > y, |_v, i| i as u32)?)
            }
            (true, false) => {
                CpuStorage::U32(self.fold_impl(src, src_l, |x, y| x < y, |_v, i| i as u32)?)
            }
        };
        Ok(dst)
    }
}

pub(super) struct ReduceSum<'a> {
    pub(super) dst_shape: &'a Shape,
    pub(super) reduce_dims: &'a [usize],
    pub(super) reduce_dims_and_stride: Vec<(usize, usize)>,
}

impl ReduceSum<'_> {
    #[inline(always)]
    pub(super) fn fold_impl<T>(&self, src: &[T], src_l: &Layout, start_elt: T) -> Result<Vec<T>>
    where
        T: WithDType,
    {
        let mut dst = vec![start_elt; self.dst_shape.elem_count()];
        match src_l.contiguous_offsets() {
            Some((o1, o2)) => {
                let src = &src[o1..o2];
                // Handle the case where we reduce over the last dimensions separately as it is
                // fairly common and easy to optimize. This rely on the layout being contiguous!
                // reduce_dims is sorted, check if it is ranging from a to n-1.
                let reduce_over_last_dims = self
                    .reduce_dims
                    .iter()
                    .rev()
                    .enumerate()
                    .all(|(i, &v)| v == src_l.shape().rank() - 1 - i);
                if reduce_over_last_dims {
                    let reduce_sz = self
                        .reduce_dims_and_stride
                        .iter()
                        .map(|(u, _)| u)
                        .product::<usize>();
                    for (dst_i, dst_v) in dst.iter_mut().enumerate() {
                        let src_i = dst_i * reduce_sz;
                        unsafe {
                            T::vec_reduce_sum(
                                src[src_i..src_i + reduce_sz].as_ptr(),
                                dst_v,
                                reduce_sz,
                            )
                        };
                    }
                    return Ok(dst);
                };
                for (unstr_index, &src) in src.iter().enumerate() {
                    let mut dst_index = unstr_index;
                    // Set the reduce_dims indexes to 0.
                    for &(dim, stride) in self.reduce_dims_and_stride.iter() {
                        // The compiler is able to optimize the following in a single divmod op.
                        let (pre, post) = (dst_index / stride, dst_index % stride);
                        dst_index = (pre / dim) * stride + post;
                    }
                    dst[dst_index] += src;
                }
            }
            None => {
                for (unstr_index, src_index) in src_l.strided_index().enumerate() {
                    let mut dst_index = unstr_index;
                    // Set the reduce_dims indexes to 0.
                    for &(dim, stride) in self.reduce_dims_and_stride.iter() {
                        // The compiler is able to optimize the following in a single divmod op.
                        let (pre, post) = (dst_index / stride, dst_index % stride);
                        dst_index = (pre / dim) * stride + post;
                    }
                    dst[dst_index] += src[src_index];
                }
            }
        }
        Ok(dst)
    }
}

impl Map1 for ReduceSum<'_> {
    #[inline(always)]
    fn f<T: WithDType>(&self, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        self.fold_impl(src, src_l, T::zero())
    }
}
