//! The gather, index-select, scatter and index-add operations on CPU storage.
//!
//! `Gather` and `IndexSelect` read elements at index positions. `Scatter` writes
//! through an `ElemUpdate`, either `Set` or `Add`, and `IndexAdd` accumulates rows
//! as PyTorch's `index_add_` does. Each is a map trait implementation driven by the
//! generic dispatch.

use super::*;

pub(super) struct Gather<'a, I: IntDType> {
    pub(super) ids: &'a [I],
    pub(super) ids_l: &'a Layout,
    pub(super) dim: usize,
}

impl<I: IntDType> Map1 for Gather<'_, I> {
    fn f<T: WithDType>(&self, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let ids = match self.ids_l.contiguous_offsets() {
            Some((a, b)) => &self.ids[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((a, b)) => &src[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let dim = self.dim;
        let ids_dims = self.ids_l.dims();
        let src_dims = src_l.dims();
        let dst_len: usize = ids_dims.iter().product();
        let dst_left_len: usize = ids_dims[..dim].iter().product();
        let dst_dim_len = ids_dims[dim];
        let dst_right_len: usize = ids_dims[dim + 1..].iter().product();

        let src_dim_len = src_dims[dim];
        let src_right_len: usize = src_dims[dim + 1..].iter().product();

        let mut dst = vec![T::zero(); dst_len];
        for left_i in 0..dst_left_len {
            let start_src_idx = left_i * src_right_len * src_dim_len;
            let start_dst_idx = left_i * dst_right_len * dst_dim_len;
            for i in 0..dst_dim_len {
                let start_dst_idx = start_dst_idx + i * dst_right_len;
                for right_i in 0..dst_right_len {
                    let dst_idx = start_dst_idx + right_i;
                    let index = ids[dst_idx];
                    if index == I::max_value() {
                        dst[dst_idx] = T::zero();
                    } else {
                        let index = index.as_usize();
                        if index >= src_dim_len {
                            Err(Error::InvalidIndex {
                                index,
                                size: src_dim_len,
                                op: "gather",
                            }
                            .bt())?
                        }
                        let src_idx = start_src_idx + index * src_right_len + right_i;
                        dst[dst_idx] = src[src_idx]
                    }
                }
            }
        }
        Ok(dst)
    }
}

pub(super) struct IndexSelect<'a, T: IntDType> {
    pub(super) ids: &'a [T],
    pub(super) ids_l: &'a Layout,
    pub(super) dim: usize,
}

impl<I: IntDType> Map1 for IndexSelect<'_, I> {
    fn f<T: WithDType>(&self, src: &[T], layout: &Layout) -> Result<Vec<T>> {
        let src = match layout.contiguous_offsets() {
            Some((a, b)) => &src[a..b],
            None => Err(Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let dim = self.dim;
        let n_ids = match self.ids_l.dims() {
            [n_ids] => *n_ids,
            d => Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: d.len(),
                shape: self.ids_l.shape().clone(),
            }
            .bt())?,
        };
        let stride_ids = self.ids_l.stride()[0];
        let mut dst_dims = layout.dims().to_vec();
        let src_dim = dst_dims[dim];
        dst_dims[dim] = n_ids;
        let dst_len: usize = dst_dims.iter().product();
        let left_len: usize = dst_dims[..dim].iter().product();
        let right_len: usize = dst_dims[dim + 1..].iter().product();
        let mut dst = vec![T::zero(); dst_len];
        for left_i in 0..left_len {
            let start_src_idx = left_i * right_len * src_dim;
            let start_dst_idx = left_i * right_len * n_ids;
            for i in 0..n_ids {
                let start_dst_idx = start_dst_idx + i * right_len;
                let index = self.ids[self.ids_l.start_offset() + stride_ids * i];
                if index == I::max_value() {
                    dst[start_dst_idx..start_dst_idx + right_len].fill(T::zero());
                } else {
                    let index = index.as_usize();
                    if index >= src_dim {
                        Err(Error::InvalidIndex {
                            index,
                            size: src_dim,
                            op: "index-select",
                        }
                        .bt())?
                    }
                    let start_src_idx = start_src_idx + index * right_len;
                    dst[start_dst_idx..start_dst_idx + right_len]
                        .copy_from_slice(&src[start_src_idx..start_src_idx + right_len])
                }
            }
        }
        Ok(dst)
    }
}

pub(super) trait ElemUpdate {
    fn f<T: WithDType>(dst: &mut T, src: T);
}

pub(super) struct Set;

pub(super) struct Add;

impl ElemUpdate for Set {
    fn f<T: WithDType>(dst: &mut T, src: T) {
        *dst = src
    }
}

impl ElemUpdate for Add {
    fn f<T: WithDType>(dst: &mut T, src: T) {
        *dst += src
    }
}

pub(super) struct Scatter<'a, I: IntDType, M: ElemUpdate> {
    pub(super) ids: &'a [I],
    pub(super) ids_l: &'a Layout,
    pub(super) dim: usize,
    pub(super) _phantom: std::marker::PhantomData<M>,
}

impl<'a, I: IntDType, M: ElemUpdate> Scatter<'a, I, M> {
    pub(super) fn new(ids: &'a [I], ids_l: &'a Layout, dim: usize) -> Self {
        Self {
            ids,
            ids_l,
            dim,
            _phantom: Default::default(),
        }
    }
}

impl<I: IntDType, M: ElemUpdate> Map2InPlace for Scatter<'_, I, M> {
    const OP: &'static str = "scatter";
    fn f<T: WithDType>(
        &self,
        dst: &mut [T],
        dst_l: &Layout,
        src: &[T],
        src_l: &Layout,
    ) -> Result<()> {
        let dst = match dst_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "scatter" }.bt())?,
            Some((o1, o2)) => &mut dst[o1..o2],
        };

        let src = match src_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "scatter" }.bt())?,
            Some((o1, o2)) => &src[o1..o2],
        };

        let dim = self.dim;
        let ids_dims = self.ids_l.dims();
        let dst_dims = dst_l.dims();
        let dst_dim_len = dst_dims[dim];
        let dst_right_len: usize = dst_dims[dim + 1..].iter().product();

        let ids_left_len: usize = ids_dims[..dim].iter().product();
        let ids_dim_len = ids_dims[dim];
        let ids_right_len: usize = ids_dims[dim + 1..].iter().product();

        let ids = match self.ids_l.contiguous_offsets() {
            Some((a, b)) => &self.ids[a..b],
            None => Err(Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        for left_i in 0..ids_left_len {
            let start_ids_idx = left_i * ids_right_len * ids_dim_len;
            let start_dst_idx = left_i * dst_right_len * dst_dim_len;
            for i in 0..ids_dim_len {
                let start_ids_idx = start_ids_idx + i * ids_right_len;
                for right_i in 0..dst_right_len {
                    let ids_idx = start_ids_idx + right_i;
                    let index = ids[ids_idx];
                    if index == I::max_value() {
                        continue;
                    }
                    let index = index.as_usize();
                    if index >= dst_dim_len {
                        Err(Error::InvalidIndex {
                            index,
                            size: dst_dim_len,
                            op: "gather",
                        }
                        .bt())?
                    }
                    let dst_idx = start_dst_idx + index * dst_right_len + right_i;
                    M::f(&mut dst[dst_idx], src[ids_idx])
                }
            }
        }

        Ok(())
    }
}

pub(super) struct IndexAdd<'a, I: IntDType> {
    pub(super) ids: &'a [I],
    pub(super) dim: usize,
}

impl<I: IntDType> Map2 for IndexAdd<'_, I> {
    const OP: &'static str = "index-add";
    // https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_
    // v1, l1 -> self
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, src: &[T], src_l: &Layout) -> Result<Vec<T>> {
        let dst_len = l1.shape().elem_count();
        let mut dst = vec![T::zero(); dst_len];
        copy_strided_src_(v1, &mut dst, 0, l1);
        let src = match src_l.contiguous_offsets() {
            None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
            Some((o1, o2)) => &src[o1..o2],
        };
        let dim = self.dim;
        let max_idx = l1.dims()[dim];
        let pre_dim = src_l.dims()[..dim].iter().product::<usize>();
        let src_dim_sz = src_l.dims()[dim];
        let post_dim = src_l.dims()[dim + 1..].iter().product::<usize>();
        if dim == 0 {
            for (src_idx, dst_idx) in self.ids.iter().enumerate() {
                if *dst_idx == I::max_value() {
                    continue;
                }
                let dst_idx = dst_idx.as_usize();
                if dst_idx >= max_idx {
                    Err(Error::InvalidIndex {
                        index: dst_idx,
                        op: "index-add",
                        size: max_idx,
                    })?
                }
                let src_idx = src_idx * post_dim;
                let dst_idx = dst_idx * post_dim;
                let src = &src[src_idx..src_idx + post_dim];
                let dst = &mut dst[dst_idx..dst_idx + post_dim];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s
                }
            }
        } else {
            for (src_idx, dst_idx) in self.ids.iter().enumerate() {
                if *dst_idx == I::max_value() {
                    continue;
                }
                let dst_idx = dst_idx.as_usize();
                if dst_idx >= max_idx {
                    Err(Error::InvalidIndex {
                        index: dst_idx,
                        op: "index-add",
                        size: max_idx,
                    })?
                }
                for pre_i in 0..pre_dim {
                    let pre_src_i = (pre_i * src_dim_sz + src_idx) * post_dim;
                    let pre_dst_i = (pre_i * max_idx + dst_idx) * post_dim;
                    let src = &src[pre_src_i..pre_src_i + post_dim];
                    let dst = &mut dst[pre_dst_i..pre_dst_i + post_dim];
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d += s
                    }
                }
            }
        }
        Ok(dst)
    }
}
