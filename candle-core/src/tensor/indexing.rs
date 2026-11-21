//! Selects, gathers and scatters elements by index, and walks a tensor's
//! storage positions.
//!
//! `embedding` and `index_select` pick rows by id, `gather` picks elements
//! along a dimension, and the scatter and `index_add` family writes values
//! back by index. `get` and `get_on_dim` take one index as a sub-tensor.
//! `strided_index` and `strided_blocks` visit the storage positions of a
//! tensor in lexicographic order, the latter reporting each contiguous block.

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
    /// Returns a tensor with the values from the `self` tensor at the index corresponding to the
    /// values hold in the `ids` tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor with dimensions `v, h`.
    /// * `ids` - A tensor with dimensions `s` and with integer values between 0 and v (exclusive).
    ///
    /// The resulting tensor has dimensions `s, h`. `s` is called the sequence length, `v` the
    /// vocabulary size, and `h` the hidden size.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let values = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let ids = Tensor::new(&[2u32, 1u32, 2u32], &Device::Cpu)?;
    /// let emb = values.embedding(&ids)?;
    /// assert_eq!(emb.to_vec2::<f32>()?, &[[4., 5.], [2., 3.], [4., 5.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn embedding(&self, ids: &Self) -> Result<Self> {
        if self.rank() != 2 || ids.rank() != 1 {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: ids.shape().clone(),
                op: "embedding",
            }
            .bt())?
        }
        self.index_select(ids, 0)
    }

    pub(super) fn scatter_checks(&self, indexes: &Self, source: &Self, dim: usize) -> Result<()> {
        let source_dims = source.dims();
        let self_dims = self.dims();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter (self, src)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        if indexes.dims() != source.dims() {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter (indexes, src)",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        Ok(())
    }

    pub fn scatter<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "scatter")?;
        self.scatter_checks(indexes, source, dim)?;
        let shape = self.shape();
        let mut storage = unsafe {
            self.device()
                .alloc_uninit_from(shape, self.dtype(), self.wave_ticket())?
        };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        let layout = Layout::contiguous(shape);
        storage.scatter_set(
            &layout,
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::Scatter(t1, t2, t3, dim)
        });
        Ok(from_storage(storage, self.shape(), op, false))
    }

    pub fn scatter_set<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<()> {
        if self.same_storage(source) {
            crate::bail!("cannot use slice_set when self and src share their storage")
        }
        let dim = dim.to_index(self.shape(), "scatter-set")?;
        self.scatter_checks(indexes, source, dim)?;
        self.storage_mut().scatter_set(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        Ok(())
    }

    pub fn scatter_add<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "scatter-add")?;
        self.scatter_checks(indexes, source, dim)?;
        let shape = self.shape();
        let mut storage = unsafe {
            self.device()
                .alloc_uninit_from(shape, self.dtype(), self.wave_ticket())?
        };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        let layout = Layout::contiguous(shape);
        storage.scatter_add(
            &layout,
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::ScatterAdd(t1, t2, t3, dim)
        });
        Ok(from_storage(storage, self.shape(), op, false))
    }

    pub fn scatter_add_set<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<()> {
        if self.same_storage(source) {
            crate::bail!("cannot use slice_set when self and src share their storage")
        }
        let dim = dim.to_index(self.shape(), "scatter-add-set")?;
        self.scatter_checks(indexes, source, dim)?;
        self.storage_mut().scatter_add(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        Ok(())
    }

    /// Embeds the values of the `src` tensor into the `self` tensor on the specified dimension.
    pub fn slice_scatter<D: Dim>(&self, src: &Self, dim: D, start: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "slice-scatter")?;
        if dim == 0 {
            self.slice_scatter0(src, start)
        } else {
            // Non-zero `dim`: reuse the dim-0 path by transposing `dim` to the front,
            // scattering, then transposing back.
            self.transpose(0, dim)?
                .slice_scatter0(&src.transpose(0, dim)?, start)?
                .transpose(0, dim)
        }
    }

    /// Embeds the values of the `src` tensor into the `self` tensor on the first dimension.
    pub fn slice_scatter0(&self, src: &Self, start: usize) -> Result<Self> {
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-scatter",
            }
            .bt())?
        }
        if self.device().location() != src.device.location() {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-scatter",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: src.shape().clone(),
            }
            .bt())?
        }
        let shape_ok =
            self.dims()
                .iter()
                .zip(src.dims().iter())
                .enumerate()
                .all(|(dim_idx, (&d1, &d2))| {
                    if 0 == dim_idx {
                        d2 + start <= d1
                    } else {
                        d1 == d2
                    }
                });
        if !shape_ok {
            Err(Error::ShapeMismatchBinaryOp {
                op: "slice-scatter (self, src)",
                lhs: self.shape().clone(),
                rhs: src.shape().clone(),
            }
            .bt())?
        }
        let mut storage = unsafe {
            self.device()
                .alloc_uninit_from(self.shape(), self.dtype(), self.wave_ticket())?
        };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        let offset = start * src.dims()[1..].iter().product::<usize>();
        src.storage()
            .copy_strided_src(&mut storage, offset, src.layout())?;
        let op = BackpropOp::new2(self, src, |t1, t2| Op::SliceScatter0(t1, t2, start));
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Accumulate element from `source` at indexes `indexes` and add them to `self`.
    pub fn index_add<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-add")?;
        let source_dims = source.dims();
        let self_dims = self.dims();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "index-add (self, source)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        // The number of element in indexes must match the dimension on which the add is
        // performed on the source tensor (and the index values from `indexes` are taken from
        // the target tensor self)
        let indexes_len = indexes.dims1()?;
        if source_dims[dim] != indexes_len {
            Err(Error::ShapeMismatchBinaryOp {
                op: "index-add (ids, source))",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        let storage = self.storage().index_add(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::IndexAdd(t1, t2, t3, dim)
        });
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Gather values across the target dimension.
    ///
    /// # Arguments
    ///
    /// * `self` - The input tensor.
    /// * `indexes` - The indices of elements to gather, this should have same number of dimensions as `self`
    ///   and indexes.dims()[d] <= self.dims()[d] for all dimensions d != dim
    /// * `dim` - the target dimension.
    ///
    /// The resulting tensor has the same shape as `indexes` and use values from `self` indexed on
    /// dimension `dim` by the values in `indexes`.
    pub fn gather<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "gather")?;

        let self_dims = self.dims();
        let indexes_dims = indexes.dims();
        let mismatch = if indexes_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(indexes_dims.iter()).enumerate() {
                if i != dim && d1 < d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "gather",
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
            }
            .bt())?
        }
        let storage =
            self.storage()
                .gather(self.layout(), &indexes.storage(), indexes.layout(), dim)?;
        let op = BackpropOp::new2(self, indexes, |t1, t2| Op::Gather(t1, t2, dim));
        Ok(from_storage(storage, indexes.shape(), op, false))
    }

    /// Select values for the input tensor at the target indexes across the specified dimension.
    ///
    /// The `indexes` is argument is an int tensor with a single dimension.
    /// The output has the same number of dimension as the `self` input. The target dimension of
    /// the output has length the length of `indexes` and the values are taken from `self` using
    /// the index from `indexes`. Other dimensions have the same number of elements as the input
    /// tensor.
    pub fn index_select<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-select")?;
        let indexes_len = match indexes.dims() {
            [l] => *l,
            _ => Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
                op: "index-select",
            }
            .bt())?,
        };
        let storage = self.storage().index_select(
            &indexes.storage(),
            self.layout(),
            indexes.layout(),
            dim,
        )?;
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes_len;
        let op = BackpropOp::new2(self, indexes, |t1, t2| Op::IndexSelect(t1, t2, dim));
        Ok(from_storage(storage, dims, op, false))
    }

    /// Returns an iterator over position of the elements in the storage when ranging over the
    /// index tuples in lexicographic order.
    pub fn strided_index(&self) -> crate::StridedIndex<'_> {
        self.layout.strided_index()
    }

    /// Similar to `strided_index` but returns the position of the start of each contiguous block
    /// as well as the length of the contiguous blocks. For a contiguous tensor, the index iterator
    /// will only return the start offset and the size would be the number of elements in the
    /// tensor.
    pub fn strided_blocks(&self) -> crate::StridedBlocks<'_> {
        self.layout.strided_blocks()
    }

    /// Returns the sub-tensor fixing the index at `i` on the first dimension.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let t = tensor.get(0)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[0., 1.]);
    /// let t = tensor.get(1)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[2., 3.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn get(&self, i: usize) -> Result<Self> {
        let dims = self.dims();
        if dims.is_empty() {
            Ok(self.clone())
        } else {
            self.narrow(0, i, 1)?.reshape(&dims[1..])
        }
    }

    /// Returns the sub-tensor fixing the index at `index` on the dimension `dim`.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let t = tensor.get_on_dim(1, 0)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[0., 2., 4.]);
    /// let t = tensor.get_on_dim(1, 1)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[1., 3., 5.]);
    /// let t = tensor.get_on_dim(0, 1)?;
    /// assert_eq!(t.to_vec1::<f32>()?, &[2., 3.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn get_on_dim<D: Dim>(&self, dim: D, index: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "get_on_dim")?;
        self.narrow(dim, index, 1)?.squeeze(dim)
    }

    /// Returns a copy of `self` where the values within `ranges` have been replaced with the
    /// content of `src`.
    pub fn slice_assign<D: std::ops::RangeBounds<usize>>(
        &self,
        ranges: &[D],
        src: &Tensor,
    ) -> Result<Self> {
        let src_dims = src.dims();
        let self_dims = self.dims();
        if self_dims.len() != src_dims.len() {
            bail!(
                "slice-assign requires input with the same rank {} <> {}",
                self_dims.len(),
                src_dims.len()
            )
        }
        if self_dims.len() != ranges.len() {
            bail!(
                "slice-assign requires input with the same rank as there are ranges {} <> {}",
                self_dims.len(),
                ranges.len()
            )
        }
        let mut src = src.clone();
        let mut mask = Self::ones(src.shape(), DType::U8, src.device())?;
        for (i, range) in ranges.iter().enumerate() {
            let start_included = match range.start_bound() {
                std::ops::Bound::Unbounded => 0,
                std::ops::Bound::Included(v) => *v,
                std::ops::Bound::Excluded(v) => *v + 1,
            };
            let end_excluded = match range.end_bound() {
                std::ops::Bound::Unbounded => self_dims[i],
                std::ops::Bound::Included(v) => *v + 1,
                std::ops::Bound::Excluded(v) => *v,
            };
            if end_excluded <= start_included {
                bail!("slice-assign: empty range for dim {i}, {start_included} {end_excluded}")
            }
            if self_dims[i] < end_excluded {
                bail!(
                    "slice-assign: upper bound is out of range for dim {i}, {end_excluded} {}",
                    self_dims[i]
                )
            }
            if end_excluded - start_included != src_dims[i] {
                bail!(
                    "slice-assign: the range for dim {i} ({start_included}..{end_excluded}) does not match the size of src {}", src_dims[i]
                )
            }
            src = src.pad_with_zeros(i, start_included, self_dims[i] - end_excluded)?;
            mask = mask.pad_with_zeros(i, start_included, self_dims[i] - end_excluded)?
        }
        mask.where_cond(/* on_true= */ &src, /* on_false= */ self)
    }

    /// Returns a new tensor with the order of elements reversed along the specified dimensions.
    /// This function makes a copy of the tensor’s data.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, Device};
    /// let t = Tensor::arange(0., 6., &Device::Cpu)?.reshape((2, 3))?;
    /// assert_eq!(t.to_vec2::<f64>()?, &[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// let t_flipped = t.flip(&[0])?;
    /// assert_eq!(t_flipped.to_vec2::<f64>()?, &[[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn flip(&self, dims: &[usize]) -> Result<Self> {
        let mut result = self.clone();
        for &dim in dims.iter() {
            let size = result.dim(dim)?;
            let indices: Vec<i64> = (0..size).rev().map(|x| x as i64).collect();
            let indices_tensor = Tensor::from_vec(indices, (size,), result.device())?;
            result = result.index_select(&indices_tensor, dim)?;
        }
        Ok(result)
    }
}
