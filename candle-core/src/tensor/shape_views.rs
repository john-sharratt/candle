//! Changes a tensor's shape, axes and layout without changing its values.
//!
//! `narrow` and `chunk` cut ranges from a dimension, and `reshape`, `squeeze`,
//! `unsqueeze`, `permute`, `t` and `transpose` rearrange its axes. `roll` shifts
//! values around a dimension, and `repeat`, `broadcast_as` and `meshgrid` extend
//! it. `contiguous` returns the tensor itself when it is already contiguous and
//! copies it otherwise, while `force_contiguous` always copies. `pad_with_zeros`
//! and `pad_with_same` add elements at the edges of a dimension.

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
    /// Repeat this tensor along the specified dimensions.
    pub fn repeat<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        // Similar to PyTorch, we extend the number of dimensions of self if needed.
        let repeats = shape.into();
        let repeats = repeats.dims();
        let mut inp = if self.rank() < repeats.len() {
            let shape = [vec![1; repeats.len() - self.rank()], self.dims().to_vec()].concat();
            self.reshape(shape)?
        } else {
            self.clone()
        };
        for (idx, &repeat) in repeats.iter().enumerate() {
            if repeat > 1 {
                inp = Self::cat(&vec![&inp; repeat], idx)?
            }
        }
        Ok(inp)
    }

    /// Creates grids of coordinates specified by the 1D inputs.
    ///
    /// # Arguments
    ///
    /// * `args` - A slice of 1D tensors.
    /// * `xy_indexing` - Whether to use xy indexing or ij indexing. If xy is selected, the
    ///   first dimension corresponds to the cardinality of the second input and the second
    ///   dimension corresponds to the cardinality of the first input. If ij is selected, the
    ///   dimensions are in the same order as the cardinality of the inputs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device, Shape};
    /// let x = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    /// let y = Tensor::new(&[4f32, 5., 6.], &Device::Cpu)?;
    ///
    /// let grids_xy = Tensor::meshgrid(&[&x, &y], true)?;
    ///
    /// assert_eq!(grids_xy.len(), 2);
    /// assert_eq!(grids_xy[0].dims(), &[3, 3]);
    ///
    /// assert_eq!(grids_xy[0].to_vec2::<f32>()?, &[[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
    /// assert_eq!(grids_xy[1].to_vec2::<f32>()?, &[[4., 4., 4.], [5., 5., 5.], [6., 6., 6.]]);
    ///
    /// let grids_ij = Tensor::meshgrid(&[&x, &y], false)?;
    ///
    /// assert_eq!(grids_ij[0].to_vec2::<f32>()?, &[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
    /// assert_eq!(grids_ij[1].to_vec2::<f32>()?, &[[4., 5., 6.], [4., 5., 6.], [4., 5., 6.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// * Will return `Err` if `args` contains less than 2 tensors.
    ///
    pub fn meshgrid<A: AsRef<LiveTensor<'w>>>(args: &[A], xy_indexing: bool) -> Result<Vec<Self>> {
        if args.len() <= 1 {
            Err(Error::OpRequiresAtLeastTwoTensors { op: "meshgrid" }.bt())?
        }
        let args: Vec<_> = if xy_indexing {
            args.iter().rev().collect()
        } else {
            args.iter().collect()
        };

        let mut shape = Vec::with_capacity(args.len());
        for arg in args.iter() {
            shape.push(arg.as_ref().dims1()?)
        }

        let mut grids = Vec::with_capacity(args.len());
        for idx in 0..args.len() {
            let mut ones = vec![1usize; args.len()];
            ones[idx] = shape[idx];
            let arg = args[idx].as_ref().reshape(ones)?;
            let mut repeats = shape.clone();
            repeats[idx] = 1;
            let repeated_tensor = arg.repeat(repeats)?;
            grids.push(repeated_tensor);
        }
        if xy_indexing {
            grids.reverse();
        }
        Ok(grids)
    }

    pub(crate) fn check_dim(&self, dim: usize, op: &'static str) -> Result<()> {
        if dim >= self.dims().len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(())
        }
    }

    /// Split a tensor into the specified number of chunks, this may return less chunks than
    /// specified.
    pub fn chunk<D: Dim>(&self, chunks: usize, dim: D) -> Result<Vec<Self>> {
        let dim = dim.to_index(self.shape(), "chunk")?;
        let size = self.dim(dim)?;
        if size < chunks {
            (0..size).map(|i| self.narrow(dim, i, 1)).collect()
        } else {
            let chunk_size = size / chunks;
            let cnt_additional = size % chunks;
            let mut tensors = vec![];
            let mut sum_chunk_size = 0;
            for i in 0..chunks {
                let chunk_size = if i < cnt_additional {
                    chunk_size + 1
                } else {
                    chunk_size
                };
                let tensor = self.narrow(dim, sum_chunk_size, chunk_size)?;
                tensors.push(tensor);
                sum_chunk_size += chunk_size
            }
            Ok(tensors)
        }
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// ```
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::new(&[
    ///     [0f32, 1., 2.],
    ///     [3.  , 4., 5.],
    ///     [6.  , 7., 8.]
    /// ], &Device::Cpu)?;
    ///
    /// let b = a.narrow(0, 1, 2)?;
    /// assert_eq!(b.shape().dims(), &[2, 3]);
    /// assert_eq!(b.to_vec2::<f32>()?, &[
    ///     [3., 4., 5.],
    ///     [6., 7., 8.]
    /// ]);
    ///
    /// let c = a.narrow(1, 1, 1)?;
    /// assert_eq!(c.shape().dims(), &[3, 1]);
    /// assert_eq!(c.to_vec2::<f32>()?, &[
    ///     [1.],
    ///     [4.],
    ///     [7.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        let err = |msg| {
            Err::<(), _>(
                Error::NarrowInvalidArgs {
                    shape: self.shape().clone(),
                    dim,
                    start,
                    len,
                    msg,
                }
                .bt(),
            )
        };
        if start > dims[dim] {
            err("start > dim_len")?
        }
        if start.saturating_add(len) > dims[dim] {
            err("start + len > dim_len")?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let op = BackpropOp::new1(self, |t| Op::Narrow(t, dim, start, len));
            let layout = self.layout().narrow(dim, start, len)?;
            let tensor_ = TensorInner {
                lease: PhantomData,
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout,
                op,
                is_variable: false,
                dtype: self.dtype,
                device: self.device.clone(),
            };
            Ok(LiveTensor(Arc::new(tensor_)))
        }
    }

    pub(super) fn squeeze_dims(self, dims: &[usize]) -> Result<Self> {
        match dims {
            [] => Ok(self),
            [i] => self.squeeze(*i),
            dims => {
                let dims = self
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(dim_idx, &v)| {
                        if dims.contains(&dim_idx) {
                            None
                        } else {
                            Some(v)
                        }
                    })
                    .collect::<Vec<_>>();
                self.reshape(dims)
            }
        }
    }

    /// Roll the tensor input along the given dimension.
    /// Elements that are shifted beyond the last position are re-introduced at the first position.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.roll(1, 0)?;
    /// assert_eq!(tensor.to_vec2::<f32>()?, &[[4., 5.], [0., 1.], [2., 3.]]);
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.roll(-1, 0)?;
    /// assert_eq!(tensor.to_vec2::<f32>()?, &[[2., 3.], [4., 5.], [0., 1.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn roll<D>(&self, shift: i32, dim: D) -> Result<Self>
    where
        D: Dim + Clone,
    {
        let dim = dim.to_index(self.shape(), "roll")?;
        let dim_size = self.dim(dim)?;
        let shift = shift.rem_euclid(dim_size as i32) as usize;
        if shift == 0 {
            Ok(self.clone())
        } else {
            let a = self.narrow(dim, 0, dim_size - shift)?;
            let b = self.narrow(dim, dim_size - shift, shift)?;
            Self::cat(&[&b, &a], dim)
        }
    }

    pub(super) fn flatten_<D1: Dim, D2: Dim>(
        &self,
        start_dim: Option<D1>,
        end_dim: Option<D2>,
    ) -> Result<Self> {
        if self.rank() == 0 {
            self.reshape(1)
        } else {
            let start_dim = match start_dim {
                None => 0,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            let end_dim = match end_dim {
                None => self.rank() - 1,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            if start_dim < end_dim {
                let dims = self.dims();
                let mut dst_dims = dims[..start_dim].to_vec();
                dst_dims.push(dims[start_dim..end_dim + 1].iter().product::<usize>());
                if end_dim + 1 < dims.len() {
                    dst_dims.extend(&dims[end_dim + 1..]);
                }
                self.reshape(dst_dims)
            } else {
                Ok(self.clone())
            }
        }
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` to `end_dim` (both
    /// inclusive).
    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> Result<Self> {
        self.flatten_(Some(start_dim), Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `0` to `end_dim` (inclusive).
    pub fn flatten_to<D: Dim>(&self, end_dim: D) -> Result<Self> {
        self.flatten_(None::<usize>, Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` (inclusive) to the last
    /// dimension.
    pub fn flatten_from<D: Dim>(&self, start_dim: D) -> Result<Self> {
        self.flatten_(Some(start_dim), None::<usize>)
    }

    /// Flattens the input tensor by reshaping it into a one dimension tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.flatten_all()?;
    /// assert_eq!(tensor.to_vec1::<f32>()?, &[0., 1., 2., 3., 4., 5.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn flatten_all(&self) -> Result<Self> {
        self.flatten_(None::<usize>, None::<usize>)
    }

    /// Returns a tensor that is a transposed version of the input, the two last dimensions of the
    /// input are swapped.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.t()?;
    /// assert_eq!(tensor.to_vec2::<f32>()?, &[[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn t(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are
    /// swapped.
    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let op = BackpropOp::new1(self, |t| Op::Transpose(t, dim1, dim2));
        let tensor_ = TensorInner {
            lease: PhantomData,
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.transpose(dim1, dim2)?,
            op,
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(LiveTensor(Arc::new(tensor_)))
    }

    /// Returns a tensor with the same data as the input where the dimensions have been permuted.
    /// dims must be a permutation, i.e. include each dimension index exactly once.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::arange(0u32, 120u32, &Device::Cpu)?.reshape((2, 3, 4, 5))?;
    /// assert_eq!(tensor.dims(), &[2, 3, 4, 5]);
    /// let tensor = tensor.permute((2, 3, 1, 0))?;
    /// assert_eq!(tensor.dims(), &[4, 5, 3, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn permute<D: Dims>(&self, dims: D) -> Result<Self> {
        let dims = dims.to_indexes(self.shape(), "permute")?;
        // O(n^2) permutation check but these arrays are small.
        let is_permutation =
            dims.len() == self.rank() && (0..dims.len()).all(|i| dims.contains(&i));
        if !is_permutation {
            bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                dims
            )
        }
        let op = BackpropOp::new1(self, |t| Op::Permute(t, dims.clone()));
        let tensor_ = TensorInner {
            lease: PhantomData,
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.permute(&dims)?,
            op,
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(LiveTensor(Arc::new(tensor_)))
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns true if the data is stored in a Fortran contiguous (aka column major) way.
    pub fn is_fortran_contiguous(&self) -> bool {
        self.layout.is_fortran_contiguous()
    }

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left<S: Into<Shape>>(&self, left_shape: S) -> Result<Self> {
        let left_shape = left_shape.into();
        let mut dims = left_shape.into_dims();
        dims.extend(self.dims());
        self.broadcast_as(dims)
    }

    /// Broadcast the input tensor to the target shape. This returns an error if the input shape is
    /// not compatible with the target shape.
    ///
    /// If the input shape is `i_1, i_2, ... i_k`, the target shape has to have `k` dimensions or
    /// more and shape `j_1, ..., j_l, t_1, t_2, ..., t_k`. The dimensions `j_1` to `j_l` can have
    /// any value, the dimension `t_a` must be equal to `i_a` if `i_a` is different from 1. If
    /// `i_a` is equal to 1, any value can be used.
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let tensor_ = TensorInner {
            lease: PhantomData,
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape)?,
            op: BackpropOp::new1(self, Op::Broadcast),
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(LiveTensor(Arc::new(tensor_)))
    }

    /// An alias for broadcast_as.
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    /// Returns a tensor that is in row major order. This is the same as the original tensor if it
    /// was already contiguous, otherwise a copy is triggered.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = unsafe {
                self.device()
                    .alloc_uninit_from(shape, self.dtype(), self.wave_ticket())?
            };
            self.storage()
                .copy_strided_src(&mut storage, 0, self.layout())?;
            let op = BackpropOp::new1(self, Op::Copy);
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    }

    /// Returns a tensor that is in row major order. This always makes a copy.
    pub fn force_contiguous(&self) -> Result<Self> {
        let shape = self.shape();
        let mut storage = unsafe {
            self.device()
                .alloc_uninit_from(shape, self.dtype(), self.wave_ticket())?
        };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        let op = BackpropOp::new1(self, Op::Copy);
        Ok(from_storage(storage, shape.clone(), op, false))
    }

    /// Reshape returns a tensor with the target shape provided that the number of elements of the
    /// original tensor is the same.
    /// If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
    /// a new storage and copies the data over, the returned tensor is always contiguous.
    ///
    /// The shape can be specified using a tuple of `usize` and at most one `()` in which case
    /// the behavior is the same as when using `-1` in PyTorch: this dimension size is adjusted so
    /// as to match the number of elements in the tensor.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.reshape((1, 6))?;
    /// assert_eq!(c.shape().dims(), &[1, 6]);
    ///
    /// let c = a.reshape((3, 2))?;
    /// assert_eq!(c.shape().dims(), &[3, 2]);
    ///
    /// let c = a.reshape((2, (), 1))?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 1]);
    ///
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn reshape<S: ShapeWithOneHole>(&self, s: S) -> Result<Self> {
        let shape = s.into_shape(self.elem_count())?;
        if shape.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            }
            .bt());
        }
        let op = BackpropOp::new1(self, Op::Reshape);
        if self.is_contiguous() {
            let tensor_ = TensorInner {
                lease: PhantomData,
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                op,
                is_variable: false,
                dtype: self.dtype,
                device: self.device.clone(),
            };
            Ok(LiveTensor(Arc::new(tensor_)))
        } else {
            let mut storage = unsafe {
                self.device()
                    .alloc_uninit_from(&shape, self.dtype(), self.wave_ticket())?
            };
            self.storage()
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(storage, shape, op, false))
        }
    }

    /// Creates a new tensor with the specified dimension removed if its size was one.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3, 1), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.squeeze(2)?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    ///
    /// let c = a.squeeze(D::Minus1)?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn squeeze<D: Dim>(&self, dim: D) -> Result<Self> {
        // The PyTorch semantics are to return the same tensor if the target dimension
        // does not have a size of 1.
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "squeeze")?;
        if dims[dim] == 1 {
            let mut dims = dims.to_vec();
            let mut strides = self.stride().to_vec();
            dims.remove(dim);
            strides.remove(dim);
            let tensor_ = TensorInner {
                lease: PhantomData,
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout: Layout::new(dims.into(), strides, self.layout.start_offset()),
                op: BackpropOp::new1(self, Op::Reshape),
                is_variable: false,
                dtype: self.dtype,
                device: self.device.clone(),
            };
            Ok(LiveTensor(Arc::new(tensor_)))
        } else {
            Ok(self.clone())
        }
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device, D};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.unsqueeze(0)?;
    /// assert_eq!(c.shape().dims(), &[1, 2, 3]);
    ///
    /// let c = a.unsqueeze(D::Minus1)?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 1]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        let mut strides = self.stride().to_vec();
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        // Cannot panic because to_index_plus_one already checks dimensions
        dims.insert(dim, 1);
        // Any stride would work here, but we pick one so as to maximize the probability to remain
        // C contiguous.
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        strides.insert(dim, stride);
        let tensor_ = TensorInner {
            lease: PhantomData,
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: Layout::new(dims.into(), strides, self.layout.start_offset()),
            op: BackpropOp::new1(self, Op::Reshape),
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(LiveTensor(Arc::new(tensor_)))
    }

    /// Pad the input tensor using 0s along dimension `dim`. This adds `left` elements before the
    /// input tensor values and `right` elements after.
    pub fn pad_with_zeros<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        if left == 0 && right == 0 {
            Ok(self.clone())
        } else if left == 0 {
            let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
            let mut dims = self.dims().to_vec();
            dims[dim] = right;
            let right = Tensor::zeros(dims.as_slice(), self.dtype, self.device())?;
            Self::cat(&[self, &right], dim)
        } else if right == 0 {
            let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
            let mut dims = self.dims().to_vec();
            dims[dim] = left;
            let left = Tensor::zeros(dims.as_slice(), self.dtype, self.device())?;
            Self::cat(&[&left, self], dim)
        } else {
            let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
            let mut dims = self.dims().to_vec();
            dims[dim] = left;
            let left = Tensor::zeros(dims.as_slice(), self.dtype, self.device())?;
            dims[dim] = right;
            let right = Tensor::zeros(dims.as_slice(), self.dtype, self.device())?;
            Self::cat(&[&left, self, &right], dim)
        }
    }

    /// Pad the input tensor using same values along dimension `dim`. This adds `left` elements before the
    /// input tensor values and `right` elements after.
    pub fn pad_with_same<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        if left == 0 && right == 0 {
            Ok(self.clone())
        } else if self.elem_count() == 0 {
            bail!("cannot use pad_with_same on an empty tensor")
        } else if left == 0 {
            let dim = dim.to_index(self.shape(), "pad_with_same")?;
            let r = self.narrow(dim, self.dim(dim)? - 1, 1)?;
            let mut v = vec![self];
            for _ in 0..right {
                v.push(&r)
            }
            Self::cat(&v, dim)
        } else if right == 0 {
            let dim = dim.to_index(self.shape(), "pad_with_same")?;
            let l = self.narrow(dim, 0, 1)?;
            let mut v = vec![];
            for _ in 0..left {
                v.push(&l)
            }
            v.push(self);
            Self::cat(&v, dim)
        } else {
            let dim = dim.to_index(self.shape(), "pad_with_same")?;
            let l = self.narrow(dim, 0, 1)?;
            let r = self.narrow(dim, self.dim(dim)? - 1, 1)?;
            let mut v = vec![];
            for _ in 0..left {
                v.push(&l)
            }
            v.push(self);
            for _ in 0..right {
                v.push(&r)
            }
            Self::cat(&v, dim)
        }
    }
}
