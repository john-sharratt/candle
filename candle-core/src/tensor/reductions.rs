//! Reductions over a dimension or over every element of a tensor.
//!
//! `sum`, `mean`, `var`, `max`, `min`, `argmax` and `argmin` each come in a
//! `_keepdim` form, which keeps the reduced dimension at size one, and a
//! squeezing form. The `_all` variants reduce every element to a scalar tensor.
//! `cumsum` accumulates along a dimension, and `log_sum_exp` computes the log
//! of a sum of exponentials.

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
    pub(super) fn reduce_impl<D: Dim>(&self, dim: D, keepdim: bool, op: ReduceOp) -> Result<Self> {
        let dim = dim.to_index(self.shape(), op.name())?;
        let storage = self.storage().reduce_op(op, self.layout(), &[dim])?;
        let mut dims = self.dims().to_vec();
        dims[dim] = 1;
        let op = match op {
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max => {
                BackpropOp::new1(self, |arg| Op::Reduce(arg, op, dims.to_vec()))
            }
            ReduceOp::ArgMin | ReduceOp::ArgMax => BackpropOp::none(),
        };
        let res = from_storage(storage, dims, op, false);
        if keepdim {
            Ok(res)
        } else {
            res.squeeze_dims(&[dim])
        }
    }

    pub(super) fn sum_impl<D: Dims>(&self, sum_dims: D, keepdim: bool) -> Result<Self> {
        let sum_dims = sum_dims.to_indexes(self.shape(), "sum")?;
        let storage = self
            .storage()
            .reduce_op(ReduceOp::Sum, self.layout(), &sum_dims)?;
        let mut dims = self.dims().to_vec();
        for &sum_dim in sum_dims.iter() {
            dims[sum_dim] = 1
        }
        let op = BackpropOp::new1(self, |a| Op::Reduce(a, ReduceOp::Sum, dims.to_vec()));
        let sum = from_storage(storage, dims, op, false);
        if keepdim {
            Ok(sum)
        } else {
            sum.squeeze_dims(&sum_dims)
        }
    }

    /// Returns the sum of all elements in the input tensor. The sum is performed over all the
    /// input dimensions.
    ///
    /// The resulting tensor has a shape that is similar to the shape of the input tensor, except
    /// that the number of elements for each dimension index in `sum_dims` is 1.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let s = a.sum_keepdim(0)?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[2., 4.]]);
    /// let s = a.sum_keepdim(1)?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[1.], [5.]]);
    /// let s = a.sum_keepdim((0, 1))?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[6.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sum_keepdim<D: Dims>(&self, sum_dims: D) -> Result<Self> {
        self.sum_impl(sum_dims, true)
    }

    /// Returns the sum of all elements in the input tensor. The sum is performed over all the
    /// input dimensions and compared to `sum_keepdim` these dimensions are squeezed rather than
    /// kept.
    pub fn sum<D: Dims>(&self, sum_dims: D) -> Result<Self> {
        self.sum_impl(sum_dims, false)
    }

    /// Returns the mean of all elements in the input tensor. The mean is performed over all the
    /// input dimensions.
    ///
    /// The resulting tensor has a shape that is similar to the shape of the input tensor, except
    /// that the number of elements for each dimension index in `mean_dims` is 1.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let s = a.mean_keepdim(0)?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[1., 2.]]);
    /// let s = a.mean_keepdim(1)?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[0.5], [2.5]]);
    /// let s = a.mean_keepdim((0, 1))?;
    /// assert_eq!(s.to_vec2::<f32>()?, &[[1.5]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn mean_keepdim<D: Dims>(&self, mean_dims: D) -> Result<Self> {
        let mean_dims = mean_dims.to_indexes(self.shape(), "mean-keepdim")?;
        let reduced_dim: usize = mean_dims.iter().map(|i| self.dims()[*i]).product();
        let scale = 1f64 / (reduced_dim as f64);
        self.sum_impl(mean_dims, true)? * scale
    }

    /// Returns the mean of all elements in the input tensor. The mean is performed over all the
    /// input dimensions and compared to `mean_keepdim` these dimensions are squeezed rather than
    /// kept.
    pub fn mean<D: Dims>(&self, mean_dims: D) -> Result<Self> {
        let mean_dims = mean_dims.to_indexes(self.shape(), "mean")?;
        let reduced_dim: usize = mean_dims.iter().map(|i| self.dims()[*i]).product();
        let scale = 1f64 / (reduced_dim as f64);
        self.sum_impl(mean_dims, false)? * scale
    }

    /// Returns the unbiased variance over the selected dimension.
    pub fn var_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "var")?;
        let mean = self.mean_keepdim(dim)?;
        let squares = self.broadcast_sub(&mean)?.sqr()?;
        squares.sum_impl(dim, true)? / (self.dim(dim)? - 1) as f64
    }

    /// Returns the unbiased variance over the selected dimension.
    pub fn var<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "var")?;
        self.var_keepdim(dim)?.squeeze(dim)
    }

    /// Gathers the maximum value across the selected dimension. The resulting shape has the same
    /// number of dimensions as the original tensor and the select dimension has a single element.
    pub fn max_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::Max)
    }

    /// Similar to `max_keepdim` but the target dimension is squeezed.
    pub fn max<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::Max)
    }

    /// Gathers the minimum value across the selected dimension. The resulting shape has the same
    /// number of dimensions as the original tensor and the select dimension has a single element.
    pub fn min_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::Min)
    }

    /// Similar to `min_keepdim` but the target dimension is squeezed.
    pub fn min<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::Min)
    }

    pub fn argmax_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::ArgMax)
    }

    /// Similar to `argmax_keepdim` but the target dimension is squeezed.
    pub fn argmax<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::ArgMax)
    }

    pub fn argmin_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::ArgMin)
    }

    /// Similar to `argmin_keepdim` but the target dimension is squeezed.
    pub fn argmin<D: Dim>(&self, dim: D) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::ArgMin)
    }

    /// Computes the max of all the elements in this tensor and returns a tensor holding this
    /// scalar with zero dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.max_all()?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 5.);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn max_all(&self) -> Result<Self> {
        if self.rank() == 0 {
            Ok(self.clone())
        } else {
            self.flatten_all()?.max(0)
        }
    }

    /// Computes the min of all the elements in this tensor and returns a tensor holding this
    /// scalar with zero dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.min_all()?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 0.);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn min_all(&self) -> Result<Self> {
        if self.rank() == 0 {
            Ok(self.clone())
        } else {
            self.flatten_all()?.min(0)
        }
    }

    /// Computes the sum of all the elements in this tensor and returns a tensor holding this
    /// scalar with zero dimensions.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.sum_all()?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 15.);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn sum_all(&self) -> Result<Self> {
        let dims: Vec<_> = (0..self.rank()).collect();
        self.sum(dims)
    }

    pub fn mean_all(&self) -> Result<Self> {
        self.sum_all()? / self.elem_count() as f64
    }

    /// Returns the cumulative sum of elements of the input tensor summed over the specified
    /// dimension.
    ///
    /// This operation is most efficient when dim is the last dimension of the tensor.
    pub fn cumsum<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "cumsum")?;
        let rank = self.rank();
        if rank == 0 {
            return Ok(self.clone());
        }
        let n_axis = self.dim(dim)?;
        let triu = Tensor::triu2(n_axis, self.dtype(), self.device())?;
        if rank == 1 {
            self.unsqueeze(0)?.matmul(&triu)?.squeeze(0)
        } else {
            let last = rank - 1;
            let t = self.transpose(dim, last)?;
            let t = t.broadcast_matmul(&triu)?;
            t.transpose(dim, last)
        }
    }

    /// Returns log(sum(exp(tensor), dim)).
    pub fn log_sum_exp<D: Dims>(&self, sum_dims: D) -> Result<Self> {
        let sum_dims = sum_dims.to_indexes(self.shape(), "log-sum-exp")?;
        if sum_dims.is_empty() {
            return Ok(self.clone());
        }
        let max = sum_dims[1..]
            .iter()
            .try_fold(self.max_keepdim(sum_dims[0])?, |max, &dim| {
                max.max_keepdim(dim)
            })?;
        let exp = self.broadcast_sub(&max)?.exp()?;
        let sum = exp.sum(sum_dims.clone())?;

        sum.log()? + max.squeeze_dims(&sum_dims)
    }
}
