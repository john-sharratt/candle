//! Matrix products, dot products and norms, and the triangular and identity
//! matrices.
//!
//! `matmul` multiplies batched matrices of shape `[.., m, k]` by `[.., k, n]`.
//! `broadcast_matmul` accepts operands whose batch dimensions differ, and
//! concretises its broadcast operands with `contiguous`. `dot` and `mv` are
//! strict about shapes and reject a mismatch. `norm` is the Frobenius norm, and
//! `tril2`, `triu2` and `eye` build the masks and the identity.

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
    /// Computes the dot product of two 1D tensors.
    ///
    /// - If inputs are 1D vectors (`[n]`), returns their scalar dot product.
    /// - Panics if shapes are not compatible
    /// - Not supported for integer dtypes
    ///
    /// # Example (vectors)
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let t1 = Tensor::new(&[1.0, 2.0, 3.0], &Device::Cpu)?;
    /// let t2 = Tensor::new(&[4.0, 5.0, 6.0], &Device::Cpu)?;
    /// let res = t1.dot(&t2)?;
    /// assert_eq!(res.to_scalar::<f64>()?, 32.);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn dot(&self, rhs: &Self) -> Result<Self> {
        if self.dims().len() != 1 || rhs.dims().len() != 1 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "dot",
            });
        }

        (self * rhs).and_then(|ret| ret.sum_all())
    }

    /// Computes the **Frobenius norm** (L2 norm of all elements) of the tensor.
    /// - Output is `sqrt(sum(x^2))`.
    /// - Always returns a scalar (`[]` shape).
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let t = Tensor::new(&[[3., 4.], [0., 0.]], &Device::Cpu)?;
    /// let norm = t.norm()?;
    /// assert_eq!(norm.to_scalar::<f64>()?, 5.);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn norm(&self) -> Result<Self> {
        if self.dtype().is_int() {
            bail!("norm not supported for integer dtypes");
        }

        self.sqr().and_then(|x| x.sum_all()).and_then(|x| x.sqrt())
    }

    /// Performs strict matrix-vector multiplication (`[m, n] * [n] = [m]`).
    ///
    /// - If `self` is a matrix (`[m, n]`) and `rhs` is a vector (`[n]`), returns a vector (`[m]`).
    /// - **No broadcasting**: Panics if `self` is not 2D or if `rhs` is not 1D with matching size.
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let mat = Tensor::new(&[[1., 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    /// let vec = Tensor::new(&[1., 1., 1.], &Device::Cpu)?;
    /// let res = mat.mv(&vec)?;
    /// assert_eq!(res.to_vec1::<f64>()?, [6., 15.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn mv(&self, rhs: &Self) -> Result<Self> {
        // Strict shape checks
        let lhs_dims = self.dims();
        let rhs_dims = rhs.dims();
        if lhs_dims.len() != 2 || rhs_dims.len() != 1 || lhs_dims[1] != rhs_dims[0] {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "mv",
            });
        }

        // Direct matmul after ensuring rhs is column vector
        self.matmul(&rhs.unsqueeze(1)?)?.squeeze(1)
    }

    /// Returns the matrix-multiplication of the input tensor with the other provided tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor with dimensions `b1, b2, ..., bi, m, k`.
    /// * `rhs` - A tensor with dimensions `b1, b2, ..., bi, k, n`.
    ///
    /// The resulting tensor has dimensions `b1, b2, ..., bi, m, n`.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        if dim < 2 || b_dims.len() != dim {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .bt())?
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        if c_shape.elem_count() == 0 || k == 0 {
            return Tensor::zeros(c_shape, self.dtype(), self.device());
        }
        let batching: usize = a_dims[..dim - 2].iter().product();
        let batching_b: usize = b_dims[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .bt())?
        }

        let storage = self.storage().matmul(
            &rhs.storage(),
            (batching, m, n, k),
            self.layout(),
            rhs.layout(),
        )?;
        let op = BackpropOp::new2(self, rhs, Op::Matmul);
        Ok(from_storage(storage, c_shape, op, false))
    }

    /// Matrix-multiplication with broadcasting support.
    ///
    /// Compared to `matmul` the two matrixes are allowed to have different dimensions as long as
    /// they are compatible for broadcast. E.g. if `self` has shape `(j, 1, n, k)` and `rhs` has
    /// shape `(l, k, m)`, the output will have shape `(j, l, n, m)`.
    pub fn broadcast_matmul(&self, rhs: &Self) -> Result<Self> {
        let lhs = self;
        let (l_shape, r_shape) = lhs.shape().broadcast_shape_matmul(rhs.shape())?;
        let l_broadcast = l_shape != *lhs.shape();
        let r_broadcast = r_shape != *rhs.shape();
        // Broadcast operands are concretised via `contiguous` here; a strided matmul
        // path would avoid the copy.
        match (l_broadcast, r_broadcast) {
            (true, true) => lhs
                .broadcast_as(&l_shape)?
                .contiguous()?
                .matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),
            (false, true) => lhs.matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),
            (true, false) => lhs.broadcast_as(&l_shape)?.contiguous()?.matmul(rhs),
            (false, false) => lhs.matmul(rhs),
        }
    }

    /// Returns a lower triangular matrix of ones of size n by n.
    pub fn tril2(n: usize, dtype: DType, device: &Device) -> Result<Self> {
        let t = Tensor::arange(0u32, n as u32, device)?;
        let t1 = t.reshape((1, n))?.broadcast_as((n, n))?;
        let t2 = t.reshape((n, 1))?.broadcast_as((n, n))?;
        t1.le(&t2)?.to_dtype(dtype)
    }

    /// Returns an upper triangular matrix of ones of size n by n.
    pub fn triu2(n: usize, dtype: DType, device: &Device) -> Result<Self> {
        let t = Tensor::arange(0u32, n as u32, device)?;
        let t1 = t.reshape((1, n))?.broadcast_as((n, n))?;
        let t2 = t.reshape((n, 1))?.broadcast_as((n, n))?;
        t1.ge(&t2)?.to_dtype(dtype)
    }

    /// Returns a matrix with a diagonal of ones of size n by n.
    pub fn eye(n: usize, dtype: DType, device: &Device) -> Result<Self> {
        let t = Tensor::arange(0u32, n as u32, device)?;
        let t1 = t.reshape((1, n))?.broadcast_as((n, n))?;
        let t2 = t.reshape((n, 1))?.broadcast_as((n, n))?;
        t1.eq(&t2)?.to_dtype(dtype)
    }
}
