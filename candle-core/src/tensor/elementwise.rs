//! Element-wise arithmetic, in-place updates and unary functions on a tensor.
//!
//! The binary operators allocate a fresh output, because reusing an operand's
//! storage could create a cycle in the compute graph. The `*_mut` variants
//! write `self` and only read `rhs`, so a wave-scoped operand can be
//! accumulated into an owned buffer. The unary functions, the broadcasting
//! forms, `affine`, `clamp` and `pow` complete the set.

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
    pub(crate) fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs != rhs {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            }
            .bt())
        } else {
            Ok(lhs)
        }
    }

    /// Returns true if the computation graph should track this op, that is if it is
    /// a variable or if it has some variable as dependencies.
    pub fn track_op(&self) -> bool {
        self.is_variable || self.op.is_some()
    }

    // These binary ops allocate a fresh output; there is no in-place / pre-allocated
    // variant, since reusing an operand's storage could create cycles in the compute graph.
    binary_op!(add, Add);

    binary_op!(mul, Mul);

    binary_op!(sub, Sub);

    binary_op!(div, Div);

    binary_op_scalar!(maximum, Maximum);

    binary_op_scalar!(minimum, Minimum);

    broadcast_binary_op!(broadcast_add, add);

    broadcast_binary_op!(broadcast_mul, mul);

    broadcast_binary_op!(broadcast_sub, sub);

    broadcast_binary_op!(broadcast_div, div);

    broadcast_binary_op!(broadcast_maximum, maximum);

    broadcast_binary_op!(broadcast_minimum, minimum);

    broadcast_binary_op!(broadcast_eq, eq);

    broadcast_binary_op!(broadcast_ne, ne);

    broadcast_binary_op!(broadcast_lt, lt);

    broadcast_binary_op!(broadcast_le, le);

    broadcast_binary_op!(broadcast_gt, gt);

    broadcast_binary_op!(broadcast_ge, ge);

    /// In-place addition: `self += rhs`.
    ///
    /// This operation modifies the tensor in-place without creating a new tensor,
    /// which is more memory-efficient for large tensors.
    ///
    /// # Requirements
    /// - `self` must be contiguous
    /// - `self` and `rhs` must have the same shape
    /// - `self` and `rhs` must have the same dtype
    /// - No autograd support (gradients are not tracked for in-place ops)
    ///
    /// # Example
    /// ```rust
    /// # use candle_core::{Tensor, Device, Result};
    /// # fn main() -> Result<()> {
    /// let mut a = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    /// let b = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?;
    /// a.add_mut(&b)?;  // a is now [1.5, 2.5, 3.5]
    /// # Ok(())
    /// # }
    /// ```
    /// **`rhs` carries its own lifetime, and every `*_mut` below does the same.**
    /// An in-place op writes `self` and only *reads* `rhs`, so the two need no
    /// relation: accumulating a wave-scoped result into an owned buffer is the
    /// residual add, and decaying a sequence's carried state by a wave-scoped
    /// factor is the DeltaNet chunk carry. Requiring `&Self` would make those
    /// unwritable without a copy, for a borrow that ends when the call does.
    pub fn add_mut(&mut self, rhs: &LiveTensor<'_>) -> Result<()> {
        self.binary_inplace_op(rhs, crate::op::BinaryInplaceOp::Add)
    }

    /// In-place subtraction: `self -= rhs`.
    ///
    /// See [`add_mut`](Self::add_mut) for details on requirements.
    pub fn sub_mut(&mut self, rhs: &LiveTensor<'_>) -> Result<()> {
        self.binary_inplace_op(rhs, crate::op::BinaryInplaceOp::Sub)
    }

    /// In-place multiplication: `self *= rhs`.
    ///
    /// See [`add_mut`](Self::add_mut) for details on requirements.
    pub fn mul_mut(&mut self, rhs: &LiveTensor<'_>) -> Result<()> {
        self.binary_inplace_op(rhs, crate::op::BinaryInplaceOp::Mul)
    }

    /// In-place division: `self /= rhs`.
    ///
    /// See [`add_mut`](Self::add_mut) for details on requirements.
    pub fn div_mut(&mut self, rhs: &LiveTensor<'_>) -> Result<()> {
        self.binary_inplace_op(rhs, crate::op::BinaryInplaceOp::Div)
    }

    /// In-place minimum: `self = min(self, rhs)`.
    ///
    /// See [`add_mut`](Self::add_mut) for details on requirements.
    /// Note: Not supported for integer dtypes (U8, U32, I64).
    pub fn minimum_mut(&mut self, rhs: &LiveTensor<'_>) -> Result<()> {
        self.binary_inplace_op(rhs, crate::op::BinaryInplaceOp::Min)
    }

    /// In-place maximum: `self = max(self, rhs)`.
    ///
    /// See [`add_mut`](Self::add_mut) for details on requirements.
    /// Note: Not supported for integer dtypes (U8, U32, I64).
    pub fn maximum_mut(&mut self, rhs: &LiveTensor<'_>) -> Result<()> {
        self.binary_inplace_op(rhs, crate::op::BinaryInplaceOp::Max)
    }

    /// Internal helper for in-place binary operations.
    pub(super) fn binary_inplace_op(
        &mut self,
        rhs: &LiveTensor<'_>,
        op: crate::op::BinaryInplaceOp,
    ) -> Result<()> {
        // Check shapes match
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape != rhs_shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
                op: "binary_inplace",
            }
            .bt());
        }

        // Check contiguous
        if !self.is_contiguous() {
            return Err(Error::RequiresContiguous {
                op: "binary_inplace",
            }
            .bt());
        }

        if self.elem_count() == 0 {
            return Ok(());
        }

        // Get mutable access to storage
        let lhs_layout = self.layout().clone();
        let rhs_layout = rhs.layout().clone();
        let mut storage = self.storage_mut();

        storage.binary_inplace_impl(op, &rhs.storage(), &lhs_layout, &rhs_layout)
    }

    unary_op!(recip, Recip);

    unary_op!(neg, Neg);

    unary_op!(exp, Exp);

    unary_op!(log, Log);

    unary_op!(sin, Sin);

    unary_op!(cos, Cos);

    unary_op!(tanh, Tanh);

    unary_op!(abs, Abs);

    unary_op!(sqr, Sqr);

    unary_op!(sqrt, Sqrt);

    unary_op!(gelu, Gelu);

    unary_op!(gelu_erf, GeluErf);

    unary_op!(erf, Erf);

    unary_op!(relu, Relu);

    unary_op!(silu, Silu);

    unary_op!(ceil, Ceil);

    unary_op!(floor, Floor);

    unary_op!(round, Round);

    unary_op!(sign, Sign);

    /// Round element of the input tensor to the nearest integer.
    ///
    /// If the number of decimals is negative, it specifies the number of positions to the left of
    /// the decimal point.
    pub fn round_to(&self, decimals: i32) -> Result<Self> {
        let mult = 10f64.powi(decimals);
        (self * mult)?.round()? * (1f64 / mult)
    }

    /// This operation multiplies the input tensor by `mul` then adds `add` and return the result.
    /// The input values `mul` and `add` are casted to the appropriate type so some rounding might
    /// be performed.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::Cpu)?;
    /// let a = a.affine(4., -2.)?;
    /// assert_eq!(a.to_vec2::<f32>()?, &[[-2.0, 2.0], [6.0, 10.0]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage().affine(self.layout(), mul, add)?;
        let op = BackpropOp::new1(self, |arg| Op::Affine { arg, mul, add });
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Applies the Exponential Linear Unit (ELU) function on each element of the input tensor.
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage().elu(self.layout(), alpha)?;
        let op = BackpropOp::new1(self, |t| Op::Elu(t, alpha));
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Raise the tensor to some float exponent `e`.
    pub fn powf(&self, e: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage().powf(self.layout(), e)?;
        let op = BackpropOp::new1(self, |t| Op::Powf(t, e));
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Clamp the tensor values to be between `min` and `max`.
    pub fn clamp<T1: TensorOrScalar<'w>, T2: TensorOrScalar<'w>>(
        &self,
        min: T1,
        max: T2,
    ) -> Result<Self> {
        self.maximum(min)?.minimum(max)
    }

    /// Pointwise pow operation.
    pub fn pow(&self, rhs: &Tensor) -> Result<Self> {
        rhs.mul(&self.log()?)?.exp()
    }

    /// Broadcasting version of `pow`.
    pub fn broadcast_pow(&self, rhs: &Tensor) -> Result<Self> {
        rhs.broadcast_mul(&self.log()?)?.exp()
    }
}
