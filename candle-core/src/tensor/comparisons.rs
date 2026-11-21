//! Element-wise comparisons and selection between tensors.
//!
//! Each comparison returns a tensor of the same shape with `u8` elements, 1
//! where the relation holds and 0 otherwise. `where_cond` reads that mask to
//! pick between two branches element by element. `cmp` takes the operation as
//! an argument, and `eq`, `ne`, `lt`, `gt`, `ge` and `le` are shorthands for it.

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
    /// Element-wise comparison between two tensors, e.g. equality, greater than, ... The actual
    /// comparison operation is specified by the `op` argument.
    ///
    /// The returned tensor has the same shape as the original tensors and uses `u8` elements.
    pub fn cmp<T: TensorOrScalar<'w>>(&self, rhs: T, op: CmpOp) -> Result<Self> {
        let rhs = match rhs.to_tensor_scalar()? {
            crate::scalar::TensorScalar::Tensor(rhs) => rhs,
            crate::scalar::TensorScalar::Scalar(rhs) => rhs
                .to_dtype(self.dtype())?
                .to_device(self.device())?
                .broadcast_as(self.shape())?,
        };
        let shape = self.same_shape_binary_op(&rhs, "cmp")?;
        let storage = self
            .storage()
            .cmp(op, &rhs.storage(), self.layout(), rhs.layout())?;
        let op = BackpropOp::new1(self, |a| Op::Cmp(a, op));
        Ok(from_storage(storage, shape.dims(), op, false))
    }

    /// Element-wise equality.
    pub fn eq<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
        self.cmp(rhs, CmpOp::Eq)
    }

    /// Element-wise non-equality.
    pub fn ne<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
        self.cmp(rhs, CmpOp::Ne)
    }

    /// Element-wise comparison with lower-than, the returned tensor uses value 1 where `self <
    /// rhs` and 0 otherwise.
    pub fn lt<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
        self.cmp(rhs, CmpOp::Lt)
    }

    /// Element-wise comparison with greater-than, the returned tensor uses value 1 where `self >
    /// rhs` and 0 otherwise.
    pub fn gt<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
        self.cmp(rhs, CmpOp::Gt)
    }

    /// Element-wise comparison with greater-equal, the returned tensor uses value 1 where `self >=
    /// rhs` and 0 otherwise.
    pub fn ge<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
        self.cmp(rhs, CmpOp::Ge)
    }

    /// Element-wise comparison with lower-equal, the returned tensor uses value 1 where `self <=
    /// rhs` and 0 otherwise.
    pub fn le<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
        self.cmp(rhs, CmpOp::Le)
    }

    /// Returns a tensor with the same shape as the input tensor, the values are taken from
    /// `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
    /// input tensor is equal to zero.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let _shap = self.same_shape_binary_op(on_true, "where_cond")?;
        let shape = self.same_shape_binary_op(on_false, "where_cond")?;
        let storage = self.storage().where_cond(
            self.layout(),
            &on_true.storage(),
            on_true.layout(),
            &on_false.storage(),
            on_false.layout(),
        )?;
        let op = BackpropOp::new3(self, on_true, on_false, Op::WhereCond);
        Ok(from_storage(storage, shape, op, false))
    }
}
