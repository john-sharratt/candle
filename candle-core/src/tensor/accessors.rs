//! Reads a tensor's metadata and storage, and hands out views that keep their
//! owner's lifetime.
//!
//! The shape, dtype, layout and device accessors do no device work. Methods
//! that return `Self` are views, and a view of a leased tensor stays a lease.
//! `same_storage` and `is_sole_owner` let a pool that hands out views of its
//! buffers decide whether a buffer may be handed out again.

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
    /// The dtype for the elements stored in the input tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device on which the input tensor is located.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The tensor shape, i.e. dimension sizes on each axis.
    pub fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    /// The dimension size for this tensor on each axis.
    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    /// The layout of the input tensor, this stores both the shape of the tensor as well as the
    /// strides and the start offset to apply to the underlying storage.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
    }

    /// The number of dimensions for this tensor, 0 for a scalar tensor, 1 for a 1D tensor, etc.
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// The number of elements stored in this tensor.
    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    /// The unique identifier for this tensor.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Whether this tensor is a variable or not. A variable is a tensor for which gradient is
    /// tracked and on which backpropagation can be performed.
    pub fn is_variable(&self) -> bool {
        self.is_variable
    }

    /// The recorded graph edge, if this tensor tracks one.
    ///
    /// `Op<'w>` and not an elided `Op<'_>`: elision would tie the operands to
    /// *this borrow* rather than to the tensor's own lifetime, which silently
    /// constrains every caller that walks the graph.
    pub(crate) fn op(&self) -> &Option<Op<'w>> {
        &self.op
    }

    pub(crate) fn storage(&self) -> std::sync::RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }

    pub(crate) fn storage_mut(&self) -> std::sync::RwLockWriteGuard<'_, Storage> {
        self.storage.write().unwrap()
    }

    // If we extend the visibility of this function to be usable outside of this crate, we should
    // make it unsafe.
    pub(crate) fn storage_mut_and_layout(
        &self,
    ) -> (std::sync::RwLockWriteGuard<'_, Storage>, &Layout) {
        let storage = self.storage.write().unwrap();
        (storage, &self.layout)
    }

    /// The storage used by this tensor, together with the layout to use to access it safely.
    pub fn storage_and_layout(&self) -> (std::sync::RwLockReadGuard<'_, Storage>, &Layout) {
        let storage = self.storage.read().unwrap();
        (storage, &self.layout)
    }

    /// Whether both tensors are views of the same allocation.
    ///
    /// Public because aliasing decides which operations are legal — `slice_set`
    /// refuses a source that shares its destination's storage — so a caller that
    /// may hold either two buffers or one handed to it twice has to be able to
    /// ask before it copies.
    pub fn same_storage(&self, rhs: &LiveTensor<'_>) -> bool {
        let lhs: &RwLock<Storage> = self.storage.as_ref();
        let rhs: &RwLock<Storage> = rhs.storage.as_ref();
        std::ptr::eq(lhs, rhs)
    }

    /// Whether this handle is the only thing reaching its storage — no clone
    /// of the handle and no view (`narrow`, `reshape`, …) of the buffer is
    /// alive anywhere else. A pool that hands out views of buffers it holds
    /// asks this to know which of them it may hand out again.
    pub fn is_sole_owner(&self) -> bool {
        Arc::strong_count(&self.0) == 1 && Arc::strong_count(&self.storage) == 1
    }
}
