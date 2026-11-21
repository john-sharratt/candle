//! The `Tensor` type and the lifetime-generic `LiveTensor` it is built on.
//!
//! A tensor is a reference-counted handle to storage and a layout. The `'w`
//! parameter of `LiveTensor` says how long its storage stays valid, so a view
//! of a wave arena cannot be used where an owned tensor is required. `Tensor`
//! is `LiveTensor<'static>`. Covariance means an owned tensor is accepted where
//! a shorter-lived one is expected, and never the reverse. The method blocks are
//! split by concern, and each file below holds one of them.
//!
//! - `accessors`: metadata, storage access and view-sharing checks.
//! - `construct`: constructors, leased and wrapped device memory.
//! - `comparisons`: element-wise comparisons and `where_cond`.
//! - `concat`: `stack` and the view-returning `cat_view`.
//! - `elementwise`: arithmetic, in-place updates and unary functions.
//! - `scalar_ops`: operators between an `f64` and a tensor.
//! - `indexing`: index-based selection, gather and scatter.
//! - `sparse_index_ops`: updates at a few indices of the last dimension.
//! - `linalg`: matrix products, dot products and norms.
//! - `reductions`: sums, means, extrema and their `_keepdim` forms.
//! - `resample`: interpolation and 2D pooling.
//! - `shape_views`: reshapes, permutations, narrowing and padding.
//! - `readback`: copies of values to the host.
//! - `transfers`: device and dtype moves, and owned copies.
//! - `debug_checks`: the `tensor-assert` finiteness probes.

#![allow(clippy::redundant_closure_call)]
use crate::backend::{BackendDevice, BackendStorage};
#[cfg(feature = "cuda")]
use crate::cuda_backend::{CudaStorage, LeaseAnchor};
use crate::op::{BackpropOp, BinaryOp, CmpOp, Op, ReduceOp, UnaryOp};
use crate::scalar::TensorOrScalar;
use crate::shape::{Dim, Dims, ShapeWithOneHole};
use crate::{bail, storage::Storage, DType, Device, Error, Layout, Result, Shape};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// The shared body of a tensor: what `Tensor`'s `Arc` points at.
///
/// `'w` is how long this tensor's storage stays valid. It has to travel through
/// `op`, which holds the tensors this one was derived from — a graph node must
/// not outlive its operands' memory — but travelling is not the same as being
/// grounded: `Op` -> `LiveTensor` -> `TensorInner` -> `BackpropOp` -> `Op` is a
/// cycle, and `storage` carries no lifetime of its own, because a lease is a
/// runtime `Backing` marker rather than a borrow. So the parameter bottoms out
/// in the marker below, and variance is stated there rather than inferred.
pub struct TensorInner<'w> {
    /// Covariant in `'w`: an owned tensor is usable wherever a shorter-lived
    /// one is expected, never the reverse. `&'w [u8]` and not `&'w mut [u8]`,
    /// which would be invariant and would reject exactly that coercion.
    lease: PhantomData<&'w [u8]>,
    id: TensorId,
    // As we provide inner mutability on the tensor content, the alternatives are:
    // - Using a mutex, this would have the highest cost when retrieving the storage but would
    //   prevent errors when concurrent access takes place. Mutex would also be subject to
    //   deadlocks for example using the current code if the same tensor is used twice by a single
    //   binary op.
    // - Using a refcell unsafe cell would have some intermediary cost, borrow checking would be
    //   verified dynamically, but the resulting tensors would not be send or sync.
    // - Using an unsafe cell would have the lowest cost but undefined behavior on concurrent
    //   accesses.
    // Ideally, we would use Arc<Storage> for tensors on which we don't plan on modifying the data
    // and Arc<Mutex<Storage>> for tensors where the data could be modified, e.g. variables but
    // that's tricky to encode in the current setup.
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    op: BackpropOp<'w>,
    is_variable: bool,
    dtype: DType,
    device: Device,
}

impl<'w> AsRef<LiveTensor<'w>> for LiveTensor<'w> {
    fn as_ref(&self) -> &LiveTensor<'w> {
        self
    }
}

// Tensors are refcounted so that cloning is cheap when building the op graph.
// Storages are also refcounted independently so that its possible to avoid
// copying the storage for operations that only modify the shape or stride.
#[derive(Clone)]
/// The core struct for manipulating tensors.
///
/// ```rust
/// use candle_core::{Tensor, DType, Device};
///
/// let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
/// let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
///
/// let c = a.matmul(&b)?;
/// # Ok::<(), candle_core::Error>(())
/// ```
///
/// Tensors are reference counted with [`Arc`] so cloning them is cheap.
///
/// # Lifetime
///
/// `'w` is how long this tensor's storage stays valid. Anything allocated from
/// the device pool owns its memory and so is [`Tensor`] — `LiveTensor<'static>`
/// — which is what nearly all code means and writes. A tensor that *views*
/// memory owned elsewhere carries that owner's lifetime instead: a KV arena
/// slot, or a range of an in-flight inference wave's transient half, which is
/// reclaimed the moment the wave's guard drops.
///
/// The parameter is covariant, so an owned tensor is accepted anywhere a
/// shorter-lived one is expected and never the reverse:
///
/// ```
/// use candle_core::{LiveTensor, Tensor};
/// fn shorten<'a>(t: Tensor) -> LiveTensor<'a> { t }
/// ```
///
/// ```compile_fail
/// use candle_core::{LiveTensor, Tensor};
/// fn lengthen<'a>(t: LiveTensor<'a>) -> Tensor { t }
/// ```
///
/// That asymmetry is the whole point: it is what stops a wave buffer escaping
/// the scope that will free it, without anyone having to order `drop` calls by
/// hand. [`crate::Module`] and the `CustomOp` traits are deliberately left at
/// `Tensor`, so a borrowed tensor cannot be handed to something that might
/// store it.
pub struct LiveTensor<'w>(Arc<TensorInner<'w>>);

/// The everyday tensor: it owns its storage, so that storage is live for as
/// long as anything holds a reference to it.
pub type Tensor = LiveTensor<'static>;

impl<'w> std::ops::Deref for LiveTensor<'w> {
    type Target = TensorInner<'w>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self
                .storage()
                .unary_impl::<crate::op::$op_name>(self.layout())?;
            let op = BackpropOp::new1(self, |s| Op::Unary(s, UnaryOp::$op_name));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            if shape.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self.storage().binary_impl::<crate::op::$op_name>(
                &*rhs.storage(),
                self.layout(),
                rhs.layout(),
            )?;
            let op = BackpropOp::new2(self, rhs, |t1, t2| Op::Binary(t1, t2, BinaryOp::$op_name));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! binary_op_scalar {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: TensorOrScalar<'w>>(&self, rhs: T) -> Result<Self> {
            let rhs = match rhs.to_tensor_scalar()? {
                crate::scalar::TensorScalar::Tensor(rhs) => rhs,
                crate::scalar::TensorScalar::Scalar(rhs) => rhs
                    .to_dtype(self.dtype())?
                    .to_device(self.device())?
                    .broadcast_as(self.shape())?,
            };
            let shape = self.same_shape_binary_op(&rhs, stringify!($fn_name))?;
            if self.elem_count() == 0 {
                return Ok(self.clone());
            }
            let storage = self.storage().binary_impl::<crate::op::$op_name>(
                &*rhs.storage(),
                self.layout(),
                rhs.layout(),
            )?;
            let op = BackpropOp::new2(self, &rhs, |t1, t2| Op::Binary(t1, t2, BinaryOp::$op_name));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! broadcast_binary_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let lhs = self;
            let shape = lhs
                .shape()
                .broadcast_shape_binary_op(rhs.shape(), stringify!($fn_name))?;
            let l_broadcast = shape != *lhs.shape();
            let r_broadcast = shape != *rhs.shape();
            match (l_broadcast, r_broadcast) {
                (true, true) => lhs
                    .broadcast_as(&shape)?
                    .$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

/// Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
///
/// `'w` comes from `op`: the result must not outlive the operands recorded in
/// it. A storage built from a pool allocation carries no constraint of its own,
/// so an owned input yields `'static` and the caller sees a plain `Tensor`.
pub(crate) fn from_storage<'w, S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: BackpropOp<'w>,
    is_variable: bool,
) -> LiveTensor<'w> {
    let dtype = storage.dtype();
    let device = storage.device();
    let tensor_ = TensorInner {
        lease: PhantomData,
        id: TensorId::new(),
        storage: Arc::new(RwLock::new(storage)),
        layout: Layout::contiguous(shape),
        op,
        is_variable,
        dtype,
        device,
    };
    LiveTensor(Arc::new(tensor_))
}

/// The arithmetic operator impls, generic over `'w` so that `a + b` on wave
/// buffers yields a wave-scoped result instead of one claiming `'static`.
///
/// Both operands share a single `'w`. That is not a restriction in practice:
/// `LiveTensor` is covariant, so mixing an owned tensor with a leased one
/// simply resolves `'w` to the shorter of the two — which is exactly the
/// lifetime the result is valid for.
macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<'w, B: std::borrow::Borrow<LiveTensor<'w>>> std::ops::$trait<B> for LiveTensor<'w> {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: B) -> Self::Output {
                LiveTensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<'w, B: std::borrow::Borrow<LiveTensor<'w>>> std::ops::$trait<B> for &LiveTensor<'w> {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: B) -> Self::Output {
                LiveTensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<'w, B: std::borrow::Borrow<LiveTensor<'w>>> std::ops::$trait<LiveTensor<'w>>
            for Result<B>
        {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: LiveTensor<'w>) -> Self::Output {
                LiveTensor::$fn1(self?.borrow(), &rhs)
            }
        }

        impl<'w, B: std::borrow::Borrow<LiveTensor<'w>>> std::ops::$trait<&LiveTensor<'w>>
            for Result<B>
        {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: &LiveTensor<'w>) -> Self::Output {
                LiveTensor::$fn1(self?.borrow(), rhs)
            }
        }

        impl<'w, B: std::borrow::Borrow<LiveTensor<'w>>> std::ops::$trait<Result<B>>
            for LiveTensor<'w>
        {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                LiveTensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<'w, B: std::borrow::Borrow<LiveTensor<'w>>> std::ops::$trait<Result<B>>
            for &LiveTensor<'w>
        {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                LiveTensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<'w> std::ops::$trait<f64> for LiveTensor<'w> {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }

        impl<'w> std::ops::$trait<f64> for &LiveTensor<'w> {
            type Output = Result<LiveTensor<'w>>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

/// The methods that hand `self` to something which may outlive the call.
///
/// [`crate::Module`] and [`crate::ModuleT`] take `&Tensor`, so a module is free
/// to keep what it is given — cache it, close over it, store it beside its
/// weights. That is why they are not generic over `'w`, and why these two
/// methods sit here rather than in the main block: running a module over a
/// wave-scoped buffer is refused at compile time instead of producing a tensor
/// that outlives the half it points into. Copy it off the wave first
/// (`.contiguous()` onto owned storage) if that is really what you want.
impl Tensor {
    /// Run the `forward` method of `m` on `self`.
    pub fn apply<M: crate::Module>(&self, m: &M) -> Result<Self> {
        m.forward(self)
    }

    /// Run the `forward` method of `m` on `self`.
    pub fn apply_t<M: crate::ModuleT>(&self, m: &M, train: bool) -> Result<Self> {
        m.forward_t(self, train)
    }
}

mod accessors;
mod comparisons;
mod concat;
mod construct;
mod debug_checks;
mod elementwise;
mod indexing;
mod linalg;
mod readback;
mod reductions;
mod resample;
mod scalar_ops;
mod shape_views;
mod sparse_index_ops;
mod transfers;
