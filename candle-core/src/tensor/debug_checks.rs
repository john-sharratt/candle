//! Records a tensor's finite-value statistics under a name for the
//! `tensor-assert` harness.
//!
//! `assert` folds the NaN count, the Inf count and the finite minimum and
//! maximum into a named slot, and returns the tensor so the call chains inline.
//! It is infallible on purpose, because a probe that can fail the program it
//! observes changes what that program does. Without the `tensor-assert`
//! feature it compiles to nothing. `assert_once` records a name only the first
//! time it is seen in an epoch, for values that do not change between forwards.

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
    /// Fold this tensor's NaN count, Inf count and finite min/max into `name`'s
    /// assert slot, and return the tensor so the call chains inline:
    ///
    /// ```rust,ignore
    /// let y = x.assert("attn.qkv_in").matmul(&w)?;
    /// ```
    ///
    /// Under the `tensor-assert` feature this is one kernel launch that reads
    /// the tensor where it already lives — no synchronisation, no allocation,
    /// no dtype conversion, no copy. Results are read by
    /// `tensor_assert::report` at a synchronisation the caller already
    /// performs. Without the feature it compiles to nothing.
    ///
    /// It is infallible on purpose. A probe that can fail the program it
    /// observes changes what the program does, which is the one thing this must
    /// never do; internal errors are logged instead.
    #[cfg(feature = "tensor-assert")]
    pub fn assert(&self, name: &str) -> &Self {
        crate::tensor_assert::assert_tensor(self, name);
        self
    }

    /// No-op twin of [`Self::assert`]: the feature is off, so the call and its
    /// name literal compile away entirely.
    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    pub fn assert(&self, _name: &str) -> &Self {
        self
    }

    /// [`Self::assert`], but only the first time this name is seen in the
    /// current epoch.
    ///
    /// For values that do not change between forwards — a weight, a rope table,
    /// a norm constant — where re-reading them every layer of every wave would
    /// be exactly the bandwidth perturbation the design avoids.
    #[cfg(feature = "tensor-assert")]
    pub fn assert_once(&self, name: &str) -> &Self {
        if crate::tensor_assert::should_run_once(name) {
            crate::tensor_assert::assert_tensor(self, name);
        }
        self
    }

    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    pub fn assert_once(&self, _name: &str) -> &Self {
        self
    }
}
