//! Joins tensors along a dimension, as a view when the parts already are one
//! range.
//!
//! `stack` adds a dimension by joining equal-rank tensors. `cat_view` returns
//! the concatenation as a view with no copy when the parts are contiguous
//! ranges of one storage, each starting where the one before it ends. Otherwise
//! it returns `None` and the caller concatenates. A write through such a view
//! writes the parts it was cut from, which is what a caller that wants per
//! sequence views of a block back as one block relies on.

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
    /// Stacks two or more tensors along a particular dimension.
    ///
    /// All tensors must have the same rank, and the output has one additional rank
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::stack(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[2, 2, 3]);
    ///
    /// let c = Tensor::stack(&[&a, &b], 2)?;
    /// assert_eq!(c.shape().dims(), &[2, 3, 2]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn stack<A: AsRef<LiveTensor<'w>>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "stack" }.bt())?
        }
        let dim = dim.to_index_plus_one(args[0].as_ref().shape(), "stack")?;
        let args = args
            .iter()
            .map(|t| t.as_ref().unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::cat(&args, dim)
    }

    /// The tensor [`Self::cat`] would build from `parts` along `dim`, as a
    /// **view** — no allocation, no copy — when the parts already are that
    /// tensor: contiguous ranges of one storage, each starting where the one
    /// before it ends, with nothing ahead of `dim` but unit dims (so the
    /// concatenation is one range). `None` when they are not, and the caller
    /// concatenates.
    ///
    /// For data cut into per-sequence views and wanted whole again — the rows
    /// of one logits block handed out per sequence, then read as a block. The
    /// view aliases the parts' storage, which is what makes it free and what
    /// the caller must want: a write through it is a write to them.
    pub fn cat_view<A: AsRef<LiveTensor<'w>>, D: Dim>(parts: &[A], dim: D) -> Option<Self> {
        let first = parts.first()?.as_ref();
        let dim = dim.to_index(first.shape(), "cat_view").ok()?;
        let dims0 = first.dims();
        if dims0[..dim].iter().any(|&d| d != 1) {
            return None;
        }
        let mut cat_dims = dims0.to_vec();
        cat_dims[dim] = 0;
        let mut next = first.layout.start_offset();
        for part in parts {
            let part = part.as_ref();
            let dims = part.dims();
            if !part.is_contiguous()
                || !part.same_storage(first)
                || dims.len() != dims0.len()
                || part.layout.start_offset() != next
                || dims
                    .iter()
                    .zip(dims0)
                    .enumerate()
                    .any(|(i, (a, b))| i != dim && a != b)
            {
                return None;
            }
            cat_dims[dim] += dims[dim];
            next += part.elem_count();
        }
        let op = BackpropOp::new(parts, |args| Op::Cat(args, dim));
        Some(LiveTensor(Arc::new(TensorInner {
            lease: PhantomData,
            id: TensorId::new(),
            storage: first.storage.clone(),
            layout: Layout::contiguous_with_offset(cat_dims, first.layout.start_offset()),
            op,
            is_variable: false,
            dtype: first.dtype,
            device: first.device.clone(),
        })))
    }

    /// Normalize a 'relative' axis value: positive values are kept, negative
    /// values means counting the dimensions from the back.
    pub fn normalize_axis(&self, axis: i64) -> Result<usize> {
        let rank = self.rank() as i64;
        if rank <= axis {
            bail!("axis {axis} is too large, tensor rank {rank}")
        } else if 0 <= axis {
            Ok(axis as usize)
        } else {
            let naxis = rank + axis;
            if naxis < 0 {
                bail!("axis {axis} is too small, tensor rank {rank}")
            }
            Ok(naxis as usize)
        }
    }
}
