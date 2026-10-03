use crate::op::{BackpropOp, Op};
use crate::wave_provenance::WaveTicket;
use crate::{shape::Dim, Context, Error, LiveTensor, Result, Shape, Storage, Tensor};

impl<'w> LiveTensor<'w> {
    /// Concatenates two or more tensors along a particular dimension.
    ///
    /// All tensors must of the same rank, and the output will have
    /// the same rank
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::cat(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[4, 3]);
    ///
    /// let c = Tensor::cat(&[&a, &b], 1)?;
    /// assert_eq!(c.shape().dims(), &[2, 6]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn cat<A: AsRef<LiveTensor<'w>>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let dim = dim.to_index(arg0.shape(), "cat")?;
        for arg in args {
            arg.as_ref().check_dim(dim, "cat")?;
        }
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg0.rank() != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: arg0.rank(),
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx != dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
        }
        let all_contiguous = args.iter().all(|v| v.as_ref().is_contiguous());
        if all_contiguous {
            Self::cat_contiguous(args, dim)
        } else if dim == 0 {
            Self::cat0(args)
        } else {
            let args: Vec<Self> = args
                .iter()
                .map(|a| a.as_ref().transpose(0, dim))
                .collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<A: AsRef<LiveTensor<'w>>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
                if dim_idx != 0 && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
            let next_offset = offsets.last().context("empty offsets")? + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = BackpropOp::new(args, |args| Op::Cat(args, 0));
        // The concatenation lands in the arena its inputs came from — `arg0` is
        // representative because every argument shares one `'w`, so they cannot
        // come from different generations.
        let mut storage = unsafe { device.alloc_uninit_from(&shape, dtype, arg0.wave_ticket())? };
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage()
                .copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(crate::tensor::from_storage(storage, shape, op, false))
    }

    /// [`Self::cat`] into an **owned** allocation: a `'static` tensor from the
    /// pool rather than the arena the arguments came from — what
    /// [`Self::to_owned_tensor`] is to a single tensor. One copy, with the same
    /// run merging as `cat`, where `cat` followed by `to_owned_tensor` would
    /// copy twice and put the intermediate on the very span the result is being
    /// taken off.
    pub fn cat_owned<A: AsRef<LiveTensor<'w>>, D: Dim>(args: &[A], dim: D) -> Result<Tensor> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let dim = dim.to_index(args[0].as_ref().shape(), "cat")?;
        if !args.iter().all(|a| a.as_ref().is_contiguous()) {
            // A strided argument goes through the general concatenation; the
            // copy below reads ranges, which only a contiguous layout is.
            return Self::cat(args, dim)?.to_owned_tensor();
        }
        let (storage, shape) = Self::cat_contiguous_storage(args, dim, None)?;
        Ok(crate::tensor::from_storage(
            storage,
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    fn cat_contiguous<A: AsRef<LiveTensor<'w>>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        // The concatenation lands in the arena its inputs came from — `arg0` is
        // representative because every argument shares one `'w`, so they cannot
        // come from different generations.
        let (storage, shape) = Self::cat_contiguous_storage(args, dim, arg0.wave_ticket())?;
        let op = BackpropOp::new(args, |args| Op::Cat(args, dim));
        Ok(crate::tensor::from_storage(storage, shape, op, false))
    }

    /// The contiguous concatenation's validation and copy, into storage
    /// allocated against `ticket` (`None`: an owned pool allocation).
    fn cat_contiguous_storage<A: AsRef<LiveTensor<'w>>>(
        args: &[A],
        dim: usize,
        ticket: Option<WaveTicket>,
    ) -> Result<(Storage, Shape)> {
        let arg0 = args[0].as_ref();
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[dim] = 0;
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == dim {
                    cat_dims[dim] += v2;
                }
                if dim_idx != dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
        }
        let cat_target_dim_len = cat_dims[dim];
        let block_size: usize = cat_dims.iter().skip(1 + dim).product();
        let shape = Shape::from(cat_dims);
        let mut storage = unsafe { device.alloc_uninit_from(&shape, dtype, ticket)? };
        // Every argument shares the dims before `dim`, so this is one value.
        let d1: usize = first_dims.iter().take(dim).product();
        let dst_s = block_size * cat_target_dim_len;
        let mut dst_o = 0;
        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_ref();
            let src_o = arg.layout().start_offset();
            let mut d2 = block_size * arg.shape().dims()[dim];
            let mut j = i + 1;
            // **A run of arguments laid back to back in one storage is one copy.**
            // With nothing outside `dim` (d1 == 1) every argument is a single
            // contiguous range and the result is their ranges end to end, so
            // arguments that already sit end to end in the same buffer — the
            // per-sequence views of one upload, the rows of one logits block —
            // are one range of it. One launch per run instead of one per
            // argument; the result is still a copy, so nothing aliases.
            if d1 == 1 {
                while j < args.len() {
                    let next = args[j].as_ref();
                    if !next.same_storage(arg) || next.layout().start_offset() != src_o + d2 {
                        break;
                    }
                    d2 += block_size * next.shape().dims()[dim];
                    j += 1;
                }
            }
            arg.storage().copy2d(
                &mut storage,
                d1,
                d2,
                /* src_s */ d2,
                dst_s,
                src_o,
                dst_o,
            )?;
            dst_o += d2;
            i = j;
        }
        Ok((storage, shape))
    }

    /// Set the values on `self` using values from `src`. The copy starts at the specified
    /// `offset` for the target dimension `dim` on `self`.
    /// `self` and `src` must have the same shape except on dimension `dim` where the `self` size
    /// has to be greater than or equal to `offset` plus the `src` size.
    ///
    /// Note that this modifies `self` in place and as such is not compatible with
    /// back-propagation.
    ///
    /// `src` carries its own lifetime for the same reason the `*_mut` ops do:
    /// this writes `self` and only reads `src`, so writing a wave-scoped result
    /// into an owned buffer — a scan's chunk output into the segment buffer, a
    /// conv's advanced tail into the carried state — needs no relation between
    /// the two.
    pub fn slice_set<D: Dim>(&self, src: &LiveTensor<'_>, dim: D, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set")?;
        if !self.is_contiguous() || !src.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set" }.bt())?
        }
        if self.same_storage(src) {
            crate::bail!("cannot use slice_set when self and src share their storage")
        }
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.device().location() != src.device().location() {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim && *v2 + offset > *v1 {
                crate::bail!("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
            }
            if dim_idx != dim && v1 != v2 {
                crate::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o = self.layout().start_offset() + offset * block_size;
        let src_o = src.layout().start_offset();
        src.storage().copy2d(
            &mut self.storage_mut(),
            d1,
            d2,
            /* src_s */ d2,
            /* dst_s */ block_size * self.dims()[dim],
            src_o,
            dst_o,
        )?;

        Ok(())
    }
}
