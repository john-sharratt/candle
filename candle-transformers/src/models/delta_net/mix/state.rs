//! The carried per-sequence state of one DeltaNet layer: the recurrent matrix and
//! the convolution tail.
//!
//! The buffers are written in place, so the type is deliberately not cloneable.
//! A shallow clone would track later writes as though it were a snapshot, and a
//! snapshot is an explicit deep copy. A state can be a view over memory in the
//! device reservation, anchored to its owner, so that the owner cannot reuse the
//! memory while any handle still reads it. Wave work allocates its write buffers
//! without zeroing, since the kernels overwrite them completely. A state at
//! sequence start is zeroed, because it is read at zero.

use super::*;

/// The carried per-sequence state of one DeltaNet layer.
///
/// **Deliberately not `Clone`.** The buffers are written in place, and
/// `Tensor::clone` is a shallow handle clone that shares storage — so a derived
/// `Clone` would hand back something that *looks* like a snapshot and then
/// tracks every subsequent mutation. That is not a hypothetical: it silently
/// broke snapshot-and-resume the moment the state stopped being replaced.
/// [`Self::snapshot`] is the deep copy, and it is the only way to get a second
/// one.
#[derive(Debug)]
pub struct DeltaNetState {
    /// `[n_v_heads, d_v, d_k]`, F32.
    pub s: Tensor,
    /// The last `conv_kernel − 1` conv inputs, `[conv_dim, conv_kernel − 1]`,
    /// F32. Zeros at sequence start (causal left-padding).
    pub conv_tail: Tensor,
}

impl DeltaNetState {
    /// An independent copy of both buffers — what a snapshot has to be.
    ///
    /// Allocates, so it belongs to the reference model's per-layer state
    /// bookkeeping, not to anything on the wave path (a wave preserves the
    /// entering state by not writing it — see `RecurrentStateStore`).
    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            s: self.s.copy()?,
            conv_tail: self.conv_tail.copy()?,
        })
    }

    /// Overwrite both buffers with `src`'s, in place.
    ///
    /// The same *arithmetic* as [`Self::snapshot`] into a destination that
    /// already exists, which is what a **restore** wants: `import` has bytes
    /// from a sealed snapshot and a live buffer whose address other code has
    /// already resolved, so it fills that buffer rather than replacing it.
    /// Snapshotting by allocation made the store the largest device allocator in
    /// the decode loop: 679 MB across one measured window on the 0.8B at batch
    /// 4, more than half of everything the loop allocated, in an allocate/free
    /// pair per layer per session per token.
    ///
    /// **The wave path does not use this.** It used to: every wave copied every
    /// layer's state aside so a failure could restore it, ~2 MB and two launches
    /// per layer per wave. A wave now reads one buffer and writes the other, so
    /// the entering state is preserved by not being touched and `commit_wave`
    /// installs the result with a pointer swap.
    pub fn copy_from(&mut self, src: &Self) -> Result<()> {
        self.s.slice_set(&src.s, 0, 0)?;
        self.conv_tail.slice_set(&src.conv_tail, 0, 0)
    }

    /// Overwrite both buffers from host values, in place — a restore's write.
    ///
    /// On CUDA the bytes go straight into the slot's own storage, one upload per
    /// buffer, and nothing else is touched on the device. Restoring through
    /// [`Self::copy_from`] built a device-side source first, so every resume
    /// allocated a whole layer state from the driver to copy it once and drop
    /// it: 3.2 MiB per layer on Flash-Next, outside the reservation.
    pub fn write_host(&mut self, s: &[f32], conv_tail: &[f32]) -> Result<()> {
        match self.s.device() {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                upload_into(&self.s, s)?;
                upload_into(&self.conv_tail, conv_tail)
            }
            dev => {
                let dev = dev.clone();
                self.s
                    .slice_set(&Tensor::from_vec(s.to_vec(), self.s.shape(), &dev)?, 0, 0)?;
                self.conv_tail.slice_set(
                    &Tensor::from_vec(conv_tail.to_vec(), self.conv_tail.shape(), &dev)?,
                    0,
                    0,
                )
            }
        }
    }

    pub fn zeros(dims: &DeltaNetDims, dev: &Device) -> Result<Self> {
        Ok(Self {
            s: Tensor::zeros(
                (dims.n_v_heads, dims.head_dim, dims.head_dim),
                DType::F32,
                dev,
            )?,
            conv_tail: Tensor::zeros((dims.conv_dim(), dims.conv_kernel - 1), DType::F32, dev)?,
        })
    }

    /// Bytes one state's two buffers occupy, in the order [`Self::at`] lays
    /// them out. `(s_bytes, conv_bytes)`.
    pub fn byte_sizes(dims: &DeltaNetDims) -> (usize, usize) {
        let f32_bytes = DType::F32.size_in_bytes();
        (
            dims.n_v_heads * dims.head_dim * dims.head_dim * f32_bytes,
            dims.conv_dim() * (dims.conv_kernel - 1) * f32_bytes,
        )
    }

    /// A state whose two buffers are **views** over memory owned elsewhere —
    /// `s` at `s_ptr`, the conv tail at `conv_ptr`.
    ///
    /// This is how a sequence's recurrent memory lives inside the device
    /// reservation instead of the CUDA pool. It is a per-sequence, fixed-size,
    /// uniform allocation — the same shape as a KV arena's regions or an expert
    /// slot — and allocating it from the pool instead put ~126 MiB per sequence
    /// outside the reservation, which is memory the span was never sized for
    /// and which WDDM demotes to host RAM rather than refusing. On a hybrid
    /// stack that is three quarters of the model's per-sequence memory.
    ///
    /// **Anchored to its owner.** Both buffers hold `anchor`, and so does every
    /// clone, view and re-lease of them, so the memory's owner cannot hand it to
    /// another sequence while anything still reads this state through a handle it
    /// took along the way — a `write_half`, a captured view, a re-lease. The worst a
    /// forgotten handle can then cost is ground not yet reclaimed, never a read of
    /// someone else's state.
    ///
    /// # Safety
    ///
    /// Both pointers must name at least [`Self::byte_sizes`] bytes of correctly
    /// aligned memory that stays live while `anchor` is held, and that nothing
    /// else writes through another alias.
    #[cfg(feature = "cuda")]
    pub unsafe fn at(
        dims: &DeltaNetDims,
        dev: &Device,
        s_ptr: u64,
        conv_ptr: u64,
        anchor: LeaseAnchor,
    ) -> Result<Self> {
        Ok(Self {
            s: Tensor::from_anchored_cuda_ptr(
                s_ptr,
                DType::F32,
                (dims.n_v_heads, dims.head_dim, dims.head_dim),
                dev,
                anchor.clone(),
            )?,
            conv_tail: Tensor::from_anchored_cuda_ptr(
                conv_ptr,
                DType::F32,
                (dims.conv_dim(), dims.conv_kernel - 1),
                dev,
                anchor,
            )?,
        })
    }

    /// Handles onto both buffers, as the buffers a long span WRITES — the
    /// prefill scratch the store hands it, which `commit_wave` exchanges in.
    pub fn write_half(&self) -> DeltaNetOut {
        DeltaNetOut {
            s: self.s.clone(),
            conv_tail: self.conv_tail.clone(),
        }
    }

    /// A write half for a caller carrying **one** state — the single-span entry
    /// points and the reference tests, which have no wave to roll back.
    ///
    /// `s` is the same buffer, which is what those paths have always done and is
    /// safe: the state kernels hold their `S` tile in registers for the whole
    /// sequence, so nothing reads an element after another block wrote it. The
    /// conv tail is **not** safe that way — the prefill conv's blocks for the
    /// first `kwidth − 1` outputs read the entering tail while the block owning
    /// the segment's end writes the advanced one — so it gets a scratch buffer
    /// that [`Self::absorb_solo`] folds back once the launch is behind us.
    pub fn solo_out(&self) -> Result<DeltaNetOut> {
        Ok(DeltaNetOut {
            s: self.s.clone(),
            conv_tail: Tensor::zeros(self.conv_tail.shape(), DType::F32, self.conv_tail.device())?,
        })
    }

    /// Install the advance a [`Self::solo_out`] write half received. `s` shares
    /// its buffer and is already current; the conv tail is the copy back.
    pub fn absorb_solo(&mut self, out: &DeltaNetOut) -> Result<()> {
        self.conv_tail.slice_set(&out.conv_tail, 0, 0)
    }

    /// Allocate the buffers **without zeroing** — for a write half the kernels
    /// fully overwrite before anything reads it.
    ///
    /// Hot-path invariant 6 (`AGENTS.md`): a buffer a kernel stamps end to end
    /// must not be memset first. The zeroed twin costs a full-width device
    /// memset per layer (~2 MB × 30 layers ≈ 63 MiB per fork) on the exact
    /// bytes the next wave overwrites, and forks run ~3× per turn.
    ///
    /// Sound for the prefill scratch: [`commit_wave`] installs `s` and the conv
    /// tail together because a long span's kernels write both into it (the conv
    /// kernels take the entering and advanced tails as two pointers).
    /// [`Self::zeros`] remains the constructor for a state that is genuinely
    /// read at zero — a sequence-start value — and [`Self::solo_out`] keeps its
    /// zeroed scratch tail deliberately.
    ///
    /// [`commit_wave`]: super::state_store::RecurrentStateStore::commit_wave
    pub fn uninit(dims: &DeltaNetDims, dev: &Device) -> Result<Self> {
        Ok(Self {
            s: Tensor::empty(
                (dims.n_v_heads, dims.head_dim, dims.head_dim),
                DType::F32,
                dev,
            )?,
            conv_tail: Tensor::empty((dims.conv_dim(), dims.conv_kernel - 1), DType::F32, dev)?,
        })
    }
}

/// Where a long span WRITES one sequence's advanced state for a layer: the
/// store's prefill scratch, which `commit_wave` exchanges in.
///
/// Both buffers are handles rather than borrows, because the reference paths
/// have a single state and name the same tensors here and in the
/// [`DeltaNetState`] they read — and the kernels take addresses either way.
#[derive(Debug, Clone)]
pub struct DeltaNetOut {
    /// `[n_v_heads, d_v, d_k]`, F32 — the advanced `S`.
    pub s: Tensor,
    /// `[conv_dim, conv_kernel − 1]`, F32 — the advanced conv tail.
    pub conv_tail: Tensor,
}
