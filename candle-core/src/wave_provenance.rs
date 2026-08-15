//! Where a leased storage came from, so an op's output can be allocated from
//! the same place.
//!
//! # The rule
//!
//! An op writes its output into whichever arena its *operand* came from. That is
//! the whole mechanism: there is no ambient "current wave" to consult and
//! nothing to keep in sync at a layer boundary, because the answer travels with
//! the data. Two waves in flight stay separate for free — a value carries its
//! own arena, so a kernel reading it cannot land in the other one.
//!
//! It also makes the lifetime honest. `LiveTensor<'w>` already propagates `'w`
//! from operand to result (`from_storage` derives it from the operand graph), so
//! before this existed `'w` was a safe over-approximation that bought nothing:
//! the type said "may be wave-backed" while the allocation always came from the
//! pool. Once the allocation follows the operand, the type is *true*, and the
//! borrow checker is what stops a wave-backed value outliving its generation.
//!
//! # Why a ticket and a callback rather than a pointer to the arena
//!
//! The arenas live in `candle-nn`, which depends on this crate, so this crate
//! cannot name them. A [`WaveTicket`] is an opaque, `Copy` coordinate that
//! candle-nn can resolve, and [`install_wave_allocator`] is how it hands over
//! the resolver. Keeping the ticket `Copy` is what lets
//! [`crate::cuda_backend::Backing`] stay `Copy` and avoids an `Arc` on every
//! storage.
//!
//! # Why a stray free cannot corrupt anything
//!
//! A wave range is carved from the device's **VMM reservation**
//! (`candle_nn::kv_cache::chunked::region_pool::carve_transient`), not from the
//! stream-ordered pool. So if a `CudaSlice` over wave memory is ever dropped
//! bare — the window between allocating an output and wrapping it in a
//! `Backing::Lease` storage, which a `?` on a failed kernel launch can hit —
//! the resulting `cuMemFreeAsync` names memory the pool never allocated. The
//! driver rejects it and cudarc records the error; nothing is freed. The hazard
//! is structural, not something each call site has to be careful about.

use std::sync::OnceLock;

/// Which wave arena a leased storage was allocated from.
///
/// Deliberately a plain `Copy` coordinate rather than a handle: it rides on
/// every [`crate::cuda_backend::Backing::Lease`], so it has to be cheap to copy
/// and free of allocation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct WaveTicket {
    /// The transient domain — the CUDA stream ordinal that owns the arenas.
    pub domain: u32,
    /// Which arena within the domain (one per layer phase).
    pub arena: u32,
    /// The generation open when this range was handed out.
    ///
    /// A generation bumps its epoch when its cursor rewinds, so a ticket from a
    /// closed generation resolves to `None` instead of carving from whatever
    /// occupies that span now. `LiveTensor<'w>` already makes a stale ticket
    /// unreachable at compile time; this is the runtime backstop for the
    /// `unsafe` constructors that mint leases from raw pointers.
    pub epoch: u64,
}

/// Carve `bytes` from the arena `ticket` names, or `None` if that generation has
/// closed or the arena has no room.
///
/// Installed by candle-nn, which owns the arenas.
pub type WaveAllocFn = fn(WaveTicket, usize, usize) -> Option<u64>;

static WAVE_ALLOC: OnceLock<WaveAllocFn> = OnceLock::new();

/// Register the resolver for [`WaveTicket`]s. Idempotent; later calls are
/// ignored, since there is one arena owner per process.
pub fn install_wave_allocator(f: WaveAllocFn) {
    let _ = WAVE_ALLOC.set(f);
}

/// Carve `bytes` (aligned to `align`) from the arena `ticket` names.
///
/// `None` means "allocate from the pool instead" and is a normal outcome, not an
/// error: it is what a closed generation, a full arena, or a process that never
/// installed a resolver all return. The caller always has the pool to fall back
/// on, so a miss costs an allocation rather than a failure.
pub fn wave_alloc(ticket: WaveTicket, bytes: usize, align: usize) -> Option<u64> {
    WAVE_ALLOC.get()?(ticket, bytes, align)
}

/// What owns the memory behind a [`crate::cuda_backend::Backing::Lease`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum LeaseOrigin {
    /// Memory owned by something with no allocator to inherit — a KV arena
    /// slot, a pinned staging buffer, a caller-supplied pointer. An op reading
    /// one of these allocates its output from the pool, because there is
    /// nowhere else for it to come from and the arena is not a scratch space.
    Foreign,
    /// A wave generation. An op reading this allocates its output from the same
    /// generation, which is what makes the `'w` on the result true rather than
    /// merely permitted.
    Wave(WaveTicket),
}

impl LeaseOrigin {
    /// The ticket to allocate an inherited output from, if there is one.
    pub fn ticket(&self) -> Option<WaveTicket> {
        match self {
            Self::Foreign => None,
            Self::Wave(t) => Some(*t),
        }
    }
}
