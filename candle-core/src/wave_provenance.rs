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

use std::sync::atomic::{AtomicU64, Ordering};
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

/// Why an allocation that *could* have been served from a wave arena was not.
///
/// **The two have opposite fixes, and telling them apart is the whole point.**
/// `NoTicket` is a provenance break: something upstream produced a tensor with
/// no wave backing, and everything derived from it inherits the pool — so the
/// fix is at that root, possibly many frames above the site that shows up in a
/// report. `ArenaFull` is a sizing problem: provenance is intact and the arena
/// simply had no room, so the fix is the arena's width and nothing about the
/// call site is wrong.
///
/// Guessing between them was costing real time. The fallback in
/// [`crate::cuda_backend::alloc_inheriting`] is silent and both paths land on
/// `CudaDevice::alloc`, so a forbidden-allocation report shows an identical row
/// either way — a site that has lost its provenance and a site whose arena
/// overflowed are the same line of output.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ArenaDecline {
    /// The origin carried no wave ticket — a `Foreign` lease, an owned pool
    /// allocation, or a tensor that never had provenance to begin with.
    NoTicket,
    /// The origin had a ticket and the arena refused: no room, or a generation
    /// that has since closed.
    ///
    /// Also covers a process with no resolver installed at all, which in
    /// practice means before candle-nn registers one at startup — a window with
    /// no waves in it, so it contributes nothing to a steady-state reading.
    ArenaFull,
}

/// `[NoTicket, ArenaFull]` — calls, then bytes, indexed by `ArenaDecline as
/// usize`. Relaxed throughout: these are counters read by a report, never a
/// value anything orders against.
static DECLINE_CALLS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static DECLINE_BYTES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Carve from `from`'s arena, recording why if that is not possible.
///
/// The single decision point for "arena or pool", so the accounting cannot drift
/// from the behaviour: a caller that carves without asking here is a caller that
/// does not appear in the totals.
pub fn wave_alloc_attributed(from: Option<WaveTicket>, bytes: usize, align: usize) -> Option<u64> {
    let Some(ticket) = from else {
        record_decline(ArenaDecline::NoTicket, bytes);
        return None;
    };
    match wave_alloc(ticket, bytes, align) {
        Some(ptr) => Some(ptr),
        None => {
            record_decline(ArenaDecline::ArenaFull, bytes);
            None
        }
    }
}

fn record_decline(why: ArenaDecline, bytes: usize) {
    let i = why as usize;
    DECLINE_CALLS[i].fetch_add(1, Ordering::Relaxed);
    DECLINE_BYTES[i].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Cumulative `(calls, bytes)` for one decline reason since process start.
pub fn arena_declines(why: ArenaDecline) -> (u64, u64) {
    let i = why as usize;
    (
        DECLINE_CALLS[i].load(Ordering::Relaxed),
        DECLINE_BYTES[i].load(Ordering::Relaxed),
    )
}

/// Zero both counters.
///
/// Test-only: production measures an interval with [`DeclineSnapshot`], which
/// subtracts two readings instead and so cannot lose a count to the persistence
/// thread racing between the zeroing and the read.
#[cfg(test)]
pub fn reset_arena_declines() {
    for i in 0..2 {
        DECLINE_CALLS[i].store(0, Ordering::Relaxed);
        DECLINE_BYTES[i].store(0, Ordering::Relaxed);
    }
}

/// A reading of both counters, for measuring an interval by subtraction.
///
/// **The lifetime totals do not mean what they look like they mean.** They count
/// every declined allocation in the process, and most declines are by design: an
/// op on the residual stream has no ticket to inherit because the residual
/// crosses layers and belongs on the pool, model loading has no wave at all, and
/// neither is a defect. Read cumulatively, `NoTicket` is dominated by exactly
/// those and says nothing about whether provenance is broken.
///
/// What answers that question is the delta across ONE WAVE, where every
/// allocation should be inheriting. Monotonic counters and a subtraction, rather
/// than a reset, so a concurrent persistence thread cannot lose a count between
/// the zeroing and the read.
#[derive(Clone, Copy, Debug)]
pub struct DeclineSnapshot {
    no_ticket: (u64, u64),
    arena_full: (u64, u64),
}

impl DeclineSnapshot {
    /// Read both counters now.
    pub fn now() -> Self {
        Self {
            no_ticket: arena_declines(ArenaDecline::NoTicket),
            arena_full: arena_declines(ArenaDecline::ArenaFull),
        }
    }

    /// `(no_ticket_bytes, arena_full_bytes)` accumulated since `self` was taken.
    pub fn bytes_since(&self) -> (u64, u64) {
        let now = Self::now();
        (
            now.no_ticket.1.saturating_sub(self.no_ticket.1),
            now.arena_full.1.saturating_sub(self.arena_full.1),
        )
    }
}

/// The last completed wave's decline bytes, `(no_ticket, arena_full)`.
static LAST_WAVE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Publish one wave's decline delta. Called by the scheduler at wave end.
pub fn publish_wave_declines(no_ticket_bytes: u64, arena_full_bytes: u64) {
    LAST_WAVE[0].store(no_ticket_bytes, Ordering::Relaxed);
    LAST_WAVE[1].store(arena_full_bytes, Ordering::Relaxed);
}

/// The last wave's decline bytes — the figure the memory report should show,
/// since it is the one scoped to a window where inheriting is expected.
pub fn last_wave_declines() -> (u64, u64) {
    (
        LAST_WAVE[0].load(Ordering::Relaxed),
        LAST_WAVE[1].load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod decline_tests {
    use super::{
        arena_declines, reset_arena_declines, wave_alloc_attributed, ArenaDecline, WaveTicket,
    };

    /// A ticketless origin is charged to `NoTicket`, with its bytes.
    ///
    /// The counters are process-global, so this test resets first and asserts on
    /// the delta it creates rather than on absolutes. It cannot run beside
    /// another test that allocates — there is none in this crate that installs a
    /// resolver, and the whole point of the split is that it needs no device.
    #[test]
    fn a_ticketless_origin_is_charged_to_no_ticket() {
        reset_arena_declines();
        assert_eq!(wave_alloc_attributed(None, 4096, 256), None);
        let (calls, bytes) = arena_declines(ArenaDecline::NoTicket);
        assert_eq!((calls, bytes), (1, 4096));
        assert_eq!(
            arena_declines(ArenaDecline::ArenaFull),
            (0, 0),
            "a missing ticket is not an arena that refused — the two have \
             opposite fixes and must not share a counter",
        );
    }

    /// A ticket whose arena will not serve it is charged to `ArenaFull`.
    ///
    /// With no resolver installed every ticket declines, which is exactly the
    /// path a closed generation or a full arena takes.
    #[test]
    fn a_refused_ticket_is_charged_to_arena_full() {
        reset_arena_declines();
        let ticket = WaveTicket {
            domain: 0,
            arena: 0,
            epoch: 0,
        };
        assert_eq!(wave_alloc_attributed(Some(ticket), 8192, 256), None);
        assert_eq!(arena_declines(ArenaDecline::ArenaFull), (1, 8192));
        assert_eq!(arena_declines(ArenaDecline::NoTicket), (0, 0));
    }
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
