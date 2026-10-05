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

/// The arena index a co-resident guest's own bump answers to.
///
/// Not one of the wave domain's per-phase arenas — a guest is not a layer — so
/// it is resolved from its own registry. Defined here rather than beside that
/// registry because a [`WaveTicket`] naming it is minted at the guest's
/// *placement* sites, which are further down the stack than the owner.
pub const GUEST_ARENA: u32 = 3;

/// The epoch of a ticket that means "whichever guest generation is open now".
///
/// # Why a guest's weights need this and a wave's operands do not
///
/// An epoch exists so a ticket cannot outlive the generation it came from: a
/// generation bumps its epoch when it rewinds, and a stale ticket then resolves
/// to nothing instead of carving from whatever occupies that span next. That is
/// exactly right for an operand, whose life *is* one generation.
///
/// A guest's weights are not operands. They are placed once at load, live for
/// the whole drain, and are read by every stage — the encode, each denoise step,
/// each decode tile — each of which is its own generation. A weight carrying a
/// real epoch would route the first stage's activations into the arena and
/// nothing after it, which is worse than never routing at all because it looks
/// like it works.
///
/// So a weight carries a routing *seed* rather than a provenance: it says
/// "allocate from the guest arena's open generation", and resolves to the pool
/// when none is open. It never names a range and so can never alias one.
pub const GUEST_ANY_EPOCH: u64 = u64::MAX;

impl WaveTicket {
    /// The routing seed a guest stamps on memory it placed in its own ground.
    ///
    /// `domain` is the stream ordinal, as for any ticket. See
    /// [`GUEST_ANY_EPOCH`] for why the epoch is a sentinel.
    pub fn guest(domain: u32) -> Self {
        Self {
            domain,
            arena: GUEST_ARENA,
            epoch: GUEST_ANY_EPOCH,
        }
    }
}

/// What the arena a ticket names did with a request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaveCarve {
    /// Served, at this device address.
    Carved(u64),
    /// The generation the ticket names has closed — or there is no such arena,
    /// or no resolver at all. The tensor descends from memory a finished phase
    /// owned, so its new data has no phase to live in and the pool is its home.
    Closed,
    /// The generation is open and its span cannot hold the request. Not a
    /// pool allocation waiting to happen: the wave plan priced this phase
    /// short of what it carves, and the allocation is refused so the site that
    /// overran is named here rather than found later as memory the reservation
    /// never accounted for.
    Exhausted,
}

/// Carve `bytes` (aligned to the third argument) from the arena a ticket names.
///
/// Installed by candle-nn, which owns the arenas.
pub type WaveAllocFn = fn(WaveTicket, usize, usize) -> WaveCarve;

static WAVE_ALLOC: OnceLock<WaveAllocFn> = OnceLock::new();

/// Register the resolver for [`WaveTicket`]s. Idempotent; later calls are
/// ignored, since there is one arena owner per process.
pub fn install_wave_allocator(f: WaveAllocFn) {
    let _ = WAVE_ALLOC.set(f);
}

/// Carve `bytes` (aligned to `align`) from the arena `ticket` names.
///
/// A closed generation — or a process that never installed a resolver — is
/// [`WaveCarve::Closed`], and the caller allocates from the pool. An open one
/// that cannot hold the request is [`WaveCarve::Exhausted`], which every caller
/// turns into an error: see [`exhausted`].
pub fn wave_alloc(ticket: WaveTicket, bytes: usize, align: usize) -> WaveCarve {
    match WAVE_ALLOC.get() {
        Some(f) => f(ticket, bytes, align),
        None => WaveCarve::Closed,
    }
}

/// The error an open generation's overrun becomes.
///
/// **There is no fallback for it.** A request that overruns an open phase is a
/// buffer the wave plan did not price, and serving it from the pool is how the
/// pool came to hold gigabytes of forward-pass memory the reservation never
/// accounted for — until the card ran out with nothing pointing at the cause.
///
/// The message carries the stack that asked: the generic op that allocated is
/// never the answer, the model code above it is. Captured only here, on the
/// error path, so it costs nothing on any allocation that succeeds.
#[track_caller]
pub fn exhausted(ticket: WaveTicket, bytes: usize) -> crate::Error {
    let at = std::panic::Location::caller();
    let stack = std::backtrace::Backtrace::force_capture().to_string();
    let asked_by: Vec<&str> = stack
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.starts_with("at ")
                && (l.contains("candle_transformers::") || l.contains("candle_conversation::"))
        })
        .take(8)
        .collect();
    crate::Error::Msg(format!(
        "wave arena exhausted: {bytes} B asked at {}:{} of arena {} (domain {}, epoch {}), \
         past what the wave plan priced for this phase — price the buffer, do not \
         allocate it elsewhere. Asked by: {}",
        at.file(),
        at.line(),
        ticket.arena,
        ticket.domain,
        ticket.epoch,
        asked_by.join(" <- "),
    ))
}

/// Why an allocation that inherited from an operand went to the pool.
///
/// **Neither is an arena that ran out of room** — that is an error
/// ([`WaveCarve::Exhausted`]), never a pool allocation. Both of these are the
/// operand's provenance: `NoTicket` is a tensor with no wave backing at all, and
/// everything derived from it inherits the pool, so the fix is at that root,
/// possibly many frames above the site that shows up in a report. `Closed` is a
/// tensor whose phase has finished — a value read after its generation, whose
/// derived data has no phase left to live in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ArenaDecline {
    /// The origin carried no wave ticket — a `Foreign` lease, an owned pool
    /// allocation, or a tensor that never had provenance to begin with.
    NoTicket,
    /// The origin's generation has closed.
    ///
    /// Also covers a process with no resolver installed at all, which in
    /// practice means before candle-nn registers one at startup — a window with
    /// no waves in it, so it contributes nothing to a steady-state reading.
    Closed,
}

/// `[NoTicket, Closed]` — calls, then bytes, indexed by `ArenaDecline as
/// usize`. Relaxed throughout: these are counters read by a report, never a
/// value anything orders against.
static DECLINE_CALLS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static DECLINE_BYTES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Carve from `from`'s arena, recording why if the pool is the home instead.
///
/// The single decision point for "arena or pool", so the accounting cannot drift
/// from the behaviour: a caller that carves without asking here is a caller that
/// does not appear in the totals. `Ok(None)` sends the caller to the pool; an
/// open generation that cannot hold the request is an error ([`exhausted`]).
#[track_caller]
pub fn wave_alloc_attributed(
    from: Option<WaveTicket>,
    bytes: usize,
    align: usize,
) -> crate::Result<Option<u64>> {
    let Some(ticket) = from else {
        record_decline(ArenaDecline::NoTicket, bytes);
        return Ok(None);
    };
    match wave_alloc(ticket, bytes, align) {
        WaveCarve::Carved(ptr) => Ok(Some(ptr)),
        WaveCarve::Closed => {
            record_decline(ArenaDecline::Closed, bytes);
            Ok(None)
        }
        WaveCarve::Exhausted => Err(exhausted(ticket, bytes)),
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
    closed: (u64, u64),
}

impl DeclineSnapshot {
    /// Read both counters now.
    pub fn now() -> Self {
        Self {
            no_ticket: arena_declines(ArenaDecline::NoTicket),
            closed: arena_declines(ArenaDecline::Closed),
        }
    }

    /// `(no_ticket_bytes, closed_bytes)` accumulated since `self` was taken.
    pub fn bytes_since(&self) -> (u64, u64) {
        let now = Self::now();
        (
            now.no_ticket.1.saturating_sub(self.no_ticket.1),
            now.closed.1.saturating_sub(self.closed.1),
        )
    }
}

/// The last completed wave's decline bytes, `(no_ticket, closed)`.
static LAST_WAVE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Publish one wave's decline delta. Called by the scheduler at wave end.
pub fn publish_wave_declines(no_ticket_bytes: u64, closed_bytes: u64) {
    LAST_WAVE[0].store(no_ticket_bytes, Ordering::Relaxed);
    LAST_WAVE[1].store(closed_bytes, Ordering::Relaxed);
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
        arena_declines, exhausted, reset_arena_declines, wave_alloc_attributed, ArenaDecline,
        WaveTicket,
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
        assert_eq!(wave_alloc_attributed(None, 4096, 256).unwrap(), None);
        let (calls, bytes) = arena_declines(ArenaDecline::NoTicket);
        assert_eq!((calls, bytes), (1, 4096));
        assert_eq!(
            arena_declines(ArenaDecline::Closed),
            (0, 0),
            "a missing ticket is not a closed generation — the two have \
             different roots and must not share a counter",
        );
    }

    /// A ticket whose generation is gone is charged to `Closed`.
    ///
    /// With no resolver installed every ticket reads as closed, which is
    /// exactly the path a finished generation takes.
    #[test]
    fn a_closed_ticket_is_charged_to_closed() {
        reset_arena_declines();
        let ticket = WaveTicket {
            domain: 0,
            arena: 0,
            epoch: 0,
        };
        assert_eq!(
            wave_alloc_attributed(Some(ticket), 8192, 256).unwrap(),
            None
        );
        assert_eq!(arena_declines(ArenaDecline::Closed), (1, 8192));
        assert_eq!(arena_declines(ArenaDecline::NoTicket), (0, 0));
    }

    /// An exhausted open generation is an error naming the request, never a
    /// pool allocation.
    #[test]
    fn an_exhausted_generation_is_refused() {
        let ticket = WaveTicket {
            domain: 3,
            arena: 1,
            epoch: 9,
        };
        let msg = exhausted(ticket, 4096).to_string();
        assert!(msg.contains("wave arena exhausted: 4096 B"), "{msg}");
        assert!(msg.contains("arena 1 (domain 3, epoch 9)"), "{msg}");
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
