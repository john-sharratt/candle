//! Per-domain device bump allocators for the transient tier's intermediates.
//!
//! Inference-loop transients bump-allocate from a span that is never freed. A
//! generation hands out disjoint ranges by advancing a cursor, and the last
//! generation to close rewinds the cursor to zero, so steady-state waves make
//! no allocator calls. Safety rests on three things: a `BumpRange` borrows its
//! generation, a reset is refused while any generation is live, and for the
//! persistence domain a fence orders the device before the cursor moves. Each
//! domain is a disjoint range of the device reservation; nested generations
//! release only to their own mark.
//!
//! - `wave_domain`: the wave domain's per-phase arenas and tier state.
//! - `arena`: one domain's bump cursor and its three kinds of backing.
//! - `wave`: opening a layer phase's generation and settling its tier.
//! - `plan`: pricing and reserving the tier for each forward.
//! - `window`: the arena window that lets an arena be created between forwards.
//! - `turn`: serialises generations on a domain shared by several threads.
//! - `forward_scratch`: the forward-wide span launch scratch is carved from.
//! - `guest_arena`: recycling a guest pipeline's stages on its claimed ground.
//! - `tests_guest`, `tests_plan`, `tests_provenance`, `tests_turn`, `tests_wave`, `tests_window`: tests.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{current, ThreadId};

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::cuda_backend::wave_provenance::{SpanState, WaveCarve, WaveTicket, ASIDE_ARENA};
use candle::cuda_backend::{
    close_scratch_scope, install_scope_carve, open_scratch_scope, scratch_reserve, StreamLease,
};
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::region_pool::{carve_persist, place_transient, release_transient};
use super::wave_census::{self, Carve};
use super::wave_plan::{align_phase_plan, launch_plan, phase_offsets, tier_footprint, LayerPhase};
use super::wave_spans::{WAVE_ATTN_BYTES, WAVE_FFN_BYTES, WAVE_FORWARD_BYTES};

/// A range handed out by a [`Generation`].
///
/// Deliberately **not** RAII: a bump range is freed by its generation's reset,
/// never individually, so a `Drop` impl would be a lie. What bounds it instead
/// is `'w` — a borrow of the [`Generation`] that handed it out. The cursor
/// cannot rewind while that guard is alive, so "valid for `'w`" and "valid
/// until the generation resets" are the same statement, and the borrow checker
/// can hold the compiler to it.
#[derive(Debug, Clone, Copy)]
pub struct BumpRange<'w> {
    /// Device address of the range's first byte.
    pub ptr: u64,
    /// Bytes reserved.
    pub len: usize,
    /// Covariant in `'w`: this range is valid for at most as long as the guard
    /// it came from.
    wave: PhantomData<&'w ()>,
}

/// How a domain makes a reset safe.
///
/// Resetting hands memory back out while kernels may still be reading it. What
/// prevents that differs by domain, and the difference is worth naming: one
/// costs a pipeline stall and the other costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reclaim {
    /// Fence the stream before the cursor moves.
    ///
    /// For domains whose ranges are read from one stream and overwritten from
    /// another — the persistence thread stages on its copy stream while the
    /// compute stream runs — where nothing else orders the two.
    Fence,
    /// Rely on stream order alone.
    ///
    /// For domains that are double-buffered on a single stream. By the time a
    /// half is handed out again, an entire other wave's work sits between the
    /// reads and the writes *on the same stream*, and same-stream launches
    /// complete in issue order. A fence would buy nothing and cost a full
    /// device sync on the forward's critical path — which is precisely the
    /// wave-path stall §3.6 exists to remove.
    StreamOrdered,
}

struct Inner {
    base: u64,
    capacity: usize,
    /// Bytes handed out since the last reset.
    cursor: usize,
    /// Bytes held at the **top** of the span, growing down toward `cursor`.
    ///
    /// A guest stage's result: it has to outlive the stage that made it, and
    /// the stage's own range is released to its mark the moment it closes. So
    /// it is carried at the other end, owned by the enclosing generation and
    /// released with it ([`guest_stage`]). Always zero outside a guest arena.
    carry: usize,
    /// Live [`Generation`] guards. The cursor cannot move while this is > 0.
    live: usize,
    /// Whether anything has been handed out since the last reset.
    dirty: bool,
    /// High-water mark of `cursor`, for the watermark that sizes the span.
    peak: usize,
    /// The largest generation the census has already itemised, **per carve
    /// count**.
    ///
    /// Kept apart from `peak` because the two answer different questions: `peak`
    /// is what the span must hold, this is what has already been reported.
    ///
    /// Keyed by carve count rather than held as one number because an arena
    /// serves more than one chain, and the widest is not the only one worth
    /// seeing. The attention span carries a twelve-carve prefill chain and a
    /// narrower decode chain that reaches `o_proj` through a different operand;
    /// a single high-water mark would only ever print the first, and the plan
    /// has to upper-bound both. The count is what distinguishes them, and it
    /// takes a handful of distinct values, so this stays small.
    reported_peak: HashMap<usize, usize>,
    /// Ranges carved in the current generation, when
    /// [`super::wave_census::enabled`]. Empty in every ordinary run.
    census: Vec<Carve>,
    /// Bumped every time the cursor rewinds.
    ///
    /// A [`WaveTicket`] carries the epoch it was minted in, so a ticket from a
    /// closed generation resolves to nothing rather than carving from whatever
    /// occupies the span now. `LiveTensor<'w>` already makes that unreachable at
    /// compile time; this is the backstop for the `unsafe` constructors that
    /// mint leases from raw pointers, where the compiler has nothing to check.
    epoch: u64,
    stream: Arc<CudaStream>,
    reclaim: Reclaim,
}

/// A live claim on a [`BumpArena`]'s current contents.
///
/// While one exists the cursor cannot reset, so every range handed out since
/// the last reset stays valid. The last guard to drop fences the domain's
/// stream and resets.
pub struct Generation {
    inner: Arc<Mutex<Inner>>,
    name: &'static str,
    /// Coordinates of the arena this guard is open on, so the generation can
    /// mint a [`WaveTicket`] for the storages allocated inside it.
    domain: u32,
    arena: u32,
    /// The cursor as this generation opened, so a **nested** one releases back
    /// to where it began.
    ///
    /// Generations are scope guards and therefore strictly LIFO, which is the
    /// stack discipline a bump wants: the outermost rewinds to zero and every
    /// one inside it is a mark/release pair. Without this an inner generation
    /// was a no-op — `live` was still positive when it dropped, so nothing
    /// moved — and staging a decoder's blocks inside a per-tile generation
    /// bounded nothing at all while looking like it did.
    ///
    /// Releasing is only sound because a stage copies its result out before it
    /// closes ([`guest_stage`]); anything still living in the released range
    /// when the mark unwinds would be handed to the next carve.
    entry_cursor: usize,
    /// The carry as this generation opened, released back to the same way —
    /// what a stage nested inside this one carried out belongs to this one.
    entry_carry: usize,
    /// This thread's hold on a shared domain. The last field, so it is given
    /// back only after [`Drop::drop`] has fenced and rewound the cursor. Held
    /// for its drop alone.
    _turn: Option<TurnHeld>,
}

impl Generation {
    /// The coordinate an op allocating from this generation should carry.
    ///
    /// Stamped with the arena's current epoch, so the ticket stops resolving the
    /// moment this generation's cursor rewinds.
    pub fn ticket(&self) -> WaveTicket {
        WaveTicket {
            domain: self.domain,
            arena: self.arena,
            epoch: self.inner.lock().unwrap().epoch,
        }
    }

    /// Bump `len` bytes, aligned to `align`, from the half this guard pins.
    ///
    /// The returned range borrows `self`, so it cannot outlive the guard whose
    /// drop resets the cursor underneath it. That is the whole safety argument
    /// for wave buffers, moved from a runtime count to the type system:
    /// a caller who holds a `BumpRange<'w>` provably still holds the guard.
    ///
    /// Allocating *through the guard* also removes an ambiguity the arena
    /// cannot resolve. A bump range has to come from a specific half, and with
    /// two waves in flight there is no way to tell from the outside which half
    /// a given call site belongs to — the older wave's late allocations would
    /// land in the newer wave's half and be freed by its reset. Here the guard
    /// names its own half, so the question never arises.
    ///
    /// A range cannot outlive the guard that reclaims it:
    ///
    /// ```compile_fail
    /// # use candle_nn::kv_cache::{begin_wave, BumpRange, LayerPhase};
    /// # fn f(stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>)
    /// #     -> candle::Result<()> {
    /// let range = {
    ///     let wave = begin_wave(stream, LayerPhase::Attention)?;
    ///     wave.alloc(1024, 256)?
    /// };
    /// println!("{}", range.ptr);
    /// # Ok(()) }
    /// ```
    ///
    /// while using it inside the guard's scope is fine — the control that keeps
    /// the case above from failing for an unrelated reason:
    ///
    /// ```no_run
    /// # use candle_nn::kv_cache::{begin_wave, LayerPhase};
    /// # fn f(stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>)
    /// #     -> candle::Result<()> {
    /// let wave = begin_wave(stream, LayerPhase::Attention)?;
    /// let range = wave.alloc(1024, 256)?;
    /// println!("{}", range.ptr);
    /// # Ok(()) }
    /// ```
    pub fn alloc(&self, len: usize, align: usize) -> Result<BumpRange<'_>> {
        bump(&self.inner, self.name, len, align)
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        let (should_reset, stream, reclaim, closing) = {
            let mut inner = self.inner.lock().unwrap();
            inner.live -= 1;
            // A nested generation releases to its mark and stops here. It does
            // not touch `dirty` or `epoch`: the outer generation is still open,
            // its tickets must keep resolving, and the ranges below the mark are
            // still live. Only the outermost rewinds the arena.
            //
            // **Guest arenas only.** Releasing early hands the range back to the
            // next carve, which is sound exactly where every value that outlives
            // the scope has been copied out of it first — the contract
            // [`guest_stage`] keeps, and the reason a guest stage may nest. The
            // wave path has no such contract: its generations exist to bound a
            // layer phase, tensors cross between them freely, and reclaiming a
            // nested one's bytes there hands live attention state to the next
            // allocation. That shows up as an unspecified launch failure in a
            // test that has nothing to do with the scope that moved the cursor.
            //
            // No fence, unlike the reset below. The reset needs one because the
            // ranges it frees may be read by a domain that does not share this
            // stream; a nested release stays inside one generation on one
            // stream, where the next kernel to write a released range is
            // ordered behind the kernel that read it. Adding a synchronise here
            // would put one in the innermost scope of the decoder — per resnet,
            // per block, per tile.
            if inner.live > 0 && self.arena == GUEST_ARENA {
                if inner.cursor + inner.carry > inner.peak {
                    inner.peak = inner.cursor + inner.carry;
                }
                inner.cursor = self.entry_cursor;
                inner.carry = self.entry_carry;
                // No tier reference to give back: a guest arena stands in the
                // guest's own claimed ground, so this generation never took one.
                return;
            }
            (
                inner.live == 0 && inner.dirty,
                inner.stream.clone(),
                inner.reclaim,
                inner.live == 0,
            )
        };
        // The forward-wide span's last generation closing ends the forward's
        // scratch scope — before the cursor rewinds below, so no launch is handed
        // a block in ground about to be reused. Read under the lock that counted
        // the drop, so a generation opening in between cannot be mistaken for it.
        if closing && self.arena == FORWARD_ARENA && self.domain != NOT_A_WAVE {
            close_forward_scratch(self.domain as usize);
        }
        // The tier's reference count moves on **every** drop, not only the ones
        // that reset a cursor: a clean generation still held the tier open.
        //
        // A guest generation is excluded for the same reason the persistence
        // domain is, by a different route: its arena stands in the guest's own
        // claimed ground rather than in the tier, so there is no tier reference
        // for it to have taken and none to give back. Releasing one here would
        // decrement a count this generation never incremented.
        let release = |g: &Generation| {
            if g.domain != NOT_A_WAVE && g.arena != GUEST_ARENA {
                release_if_last(g.domain as usize);
            }
        };
        if !should_reset {
            release(self);
            return;
        }
        // Fence before the cursor moves: the ranges about to become reusable
        // may still be under a kernel's read, and for this domain nothing else
        // orders that read against the next writer.
        if reclaim == Reclaim::Fence {
            if let Err(e) = stream.synchronize() {
                // Refuse to reset rather than hand out memory the GPU may still
                // be reading. The span leaks for one generation; the
                // alternative is a data race (principle 7).
                log::error!(
                    "{}: transient reset skipped — stream fence failed: {e:?}",
                    self.name
                );
                release(self);
                return;
            }
        }
        let mut inner = self.inner.lock().unwrap();
        // The generation is closing, so its cursor is this layer phase's whole
        // cost. Itemise it if it is the worst yet for its chain — see
        // `wave_census`.
        if wave_census::enabled() {
            let (cursor, carves, capacity) = (inner.cursor, inner.census.len(), inner.capacity);
            let seen = inner.reported_peak.entry(carves).or_insert(0);
            if cursor > *seen {
                *seen = cursor;
                wave_census::report(self.name, cursor, capacity, &inner.census);
            }
        }
        inner.census.clear();
        inner.cursor = 0;
        inner.carry = 0;
        inner.dirty = false;
        inner.epoch += 1;
        drop(inner);
        release(self);
    }
}

/// Arena coordinate for a co-resident guest's activations.
///
/// A wave domain's arenas are indexed by [`LayerPhase`] — 0, 1, 2 — so this is
/// the first coordinate that names no phase, and `WaveDomain::arenas.get(3)`
/// already answers `None` for it. That is what makes it safe to overload the
/// field rather than widen the ticket: a guest ticket cannot be mistaken for a
/// wave one by any code that does not know about guests, and the miss it
/// produces there is the ordinary "allocate from the pool" answer.
/// Re-exported rather than defined twice: a ticket naming it is minted at the
/// guest's *placement* sites, which sit below this crate, so the constant lives
/// beside [`WaveTicket`] and the registry that answers it lives here.
pub use candle::wave_provenance::{GUEST_ANY_EPOCH, GUEST_ARENA};

pub use super::backing::SPANS_OUTGROWN_MARKER;

mod arena;
mod forward_scratch;
mod guest_arena;
mod plan;
#[cfg(test)]
mod tests_guest;
#[cfg(test)]
mod tests_plan;
#[cfg(test)]
mod tests_provenance;
#[cfg(test)]
mod tests_turn;
#[cfg(test)]
mod tests_wave;
#[cfg(test)]
mod tests_window;
mod turn;
mod wave;
mod wave_domain;
mod window;

pub use arena::*;
use forward_scratch::*;
pub use guest_arena::*;
pub use plan::*;
use turn::*;
pub use wave::*;
pub(crate) use wave_domain::*;
pub use window::*;
