//! Advisory counter: is a hot→warm migrate copying residences right now?
//!
//! Used to defer work that would duplicate a migrate's effort — the scheduler
//! postpones a section quantize while a migrate is converting the same
//! residences, so the two don't both do it. That is a *work* decision, not a
//! safety one, and nothing here provides mutual exclusion.
//!
//! # What used to live here, and why it doesn't
//!
//! This file was an arena-topology `RwLock`: shared for operations that
//! captured raw arena base pointers (the persistence thread's migrate, the
//! scheduler's elevate), exclusive for operations that invalidated them (arena
//! free, defrag relocate, arena-vector truncate, `cuMemPoolTrimTo`). A migrate
//! built a per-head base-pointer table, uploaded it, and launched a kernel that
//! dereferenced those pointers — all with no storage lock held — while the
//! scheduler thread was free to unmap the memory underneath it. The lock made
//! the two exclusive, at the cost of a process-global read acquisition on every
//! migrate and every elevate.
//!
//! Two independent changes retired it:
//!
//! - **Nothing invalidates a base pointer any more.** A region of the
//!   reservation is mapped once and stays mapped at the same address for the
//!   process lifetime; "freeing" an arena moves its region between two lists.
//!   Defrag relocation is gone, the arena vector is not truncated, and the pool
//!   trim went with the pool's KV. The ordering that *is* still required — not
//!   re-tenanting a region while an earlier kernel may still be reading it —
//!   belongs to whoever re-tenants, and lives in `region_pool::claim_region`.
//! - **The table stopped being dense over storage.** It is sized from the job
//!   list now, so every pointer in it comes from a gid the caller has pinned.
//!   Even under the old allocator that would have made the neighbour-arena
//!   hazard unreachable; the pin already protected every arena the kernel could
//!   address.
//!
//! `docs/archived/arena_unification.md` §5 (audit A4).
//!
//! # What a compaction is held off from, and what it merely steps around
//!
//! A migrate **writes** device slots for a short while: the quantize claims its
//! destinations and fills their records, and the convert writes their bytes. A
//! pass that copied one of those slots before the write landed would hand its
//! holders a copy of nothing, so for that phase the migrate holds chunk
//! locations ([`CHUNK_LOCATIONS`]) and a pass is refused. It settles the hold
//! ([`MigrateFlight::settle`]) once the writes are fenced.
//!
//! The rest of the group — the DtoH to warm, which is nine tenths of it, and the
//! install — only **reads** slots it holds by its own handles. A pass may run
//! beside that: nothing frees a slot anyone still names, so a holder the pass
//! does not reach costs only the reclaim, never a byte. The pass steps around
//! the group's chunks instead ([`HeldChunks`]), because moving them would
//! claim destinations no holder it can reach would take up.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use ahash::AHashMap;

use super::types::SealedSequence;

/// Migrates currently copying residences.
static MIGRATE_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// [`CHUNK_LOCATIONS`]' frozen bit: a compaction is relocating chunks.
const FROZEN: usize = 1 << (usize::BITS - 1);

/// Held shared by a migrate while it **writes** device slots, and by a free or
/// snapshot off the compacting thread; exclusively by a KV compaction while it
/// relocates chunks: the count of shared holders, with [`FROZEN`] set while a
/// compaction holds it. Every step is one compare-and-swap; nothing waits on it.
///
/// # Why an exclusion is back, when the header says it went away
///
/// The lock this file used to hold guarded **arena base pointers**, and that
/// hazard is genuinely gone: a region is mapped once, at one address, for the
/// process lifetime. What came back is finer-grained. A pass plans from an
/// occupancy census and copies each source slot as it stands at the copy, so a
/// slot whose bytes are still being written — a migrate's quantize destination
/// before its convert lands — is copied unfinished, and the holder the sweep
/// rewrites to that copy reads it as its KV. It does not fault, and the wrong
/// bytes surface later as a sequence answering from another sequence's KV.
///
/// Measured while the hold spanned the whole group: with compaction gated on a 2 s
/// window, 8 of 8 sessions rewrote the story correctly across many runs; with the
/// same code gated per wave — 114 passes instead of 16 — it fell to 4 of 8.
///
/// **A migrate holds it only while it writes.** Its reads need no exclusion —
/// every slot it reads is one it names, which no pass frees or overwrites — and
/// the reads are what take the time: on BW, 2026-11-08, the quantize and convert
/// were 0–42 ms of groups lasting 20–922 ms, the DtoH the rest. A hold that
/// spanned the group refused 586 passes in five minutes.
///
/// # Neither side ever blocks
///
/// Both ask and defer when refused. That is not an optimisation: either side
/// waiting on the other would put a wait on a path that has none. Compaction
/// retries on its next wave, a migrate on the persistence thread's next pass;
/// both are periodic and neither is on a critical path.
static CHUNK_LOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// A shared hold on [`CHUNK_LOCATIONS`], or `None` while a compaction holds it.
fn try_share() -> Option<Shared> {
    let mut state = CHUNK_LOCATIONS.load(Ordering::SeqCst);
    loop {
        if state & FROZEN != 0 {
            return None;
        }
        match CHUNK_LOCATIONS.compare_exchange_weak(
            state,
            state + 1,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => return Some(Shared),
            Err(now) => state = now,
        }
    }
}

/// One shared hold on [`CHUNK_LOCATIONS`], released on drop.
struct Shared;

impl Drop for Shared {
    fn drop(&mut self) {
        CHUNK_LOCATIONS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// RAII marker: a migrate is copying residences until this drops. Drop-based so
/// an early return or an error still clears the count.
#[must_use = "the flight marker clears on drop; bind it for the migrate's scope"]
pub struct MigrateFlight {
    /// Keeps a compaction from copying slots this migrate is still writing, until
    /// [`Self::settle`]. See [`CHUNK_LOCATIONS`].
    locations: Option<Shared>,
}

impl MigrateFlight {
    /// **The migrate's device writes have landed** — its destinations claimed, its
    /// records filled, its convert fenced — so a compaction may run beside the rest
    /// of it. The in-flight count stays until the marker drops: it says the
    /// residences are being migrated, which they still are.
    ///
    /// Register what the group holds first ([`HeldChunks`]), so a pass that starts
    /// the moment this returns already steps around it.
    pub fn settle(&mut self) {
        self.locations = None;
    }
}

impl Drop for MigrateFlight {
    fn drop(&mut self) {
        MIGRATE_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A compaction pass has been refused and is waiting for a gap.
///
/// **The fairness half of the exclusion, and it is needed in one direction only.**
/// Neither side blocks, so whoever asks during the other's window simply loses — and
/// the two are not symmetric in how often they ask. A hot→warm batch is one long
/// hold per pass; a compaction asks between forwards and is bounded to tens of
/// milliseconds. Left to chance, the long holder wins nearly every time: measured, 78
/// refusals against 107 attempts, precisely during the mass eviction that produced
/// both the fragmentation and the migrate work.
///
/// So a refused pass says so, and the next migrate that would have taken the guard
/// steps aside for one round instead. A compaction clears it when it gets in, and its
/// own gate only lets it ask every few hundred milliseconds, so the migrate keeps the
/// large majority of the time.
static COMPACTION_WAITING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Mark a migrate as in flight until the returned marker drops, or `None` while a
/// compaction holds — or is waiting for — chunk locations.
///
/// **Never blocks, and never waits its turn.** A refusal means the caller should
/// come back on its next pass — see [`CHUNK_LOCATIONS`] for why waiting is the one
/// thing neither side may do. Concurrent migrates simply both count.
pub fn try_migrate_flight() -> Option<MigrateFlight> {
    if COMPACTION_WAITING.load(Ordering::SeqCst) {
        return None;
    }
    let locations = try_share()?;
    MIGRATE_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    Some(MigrateFlight {
        locations: Some(locations),
    })
}

/// Chunk slots held by an off-thread migrate's own handles, with how many
/// registrations name each — two groups can share a chunk.
fn held() -> &'static Mutex<AHashMap<i64, u32>> {
    static HELD: OnceLock<Mutex<AHashMap<i64, u32>>> = OnceLock::new();
    HELD.get_or_init(Mutex::default)
}

/// **The chunks a migrate group holds, which a compaction steps around** —
/// every band and record slot of the sequences it registered, until it drops.
///
/// The group's sequences live on the persistence thread's stack, where no sweep
/// can reach them. A pass that moved one would copy it, rewrite the holders it
/// can reach, and leave the group's handle keeping the source alive: claims
/// spent on ground nothing it reached takes up, reported `unwitnessed`, with the
/// source not reclaimed either. So a pass skips them
/// ([`retain_unheld`]), and the group's arenas pack once it has installed.
///
/// Registered while the migrate still holds chunk locations — before
/// [`MigrateFlight::settle`] — so no pass can start between the two and miss
/// it. A skipped move costs nothing: its destination was never claimed.
#[must_use = "the chunks are held only until this drops; bind it for the group's scope"]
#[derive(Default)]
pub struct HeldChunks {
    raws: Vec<i64>,
}

impl HeldChunks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold every band and record slot of `seqs`.
    pub fn hold<'a>(&mut self, seqs: impl IntoIterator<Item = &'a SealedSequence>) {
        let mut raws = Vec::new();
        for seq in seqs {
            for chunk in &seq.chunks {
                raws.extend(chunk.bands().map(|(gid, _)| gid.raw()));
                raws.extend(chunk.meta.as_ref().map(|m| m.raw()));
            }
        }
        self.hold_raws(raws);
    }

    /// Hold these raw slots.
    fn hold_raws(&mut self, raws: Vec<i64>) {
        let mut map = held().lock().unwrap_or_else(|e| e.into_inner());
        for &raw in &raws {
            *map.entry(raw).or_insert(0) += 1;
        }
        drop(map);
        self.raws.extend(raws);
    }

    /// Slots this registration holds, counting a shared one once per mention.
    pub fn len(&self) -> usize {
        self.raws.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raws.is_empty()
    }
}

impl Drop for HeldChunks {
    fn drop(&mut self) {
        if self.raws.is_empty() {
            return;
        }
        let mut map = held().lock().unwrap_or_else(|e| e.into_inner());
        for raw in &self.raws {
            if let Some(n) = map.get_mut(raw) {
                *n -= 1;
                if *n == 0 {
                    map.remove(raw);
                }
            }
        }
    }
}

/// Keep the items of `items` whose slot (`raw_of`) no migrate holds
/// ([`HeldChunks`]), in order, and answer how many were dropped. One lock for
/// the whole list.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn retain_unheld<T>(items: &mut Vec<T>, raw_of: impl Fn(&T) -> i64) -> usize {
    let map = held().lock().unwrap_or_else(|e| e.into_inner());
    if map.is_empty() {
        return 0;
    }
    let before = items.len();
    items.retain(|item| !map.contains_key(&raw_of(item)));
    before - items.len()
}

/// A shared hold on chunk locations taken to **free** or **snapshot** device chunks
/// off the thread that compacts — see [`try_hold_chunk_locations`].
#[must_use = "the hold lifts on drop; bind it for the free's scope"]
pub struct LocationHold {
    _locations: Shared,
}

/// Hold chunk locations still while dropping device chunks on a thread other than
/// the one that compacts, or `None` while a compaction holds — or is waiting for —
/// them.
///
/// **Freeing a chunk under a pass breaks the pass as surely as moving one does.** A
/// pass plans from an occupancy census, then claims, copies and sweeps. A slot freed
/// on another thread inside that window leaves its planned copy named by no holder
/// (counted as `unwitnessed`), and puts the slot back on the free list, where the
/// pass's own claims are handed it as a destination it had planned to read — a move
/// declined (`source_collisions`). Neither reads a wrong byte; both leave the pass
/// unable to finish packing, and the frontier stays where it was. Measured on the
/// Qwen3-30B engine probe through a mass retirement: 58,745 slots freed on the
/// persistence and cold-writer threads inside one pass, 15,252 of its moves wasted,
/// and the VRAM-efficiency gate failing on the samples that followed.
///
/// **Holding a snapshot of them across a pass breaks it the same way.** A pass sweeps
/// the holders it can reach, and another thread's locals are not among them: a clone of
/// a residence's chunks keeps the source slots alive after the residence has moved on,
/// so every later pass finds them live, plans them again, and reports the copy
/// `unwitnessed` for as long as the clone lives. A thread that reads device chunks
/// therefore snapshots and gathers them under this hold and drops the snapshot before
/// it goes back — the persistence thread's section persist, ~180,000 slots a pass on
/// the Qwen3-30B while the tool catalog reached the log.
///
/// The same word and the same rule as [`try_migrate_flight`] — shared, never
/// waits, defers to a waiting pass — without the in-flight count, which is a
/// statement about migrate work. A caller that is refused keeps the chunks and
/// lets a later pass of its own drop or read them.
pub fn try_hold_chunk_locations() -> Option<LocationHold> {
    if COMPACTION_WAITING.load(Ordering::SeqCst) {
        return None;
    }
    let locations = try_share()?;
    Some(LocationHold {
        _locations: locations,
    })
}

/// Exclusive hold on chunk locations for a compaction pass, or `None` while a
/// migrate is acting on addresses it has already captured.
///
/// Never blocks, for the reason on [`CHUNK_LOCATIONS`].
#[must_use = "the freeze lifts on drop; bind it for the compaction's scope"]
pub struct LocationFreeze(());

impl Drop for LocationFreeze {
    fn drop(&mut self) {
        CHUNK_LOCATIONS.fetch_and(!FROZEN, Ordering::SeqCst);
    }
}

pub fn try_freeze_chunk_locations() -> Option<LocationFreeze> {
    match CHUNK_LOCATIONS.compare_exchange(0, FROZEN, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => {
            // In, so stop holding migrates off — the freeze itself does that now.
            COMPACTION_WAITING.store(false, Ordering::SeqCst);
            Some(LocationFreeze(()))
        }
        Err(_) => {
            // Refused. Say so, so the next migrate that would have renewed the hold
            // steps aside for a round — see [`COMPACTION_WAITING`].
            COMPACTION_WAITING.store(true, Ordering::SeqCst);
            None
        }
    }
}

/// Stop holding migrates off after a compaction decided not to ask.
///
/// The flag is set by a refusal and cleared by the pass that follows it, so a caller
/// whose gate has since closed — the pools packed themselves, the pass is not worth
/// making — must clear it rather than leave migrates deferring for a pass that is
/// never coming.
pub fn clear_compaction_waiting() {
    COMPACTION_WAITING.store(false, Ordering::SeqCst);
}

/// Whether any migrate is copying residences right now. For deferring duplicated
/// *work* only — the safety question is answered by the guards above, not by this
/// counter, because a counter read cannot keep the answer true for the caller's
/// next instruction.
pub fn migrate_in_flight() -> bool {
    MIGRATE_IN_FLIGHT.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test for the counter and the lock: both are process-global, so
    // splitting them across tests that run in parallel would race on them.
    #[test]
    fn flights_compose_and_exclude_a_compaction() {
        assert!(!migrate_in_flight(), "clean start");
        assert!(
            try_freeze_chunk_locations().is_some(),
            "nothing is holding locations, so a pass may freeze them"
        );

        let a = try_migrate_flight().expect("no compaction is holding the freeze");
        let b = try_migrate_flight().expect("migrates compose");
        assert!(migrate_in_flight(), "in flight while held");
        assert!(
            try_freeze_chunk_locations().is_none(),
            "a compaction must refuse while a migrate holds captured addresses — \
             relocating them is what put the wrong KV in the warm tier"
        );

        drop(a);
        assert!(migrate_in_flight(), "still held by the second marker");
        assert!(
            try_freeze_chunk_locations().is_none(),
            "one remaining migrate is still one too many"
        );

        // A refusal leaves the waiting flag set, which is what stops the long holder
        // from renewing its hold indefinitely — so clear it before asserting that a
        // migrate can proceed again.
        clear_compaction_waiting();

        drop(b);
        assert!(!migrate_in_flight(), "cleared when the last marker drops");
        let freeze = try_freeze_chunk_locations().expect("the last migrate let go");
        assert!(
            try_migrate_flight().is_none(),
            "and the exclusion holds the other way: a migrate defers to the pass \
             rather than capturing addresses it is mid-way through rewriting"
        );
        drop(freeze);
        let resumed = try_migrate_flight().expect("the pass ended, so a migrate proceeds");

        // ── Fairness: a refused pass holds the next migrate off ──────────────
        //
        // Without it the long holder wins nearly every contest — a hot→warm batch is
        // one long hold per pass and a compaction asks between forwards — and chance
        // alone left 78 of 107 passes refused during the very burst that needed them.
        assert!(
            try_freeze_chunk_locations().is_none(),
            "refused while that migrate holds"
        );
        drop(resumed);
        assert!(
            try_migrate_flight().is_none(),
            "the refusal is remembered, so the next migrate steps aside instead of \
             renewing the hold"
        );
        let after_refusal = try_freeze_chunk_locations().expect("the gap the step-aside opened");
        drop(after_refusal);
        let held = try_migrate_flight().expect("the pass getting in released the hold");

        // ── And a pass that stands down must not leave migrates waiting ──────
        assert!(try_freeze_chunk_locations().is_none(), "refused again");
        drop(held);
        clear_compaction_waiting();
        assert!(
            try_migrate_flight().is_some(),
            "the pass stood down, so nothing is holding the migrate off"
        );

        // ── A free holds locations the same way, and counts no migrate ───────
        let hold = try_hold_chunk_locations().expect("nothing is compacting");
        assert!(
            !migrate_in_flight(),
            "a free is not migrate work — the advisory count stays clear"
        );
        assert!(
            try_freeze_chunk_locations().is_none(),
            "a pass refuses while a free off its thread is in progress"
        );
        drop(hold);
        assert!(
            try_hold_chunk_locations().is_none(),
            "and the refused pass is waited for: the next free defers to it"
        );
        let freeze = try_freeze_chunk_locations().expect("the gap the deferral opened");
        assert!(
            try_hold_chunk_locations().is_none(),
            "no free off the compacting thread while a pass holds the locations"
        );
        drop(freeze);
        assert!(try_hold_chunk_locations().is_some(), "the pass ended");

        // ── A settled migrate lets a pass in and is still in flight ──────────
        let mut flight = try_migrate_flight().expect("nothing is compacting");
        assert!(
            try_freeze_chunk_locations().is_none(),
            "a pass refuses while the migrate is still writing slots"
        );
        clear_compaction_waiting();
        flight.settle();
        assert!(
            migrate_in_flight(),
            "settled is not finished: its residences are still being migrated"
        );
        let freeze = try_freeze_chunk_locations().expect("the writes landed, so a pass may run");
        assert!(
            try_migrate_flight().is_none(),
            "and a new migrate still waits for the pass before it writes"
        );
        drop(freeze);
        drop(flight);
        assert!(
            !migrate_in_flight(),
            "cleared when the settled marker drops"
        );
    }

    /// **A pass steps around what a migrate holds, for as long as it holds it**,
    /// counting a slot two groups share until both let go.
    #[test]
    fn held_chunks_are_skipped_until_every_holder_lets_go() {
        // Raws no other test registers: the registry is process-global.
        let (a, b, c) = (-1_000_001i64, -1_000_002, -1_000_003);
        let mut first = HeldChunks::new();
        first.hold_raws(vec![a, b]);
        let mut second = HeldChunks::new();
        second.hold_raws(vec![b]);

        let mut moves = vec![a, b, c];
        assert_eq!(retain_unheld(&mut moves, |&r| r), 2);
        assert_eq!(moves, vec![c], "only the unheld slot stays planned");

        drop(first);
        let mut moves = vec![a, b, c];
        assert_eq!(retain_unheld(&mut moves, |&r| r), 1);
        assert_eq!(
            moves,
            vec![a, c],
            "the shared slot is still held by the second"
        );

        drop(second);
        let mut moves = vec![a, b, c];
        assert_eq!(retain_unheld(&mut moves, |&r| r), 0);
        assert_eq!(moves, vec![a, b, c], "every holder let go");
    }
}
