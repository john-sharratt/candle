//! The scheduler's wave domain: one arena per layer phase, plus the forward span.
//!
//! A layer opens its attention generation and then its FFN generation, so each
//! phase gets an arena sized from its own footprint. A phase's span is reset
//! when its guard drops and reused a layer later. That reuse is safe because the
//! other phase's whole kernel sequence sits between the two on the same stream,
//! and same-stream work completes in issue order. The domain also records the
//! plan the next placement reserves, the tier's placed position and size, and
//! whether a forward is between its first and last line, since the generation
//! count reads zero at every phase boundary.

use super::*;

/// The scheduler's wave domain: one arena per layer phase.
///
/// A layer opens exactly two generations — attention, then the FFN — so with the
/// arenas named by phase, attention reuses one span every layer and the FFN
/// reuses the other. That is what the alternating pair already produced by
/// parity; naming it removes the dependence on the count of `begin_wave` calls
/// per layer staying even, and lets each span be sized from *its own* phase
/// rather than both from the larger.
///
/// The sizing difference is not small: on Qwen3-30B-A3B at ten concurrent
/// contexts the attention phase sums to 297 MiB and the FFN to at least 511 MiB,
/// so two max-phase spans would reserve 1022 MiB where 808 MiB does.
///
/// # Why resetting a phase's span every layer is safe
///
/// Unchanged from the alternating pair, because the ordering argument was never
/// about which half: a span is reset when its guard drops and handed out again
/// one layer later, and between those two points sits the *other* phase's entire
/// kernel sequence on the same stream. Same-stream launches complete in issue
/// order, so the previous layer's reads have drained before the next layer's
/// writes are issued ([`Reclaim::StreamOrdered`]).
pub(super) struct WaveDomain {
    /// Indexed by [`LayerPhase`] via [`phase_index`].
    pub(super) arenas: [BumpArena; 3],
    /// Generations open across **all three** arenas.
    ///
    /// The per-arena `live` count bounds one span's cursor; this bounds the
    /// tier's *existence*. It goes 0 → 1 when a forward opens its first phase
    /// and 1 → 0 when the last guard drops, and those two moments are exactly
    /// where the tier is placed and where it vanishes.
    pub(super) live_generations: usize,
    /// What the next placement should reserve per phase, set by the forward
    /// before it opens anything ([`plan_wave_transient`]).
    ///
    /// `None` until a forward has priced itself, which is the case for every
    /// path that opens a wave without going through the batched forward — tests,
    /// the migration helpers — and those get [`fallback_plan`].
    pub(super) planned: Option<[usize; 3]>,
    /// The bytes of `planned`'s forward-wide span that are the launchers'
    /// scratch reserve (`candle::cuda_backend::scratch_reserve`) — what a
    /// forward's scratch scope may carve. Zero with no plan: [`fallback_plan`]
    /// prices no scratch.
    pub(super) scratch_priced: usize,
    /// `planned` belongs to a forward that has not laid its spans out yet: it
    /// priced itself and has still to open its first phase.
    ///
    /// Such a plan is not a finished forward's width but the next layout's, so
    /// an arena window that hands the tier back keeps it — the forward's guard
    /// re-places it at the frontier the window left. Dropped there, the guard
    /// laid the fallback out under the forward instead: on WhiteWorld
    /// (a965e3a07, 2026-10-19) three arenas created between a creep step's
    /// placement and its forward span left a 74 MB residual asking a 16 MiB
    /// span, with no generation live for the outgrown-spans refusal to see.
    pub(super) plan_pending: bool,
    /// The width the caller running **between forwards** — a draft walk —
    /// priced for itself through [`cover_wave_transient`], while it holds the
    /// [`Covering`] that call returned.
    ///
    /// An arena window hands the standing tier back and drops `planned`, and a
    /// walk creates arenas mid-walk (its module docs say why no pre-ensure
    /// reaches every key). With no plan, the walk's next phase re-placed the
    /// worst-case [`fallback_plan`] from inside its own recording, where the
    /// weight side refuses to concede: Flash-Next's BF16 ×4 gate row failed on
    /// a 912 MiB tier 19 regions short. The window leaves this as the plan
    /// instead, so the walk re-places only its own width.
    pub(super) covering: Option<[usize; 3]>,
    /// `covering`'s launch-scratch reserve, as `scratch_priced` is `planned`'s.
    pub(super) covering_scratch: usize,
    /// Where the tier currently sits, while it exists.
    pub(super) placed_at: Option<u64>,
    /// **How much ground that placement bought**, which is the only figure the
    /// layout may be checked against.
    ///
    /// Recorded because the alternative was reading
    /// `region_stats(..).transient_bytes`, and that answers a different
    /// question: it is live occupancy *right now* — explicitly "a snapshot
    /// taken mid-wave, not a reservation" — and it is a property of the pool,
    /// not of this domain's tier. When the snapshot happened to read at least
    /// as large as the plan while this domain's own tier was narrower, the
    /// replan was skipped, the arenas were laid out with the new plan's widths
    /// on the old tier's base, and the spans ran off its top into the expert
    /// weight zone. The reservation is what the layout has to fit inside, so
    /// the reservation is what gets stored.
    ///
    /// `Some` exactly when `placed_at` is: the two are set and cleared together.
    pub(super) placed_bytes: Option<usize>,
    /// Whether that placement belongs to a **forward** or to a lone guard.
    ///
    /// [`plan_wave_transient`] reserves for the forward, so the reservation has
    /// to outlive the phase boundaries inside it and is released by the *next*
    /// forward. A caller that opens a guard without ever pricing itself — a
    /// test, a migration helper, anything at load — has no forward to hand it
    /// back, so its reservation is scoped to the guard and
    /// [`release_if_last`] returns it.
    ///
    /// Without the distinction a single unplanned `begin_wave` at load pins the
    /// tier at whatever the frontier was then, for the life of the process.
    pub(super) reserved_by_forward: bool,
    /// Callers currently creating KV arenas outside a wave — see
    /// [`enter_arena_window`].
    ///
    /// **The wave is pre-allocated end to end**, and a new arena is the one
    /// allocation that cannot be: it claims a region the partition did not have
    /// spoken for, and the tier's placement is measured against the arena
    /// frontier at the instant the forward begins. A count rather than a flag
    /// because the compressor and the fork path can both be inside one.
    pub(super) arena_windows: usize,
    /// Whether a forward is between its first and last line.
    ///
    /// **`live_generations == 0` does not mean "no forward is running".** It
    /// falls to zero at every phase boundary — a layer's attention guard drops
    /// before its FFN guard opens, and the head span opens after the last
    /// layer's has gone — so a thread that read it as "the partition is idle"
    /// would take the gap in the middle of a forward and pull the tier out from
    /// under the next phase. That is the §13b failure.
    ///
    /// This is set from the end of admit to the end of `forward_wave_contexts`
    /// and is what [`enter_arena_window`] actually waits on; `live_generations`
    /// then covers the tail, where the forward has returned but its logits still
    /// sit on the head span.
    pub(super) forward_open: bool,
    /// Which thread opened the forward, so [`enter_arena_window`] can tell a
    /// *wait* from a **self-deadlock**.
    pub(super) forward_thread: Option<ThreadId>,
    /// Where the forward span now live was opened — what a refusal of a plan
    /// that outgrew the spans names as the generation holding their layout.
    pub(super) forward_opener: Option<&'static std::panic::Location<'static>>,
}

/// What to reserve per phase when a wave opens without having priced itself.
///
/// The old fixed constants. A caller that goes through the batched forward
/// always sets a real plan; this covers the paths that do not, and it is
/// deliberately the worst case rather than a guess — an under-sized fallback
/// would exhaust a span rather than merely waste one.
pub(super) fn fallback_plan() -> [usize; 3] {
    align_phase_plan([WAVE_ATTN_BYTES, WAVE_FFN_BYTES, WAVE_FORWARD_BYTES])
}

/// Arena coordinate for a domain that is **not** a wave half.
///
/// The persistence domain stages on the copy stream and reclaims by fence, so a
/// range from it must never be inherited by an op the way a wave range is: the
/// two are not ordered against each other. A ticket carrying this resolves to
/// nothing, so an op reading persistence-staged memory allocates from the pool —
/// which is the correct answer, not a fallback.
pub(crate) const NOT_A_WAVE: u32 = u32::MAX;

/// Slot for a phase's arena.
///
/// A `match` rather than a cast so adding a phase is a compile error here
/// instead of an out-of-bounds index at run time.
pub(super) fn phase_index(phase: LayerPhase) -> usize {
    match phase {
        LayerPhase::Attention => 0,
        LayerPhase::Ffn => 1,
        LayerPhase::Forward => FORWARD_ARENA as usize,
    }
}

/// The layer phase whose span shares `phase`'s ground ([`phase_offsets`]) —
/// `None` for the forward span, which has its own.
pub(super) fn other_layer_phase(phase: LayerPhase) -> Option<LayerPhase> {
    match phase {
        LayerPhase::Attention => Some(LayerPhase::Ffn),
        LayerPhase::Ffn => Some(LayerPhase::Attention),
        LayerPhase::Forward => None,
    }
}

/// The forward-wide span's arena index in a wave domain — the span a forward's
/// launch scratch is carved from.
pub(super) const FORWARD_ARENA: u32 = 2;

pub(super) fn wave_domains() -> &'static Mutex<HashMap<usize, WaveDomain>> {
    static DOMAINS: OnceLock<Mutex<HashMap<usize, WaveDomain>>> = OnceLock::new();
    DOMAINS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Signalled whenever a domain's `placed_at`, `live_generations` or
/// `arena_windows` changes — the three fields the two waits below read.
pub(super) fn wave_gate() -> &'static Condvar {
    static GATE: OnceLock<Condvar> = OnceLock::new();
    GATE.get_or_init(Condvar::new)
}

pub(super) type DomainMap = HashMap<usize, WaveDomain>;

/// The domain for `ordinal`, created on first touch.
///
/// Split out of [`with_wave_domain`] because the waits below have to re-read the
/// domain after each wake-up, and the guard they hold is the raw map lock.
pub(super) fn domain_entry<'a>(
    map: &'a mut DomainMap,
    stream: &Arc<CudaStream>,
) -> (usize, &'a mut WaveDomain) {
    let ordinal = stream.context().ordinal();
    let domain = match map.entry(ordinal) {
        Entry::Occupied(o) => o.into_mut(),
        Entry::Vacant(v) => v.insert(WaveDomain {
            // Created with no backing at all: the tier does not exist between
            // forwards, so there is nothing to carve until one opens.
            arenas: [
                BumpArena::detached(stream, "wave-attn"),
                BumpArena::detached(stream, "wave-ffn"),
                BumpArena::detached(stream, "wave-forward"),
            ],
            live_generations: 0,
            planned: None,
            scratch_priced: 0,
            plan_pending: false,
            covering: None,
            covering_scratch: 0,
            placed_at: None,
            placed_bytes: None,
            reserved_by_forward: false,
            arena_windows: 0,
            forward_open: false,
            forward_opener: None,
            forward_thread: None,
        }),
    };
    (ordinal, domain)
}

pub(super) fn lock_domains() -> MutexGuard<'static, DomainMap> {
    wave_domains().lock().unwrap_or_else(|e| e.into_inner())
}

pub(super) fn with_wave_domain<R>(
    stream: &Arc<CudaStream>,
    f: impl FnOnce(&mut WaveDomain) -> Result<R>,
) -> Result<R> {
    let mut map = lock_domains();
    let (_, domain) = domain_entry(&mut map, stream);
    let r = f(domain);
    drop(map);
    wave_gate().notify_all();
    r
}
