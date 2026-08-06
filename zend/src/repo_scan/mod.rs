//! `repo_map` layer ingestion — **one conversation per directory**.
//!
//! Walks the workspace, groups the files by their directory, and explores each
//! directory as TWO `code_read`-shaped tool round-trips — list it, then read its
//! module doc — the last of which DECODES a two-sentence summary of what the
//! folder is for (see [`render`] for the turn shape). That summary is the
//! layer's retrieval surface: a query about "the KV cache paging code" matches
//! prose describing that folder, where a bare file listing would only match on a
//! filename the asker already knew.
//!
//! Structurally this mirrors [`crate::code_read`]: a bounded worker pool over
//! per-unit conversations, each tagged with a content hash that serves as the
//! restart-resume cache key and the refresh's change detector, and each freed
//! after its turns seal into the substrate. [`DirState`] records those hashes so
//! a filesystem event re-ingests only the directories that actually changed.

pub mod anchor;
pub mod binary_sniff;
pub mod dir_unit;
pub mod render;
pub mod types;
pub mod walk;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::memory_report::MemoryReport;
use candle_conversation::projection::{self, GroupId, LayerId, TimelineId};
use candle_conversation::{ConversationEngine, SequenceConfig};
use zend_tools::ToolContext;

use crate::ingest_report::{Failures, IngestReport};
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::turn_sink::{InsertTurnSink, SequenceTurnSink};

pub use binary_sniff::is_binary_sample;
pub use dir_unit::{build_units, DirRecord, DirState, DirUnit};
pub use types::{FileEntry, Language, RepoMap};
pub use walk::{walk_workspace, MAX_FILE_BYTES};

/// Directories ingested concurrently. Each unit is ONE conversation running a
/// short chain (two prefills + one bounded decode), so this is the whole
/// concurrent conversation count for the layer.
///
/// Sized to feed BOTH row-groups. A directory's worker is a three-phase state
/// machine — prefill, prefill, decode — so width buys sequences in each. At 8,
/// 45% of waves ran a prefill with the decode row idle, and the extra width
/// exists to fill those.
///
/// Width also widens prefill itself, which is where most of a directory's wall
/// clock goes. A prefill wave carries as many sequences as are ready, and its
/// throughput scales close to linearly with that count — measured across 981
/// waves on Qwen3-30B-A3B: 188 t/s at one sequence, 357 at two, 457 at three,
/// 699 at four, 1427 at five. A forward carried 10.9x the tokens for 7.2x the
/// time as the batch grew, because the expert weight load amortizes across the
/// batch. Sustaining four or more ready sequences takes roughly twice that many
/// open conversations, since each spends part of its chain decoding.
///
/// 24 matches the scheduler's own `MAX_PREFILL_WIDTH` ceiling, so the pool never
/// asks for more concurrency than a wave can carry, and sits below `code_read`'s
/// effective 48 (12 file workers × 4 forked scopes). The trade is VRAM: every
/// admitted conversation pins its K/V, which feeds the same pressure that caps
/// prefill — so on a tight card the useful width is bounded by eviction churn
/// rather than by this constant.
pub const REPO_MAP_PARALLELISM: usize = 24;

/// Longest a worker will wait for VRAM before claiming its unit anyway.
///
/// A bound, not a timeout to rely on: without it a pathological state where the
/// pool never drops would stall the scan forever. Reaching it means the gate
/// failed to help, and the arena allocator's own refusal is the next line of
/// defence.
const SCAN_POOL_WAIT_CAP: std::time::Duration = std::time::Duration::from_secs(20);

/// How many conversations this pool may hold open at once, derived from the
/// card rather than from a thread count.
///
/// Two earlier shapes failed, and both failures are instructive:
///
/// 1. A snapshot test ("is the pool full?") blocks new claims but cannot
///    un-open the conversations already running, so the pool sailed past the
///    mark while 24 chains were in flight.
/// 2. A forward-looking test ("would ONE more fit?") is still evaluated by
///    every worker against the same pre-allocation state. At scan start all 24
///    read a nearly-empty pool, all passed, and all allocated together — a
///    thundering herd that put the whole burst over the wall at once. That is
///    why every configuration produced exactly `n_failed = 24`: one burst, one
///    wall, 24 casualties.
///
/// So the bound is a COUNT, evaluated under a lock, against memory that is not
/// KV: `(limit - fixed_footprint) / per_conversation`. `fixed_footprint` is the
/// expert cache plus dense weights — everything the pool holds that a scan can
/// never free — and `per_conversation` is measured live. Workers then queue on
/// the count instead of racing a gauge that lags them.
fn max_live_conversations() -> Option<usize> {
    let (report, age_ms) = candle_conversation::memory_report::latest()?;
    if age_ms > 10_000 {
        return None;
    }
    let kv = kv_reserved(&report);
    let vram = report.vram?;
    let governor = vram.governor?;
    if governor.capacity_bytes == 0 {
        return None;
    }
    Some(scan_width(
        governor.capacity_bytes,
        governor.scratch_margin_bytes,
        vram.pool_used_bytes,
        kv,
        SCAN_KV_BASELINE.load(Ordering::Relaxed),
        SCAN_LIVE_CONVS.load(Ordering::Relaxed),
    ))
}

/// Total reserved KV arena bytes in a memory report — the quantity the arena
/// allocator's ceiling actually counts, across both backings.
fn kv_reserved(report: &MemoryReport) -> u64 {
    report
        .kv
        .float_reserved_bytes
        .saturating_add(report.kv.quant_reserved_bytes)
}

/// The count bound, as arithmetic over the report's figures alone.
///
/// `(capacity - scratch_margin - fixed - baseline) / per_conversation`, where
/// `fixed` is what the pool holds that no scan decision can release — the expert
/// cache plus dense weights — `baseline` is the inherited pre-scan KV corpus
/// (see [`SCAN_KV_BASELINE`]), and `per_conversation` is measured from the
/// pool's own growth.
///
/// Numerator and denominator must price the same bytes. `fixed` is
/// `pool_used - kv`, so subtracting it alone hands *every* KV byte back as room
/// for new conversations — including the inherited corpus, which is resident,
/// is not the scan's to spend, and is precisely what `per_conversation_kv`
/// excludes on the other side of the divide. On the measured report below that
/// counted ~2 GiB of standing arenas as free space and opened ten directories
/// against room for six.
///
/// The thing that costs VRAM is an *open conversation*: it pins its K/V until
/// its chain completes, and while it is live its K sits in `R16` (4 bytes per
/// element) and its V in F16, so a directory costs several times what the same
/// turns cost once sealed and quantized. Admission cannot substitute for this
/// bound — it throttles prefill *submission*, and by then the conversation
/// exists and its K/V is already pinned.
///
/// The margin held back is the governor's OWN `scratch_margin`, not a fraction
/// of capacity. A fraction double-charges the expert cache: `fixed` subtracts it
/// explicitly, then the fraction holds back a share of capacity that is mostly
/// the same bytes again. Measured on the 16 GiB card, a 0.70 fraction left the
/// gate 1.87 GiB for scan KV while the governor had ~5 GiB floored for exactly
/// that — under half the room, and width is what the whole pool is for: up to
/// the point the card sustains, a wave's throughput scales with the sequences it
/// carries, because the expert load amortizes across them (prefill 188 t/s at
/// one sequence against 699 at four; decode 3.4 against 15.9 at nine). Holding
/// back the scratch margin instead spends the room the governor already reserved
/// for KV — while [`SCAN_CONV_KV_MIN`] keeps the resulting width on the safe
/// side of that point.
fn scan_width(
    capacity: u64,
    scratch_margin: u64,
    pool_used: u64,
    kv: u64,
    baseline: u64,
    live: usize,
) -> usize {
    let fixed = pool_used.saturating_sub(kv);
    let for_kv = capacity
        .saturating_sub(scratch_margin)
        .saturating_sub(fixed)
        .saturating_sub(baseline);
    let per_conv = per_conversation_kv(kv, baseline, live);
    ((for_kv / per_conv.max(1)) as usize).clamp(1, REPO_MAP_PARALLELISM)
}

/// What one more open directory conversation is expected to cost the pool.
///
/// Measured rather than assumed, and measured against the pool's OWN growth:
/// the estimate prices `kv - baseline`, never the whole process's arenas (see
/// [`SCAN_KV_BASELINE`]). Clamped so a cold start cannot admit unboundedly and
/// one atypical directory cannot stall the scan.
fn per_conversation_kv(kv: u64, baseline: u64, live: usize) -> u64 {
    let grown = kv.saturating_sub(baseline);
    if live > 0 && grown > 0 {
        (grown / live as u64).clamp(SCAN_CONV_KV_MIN, SCAN_CONV_KV_MAX)
    } else {
        SCAN_CONV_KV_MIN
    }
}

/// Reserved KV present when the pool opened — the corpus the scan did not
/// create, and cannot free by finishing a directory.
///
/// The tool sections, the base builder's prefill and the calibration exemplars
/// all live in the same arenas the report totals, and together they run to
/// gigabytes before a single directory is opened. Charging that to the handful
/// of conversations in flight pins [`per_conversation_kv`] to
/// [`SCAN_CONV_KV_MAX`] and throttles the pool to a width of two on a card with
/// room for more — and the error compounds in the wrong direction, since fewer
/// live conversations divide the same fixed corpus into a larger per-conversation
/// estimate. Pricing the delta instead leaves exactly the part a scan decision
/// influences: a completed directory demotes to the warm tier and gives its
/// arenas back, so the delta tracks what is genuinely in flight.
static SCAN_KV_BASELINE: AtomicU64 = AtomicU64::new(0);

/// Re-anchor [`SCAN_KV_BASELINE`] on the arenas already in place, so the pool
/// about to open prices only the KV it goes on to create. Called once per pass,
/// before any worker claims a directory.
fn anchor_scan_kv_baseline() {
    let baseline = candle_conversation::memory_report::latest()
        .map(|(report, _age_ms)| kv_reserved(&report))
        .unwrap_or(0);
    SCAN_KV_BASELINE.store(baseline, Ordering::Relaxed);
    tracing::debug!(
        target: "zend::repo_scan",
        baseline_bytes = baseline,
        "scan pool: per-conversation KV estimate anchored on the pre-scan corpus",
    );
}

/// Pool width from the VRAM governor, for the moment BEFORE any memory report
/// exists.
///
/// The scan starts before the scheduler has published its first report — the
/// walk runs between calibration and the pool — so [`max_live_conversations`]
/// has nothing to read and every worker would fall back to the full constant.
/// The governor is live from model load, and its headroom at this instant is
/// exactly the VRAM left after weights and the expert cache: the room a scan's
/// conversations have to share.
///
/// A slice is held back for wave activations (measured at ~1 GiB for a wide
/// prefill, and the wall is only reached when a wave lands on a high base), and
/// the rest is divided by the per-conversation KV estimate.
fn scan_width_from_governor() -> Option<usize> {
    let gov = candle::vram::get(0)?;
    let headroom = gov.measure().ok()?.headroom;
    let for_kv = headroom.saturating_sub(SCAN_ACTIVATION_RESERVE);
    Some(((for_kv / SCAN_CONV_KV_MIN) as usize).clamp(1, REPO_MAP_PARALLELISM))
}

/// Held back from the scan's share for a wave's transient activations. Measured
/// at 666-1005 MiB across prefill waves on Qwen3-30B-A3B; the failures all
/// occurred when a wave of that size landed on an already-high base.
const SCAN_ACTIVATION_RESERVE: u64 = 1536 * 1024 * 1024;

/// Live conversation count, readable by [`max_live_conversations`] for its
/// per-conversation estimate without threading the counter through.
static SCAN_LIVE_CONVS: AtomicUsize = AtomicUsize::new(0);

/// Floor/ceiling on the per-conversation KV estimate.
///
/// The floor is measured, not guessed: a 24-conversation burst drove KV arenas
/// to 5760 MiB, i.e. ~240 MiB of LIVE KV each — and arenas reserve roughly
/// twice what they hold live (2560 MiB reserved against 1360 live in a steady
/// sample), because each format keeps its own partially-filled 16 MiB slabs.
/// So a conversation costs the pool about 480 MiB of *reserved* arena, which is
/// the quantity the allocator's ceiling actually counts.
///
/// The doubling is NOT slack, and halving it to price only the live half is a
/// measured dead end. The argument for halving is that slab waste belongs to a
/// *format* rather than to a conversation, so the marginal directory should not
/// pay it twice; the card says otherwise. At 256 MiB the gate opened 19
/// directories on the 16 GiB card and the pass hit the wall inside two minutes:
/// `pool_reserved=14976MiB` against `pool_used=13560MiB` — a 1415 MiB gap of
/// reserved-but-unfilled arena — 18 MiB free, a 16 MiB quantized arena refused
/// with `arenas_freed=0`, a device OOM that halved the admission budget, and 15
/// directories lost in a single millisecond. The gap is exactly the slab
/// overhead, and it scales with concurrently-live conversations, because each
/// one's chunks land in different format arenas at different fill levels.
///
/// It was also slower, which is the part worth remembering: 10 s per directory
/// against 6.82 at width 10, with decode throughput at nine co-batched sequences
/// falling from 15.9 tok/s to 7.9 and degrading further past ten. Past the wall
/// the extra width buys eviction churn, not throughput.
const SCAN_CONV_KV_MIN: u64 = 480 * 1024 * 1024;
const SCAN_CONV_KV_MAX: u64 = 768 * 1024 * 1024;

/// Block until the KV pool has room for another open conversation, then COUNT
/// THIS ONE IN before releasing the gate.
///
/// `live` is the count of conversations this pool currently holds open. When it
/// is zero the wait is skipped unconditionally — one conversation must always be
/// able to proceed, or a pool whose only route to freeing VRAM is finishing the
/// work it is not allowed to start would deadlock.
///
/// The reservation happens under the SAME lock as the decision, and that is the
/// whole point. Deciding under the lock and incrementing after it re-opens the
/// race the gate exists to close: every waiting worker reads the same
/// pre-increment count, each concludes there is room for one more, and they all
/// proceed — a narrower replay of the herd that put exactly 24 conversations on
/// the card at once and produced `n_failed = 24` under every configuration.
/// Callers must therefore pair this with [`release_scan_slot`] on every exit
/// path, including failures.
fn reserve_scan_slot(live: &AtomicUsize) {
    // Serialises the decision so workers cannot all read the same pre-allocation
    // state and admit together (see `max_live_conversations`).
    static GATE: Mutex<()> = Mutex::new(());
    let start = std::time::Instant::now();
    let mut logged = false;
    loop {
        let cap = {
            let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
            let cap = max_live_conversations();
            let now = live.load(Ordering::Relaxed);
            // Always let one through: a pool whose only route to freeing VRAM is
            // finishing work it is not allowed to start would deadlock.
            match cap {
                Some(c) if now >= c && now > 0 => c,
                _ => {
                    take_scan_slot(live);
                    return;
                }
            }
        };
        if start.elapsed() >= SCAN_POOL_WAIT_CAP {
            // Waited long enough that stalling the whole ingest is the worse
            // outcome. Still taken under the gate, so the count stays exact.
            let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
            take_scan_slot(live);
            return;
        }
        if !logged {
            logged = true;
            tracing::debug!(
                target: "zend::repo_scan",
                live_conversations = live.load(Ordering::Relaxed),
                max_live = cap,
                "scan pool: waiting for KV room before opening another directory",
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Count one conversation in, on both the pool-local and process-wide gauges.
/// Called only with the reservation gate held.
fn take_scan_slot(live: &AtomicUsize) {
    live.fetch_add(1, Ordering::Relaxed);
    SCAN_LIVE_CONVS.fetch_add(1, Ordering::Relaxed);
}

/// Holds one admitted conversation's slot on both gauges, and gives it back on
/// drop.
///
/// A guard rather than a paired release call: `SCAN_LIVE_CONVS` is
/// process-global and never reset, so a slot lost to an unwind is lost for the
/// life of the daemon. One panic inside an ingest — where the failure paths
/// already tolerate a directory going wrong — would leave the gate permanently
/// believing a conversation is open, and enough of them throttle every later
/// pass toward a width of one.
struct ScanSlot<'a> {
    live: &'a AtomicUsize,
}

impl<'a> ScanSlot<'a> {
    /// Wait for room, then take a slot. Returns holding it, so no other worker
    /// can decide against this state.
    fn reserve(live: &'a AtomicUsize) -> Self {
        reserve_scan_slot(live);
        Self { live }
    }
}

impl Drop for ScanSlot<'_> {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
        SCAN_LIVE_CONVS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Hard `max_tokens` on a folder summary decode. The request asks for two
/// sentences; this bounds the runaway case without clipping a summary that
/// enumerates a few of the folder's parts. Matches the `code_reading` scope
/// budget, which produces summaries of the same shape and length.
const FOLDER_SUMMARY_MAX_TOKENS: usize = 200;

/// Tolerated per-directory ingest failures before the whole pass aborts. A
/// handful of unreadable directories shouldn't sink a workspace scan; a flood
/// means something systemic (out of KV VRAM) and continuing just burns GPU.
const MAX_DECODE_FAILURES: usize = 24;

/// Key this pass reports completeness under (see [`crate::ingest_report`]).
pub const PASS_NAME: &str = "repo_map";

/// Gather-scope tags for a directory's turns: `["repo_map", <dir>]`. The second
/// tag is the unit's directory (`"."` for the workspace root), so a tag-scoped
/// provenance gallery can admit exactly one folder's turns.
fn dir_tags(unit: &DirUnit) -> Vec<String> {
    vec!["repo_map".to_string(), unit.dir.clone()]
}

/// Strip auto-summarization from a [`SequenceConfig`] before using
/// it to mint a utility-layer conversation (repo_map, code_reading).
///
/// The legacy per-turn tree summarization (`summarize_every`) runs synchronously
/// inside `finalize_turn_post_done` — `drain_cognitive_tasks` spin-polls each
/// task to completion before `insert_turn` returns. For repo_map / code_reading,
/// which carry hundreds of small structured turns (folder chains, scope reads),
/// that would stall every unit behind the summarizer, so it stays off here. The
/// async AVL summariser is separate and unaffected: it runs on its own thread
/// (wave-driven compression) and summarises every layer — including these —
/// without blocking ingest, and provenance scans expand the compressed nodes on
/// retrieval.
pub(crate) fn utility_config(mut config: SequenceConfig) -> SequenceConfig {
    config.tree.summarize_every = 0;
    config.tree.segment_summarize_every = 0;
    // Utility ingests (repo_map, code_reading) are append-only cumulative
    // trunks — each turn just extends the layer. Skip the per-turn projection
    // rebuild (reset + re-project the whole trunk, which is O(n²) and serial on
    // the scheduler thread); turns still seal into the substrate. This lets the
    // parallel workers' prefills/decodes actually batch instead of serialising
    // behind reprojection.
    config.disable_reprojection = true;
    // Utility ingests quantize at C5, fully adaptive for both K and V (the
    // engine-wide uniform-K pin is off in this config). The code-reading layer
    // inherits this same C5 level via `code_read_config`.
    config.kv_compression_level = Some(5);
    config
}

/// Top-level `repo_map` ingestion.
///
/// Walks `workspace`, builds one [`DirUnit`] per directory holding files, and
/// runs the units through a bounded worker pool. Returns the walked [`RepoMap`]
/// so a co-located `code_reading` pass doesn't re-walk, plus the [`DirState`]
/// the refresh path compares against.
///
/// The closing decode is a summary *of the folder* because
/// [`layer_system_prompt`] frames the conversation with [`SUMMARIZE_BRANCH`] —
/// the summarizer persona plus the FOLDER-shaped worked examples. Both parts
/// matter, and neither can be selected at runtime: the prompt is a static string
/// and `disable_reprojection` (see [`utility_config`]) means the conversation
/// never re-projects, so the selections
/// `ingest_roundtrip_chain_indices` sets can never materialise.
#[allow(clippy::too_many_arguments)]
pub fn ingest_repo_map(
    engine: &Mutex<ConversationEngine>,
    proj_builder: projection::Builder,
    workspace: &Path,
    config: SequenceConfig,
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<(RepoMap, DirState, IngestReport)> {
    let map = walk_workspace(workspace);
    let units = build_units(&map, workspace);

    tracing::info!(
        n_files = map.files.len(),
        n_dirs = units.len(),
        n_anchored = units.iter().filter(|u| u.anchor.is_some()).count(),
        skipped_extension = map.files_skipped_extension,
        skipped_oversize = map.files_skipped_oversize,
        skipped_binary = map.files_skipped_binary,
        "repo map walk complete; ingesting one conversation per directory",
    );

    let plan = IngestPlan::new(engine, &proj_builder, &config, layer_name, group_name)?;
    // Retire conversations for directories that no longer exist, then snapshot
    // the surviving hashes once for O(1) per-unit resume-cache probes.
    let present: HashSet<&str> = units.iter().map(|u| u.dir.as_str()).collect();
    reconcile_deleted(engine, &present);
    let report = run_dir_pool(engine, &plan, workspace, &units, progress);
    Ok((map, dir_state_from_substrate(engine), report))
}

/// Outcome of a [`refresh_repo_map`] call. `Replaced` carries only the new
/// per-directory hash record — per-unit conversations are freed once their
/// turns seal, so there is no live sequence to swap.
pub enum RefreshOutcome {
    NoOp,
    Replaced { state: DirState },
}

/// Selective refresh of the `repo_map` layer.
///
/// Re-derives the units from `map` (which the caller already walked, usually
/// once per filesystem-event burst and shared with the `code_reading` refresh)
/// and returns `NoOp` when no directory's hash moved. Otherwise it runs the same
/// reconcile + pool as [`ingest_repo_map`]: directories whose hash is unchanged
/// hit the resume-cache snapshot and are skipped, so only changed, added, or
/// removed directories cost anything.
///
/// The engine mutex is taken only for the quick create/tombstone ops inside the
/// pool — never across a decode — so chat consumers keep running throughout.
pub fn refresh_repo_map(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    map: &RepoMap,
    prior: &DirState,
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<RefreshOutcome> {
    let units = build_units(map, workspace);
    if prior.equivalent_to(&units) {
        tracing::trace!("repo map refresh: no directory hash changed, skipping refresh");
        return Ok(RefreshOutcome::NoOp);
    }

    let changed = prior.changed_dirs(&units);
    tracing::info!(
        n_changed = changed.len(),
        sample_changed = ?changed.iter().take(5).collect::<Vec<_>>(),
        n_total_dirs = units.len(),
        "repo map refresh: re-ingesting changed directories",
    );

    let plan = IngestPlan::new(
        ctx.engine,
        &ctx.proj_builder,
        &ctx.config,
        layer_name,
        group_name,
    )?;
    let present: HashSet<&str> = units.iter().map(|u| u.dir.as_str()).collect();
    reconcile_deleted(ctx.engine, &present);
    let report = run_dir_pool(ctx.engine, &plan, workspace, &units, progress);
    crate::ingest_report::publish(PASS_NAME, report);
    Ok(RefreshOutcome::Replaced {
        state: dir_state_from_substrate(ctx.engine),
    })
}

/// The per-pass constants every worker needs to mint its unit's conversation:
/// the resolved layer/group ids, the shared system prompt, and the utility
/// config. Assembled once so the pool's per-unit signature stays small.
struct IngestPlan {
    layer: LayerId,
    group: GroupId,
    proj_builder: projection::Builder,
    system_prompt: String,
    config: SequenceConfig,
}

impl IngestPlan {
    fn new(
        engine: &Mutex<ConversationEngine>,
        proj_builder: &projection::Builder,
        config: &SequenceConfig,
        layer_name: &str,
        group_name: &str,
    ) -> anyhow::Result<Self> {
        let layer = proj_builder
            .id_for_layer(layer_name)
            .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
        let group = proj_builder
            .id_for_group(group_name)
            .ok_or_else(|| anyhow::anyhow!("projection schema missing '{group_name}' group"))?;
        // Append-only ingest layer (in-memory flag, re-applied every load): folder
        // summaries score self-local during ingest, so a summary is grounded in its
        // own folder rather than derailed by cross-directory retrieval.
        validate_summarize_branch(proj_builder)?;
        engine.lock().unwrap().mark_layer_append_only(layer);
        Ok(Self {
            layer,
            group,
            proj_builder: proj_builder.clone(),
            system_prompt: layer_system_prompt(proj_builder, layer_name, config),
            config: utility_config(config.clone()),
        })
    }
}

/// Tombstone every live `repo_map` conversation whose directory is no longer
/// present. Covers directories deleted while the daemon was down (the walk only
/// visits directories that still exist) and those removed between refreshes.
/// A still-present *changed* directory is handled by [`process_one_dir`], which
/// supersedes its own stale generation.
fn reconcile_deleted(engine: &Mutex<ConversationEngine>, present: &HashSet<&str>) {
    let e = engine.lock().unwrap();
    for (tl, dir) in e.conversations_with_metadata_key(DIR_KEY) {
        if !present.contains(dir.as_str()) {
            if let Err(err) = e.tombstone_timeline(tl) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %dir,
                    "tombstone of removed directory's conversation failed: {err:#}",
                );
            }
        }
    }
}

/// Metadata key holding a unit's directory — the invalidation-scan key. Distinct
/// from `code_read`'s `path` so the two layers' reconcile sweeps never touch each
/// other's conversations.
const DIR_KEY: &str = "dir";

/// Metadata key holding a unit's content hash — the resume-cache key, written
/// only after the unit's ingest succeeds.
const HASH_KEY: &str = "content_sha256";

/// Rebuild the [`DirState`] from what the substrate has ACTUALLY ingested,
/// joining each conversation's [`DIR_KEY`] and [`HASH_KEY`] metadata by timeline
/// — the same durable record `code_read` derives its state from.
///
/// The state must come from the substrate, never from the walk. A walk-derived
/// state records every directory the pass *attempted*, so a directory whose
/// ingest failed still gets its content hash stored as ingested; the next walk
/// then sees an unchanged hash, `equivalent_to` returns true, the refresh is a
/// `NoOp`, and the directory is never retried — silently absent from the repo
/// map for the life of the workspace, while `process_one_dir` logs that it will
/// be picked up next run. [`HASH_KEY`] is written only after a unit's ingest
/// succeeds, so joining on it records exactly what is really there.
fn dir_state_from_substrate(engine: &Mutex<ConversationEngine>) -> DirState {
    let e = engine.lock().unwrap();
    let hashes: HashMap<TimelineId, String> = e
        .conversations_with_metadata_key(HASH_KEY)
        .into_iter()
        .collect();
    let mut units: Vec<DirRecord> = e
        .conversations_with_metadata_key(DIR_KEY)
        .into_iter()
        .filter_map(|(tl, dir)| {
            hashes.get(&tl).map(|content_hash| DirRecord {
                dir,
                content_hash: content_hash.clone(),
            })
        })
        .collect();
    units.sort_by(|a, b| a.dir.cmp(&b.dir));
    DirState { units }
}

/// Drive a bounded worker pool over `units`: each worker pulls the next
/// directory from a shared cursor and runs [`process_one_dir`]. Workers share
/// the progress counter, a tolerated-failure counter, and an abort flag (the
/// first hard error stops the rest).
fn run_dir_pool(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    workspace: &Path,
    units: &[DirUnit],
    progress: &Arc<LoadProgress>,
) -> IngestReport {
    let total = units.len();
    progress.set_step_progress(0, total as u64);
    // Price the pool against the arenas it is about to add, not the ones it
    // inherits. Must precede the width sizing below, which reads the estimate.
    anchor_scan_kv_baseline();
    // One snapshot of the live hashes drives every worker's O(1) resume probe.
    let present_hashes = engine
        .lock()
        .unwrap()
        .conversation_metadata_values(HASH_KEY);

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    // Conversations this pool currently holds open — the quantity that actually
    // costs VRAM (each pins its K/V until its chain completes). Gates worker
    // claims via `reserve_scan_slot`.
    let live_convs = AtomicUsize::new(0);
    // Per-directory failures are RECORDED, never propagated: a directory that
    // fails keeps its prior generation live and the pass carries on, so one bad
    // folder (or a systemic VRAM squeeze) degrades the map instead of killing
    // the daemon. The cap still stops a flood — `failures.set_abort()` — but as
    // reported state, not as a fatal error. See `report`.
    let failures = Failures::new();

    std::thread::scope(|s| {
        // Size the pool to the CARD before spawning, not to a constant.
        //
        // The runtime gate alone cannot bound the opening burst: it reads the
        // scheduler's published memory report, and at scan start that report is
        // either absent or predates the scan, so every worker sees an empty pool
        // and claims. Measured: the gate computed `max_live=6` while 21
        // conversations were already open. Deciding the width once, up front,
        // removes the race entirely; the runtime gate then handles drift as
        // directories vary in size.
        let n_workers = max_live_conversations()
            .or_else(scan_width_from_governor)
            .unwrap_or(REPO_MAP_PARALLELISM);
        tracing::info!(
            target: "zend::repo_scan",
            n_workers,
            ceiling = REPO_MAP_PARALLELISM,
            "repo map pool width sized to available KV",
        );
        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers.max(1) {
            handles.push(s.spawn(|| {
                // One overlay context per worker: the prefilled `file_list`
                // response is produced by RUNNING the tool, so the listing the
                // model sees can never drift from the live tool's output.
                let ctx = ToolContext::with_workspace(workspace);
                loop {
                    // Stop before claiming the next directory on a first-error
                    // abort OR a shutdown cancel. The in-flight conversation
                    // finishes (the scheduler is still live), so the engine can
                    // drain rather than the shutdown losing the un-drained tier
                    // tail.
                    if failures.aborted() || candle_conversation::ingest_cancelled() {
                        return;
                    }
                    // Claim the unit BEFORE reserving a slot: a worker that
                    // finds the queue empty must not be holding one, or the
                    // last workers to notice would each pin a slot against the
                    // conversations still finishing.
                    let idx = cursor.fetch_add(1, Ordering::Relaxed);
                    if idx >= units.len() {
                        return;
                    }
                    // Probe the resume cache BEFORE queueing for VRAM. A hit
                    // opens no conversation and costs one hash lookup, so
                    // waiting on the KV gate for it buys nothing — and on a
                    // fully-cached restart every unit is a hit, which put ~700
                    // free lookups through a gate that admits a handful at a
                    // time.
                    let result = if present_hashes.contains(&units[idx].content_hash) {
                        tracing::debug!(
                            target: "zend::repo_scan",
                            dir = %units[idx].dir,
                            "skip: directory already in substrate (resume cache hit)",
                        );
                        Ok(())
                    } else {
                        // Bound open conversations by the card, not by the
                        // thread count — see `scan_width`. The guard holds the
                        // slot for the whole ingest, so no other worker can
                        // decide against this state, and an unwind cannot leak
                        // it from the process-global gauge.
                        let _slot = ScanSlot::reserve(&live_convs);
                        process_one_dir(engine, plan, &ctx, &units[idx], &failures)
                    };
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.set_step_progress(d as u64, total as u64);
                    // An error escaping `process_one_dir` is an unexpected one
                    // (its own two failure modes record and return Ok). Record
                    // it the same way so it lands in the report rather than
                    // vanishing, and let the cap decide whether to stop.
                    if let Err(e) = result {
                        let n = failures.record(&units[idx].dir, format!("{e:#}"));
                        tracing::warn!(
                            target: "zend::repo_scan",
                            dir = %units[idx].dir,
                            "directory ingest failed (will retry next run): {e:#}",
                        );
                        if n > MAX_DECODE_FAILURES {
                            failures.set_abort();
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("repo_map worker panicked");
        }
    });

    // Pin the bar to 100%: workers store from their own `done` snapshot without a
    // max, so the last stored value can settle a step short even though every
    // unit ran.
    progress.set_step_progress(total as u64, total as u64);
    let report = failures.into_report(total);
    // Say "incomplete" when it is incomplete. The old line said "complete" with
    // the failure count as a field, so a quarter-empty map read as success at a
    // glance; the first failure's cause is carried here too, so the summary
    // alone explains the pass without scrolling back through the warnings.
    if report.is_incomplete() {
        tracing::error!(
            target: "zend::repo_scan",
            n_dirs = total,
            n_failed = report.n_failed,
            aborted = report.aborted,
            first_failure = report.failures.first().map(|f| f.error.as_str()).unwrap_or("-"),
            first_failure_dir = report.failures.first().map(|f| f.unit.as_str()).unwrap_or("-"),
            "repo map per-directory ingest INCOMPLETE — affected directories keep \
             their prior generation and retry next pass (GET /v1/repo_map)",
        );
    } else {
        tracing::info!(n_dirs = total, "repo map per-directory ingest complete",);
    }
    report
}

/// Ingest one directory into a fresh conversation: render the folder's
/// round-trip chain, run it (two prefills + the summary decode), tag the
/// conversation, and free it.
///
/// The caller has already established this unit is not a resume-cache hit, and
/// holds its [`ScanSlot`] for the duration.
///
/// A per-directory ingest failure is TOLERATED up to [`MAX_DECODE_FAILURES`]:
/// the attempt's partial is tombstoned, the prior generation is left live, and
/// the unit simply misses the resume cache next run and is retried.
fn process_one_dir(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    ctx: &ToolContext,
    unit: &DirUnit,
    failures: &Failures,
) -> anyhow::Result<()> {
    // Render BEFORE minting anything: the tool responses come from actually
    // running the tools, so a directory the tools can't read is caught here and
    // costs no conversation. Prefilling an error body would be worse than
    // skipping — it teaches the model a tool interaction that failed.
    let (prefilled, decode_user) = render::render_chain(ctx, unit);
    if let Some(detail) = render::chain_error(&prefilled, &decode_user) {
        let n = failures.record(&unit.dir, format!("file_list failed: {detail}"));
        tracing::warn!(
            target: "zend::repo_scan",
            dir = %unit.dir,
            "skip: file_list failed for this directory ({detail}); not prefilling an error body",
        );
        if n > MAX_DECODE_FAILURES {
            tracing::error!(
                target: "zend::repo_scan",
                n, cap = MAX_DECODE_FAILURES,
                "repo map ingest stopping early: failure cap reached (last: {detail})",
            );
            failures.set_abort();
        }
        return Ok(());
    }

    // Cache miss → new / changed / crashed-partial directory. Reconcile this
    // directory's existing conversations WITHOUT invalidating good content up
    // front — a DEFERRED tombstone. A PARTIAL (carries `dir` but no
    // `content_sha256`: a crashed prior attempt) has nothing to lose and goes
    // now; a GOOD generation is deferred into `superseded`, staying live as the
    // folder's content until this ingest commits its own hash. Without the
    // deferral a failed re-ingest would destroy the only summary the folder has —
    // and two live generations would both vote in the same provenance scan.
    let (mut conv, superseded) = {
        let e = engine.lock().unwrap();
        let mut superseded = Vec::new();
        for tl in e.find_conversations_by_metadata(DIR_KEY, &unit.dir) {
            let is_good = e
                .conversation_metadata(tl)
                .is_some_and(|m| m.contains_key(HASH_KEY));
            if is_good {
                superseded.push(tl);
            } else if let Err(err) = e.tombstone_timeline(tl) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "tombstone of stale partial conversation failed: {err:#}",
                );
            }
        }
        let conv = e
            .new_conversation_with_projection(
                &plan.system_prompt,
                plan.proj_builder.clone(),
                plan.layer,
                plan.group,
                plan.config.clone(),
            )
            .map_err(|err| anyhow::anyhow!("repo_map conv create: {err}"))?;
        // The folder's closing turn is its own decoded summary, so the AVL
        // summariser must not compress these turns into a second summary tree.
        e.set_timeline_summarize(conv.timeline_id(), false);
        // The conversation carries no conv_id (it is not a dialogue), so without a
        // label the substrate viewer renders it as "(untitled)".
        if let Err(err) = e.set_conversation_label(conv.timeline_id(), &unit.dir) {
            tracing::warn!(target: "zend::repo_scan", "repo_map label set failed: {err:#}");
        }
        (conv, superseded)
    };

    // Tag `dir` IMMEDIATELY — before the decode below that can fail. A partial
    // left by such a failure then still names the directory it covers, so it (a)
    // shows in the substrate as that folder rather than "(untitled)", and (b) is
    // found by the invalidation scan above on the next run, which tombstones it
    // and retries. The resume-cache key is withheld until success, so a partial is
    // never mistaken for a completed ingest and skipped.
    {
        let mut early = BTreeMap::new();
        early.insert("kind".to_string(), "repo_map".to_string());
        early.insert(DIR_KEY.to_string(), unit.dir.clone());
        if let Err(e) = conv.set_metadata_many(&early) {
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                "failed to tag dir metadata at conversation creation: {e:#}",
            );
        }
    }

    // One chain on this conversation: request → list → read → DECODE. The
    // conversation projects its own turns (`target_is_ingest_self`), so the
    // request is in the decode's context where it belongs, and there is no
    // throwaway intermediate decode to set the wrong style.
    let force_tools: Vec<String> = render::CHAIN_TOOLS.iter().map(|t| t.to_string()).collect();
    let emit = {
        let mut sink = SequenceTurnSink::new(&mut conv);
        sink.ingest_chain(
            &prefilled,
            &decode_user,
            dir_tags(unit),
            FOLDER_SUMMARY_MAX_TOKENS,
            &force_tools,
        )
    };

    let tokens = match emit {
        Ok(tokens) => tokens,
        Err(e) => {
            // The deferred tombstone is the safety net: `superseded` was never
            // tombstoned, so the prior generation stays live and its resume hash
            // stays in the cache — this failed attempt invalidates nothing. Drop
            // only THIS attempt's partial; the retry re-mints cleanly.
            {
                let en = engine.lock().unwrap();
                if let Err(err) = en.tombstone_timeline(conv.timeline_id()) {
                    tracing::warn!(
                        target: "zend::repo_scan",
                        dir = %unit.dir,
                        "tombstone of failed-attempt partial failed: {err:#}",
                    );
                }
            }
            // If a graceful shutdown latched the cancel flag, this `Err` is the
            // interruptible decode-wait unwinding (`wait_cancellable` →
            // `IngestCancelled`), not a genuine ingest failure — the anyhow
            // layer has erased the variant, so the global flag is the source of
            // truth. Don't record it against the failure cap: a Ctrl-C with 24
            // workers in flight would otherwise book 24 failures at once, trip
            // the abort, and report the pass as incomplete. The partial was just
            // tombstoned, so the directory re-ingests next run.
            if candle_conversation::ingest_cancelled() {
                tracing::debug!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "shutdown cancelled decode mid-directory — dropped partial; will re-ingest next run",
                );
                return Ok(());
            }
            let n = failures.record(&unit.dir, format!("{e:#}"));
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                superseded_kept = superseded.len(),
                "directory ingest failed (will retry next run; prior generation kept live): {e:#}",
            );
            if n > MAX_DECODE_FAILURES {
                // Carry the CAUSE, not just the count: the fatal-looking line
                // used to say only "N failed (cap = M)", so every diagnosis
                // began by scrolling back through N warnings to find the root.
                tracing::error!(
                    target: "zend::repo_scan",
                    n, cap = MAX_DECODE_FAILURES,
                    "repo map ingest stopping early: failure cap reached (last: {e:#})",
                );
                failures.set_abort();
            }
            return Ok(());
        }
    };

    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), "repo_map".to_string());
    tags.insert(DIR_KEY.to_string(), unit.dir.clone());
    tags.insert(HASH_KEY.to_string(), unit.content_hash.clone());
    tags.insert("files".to_string(), unit.files.len().to_string());
    if let Some(a) = &unit.anchor {
        tags.insert("anchor".to_string(), a.path.clone());
    }
    // The tag write is what commits the new generation. If it fails, this
    // attempt has to go: keeping the prior generation live is right, but keeping
    // BOTH is not — the untagged replacement is invisible to the resume cache
    // and to the invalidation sweep, yet its turns are perfectly visible to
    // provenance, so the folder would vote twice in every scan from then on,
    // permanently. Drop it and let the unit retry, exactly like a decode
    // failure.
    if let Err(e) = conv.set_metadata_many(&tags) {
        {
            let en = engine.lock().unwrap();
            if let Err(err) = en.tombstone_timeline(conv.timeline_id()) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "tombstone of untagged replacement failed — TWO generations of this \
                     directory are now live: {err:#}",
                );
            }
        }
        let n = failures.record(&unit.dir, format!("metadata tag write failed: {e:#}"));
        tracing::warn!(
            target: "zend::repo_scan",
            dir = %unit.dir,
            superseded_kept = superseded.len(),
            "failed to tag conversation metadata (resume cache); dropped the replacement \
             and kept the prior generation: {e:#}",
        );
        if n > MAX_DECODE_FAILURES {
            tracing::error!(
                target: "zend::repo_scan",
                n, cap = MAX_DECODE_FAILURES,
                "repo map ingest stopping early: failure cap reached (last: {e:#})",
            );
            failures.set_abort();
        }
        return Ok(());
    }

    // Deferred tombstone ACTIVATES, now that the new generation is truly
    // committed (its hash landed above). Until this instant a projection saw the
    // prior generation (stale but present); from here it sees this one, and only
    // this one.
    if !superseded.is_empty() {
        let e = engine.lock().unwrap();
        for tl in &superseded {
            if let Err(err) = e.tombstone_timeline(*tl) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "deferred tombstone of superseded generation failed: {err:#}",
                );
            }
        }
    }

    // Nothing attends this folder again until a projection retrieves it. Flag it
    // for full KV eviction so the persistence pipeline offloads its turns to cold
    // and frees both the VRAM and RAM copies; `elevate_to_hot` pulls them back on
    // demand. `FreeSequence` on drop only releases the batch slot, not the sealed
    // KV, so this is what actually reclaims the space across a large scan.
    engine
        .lock()
        .unwrap()
        .evict_ingest_timeline(conv.timeline_id());
    tracing::debug!(
        target: "zend::repo_scan",
        dir = %unit.dir,
        tokens,
        "directory ingested (chain prefilled + summary decoded)",
    );
    Ok(())
}

/// The section-tree branch an ingest conversation frames on: the terse
/// code-summarization engine, with the worked request→summary examples stuffed
/// in. Node id → option id; a node not named here keeps its schema default.
///
/// This has to be resolved into the STATIC system prompt rather than left to a
/// runtime `Selection`: an ingest conversation is created with an explicit
/// prompt string and runs with `disable_reprojection` (see [`utility_config`]),
/// so it never re-projects and a later selection change can never materialise.
const SUMMARIZE_BRANCH: &[(&str, &str)] = &[
    // The conversational "You are Zen, pair programming…" frame makes the model
    // reason aloud, refuse, or chat; `summarize` pins content-is-provided,
    // English, summary-only.
    ("persona", "summarize"),
    // `standard` says "a short paragraph or two", which fights "two sentences".
    ("response_length", "terse"),
    // Worked examples of THIS ingest's round-trip. The shape teaches the
    // subject: shown the `code_reading` file examples, a folder decode
    // faithfully summarises the excerpt it was just handed ("The `mod.rs` file
    // outlines…") rather than the directory it was asked about.
    ("summarize_examples", "folder"),
    // An ingest turn supplies its content; there is nothing to reason about.
    ("thinking_effort", "off"),
];

/// Pull the layer's system prompt out of the schema and wrap it with the
/// engine's dialect markers.
///
/// Mirrors the dialogue layer's `pre_collection_prelude` (`session.rs`): fixed
/// sections verbatim, plus each section-tree node's *chosen* option — chosen
/// here by [`SUMMARIZE_BRANCH`] rather than by the tree's defaults, so the
/// conversation is framed as the summarizer it is.
fn layer_system_prompt(
    builder: &projection::Builder,
    layer_name: &str,
    config: &SequenceConfig,
) -> String {
    debug_assert!(
        builder.schema().layers.iter().any(|l| l.name == layer_name),
        "projection schema missing '{layer_name}' layer"
    );
    config
        .dialect
        .format_system_prompt(&ingest_prompt_body(builder))
}

/// The unwrapped body of [`layer_system_prompt`] — the schema walk, with no
/// dialect framing. Split out so the assembled prompt can be asserted directly.
fn ingest_prompt_body(builder: &projection::Builder) -> String {
    use projection::SystemPromptItem;
    let mut body = String::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => body.push_str(&s.content),
            SystemPromptItem::SectionTree(tree) => {
                let (selection, _) = summarize_selection(tree);
                for node in &tree.nodes {
                    // A collection node has no options of its own; its members
                    // are provenance-selected and live-prefilled at projection.
                    // Skip it and keep walking — the nodes below it are ordinary
                    // content, and stopping here (as the dialogue prelude does,
                    // its contract being "the text before the tools") would drop
                    // them.
                    if node.collection.is_some() {
                        continue;
                    }
                    // Structural markers (`<tools>` …) are generated at
                    // projection, never part of the static prelude.
                    if node.glue.is_some() {
                        continue;
                    }
                    if let Some(option) = node.options.get(node.chosen(&selection)) {
                        body.push_str(&option.content);
                    }
                }
            }
            // Same reasoning as the in-tree collection node above.
            SystemPromptItem::Collection(_) => continue,
        }
    }
    body
}

/// `tree`'s default selection with [`SUMMARIZE_BRANCH`] applied, plus the branch
/// entries this tree could not resolve.
///
/// A node the tree does not declare at all is not a miss — the branch spans two
/// trees, so each sees only its own nodes. A node that IS declared but lacks the
/// named option is a miss: the schema and this ingest disagree about what the
/// option is called, and the prompt silently loses that framing.
fn summarize_selection(tree: &projection::SectionTree) -> (Vec<u8>, Vec<String>) {
    let mut selection = tree.default_selection.clone();
    let mut unresolved = Vec::new();
    for (node_id, option_id) in SUMMARIZE_BRANCH {
        let Some(node) = tree.nodes.iter().find(|n| n.name == *node_id) else {
            continue;
        };
        let Some(dim) = node.dim else {
            continue; // mandatory node — one option, nothing to select
        };
        let Some(idx) = node.options.iter().position(|o| o.id == *option_id) else {
            let have: Vec<&str> = node.options.iter().map(|o| o.id.as_str()).collect();
            unresolved.push(format!(
                "node {node_id:?} has no option {option_id:?} (declares {have:?})"
            ));
            continue;
        };
        if let Some(slot) = selection.get_mut(dim) {
            *slot = idx as u8;
        }
    }
    (selection, unresolved)
}

/// Fail the ingest if the schema cannot supply the summarizer framing.
///
/// This is deliberately a hard error, not a warning. Without the branch the
/// decode runs as the dialogue agent and writes chat — "would you like me to
/// read any of these files?" — or an implementation plan, and every folder in
/// the layer is quietly worthless. A stale or hand-edited workspace
/// `projection.yaml` (`--working-dir`) is exactly how that happens, and it
/// happened: the bundled schema had the option, the workspace copy did not, and
/// three ingest runs produced garbage behind a single warning line.
fn validate_summarize_branch(builder: &projection::Builder) -> anyhow::Result<()> {
    use projection::SystemPromptItem;
    let unresolved: Vec<String> = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .filter_map(|i| match i {
            SystemPromptItem::SectionTree(t) => Some(t),
            _ => None,
        })
        .flat_map(|t| summarize_selection(t).1)
        .collect();
    if unresolved.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "projection schema cannot supply the repo_map summarizer framing: {}.          The ingest would decode chat instead of folder summaries — check the          workspace `projection.yaml` is in step with the bundled one.",
        unresolved.join("; "),
    ))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::types::Language;

    fn unit(dir: &str) -> DirUnit {
        DirUnit {
            dir: dir.to_string(),
            files: vec![FileEntry {
                path: format!("{dir}x.rs"),
                line_count: 1,
                language: Language::Rust,
                size_bytes: 1,
                module_hint: None,
            }],
            listed: vec![format!("{dir}x.rs")],
            anchor: None,
            content_hash: "abc".to_string(),
        }
    }

    /// The gather-scope tags name the layer and the exact directory, so a
    /// tag-scoped gallery admits one folder's turns and no other's.
    #[test]
    fn tags_carry_the_layer_and_the_directory() {
        assert_eq!(
            dir_tags(&unit("zend/src/")),
            vec!["repo_map".to_string(), "zend/src/".to_string()],
        );
    }

    #[test]
    fn the_root_unit_tags_with_a_usable_label() {
        assert_eq!(
            dir_tags(&unit(".")),
            vec!["repo_map".to_string(), ".".to_string()],
        );
    }

    /// Every tool the prefilled chain calls must be pinned into the catalog, or
    /// the projection presents a `<tool_call>` for a tool it never defined.
    #[test]
    fn every_tool_the_chain_calls_is_pinned() {
        for tool in render::CHAIN_TOOLS {
            assert!(
                zend_tools::registry::find(tool).is_some(),
                "{tool} is called by the chain but not registered",
            );
        }
    }

    /// Parse the bundled projection.yaml the way the daemon does.
    fn bundled_builder() -> projection::Builder {
        let dialect = candle_conversation::models::Dialect::chat_ml();
        projection::Builder::from_yaml_with_vars_and_dialect(
            include_str!("../prompts/projection.yaml"),
            &[("workspace", "test")],
            Some(&dialect),
        )
        .expect("projection.yaml must parse")
    }

    /// The system prompt an ingest conversation is actually created with — the
    /// artifact, not a proxy for it. It must frame the model as the summarizer
    /// and carry the worked examples, or the folder decode answers a
    /// summary request conversationally.
    #[test]
    fn the_ingest_prompt_is_framed_as_the_summarizer() {
        let builder = bundled_builder();
        let prompt = ingest_prompt_body(&builder);

        assert!(
            prompt.contains("You are a code-summarization engine"),
            "the summarize persona must be in the prompt",
        );
        assert!(
            !prompt.contains("You are Zen"),
            "the dialogue persona must NOT be — it is what produces chat replies",
        );
        // The examples must be the FOLDER shape and ONLY the folder shape. An
        // example teaches the subject as much as the format: shown a scope read,
        // the folder decode summarises the excerpt it was handed ("The `mod.rs`
        // file outlines…") instead of the directory it was asked about.
        assert!(
            prompt.contains("Summarize the `worker/scheduling/` folder"),
            "the worked FOLDER examples must survive the walk past the tool catalog",
        );
        assert!(
            prompt.contains("This folder throttles requests per tenant"),
            "both folder examples must be present",
        );
        assert!(
            !prompt.contains("Jitter returns"),
            "the code_reading FILE examples must NOT be — they teach the wrong subject",
        );
    }

    /// A schema that cannot supply the summarizer framing must FAIL the ingest,
    /// not warn. A stale workspace `projection.yaml` did exactly this: the
    /// bundled schema declared the option, the `--working-dir` copy did not, and
    /// three ingest runs decoded chat instead of folder summaries behind a single
    /// warning line.
    #[test]
    fn the_bundled_schema_supplies_the_whole_summarizer_branch() {
        validate_summarize_branch(&bundled_builder())
            .expect("the bundled schema must declare every SUMMARIZE_BRANCH option");
    }

    /// The same schema with the examples option renamed — the exact shape of the
    /// stale-copy bug — is rejected, naming what is missing.
    #[test]
    fn a_schema_missing_a_branch_option_is_rejected() {
        let yaml = include_str!("../prompts/projection.yaml").replace(
            "            - id: folder
",
            "            - id: renamed_away
",
        );
        let dialect = candle_conversation::models::Dialect::chat_ml();
        let builder = projection::Builder::from_yaml_with_vars_and_dialect(
            &yaml,
            &[("workspace", "test")],
            Some(&dialect),
        )
        .expect("still parses");

        let err = validate_summarize_branch(&builder)
            .expect_err("a schema without the folder examples must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("summarize_examples"), "{msg}");
        assert!(msg.contains("folder"), "names the missing option: {msg}");
    }

    /// `repo_map` keys its invalidation sweep on `dir`, `code_read` on `path`.
    /// If they shared a key, each layer's reconcile would tombstone the other's
    /// conversations (a directory is never in the file walk, and vice versa).
    #[test]
    fn the_invalidation_key_is_distinct_from_code_reads() {
        assert_eq!(DIR_KEY, "dir");
        assert_ne!(DIR_KEY, "path");
    }

    /// Measured report from a `repo_map` pass on the 16 GiB card, 21 directories
    /// in: capacity 13.17 GiB, scratch margin 1 GiB, `pool_used` 9.67 GiB, KV
    /// arenas 2.31 GiB.
    const CAPACITY: u64 = 14_143_193_088;
    const SCRATCH: u64 = 1_073_741_824;
    const POOL_USED: u64 = 10_380_898_464;
    const KV: u64 = 2_483_027_968;
    /// Arenas standing before the first directory opened — tool sections, the
    /// base builder's prefill, and the calibration exemplars.
    const PRE_SCAN: u64 = 2_000_000_000;

    /// The estimate must price the KV a scan ADDS. Charged the whole process's
    /// arenas instead, the per-conversation estimate pinned to its ceiling — and
    /// the error compounds, since a narrower pool divides the same fixed corpus
    /// into a still-larger estimate.
    #[test]
    fn the_estimate_prices_the_scans_own_kv_not_the_inherited_corpus() {
        assert!(per_conversation_kv(KV, PRE_SCAN, 2) < per_conversation_kv(KV, 0, 2));
        assert_eq!(per_conversation_kv(KV, 0, 2), SCAN_CONV_KV_MAX);
    }

    /// ...and the numerator must price it the same way. `fixed` is
    /// `pool_used - kv`, so subtracting only `fixed` returns every KV byte as
    /// free room — including the inherited corpus the denominator deliberately
    /// excludes. Growing that corpus leaves the same room for new work, so the
    /// pool must get narrower; unfixed, it stayed exactly as wide.
    #[test]
    fn the_inherited_corpus_is_not_free_room() {
        const EXTRA: u64 = 2 * 1024 * 1024 * 1024;
        let base = scan_width(CAPACITY, SCRATCH, POOL_USED, KV, PRE_SCAN, 2);
        let with_corpus = scan_width(
            CAPACITY,
            SCRATCH,
            POOL_USED + EXTRA,
            KV + EXTRA,
            PRE_SCAN + EXTRA,
            2,
        );
        assert!(with_corpus < base, "{with_corpus} vs {base}");
    }

    /// Holding back a FRACTION of capacity double-charges the expert cache:
    /// `fixed` subtracts it explicitly, then the fraction holds back a share of
    /// capacity that is mostly the same bytes again. On the measured report a
    /// 0.70 fraction left 1.87 GiB for scan KV against the governor's own
    /// ~5 GiB KV floor — under half the room, and the pool ran that much
    /// narrower for it.
    #[test]
    fn the_margin_held_back_is_the_governors_not_a_fraction_of_capacity() {
        let per_conv = per_conversation_kv(KV, PRE_SCAN, 2);
        let held_by_fraction = CAPACITY - ((CAPACITY as f64) * 0.70) as u64;
        assert!(held_by_fraction > SCRATCH * 3, "{held_by_fraction}");
        let by_fraction =
            (((CAPACITY as f64) * 0.70) as u64).saturating_sub(POOL_USED - KV + PRE_SCAN) / per_conv;
        let by_governor = scan_width(CAPACITY, SCRATCH, POOL_USED, KV, PRE_SCAN, 2) as u64;
        assert_eq!(by_fraction, 0);
        assert_eq!(by_governor, 6);
    }

    /// The floor charges a conversation for RESERVED arena, not for the live
    /// half — and that is the width the card actually sustains.
    ///
    /// Pricing only the live ~240 MiB (on the theory that slab waste belongs to
    /// a format rather than to a conversation) opens 19 directories on this
    /// report and walks straight into the wall: measured, 15 directories lost in
    /// one millisecond, a device OOM, 18 MiB free, and a 1415 MiB gap between
    /// reserved and used arena that is precisely the overhead the halving
    /// assumed away. Ten is the width that ran clean, and faster.
    #[test]
    fn the_floor_charges_reserved_arena_not_just_the_live_half() {
        assert_eq!(SCAN_CONV_KV_MIN, 480 * 1024 * 1024);
        let for_kv = CAPACITY - SCRATCH - (POOL_USED - KV) - PRE_SCAN;
        assert_eq!(for_kv / SCAN_CONV_KV_MIN, 6);
        // Halving the floor nearly doubles the width — the same argument that
        // opened 19 directories on this card and walked into the wall.
        assert_eq!(for_kv / (256 * 1024 * 1024), 11);
    }

    /// Growing the inherited corpus must not inflate the per-conversation
    /// estimate: the anchor moves with it, so the same in-flight KV is priced
    /// the same however large the corpus underneath it grows.
    #[test]
    fn a_larger_inherited_corpus_does_not_inflate_the_estimate() {
        const EXTRA: u64 = 4 * 1024 * 1024 * 1024;
        assert_eq!(
            per_conversation_kv(KV, PRE_SCAN, 2),
            per_conversation_kv(KV + EXTRA, PRE_SCAN + EXTRA, 2),
        );
    }

    /// A pool holding nothing open has added nothing, so the estimate is the
    /// measured floor rather than a division over a corpus it never created.
    #[test]
    fn an_idle_pool_estimates_at_the_floor() {
        assert_eq!(per_conversation_kv(9_000_000_000, 0, 0), SCAN_CONV_KV_MIN);
        assert_eq!(
            per_conversation_kv(9_000_000_000, 9_000_000_000, 4),
            SCAN_CONV_KV_MIN,
        );
        // Demotion can hand back more than the scan added; the delta floors.
        assert_eq!(
            per_conversation_kv(1_000, 9_000_000_000, 4),
            SCAN_CONV_KV_MIN
        );
    }

    /// The estimate stays inside its measured clamps at both ends: a cold start
    /// cannot admit unboundedly, and one atypical directory cannot stall the
    /// scan by pricing every later one off the card.
    #[test]
    fn the_per_conversation_estimate_stays_within_its_measured_clamps() {
        assert_eq!(
            per_conversation_kv(4 * SCAN_CONV_KV_MAX, 0, 1),
            SCAN_CONV_KV_MAX,
        );
        assert_eq!(
            per_conversation_kv(SCAN_CONV_KV_MIN / 4, 0, 1),
            SCAN_CONV_KV_MIN,
        );
        assert_eq!(
            per_conversation_kv(2 * SCAN_CONV_KV_MIN, 0, 2),
            SCAN_CONV_KV_MIN,
        );
    }

    /// The bound is a width, so it never reaches zero — a pool that may not open
    /// a conversation can never free the VRAM it is waiting on — and never
    /// exceeds the thread count the pool actually spawns.
    #[test]
    fn the_width_stays_between_one_and_the_pool_ceiling() {
        assert_eq!(scan_width(CAPACITY, SCRATCH, 14_000_000_000, 0, 0, 0), 1);
        // A margin wider than the card leaves nothing, and still not zero.
        assert_eq!(scan_width(CAPACITY, 2 * CAPACITY, 0, 0, 0, 0), 1);
        assert_eq!(
            scan_width(1024 * 1024 * 1024 * 1024, SCRATCH, 0, 0, 0, 0),
            REPO_MAP_PARALLELISM,
        );
    }
}
