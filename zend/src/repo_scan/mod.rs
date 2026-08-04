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

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{self, GroupId, LayerId};
use candle_conversation::{ConversationEngine, SequenceConfig};
use zend_tools::ToolContext;

use crate::ingest_report::{Failures, IngestReport};
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::turn_sink::{InsertTurnSink, SequenceTurnSink};

pub use binary_sniff::is_binary_sample;
pub use dir_unit::{build_units, DirState, DirUnit};
pub use types::{FileEntry, Language, RepoMap};
pub use walk::{walk_workspace, MAX_FILE_BYTES};

/// Directories ingested concurrently. Each unit is ONE conversation running a
/// short chain (two prefills + one bounded decode), so this is the whole
/// concurrent conversation count for the layer.
///
/// Sized to keep the DECODE row-group fed, not to widen prefill. A directory's
/// worker is a three-phase state machine — prefill, prefill, decode — and
/// `promote_new_prefills` admits exactly one prefill while
/// `vram_under_pressure()` holds, so workers queued behind it contribute no
/// decode work while they wait. At 8, 45% of waves ran a prefill with the decode
/// row idle; the extra width exists to fill those. It cannot make prefill
/// itself faster — that floor is one submission at a time.
///
/// 24 matches the scheduler's own `MAX_PREFILL_WIDTH` ceiling, so the pool never
/// asks for more concurrency than a wave can carry, and sits below `code_read`'s
/// effective 48 (12 file workers × 4 forked scopes). The trade is VRAM: every
/// admitted conversation pins its K/V, which feeds the same pressure that caps
/// prefill — so on a tight card the useful width is bounded by eviction churn
/// rather than by this constant.
pub const REPO_MAP_PARALLELISM: usize = 24;

/// Fraction of the governor's KV capacity above which a worker must not OPEN a
/// new directory conversation.
///
/// [`REPO_MAP_PARALLELISM`] bounds worker threads, not VRAM — and the thing that
/// costs VRAM is an *open conversation*, because it pins its K/V until its chain
/// completes. While a conversation is live its K sits in `R16` (4 bytes/element)
/// and its V in F16, so a directory costs several times what the same turns cost
/// once sealed and quantized.
///
/// 24 of those at once is right for a large card and far too many for a 16 GiB
/// one whose expert cache already holds 7.5 GiB: measured, the pool reached
/// 5.6 GiB of KV — 94% of it unsealed — and drove `pool_used` into the arena
/// allocator's ceiling, refusing directory after directory. Admission cannot
/// help: it throttles prefill *submission*, and by then the conversation exists
/// and its K/V is already pinned.
///
/// So the pool is bounded by the card instead of by a constant. A worker about
/// to claim a unit waits while the pool is above this fraction, which lets the
/// in-flight conversations seal, quantize, and drain first. Self-regulating: it
/// adapts to whatever the expert cache and model leave behind, on any card.
const SCAN_POOL_HIGH_WATER: f64 = 0.70;

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
    let vram = report.vram?;
    let capacity = vram.governor?.capacity_bytes;
    if capacity == 0 {
        return None;
    }
    let kv = report
        .kv
        .float_reserved_bytes
        .saturating_add(report.kv.quant_reserved_bytes);
    // What the pool holds that no scan decision can release.
    let fixed = vram.pool_used_bytes.saturating_sub(kv);
    let limit = ((capacity as f64) * SCAN_POOL_HIGH_WATER) as u64;
    let for_kv = limit.saturating_sub(fixed);
    // Per-conversation KV, measured rather than assumed. Clamped so a cold start
    // cannot admit unboundedly and one atypical directory cannot stall the scan.
    let live = SCAN_LIVE_CONVS.load(Ordering::Relaxed);
    let per_conv = if live > 0 && kv > 0 {
        (kv / live as u64).clamp(SCAN_CONV_KV_MIN, SCAN_CONV_KV_MAX)
    } else {
        SCAN_CONV_KV_MIN
    };
    Some(((for_kv / per_conv.max(1)) as usize).clamp(1, REPO_MAP_PARALLELISM))
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

/// Give the slot back. Must run on every exit path from a reserved section, or
/// the gauges drift up and the pool throttles itself toward a width of one.
fn release_scan_slot(live: &AtomicUsize) {
    live.fetch_sub(1, Ordering::Relaxed);
    SCAN_LIVE_CONVS.fetch_sub(1, Ordering::Relaxed);
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
    let state = DirState::from_units(&units);

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
    Ok((map, state, report))
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
        state: DirState::from_units(&units),
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
                    // Bound open conversations by the card, not by the thread
                    // count — see `SCAN_POOL_HIGH_WATER`. Returns holding the
                    // slot, so no other worker can decide against this state.
                    reserve_scan_slot(&live_convs);
                    let result = process_one_dir(
                        engine,
                        plan,
                        &ctx,
                        &units[idx],
                        &present_hashes,
                        &failures,
                    );
                    release_scan_slot(&live_convs);
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

/// Ingest one directory into a fresh conversation: skip via the resume-cache
/// snapshot if its content hash is already live; otherwise render the folder's
/// round-trip chain, run it (two prefills + the summary decode), tag the
/// conversation, and free it.
///
/// A per-directory ingest failure is TOLERATED up to [`MAX_DECODE_FAILURES`]:
/// the attempt's partial is tombstoned, the prior generation is left live, and
/// the unit simply misses the resume cache next run and is retried.
fn process_one_dir(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    ctx: &ToolContext,
    unit: &DirUnit,
    present_hashes: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    if present_hashes.contains(&unit.content_hash) {
        tracing::debug!(
            target: "zend::repo_scan",
            dir = %unit.dir,
            "skip: directory already in substrate (resume cache hit)",
        );
        return Ok(());
    }

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
    let committed = match conv.set_metadata_many(&tags) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                "failed to tag conversation metadata (resume cache): {e:#}",
            );
            false
        }
    };

    // Deferred tombstone ACTIVATES — but only once the new generation is truly
    // committed (its hash landed above). If that write failed the replacement
    // isn't resume-cached, so treat it as not-yet-committed and keep the prior
    // generation live rather than swapping to an untagged replacement. On success
    // the swap completes here: until this instant a projection saw the prior
    // generation (stale but present); from here it sees this one, and only this
    // one.
    if committed && !superseded.is_empty() {
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
}
