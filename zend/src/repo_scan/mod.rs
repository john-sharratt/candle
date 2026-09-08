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
pub mod metadata;
pub mod probe;
pub mod probe_pass;
pub mod render;
pub mod types;
pub mod walk;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::memory_report::AdmissionSection;
use candle_conversation::projection::{self, GroupId, LayerId, TimelineId};
use candle_conversation::stencil::{ThinkMode, ToolCallEnvelope, TriggerRegistry};
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
/// **This is the worker-thread count, not the number of open conversations.**
/// A worker blocked in [`reserve_scan_slot`] holds no conversation and no K/V;
/// how many are open at once is decided per claim by [`gate_decision`], from
/// what the engine publishes. It was previously 24, matched to the scheduler's
/// `MAX_PREFILL_WIDTH` on the reasoning that the pool should never ask for more
/// concurrency than a wave can carry — but that ceiling is a *prefill*
/// backstop, and it was sizing the pool for the wrong phase.
///
/// Measured over a full workspace ingest (Qwen3.6-35B-A3B, 201 waves): decode is
/// **61% of the phase's wall time against prefill's 24%**, and decode has no
/// width cap of its own. At a pool of 24 both phases averaged only ~9 wide
/// (max 23) — roughly a third of workers in a forward at any instant, since each
/// unit's chain is two prefills then a decode. Prefill is capped at 24 per wave
/// regardless, so workers beyond that queue for prefill but still add to decode
/// width, which is the phase that dominates.
///
/// Decode throughput is width-bound, not depth-bound: banding the same waves by
/// width gives 35 t/s at 1.2 sequences, 124 at 8.7, and 429 at 18.4, while
/// per-forward time FALLS from 338 ms to 43 ms — the MoE expert-weight load
/// amortising across the batch, exactly as the prefill note above describes.
///
/// **96 is a measured optimum, not a headroom guess.** Swept on a 72 GB card
/// (RTX PRO 5000, governor capacity 70.7 GB), whole-workspace ingest, decode
/// rate measured against forward time:
///
/// | ceiling | decode t/s | decode width | ingest phase |
/// |--------:|-----------:|-------------:|-------------:|
/// |      24 |      122.3 |          8.6 |        563 s |
/// |      64 |      240.2 |         32.6 |        453 s |
/// |  **96** |  **288.4** |     **56.2** |    **446 s** |
/// |     128 |      265.2 |         28.4 |        468 s |
///
/// 128 REGRESSES: the achieved width collapses to 28 because the extra workers
/// contend for admission rather than adding concurrency, so more of them sit
/// blocked than decoding. Past the knee, raising this number costs throughput.
///
/// The trade is still VRAM: every admitted conversation pins its K/V, so on a
/// tight card the useful width is bounded by eviction churn — and by the
/// warm-tier drain, which an over-wide ingest can outrun.
///
/// **Caveat worth knowing before changing this.** On the 72 GB card the engine's
/// gate has never bound: the pool logged `n_workers == ceiling` at every value
/// swept above (24, 64, 96, 128), so the ceiling — not the engine's signal — is
/// what actually limits width there, and the 128 regression was found by
/// throughput rather than refused by the gate. On a smaller card the gate binds
/// and picks the width; on a large one, treat this constant as the live limit.
pub const REPO_MAP_PARALLELISM: usize = 96;

/// **Removed, deliberately: the queue-length and open-conversation marks.**
///
/// Both scaled from the engine's `wave_width` and both were flow control. The
/// open mark held the pool at ~21 conversations against a mark of 15-19 for 409
/// seconds of one run, starving the queue to a mean depth of 6 against a
/// ten-wide wave — while the failure it was written to prevent (free KV regions
/// reaching zero) happened anyway at a *constant* 21 open, and cleared again at
/// the same 21. It cost throughput and protected nothing.
///
/// What they were reaching for is real: enough open conversations will push the
/// elastic boundary down and evict the resident experts. [`gate_decision`] now
/// reads that directly, as the weight zone against its hold, instead of
/// counting conversations as a stand-in for it.
/// The engine's admission section, or `None` when nothing fresh has been
/// published (the normal state at scan start).
///
/// A stale report is worse than none: pacing on figures from ten seconds ago
/// would let the pool run away exactly when the engine has stalled. Stale is
/// measured against the engine's own cadence — two of its publishing intervals
/// plus a wave's worth of slack — not a constant: a fixed five seconds read a
/// report one wave old as stale for 95 fills of one run, with waves running
/// three to five seconds, and held the producer while the queue ran empty.
fn engine_admission() -> Option<AdmissionSection> {
    let (report, age_ms) = candle_conversation::memory_report::latest()?;
    let cadence = report.admission.publish_interval_ms.max(1_000);
    if age_ms > 2 * cadence + 2_000 {
        // At most one line a second across every worker: the numbers are what
        // matter, and every held worker asks ten times a second.
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1)) {
            *last = Some(std::time::Instant::now());
            tracing::debug!(
                target: "zend::repo_scan",
                age_ms,
                cadence_ms = cadence,
                "scan pool: engine report is stale",
            );
        }
        return None;
    }
    Some(report.admission)
}

/// Why the gate is holding a worker back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hold {
    /// This pool has opened conversations the engine's report does not yet
    /// show, beyond the ramp slack — the report is a wave behind, so the pool
    /// would be opening blind.
    Unreflected { live: usize, reflected: usize },
    /// The pool has run away: more conversations open than any healthy state
    /// explains. See [`SCAN_RUNAWAY_CEILING`].
    Runaway { live: usize, ceiling: usize },
    /// The weight zone has been pushed down to the hold: one more conversation
    /// would come out of the resident experts.
    Residency { zone_mib: u64, hold_mib: u64 },
}

impl fmt::Display for Hold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Hold::Unreflected { live, reflected } => write!(
                f,
                "{live} open here but the engine reports {reflected} — waiting for it to catch up"
            ),
            Hold::Runaway { live, ceiling } => write!(
                f,
                "{live} conversations open, past the runaway ceiling of {ceiling}"
            ),
            Hold::Residency { zone_mib, hold_mib } => write!(
                f,
                "weight zone {zone_mib}MiB is down to its {hold_mib}MiB hold — \
                 one more would evict resident experts"
            ),
        }
    }
}

/// Conversations this pool may run ahead of the engine's report of them.
///
/// The report publishes at the wave cadence, ~2 s; a worker opens and submits
/// in well under that. Without this bound the gate reads the same "empty"
/// report for every worker in the burst and admits them all — measured, 96
/// opened in the first second of a pass, which is the herd every other signal
/// exists to prevent. Four keeps the pool feeding a report interval ahead
/// without ever being more than one interval blind.
const SCAN_OPEN_SLACK: usize = 4;

/// Open conversations past which this pool is malfunctioning rather than busy.
///
/// **Runaway protection only — never tune this for throughput.** It exists so a
/// bug (a worker loop that never closes a conversation, a report that never
/// arrives) cannot open conversations without bound. It is deliberately far
/// above any healthy state: a normal pass on the 16 GB card sits around twenty,
/// and the worst measured pathology reached eighty-three. If a run is ever held
/// here, that is a defect to find, not a number to lower.
///
/// It is a flat count and not scaled from the card, the model or the engine's
/// width, because it is not sizing anything — the wave's own claim-and-refuse
/// does the sizing on every machine. A larger card simply never approaches it.
const SCAN_RUNAWAY_CEILING: usize = 256;

/// Whether a worker may open a conversation while `live` are already open.
///
/// # This gate is runaway protection. It is NOT flow control.
///
/// **Never pace the wave from here.** The wave paces itself: every admission in
/// `interleave::fill` claims that sequence's KV and its per-sequence model state
/// through the real allocators, and a refusal is the device itself saying the
/// wave is as wide as it carries. Nothing this producer can compute adds to
/// that answer, and anything it computes *instead* is a proxy for it.
///
/// Every proxy tried here has been falsified on hardware
/// (`docs/wave_feeder.md` §4.5–§4.11): a residency estimate, a controller
/// climbing open conversations against measured throughput, a backlog
/// watermark, a queue-length mark, and an open-conversation mark scaled from
/// the engine's wave width. The last of those is why this function is now four
/// lines. It held the pool at ~21 conversations against a mark of 15-19 for
/// **409 seconds** of a single run, starving the queue to a mean depth of 6
/// against a ten-wide wave — while the failure it claimed to prevent happened
/// anyway, at a *constant* 21 open: free KV regions went 41 → 0 → 149 through
/// a collapse and a recovery without the open count moving at all. It was
/// throttling throughput and protecting nothing.
///
/// The deeper reason it could not work: `wave_width` is measured from the
/// expert-union curve of one forward — how many rows fit in a single step. How
/// many conversations may safely be *open* is a different quantity entirely,
/// and scaling one from the other is the proxy error this whole design arc is a
/// catalogue of.
///
/// # What is left, and why each is not flow control
///
/// * `live == 0` always opens. A pool whose only route to freeing the device is
///   finishing work it is not allowed to start would deadlock.
/// * [`SCAN_RUNAWAY_CEILING`] — an absolute bound on unbounded growth, far
///   above any healthy state, so it never binds in normal operation.
/// * The ramp bound below — how fast conversations may open *between reports*,
///   so the pool does not open as a herd against one stale reading. A bound on
///   rate, not on capacity.
/// * The weight zone against its hold — the one resource that opening a
///   conversation can spend irrecoverably, because K/V and the expert weights
///   share one elastic span. This is a *measured* quantity the engine
///   publishes, not a count standing in for it.
///
/// If the engine cannot keep up, that shows as the allocators refusing claims,
/// and the fix belongs there — in what the engine can shed — not in a producer
/// that declines to hand it work.
fn gate_decision(live: usize, admission: Option<&AdmissionSection>) -> Result<(), Hold> {
    if live == 0 {
        return Ok(());
    }
    if live >= SCAN_RUNAWAY_CEILING {
        return Err(Hold::Runaway {
            live,
            ceiling: SCAN_RUNAWAY_CEILING,
        });
    }
    // How far the pool may run ahead of the report, which is a RAMP bound, not
    // a capacity one: it says how fast conversations may be opened between two
    // reports, never how many may exist. Without it the whole pool reads the
    // same stale "empty" report and opens as one — measured, 96 in the first
    // second of a pass. No report at all is the same condition with nothing
    // reflected yet, so it ramps from zero rather than being a special case.
    let reflected = admission.map_or(0, |a| a.open_slots);
    let burst = admission.map_or(SCAN_OPEN_SLACK, |a| (a.wave_width / 2).max(SCAN_OPEN_SLACK));
    if live > reflected.saturating_add(burst) {
        return Err(Hold::Unreflected { live, reflected });
    }
    // **The one resource this gate spends, measured rather than modelled.**
    // The KV side and the weight side share one elastic span, so every open
    // conversation's K/V comes out of somewhere — and once the zone is down to
    // its hold, what it comes out of is the resident experts. An engine that
    // streams its experts is slower at everything, including finishing the
    // conversations that would give the ground back, so this is the one place
    // holding actually buys something.
    //
    // Above the hold the gate opens: headroom is there to be used, and the
    // wave's own claim-and-refuse decides what fits in any given forward.
    if let Some(a) = admission {
        if a.weight_hold_bytes > 0 && a.weight_zone_bytes <= a.weight_hold_bytes {
            return Err(Hold::Residency {
                zone_mib: a.weight_zone_bytes >> 20,
                hold_mib: a.weight_hold_bytes >> 20,
            });
        }
    }
    Ok(())
}

/// Block until the engine has room for another open conversation, then COUNT
/// THIS ONE IN before releasing the gate.
///
/// `live` is the count of conversations this pool currently holds open. The
/// decision is [`gate_decision`]; this is the loop around it.
///
/// The reservation happens under the SAME lock as the decision, and that is the
/// whole point. Deciding under the lock and incrementing after it re-opens the
/// race the gate exists to close: every waiting worker reads the same
/// pre-increment count, each concludes there is room for one more, and they all
/// proceed — a narrower replay of the herd that put exactly 24 conversations on
/// the card at once and produced `n_failed = 24` under every configuration.
///
/// **There is no wait cap.** One existed — twenty seconds, after which a worker
/// took its slot regardless — and it is how 95 conversations came to be open
/// against a limit of 7: every hold expired into an admission. A hold here ends
/// only when the engine says there is room, and there is no state in which that
/// never happens: the conversations already open are being stepped (or the
/// engine reports them starved, and steps them as ground frees), and every one
/// that finishes gives its ground back.
fn reserve_scan_slot(live: &AtomicUsize) {
    static GATE: Mutex<()> = Mutex::new(());
    let started = std::time::Instant::now();
    let mut last_logged: Option<Hold> = None;
    let mut last_log_at = std::time::Instant::now();
    loop {
        let hold = {
            let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
            let now = live.load(Ordering::Relaxed);
            match gate_decision(now, engine_admission().as_ref()) {
                Ok(()) => {
                    take_scan_slot(live);
                    if last_logged.is_some() {
                        tracing::debug!(
                            target: "zend::repo_scan",
                            held_ms = started.elapsed().as_millis() as u64,
                            live_conversations = live.load(Ordering::Relaxed),
                            "scan pool: released after holding",
                        );
                    }
                    return;
                }
                Err(hold) => hold,
            }
        };
        // Log the hold when it starts, when its reason changes, and every ten
        // seconds while it lasts — a hold logged once at its start was 86
        // workers reading "mark of 10" for twenty minutes in a run whose gate
        // had long since said something else.
        let changed = last_logged != Some(hold);
        if changed || last_log_at.elapsed() >= std::time::Duration::from_secs(10) {
            last_logged = Some(hold);
            last_log_at = std::time::Instant::now();
            tracing::debug!(
                target: "zend::repo_scan",
                live_conversations = live.load(Ordering::Relaxed),
                held_ms = started.elapsed().as_millis() as u64,
                %hold,
                "scan pool: holding before opening another directory",
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Count one conversation in. Called only with the reservation gate held.
fn take_scan_slot(live: &AtomicUsize) {
    live.fetch_add(1, Ordering::Relaxed);
}

/// Holds one admitted conversation's slot, and gives it back on drop.
///
/// A guard rather than a paired release call, so a slot lost to an unwind is
/// not lost for the pass: one panic inside an ingest — where the failure paths
/// already tolerate a directory going wrong — would otherwise leave the gate
/// believing a conversation is open for as long as the pool runs.
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
    wipe_metadata: bool,
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

    // The directory-frequency index spans the WHOLE workspace and is built
    // before any directory is ingested. A document frequency computed over a
    // partial corpus is meaningless — every term looks rare when most
    // directories have not been counted — so this cannot be built incrementally
    // as the pool advances.
    let index = Arc::new(probe::idf::TermIndex::build(&units, &map));
    tracing::info!(
        target: "zend::repo_scan::probe",
        n_dirs = index.n_dirs(),
        rarity_gate = index.rarity_gate(),
        n_symbol_files = map.symbols.len(),
        "probe term index built",
    );

    // `--wipe-metadata`: drop every `.substrate.yaml` before the skeletons are
    // seeded, so this boot regenerates them all. Keyed on the walked units, so
    // it can only remove files in directories this workspace actually walked.
    if wipe_metadata {
        let removed = metadata::wipe(workspace, &units);
        tracing::info!(
            target: "zend::repo_scan::metadata",
            removed,
            "--wipe-metadata: folder metadata deleted; it will be regenerated",
        );
    }
    // Seed a skeleton in every folder that has none, so the shape is
    // discoverable: a person browsing the tree finds the file already there with
    // its registers named, rather than having to know it could exist. Existing
    // files are never touched.
    let seeded = metadata::seed_skeletons(workspace, &units, |unit| {
        index
            .distinctive(&unit.dir, probe::render::seed_count())
            .into_iter()
            .map(str::to_string)
            .collect()
    });
    let authored = units
        .iter()
        .filter_map(|u| metadata::load(workspace, u))
        .filter(|m| m.is_complete())
        .count();
    tracing::info!(
        target: "zend::repo_scan::metadata",
        seeded,
        complete = authored,
        of = units.len(),
        "folder metadata ready ({} directories need no generation)",
        authored,
    );

    let plan = IngestPlan::new(
        engine,
        &proj_builder,
        &config,
        layer_name,
        group_name,
        Arc::clone(&index),
    )?;
    // Retire conversations for directories that no longer exist, then snapshot
    // the surviving hashes once for O(1) per-unit resume-cache probes.
    // Crashed partials are NOT swept here. The sweep has to run whether or not
    // this pass does, so it belongs to the caller — see
    // [`retire_crashed_partials`].
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

    let index = Arc::new(probe::idf::TermIndex::build(&units, map));
    // A refresh never wipes: it re-ingests changed directories only, and the
    // metadata for every unchanged one is exactly what it should keep.
    metadata::seed_skeletons(workspace, &units, |unit| {
        index
            .distinctive(&unit.dir, probe::render::seed_count())
            .into_iter()
            .map(str::to_string)
            .collect()
    });
    let plan = IngestPlan::new(
        ctx.engine,
        &ctx.proj_builder,
        &ctx.config,
        layer_name,
        group_name,
        index,
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
    /// System prompt for the throwaway probe-generation conversations — the
    /// question-writer framing rather than the summariser's.
    probe_prompt: String,
    /// System prompt for the per-folder probe-ANSWERING conversations, framed to
    /// answer with a `<think>` block rather than to summarise without one.
    answer_prompt: String,
    config: SequenceConfig,
    /// `<think>` bound to [`ThinkMode::Off`]'s tree, for every folder summary
    /// decode in the pass.
    ///
    /// Compiled ONCE here rather than per directory: the compile builds an
    /// `HfVocab`, which clones the tokenizer — a 12 MB copy per folder for a
    /// tree that is identical every time.
    triggers: Arc<TriggerRegistry>,
    /// The call syntax this checkpoint speaks, for the chain's PREFILLED
    /// `<tool_call>`s. Asked of the dialect so the prefills cannot teach a
    /// shape the chat stencil would refuse — see [`render::render_list_call`].
    envelope: ToolCallEnvelope,
    /// Directory-frequency index over the whole workspace, built once. Shared
    /// by every worker: it is read-only after construction, and it has to span
    /// the *whole* corpus for a document frequency to mean anything.
    index: Arc<probe::idf::TermIndex>,
    /// Filler that pads each stuffed probe case out to its block boundary.
    ///
    /// The dialect's turn terminator rather than an arbitrary id: it is a token
    /// the model has seen in exactly this position ten thousand times, so a run
    /// of them is the most inert tail available, and it lands in the assistant
    /// half where nothing scores it. Same choice the tool calibration makes.
    pad_token: u32,
}

impl IngestPlan {
    fn new(
        engine: &Mutex<ConversationEngine>,
        proj_builder: &projection::Builder,
        config: &SequenceConfig,
        layer_name: &str,
        group_name: &str,
        index: Arc<probe::idf::TermIndex>,
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
        let triggers = {
            let en = engine.lock().unwrap();
            en.compile_think_steering()?
                .map(|ts| ts.registry_for(&TriggerRegistry::new(), ThinkMode::Off))
                // No single `<think>`/`</think>` token in this vocabulary, so
                // there is no block to steer and nothing to bind the trigger to;
                // `compile_think_steering` has already warned. The sampler-side
                // `apply_think_mode` still runs.
                .unwrap_or_else(|| Arc::new(TriggerRegistry::new()))
        };
        let pad_token = {
            let e = engine.lock().unwrap();
            e.tokenizer()
                .encode(config.dialect.assistant_end, false)
                .ok()
                .and_then(|enc| enc.get_ids().last().copied())
                .unwrap_or(0)
        };
        engine.lock().unwrap().mark_layer_append_only(layer);
        let envelope = ToolCallEnvelope::for_dialect(&config.dialect);
        Ok(Self {
            layer,
            group,
            proj_builder: proj_builder.clone(),
            system_prompt: layer_system_prompt(proj_builder, layer_name, config),
            // Generation: thinking suppressed — its budget is for questions.
            // Answering: thinking KEPT, because the `<think>` block is exactly
            // the reasoning-shaped signature a mid-decode scan matches against.
            probe_prompt: probe_generation_prompt(proj_builder, config),
            answer_prompt: branch_system_prompt(
                proj_builder,
                config,
                probe_pass::ANSWER_BRANCH,
                false,
            ),
            config: utility_config(config.clone()),
            triggers,
            envelope,
            index,
            pad_token,
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
    // Both keys: a directory's summary conversation carries `DIR_KEY`, its probe
    // conversation `PROBE_DIR_KEY`. Sweeping only the first would leave a deleted
    // folder's probes live and voting in every scan, with no summary behind them
    // to retrieve.
    for key in [DIR_KEY, PROBE_DIR_KEY] {
        for (tl, dir) in e.conversations_with_metadata_key(key) {
            if !present.contains(dir.as_str()) {
                if let Err(err) = e.tombstone_timeline(tl) {
                    tracing::warn!(
                        target: "zend::repo_scan",
                        dir = %dir, key,
                        "tombstone of removed directory's conversation failed: {err:#}",
                    );
                }
            }
        }
    }
}

/// Retire every crashed-partial `repo_map` conversation, up front.
///
/// A partial carries [`DIR_KEY`] but no [`HASH_KEY`] — the directory tag is
/// written at conversation creation and the content hash only after the unit's
/// ingest succeeds, so the pair says "this attempt started and never finished".
///
/// **Why eagerly, when [`process_one_dir`] already retires one.** That retirement
/// is per-directory and lazy: it runs only when *that* directory comes back
/// through the pool on a resume-cache miss. Across ordinary runs the two are
/// equivalent, because a partial has no hash, so it always misses the cache and
/// always gets re-ingested. They stop being equivalent the moment the pass does
/// not run — `--skip-layer repo_map`, an aborted pass, a failure cap — and then
/// the debris simply stays live.
///
/// Live is the problem. A partial is a half-built chain: a request and a
/// `file_list` round-trip with the summary decode missing, and because it was
/// never coupled (the ingest writes its `TurnCoupling`s only after the whole
/// chain returns) nothing marks its halves as belonging together either. It
/// still carries `WideQSig`s, so it still competes in the provenance gather —
/// and a file listing is about the most promiscuous thing that can: dozens of
/// paths, matching almost any probe. Measured on one substrate after a hard kill
/// mid-ingest: 35 such chains, and the dialogue's projection selecting their
/// scaffolding turns — a `<tool_response>` listing paired with a `file_read`
/// call, from two unrelated folders, with the summary nowhere and the last call
/// answered by nothing.
///
/// Called ONCE per boot from the session's ingest pre-loop, for every layer the
/// operator did not `--disable-layer` — so a `--skip-layer repo_map` boot, which
/// runs no pass at all, still leaves with its debris retired. That is why the
/// call does not live in [`ingest_repo_map`]: a sweep that only runs when the
/// pass runs cannot clean up the one case that produces debris and then declines
/// to re-ingest it.
///
/// Never called from [`refresh_repo_map`]. A refresh can overlap a pool that is
/// mid-flight, and an in-flight unit is indistinguishable from a crashed one by
/// metadata alone — both carry `dir` with no hash — so sweeping there would
/// tombstone the conversation a worker is still building. The pre-loop runs
/// before any pool starts, so nothing is in flight.
pub(crate) fn retire_crashed_partials(engine: &Mutex<ConversationEngine>) {
    let e = engine.lock().unwrap();
    let mut retired = 0usize;
    for (tl, dir) in e.conversations_with_metadata_key(DIR_KEY) {
        if e.conversation_metadata(tl)
            .is_some_and(|m| ingest_committed(&m))
        {
            continue;
        }
        match e.tombstone_timeline(tl) {
            Ok(()) => retired += 1,
            Err(err) => tracing::warn!(
                target: "zend::repo_scan",
                dir = %dir,
                "tombstone of crashed-partial conversation failed: {err:#}",
            ),
        }
    }
    if retired > 0 {
        tracing::info!(
            target: "zend::repo_scan",
            retired,
            "retired crashed-partial repo_map conversations (no content hash) \
             so their half-built chains leave the provenance gather",
        );
    }
}

/// Whether a `repo_map` conversation's ingest ever committed.
///
/// The two keys are a completion protocol, not two independent tags: [`DIR_KEY`]
/// is written at conversation creation and [`HASH_KEY`] only after the unit's
/// ingest succeeds, so their combination is the only durable record of whether
/// an attempt finished. `dir` alone therefore means "started, never committed" —
/// a crashed partial — and that is what both the eager sweep
/// ([`retire_crashed_partials`]) and the per-directory deferred tombstone in
/// [`process_one_dir`] key on.
///
/// Named and separate so the rule is stated once and testable without an
/// engine; the sweep that applies it needs a loaded model and so is covered by
/// the engine-backed suite rather than here.
fn ingest_committed(meta: &BTreeMap<String, String>) -> bool {
    meta.contains_key(HASH_KEY)
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
    // Every directory's admitted + held-out probes, collected for the holdout
    // file and the pass summary. Bounded by the workspace's directory count, so
    // a few hundred small structs.
    let probes: Mutex<Vec<probe::ProbeSet>> = Mutex::new(Vec::new());
    let probe_stats: Mutex<probe_pass::ProbeStats> = Mutex::new(Default::default());

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
        // The fallback is the BLIND width, taken when neither the memory report
        // nor the governor can say anything — which the call site above notes is
        // normal at scan start. It must NOT be `REPO_MAP_PARALLELISM`: that
        // constant is a measured ceiling for a 72 GB card, and defaulting to it
        // would open 96 conversations on any card with zero memory input.
        // Over-subscription there is not a slowdown — the wave's transient tier
        // fails and the ingest aborts, losing files (see the table on
        // `MAX_SCOPE_LINES`). Start conservative and let the runtime gate widen.
        // **The pool starts WIDE and is paced by the queue, not sized by the
        // card.** A worker blocked in `reserve_scan_slot` holds no conversation
        // and no K/V — it is a parked thread — so the thread count costs
        // essentially nothing and only decides how quickly the queue can be
        // refilled once it drains. Sizing it from residency is what produced
        // `n_workers=1` against a healthy expert cache, and a queue that never
        // held more than one candidate across 226 waves.
        //
        // Bounded by the smaller of the measured ceiling and the directories
        // there actually are: spawning more threads than units is pure waste.
        let n_workers = REPO_MAP_PARALLELISM.min(units.len().max(1));
        tracing::info!(
            target: "zend::repo_scan",
            n_workers,
            "repo map pool started wide; paced only by the engine's residency",
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
                        process_one_dir(
                            engine,
                            plan,
                            &ctx,
                            &units[idx],
                            &failures,
                            &probes,
                            &probe_stats,
                            workspace,
                        )
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

    // The holdout file is written OUTSIDE the substrate, deliberately. These
    // queries exist to be scored against a corpus that has never seen them, so
    // persisting them anywhere the scan can reach would destroy the only
    // measurement that says whether any of this worked.
    let sets = probes.into_inner().unwrap_or_else(|e| e.into_inner());
    let stats = probe_stats.into_inner().unwrap_or_else(|e| e.into_inner());
    // A final rewrite. Each directory already wrote the file as it finished (see
    // `process_one_dir`), so this only closes the last one out — but it is the
    // write that runs even when the pass admitted nothing at all, which is the
    // difference between an empty holdout file and no file for the harness to
    // open.
    if let Err(e) = probe_pass::write_holdout(workspace, &sets) {
        tracing::warn!(
            target: "zend::repo_scan::probe",
            "probe holdout file not written — the retrieval harness has nothing to score: {e:#}",
        );
    }
    tracing::info!(
        target: "zend::repo_scan::probe",
        dirs_generated = stats.dirs_generated,
        dirs_complete = stats.dirs_complete,
        probes_ingested = stats.probes_ingested,
        held_out = stats.held_out,
        dirs_without_seeds = stats.dirs_without_seeds,
        by_register = ?stats.by_register,
        rejections = ?stats.rejections,
        "probe layer complete",
    );

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
#[allow(clippy::too_many_arguments)]
fn process_one_dir(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    ctx: &ToolContext,
    unit: &DirUnit,
    failures: &Failures,
    probes: &Mutex<Vec<probe::ProbeSet>>,
    stats: &Mutex<probe_pass::ProbeStats>,
    holdout_root: &Path,
) -> anyhow::Result<()> {
    // Render BEFORE minting anything: the tool responses come from actually
    // running the tools, so a directory the tools can't read is caught here and
    // costs no conversation. Prefilling an error body would be worse than
    // skipping — it teaches the model a tool interaction that failed.
    let (prefilled, decode_user) = render::render_chain(ctx, unit, &plan.envelope);
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
        // The prior generation's PROBE conversation is superseded too. It carries
        // its own key, so the `DIR_KEY` scan below cannot see it — and left alone
        // it would survive this re-ingest and vote in every scan alongside the
        // replacement probes, permanently doubling the folder's weight with
        // questions written against content that has since changed.
        superseded.extend(e.find_conversations_by_metadata(PROBE_DIR_KEY, &unit.dir));
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
        let mut sink = SequenceTurnSink::new(&mut conv, Arc::clone(&plan.triggers));
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

    // ── Probe layer ──────────────────────────────────────────────────────────
    // Generate this folder's questions, admit them, and decode each as a turn
    // on THIS conversation.
    //
    // Runs BEFORE the hash tag below, which is what commits the generation. A
    // crash here therefore re-ingests the whole directory next run — summary and
    // probes together — rather than leaving it marked complete with a partial
    // probe set that the resume cache would skip forever.
    let probe_set = run_probe_pass(engine, plan, &mut conv, unit, holdout_root);
    if let Some((set, had_seeds)) = probe_set {
        stats.lock().unwrap().merge(&set, had_seeds);
        // Rewritten after EVERY directory, not once when the pool drains.
        //
        // The held-out queries exist only in memory until this lands, and they
        // are produced only for directories processed in *this* pass — a
        // directory that resume-cache hits contributes nothing. So a pool that
        // is interrupted, aborted, or simply still running at the end of a
        // session would leave no scoreable corpus at all, and re-running would
        // not recover it: the directories would hit the cache and be skipped.
        //
        // The file is a few hundred KB and the pool completes a directory every
        // few minutes, so rewriting it whole is cheaper than the bookkeeping an
        // append would need to stay deterministic.
        let mut all = probes.lock().unwrap();
        all.push(set);
        if let Err(e) = probe_pass::write_holdout(holdout_root, &all) {
            tracing::warn!(
                target: "zend::repo_scan::probe",
                "probe holdout write failed: {e:#}",
            );
        }
    }

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

/// Metadata key naming the directory a PROBE conversation belongs to.
///
/// Separate from [`DIR_KEY`] on purpose — see the tagging site in
/// [`run_probe_pass`]. Both are swept by [`reconcile_deleted`] so a removed
/// directory takes its probes with it.
const PROBE_DIR_KEY: &str = "probe_dir";

/// Create the conversation a folder's probes are answered on: framed to answer,
/// seeded with the folder's summary, and never cold-persisted before its own
/// eviction.
fn new_probe_conversation(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    unit: &DirUnit,
    summary: &str,
    chunk: &[probe::Probe],
) -> Option<candle_conversation::Sequence> {
    if chunk.is_empty() {
        return None;
    }
    let mut conv = {
        let e = engine.lock().unwrap();
        match e.new_conversation_with_projection(
            &plan.answer_prompt,
            plan.proj_builder.clone(),
            plan.layer,
            plan.group,
            plan.config.clone(),
        ) {
            Ok(c) => {
                e.set_timeline_summarize(c.timeline_id(), false);
                // Labelled with the BARE directory, exactly as its summary
                // conversation is. The label is what a projection tile carries,
                // and the retrieval harness matches tiles against the directory
                // a probe was written about — so a decorated label ("… probes")
                // makes every trial read as a miss and reports a formatting
                // artifact as 0% retrieval. The `kind` metadata is what
                // distinguishes the two conversations; the label is not.
                if let Err(err) = e.set_conversation_label(c.timeline_id(), &unit.dir) {
                    tracing::warn!(target: "zend::repo_scan::probe", "probe label set failed: {err:#}");
                }
                c
            }
            Err(err) => {
                tracing::warn!(
                    target: "zend::repo_scan::probe",
                    dir = %unit.dir,
                    "probe conversation create failed: {err}",
                );
                return None;
            }
        }
    };
    if let Err(err) = probe_pass::seed_context(&mut conv, unit, summary) {
        tracing::warn!(
            target: "zend::repo_scan::probe",
            dir = %unit.dir,
            "probe context seed failed; probes will answer unGROUNDED: {err:#}",
        );
    }
    Some(conv)
}

/// Generate, admit and ingest one directory's probes.
///
/// Returns the admitted set and whether the directory had any distinctive term
/// to seed with, or `None` when generation failed outright.
///
/// **Never fatal.** A directory whose probes fail keeps its summary and its
/// place in the map; the probe layer is an index over that summary, and a
/// missing index entry costs retrieval quality for one folder where a propagated
/// error would cost the folder itself. Every failure path here logs and returns.
fn run_probe_pass(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    conv: &mut candle_conversation::Sequence,
    unit: &DirUnit,
    root: &Path,
) -> Option<(probe::ProbeSet, bool)> {
    let seeds = plan
        .index
        .distinctive(&unit.dir, probe::render::seed_count());
    let had_seeds = !seeds.is_empty();
    let summary = probe_pass::last_summary(conv);

    // The folder's own `.substrate.yaml` comes first. A complete file means the
    // questions are already written — by a previous run or, better, by a person —
    // and the model is not asked for any. That is the whole point of the file:
    // the expensive artifact survives `--wipe-substrate`, and a fresh substrate
    // costs a prefill instead of a generation.
    let mut meta = metadata::load(root, unit)
        .unwrap_or_else(|| metadata::FolderMetadata::skeleton(unit, Vec::new()));
    if meta.is_complete() {
        let set = probe_pass::admit_authored(unit, &plan.index, &meta);
        tracing::debug!(
            target: "zend::repo_scan::probe",
            dir = %unit.dir,
            authored = meta.question_count(),
            admitted = set.probes.len(),
            "probes taken from folder metadata; no generation needed",
        );
        let ingested = ingest_probe_chunks(engine, plan, unit, &summary, &set);
        tracing::debug!(
            target: "zend::repo_scan::probe",
            dir = %unit.dir, ingested, "authored probes ingested",
        );
        return Some((set, had_seeds));
    }

    // The generation conversation is a throwaway: it is created only to hold
    // the evidence block and decode a list of questions, and its answer must
    // never reach the layer (see `probe_pass`). Marked transient so its K/V is
    // never cold-persisted, and tombstoned below on every path.
    let mut gen_conv = {
        let e = engine.lock().unwrap();
        let created = e.new_conversation_with_projection(
            &plan.probe_prompt,
            plan.proj_builder.clone(),
            plan.layer,
            plan.group,
            plan.config.clone(),
        );
        match created {
            Ok(c) => {
                e.conversation().mark_timeline_transient(c.timeline_id());
                e.set_timeline_summarize(c.timeline_id(), false);
                c
            }
            Err(err) => {
                tracing::warn!(
                    target: "zend::repo_scan::probe",
                    dir = %unit.dir,
                    "probe generation conversation create failed: {err}",
                );
                return None;
            }
        }
    };

    let candidates = probe_pass::generate(&mut gen_conv, unit, &summary, &seeds);
    let gen_timeline = gen_conv.timeline_id();
    drop(gen_conv);
    {
        let e = engine.lock().unwrap();
        if let Err(err) = e.tombstone_timeline(gen_timeline) {
            tracing::warn!(
                target: "zend::repo_scan::probe",
                dir = %unit.dir,
                "tombstone of probe generation conversation failed — its question list \
                 is now live in the layer: {err:#}",
            );
        }
    }

    let candidates = match candidates {
        Ok(c) => c,
        Err(err) => {
            tracing::warn!(
                target: "zend::repo_scan::probe",
                dir = %unit.dir,
                "probe generation failed; folder keeps its summary and no probes: {err:#}",
            );
            return None;
        }
    };

    let set = probe_pass::admit(unit, &plan.index, &candidates);

    // Write what was generated back to `.substrate.yaml` BEFORE ingesting it, so
    // the cost is banked even if the ingest below is interrupted. Held-out
    // queries are written too: they are the same quality of question and a
    // person reading the file should see the folder's whole harvest, not the
    // three-quarters that happened to fit.
    meta.folder.path = unit.dir.clone();
    meta.folder.content_hash = unit.content_hash.clone();
    meta.folder.summary = summary.trim().to_string();
    meta.folder.distinctive_terms = seeds.iter().map(|s| s.to_string()).collect();
    for register in probe::Register::ALL {
        let mut authored = meta.register(register);
        for probe in set
            .probes
            .iter()
            .chain(set.held_out.iter())
            .filter(|p| p.register == register)
        {
            if !authored.iter().any(|q| q == &probe.text) {
                authored.push(probe.text.clone());
            }
        }
        meta.set_register(register, authored);
    }
    if let Err(e) = metadata::save(root, unit, &meta) {
        tracing::warn!(
            target: "zend::repo_scan::metadata",
            dir = %unit.dir,
            "could not save generated questions — they will be regenerated: {e:#}",
        );
    }

    let ingested = ingest_probe_chunks(engine, plan, unit, &summary, &set);
    tracing::debug!(
        target: "zend::repo_scan::probe",
        dir = %unit.dir,
        seeds = seeds.len(),
        candidates = candidates.iter().map(|(_, q)| q.len()).sum::<usize>(),
        admitted = set.probes.len(),
        ingested,
        held_out = set.held_out.len(),
        rejected = set.rejected.len(),
        "probes generated",
    );
    Some((set, had_seeds))
}

/// Answer a directory's probes in bounded chunks, one conversation each.
///
/// The probes are answered on their OWN conversation, framed to answer rather
/// than to summarise — see [`probe_pass`]. Under the folder conversation's
/// summariser framing (`thinking_effort: off`) a probe would decode two terse
/// sentences and no `<think>` block, and the reasoning-shaped signatures a
/// mid-decode scan matches against would not exist at all.
///
/// Chunked because one conversation for all 24 accumulates ~8,000 tokens of live
/// K/V per directory, which is what put the first full-workspace run into the
/// partition wall at `dirs_generated=0`. Each chunk is evicted as soon as it is
/// answered, so a directory's peak is its folder chain plus one chunk.
fn ingest_probe_chunks(
    engine: &Mutex<ConversationEngine>,
    plan: &IngestPlan,
    unit: &DirUnit,
    summary: &str,
    set: &probe::ProbeSet,
) -> usize {
    if candle_conversation::ingest_cancelled() {
        return 0;
    }
    let Some(mut probe_conv) = new_probe_conversation(engine, plan, unit, summary, &set.probes)
    else {
        return 0;
    };
    let ingested = probe_pass::ingest(&mut probe_conv, &set.probes, &unit.dir, plan.pad_token);
    let timeline = probe_conv.timeline_id();
    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), "repo_map_probe".to_string());
    // A key of its OWN, deliberately not `DIR_KEY`. `dir_state_from_substrate`
    // joins DIR_KEY against HASH_KEY to rebuild the resume record, so a second
    // conversation carrying DIR_KEY for the same directory would double every
    // unit in that state — `equivalent_to`'s length check would then never
    // match a fresh walk and the whole workspace would re-ingest on every
    // filesystem event.
    tags.insert(PROBE_DIR_KEY.to_string(), unit.dir.clone());
    if let Err(e) = probe_conv.set_metadata_many(&tags) {
        tracing::warn!(
            target: "zend::repo_scan::probe",
            dir = %unit.dir,
            "probe conversation tagging failed: {e:#}",
        );
    }
    drop(probe_conv);
    engine.lock().unwrap().evict_ingest_timeline(timeline);
    ingested
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

/// The system prompt for a conversation framed on an arbitrary branch.
///
/// The same schema walk as [`layer_system_prompt`], resolved against `branch`.
/// The probe layer needs two of these and neither is the summariser's: one that
/// writes questions ([`probe_pass::GENERATE_BRANCH`]) and one that answers them
/// with a `<think>` block ([`probe_pass::ANSWER_BRANCH`]).
///
/// `suppress_thinking` prepends `/no_think` to the assembled body, which is how
/// the model's soft switch is actually driven (`models::builder`). It is not
/// reachable through the section tree: the schema's `no_think` toggle carries a
/// *dialect* marker rather than prompt content, and a conversation created with
/// an explicit prompt string under `disable_reprojection` never re-projects, so
/// selecting it resolves cleanly and contributes nothing at all.
fn branch_system_prompt(
    builder: &projection::Builder,
    config: &SequenceConfig,
    branch: &[(&str, &str)],
    suppress_thinking: bool,
) -> String {
    config
        .dialect
        .format_system_prompt(&branch_prompt_body(builder, branch, suppress_thinking))
}

/// The system prompt for the probe-GENERATION conversation: one option's text
/// and nothing else.
///
/// Deliberately **not** the schema walk every other ingest prompt uses. That
/// walk emits every fixed section in the shared system prompt — the coding
/// assistant's framing, the grounding rules, the history stance — and the
/// question-writer persona is one paragraph inside thousands of tokens of it.
/// The result was a generator that reasoned at length about what was being asked
/// of it and never got to the questions: measured blocks of 5,342, 15,102 and
/// 15,140 characters, one of them concluding the task was to "generate 50 short
/// paragraphs".
///
/// The persona text still comes from the schema, so the prompt stays a config
/// item rather than a string literal in Rust. What is dropped is only the
/// scaffolding that belongs to a *dialogue* — which this conversation is not.
fn probe_generation_prompt(builder: &projection::Builder, config: &SequenceConfig) -> String {
    use projection::SystemPromptItem;
    let persona = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            SystemPromptItem::SectionTree(tree) => tree
                .nodes
                .iter()
                .find(|n| n.name == "persona")?
                .options
                .iter()
                .find(|o| o.id == "question_writer")
                .map(|o| o.content.clone()),
            _ => None,
        })
        .unwrap_or_default();
    config
        .dialect
        .format_system_prompt(&format!("/no_think\n{}", persona.trim()))
}

/// The unwrapped body [`branch_system_prompt`] frames — split out so the
/// `/no_think` decision can be asserted without constructing a whole
/// [`SequenceConfig`].
fn branch_prompt_body(
    builder: &projection::Builder,
    branch: &[(&str, &str)],
    suppress_thinking: bool,
) -> String {
    let body = prompt_body_for(builder, branch);
    if suppress_thinking {
        format!("/no_think\n{body}")
    } else {
        body
    }
}

/// The unwrapped body of [`layer_system_prompt`] — the schema walk, with no
/// dialect framing. Split out so the assembled prompt can be asserted directly.
fn ingest_prompt_body(builder: &projection::Builder) -> String {
    prompt_body_for(builder, SUMMARIZE_BRANCH)
}

/// [`ingest_prompt_body`] over an arbitrary branch.
fn prompt_body_for(builder: &projection::Builder, branch: &[(&str, &str)]) -> String {
    use projection::SystemPromptItem;
    let mut body = String::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => body.push_str(&s.content),
            SystemPromptItem::SectionTree(tree) => {
                let (selection, _) = branch_selection(tree, branch);
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

/// `tree`'s default selection with `branch` applied, plus the branch entries
/// this tree could not resolve.
///
/// Used by both ingest framings — [`SUMMARIZE_BRANCH`] for the folder summary
/// and [`probe_pass::GENERATE_BRANCH`] for the probe questions.
///
/// A node the tree does not declare at all is not a miss — a branch spans two
/// trees, so each sees only its own nodes. A node that IS declared but lacks the
/// named option is a miss: the schema and this ingest disagree about what the
/// option is called, and the prompt silently loses that framing.
fn branch_selection(
    tree: &projection::SectionTree,
    branch: &[(&str, &str)],
) -> (Vec<u8>, Vec<String>) {
    let mut selection = tree.default_selection.clone();
    let mut unresolved = Vec::new();
    for (node_id, option_id) in branch {
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
    validate_branch(builder, SUMMARIZE_BRANCH, "summarizer")?;
    // The probe-generation branch fails the same way and just as silently: with
    // the `question_writer` persona missing, generation falls back to whatever
    // the schema defaults to and returns a folder description where the parser
    // expects forty-eight questions. Every probe for every directory is then
    // lost behind one warning.
    validate_branch(builder, probe_pass::GENERATE_BRANCH, "probe-generation")?;
    // And the answering branch, whose silent failure is the subtlest of the
    // three: a missing `thinking_effort` option leaves the probes decoding
    // without a `<think>` block, so they still look like perfectly good turns
    // while carrying none of the reasoning-shaped signatures they exist for.
    validate_branch(builder, probe_pass::ANSWER_BRANCH, "probe-answering")
}

/// Fail the ingest if `branch` cannot be resolved against the schema.
fn validate_branch(
    builder: &projection::Builder,
    branch: &[(&str, &str)],
    what: &str,
) -> anyhow::Result<()> {
    let unresolved: Vec<String> = builder
        .schema()
        .system_prompt
        .section_trees()
        .flat_map(|t| branch_selection(t, branch).1)
        .collect();
    if unresolved.is_empty() {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "projection schema cannot supply the repo_map {what} framing: {}. The ingest \
         would decode the wrong thing entirely — check the workspace `projection.yaml` \
         is in step with the bundled one.",
        unresolved.join("; "),
    ))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::types::Language;

    /// The completion protocol the crashed-partial sweep rests on: the directory
    /// tag is written at creation, the content hash only on success, so `dir`
    /// without a hash is the signature of an attempt that never finished.
    ///
    /// Pinned because the sweep tombstones on it. Keying on the wrong half would
    /// retire every GOOD generation on the next pass — the folder summaries are
    /// exactly the conversations that DO carry a hash — and the blast radius is
    /// the whole layer, silently, one boot later.
    #[test]
    fn a_conversation_is_committed_only_once_it_carries_a_content_hash() {
        let mut meta = BTreeMap::new();
        assert!(
            !ingest_committed(&meta),
            "no metadata at all is not committed"
        );

        meta.insert(DIR_KEY.to_string(), "candle-nn/src/".to_string());
        assert!(
            !ingest_committed(&meta),
            "the directory tag alone is a crashed partial — it is written at \
             conversation creation, before any work"
        );

        meta.insert(HASH_KEY.to_string(), "abc123".to_string());
        assert!(
            ingest_committed(&meta),
            "the content hash is the commit, and is written only on success"
        );

        // The hash is what counts, not the tag: a generation that somehow lost
        // its directory tag has still committed its content and must not be
        // swept. (`retire_crashed_partials` only visits `DIR_KEY` holders, so
        // this is belt-and-braces on the predicate itself.)
        let mut hash_only = BTreeMap::new();
        hash_only.insert(HASH_KEY.to_string(), "abc123".to_string());
        assert!(ingest_committed(&hash_only));
    }

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

    /// `thinking_effort: off` is only a *sentence* in the prompt ("Answer
    /// directly, without deliberating first"), not a mechanism. On the
    /// probe-generation turn the model overrode it routinely and spent the whole
    /// budget reasoning — 6,363 and 6,735 characters of it, not one question —
    /// so both directories lost their entire probe set.
    ///
    /// The mechanism is `/no_think` on the system prompt body. Selecting the
    /// schema's `no_think` toggle does NOT achieve it: that node carries a
    /// dialect marker rather than content, so it resolves cleanly and emits
    /// nothing into a statically-assembled prompt.
    #[test]
    fn the_generation_prompt_suppresses_thinking_and_the_answer_prompt_keeps_it() {
        let builder = bundled_builder();
        let answer = branch_prompt_body(&builder, probe_pass::ANSWER_BRANCH, false);
        assert!(
            !answer.contains("/no_think"),
            "the probe ANSWER must think — the block is the signature a \
             mid-decode scan matches against",
        );
    }

    /// The generation prompt is the persona ALONE. Carrying the whole shared
    /// system prompt buried the instruction in a coding assistant's framing and
    /// the generator spent its budget reasoning about the request instead of
    /// answering it.
    #[test]
    fn the_generation_prompt_is_the_persona_alone() {
        let builder = bundled_builder();
        let dialect = candle_conversation::models::Dialect::chat_ml();
        let full = dialect.format_system_prompt(&ingest_prompt_body(&builder));
        let persona = builder
            .schema()
            .system_prompt
            .items
            .iter()
            .find_map(|item| match item {
                projection::SystemPromptItem::SectionTree(tree) => tree
                    .nodes
                    .iter()
                    .find(|n| n.name == "persona")?
                    .options
                    .iter()
                    .find(|o| o.id == "question_writer")
                    .map(|o| o.content.clone()),
                _ => None,
            })
            .expect("the schema must declare the question_writer persona");

        assert!(
            persona.contains("You write the questions"),
            "{persona:.120}",
        );
        // Far shorter than the full walk, and that difference is the fix.
        assert!(
            persona.len() * 4 < full.len(),
            "persona {} vs full walk {}",
            persona.len(),
            full.len(),
        );
    }

    /// Why selecting the schema's `no_think` toggle did not work, pinned so the
    /// shortcut is not retried.
    ///
    /// It is not a no-op — it really does emit the marker — but a
    /// statically-assembled prompt walks the schema in declaration order, and the
    /// toggle sits inside the section tree, *after* the fixed sections above it.
    /// The soft switch is only read at the very start of the prompt, so selecting
    /// it buries the marker mid-body where the model ignores it: measured, the
    /// generation still produced 6,700 characters of reasoning and no questions.
    /// Prepending to the assembled body is what actually puts it at position 0.
    #[test]
    fn the_schema_no_think_toggle_lands_mid_prompt_not_at_the_start() {
        let builder = bundled_builder();
        let selected = prompt_body_for(&builder, &[("no_think", "present")]);
        let absent = prompt_body_for(&builder, &[("no_think", "absent")]);
        assert_ne!(selected, absent, "the toggle does emit its marker");
        assert!(
            !selected.starts_with("/no_think"),
            "…but not where the switch is read — which is the whole defect",
        );
        assert!(
            branch_prompt_body(&builder, &[], true).starts_with("/no_think\n"),
            "prepending to the body is what puts it at position 0",
        );
    }

    /// The `repo_map` group's candidates are whole folders, so its scoring has
    /// to normalize the competition BETWEEN them. Without this the flag parses,
    /// the daemon starts, retrieval runs — and every folder is quietly scaled
    /// against its own denominator, so the one carrying the most turns wins.
    /// Nothing about that failure is visible except a bad hit rate.
    #[test]
    fn the_repo_map_group_normalizes_between_folders() {
        let builder = bundled_builder();
        let layer = builder
            .schema()
            .layers
            .iter()
            .find(|l| l.name == "repo_map")
            .expect("repo_map layer");
        let group = layer.groups.first().expect("structure group");
        assert!(
            group.policy.scan.member_normalization,
            "repo_map/{} must normalize across folders — see the tools collection, \
             which is the shape this mirrors",
            group.name,
        );
    }

    /// …and the dialogue layer must NOT, because its candidates are moments in
    /// one thread rather than competing conversations.
    #[test]
    fn the_dialogue_group_keeps_per_timeline_scoping() {
        let builder = bundled_builder();
        for layer in &builder.schema().layers {
            if layer.name == "repo_map" {
                continue;
            }
            for group in &layer.groups {
                assert!(
                    !group.policy.scan.member_normalization,
                    "{}/{} should scope per timeline",
                    layer.name, group.name,
                );
            }
        }
    }

    /// An admission section as the engine would publish it, with `open` slots
    /// showing above the pass's base and the weight zone comfortably clear of
    /// its hold unless a test says otherwise.
    fn admission(
        carried: usize,
        starved: usize,
        queued_tokens: u64,
        open: usize,
    ) -> AdmissionSection {
        AdmissionSection {
            ceiling_bytes: 0,
            prefill_width: 0,
            section_width: 0,
            decode_width: carried,
            queued_prefills: 0,
            queued_prefill_tokens: queued_tokens,
            completed_tokens: 0,
            decode_carried: carried,
            decode_starved: starved,
            open_slots: open,
            wave_width: 0,
            publish_interval_ms: 0,
            weight_zone_bytes: 7 << 30,
            weight_hold_bytes: 5 << 30,
        }
    }

    /// One conversation must always be able to proceed, whatever the engine
    /// says or fails to say — a pool whose only route to freeing the device is
    /// finishing work it may not start would deadlock.
    #[test]
    fn the_first_conversation_always_opens() {
        assert_eq!(gate_decision(0, None), Ok(()));
        let mut a = admission(4, 30, u64::MAX, 0);
        a.weight_zone_bytes = 0;
        assert_eq!(gate_decision(0, Some(&a)), Ok(()));
    }

    /// **No report is a cold start, not a hold.** With nothing reflected the
    /// ramp bound applies from zero, so the pool opens its first slack's worth
    /// and then waits for the engine to report them — which is the same rule
    /// that governs every later burst, rather than a special case.
    #[test]
    fn no_report_ramps_from_zero_rather_than_holding() {
        assert_eq!(gate_decision(1, None), Ok(()));
        assert_eq!(gate_decision(SCAN_OPEN_SLACK, None), Ok(()));
        assert_eq!(
            gate_decision(SCAN_OPEN_SLACK + 1, None),
            Err(Hold::Unreflected {
                live: SCAN_OPEN_SLACK + 1,
                reflected: 0,
            }),
        );
    }

    /// **The burst is bounded by what the engine has seen.** A report a wave
    /// behind reads "empty" for every worker in the burst; the pool may run at
    /// most the slack ahead of the slots the engine reports. Measured, 96 opened
    /// in one second without this.
    #[test]
    fn openings_the_engine_has_not_reflected_hold_beyond_the_slack() {
        // Four open here, the report shows none of them yet: within the slack.
        assert_eq!(
            gate_decision(SCAN_OPEN_SLACK, Some(&admission(0, 0, 0, 0))),
            Ok(()),
        );
        // A fifth would be one past it.
        assert_eq!(
            gate_decision(SCAN_OPEN_SLACK + 1, Some(&admission(0, 0, 0, 0))),
            Err(Hold::Unreflected {
                live: SCAN_OPEN_SLACK + 1,
                reflected: 0,
            }),
        );
        // Once the report shows them, the same worker opens.
        assert_eq!(
            gate_decision(SCAN_OPEN_SLACK + 1, Some(&admission(0, 0, 0, 2))),
            Ok(()),
        );
        // Slots this pool did not open only loosen the bound — never a base to
        // subtract, which went stale and held a pass to five conversations.
        assert_eq!(
            gate_decision(SCAN_OPEN_SLACK + 1, Some(&admission(0, 0, 0, 30))),
            Ok(()),
        );
    }

    /// **The one thing worth holding for: the resident experts.**
    ///
    /// K/V and the expert weights share one elastic span, so once the zone is
    /// down to its hold, another conversation's K/V comes out of the experts —
    /// and an engine that streams its experts is slower at everything,
    /// including finishing the conversations that would give the ground back.
    #[test]
    fn a_zone_down_to_its_hold_holds_the_producer() {
        let mut a = admission(8, 0, 0, 8);
        a.weight_zone_bytes = 5 << 30;
        a.weight_hold_bytes = 5 << 30;
        assert_eq!(
            gate_decision(8, Some(&a)),
            Err(Hold::Residency {
                zone_mib: 5 << 10,
                hold_mib: 5 << 10,
            }),
        );
        // A megabyte of headroom is headroom: the gate opens and the wave's own
        // claim-and-refuse decides what actually fits.
        a.weight_zone_bytes = (5 << 30) + (1 << 20);
        assert_eq!(gate_decision(8, Some(&a)), Ok(()));
    }

    /// A model with nothing to defend — no reservation, so no hold — is never
    /// held by this. The generality case: a dense checkpoint sitting resident
    /// with room over publishes a zero hold and the gate opens every time.
    #[test]
    fn no_hold_to_defend_never_holds() {
        let mut a = admission(8, 0, 0, 8);
        a.weight_zone_bytes = 0;
        a.weight_hold_bytes = 0;
        assert_eq!(gate_decision(8, Some(&a)), Ok(()));
    }

    /// **Runaway protection, and nothing finer.** Far above any healthy state,
    /// so it never binds in normal operation — a normal pass sits near twenty
    /// and the worst measured pathology reached eighty-three.
    #[test]
    fn the_runaway_ceiling_is_the_last_resort_only() {
        assert_eq!(
            gate_decision(SCAN_RUNAWAY_CEILING, Some(&admission(60, 0, 0, 300))),
            Err(Hold::Runaway {
                live: SCAN_RUNAWAY_CEILING,
                ceiling: SCAN_RUNAWAY_CEILING,
            }),
        );
        assert_eq!(
            gate_decision(SCAN_RUNAWAY_CEILING - 1, Some(&admission(60, 0, 0, 300))),
            Ok(()),
        );
        const _: () = assert!(SCAN_RUNAWAY_CEILING > 83);
    }

    /// **The producer does not pace the wave.** A starved engine, a deep queue
    /// and a long backlog are all the wave's business — it claims through the
    /// real allocators and refuses what will not fit. None of them holds the
    /// producer, because every proxy for them that has been tried here
    /// throttled throughput without protecting anything.
    #[test]
    fn engine_load_alone_never_holds_the_producer() {
        let mut a = admission(5, 3, u64::MAX, 5);
        a.queued_prefills = 500;
        assert_eq!(
            gate_decision(5, Some(&a)),
            Ok(()),
            "starved decodes, a full backlog and a 500-deep queue are not this gate's business",
        );
    }

    /// The generality case: an engine carrying everything it holds, with room
    /// above its hold, opens the gate every time, on any card.
    #[test]
    fn an_engine_with_residency_to_spare_always_opens() {
        assert_eq!(gate_decision(60, Some(&admission(60, 0, 100, 60))), Ok(()));
        assert_eq!(gate_decision(60, Some(&admission(60, 0, 0, 60))), Ok(()));
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
        // `include_str!` hands back the checkout's own line endings, while Rust
        // normalizes the CRLF in THIS file's string literals to LF. On a CRLF
        // checkout (`core.autocrlf=true`, the Windows default) an exact-line
        // pattern therefore matches nothing, and `replace` reports success
        // having changed nothing — leaving the schema intact and this test
        // asserting that a VALID schema is rejected. Normalize first, then prove
        // the edit actually landed.
        let yaml = include_str!("../prompts/projection.yaml").replace("\r\n", "\n");
        let mutated = yaml.replace(
            "            - id: folder\n",
            "            - id: renamed_away\n",
        );
        assert_ne!(
            mutated, yaml,
            "the `- id: folder` option moved or was re-indented — this test's \
             pattern is stale and was silently mutating nothing"
        );
        let yaml = mutated;
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
