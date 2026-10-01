//! The run itself: drive the real engine into KV fragmentation, then measure what it
//! costs.
//!
//! # Why this is not a row of the forward gate
//!
//! `test_parallel_batched_forwarding` measures each configuration from a **clean
//! slate**: every sequence is freed, `release_empty_arenas` runs, the session is
//! dropped and the device synchronised, and then a gate asserts nothing is live
//! before the next row starts ("the KV must be gone before the next config
//! starts", `batch_test/utils.rs`). So those rows cannot fragment — by
//! construction and by assertion — which is exactly what makes them good
//! regression and throughput baselines and useless for this question.
//!
//! It also cannot live in `candle-transformers` at all. Fragmentation is produced
//! by *admission* — which conversations are allowed to run concurrently, and
//! therefore which arenas are live at once — and the admission model
//! (`scheduler::admit::rate`) is in this crate, one level **above** the gate's.
//! Driving `BatchedInferenceSession` directly would exercise a different admitter
//! and measure a different machine.
//!
//! # What fragmentation actually is here
//!
//! KV lives in fixed 16 MiB regions, each handed to one arena, and **an arena
//! keeps its region for as long as it lives**. The wave transient tier must be
//! placed above every live arena, so the figure that matters is not how many
//! regions are live but how high the highest one sits: `live_watermark`. Free
//! regions *below* that frontier are ground the KV side owns and reports as
//! available, which the tier cannot use — so they push `weight_floor` right and
//! cost expert residency, which costs decode.
//!
//! `holes = live_watermark - live` is therefore the damage, and it is exact.
//!
//! # The shape of the run
//!
//! **Phase A — overlapping churn.** Conversations are started on a stagger and
//! retired at *different ages*, so that at every moment some are growing while
//! others are being freed. That is what strands a long-lived arena high in the
//! span while lower ones drain around it. Concurrency is raised until the pool
//! stops accepting more — the point of saturation — because a fragmentation
//! measurement taken with the pool half empty measures nothing.
//!
//! **Phase B — the comparison batch.** The same shape as one clean row of the
//! gate (concurrent prefills, then concurrent decodes) run against the *fragmented*
//! pool. The delta against the clean row is the cost, measured rather than
//! inferred.
//!
//! Run it through the `kv_fragmentation` example, or call [`run`] from a test — see
//! the module above.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle::Device;
use candle_nn::kv_cache::{region_stats, REGION_BYTES};
use candle_transformers::models::batch_test::fixtures;
use candle_transformers::models::batch_test::story_normalize::normalize_story;

use super::probe::{Probe, ProbeOutcome};
use super::profile::StoryGate;
use crate::guest::GuestRegistry;
use crate::memory_report;
use crate::scratch_substrate::ScratchSubstrate;
use crate::{
    ConversationEngine, OptionalState, SamplingConfig, SelectionState, SequenceConfig, TurnOptions,
};
use crate::{Sequence, NO_THINK_SELECTOR};

/// One sample of the pool's geometry.
#[derive(Clone, Copy, Debug, Default)]
struct Geometry {
    live: usize,
    watermark: usize,
    free: usize,
    weight_mib: usize,
    /// A tier was standing when this was read, so `free` is the ceiling's answer
    /// rather than the pool's — see [`sample`].
    mid_wave: bool,
}

impl Geometry {
    /// Free regions stranded below the arena frontier — the damage, in regions.
    fn holes(&self) -> usize {
        self.watermark.saturating_sub(self.live)
    }
}

/// The VRAM composition below the frontier, **as one publish of the engine's own
/// memory report sees it**.
///
/// Every field is in regions except [`Self::denied_mib`], and together they
/// decompose the frontier: `frontier = packed + span + sparsity + holes`, where
/// sparsity is `arenas - packed` and holes is `frontier - live`. Kept as a struct
/// rather than a tuple because the figures are only interpretable together — an
/// efficiency percentage on its own cannot say whether the loss is air inside the
/// arenas, which a pack removes, or free regions stranded below the frontier, which
/// the next claim takes by itself.
///
/// `span` is the term that is not a loss: whole regions a span tenant holds — a
/// sequence's recurrent state store, the provenance gallery. Without it the figure is
/// unreadable on any model that has one. Qwen3.8-Flash-Next, whose DeltaNet layers
/// keep their state there, read 4% efficiency with the KV pools packed to within two
/// arenas of perfect, because a few hundred regions of live recurrent state were being
/// charged to compaction as fragmentation.
///
/// All of it comes from the report and none from a second `region_stats` call, so the
/// numerator and the denominator are one moment. See [`Composition::frontier`].
#[derive(Clone, Copy, Debug, Default)]
struct Composition {
    /// What a perfect pack of the KV pools would return.
    denied_mib: usize,
    /// `(packed + span) / frontier`, as a percentage.
    eff: usize,
    /// Arenas the GPU KV pools hold.
    arenas: usize,
    /// Arenas those pools would need if packed.
    packed: usize,
    /// The arena frontier at the same instant as the rows above.
    frontier: usize,
    /// Regions held by any tenant at that instant.
    live: usize,
    /// Of `live`, the regions a span tenant holds. In use, and not packable.
    span: usize,
    /// Of `live`, the regions `KvHead` record arenas hold. In use, not packable, and in no
    /// size-class row — so absent from `packed` and `arenas` both.
    record: usize,
    /// Everything below the frontier that is holding nothing, in MiB — sparsity plus
    /// holes. The absolute form of [`Self::eff`], and the unit the weight side
    /// actually loses.
    loss_mib: usize,
    /// When the engine published this report, in milliseconds since the epoch.
    ///
    /// **The identity of the observation, not decoration.** The report is published on
    /// the scheduler's telemetry cadence and this harness samples faster than that, so
    /// consecutive samples routinely read the *same* publish. Anything asking whether a
    /// figure persisted has to compare distinct publishes, or it is comparing one
    /// observation with itself and finding, unsurprisingly, that it agrees.
    captured_ms: u64,
}

/// One raw reading, whatever the wave is doing.
/// Break the `span` column down by tenant, **for the sample that column came
/// from**.
///
/// **`span` on its own cannot be acted on.** It is every span tenant's regions as
/// one number, counted as legitimately in use — so a tenant standing on thirty
/// mostly-empty arenas reads exactly like one standing on thirty full ones, and a
/// compaction that reclaimed nothing reads exactly like one that reclaimed
/// everything. `use%` is what says whether a tenant has room to give back, and
/// `pools` is what says whether a packing walk could ever recover it: a tenant
/// spread over several strides pays at least one region per stride however small
/// its slots are, and no walk recovers that — only giving it fewer strides does.
///
/// **Taken from the report's own publish, not sampled here.** The first version
/// of this called `arena_census` at print time, minutes after the phase-B row it
/// was meant to explain, and the two disagreed by an order of magnitude — 18
/// tenant regions against a `span` of 219. That reads exactly like an uncounted
/// tenant holding 201 regions, and it is nothing but two moments compared as one.
/// The rows now arrive in `KvSection` beside `span_regions`, so the two are one
/// moment and the difference below is real rather than skew.
///
/// **A difference is not necessarily a defect.** `span_regions` counts every
/// `SpanRegion` handle, and the slot arenas are not its only holder: a guest
/// model's ground claims them too, and anything else that takes one in future
/// will. So the line reports the residual and names what it could be, rather than
/// asserting an identity that only holds while no guest is resident.
fn print_tenant_census(label: &str) {
    let Some((report, _)) = memory_report::latest() else {
        return;
    };
    let rows = &report.kv.span_tenants;
    let span = report.kv.span_regions;
    println!("\n=== Span tenants ({label}) ===\n");
    println!(
        "  {:<28} {:>7} {:>9} {:>6} {:>6}",
        "tenant", "regions", "held MiB", "use%", "pools"
    );
    for r in rows {
        let reserved = r.regions * REGION_BYTES;
        let use_pct = (r.held_bytes * 100).checked_div(reserved).unwrap_or(0);
        println!(
            "  {:<28} {:>7} {:>9} {:>6} {:>6}",
            r.tenant,
            r.regions,
            r.held_bytes >> 20,
            use_pct,
            r.pools,
        );
    }
    let total: usize = rows.iter().map(|r| r.regions).sum();
    // The reconciliation is the point of printing this at all: the residual is
    // ground held by a `SpanRegion` holder that is not a slot arena — a guest
    // model's ground today — and naming it is what keeps it from being read as an
    // uncounted tenant or as sampling skew.
    println!(
        "  {:<28} {:>7}   ({})",
        "total",
        total,
        match span.checked_sub(total) {
            Some(0) => format!("= span {span}"),
            Some(rest) => format!("span {span}; {rest} in non-arena holders (guest ground)"),
            // The census is taken under the pools' lock and `span_regions` under
            // the region pool's, so a release between them can leave the rows
            // ahead. Worth saying, not worth calling a fault.
            None => format!("span {span}; census ahead by {}", total - span),
        },
    );
}

fn sample_now(device: &Device) -> Option<Geometry> {
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        return None;
    };
    region_stats(gpu_id).map(|s| Geometry {
        live: s.live,
        watermark: s.live_watermark,
        free: s.free,
        weight_mib: s.weight_bytes >> 20,
        mid_wave: s.transient_bytes > 0,
    })
}

/// A reading taken **between forwards**, retrying briefly for one.
///
/// Sampling indiscriminately reports nonsense for `free`: while a forward is open
/// the transient tier is placed, and the region ceiling then puts every region
/// above the tier's base out of reach, so `free` reads 0 no matter how empty the
/// pool is. A saturation test fed those samples declares victory on the first
/// forward it happens to catch — which is exactly what the first version of this
/// harness did, reporting "saturated after 3.0s" against a pool with 1,700 free
/// regions.
///
/// `transient_bytes > 0` is the discriminator: non-zero means a tier stands, so
/// the reading is mid-wave. Falls back to the last reading if no quiet moment
/// appears, flagged so the caller can say so rather than quoting it as fact.
fn sample(device: &Device) -> Option<Geometry> {
    let mut last = None;
    for _ in 0..40 {
        let g = sample_now(device)?;
        if !g.mid_wave {
            return Some(g);
        }
        last = Some(g);
        std::thread::sleep(Duration::from_millis(25));
    }
    last
}

/// A prefix of the forward gate's **own** story fixture, `fraction`/8ths long.
///
/// The gate's fixture rather than a fixture of this example's, for two reasons.
/// Phase B is meant to be read against the gate's clean rows, and a comparison
/// between two different prompts is not a comparison — same text, same tokenizer,
/// same token count. And `story_prompt`'s own doc records why a second copy is
/// worse than it looks: the file is checked in with CRLF, so an independent
/// `include_str!` of it "would be a different prompt that merely looked identical
/// in the source".
///
/// Phase A varies the fraction so conversations differ in KV size, which is what
/// spreads occupancy across size classes instead of pouring it all into one.
/// The reply with a leading reasoning block removed.
///
/// A suppressed turn still carries the dialect's *framing* — Qwen3 opens a turn
/// with a pre-closed `<think> </think>` — and that is engine structure, not model
/// output: the gate drives the session directly and never sees it. Stripping it is
/// what makes the two comparable; leaving it in fails every session on a prefix the
/// model was never asked to produce.
fn strip_think(reply: &str) -> &str {
    let t = reply.trim_start();
    match t.find("</think>") {
        Some(end) if t.starts_with("<think>") => &t[end + "</think>".len()..],
        _ => reply,
    }
}

/// The conversation's own sampling, made deterministic.
///
/// Derived from the model's configured sampling rather than built from scratch, so
/// every other lever — the penalties, the banned tokens, the segment behaviour —
/// stays as the model ships it and only the randomness goes. `temperature = 0` is
/// argmax; `top_k = 1` and `top_p = 1` remove the two ways a nucleus could still
/// widen the choice.
fn greedy_sampling(config: SequenceConfig) -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        segment_temp_boost: 0.0,
        top_k: 1,
        top_p: 1.0,
        ..config.sampling
    }
}

/// The composer's thinking dial, off.
///
/// `Present` on [`NO_THINK_SELECTOR`] is what the turn assembler reads to emit the
/// dialect's suppression — `/no_think` as live glue for the families that carry the
/// switch in the user turn, a pre-closed block for the rest. Named here because
/// every turn this harness submits wants it and a missing one is not an error, just
/// a reply that spends its budget reasoning.
fn no_think() -> SelectionState {
    let mut s = SelectionState::default();
    s.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    s
}

fn story_slice(story: &str, fraction: usize) -> &str {
    let want = story.len() * fraction.clamp(1, 8) / 8;
    // Cut on a char boundary — the story is not ASCII.
    let mut end = want.min(story.len());
    while end > 0 && !story.is_char_boundary(end) {
        end -= 1;
    }
    &story[..end]
}

/// Run one probe, and return what it measured.
///
/// Loads the model, stands up an engine on a scratch substrate, runs the three phases,
/// prints the tables as it goes, and evaluates the gates. Every gate is evaluated and
/// reported before any of them fails the run — see [`ProbeOutcome`].
///
/// The caller owns the decision about tracing. Nothing here initialises a subscriber,
/// because a test process has usually already installed one and a second `try_init`
/// would silently do nothing; a driver that wants the compaction pass and the weight
/// side's reclaim visible installs INFO before calling.
pub fn run(probe: &Probe) -> anyhow::Result<ProbeOutcome> {
    let device = Device::new_cuda(probe.device)?;
    let builder = probe.builder();
    let (model_path, tokenizer_path) = builder.resolve_paths_pub()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    println!("Loading {model_path:?} …");
    let model = builder.load_model(&model_path, &device, None)?;
    run_on_model(probe, &device, tokenizer, model)
}

/// Run the probe against a model the **caller** loaded.
///
/// The seam exists so one process can measure both halves of the comparison table off a
/// single load: the harness rows want `&M` for `forward_wave`, these rows want the model
/// *moved into* a `ConversationEngine`, and a 17 GB checkpoint is not worth opening twice
/// — quite apart from the span reservation, which is process-global and does not expect a
/// second weight zone to be carved beside the first.
///
/// The caller passes the model already batched. Everything else — the scratch substrate,
/// the engine, the phases — is set up here, because none of it is the caller's business.
pub fn run_on_model(
    probe: &Probe,
    device: &Device,
    tokenizer: tokenizers::Tokenizer,
    model: Box<dyn crate::ManagedBatchedModel + Send>,
) -> anyhow::Result<ProbeOutcome> {
    let args = probe;
    let device = device.clone();
    let builder = probe.builder();

    // **A scratch substrate, never the workspace's own.**
    //
    // `EngineConfig::workspace_path` defaults to `None`, which opens the
    // substrate under the *process working directory* — so run from the repo root
    // this harness appends its conversations to the live `.substrate` and reloads
    // the user's real history on the way in. Held for the engine's whole life;
    // dropping it removes the store, and a previous run's corpse is swept on
    // creation. See `scratch_substrate` for what that does and does not promise.
    let scratch = ScratchSubstrate::new()?;
    if scratch.swept() > 0 {
        println!(
            "swept {} scratch substrate(s) left by a previous run",
            scratch.swept(),
        );
    }
    println!("substrate: {:?}", scratch.path());
    let mut engine_config = builder.engine_config(&tokenizer);
    engine_config.workspace_path = Some(scratch.path().to_path_buf());

    let engine = Arc::new(ConversationEngine::new(
        model,
        tokenizer.clone(),
        engine_config,
        GuestRegistry::new(),
    )?);
    println!("Engine up.\n");

    // The forward gate's own fixture — see `story_slice`.
    let story = candle_transformers::models::batch_test::fixtures::story_prompt();

    let base = sample(&device).unwrap_or_default();
    println!(
        "at rest:            live={:4}  watermark={:4}  holes={:4}  free={:4}  weights={} MiB",
        base.live,
        base.watermark,
        base.holes(),
        base.free,
        base.weight_mib,
    );

    // ── The immovable neighbours ─────────────────────────────────────────────
    //
    // Created before the churn and held until the process ends. Their arenas never
    // come back, so the frees happening around them cannot be re-packed from the
    // bottom — which is the whole mechanism. Held in a `Vec` that outlives phase
    // B on purpose; dropping them early would let the pool tidy itself and the
    // measurement would quietly become the uniform-churn one that showed nothing.
    let mut pinned: Vec<Sequence> = Vec::with_capacity(args.profile.pinned);
    for i in 0..args.profile.pinned {
        let mut c = engine.new_conversation(
            &builder.format_system_prompt(),
            builder.conversation_config(),
        )?;
        // Several turns each, so a pinned resident spans arenas rather than
        // sitting in one.
        for t in 0..3 {
            c.insert_turn(
                story_slice(&story, 4 + (i % 4)),
                story_slice(&story, 1 + (t % 3)),
            )?;
        }
        pinned.push(c);
    }
    let after_pinned = sample(&device).unwrap_or_default();
    println!(
        "pinned {} held:     live={:4}  watermark={:4}  holes={:4}  free={:4}",
        args.profile.pinned,
        after_pinned.live,
        after_pinned.watermark,
        after_pinned.holes(),
        after_pinned.free,
    );

    // ── Phase A: overlapping churn ───────────────────────────────────────────
    //
    // Each worker owns one conversation at a time and retires it at an age that
    // differs per worker, so retirements are spread rather than synchronised. A
    // synchronised retirement frees every arena at once and leaves the pool
    // *packed*, which is the opposite of what this is for.
    let stop = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let live_now = Arc::new(AtomicUsize::new(0));
    // Turn residences the eviction actually flagged, and retirements where it
    // flagged nothing — the difference between churn and the appearance of churn.
    let flagged_total = Arc::new(AtomicUsize::new(0));
    let evict_noops = Arc::new(AtomicUsize::new(0));
    let stragglers_held = Arc::new(AtomicUsize::new(0));
    let args_straggler_every = args.straggler_every;
    let args_straggler_hold = args.straggler_hold_secs;

    let mut workers = Vec::new();
    let concurrency = Arc::new(AtomicUsize::new(args.profile.concurrency));
    let stagger = Arc::new(AtomicUsize::new(args.stagger_ms as usize));

    for id in 0..args.profile.max_concurrency {
        let engine = Arc::clone(&engine);
        let builder = builder.clone();
        let stop = Arc::clone(&stop);
        let started = Arc::clone(&started);
        let retired = Arc::clone(&retired);
        let errors = Arc::clone(&errors);
        let live_now = Arc::clone(&live_now);
        let concurrency = Arc::clone(&concurrency);
        let stagger = Arc::clone(&stagger);
        let flagged_total = Arc::clone(&flagged_total);
        let evict_noops = Arc::clone(&evict_noops);
        let stragglers_held = Arc::clone(&stragglers_held);
        let story = story.clone();

        workers.push(std::thread::spawn(move || {
            // Turns before this worker retires its conversation. Coprime-ish
            // spread so the workers never fall into step.
            let lifetime = 2 + (id % 5);
            let mut round = 0usize;
            while !stop.load(Ordering::Relaxed) {
                // Workers above the current concurrency idle, so saturation can
                // be tuned without respawning threads.
                if id >= concurrency.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                // Stagger the START, which is what interleaves allocation with
                // the other workers' frees.
                let wait = stagger.load(Ordering::Relaxed) as u64 * (1 + id as u64 % 3);
                std::thread::sleep(Duration::from_millis(wait));
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let mut conv: Sequence = match engine.new_conversation(
                    &builder.format_system_prompt(),
                    builder.conversation_config(),
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        errors.lock().unwrap().push(format!("open: {e}"));
                        std::thread::sleep(Duration::from_millis(200));
                        continue;
                    }
                };
                started.fetch_add(1, Ordering::Relaxed);
                live_now.fetch_add(1, Ordering::Relaxed);

                // Grow the conversation's KV over several turns. `insert_turn`
                // prefills both halves with no decode, which is the cheapest way
                // to make a lot of KV — the point here is arena occupancy, not
                // tokens per second.
                for t in 0..lifetime {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let user = story_slice(&story, 2 + ((round + t) % 7));
                    let assistant = story_slice(&story, 1 + (id % 3));
                    if let Err(e) = conv.insert_turn(user, assistant) {
                        errors.lock().unwrap().push(format!("insert_turn: {e}"));
                        break;
                    }
                }

                // A straggler holds on, keeping a high arena live while its
                // burst-mates free around it. Slept rather than turned into
                // another population so that it is the SAME conversation shape as
                // the churn — the only difference is when it lets go, which is the
                // variable under test.
                let straggler =
                    args_straggler_every > 0 && round.is_multiple_of(args_straggler_every);
                if straggler {
                    let hold = Duration::from_secs(args_straggler_hold);
                    let until = Instant::now() + hold;
                    while Instant::now() < until && !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    stragglers_held.fetch_add(1, Ordering::Relaxed);
                }

                // Retire, **the way the ingest layers retire a unit**.
                //
                // Dropping the `Sequence` is not enough and that is the first
                // thing this harness got wrong: `FreeSequence` on drop releases
                // the batch SLOT, not the sealed KV, which the substrate keeps hot
                // until the idle demote sheds it after `IDLE_DEMOTE_GRACE_EPOCHS`
                // — at minimum ~80 s of dormancy. A 90-second run therefore freed
                // nothing at all and `live` only ever climbed.
                //
                // `evict_ingest_timeline` is what `repo_scan::process_one_dir` and
                // `code_read::process_one_file` call for exactly this reason: the
                // unit is complete, nothing will attend it again until a
                // projection retrieves it, so flag every residence
                // `evict_when_cold` and wake the persistence thread. The frees
                // land asynchronously as durability does.
                // The return value is the number of turn residences flagged, and
                // it is NOT ignorable: a zero means the timeline had nothing the
                // eviction could reach, so the retirement freed nothing and the
                // churn is not churning. Accumulated and reported rather than
                // discarded — the first two versions of this harness both failed
                // silently for want of exactly this number.
                let flagged = engine.evict_ingest_timeline(conv.timeline_id());
                flagged_total.fetch_add(flagged, Ordering::Relaxed);
                if flagged == 0 {
                    evict_noops.fetch_add(1, Ordering::Relaxed);
                }
                drop(conv);
                live_now.fetch_sub(1, Ordering::Relaxed);
                retired.fetch_add(1, Ordering::Relaxed);
                round += 1;
            }
        }));
    }

    // Saturation loop: raise concurrency and tighten the stagger until free
    // regions fall to the target, then hold.
    // Fragmentation as the engine itself measures it, read through the same
    // accessor the scheduler logs from — one definition of the figure, not a
    // second opinion computed here.
    // Read from the report the SCHEDULER publishes, not recomputed here: the
    // scheduler fills these rows from `kv_fragmentation`, which is also what its
    // own per-wave log line reads, so the harness cannot disagree with the engine
    // about how fragmented the engine is.
    //
    // Published on the 2 s telemetry cadence, so an early sample may find none.
    let frag_totals = || -> Composition {
        let Some((report, _)) = memory_report::latest() else {
            return Composition {
                eff: 100,
                ..Default::default()
            };
        };
        let freeable: usize = report.kv.classes.iter().map(|c| c.freeable_arenas).sum();
        let packed: usize = report.kv.classes.iter().map(|c| c.packed_arenas).sum();
        let arenas: usize = report.kv.classes.iter().map(|c| c.arenas).sum();
        // **Every figure from the one publish, the frontier included.** Efficiency
        // is `(packed + span) / frontier`, matching `GroundLost::efficiency_pct`: a
        // region below the frontier costs the weight side whether it is live, sparse
        // or free, so the live count is the wrong denominator. No frontier means
        // nothing denied, so a pool at rest is fully efficient.
        //
        // Reading the frontier from `region_stats` here instead — a sample taken
        // between forwards, up to a publish interval away from the class rows —
        // produced 734 regions the class rows could not account for, which reads
        // exactly like a second tenant holding ground below the frontier rather
        // than like the sampling skew it was.
        let frontier = report.kv.frontier_regions;
        // **The span tenants are in the numerator.** They hold whole regions of live
        // data from the same free list and no compaction can pack them, so charging
        // them as fragmentation makes the figure measure the model's architecture
        // rather than the pass: Flash-Next's recurrent state store alone read the gate
        // down from 97% to 4%. Clamped because the two halves come from one publish
        // but not from one lock, so a region released between them must not overflow.
        // Both kinds of in-use-but-unpackable ground. `span` is a tenant's (a recurrent
        // state store, the gallery); `record` is the `KvHead` record arenas, which appear in
        // no size-class row because `gpu_class_stats` reports band pools only. Charging
        // either as waste makes the figure measure the model's architecture rather than the
        // pass — the state store alone read the gate down from 97% to 4% on Flash-Next, and
        // record arenas cost the 30B 11 of its 62 "denied" regions.
        let span = report.kv.span_regions;
        let record = report.kv.record_regions;
        let in_use = (packed + span + record).min(frontier);
        let eff = (in_use * 100).checked_div(frontier).unwrap_or(100);
        Composition {
            denied_mib: freeable * (candle_nn::kv_cache::REGION_BYTES >> 20),
            eff,
            arenas,
            packed,
            frontier,
            live: report.kv.live_regions,
            span,
            record,
            loss_mib: frontier.saturating_sub(in_use) * (candle_nn::kv_cache::REGION_BYTES >> 20),
            captured_ms: report.captured_unix_ms,
        }
    };

    let t_churn = Instant::now();
    let mut worst = Geometry::default();
    let mut worst_freeable_mib = 0usize;
    /// Ground below which a sample's loss is not charged to compaction, in MiB.
    ///
    /// `KV_REGION_SLACK + EXPERT_MIN_GRANT_REGIONS` regions, from
    /// `expert_lre::pipeline`: the weight side is offered spare KV ground only past
    /// 32 regions of standing slack and only in grants of at least 8, so a smaller
    /// loss is ground the boundary negotiation would decline to move for. See the
    /// comment at the judging site.
    const MIN_JUDGED_LOSS_MIB: usize = (32 + 8) * (candle_nn::kv_cache::REGION_BYTES >> 20);

    let mut worst_efficiency = 100usize;
    let mut worst_eff_comp = Composition::default();
    // The worst sample whatever its size or its persistence, so the floors above hide
    // nothing.
    let mut worst_reported = 100usize;
    let mut worst_reported_comp = Composition::default();
    // The previous sample's efficiency, for the sustained measure at the judging site.
    // Starts at 100: the first sample has no predecessor, so it cannot yet be shown to
    // persist, and pairing it with a perfect one is what says so.
    let mut prev_eff = 100usize;
    // The publish the previous judged sample came from, so a report read twice is
    // never mistaken for a figure that persisted. See the judging site.
    let mut prev_captured_ms = 0u64;
    let mut saturated_at: Option<Duration> = None;
    println!("\nphase A — overlapping churn\n");
    println!(
        "   t(s)  conc  live  wmark  holes  free  weightMiB  started  retired  \
         strag  rFront  rLive  rSpan  kvArena  packed  freeableMiB  eff%"
    );
    while t_churn.elapsed() < Duration::from_secs(args.churn_secs) {
        std::thread::sleep(Duration::from_millis(1500));
        let g = sample(&device).unwrap_or_default();
        if g.holes() > worst.holes() {
            worst = g;
        }
        let conc = concurrency.load(Ordering::Relaxed);
        if g.free <= args.saturated_free && saturated_at.is_none() {
            saturated_at = Some(t_churn.elapsed());
        }
        if args.saturate && g.free > args.saturated_free {
            // Not saturated yet: widen, and tighten the stagger once the width
            // ceiling is reached.
            if conc < args.profile.max_concurrency {
                concurrency.store(conc + 2, Ordering::Relaxed);
            } else {
                let s = stagger.load(Ordering::Relaxed);
                stagger.store((s * 3 / 4).max(5), Ordering::Relaxed);
            }
        }
        // The VRAM composition as the row runs — the arena ladder's own view,
        // which is where sparsity lives. `freeable` is what a perfect pack would
        // return; `occ` is how full the arenas the pools hold actually are.
        let frag = frag_totals();
        if frag.eff < worst_reported {
            worst_reported = frag.eff;
            worst_reported_comp = frag;
        }
        // **Judged on a loss that PERSISTS, and only where it is ground the weight
        // side could have been given.**
        //
        // Two floors, and each answers a different way the instantaneous ratio lies.
        //
        // *Absolute.* The weight side is offered KV ground in whole regions, past a
        // standing slack, and never in a grant smaller than its minimum — so a loss
        // under that sum is ground the negotiation would refuse to move for even if
        // compaction handed it back. Without this floor the threshold becomes a
        // statement about the denominator: a handful of just-freed regions reads as
        // 84% at a frontier of 201 and as 97% at a frontier of 900, for the same MiB.
        //
        // *Sustained.* `weight_floor` only moves when the reclaim negotiation runs,
        // and that negotiation holds `KV_REGION_SLACK` regions back precisely so
        // momentary churn does not move the boundary. So a loss that is gone by the
        // next sample never cost the weight side anything — and the loss that shows up
        // in a single sample is exactly the one the design says is not compaction's:
        // free regions below the frontier are taken by the next claim, the region free
        // list being lowest-index-first. Measured: 24 conversations retiring at once
        // leaves one 1.5 s sample at 68% with 85 holes and 11 sparse arenas, and the
        // sample after it reads 90%.
        //
        // Requiring two consecutive *publishes* charges the sparsity, which persists,
        // and not the hole burst, which does not. Publishes, not samples: this loop
        // samples every 1.5 s and the engine publishes every 2, so a pair of samples
        // frequently reads one report twice — which any persistence test would then
        // pass trivially, and did: a dip lasting one publish read as two consecutive
        // bad samples and failed the gate. Both worsts are reported either way.
        let fresh = frag.captured_ms != prev_captured_ms;
        let sustained = if fresh { frag.eff.max(prev_eff) } else { 100 };
        if fresh {
            prev_eff = frag.eff;
            prev_captured_ms = frag.captured_ms;
        }
        if frag.loss_mib >= MIN_JUDGED_LOSS_MIB && sustained < worst_efficiency {
            // The whole composition at THIS moment, not the worst-holes sample's.
            // The two minima need not coincide, and a row pairing one sample's
            // frontier with another's efficiency is a number nobody can reproduce.
            worst_efficiency = sustained;
            // The judged percentage, not this sample's, so the row's `eff%` is the
            // number the gate compared against its threshold. Everything else is this
            // sample's, which is the sample that made the pair as bad as it was.
            worst_eff_comp = Composition {
                eff: sustained,
                ..frag
            };
        }
        println!(
            "  {:5.1}  {:4}  {:4}  {:5}  {:5}  {:4}  {:9}  {:7}  {:7}  {:5}  \
             {:6}  {:5}  {:5}  {:7}  {:6}  {:11}  {:4}",
            t_churn.elapsed().as_secs_f64(),
            conc,
            g.live,
            g.watermark,
            g.holes(),
            g.free,
            g.weight_mib,
            started.load(Ordering::Relaxed),
            retired.load(Ordering::Relaxed),
            stragglers_held.load(Ordering::Relaxed),
            frag.frontier,
            frag.live,
            frag.span + frag.record,
            frag.arenas,
            frag.packed,
            frag.denied_mib,
            frag.eff,
        );
        if frag.denied_mib > worst_freeable_mib {
            worst_freeable_mib = frag.denied_mib;
        }
        let _ = std::io::stdout().flush();
    }

    stop.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.join();
    }
    let after_churn = sample(&device).unwrap_or_default();

    println!(
        "\nworst during churn: live={:4}  watermark={:4}  holes={:4}  free={:4}",
        worst.live,
        worst.watermark,
        worst.holes(),
        worst.free,
    );
    println!(
        "after churn:        live={:4}  watermark={:4}  holes={:4}  free={:4}  weights={} MiB",
        after_churn.live,
        after_churn.watermark,
        after_churn.holes(),
        after_churn.free,
        after_churn.weight_mib,
    );
    println!(
        "evictions:          {} residences flagged over {} retirements, {} of which \
         flagged NOTHING",
        flagged_total.load(Ordering::Relaxed),
        retired.load(Ordering::Relaxed),
        evict_noops.load(Ordering::Relaxed),
    );
    match saturated_at {
        Some(d) => println!("saturated after {:.1}s", d.as_secs_f64()),
        None => println!(
            "NEVER SATURATED — free never fell to {}. Raise --max-concurrency or \
             lower --stagger-ms; a fragmentation figure taken on a half-empty pool \
             measures nothing.",
            args.saturated_free,
        ),
    }

    // ── Phase B: the comparison batch, against the fragmented pool ───────────
    //
    // Deliberately NOT preceded by a cleanup: the whole point is to run the
    // gate's shape on the pool as churn left it.
    println!("\nphase B — concurrent prefill + decode on the fragmented pool\n");
    let before = sample(&device).unwrap_or_default();

    // **A real StoryRewrite, per session, validated like the gate validates it.**
    //
    // Throughput on a fragmented pool is only half the question; the other half is
    // whether the answers are still right. A compaction that relocates a chunk and
    // leaves one band pointer stale does not fault — every address in the
    // reservation is mapped — so it surfaces as a session quietly reading another
    // session's KV. Timings cannot see that. A rewrite can: each session is given a
    // DIFFERENT protagonist name, so cross-contamination shows up as the wrong name
    // or the wrong text, and the per-session names make it detectable rather than
    // merely likely to look odd.
    let names = candle_transformers::models::batch_test::fixtures::session_names();
    let t_prefill = Instant::now();
    let mut convs: Vec<Sequence> = Vec::with_capacity(args.profile.batch);
    let mut expected: Vec<String> = Vec::with_capacity(args.profile.batch);
    let mut prompts: Vec<String> = Vec::with_capacity(args.profile.batch);
    let mut prefill_tokens = 0usize;
    for i in 0..args.profile.batch {
        // Indexed exactly as the gate indexes it, so session identities match.
        let name = &names[i % names.len()];
        // **The gate's own system prompt, with the name substituted the way the
        // gate substitutes it.** The rewrite is entirely a function of this text:
        // "Output ONLY the story text", "keep the title, punctuation, spelling,
        // casing … EXACTLY the same". Under the daemon's own system prompt the model
        // answers helpfully instead — measured, every session replied
        // `**Title:** …`, which is a correct answer to a different question and
        // matches no prefix of the expected rewrite.
        let system = fixtures::system_prompt().replace("{INSERT_NAME}", name);
        let c = engine.new_conversation(&system, builder.conversation_config())?;
        let user = fixtures::story_rewrite_prompt(&story, name);
        expected.push(fixtures::story_rewrite_expected(&story, name));
        prefill_tokens += tokenizer
            .encode(user.as_str(), false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);
        prompts.push(user);
        convs.push(c);
    }

    // **One turn, whose user half IS the story prompt.** The gate's `StoryRewrite`
    // is a single exchange: the fixture carries its own rename instruction, and the
    // assistant's reply to it is the rewrite. Prefilling the story as its own turn
    // and then asking a *second* question produces an answer to that question, which
    // is not a prefix of the expected rewrite and fails every session — measured,
    // 0/20, and it was this harness's mistake rather than the engine's.
    let t_decode = Instant::now();
    let mut handles = Vec::with_capacity(convs.len());
    for (c, prompt) in convs.iter_mut().zip(prompts.iter()) {
        let opts = TurnOptions {
            // Enough of the rewrite to catch a wrong name or wrong text without
            // paying for the whole story. The comparison is prefix-based.
            max_tokens: Some(args.batch_decode.max(48)),
            // **Thinking off, or the answer never starts.** The forward gate drives
            // the session directly with a raw prompt, so its `StoryRewrite` reply
            // begins with the rewrite. A turn through the conversation engine opens
            // a reasoning block first, and the whole token budget goes into
            // `<think>\nOkay, let me try …` — measured, 0/20 sessions, with the
            // decode never reaching the story at all.
            selection: no_think(),
            // **Greedy, because this check is about KV and not about sampling.** The
            // rewrite is a verbatim reproduction, so with any temperature at all a
            // session can diverge honestly: measured, one session of twenty matched 195
            // characters exactly and then paraphrased the next clause, which is a
            // sampler result and reads in the report as the corruption this check
            // exists to find. Greedy makes a divergence mean what the check says it
            // means.
            sampling: Some(greedy_sampling(builder.conversation_config())),
            ..Default::default()
        };
        handles.push(c.submit_turn_with_options(prompt.as_str(), opts)?);
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    let mut decoded = 0usize;
    let mut outputs: Vec<String> = vec![String::new(); convs.len()];
    for (i, (c, h)) in convs.iter_mut().zip(handles).enumerate() {
        match h.wait_cancellable() {
            Ok(resp) => {
                decoded += resp.token_ids.len();
                outputs[i] = resp.text.clone();
                // Seal it the way a live turn does, so the KV this batch created
                // is recorded rather than abandoned — abandoning would free it
                // early and quietly flatter the geometry this example reports.
                if let Err(e) = c.finish_turn(h, &resp) {
                    errors.lock().unwrap().push(format!("finish_turn: {e}"));
                }
            }
            Err(e) => errors.lock().unwrap().push(format!("decode: {e}")),
        }
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    let fragmented = sample(&device).unwrap_or_default();
    // **Read the composition here, beside the geometry it describes.** Taken at the
    // end of the run instead, it reported phase B's frontier against the *drain's*
    // arena counts — 757 regions denied with 21 arenas live — and called that 98%
    // efficient. One row, one moment.
    let end = frag_totals();
    // The breakdown of the row above, read from the same publish `frag_totals`
    // just took — so it explains phase B rather than whatever the drain leaves
    // behind minutes later.
    print_tenant_census("after phase B");

    // ── Story validation, by the profile's gate ──────────────────────────────
    //
    // `StoryGate::Verbatim` is the forward gate's own rule: `normalize_story`, then a
    // common-prefix comparison with a 5-char tolerance, read from the same module so
    // "correct" means here what it means there. The output is short next to the whole
    // story, so it asks whether the model produced a correct PREFIX of the rewrite —
    // which is what catches a wrong name, wrong content, or broken attention.
    //
    // `StoryGate::OwnName` is the floor for a model that cannot reproduce prose
    // exactly. It asks only that each session names its own protagonist, which still
    // catches the failure this check exists for: a session reading another session's KV
    // renames to *that* session's protagonist, so its own name never appears.
    let mut story_pass = 0usize;
    let mut story_fail: Vec<String> = Vec::new();
    for (i, out) in outputs.iter().enumerate() {
        let got = normalize_story(strip_think(out).trim());
        let want = normalize_story(expected[i].trim());
        let name = &names[i % names.len()];
        if args.story_gate() == StoryGate::OwnName {
            if got.contains(name.as_str()) {
                story_pass += 1;
            } else {
                let show = got.chars().take(120).collect::<String>();
                story_fail.push(format!(
                    "session {i}: its own protagonist {name:?} is absent from the reply\n      got:  {show:?}",
                ));
            }
            continue;
        }
        let g: Vec<char> = got.chars().collect();
        let w: Vec<char> = want.chars().collect();
        let common = g.iter().zip(w.iter()).take_while(|(a, b)| a == b).count();
        let min_len = g.len().min(w.len());
        const TOLERANCE: usize = 5;
        // An empty or near-empty decode is a failure, not a vacuous pass: with
        // `min_len` at zero the tolerance would make `required` zero and anything
        // would match.
        let required = min_len.saturating_sub(TOLERANCE);
        if min_len >= 16 && common >= required {
            story_pass += 1;
        } else {
            let show = min_len.min(common + 24);
            story_fail.push(format!(
                "session {i} (name {name}): matched {common}/{min_len} chars\n      got:  {:?}\n      want: {:?}",
                g[..show.min(g.len())].iter().collect::<String>(),
                w[..show.min(w.len())].iter().collect::<String>(),
            ));
        }
    }

    println!(
        "  prefill: {:.0} tok in {:.2}s  = {:.1} t/s",
        prefill_tokens as f64,
        prefill_s,
        prefill_tokens as f64 / prefill_s.max(1e-9),
    );
    println!(
        "  decode:  {:.0} tok in {:.2}s  = {:.1} t/s   ({} sequences)",
        decoded as f64,
        decode_s,
        decoded as f64 / decode_s.max(1e-9),
        args.profile.batch,
    );
    println!(
        "  geometry before batch: live={} watermark={} holes={}",
        before.live,
        before.watermark,
        before.holes(),
    );
    println!(
        "  geometry after  batch: live={} watermark={} holes={}  weights={} MiB",
        fragmented.live,
        fragmented.watermark,
        fragmented.holes(),
        fragmented.weight_mib,
    );

    let errs = errors.lock().unwrap();
    if !errs.is_empty() {
        println!("\n{} error(s) during the run:", errs.len());
        for e in errs.iter().take(12) {
            println!("  {e}");
        }
    }

    if worst.mid_wave || after_churn.mid_wave || before.mid_wave {
        println!(
            "\nNOTE: at least one reading was taken mid-wave (a tier was standing), \
             so its `free` is the region ceiling's answer and not the pool's. \
             Treat that row as indicative only."
        );
    }
    println!(
        "\n{} pinned conversation(s) still held through phases A and B, so nothing \
         re-packed underneath the measurement. Phase C releases them.",
        pinned.len(),
    );

    // ── Phase C: the drain, and whether the weight side takes the ground ─────
    //
    // Evict everything and watch two numbers that must move together: the arena
    // frontier falling, and the weight zone growing by what the frontier gave up.
    // The first is the KV side letting go; the second is the weight side taking,
    // and only the pair of them is a reclaim. A run where the frontier falls by a
    // gigabyte and `weights` does not budge has freed nothing anyone can use.
    println!("\nphase C — drain, and the weight side's uptake\n");
    let before_drain = sample(&device).unwrap_or_default();
    for c in convs.iter() {
        let _ = engine.evict_ingest_timeline(c.timeline_id());
    }
    drop(convs);
    for c in pinned.iter() {
        let _ = engine.evict_ingest_timeline(c.timeline_id());
    }
    let pinned_count = pinned.len();
    drop(pinned);
    println!(
        "  evicted and dropped {} batch + {} pinned conversations",
        args.profile.batch, pinned_count,
    );

    // **A keepalive, because the reclaim only runs inside the wave loop.**
    //
    // `reclaim_spare_ground` — the thing that actually moves `weight_floor` left —
    // is driven from the wave loop, at the point where no wave generation is live.
    // A drain with nothing decoding turns that loop zero times, so the weight side
    // is never asked and `weights` reads flat however much ground the KV side let
    // go. Measured twice before this existed: the frontier fell 272 and then 140
    // regions, and `weights` did not move by one MiB either time. That was the
    // harness idling, not the engine refusing.
    //
    // One short turn every couple of seconds is enough to turn the loop, and it is
    // also the honest shape: a daemon always has something in flight.
    let mut keepalive = engine.new_conversation(
        &builder.format_system_prompt(),
        builder.conversation_config(),
    )?;

    println!(
        "   t(s)  frontier  live  free  weightMiB  rFront  rLive  rSpan  kvArena  packed  eff%"
    );
    let t_drain = Instant::now();
    // The lowest frontier and the highest weight zone the drain reached — the pair
    // the uptake gate is judged on. Extremes rather than the final sample, because
    // the two move on different clocks: the frontier falls when the last chunk of an
    // arena goes, and the weight side takes the ground on the next wave that asks.
    // A final-sample comparison scores whichever happened to be mid-flight when the
    // clock ran out.
    let mut lowest_frontier = before_drain.watermark;
    let mut highest_weight_mib = before_drain.weight_mib;
    while t_drain.elapsed() < Duration::from_secs(args.drain_secs) {
        // Turn the wave loop so the reclaim gets a chance to run.
        let opts = TurnOptions {
            max_tokens: Some(4),
            selection: no_think(),
            ..Default::default()
        };
        match keepalive.submit_turn_with_options("Say ok.", opts) {
            Ok(h) => match h.wait_cancellable() {
                Ok(resp) => {
                    let _ = keepalive.finish_turn(h, &resp);
                }
                Err(e) => errors.lock().unwrap().push(format!("keepalive: {e}")),
            },
            Err(e) => errors
                .lock()
                .unwrap()
                .push(format!("keepalive submit: {e}")),
        }
        // **And evict what it just sealed.** A keepalive that keeps its turns is a
        // conversation growing through the whole drain: measured, the frontier rose
        // 1,472 → 1,616 over 60 s and the phase reported zero regions released,
        // because the thing turning the wave loop was also filling the pool it was
        // meant to be watching empty.
        let _ = engine.evict_ingest_timeline(keepalive.timeline_id());
        std::thread::sleep(Duration::from_millis(1500));
        let g = sample(&device).unwrap_or_default();
        lowest_frontier = lowest_frontier.min(g.watermark);
        highest_weight_mib = highest_weight_mib.max(g.weight_mib);
        let c = frag_totals();
        println!(
            "  {:5.1}  {:8}  {:4}  {:4}  {:9}  {:6}  {:5}  {:5}  {:7}  {:6}  {:4}",
            t_drain.elapsed().as_secs_f64(),
            g.watermark,
            g.live,
            g.free,
            g.weight_mib,
            c.frontier,
            c.live,
            c.span + c.record,
            c.arenas,
            c.packed,
            c.eff,
        );
        let _ = std::io::stdout().flush();
    }
    let after_drain = sample(&device).unwrap_or_default();

    let regions_released = before_drain.watermark.saturating_sub(lowest_frontier);
    let mib_released = regions_released * (candle_nn::kv_cache::REGION_BYTES >> 20);
    let weight_growth_mib = highest_weight_mib.saturating_sub(before_drain.weight_mib);
    // Nothing released means nothing was owed, so a drain with nothing to give
    // reports 100 and cannot fail the threshold.
    let uptake_pct = (weight_growth_mib * 100)
        .checked_div(mib_released)
        .unwrap_or(100);
    println!(
        "\n  frontier {} -> {} (lowest {}; {} regions, {} MiB released)",
        before_drain.watermark,
        after_drain.watermark,
        lowest_frontier,
        regions_released,
        mib_released,
    );
    println!(
        "  weights  {} -> {} MiB (peak {}; grew {} MiB = {}% of what the KV side released)",
        before_drain.weight_mib,
        after_drain.weight_mib,
        highest_weight_mib,
        weight_growth_mib,
        uptake_pct,
    );

    // ── The engine's own ledgers ─────────────────────────────────────────────
    //
    // Both failures below are about work the engine either did not get to do or
    // could not do, and neither is legible from a geometry sample. The compaction
    // tally separates "the gate never opened" from "every pass was refused the
    // window", which read identically from outside and want opposite fixes. The
    // growth tally does the same for the weight side: `at_limit` means the expert
    // cache already holds every expert and has nothing to take, which is not the
    // same failure as `floor_refused`.
    let t = candle_nn::kv_cache::compaction_tally();
    println!(
        "\ncompaction: attempts={} passes={} refused(wave={} migrate={} packed={} failed={} \
         nodev={}) clipped={} moves={} arenas_released={} regions_reclaimed={}",
        t.attempts,
        t.passes,
        t.wave_in_flight,
        t.migrate_in_flight,
        t.already_packed,
        t.step_failed,
        t.no_device,
        t.clipped,
        t.moves,
        t.arenas_released,
        t.regions_reclaimed,
    );
    let g = candle_transformers::models::expert_lre::grow_tally();
    println!(
        "weight grow: asked={} no_spare={} spare_offered={} target_unchanged={} \
         target_backwards={} floor_refused={} at_limit={} slots_gained={}",
        g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
    );

    // ── Results ──────────────────────────────────────────────────────────────
    println!("\n=== VRAM efficiency ===\n");
    println!(
        "  phase                    frontier  live  span  rec  kvArena  packed  eff%   lossMiB"
    );
    // Every column of a row comes from one `Composition`, so the frontier printed
    // is the frontier the percentage was divided by.
    let row = |name: &str, c: &Composition| {
        println!(
            "  {name:<22}  {:8}  {:4}  {:4}  {:3}  {:7}  {:6}  {:4}   {:7}",
            c.frontier, c.live, c.span, c.record, c.arenas, c.packed, c.eff, c.loss_mib,
        );
    };
    row(
        "at rest",
        &Composition {
            eff: 100,
            ..Default::default()
        },
    );
    // A run where no sample was both persistent and large enough has nothing to show
    // here, and a row of zeros would read as a frontier of nothing at 0% efficiency.
    if worst_efficiency < 100 {
        row("worst sustained", &worst_eff_comp);
    } else {
        println!(
            "  {:<22}  none — no sample was both big enough and persistent enough to judge",
            "worst sustained"
        );
    }
    row("worst single sample", &worst_reported_comp);
    row("after phase B", &end);
    println!(
        "\n  The judged figure is the worst efficiency that PERSISTED across two \
         samples (3 s) and\n  whose loss exceeded {} MiB — the standing KV slack plus \
         the weight side's minimum\n  grant, below which the boundary negotiation \
         would not move for it. `weight_floor`\n  only moves when that negotiation \
         runs, so a loss gone by the next sample never cost\n  the weight side \
         anything: free regions below the frontier are taken by the next\n  claim, the \
         region free list being lowest-index-first. The worst single sample is \
         shown\n  beside it so neither floor hides anything.",
        MIN_JUDGED_LOSS_MIB,
    );
    // Reported separately because the worst-efficiency sample and the
    // worst-holes sample need not be the same moment, and a row that mixes two
    // samples' columns is a number nobody can reproduce.
    println!(
        "  (most holes seen: frontier {} over {} live = {} stranded regions)",
        worst.watermark,
        worst.live,
        worst.holes(),
    );

    println!(
        "\n  efficiency = (packed + span + rec) / frontier — of the ground denied to \
         the weight side,\n  the share actually holding something. The frontier is the \
         denominator because the\n  frontier is what the weight side loses: the tier \
         stands above the highest live\n  arena and `weight_floor` is measured from \
         there.\n\n  `span` is whole regions a span tenant holds — a sequence's \
         recurrent state store, the\n  provenance gallery. `rec` is the `KvHead` record \
         arenas, which are in no size-class\n  row at all because the class stats report \
         band pools only. Both are in use and\n  neither is packable, so they belong \
         beside `packed` and not in the loss; charged as\n  waste they make a perfectly \
         packed pool look fragmented.\n\n  The remainder splits into the two real \
         losses: air inside the arenas\n  (kvArena - packed), which a pack removes, and \
         free regions stranded below the\n  frontier (frontier - live), which only a \
         falling frontier removes."
    );

    print_tenant_census("end of run");

    println!(
        "\nCompare the phase-B figures against the forward gate's CLEAN rows for \
         the same width. The gap is what fragmentation costs."
    );

    // ── The gates ────────────────────────────────────────────────────────────
    //
    // Both are evaluated, both are reported, and only then does the run fail. They
    // are the two halves of one claim — pack the KV, and let the weight side have
    // what packing released — so failing on the first and never printing the second
    // would hide half the specification. A threshold that suppresses its sibling's
    // evidence is worse than no threshold.
    let mut failures: Vec<String> = Vec::new();
    // **Correctness first, because it outranks both memory thresholds.** A
    // fragmented pool that answers correctly is a performance problem; a packed
    // pool that answers wrongly is a corruption. If a later compaction pass ever
    // trades the second for the first, this is the line that says so.
    if story_pass == outputs.len() {
        println!(
            "\nPASS  {}/{} sessions rewrote the story correctly.",
            story_pass,
            outputs.len(),
        );
    } else {
        for f in &story_fail {
            println!("\n  {f}");
        }
        failures.push(format!(
            "only {}/{} sessions rewrote the story correctly — a session reading \
             another session's KV does not fault, it answers wrongly, so this is \
             the check that sees it",
            story_pass,
            outputs.len(),
        ));
    }
    if worst_efficiency < args.profile.min_efficiency {
        failures.push(format!(
            "VRAM efficiency fell to {}% (threshold {}%): {} MiB of the ground denied \
             to the weight side was not holding KV",
            worst_efficiency, args.profile.min_efficiency, worst_eff_comp.loss_mib,
        ));
    } else {
        println!(
            "\nPASS  VRAM efficiency held at or above {}% (worst {}%).",
            args.profile.min_efficiency, worst_efficiency,
        );
    }
    // **A weight side that already holds every expert has nothing to take, and that
    // is not a failure of compaction.** `capacity_for_frontier` clamps to the zone's
    // limit — the slots the model actually has — so on a card with room for the whole
    // checkpoint the zone cannot grow however much ground the KV side hands back, and
    // the engine records that as `at_limit`. Measured here: 1,014 `at_limit` against
    // 1.3 million regions offered, with a 19,296 MiB zone holding a 19,296 MiB model.
    //
    // Read from the ledger rather than inferred, and narrow: it excuses nothing when
    // the zone had room (`target_unchanged`) or when the floor refused the move
    // (`floor_refused`), which are the two real defects this gate exists to catch.
    let at_limit = g[6] > 0 && g[3] == 0 && g[5] == 0 && g[7] == 0;
    if at_limit {
        println!(
            "PASS  the weight side is at its limit ({} at-limit answers): every expert \
             slot the model has is resident, so there is no residency for the {} MiB \
             released to buy. On a card that cannot hold the whole checkpoint this is \
             where the gain would land.",
            g[6], mib_released,
        );
    } else if uptake_pct < args.profile.min_weight_uptake {
        failures.push(format!(
            "the weight side took only {}% of the {} MiB the KV side released \
             (threshold {}%): the frontier fell by {} regions and `weights` grew by \
             {} MiB — freed ground that nothing claims buys no decode",
            uptake_pct,
            mib_released,
            args.profile.min_weight_uptake,
            regions_released,
            weight_growth_mib,
        ));
    } else {
        println!(
            "PASS  the weight side took {}% of the {} MiB released.",
            uptake_pct, mib_released,
        );
    }

    for f in &failures {
        println!("\nFAIL  {f}");
    }
    // Returned rather than raised. The caller decides what a failure means — a test
    // asserts, a driver exits non-zero — and either way it has the measurements, which
    // an error carrying only a count would have thrown away.
    Ok(ProbeOutcome {
        story_pass,
        story_total: outputs.len(),
        worst_sustained_efficiency: worst_efficiency,
        worst_single_efficiency: worst_reported,
        weight_uptake_pct: uptake_pct,
        weight_at_limit: at_limit,
        prefill_tps: prefill_tokens as f64 / prefill_s.max(1e-9),
        decode_tps: decoded as f64 / decode_s.max(1e-9),
        // The geometry phase B was delivered on, so a table row states rate and
        // fragmentation as one observation rather than two.
        frontier_regions: end.frontier,
        efficiency_pct: end.eff,
        peak_tokens: prefill_tokens,
        failures,
    })
}
