//! Drive the real engine into KV fragmentation, then measure what it costs.
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
//! Run it:
//!
//! ```text
//! cargo run --release --features cuda --example kv_fragmentation -- --saturate
//! ```

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::scratch_substrate::ScratchSubstrate;
use candle_conversation::{ConversationEngine, Sequence};
use candle_nn::kv_cache::region_stats;
use candle_transformers::models::batch_test::fixtures;
use candle_transformers::models::batch_test::story_normalize::normalize_story;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Fragment the KV pool with overlapping conversations, then measure the cost")]
struct Args {
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Conversations alive at once when the run starts. `--saturate` raises it.
    #[arg(long, default_value_t = 8)]
    concurrency: usize,

    /// Ceiling on concurrency while saturating.
    #[arg(long, default_value_t = 48)]
    max_concurrency: usize,

    /// Delay between starting new conversations, in milliseconds. Lower is more
    /// overlap; the saturation loop lowers it as well as raising concurrency.
    #[arg(long, default_value_t = 120)]
    stagger_ms: u64,

    /// Raise concurrency and tighten the stagger until the pool saturates.
    #[arg(long, default_value_t = false)]
    saturate: bool,

    /// Free regions at or below which the pool counts as saturated.
    #[arg(long, default_value_t = 24)]
    saturated_free: usize,

    /// Seconds to hold the overlapping churn once saturated.
    #[arg(long, default_value_t = 90)]
    churn_secs: u64,

    /// Long-lived conversations created once and held for the whole run — the
    /// immovable neighbours.
    ///
    /// **This is the half that makes holes.** Uniform churn does not fragment:
    /// every arena empties eventually and the pool re-packs from the bottom.
    /// Fragmentation needs frees landing *around things that never move*, which in
    /// the daemon is the priming chain, the dialogue base conversations and the
    /// section ingests sitting resident while ingest units are evicted beneath
    /// them.
    #[arg(long, default_value_t = 6)]
    pinned: usize,

    /// One retirement in this many is a **straggler**: held for
    /// `--straggler-hold-secs` before being evicted, instead of retiring with its
    /// burst.
    ///
    /// **This is what keeps the RIGHT side alive.** The pinned conversations
    /// cannot: they are created before the churn, so they sit at the *lowest*
    /// arena indices and the frontier rises past them immediately. A straggler is
    /// claimed when the frontier is already high and then survives while
    /// everything around it frees — so it pins the watermark up there while the
    /// churn below it opens holes. Each burst leaves one a little higher than the
    /// last, and the frontier ratchets.
    ///
    /// Without this the allocator wins: its free list is lowest-index-first, so a
    /// hole is exactly what the next claim consumes and the live set stays
    /// contiguous from zero. Measured — holes peaked at 34 and settled back to 1.
    #[arg(long, default_value_t = 5)]
    straggler_every: usize,

    /// How long a straggler holds its KV before being evicted.
    #[arg(long, default_value_t = 25)]
    straggler_hold_secs: u64,

    /// Minimum VRAM efficiency, as a percentage, below which this run **fails**.
    ///
    /// Efficiency is `packed_arenas / frontier`: of the ground denied to the weight
    /// side, how much is actually holding KV. The frontier is the denominator
    /// because the frontier is what the weight side loses.
    ///
    /// **This is expected to FAIL until continuous compaction lands, and that is
    /// the point.** The failure is the specification: it says how much VRAM
    /// fragmentation is costing, in the one unit that matters, and the same number
    /// passing is what says compaction worked. A run that cannot fail cannot tell
    /// you that.
    #[arg(long, default_value_t = 90)]
    min_efficiency: usize,

    /// Of the KV ground released during the drain phase, the minimum percentage the
    /// **weight side must take**, below which this run fails.
    ///
    /// **Freeing ground is only half the job.** The span is
    /// `| persist | KV regions | tier | expert weights |` with `weight_floor`
    /// between the last two, and lowering the arena frontier merely makes it
    /// *possible* for that floor to move left. Something has to actually move it. If
    /// it does not, compaction hands back regions nobody claims and decode is
    /// exactly as slow as before — the work would be invisible in every metric
    /// except the one that matters.
    ///
    /// This is a known gap, not a hypothetical: the elastic-partition design ledger
    /// lists "the weight side *taking* the ground the tier gives back" as **open**,
    /// and across every run of this harness `weights` has not moved by a single MiB.
    /// So expect this threshold to fail alongside `--min-efficiency` until both
    /// halves land.
    #[arg(long, default_value_t = 50)]
    min_weight_uptake: usize,

    /// Seconds to wait for the drain after everything is evicted, watching the
    /// frontier fall and the weight zone grow.
    #[arg(long, default_value_t = 90)]
    drain_secs: u64,

    /// Concurrent sequences in the phase-B comparison batch. 20 matches the
    /// gate's `Q8_0 × 20` row.
    #[arg(long, default_value_t = 20)]
    batch: usize,

    /// Tokens each phase-B sequence decodes.
    #[arg(long, default_value_t = 32)]
    batch_decode: usize,
}

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

/// One raw reading, whatever the wave is doing.
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
fn story_slice(story: &str, fraction: usize) -> &str {
    let want = story.len() * fraction.clamp(1, 8) / 8;
    // Cut on a char boundary — the story is not ASCII.
    let mut end = want.min(story.len());
    while end > 0 && !story.is_char_boundary(end) {
        end -= 1;
    }
    &story[..end]
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init()
        .ok();

    let device = Device::new_cuda(args.device)?;

    // Q4_K_M on the 30B-A3B, matching the forward gate's own checkpoint so the
    // phase-B figures are comparable to its rows rather than to another model.
    let builder = Model::Qwen3_30B_A3B_Q4
        .builder()
        .max_concurrent(args.max_concurrency + args.batch + 4)
        .max_seq_len(8192);
    let (model_path, tokenizer_path) = builder.resolve_paths_pub()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    println!("Loading {model_path:?} …");
    let model = builder.load_model(&model_path, &device, None)?;

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
        candle_conversation::guest::GuestRegistry::new(),
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
    let mut pinned: Vec<Sequence> = Vec::with_capacity(args.pinned);
    for i in 0..args.pinned {
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
        args.pinned,
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
    let concurrency = Arc::new(AtomicUsize::new(args.concurrency));
    let stagger = Arc::new(AtomicUsize::new(args.stagger_ms as usize));

    for id in 0..args.max_concurrency {
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
    let frag_totals = |g: &Geometry| -> (usize, usize) {
        let Some((report, _)) = candle_conversation::memory_report::latest() else {
            return (0, 100);
        };
        let freeable: usize = report.kv.classes.iter().map(|c| c.freeable_arenas).sum();
        let packed: usize = report.kv.classes.iter().map(|c| c.packed_arenas).sum();
        // Efficiency against the FRONTIER, matching `GroundLost::efficiency_pct`:
        // a region below the frontier costs the weight side whether it is live,
        // sparse or free, so the live count is the wrong denominator.
        // No frontier means nothing denied, so a pool at rest is fully efficient.
        let eff = (packed * 100).checked_div(g.watermark).unwrap_or(100);
        (freeable * (candle_nn::kv_cache::REGION_BYTES >> 20), eff)
    };

    let t_churn = Instant::now();
    let mut worst = Geometry::default();
    let mut worst_freeable_mib = 0usize;
    let mut worst_efficiency = 100usize;
    let mut worst_eff_geom = Geometry::default();
    let mut worst_eff_mib = 0usize;
    let mut saturated_at: Option<Duration> = None;
    println!("\nphase A — overlapping churn\n");
    println!(
        "   t(s)  conc  live  wmark  holes  free  weightMiB  started  retired  \
         strag  freeableMiB  occ%"
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
            if conc < args.max_concurrency {
                concurrency.store(conc + 2, Ordering::Relaxed);
            } else {
                let s = stagger.load(Ordering::Relaxed);
                stagger.store((s * 3 / 4).max(5), Ordering::Relaxed);
            }
        }
        // The VRAM composition as the row runs — the arena ladder's own view,
        // which is where sparsity lives. `freeable` is what a perfect pack would
        // return; `occ` is how full the arenas the pools hold actually are.
        let frag = frag_totals(&g);
        if frag.1 < worst_efficiency {
            // Capture the geometry AT THIS MOMENT, not the worst-holes sample's.
            // The two minima need not coincide, and a row pairing one sample's
            // frontier with another's efficiency is a number nobody can reproduce.
            worst_efficiency = frag.1;
            worst_eff_geom = g;
            worst_eff_mib = frag.0;
        }
        println!(
            "  {:5.1}  {:4}  {:4}  {:5}  {:5}  {:4}  {:9}  {:7}  {:7}  {:5}  {:11}  {:4}",
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
            frag.0,
            frag.1,
        );
        if frag.0 > worst_freeable_mib {
            worst_freeable_mib = frag.0;
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
    let mut convs: Vec<Sequence> = Vec::with_capacity(args.batch);
    let mut expected: Vec<String> = Vec::with_capacity(args.batch);
    let mut prefill_tokens = 0usize;
    for i in 0..args.batch {
        let mut c = engine.new_conversation(
            &builder.format_system_prompt(),
            builder.conversation_config(),
        )?;
        // Indexed exactly as the gate indexes it, so session identities match.
        let name = &names[i % names.len()];
        let user = fixtures::story_rewrite_prompt(&story, name);
        expected.push(fixtures::story_rewrite_expected(&story, name));
        prefill_tokens += tokenizer
            .encode(user.as_str(), false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);
        // Prefill the prompt only. The rewrite is DECODED below — prefilling an
        // assistant half here would put the answer in the context and measure
        // nothing.
        c.insert_turn(&user, "")?;
        convs.push(c);
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // Decode all of them concurrently: submit every turn first, then wait, so the
    // scheduler sees one wide decode rather than a series of narrow ones.
    let t_decode = Instant::now();
    let mut handles = Vec::with_capacity(convs.len());
    for c in convs.iter_mut() {
        handles.push(c.submit_turn("Rewrite the story exactly as instructed.")?);
    }
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

    // ── Story validation, by the gate's own rule ─────────────────────────────
    //
    // `normalize_story` then a common-prefix comparison with a 5-char tolerance:
    // the same normalisation and the same tolerance `utils::validate_and_print_results`
    // applies, read from the same module, so "correct" means here what it means
    // there. The output is short next to the whole story, so this asks whether the
    // model produced a correct PREFIX of the rewrite — which is what catches a
    // wrong name, wrong content, or broken attention.
    let mut story_pass = 0usize;
    let mut story_fail: Vec<String> = Vec::new();
    for (i, out) in outputs.iter().enumerate() {
        let got = normalize_story(out.trim());
        let want = normalize_story(expected[i].trim());
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
                "session {i} (name {}): matched {common}/{min_len} chars\n      got:  {:?}\n      want: {:?}",
                names[i % names.len()],
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
        args.batch,
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
        args.batch, pinned_count,
    );

    println!("   t(s)  frontier  live  free  weightMiB  eff%");
    let t_drain = Instant::now();
    let mut best_drain = before_drain;
    while t_drain.elapsed() < Duration::from_secs(args.drain_secs) {
        std::thread::sleep(Duration::from_millis(2000));
        let g = sample(&device).unwrap_or_default();
        // Track the LOWEST frontier the drain reached: the reclaim's high-water
        // mark, and what the weight side had the opportunity to take.
        if g.watermark < best_drain.watermark {
            best_drain = g;
        }
        println!(
            "  {:5.1}  {:8}  {:4}  {:4}  {:9}  {:4}",
            t_drain.elapsed().as_secs_f64(),
            g.watermark,
            g.live,
            g.free,
            g.weight_mib,
            frag_totals(&g).1,
        );
        let _ = std::io::stdout().flush();
    }
    let after_drain = sample(&device).unwrap_or_default();

    let regions_released = before_drain.watermark.saturating_sub(after_drain.watermark);
    let mib_released = regions_released * (candle_nn::kv_cache::REGION_BYTES >> 20);
    let weight_growth_mib = after_drain
        .weight_mib
        .saturating_sub(before_drain.weight_mib);
    // Nothing released means nothing was owed, so a drain with nothing to give
    // reports 100 and cannot fail the threshold.
    let uptake_pct = (weight_growth_mib * 100)
        .checked_div(mib_released)
        .unwrap_or(100);
    println!(
        "\n  frontier {} -> {} ({} regions, {} MiB released)",
        before_drain.watermark, after_drain.watermark, regions_released, mib_released,
    );
    println!(
        "  weights  {} -> {} MiB (grew {} MiB = {}% of what the KV side released)",
        before_drain.weight_mib, after_drain.weight_mib, weight_growth_mib, uptake_pct,
    );

    // ── Results ──────────────────────────────────────────────────────────────
    let end = frag_totals(&fragmented);
    println!("\n=== VRAM efficiency ===\n");
    println!("  phase                    frontier  packed  eff%   deniedMiB");
    let row = |name: &str, g: &Geometry, eff: usize, mib: usize| {
        println!(
            "  {name:<22}  {:8}  {:6}  {:4}   {:9}",
            g.watermark,
            // Derived from the efficiency and the frontier it was measured
            // against, so all three columns of a row are one sample's.
            g.watermark * eff / 100,
            eff,
            mib,
        );
    };
    row("at rest", &base, 100, 0);
    row(
        "worst efficiency",
        &worst_eff_geom,
        worst_efficiency,
        worst_eff_mib,
    );
    row("after phase B", &fragmented, end.1, end.0);
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
        "\n  efficiency = packed_arenas / frontier — of the ground denied to the \
         weight side,\n  the share actually holding KV. The frontier is the \
         denominator because the\n  frontier is what the weight side loses: the \
         tier stands above the highest live\n  arena and `weight_floor` is measured \
         from there."
    );

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
    if worst_efficiency < args.min_efficiency {
        failures.push(format!(
            "VRAM efficiency fell to {}% (threshold {}%): {} MiB of the ground denied \
             to the weight side was not holding KV",
            worst_efficiency, args.min_efficiency, worst_eff_mib,
        ));
    } else {
        println!(
            "\nPASS  VRAM efficiency held at or above {}% (worst {}%).",
            args.min_efficiency, worst_efficiency,
        );
    }
    if uptake_pct < args.min_weight_uptake {
        failures.push(format!(
            "the weight side took only {}% of the {} MiB the KV side released \
             (threshold {}%): the frontier fell by {} regions and `weights` grew by \
             {} MiB — freed ground that nothing claims buys no decode",
            uptake_pct, mib_released, args.min_weight_uptake, regions_released, weight_growth_mib,
        ));
    } else {
        println!(
            "PASS  the weight side took {}% of the {} MiB released.",
            uptake_pct, mib_released,
        );
    }

    if !failures.is_empty() {
        for f in &failures {
            println!("\nFAIL  {f}");
        }
        anyhow::bail!(
            "{} of 3 checks failed (story correctness, VRAM efficiency, weight \
             uptake). The two VRAM failures are expected until continuous \
             compaction AND the weight side's uptake both land — those ARE the \
             specification for them. A STORY failure is not expected and is not a \
             specification: it means answers are wrong.",
            failures.len(),
        );
    }
    Ok(())
}
