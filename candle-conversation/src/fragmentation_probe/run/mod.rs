//! Drives the real engine into KV fragmentation by churn, then measures what that
//! costs the decode rate and the weight side.
//!
//! The forward gate cannot produce a fragmented pool, because it clears its rows
//! between configurations, so this runs the engine's own admission and retirement
//! instead. Free regions below the arena frontier are ground the KV side owns but
//! the wave transient tier cannot use, so the holes below it are the damage that
//! the weight side pays for. Efficiency is judged against two floors and on a loss
//! that persists across consecutive publishes, so a single burst of frees is not
//! charged as compaction. Correctness is judged first, since a packed pool that
//! answers wrongly is corruption.
//!
//! - `baseline`: the clean baseline row, judged before any churn.
//! - `engine_setup`: the scratch engine and the probe's phases.
//! - `geometry`: the pool's composition, sampled between forwards.
//! - `report_print`: the timing and tenant tables.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle::Device;
use candle_nn::kv_cache::{region_stats, REGION_BYTES};
use candle_transformers::models::expert_lre::grow_tally;
use candle_transformers::models::profile::pipeline_snapshot_and_reset;
use story_rewrite::fixtures;

use super::batch::{no_think, run_story_batch, BatchTiming};
use super::grow_window::GrowWindow;
use super::probe::{BaselineRow, Probe, ProbeOutcome};
use crate::guest::GuestRegistry;
use crate::memory_report;
use crate::scratch_substrate::ScratchSubstrate;
use crate::{ConversationEngine, Sequence, TurnOptions};

/// Run the probe against a model the **caller** loaded.
///
/// The seam exists so one process can measure both halves of the comparison table off a
/// single load: the harness rows want `&M` for `forward_wave`, these rows want the model
/// *moved into* a `ConversationEngine`, and a 17 GB checkpoint is not worth opening twice
/// — quite apart from the span reservation, which is process-global and does not expect a
/// second weight zone to be carved beside the first.
///
/// The caller passes the model already batched, with the pack it was loaded from.
/// Everything else — the scratch substrate, the engine, the phases — is set up here,
/// because none of it is the caller's business.
pub fn run_on_model(
    probe: &Probe,
    device: &Device,
    pack: &Path,
    tokenizer: tokenizers::Tokenizer,
    model: Box<dyn crate::ManagedBatchedModel + Send>,
) -> anyhow::Result<ProbeOutcome> {
    let args = probe;
    let device = device.clone();
    let builder = probe.builder();
    // Held for the engine's whole life — see `start_engine`.
    let (_scratch, engine) = start_engine(probe, &device, pack, &tokenizer, model)?;

    // The forward gate's own fixture — see `story_slice`.
    let story = fixtures::story_prompt();

    // ── Baseline: the gate's C5 ×8 row, through the engine, on a clean pool ──
    //
    // Before anything else touches the engine's pool, so it is measured on the state
    // the gate's row starts from. Its conversations are evicted and dropped straight
    // after, so phase A begins on the pool it always began on.
    let (baseline, baseline_fail) = clean_baseline(&engine, probe)?;

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
    // merely likely to look odd. Timed with the gate's two windows — see `batch`.
    //
    // Enough of the rewrite to catch a wrong name or wrong text without paying for
    // the whole story; the comparison is prefix-based.
    let batch = run_story_batch(
        &engine,
        &builder,
        args.profile.batch,
        args.batch_decode.max(48),
        args.story_gate(),
    )?;
    errors.lock().unwrap().extend(batch.errors.iter().cloned());
    let timing = batch.timing;
    let story_pass = batch.story_pass;
    let story_fail = batch.story_fail;
    let convs = batch.conversations;
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

    print_timing(&timing, args.profile.batch);
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
    let grow_before_drain = grow_tally();
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
            Ok(h) => match h.wait_cancellable(keepalive.substrate.cancel_token()) {
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
    let grow_since_boot = grow_tally();
    let drain_window = GrowWindow::between(grow_before_drain, grow_since_boot);
    for (label, g) in [("", grow_since_boot), (" in drain", drain_window.counts())] {
        println!(
            "weight grow{label}: asked={} no_spare={} spare_offered={} target_unchanged={} \
             target_backwards={} floor_refused={} at_limit={} slots_gained={}",
            g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
        );
    }

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
    // The clean baseline is judged by the same rule. It ran before any churn, so a
    // failure there is not fragmentation — it is the engine answering wrongly on a
    // pool that never moved, and it outranks everything below as much as phase B does.
    if baseline.story_pass != baseline.width || !baseline_fail.is_empty() {
        for f in &baseline_fail {
            println!("\n  {f}");
        }
        failures.push(format!(
            "the clean C{}×{} baseline: {}/{} sessions rewrote the story correctly, {} \
             failure line(s) above",
            baseline.level,
            baseline.width,
            baseline.story_pass,
            baseline.width,
            baseline_fail.len(),
        ));
    }
    if story_pass == args.profile.batch {
        println!(
            "\nPASS  {}/{} sessions rewrote the story correctly.",
            story_pass, args.profile.batch,
        );
    } else {
        for f in &story_fail {
            println!("\n  {f}");
        }
        failures.push(format!(
            "only {}/{} sessions rewrote the story correctly — a session reading \
             another session's KV does not fault, it answers wrongly, so this is \
             the check that sees it",
            story_pass, args.profile.batch,
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
    // **A weight side at its limit has nothing to take, and that is not a failure of
    // compaction.** `capacity_for_frontier` clamps to the zone's limit — the slots the
    // model has, or the most the span lets the zone hold, whichever is smaller — so
    // past it the zone cannot grow however much ground the KV side hands back, and the
    // engine records that as `at_limit`. Measured here: 1,014 `at_limit` against 1.3
    // million regions offered, with a 19,296 MiB zone holding a 19,296 MiB model.
    //
    // Read from the ledger over the drain — the window the uptake is measured on —
    // rather than inferred, and narrow: it excuses nothing when the zone had room
    // (`target_unchanged`), aimed below itself (`target_backwards`) or was refused
    // by the floor (`floor_refused`), which are the real defects this gate exists to
    // catch. A zone that grew and then met its limit took every byte it had a slot
    // for, and is excused for the rest.
    let at_limit = drain_window.at_limit();
    if at_limit {
        println!(
            "PASS  the weight side reached its limit ({} at-limit answers in the drain, \
             {} MiB taken on the way, {}% of the {} MiB released): the zone holds every \
             slot it may — the whole checkpoint, or the most the span lets it have — so \
             the rest of the released ground has no residency to buy.",
            drain_window.counts()[6],
            weight_growth_mib,
            uptake_pct,
            mib_released,
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
        story_total: args.profile.batch,
        worst_sustained_efficiency: worst_efficiency,
        worst_single_efficiency: worst_reported,
        weight_uptake_pct: uptake_pct,
        weight_at_limit: at_limit,
        prefill_tps: timing.prefill_tps(),
        decode_tps: timing.decode_tps(),
        // The geometry phase B was delivered on, so a table row states rate and
        // fragmentation as one observation rather than two.
        frontier_regions: end.frontier,
        efficiency_pct: end.eff,
        peak_tokens: timing.peak_tokens,
        baseline,
        failures,
    })
}

mod baseline;
mod engine_setup;
mod geometry;
mod report_print;

pub use baseline::*;
pub use engine_setup::*;
use geometry::*;
use report_print::*;
