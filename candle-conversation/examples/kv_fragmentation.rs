//! Drive the real engine into KV fragmentation and report what it costs.
//!
//! A thin driver over [`candle_conversation::fragmentation_probe`], which is where the
//! run, the gates and the per-model profiles live. This file is flags and an exit code;
//! the same function backs the integration tests, so a command line and a test case
//! measure the same thing.
//!
//! ```text
//! cargo run --release --features hub --example kv_fragmentation -- --model qwen3-30b-a3b-q4
//! ```
//!
//! Exits non-zero when any gate fails, so it is usable from a script without parsing the
//! output.

use candle_conversation::fragmentation_probe::{names, profile, run, Probe};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Fragment the KV pool with overlapping conversations, then measure the cost")]
struct Args {
    /// Which model profile to run. The profile carries the widths and the thresholds,
    /// because both are properties of the model's KV geometry.
    #[arg(long, default_value = "qwen3-30b-a3b-q4")]
    model: String,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Delay between starting new conversations, in milliseconds. Lower is more overlap;
    /// the saturation loop lowers it as well as raising concurrency.
    #[arg(long, default_value_t = 120)]
    stagger_ms: u64,

    /// Raise concurrency and tighten the stagger until the pool saturates.
    #[arg(long, default_value_t = false)]
    saturate: bool,

    /// Free regions at or below which the pool counts as saturated.
    #[arg(long, default_value_t = 24)]
    saturated_free: usize,

    /// Seconds to hold the overlapping churn.
    #[arg(long, default_value_t = 90)]
    churn_secs: u64,

    /// One retirement in this many is a straggler. See the probe's own docs — this is
    /// what keeps the high end of the span alive.
    #[arg(long, default_value_t = 4)]
    straggler_every: usize,

    /// How long a straggler holds its KV before being evicted.
    #[arg(long, default_value_t = 20)]
    straggler_hold_secs: u64,

    /// Seconds to drain after everything is evicted, watching the frontier fall and the
    /// weight zone grow.
    #[arg(long, default_value_t = 40)]
    drain_secs: u64,

    /// Tokens each phase-B sequence decodes.
    #[arg(long, default_value_t = 48)]
    batch_decode: usize,

    /// Conversations alive at once through the churn. Defaults to the profile's.
    #[arg(long)]
    concurrency: Option<usize>,

    /// Concurrent sequences in the phase-B comparison batch. Defaults to the profile's.
    #[arg(long)]
    batch: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // INFO, not WARN: the compaction pass and the weight side's reclaim both report at
    // INFO, and a run that cannot see whether compaction fired cannot tell a working
    // pass from one that never ran. `RUST_LOG` still overrides.
    // Through `RUST_LOG`, defaulting to INFO, rather than a fixed level. A fixed
    // `with_max_level` makes one class of question unreachable without editing this file:
    // the admission pass and the relief ladder both report their verdicts at debug, so a
    // run that shows a width-1 wave cannot also show *why* it was width 1.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init()
        .ok();

    let Some(mut model_profile) = profile(&args.model) else {
        anyhow::bail!(
            "no model profile named {:?}. Known profiles: {}. A model the probe has not \
             measured needs a row in `fragmentation_probe::profile`, whose thresholds are \
             measured against that model rather than copied from another.",
            args.model,
            names().join(", "),
        );
    };
    // Overriding a width is for tuning a run, not for changing what the gate demands —
    // the thresholds stay the profile's, so a narrowed run that passes still passes the
    // same bar.
    if let Some(c) = args.concurrency {
        model_profile.concurrency = c;
        model_profile.max_concurrency = model_profile.max_concurrency.max(c);
    }
    if let Some(b) = args.batch {
        model_profile.batch = b;
    }

    let probe = Probe {
        device: args.device,
        stagger_ms: args.stagger_ms,
        saturate: args.saturate,
        saturated_free: args.saturated_free,
        churn_secs: args.churn_secs,
        straggler_every: args.straggler_every,
        straggler_hold_secs: args.straggler_hold_secs,
        drain_secs: args.drain_secs,
        batch_decode: args.batch_decode,
        ..Probe::new(model_profile)
    };

    let outcome = run(&probe)?;
    if !outcome.passed() {
        anyhow::bail!(
            "{} of 3 checks failed (story correctness, VRAM efficiency, weight uptake). \
             A STORY failure means answers are wrong and is never a tuning matter; the \
             two VRAM failures name the ground fragmentation is denying the weight side.",
            outcome.failures.len(),
        );
    }
    Ok(())
}
