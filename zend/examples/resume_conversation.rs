//! `resume_conversation` — replay a recorded conversation to one of its turns
//! and decode that turn live, reading the workspace's substrate READ-ONLY.
//!
//! The harness for reproducing where a conversation went wrong. Name the turn
//! whose decode lost its way: every turn before it goes back in through the
//! real turn path (decoded again with its recorded ids forced), then that turn is
//! decoded again — once per seed, and greedily with `--argmax` — and each run is
//! printed beside the recorded reply, with a count of the runs that matched a
//! `--fail-if` pattern. The history is replayed once and every run decodes on a
//! fork of it, so a run costs one decode, not a replay. Before the runs it lists
//! the replayed turns whose sealed ids or index pages differ from the
//! recording, with the text where they part — a replay that passes where the
//! recording failed points at those turns.
//! Nothing is written to disk: the substrate is opened read-only and every
//! replayed turn lives in RAM (see `zend::session::replay`).
//!
//! The model takes the whole card, so stop the zend daemon first:
//! ```sh
//! cargo run -p zend --release --example resume_conversation -- \
//!   --workspace D:\prog\candle --conv cc89befa-04d9-4c32-9e03-1076e691d754 \
//!   --resume 14 --max-tokens 400 --seeds 1,2,3 --argmax \
//!   --demonstration every-turn --fail-if "example\.com"
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context};
use candle_conversation::SealedPages;
use clap::{Parser, ValueEnum};
use regex::Regex;
use tracing_subscriber::EnvFilter;
use zend::api::chat::{apply_tools_dial, dial_selection};
use zend::config::{layer_flag_sets, DaemonConfig};
use zend::log_broadcast::LogBus;
use zend::session::{
    Demonstration, ReplayOutcome, ReplaySampling, ReplaySpec, TurnComparison, ZendSession,
};
use zend::types::ToolMode;

#[derive(Parser)]
#[command(about = "Replay a recorded conversation to a turn and decode that turn live, read-only")]
struct Args {
    /// The workspace whose `.substrate/` holds the conversation. Opened
    /// read-only; nothing is written.
    #[arg(long)]
    workspace: PathBuf,
    /// The conversation's id.
    #[arg(long)]
    conv: String,
    /// The turn to decode, counted from 0; every turn before it is replayed.
    #[arg(long)]
    resume: usize,
    /// The decode limit, in tokens.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Seeds to decode with, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
    seeds: Vec<u64>,
    /// Also decode greedily.
    #[arg(long)]
    argmax: bool,
    /// The thinking-effort dial, 0 (off) to 4.
    #[arg(long, default_value_t = 2)]
    effort: u8,
    /// The answer-length dial, 0 (terse) to 4.
    #[arg(long, default_value_t = 2)]
    verbosity: u8,
    /// The tools dial.
    #[arg(long, value_enum, default_value_t = Tools::Comprehensive)]
    tools: Tools,
    /// Where the tool-call demonstration is projected: `first-step` is the
    /// chat as it runs now; `every-turn` is how conversations recorded before
    /// tool rounds dropped it ran.
    #[arg(long, value_enum, default_value_t = DemonstrationArg::FirstStep)]
    demonstration: DemonstrationArg,
    /// Layers out of service, as the daemon's `--disable-layer`.
    #[arg(long = "disable-layer", default_value = "code_reading")]
    disable_layer: Vec<String>,
    /// Layers in service but not read, as the daemon's `--skip-layer`.
    #[arg(long = "skip-layer", default_value = "repo_map")]
    skip_layer: Vec<String>,
    /// A run whose output matches any of these patterns counts as a failure.
    #[arg(long = "fail-if")]
    fail_if: Vec<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Tools {
    None,
    Restricted,
    Comprehensive,
}

#[derive(Clone, Copy, ValueEnum)]
enum DemonstrationArg {
    EveryTurn,
    FirstStep,
}

fn label(sampling: ReplaySampling) -> String {
    match sampling {
        ReplaySampling::Argmax => "argmax".to_string(),
        ReplaySampling::Seed(seed) => format!("seed {seed}"),
    }
}

/// The patterns `text` matches.
fn matched<'a>(text: &str, patterns: &'a [Regex]) -> Vec<&'a str> {
    patterns
        .iter()
        .filter(|p| p.is_match(text))
        .map(Regex::as_str)
        .collect()
}

fn print_run(run: usize, outcome: &ReplayOutcome, patterns: &[Regex]) {
    let hits = matched(&outcome.text, patterns);
    let verdict = if patterns.is_empty() {
        String::new()
    } else if hits.is_empty() {
        "  PASS".to_string()
    } else {
        format!("  FAIL ({})", hits.join(", "))
    };
    println!(
        "\n===== run {run} · {} · {} tokens · prefill {:.0} ms · {:.1} t/s · {} projections · \
         decoded in {:.1} s{verdict}",
        label(outcome.sampling),
        outcome.tokens,
        outcome.prefill_ms,
        outcome.tokens_per_second,
        outcome.projections,
        outcome.decode_secs,
    );
    println!("{}", outcome.text);
}

fn pages(pages: &SealedPages) -> String {
    match pages {
        SealedPages::Absent => "none".to_string(),
        SealedPages::Malformed(e) => format!("malformed ({e})"),
        SealedPages::Widths(widths) => format!("{widths:?}"),
    }
}

/// Each replayed turn that did not seal the recorded ids under the recorded
/// page widths, with the text where its ids first part.
fn print_turns(turns: &[TurnComparison]) {
    let diverged: Vec<&TurnComparison> = turns.iter().filter(|t| !t.matches()).collect();
    if diverged.is_empty() {
        println!(
            "----- the {} replayed turns seal the recorded ids under the recorded page widths",
            turns.len()
        );
        return;
    }
    println!(
        "----- {} of {} replayed turns differ from the recording",
        diverged.len(),
        turns.len()
    );
    for t in diverged {
        let replayed = t.replayed.as_ref();
        println!(
            "turn {:>2}: {} ids recorded, {} replayed; pages recorded {}, replayed {}",
            t.turn,
            t.recorded.token_ids.len(),
            replayed.map_or(0, |r| r.token_ids.len()),
            pages(&t.recorded.pages),
            replayed.map_or_else(|| "-".to_string(), |r| pages(&r.pages)),
        );
        if let Some(d) = &t.difference {
            println!("  first difference at id {}", d.at);
            println!("  recorded: {:?}", d.recorded);
            println!("  replayed: {:?}", d.replayed);
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();

    let patterns = args
        .fail_if
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("--fail-if {p:?} is not a regex")))
        .collect::<anyhow::Result<Vec<_>>>()?;
    if !args.workspace.join(".substrate").is_dir() {
        bail!("{} holds no .substrate directory", args.workspace.display());
    }
    let tools_mode = match args.tools {
        Tools::None => ToolMode::None,
        Tools::Restricted => ToolMode::Restricted,
        Tools::Comprehensive => ToolMode::Comprehensive,
    };
    let demonstration = match args.demonstration {
        DemonstrationArg::EveryTurn => Demonstration::EveryTurn,
        DemonstrationArg::FirstStep => Demonstration::FirstStepOnly,
    };

    let (disabled_layers, skipped_layers) = layer_flag_sets(&args.disable_layer, &args.skip_layer);
    let config = DaemonConfig {
        workspace: args.workspace.clone(),
        disabled_layers,
        skipped_layers,
        read_only_substrate: true,
        ..Default::default()
    };
    let session = Arc::new(ZendSession::new(config, LogBus::new()));
    session.start_loading();
    session.wait_ready().await;

    let turns = session.recorded_conversation(&args.conv)?;
    let Some(recorded) = turns.get(args.resume) else {
        bail!(
            "{} has {} turns; there is no turn {} to resume at",
            args.conv,
            turns.len(),
            args.resume
        );
    };
    println!(
        "{} · {} turns · resuming at turn {} ({} replayed)",
        args.conv,
        turns.len(),
        args.resume,
        args.resume
    );
    println!("\n===== turn {} user half (first 400 chars)", args.resume);
    println!("{}", recorded.user.chars().take(400).collect::<String>());
    println!("\n===== turn {} as recorded", args.resume);
    println!("{}", recorded.assistant);

    let mut selection = dial_selection(
        Some(args.effort),
        Some(args.verbosity),
        Some(args.effort != 0),
    );
    apply_tools_dial(&mut selection, tools_mode);
    let mut runs: Vec<ReplaySampling> = args
        .seeds
        .iter()
        .map(|&s| ReplaySampling::Seed(s))
        .collect();
    if args.argmax {
        runs.insert(0, ReplaySampling::Argmax);
    }
    let spec = ReplaySpec {
        conv_id: args.conv.clone(),
        resume_turn: args.resume,
        max_tokens: args.max_tokens,
        selection,
        tools_mode,
        demonstration,
        runs,
    };

    let runner = Arc::clone(&session);
    let run_patterns = patterns.clone();
    let report = tokio::task::spawn_blocking(move || {
        runner.replay(
            &spec,
            &mut |turns, secs| {
                println!("\n===== replayed the {} turns in {secs:.1} s", turns.len());
                print_turns(turns);
            },
            &mut |run, outcome| print_run(run, outcome, &run_patterns),
        )
    })
    .await
    .context("the replay thread panicked")??;

    if !patterns.is_empty() {
        let failed = report
            .outcomes
            .iter()
            .filter(|o| !matched(&o.text, &patterns).is_empty())
            .count();
        println!("\nfailed {failed}/{}", report.outcomes.len());
    }
    Ok(())
}
