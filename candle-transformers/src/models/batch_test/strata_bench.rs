//! Strata's published single-session speed benchmark, run on this engine.
//!
//! Strata (github.com/Niko1221/Strata) publishes Qwen3.8-Flash-Next decode rates
//! from one script, `bench/results/2026-09-30-community-rtx-5090/benchmark.py`
//! (engine 0.1.29, commit `d6708a4`): 179.4 / 175.7 / 165.0 t/s median decode at
//! 4,096 / 32,768 / 128,000 prompt tokens on an RTX 5090. This module rebuilds
//! that script's requests exactly, so the two engines can be compared on the
//! same work rather than on whatever task each happens to measure:
//!
//! - **One user message**, no system message: a nonce line naming the length and
//!   the run, a module of synthetic one-line Python functions, and a request for
//!   a 600-word explanation of it — free text the drafter has to guess.
//! - **Sized to the token**: the function lines are cut by a binary search over
//!   their character length until the whole rendered chat prompt is at most the
//!   target, and the result must land within 20 tokens under it.
//! - **Greedy, reasoning off, 256 tokens**; decode is `generated / decode time`.
//! - **One short warm-up request first**, excluded, then three runs at each length
//!   in increasing order on the same loaded engine, reported as the median.
//!
//! Two differences are this harness's own and are stated in the output. The
//! harness prefills a system turn before every user turn, so the system turn is
//! left empty and its marker tokens are counted toward the target: the prompt
//! is the same length as Strata's, a handful of those tokens are an empty system
//! turn's. The warm-up decodes the same 256 tokens as the measured runs rather
//! than Strata's 16; it is excluded either way.

use candle::{Device, Result};

use super::utils::{account_model_load, TestConfig, TestMode, TestParams, TestResults};
use crate::models::batched_inference::{InferenceMode, ManagedBatchedModel};
use crate::models::dialect::Dialect;
use crate::models::profile::ProfileSnapshot;
use candle::quantized::Int8Mode;

/// Prompt lengths, in tokens of the rendered chat prompt — `--targets`.
pub const TARGETS: [usize; 3] = [4_096, 32_768, 128_000];

/// Runs at each length — `--runs`.
pub const RUNS: usize = 3;

/// Tokens each request generates — `max_tokens`.
pub const GENERATE: usize = 256;

/// The warm-up request, sent once before any measured run and excluded.
pub const WARMUP: &str = "Reply with exactly the word READY.";

/// Lines of synthetic Python the prompts are cut from.
const FILLER_LINES: usize = 12_000;

/// What follows the module in every request.
const ENDING: &str = "\n\nWrite a detailed explanation of the code above. Discuss deterministic \
                      integer transforms, modulo arithmetic, testing, naming, complexity, and \
                      maintainability. Write at least 600 words.";

/// How far under its target a prompt may land.
const TOLERANCE: usize = 20;

/// Strata's published medians on the RTX 5090 (E91), per target: prompt
/// tokens/s and decode tokens/s.
pub const STRATA_RTX_5090: [(f64, f64); 3] = [(4_269.8, 179.4), (5_543.2, 175.7), (5_778.7, 165.0)];

/// The whole synthetic module: one function per line, `FILLER_LINES` of them.
pub fn filler() -> String {
    (0..FILLER_LINES)
        .map(|i| {
            format!(
                "def task_{i:05}(value: int) -> int: return (value * {} + {i}) % 100003",
                (i % 97) + 1
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The request's opening line, distinct per length and run so no two prompts
/// share a prefix.
pub fn prefix(target: usize, run: usize) -> String {
    format!("Benchmark nonce: series-{target}-trial-{run}.\nReview this synthetic Python module:\n")
}

/// The longest prompt `prefix + filler[..n] + ENDING` whose rendered length
/// under `count` is at most `target`, and that length. The search is the
/// script's: over `n`, the character count of the filler, from 0 to all of it.
pub fn fit_prompt(
    filler: &str,
    target: usize,
    run: usize,
    count: impl Fn(&str) -> usize,
) -> Result<(String, usize)> {
    let head = prefix(target, run);
    let build = |n: usize| format!("{head}{}{ENDING}", &filler[..n]);
    let (mut lo, mut hi) = (0usize, filler.len());
    while lo < hi {
        let middle = (lo + hi).div_ceil(2);
        if count(&build(middle)) <= target {
            lo = middle;
        } else {
            hi = middle - 1;
        }
    }
    let prompt = build(lo);
    let actual = count(&prompt);
    if actual + TOLERANCE < target || actual > target {
        candle::bail!("could not construct target {target}: {actual}");
    }
    Ok((prompt, actual))
}

/// The median of a length's runs, as Python's `statistics.median` gives it,
/// and every run in order, e.g. `103.1 (99.4, 103.1, 115.5)`.
fn median_of(xs: &[f64]) -> String {
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    let all: Vec<String> = xs.iter().map(|x| format!("{x:.1}")).collect();
    format!("{median:.1} ({})", all.join(", "))
}

/// Run the benchmark against one model load, one session at a time, at `mode`.
pub fn strata_bench<M: ManagedBatchedModel>(
    label: &str,
    int8mode: Int8Mode,
    tokenizer_json: &str,
    dialect: Dialect,
    mode: InferenceMode,
    device: &Device,
    load: impl Fn() -> Result<M>,
) -> Result<()> {
    let mut params = TestParams::new(GENERATE, tokenizer_json, dialect)
        .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
        .with_test_mode(TestMode::Skip)
        .with_system_prompt("")
        .with_suppress_thinking(true)
        .with_print_outputs(true)
        .with_int8mode(int8mode)
        .with_timeout_secs(7200);

    println!("\n=== {label}: Strata single-session benchmark ===\n");
    let filler = filler();
    let mut prompts = vec![WARMUP.to_string()];
    let mut labels = vec!["warmup".to_string()];
    for target in TARGETS {
        for run in 1..=RUNS {
            let (prompt, actual) =
                fit_prompt(&filler, target, run, |p| params.prefill_token_count(p))?;
            let label = format!("tokens-{target}-run-{run}");
            println!("  {label}: {actual} prompt tokens");
            prompts.push(prompt);
            labels.push(label);
        }
    }
    println!(
        "  (the system turn is empty and counted: {} of each prompt's tokens)\n",
        params.system_prompt_tokens(0).len()
    );
    params = params
        .with_per_config_prompts(prompts.clone())
        .with_config_labels(labels);

    crate::models::batched_model::ensure_vram_governor(device);
    let model = account_model_load(device, load)?;
    let configs: Vec<TestConfig> = prompts
        .iter()
        .map(|_| TestConfig {
            mode,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::Skip),
        })
        .collect();
    let mut results = params.run_loaded_collect(configs, &model)?;
    report(&results);
    params.validate_and_print(&mut results)
}

/// The script's summary: per length, the median prefill and decode with every
/// run beside it, and Strata's published medians.
fn report(results: &[TestResults]) {
    println!("\n=== Strata benchmark: median of {RUNS} runs (each run), warm-up excluded ===\n");
    println!("| prompt tokens | prefill t/s | decode t/s | Strata RTX 5090 prefill / decode |");
    println!("|---:|---:|---:|---:|");
    for (i, target) in TARGETS.iter().enumerate() {
        let runs = &results[1 + i * RUNS..1 + (i + 1) * RUNS];
        let prefill: Vec<f64> = runs.iter().map(|r| r.prompt_tokens_per_sec).collect();
        let decode: Vec<f64> = runs.iter().map(|r| r.generate_tokens_per_sec).collect();
        let (sp, sd) = STRATA_RTX_5090[i];
        println!(
            "| {target} | {} | {} | {sp:.1} / {sd:.1} |",
            median_of(&prefill),
            median_of(&decode),
        );
    }
    for (i, target) in TARGETS.iter().enumerate() {
        let mut merged = ProfileSnapshot::default();
        for r in &results[1 + i * RUNS..1 + (i + 1) * RUNS] {
            merged.merge(&r.pipeline_profile);
        }
        if let Some(table) = step_breakdown(&merged) {
            println!("\n  {target} tokens, per speculative step (`--features profile`):\n{table}");
        }
    }
}

/// The speculative step's host phases from a run's pipeline profile, each as
/// milliseconds per step — `None` when the profile is empty (built without
/// `--features profile`) or recorded no step.
fn step_breakdown(snap: &ProfileSnapshot) -> Option<String> {
    let steps = snap
        .entries
        .iter()
        .find(|(name, _, _)| name == "spec:verify")
        .map(|&(_, _, calls)| calls)
        .filter(|&calls| calls > 0)?;
    let lines: Vec<String> = snap
        .entries
        .iter()
        .filter(|(name, _, _)| name.starts_with("spec:"))
        .map(|(name, ms, _)| format!("    {name:<16} {:>7.3} ms", ms / steps as f64))
        .collect();
    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The module's first and last lines, byte for byte as the script's
    /// f-string renders them.
    #[test]
    fn the_filler_matches_the_scripts_lines() {
        let f = filler();
        let lines: Vec<&str> = f.split('\n').collect();
        assert_eq!(lines.len(), 12_000);
        assert_eq!(
            lines[0],
            "def task_00000(value: int) -> int: return (value * 1 + 0) % 100003"
        );
        assert_eq!(
            lines[97],
            "def task_00097(value: int) -> int: return (value * 1 + 97) % 100003"
        );
        assert_eq!(
            lines[11_999],
            "def task_11999(value: int) -> int: return (value * 69 + 11999) % 100003"
        );
        assert!(!f.ends_with('\n'), "join leaves no trailing newline");
    }

    #[test]
    fn the_prefix_names_the_length_and_the_run() {
        assert_eq!(
            prefix(4_096, 2),
            "Benchmark nonce: series-4096-trial-2.\nReview this synthetic Python module:\n"
        );
    }

    /// With one token per character, the search lands exactly on the target:
    /// the longest filler cut whose prompt fits.
    #[test]
    fn the_search_takes_the_longest_cut_that_fits() {
        let f = filler();
        let (prompt, actual) = fit_prompt(&f, 4_096, 1, |p| p.len()).unwrap();
        assert_eq!(actual, 4_096);
        assert_eq!(prompt.len(), 4_096);
        assert!(prompt.starts_with(&prefix(4_096, 1)));
        assert!(prompt.ends_with(ENDING));
    }

    /// A counter that moves in steps wider than the tolerance cannot land
    /// inside it, and the fit refuses rather than reporting a shorter prompt.
    #[test]
    fn a_target_the_counter_cannot_reach_is_refused() {
        let f = filler();
        // Steps of 128 from 50: the nearest under 4,096 is 4,018, 78 short.
        assert!(fit_prompt(&f, 4_096, 1, |p| (p.len() / 128) * 128 + 50).is_err());
    }

    #[test]
    fn the_median_leads_and_every_run_follows_in_order() {
        assert_eq!(
            median_of(&[99.4, 115.5, 103.1]),
            "103.1 (99.4, 115.5, 103.1)"
        );
        assert_eq!(median_of(&[2.0, 1.0]), "1.5 (2.0, 1.0)");
    }

    /// Each `spec:` phase divided by the step count `spec:verify` records, in
    /// recording order; other spans are left out.
    #[test]
    fn the_step_breakdown_is_per_verify_step() {
        let snap = ProfileSnapshot {
            entries: vec![
                ("spec:draft".into(), 40.0, 20),
                ("wv:sweep".into(), 999.0, 20),
                ("spec:verify".into(), 300.0, 20),
                ("spec:rollback".into(), 5.0, 20),
            ],
        };
        assert_eq!(
            step_breakdown(&snap).unwrap(),
            "    spec:draft         2.000 ms\n    spec:verify       15.000 ms\n    \
             spec:rollback      0.250 ms"
        );
        assert!(step_breakdown(&ProfileSnapshot::default()).is_none());
    }
}
