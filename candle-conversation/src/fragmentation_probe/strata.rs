//! Strata's single-session benchmark, delivered through the real engine.
//!
//! `candle_transformers::models::batch_test::strata_bench` runs Strata's published
//! requests on the bare forward path — `forward_wave` from the batched harness, BF16 KV —
//! which measures the forward's ceiling. This runs the **same requests** through a
//! [`ConversationEngine`]: admission, the turn assembler, per-turn projection, the
//! persistence thread, and the probe's KV compression level — which the engine applies
//! to a turn once it is finished, not as it is written — as a daemon runs them. The two
//! tables side by side are what the engine costs on Strata's work.
//!
//! The requests are built by the forward bench's own functions — the synthetic module,
//! the nonce line, the binary search to the token — and cut to length by the forward
//! harness's own counter, so the request text is byte for byte the forward bench's. The
//! engine renders its own turn around it; the tokens it actually prefilled are reported
//! beside the target and the prefill rate is taken over them.
//!
//! Each request is a fresh conversation with an empty system prompt, submitted greedy
//! with reasoning off for [`GENERATE`] tokens, one at a time: the short warm-up first,
//! excluded, then three runs at each length in increasing order on the same engine.
//! After each turn its timeline is evicted and the harness waits, off the clocks, for
//! it to leave the device and for the engine to settle — the idle time a daemon has
//! between one user's requests — so no run inherits another's KV or its housekeeping.
//!
//! **The clocks are the stream's.** Prefill runs from the submit to the first token —
//! so admission and assembly are in it, as they are in what a client waits for — and
//! decode from the first token to the last, over the tokens after the first, which is
//! Strata's `generated / decode time`.

use std::time::{Duration, Instant};

use candle::Device;
use candle_nn::kv_cache::spare_tally;
use candle_transformers::models::batch_test::strata_bench::{
    filler, fit_prompt, median_of, GENERATE, RUNS, STRATA_RTX_5090, TARGETS, WARMUP,
};
use candle_transformers::models::batch_test::utils::TestParams;
use candle_transformers::models::dialect::Dialect;
use candle_transformers::models::expert_lre::grow_tally;

use super::batch::{clock_turn, greedy_sampling, no_think, ClockedTurn};
use super::probe::Probe;
use super::run::start_engine;
use crate::{SamplingConfig, SequenceConfig, TurnOptions};

/// Strata's sampling: argmax over the model's raw logits.
///
/// The forward bench picks every token as the argmax of the logits the forward
/// produced, and Strata's requests are scored the same way. The conversation's
/// own sampling, made greedy ([`greedy_sampling`]), still carries the model's
/// shipped logit levers — Flash-Next's row has `repeat_penalty 1.1` — and a
/// penalised argmax is a different reply: it parts from the forward bench's
/// within a sentence, and the MTP drafter, which predicts the raw argmax, has
/// its drafts rejected wherever the penalty moved the choice. So every lever
/// that reshapes a logit is switched off here; what stays from the
/// conversation's config only bounds or frames the turn.
fn strata_sampling(config: SequenceConfig) -> SamplingConfig {
    SamplingConfig {
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        dry: None,
        cross_turn_penalty: 0.0,
        eos_boost: 0.0,
        dynamic_eos_boost: false,
        segment_suppress_penalty: 0.0,
        segment_close_boost: 0.0,
        ..greedy_sampling(config)
    }
}

/// One request's delivery.
#[derive(Clone, Debug, PartialEq)]
pub struct StrataRun {
    /// `tokens-<target>-run-<n>`, or `warmup`.
    pub label: String,
    /// The length the request was cut to, under the forward harness's counter.
    pub fitted_tokens: usize,
    /// What the engine actually prefilled for the turn.
    pub prefill_tokens: usize,
    /// Submit to first token.
    pub prefill_s: f64,
    /// Tokens after the first.
    pub decode_tokens: usize,
    /// First token to last.
    pub decode_s: f64,
    /// Decode steps that delivered the tokens after the first — the bursts the
    /// stream arrived in ([`decode_steps`]).
    pub decode_steps: usize,
    /// The widest wait between two arrivals after the first token, and how many
    /// tokens had arrived before it ([`longest_gap`]).
    pub longest_gap: (Duration, usize),
}

/// The widest wait between two consecutive arrivals from the first token on, and
/// how many tokens had arrived before it — a decode window's one stall, which an
/// average over the window spreads across every step.
fn longest_gap(token_times: &[Duration]) -> (Duration, usize) {
    token_times
        .windows(2)
        .enumerate()
        .map(|(i, w)| (w[1].saturating_sub(w[0]), i + 1))
        .fold(
            (Duration::ZERO, 0),
            |best, gap| {
                if gap.0 > best.0 {
                    gap
                } else {
                    best
                }
            },
        )
}

/// Arrivals closer together than this belong to one decode step: a verify step's
/// accepted tokens are streamed in one loop, microseconds apart, while two steps
/// are a forward apart.
const SAME_STEP: Duration = Duration::from_millis(1);

/// The decode steps that delivered the tokens after the first, read off their
/// arrivals: a step is a burst of tokens closer together than [`SAME_STEP`].
/// What the forward bench reports as `speculative: N steps`, so the two can be
/// compared step for step — tokens per step is the acceptance, decode time per
/// step the engine's step cost.
fn decode_steps(token_times: &[Duration]) -> usize {
    let decoded = token_times.get(1..).unwrap_or_default();
    match decoded.first() {
        None => 0,
        Some(_) => {
            1 + decoded
                .windows(2)
                .filter(|w| w[1].saturating_sub(w[0]) >= SAME_STEP)
                .count()
        }
    }
}

/// What the elastic boundary did over one request: how the pool answered the
/// weight side's growth negotiations, and what the expert zone took.
///
/// Both counters are process-wide since boot, so a request's share is the
/// difference of two readings ([`Self::since`]). Read beside the decode rate: a
/// prefill concedes expert slots to its transient tier, and a decode that runs
/// before the zone has grown back reads the experts it lost over the link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryTally {
    /// The pool's growth answers (`candle_nn::kv_cache::spare_tally`).
    spare: [u64; 7],
    /// The expert zone's growth outcomes (`expert_lre::grow_tally`).
    grow: [u64; 8],
}

impl BoundaryTally {
    /// The counters as they stand.
    pub fn now() -> Self {
        Self {
            spare: spare_tally(),
            grow: grow_tally(),
        }
    }

    /// What moved between `earlier` and this reading.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            spare: std::array::from_fn(|i| self.spare[i].saturating_sub(earlier.spare[i])),
            grow: std::array::from_fn(|i| self.grow[i].saturating_sub(earlier.grow[i])),
        }
    }

    /// One line: the pool's refusals by cause and its grants, then what the
    /// zone was offered and gained.
    pub fn line(&self) -> String {
        let s = &self.spare;
        let g = &self.grow;
        format!(
            "growth refused: pressure {} occupied {} fragmented {} | granted {} regions | \
             zone asked {} offered {} regions, gained {} slots",
            s[1], s[2], s[6], s[3], g[0], g[2], g[7]
        )
    }
}

impl StrataRun {
    /// A request's rates from the tokens it prefilled and its stream's arrivals,
    /// each an offset from the submit.
    pub fn from_stream(
        label: String,
        fitted_tokens: usize,
        prefill_tokens: usize,
        token_times: &[Duration],
    ) -> Self {
        let first = token_times.first().copied().unwrap_or_default();
        let last = token_times.last().copied().unwrap_or_default();
        Self {
            label,
            fitted_tokens,
            prefill_tokens,
            prefill_s: first.as_secs_f64(),
            decode_tokens: token_times.len().saturating_sub(1),
            decode_s: last.saturating_sub(first).as_secs_f64(),
            decode_steps: decode_steps(token_times),
            longest_gap: longest_gap(token_times),
        }
    }

    pub fn prefill_tps(&self) -> f64 {
        self.prefill_tokens as f64 / self.prefill_s.max(1e-9)
    }

    pub fn decode_tps(&self) -> f64 {
        self.decode_tokens as f64 / self.decode_s.max(1e-9)
    }
}

/// Load the probe's model, stand up the engine on a scratch substrate, and deliver
/// Strata's requests through it one at a time. Returns every run, the warm-up first.
///
/// `dialect` is the forward harness's for this model: it is what cuts the requests to
/// the token, so the request text matches the forward bench's.
///
/// `at_boundary` is called after the warm-up with `None` and after each length's last
/// run with `Some(target)`, so a caller can bracket one length's window — a profiled
/// build reads its span accumulator there, snapshot-and-reset, and gets each length's
/// breakdown alone rather than all three blended.
pub fn run_strata(
    probe: &Probe,
    dialect: Dialect,
    mut at_boundary: impl FnMut(Option<usize>),
) -> anyhow::Result<Vec<StrataRun>> {
    let device = Device::new_cuda(probe.device)?;
    let builder = probe.builder();
    let (model_path, tokenizer_path) = builder.resolve_paths_pub()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let tokenizer_json = std::fs::read_to_string(&tokenizer_path)?;

    // The forward harness's counter, set up as the forward bench sets it up: empty
    // system turn, reasoning suppressed.
    let counter = TestParams::new(GENERATE, &tokenizer_json, dialect)?
        .with_system_prompt("")
        .with_suppress_thinking(true);
    let module = filler();
    // Each request carries the boundary it closes: the warm-up closes the unmeasured
    // window, and a length's last run closes that length's.
    let mut requests: Vec<(String, String, usize, Option<Option<usize>>)> =
        vec![("warmup".into(), WARMUP.into(), 0, Some(None))];
    for target in TARGETS {
        for run in 1..=RUNS {
            let (prompt, fitted) =
                fit_prompt(&module, target, run, |p| counter.prefill_token_count(p))?;
            let closes = (run == RUNS).then_some(Some(target));
            requests.push((format!("tokens-{target}-run-{run}"), prompt, fitted, closes));
        }
    }

    println!("Loading {model_path:?} …");
    let model = builder.load_model(&model_path, &device, None)?.model;
    // Held for the engine's whole life — see `start_engine`.
    let (_scratch, engine) = start_engine(probe, &device, &tokenizer, model)?;
    let config = builder.conversation_config();

    println!(
        "\n=== Strata single-session benchmark, through the engine (C{}) ===\n",
        probe.compression_level
    );
    let mut runs = Vec::with_capacity(requests.len());
    for (label, prompt, fitted, closes) in requests {
        let mut conv = engine.new_conversation("", config.clone())?;
        let opts = TurnOptions {
            max_tokens: Some(GENERATE),
            selection: no_think(),
            sampling: Some(strata_sampling(config.clone())),
            ..Default::default()
        };
        let tally_before = BoundaryTally::now();
        let t0 = Instant::now();
        let handle = conv.submit_turn_with_options(&prompt, opts)?;
        let ClockedTurn {
            prefilled,
            resp,
            token_times,
            ..
        } = clock_turn(&handle, t0).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
        let run =
            StrataRun::from_stream(label, fitted, resp.stats.turn_prefill_tokens, &token_times);
        println!(
            "  {:<22} fitted {:>6}  prefilled {:>6}  prefill {:>8.1} t/s  decode {:>6.1} t/s  \
             ({} tokens, {} steps, {:.2} accepted/step, {:.1} ms/step, longest gap {:.1} ms \
             after token {})",
            run.label,
            run.fitted_tokens,
            run.prefill_tokens,
            run.prefill_tps(),
            run.decode_tps(),
            token_times.len(),
            run.decode_steps,
            run.decode_tokens as f64 / run.decode_steps.max(1) as f64,
            run.decode_s * 1e3 / run.decode_steps.max(1) as f64,
            run.longest_gap.0.as_secs_f64() * 1e3,
            run.longest_gap.1,
        );
        // How the engine framed the request, once, and how every reply opens — what
        // to read beside the forward bench's outputs when the decode rates differ.
        if runs.len() <= 1 {
            println!(
                "    framed as {:?} … {:?}",
                head(&prefilled, 64),
                tail(&prefilled, 96)
            );
        }
        println!("    reply opens {:?}", head(&resp.text, 96));
        println!(
            "    boundary: {}",
            BoundaryTally::now().since(&tally_before).line()
        );
        conv.finish_turn(handle, &resp)?;
        let timeline = conv.timeline_id();
        let _ = engine.evict_ingest_timeline(timeline);
        drop(conv);
        // **The idle time between requests.** A daemon serving one user has the gap
        // between a reply and the next request to migrate the finished turn off the
        // device and grow the weight side back; a harness that submits the next
        // request on the same instant measures that housekeeping inside the next
        // request's own forwards instead. So the bench waits here — out of every
        // clock — until the turn has left the device and the engine has settled.
        let settled = engine.settle_evicted_timeline(timeline, SETTLE_TIMEOUT)?;
        println!(
            "    settled: {} regions live, frontier {}, weight zone {} MiB",
            settled.live_regions,
            settled.frontier,
            settled.weight_bytes >> 20
        );
        runs.push(run);
        if let Some(boundary) = closes {
            at_boundary(boundary);
        }
    }
    report(&runs);
    Ok(runs)
}

/// How long a finished turn may take to leave the device before the harness calls
/// it a fault. A 128K turn is ~1.2 GB of quantized KV through a 64 MiB staging span;
/// this is minutes beyond that.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(120);

/// The first `n` characters of `s`.
fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The last `n` characters of `s`.
fn tail(s: &str, n: usize) -> String {
    let skip = s.chars().count().saturating_sub(n);
    s.chars().skip(skip).collect()
}

/// The forward bench's summary, over the engine's runs: per length, the median
/// prefill and decode with every run beside it, and Strata's published medians.
fn report(runs: &[StrataRun]) {
    println!(
        "\n=== Strata benchmark through the engine: median of {RUNS} runs (each run), \
         warm-up excluded ===\n"
    );
    println!("| prompt tokens | prefill t/s | decode t/s | Strata RTX 5090 prefill / decode |");
    println!("|---:|---:|---:|---:|");
    for (i, target) in TARGETS.iter().enumerate() {
        let measured = &runs[1 + i * RUNS..1 + (i + 1) * RUNS];
        let prefill: Vec<f64> = measured.iter().map(StrataRun::prefill_tps).collect();
        let decode: Vec<f64> = measured.iter().map(StrataRun::decode_tps).collect();
        let (sp, sd) = STRATA_RTX_5090[i];
        println!(
            "| {target} | {} | {} | {sp:.1} / {sd:.1} |",
            median_of(&prefill),
            median_of(&decode),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;

    fn ms(v: &[u64]) -> Vec<Duration> {
        v.iter().map(|&t| Duration::from_millis(t)).collect()
    }

    /// Prefill is submit to first token over what the engine prefilled; decode is the
    /// tokens after the first over first to last — Strata's `generated / decode time`.
    #[test]
    fn a_run_reads_its_rates_off_the_stream() {
        let r = StrataRun::from_stream(
            "tokens-4096-run-1".into(),
            4090,
            4100,
            &ms(&[800, 820, 840, 900]),
        );
        assert_eq!(
            r,
            StrataRun {
                label: "tokens-4096-run-1".into(),
                fitted_tokens: 4090,
                prefill_tokens: 4100,
                prefill_s: 0.8,
                decode_tokens: 3,
                decode_s: 0.1,
                decode_steps: 3,
                longest_gap: (Duration::from_millis(60), 3),
            }
        );
        assert_eq!(r.prefill_tps(), 4100.0 / 0.8);
        assert_eq!(r.decode_tps(), 3.0 / 0.1);
    }

    /// The framing excerpts count characters, not bytes, so a multi-byte character
    /// at either cut is kept whole.
    #[test]
    fn excerpts_cut_on_characters() {
        assert_eq!(head("<|im_start|>user\n", 5), "<|im_");
        assert_eq!(
            tail("assistant\n<think>\n\n</think>\n\n", 12),
            "\n\n</think>\n\n"
        );
        assert_eq!(head("→ok", 2), "→o");
        assert_eq!(tail("ok→", 1), "→");
        assert_eq!(tail("ab", 9), "ab");
    }

    /// Every logit lever the model ships is off, the greedy pick stays, and what
    /// only bounds the turn is the conversation's own. Flash-Next's own row carries
    /// the repeat penalty; the rest are layered on so each lever is seen cleared.
    #[test]
    fn strata_sampling_is_the_raw_argmax() {
        let mut config = Model::Qwen38_FlashNext_Q4KO.conversation_config();
        assert_eq!(config.sampling.repeat_penalty, 1.1, "the shipped row");
        config.sampling = config
            .sampling
            .with_presence_penalty(1.5)
            .with_frequency_penalty(0.4)
            .with_cross_turn_penalty(2.0)
            .with_dry_penalty(0.8, 1.75, 2, 512)
            .with_dynamic_eos_boost(1.0, 400, 500, 3.0)
            .with_eos_failsafe(512, 700);
        let s = strata_sampling(config);
        assert_eq!((s.temperature, s.top_k, s.top_p), (0.0, 1, 1.0), "greedy");
        assert_eq!(
            (
                s.repeat_penalty,
                s.frequency_penalty,
                s.presence_penalty,
                s.cross_turn_penalty
            ),
            (1.0, 0.0, 0.0, 0.0)
        );
        assert!(s.dry.is_none());
        assert_eq!((s.eos_boost, s.dynamic_eos_boost), (0.0, false));
        assert_eq!(
            (s.segment_suppress_penalty, s.segment_close_boost),
            (0.0, 0.0)
        );
        assert_eq!((s.graceful_eos_after, s.forced_eos_after), (512, 700));
    }

    /// A request's share of the boundary counters is the difference of two
    /// readings, named by cause.
    #[test]
    fn a_boundary_tally_reads_one_requests_share() {
        let earlier = BoundaryTally {
            spare: [1, 10, 2, 300, 0, 0, 1],
            grow: [5, 1, 120, 0, 0, 0, 0, 900],
        };
        let later = BoundaryTally {
            spare: [1, 17, 2, 340, 4, 0, 3],
            grow: [9, 1, 160, 0, 0, 0, 0, 1000],
        };
        let d = later.since(&earlier);
        assert_eq!(
            d,
            BoundaryTally {
                spare: [0, 7, 0, 40, 4, 0, 2],
                grow: [4, 0, 40, 0, 0, 0, 0, 100],
            }
        );
        assert_eq!(
            d.line(),
            "growth refused: pressure 7 occupied 0 fragmented 2 | granted 40 regions | \
             zone asked 4 offered 40 regions, gained 100 slots"
        );
    }

    /// A turn that streamed one token decoded nothing, and one that streamed none
    /// has no prefill window to divide by — zero, not a division by zero.
    #[test]
    fn a_short_stream_decodes_nothing() {
        let one = StrataRun::from_stream("x".into(), 0, 10, &ms(&[250]));
        assert_eq!((one.decode_tokens, one.decode_s), (0, 0.0));
        assert_eq!(one.decode_tps(), 0.0);
        let none = StrataRun::from_stream("x".into(), 0, 10, &[]);
        assert_eq!((none.prefill_s, none.decode_tokens), (0.0, 0));
        assert_eq!((one.decode_steps, none.decode_steps), (0, 0));
    }

    /// A step is a burst: tokens arriving within a millisecond of each other came
    /// out of one verify step, and the first token — the prefill's — is no step.
    #[test]
    fn decode_steps_count_the_bursts_after_the_first_token() {
        let us =
            |v: &[u64]| -> Vec<Duration> { v.iter().map(|&t| Duration::from_micros(t)).collect() };
        // First token at 800 ms; then a burst of three, one alone, a burst of two.
        let times = us(&[
            800_000, 820_000, 820_010, 820_020, 838_000, 856_000, 856_005,
        ]);
        assert_eq!(decode_steps(&times), 3);
        // Every decoded token its own step.
        assert_eq!(decode_steps(&us(&[0, 15_000, 30_000, 45_000])), 3);
        assert_eq!(decode_steps(&us(&[5])), 0);
    }

    /// The stall is the widest wait between arrivals, named by the tokens before
    /// it; the first of two equal waits is the one reported, and a stream of one
    /// token has none.
    #[test]
    fn the_longest_gap_names_the_stall_and_where_it_fell() {
        assert_eq!(
            longest_gap(&ms(&[800, 820, 1_900, 1_915, 1_930])),
            (Duration::from_millis(1_080), 2)
        );
        assert_eq!(
            longest_gap(&ms(&[0, 10, 20])),
            (Duration::from_millis(10), 1)
        );
        assert_eq!(longest_gap(&ms(&[250])), (Duration::ZERO, 0));
    }
}
