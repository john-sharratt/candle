//! The long-context scalability gate: the same batched forward every model's
//! ladder runs, at a KV depth of tens to hundreds of thousands of tokens.
//!
//! The per-model ladders answer "what does this model cost per token at a
//! working depth". This answers a different question — **what happens to that
//! cost as the KV cache grows** — which is the property the engine exists to
//! defend, and the one a table at 700 tokens cannot show.
//!
//! # The filler, and why it is not `ruler_gen::make_filler`
//!
//! A depth test needs a prompt of a given token length, and `ruler_gen` already
//! has one — but it **tiles a single corpus**, so a 128K prompt is that corpus
//! repeated some forty times. That is fine for RULER, whose score is retrieval
//! accuracy and whose filler is meant to be ignorable. It is not fine here,
//! because two of the numbers this gate reports are the KV **compression
//! ratio** and the throughput that follows from it, and a cache holding forty
//! copies of one passage compresses like nothing real ever will.
//!
//! So the filler here is assembled instead: paragraphs drawn from several
//! unrelated domains, ordered by a deterministic permutation, and *perturbed
//! per cycle* — the numbers, names and dates in a paragraph differ every time
//! it comes round, so no two cycles present the same token sequence. It is
//! still synthetic, and the report says so; what it is not is degenerate.
//!
//! Everything is deterministic given the seed, so a run reproduces exactly.

use candle::{Device, Result};
use tokenizers::Tokenizer;

use super::utils::{account_model_load, story_prompt, TestConfig, TestMode, TestParams};
use crate::models::batched_inference::{InferenceMode, ManagedBatchedModel};
use crate::models::dialect::Dialect;
use candle::quantized::Int8Mode;

/// Paragraph stock for the filler, one per domain, each carrying `{}` slots the
/// per-cycle perturbation fills. Unrelated subjects on purpose: adjacent
/// paragraphs should share as little vocabulary as possible, because a cache
/// whose neighbouring blocks are near-duplicates is the case that flatters
/// every compression format.
const PARAGRAPHS: &[&str] = &[
    "The survey vessel logged {N} separate soundings along the shelf edge that season, \
     each recorded against a datum the crew re-established from the tide gauge at {PLACE}. \
     Depths ran from {N2} to {N3} metres, and the bottom returned a hard echo everywhere \
     except a band of soft sediment some {N4} kilometres wide. {NAME} argued the band marked \
     an older channel, since buried; the alternative reading, that it was slump material off \
     the bank above, was never fully ruled out.",
    "Fermentation in the {PLACE} tradition begins with a mash held near {N} degrees for \
     roughly {N2} hours, long enough for the enzymes to convert the starch without letting \
     the wild yeasts take hold. {NAME} records that the older houses judged readiness by the \
     sound the paddle made rather than by any measurement, and that the change is abrupt: \
     one stroke the mash drags, the next it parts cleanly and the batch is moved.",
    "The statute was amended {N} times between its passage and the {N2}s, and the amendments \
     pull in opposite directions. Three of them narrow the class of persons who may bring an \
     action; two widen the remedies available once an action is brought. Commentators of the \
     {PLACE} school treat the result as incoherent, while {NAME} has argued that the pattern \
     is deliberate — that the legislature meant to make the right harder to invoke and more \
     valuable when invoked.",
    "Larval development in the species proceeds through {N} instars, of which the {N2}th is \
     the longest and the only one spent away from the host plant. Field counts near {PLACE} \
     put mortality across that instar at better than {N3} per cent, most of it attributable \
     to a single parasitoid. {NAME} notes that populations at the northern edge of the range \
     skip the migration entirely, which would make the instar count plastic rather than fixed.",
    "The bridge as built differs from the drawings in three respects, all of them at the \
     {PLACE} abutment. The bearing seats sit {N} millimetres lower than specified, the \
     drainage was rerouted around a services duct that appears on no drawing, and the \
     expansion joint is of a different make from the one tendered. {NAME}'s inspection put \
     the remaining life at {N2} years, contingent on the joint being replaced within {N3}.",
    "Of the {N} manuscripts that preserve the poem, only {N2} are independent of the \
     {PLACE} copy, and those two disagree with it in ways that cannot be scribal. Both omit \
     the passage about the crossing; both give the king's name in a form the {PLACE} copy \
     never uses. {NAME} takes the omission as evidence of an earlier and shorter recension, \
     against the older view that the passage was cut for length.",
    "Yield across the {N} trial plots varied by a factor of {N2}, which is more than the \
     treatments can account for. Soil cores taken afterwards showed the eastern plots sit on \
     a lens of coarse material that drains far faster than the rest of the field. {NAME} \
     recommends the trial be repeated on ground surveyed in advance, and that the {PLACE} \
     results be set aside rather than corrected.",
    "The engine was rated at {N} kilowatts continuous, but the duty it actually saw was \
     intermittent and much harder: {N2} starts a day against a load that had not been \
     allowed to run down. Bearing temperatures at {PLACE} ran {N3} degrees above the figures \
     in the manual from the first week. {NAME} put the eventual failure down to the duty \
     cycle rather than to any defect in the machine.",
];

/// Names substituted into `{NAME}`, cycled so a paragraph rarely repeats one.
const NAMES: &[&str] = &[
    "Halvorsen",
    "Okonkwo",
    "Bergström",
    "Nakamura",
    "Duarte",
    "Whitfield",
    "Ravenna",
    "Ó Cuinn",
    "Marchetti",
    "Szabó",
    "Adeyemi",
    "Lindqvist",
];

/// Places substituted into `{PLACE}`.
const PLACES: &[&str] = &[
    "Kirkwall",
    "Tampere",
    "Valparaíso",
    "Broome",
    "Trondheim",
    "Galway",
    "Otaru",
    "Ceuta",
    "Dunedin",
    "Arica",
    "Nuuk",
    "Hobart",
];

/// A tiny deterministic LCG — reproducibility matters more than statistical
/// quality, and the only thing being randomised is which paragraph comes next
/// and what numbers go in it.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next() as usize) % xs.len()]
    }

    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo).max(1)
    }
}

/// One paragraph with its slots filled, for cycle `cycle` of the filler.
fn realise(template: &str, rng: &mut Lcg) -> String {
    let name = *rng.pick(NAMES);
    let place = *rng.pick(PLACES);
    let (n, n2, n3, n4) = (
        rng.range(3, 900),
        rng.range(2, 400),
        rng.range(5, 95),
        rng.range(1, 40),
    );
    template
        .replace("{NAME}", name)
        .replace("{PLACE}", place)
        .replace("{N2}", &n2.to_string())
        .replace("{N3}", &n3.to_string())
        .replace("{N4}", &n4.to_string())
        .replace("{N}", &n.to_string())
}

/// Prose of approximately `target_tokens` tokens under `tokenizer`.
///
/// Slightly **under** the target rather than over: the caller appends an
/// instruction and the chat template adds its own tokens, and a prompt that
/// overshoots the depth being measured makes the row mean something other than
/// its label.
pub fn padding_prose(tokenizer: &Tokenizer, target_tokens: usize) -> String {
    let mut rng = Lcg(0x5eed_1234_abcd_0001);
    let mut out = String::with_capacity(target_tokens * 4);
    let mut tokens = 0usize;
    // Leave 2% headroom for the instruction and template.
    let budget = target_tokens.saturating_sub(target_tokens / 50);
    let mut i = 0usize;
    while tokens < budget {
        // Walk the stock in a rotating order so consecutive cycles do not
        // present paragraphs in the same sequence.
        let p = PARAGRAPHS[(i + (i / PARAGRAPHS.len())) % PARAGRAPHS.len()];
        let para = realise(p, &mut rng);
        let n = tokenizer
            .encode(para.as_str(), false)
            .map(|e| e.len())
            .unwrap_or(0);
        if tokens + n > budget {
            break;
        }
        out.push_str(&para);
        out.push_str("\n\n");
        tokens += n + 1;
        i += 1;
    }
    out
}

/// One depth's worth of prompt: the filler, then an instruction that names the
/// session so the harness's per-session validation still has something to check.
/// The task a depth row asks at the end of its padding.
///
/// A depth row is `padding + task`, and WHICH task decides what the row's decode
/// rate means — the two differ by more than the validation they run.
/// `Coherence` asks for a couple of sentences about material the model has just
/// read, where the draft head is guessing; `Rewrite` puts the story at the tail
/// and asks for it back with a character renamed, where the drafter is largely
/// copying tokens it can see and acceptance roughly doubles.
///
/// Both are honest numbers and neither is comparable to the other. Naming the
/// task at the call site is what stops a `Coherence` depth row being read
/// against a `Rewrite` width row and the difference being taken for a
/// regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthTask {
    /// Answer about the padding itself. Validated as [`TestMode::CoherenceCheck`].
    Coherence,
    /// Rewrite the story that follows the padding, renaming its character.
    /// Validated as [`TestMode::StoryRewrite`].
    Rewrite,
}

impl DepthTask {
    fn test_mode(self) -> TestMode {
        match self {
            Self::Coherence => TestMode::CoherenceCheck,
            Self::Rewrite => TestMode::StoryRewrite,
        }
    }
}

fn depth_prompt(tokenizer: &Tokenizer, depth: usize, task: DepthTask) -> String {
    let padding = padding_prose(tokenizer, depth);
    match task {
        DepthTask::Coherence => format!(
            "{padding}\n\nYou have just read a long set of unrelated field notes. \
             Greet {{INSERT_NAME}} by name in one short sentence, then state in one \
             further sentence what kinds of documents those notes were.",
        ),
        // The story goes AFTER the padding, so the tokens the rewrite must
        // reproduce are the most recent ones and the task is the same one the
        // width ladder runs — only the distance back to the start has changed.
        // That is the whole point of the row: hold the task fixed and vary
        // depth, so the two tables differ in one axis rather than three.
        DepthTask::Rewrite => format!("{padding}\n\n{}", story_prompt()),
    }
}

/// Run `depths` against one loaded model, one depth at a time.
///
/// **One `run_loaded` per depth, carrying every mode as a config of that call.**
/// Two constraints meet here and this is the only shape that satisfies both.
///
/// The call must cover all the modes, because a *second* `run_loaded` against an
/// already-loaded model leaks per-sequence engine state on the recurrent
/// architectures: their DeltaNet / PLE / index maps are keyed by slot id and
/// outlive the session, so the next call's fresh sequence 0 inherits the
/// previous one's offset and the reconciler refuses it — measured as "cannot
/// truncate sequence 0 to 62934 tokens — it stands at 32172". Every ladder in
/// the tree passes many configs to one call for this reason; a call per mode
/// does not work and never did.
///
/// The call must *not* also cover all the depths, because the harness prints its
/// comparison table once at the end, so a config that fails hard — an OOM at the
/// deepest rung is the realistic case — takes every shallower row's numbers down
/// with it. A call per depth means a depth that cannot run costs only its own
/// rows. (Validation failures are unaffected either way: the table prints before
/// the error.)
///
/// Depths belong to the caller because they are a property of the checkpoint:
/// a model whose rope tables stop at 32K has nothing to say about 128K, and
/// asking anyway measures the error path. `native_context` is what makes that
/// binding rather than advisory — see the check in [`long_context_gate`].
/// One depth and the compression rungs to measure it at.
///
/// Per depth rather than one list for all of them, because the deep rung costs
/// far more than the shallow one: full attention over 4× the tokens is ~16× the
/// prefill work, measured at over 30 minutes for a single 8B rung. The shallow
/// depth can afford the whole ladder; the deep one is best spent on the two
/// endpoints — uncompressed, and maximum compression — which is what the
/// depth-vs-compression question actually turns on.
pub type DepthPlan<'a> = (usize, &'a [InferenceMode]);

/// The checkpoint's trained context window, in tokens, as its own metadata
/// declares it (`<arch>.context_length` in the GGUF).
///
/// Every rung must fit inside it, prompt **and** generated tokens together. A
/// row past the window is not a deeper measurement of this engine; it is a
/// measurement of whatever the rope tables extrapolate to out there, and it
/// moves with the position-scaling scheme rather than with anything the cache
/// or the kernels do. Reporting one next to an in-window row invites exactly
/// the comparison it cannot support.
///
/// This is a hard error rather than a skipped row on purpose. Every ladder in
/// the tree began as a copy of the same `(32_768, …), (131_072, …)` pair, which
/// is right for a 262K checkpoint and wrong for a 32K one, and nothing about a
/// silently-extrapolating run looks wrong in the output table — the tokens/s
/// and the compression ratio are all perfectly plausible. Failing at the call
/// site is what makes the mismatch visible at the moment it is introduced.
pub type NativeContext = usize;

/// The rungs of `ladder` that do not fit inside `native_context`, described for
/// the error message. Each entry is `(depth label, measured prompt tokens)`.
///
/// The generated tokens count against the window as much as the prompt does —
/// they occupy positions past the end of the prefill, which is precisely where
/// a checkpoint at the edge of its training runs out. A rung whose prompt fits
/// with room for fewer than `generate` more positions still finishes outside
/// the window, so the check is on the sum.
fn rungs_past_window(
    ladder: &[(usize, usize)],
    generate: usize,
    native_context: NativeContext,
) -> Vec<String> {
    ladder
        .iter()
        .filter(|(_, actual)| actual + generate > native_context)
        .map(|(depth, actual)| format!("depth {depth} ({actual} prompt + {generate} new)"))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn long_context_gate<M: ManagedBatchedModel>(
    label: &str,
    int8mode: Int8Mode,
    tokenizer_json: &str,
    dialect: Dialect,
    native_context: NativeContext,
    plan: &[DepthPlan<'_>],
    contexts: usize,
    generate: usize,
    task: DepthTask,
    device: &Device,
    load: impl Fn() -> Result<M>,
) -> Result<()> {
    let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
        .map_err(|e| candle::Error::Msg(format!("tokenizer: {e}")))?;
    println!("\n=== {label}: long-context scalability ===\n");
    println!("  native context window: {native_context} tokens\n");

    // Realise every rung's prompt up front, and measure it, so the window check
    // below runs against the token count that will actually be prefilled rather
    // than against the depth label — `padding_prose` deliberately lands a couple
    // of per cent under its target, and at the top rung that difference decides
    // whether the row fits. Doing it here also costs the tokenizer pass once
    // instead of once per rung, and puts the failure before the model load.
    let mut rungs: Vec<(usize, &[InferenceMode], String, usize)> = Vec::with_capacity(plan.len());
    for &(depth, modes) in plan {
        let prompt = depth_prompt(&tokenizer, depth, task);
        let actual = tokenizer
            .encode(prompt.as_str(), false)
            .map(|e| e.len())
            .unwrap_or(0);
        rungs.push((depth, modes, prompt, actual));
    }
    let measured: Vec<(usize, usize)> = rungs.iter().map(|(d, _, _, a)| (*d, *a)).collect();
    let past_window = rungs_past_window(&measured, generate, native_context);
    if !past_window.is_empty() {
        candle::bail!(
            "{label} was trained to hold {native_context} tokens, but its ladder asks for {}. \
             Past the trained window a row measures rope extrapolation rather than this engine, \
             so it cannot be reported beside an in-window row. Cap the ladder at the \
             checkpoint's own context_length.",
            past_window.join(", ")
        );
    }
    // **The span is sized here, before the weights land.** Without a governor
    // the KV reservation falls back to a small test constant — a measured
    // 3,170,893,824 B with the weight floor at its very top, so a wave needing
    // 872 MB of transient tier is refused on a card with 72 GB free. The
    // qwen4exp engine calls this inside its own loader; the models that reach
    // the generic batched path do not, and a depth gate is exactly where that
    // shows. Every engine tolerates being asked twice.
    crate::models::batched_model::ensure_vram_governor(device);
    let model = account_model_load(device, load)?;

    let mut failures: Vec<String> = Vec::new();
    for (depth, modes, prompt, actual) in &rungs {
        let modes_label: Vec<String> = modes.iter().map(|m| format!("{m:?}")).collect();
        println!(
            "\n--- {label} | depth {depth} ({actual} prompt tokens) | {} | {contexts} context(s) ---\n",
            modes_label.join(", ")
        );
        let params = TestParams::new(generate, tokenizer_json, dialect.clone())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(7200)
            // The validation follows the task, so a `Coherence` row is judged
            // on whether it still decodes sense at this depth and a `Rewrite`
            // row on whether it reproduced the story.
            .with_test_mode(task.test_mode())
            // One prompt per config, all of them this depth's.
            .with_per_config_prompts(vec![prompt.clone(); modes.len()]);
        let cfgs: Vec<TestConfig> = modes
            .iter()
            .map(|&mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: contexts,
                num_repeats: 1,
                test_mode: Some(task.test_mode()),
            })
            .collect();
        if let Err(e) = params.run_loaded(cfgs, &model) {
            println!("  !! {label} depth {depth}: {e}");
            failures.push(format!("{label} depth {depth}: {e}"));
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "long-context gate: {} of the rungs failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The filler must not be a tiled corpus: no two cycles may present the
    /// same paragraph text. This is the whole reason the module exists rather
    /// than calling `ruler_gen::make_filler`, so it is the thing to pin.
    #[test]
    fn cycles_do_not_repeat_verbatim() {
        let mut rng = Lcg(0x5eed_1234_abcd_0001);
        let mut seen = std::collections::HashSet::new();
        // Ten cycles of the whole stock — far more than a 128K prompt needs of
        // any single paragraph.
        for _ in 0..10 {
            for p in PARAGRAPHS {
                let r = realise(p, &mut rng);
                assert!(
                    seen.insert(r.clone()),
                    "a realised paragraph repeated verbatim, which is exactly the \
                     degenerate filler this module exists to avoid:\n{r}"
                );
            }
        }
    }

    /// Every slot is filled — a `{N}`/`{NAME}` left in the prose would be a
    /// literal brace in the prompt and a giveaway that the perturbation missed.
    #[test]
    fn every_slot_is_substituted() {
        let mut rng = Lcg(7);
        for p in PARAGRAPHS {
            let r = realise(p, &mut rng);
            assert!(!r.contains('{'), "unsubstituted slot in:\n{r}");
            assert!(!r.contains('}'), "unsubstituted slot in:\n{r}");
        }
    }

    /// The generator is deterministic: same seed, same prose. A depth row that
    /// cannot be reproduced is not a measurement.
    #[test]
    fn the_same_seed_gives_the_same_prose() {
        let mut a = Lcg(42);
        let mut b = Lcg(42);
        for p in PARAGRAPHS {
            assert_eq!(realise(p, &mut a), realise(p, &mut b));
        }
    }

    /// A ladder that fits reports nothing — the common case, and the one that
    /// must not cost a false failure.
    #[test]
    fn a_ladder_inside_the_window_passes() {
        let ladder = [(8_192, 8_010), (32_768, 32_100)];
        assert!(rungs_past_window(&ladder, 64, 32_768).is_empty());
    }

    /// The 128K rung on a 32K checkpoint: the case every ladder in the tree had,
    /// and the reason this check exists.
    #[test]
    fn a_rung_past_the_window_is_named_with_its_depth() {
        let ladder = [(32_768, 32_100), (131_072, 128_500)];
        let over = rungs_past_window(&ladder, 64, 32_768);
        assert_eq!(over.len(), 1, "only the 128K rung is outside a 32K window");
        assert!(
            over[0].contains("131072"),
            "the message must name the offending depth, got: {}",
            over[0]
        );
    }

    /// The generated tokens count against the window too. A prompt that fits
    /// with 32 positions to spare does not fit once 64 more are decoded — and
    /// this is a real boundary, not a hypothetical one: Llama-3.2-3B's window
    /// is exactly the 131,072 its top rung is labelled with.
    #[test]
    fn a_prompt_that_fits_but_leaves_no_room_to_decode_is_caught() {
        let ladder = [(131_072, 131_040)];
        assert!(
            rungs_past_window(&ladder, 64, 131_072).len() == 1,
            "131,040 prompt + 64 generated exceeds a 131,072 window"
        );
        // The same rung as actually realised — `padding_prose` lands a couple of
        // per cent under the label, which is what buys the room to decode.
        assert!(rungs_past_window(&[(131_072, 128_500)], 64, 131_072).is_empty());
    }

    /// Adjacent paragraphs must not be the same template: a depth row's
    /// compression number is only meaningful if neighbouring KV blocks hold
    /// genuinely different text, and the walk order is what guarantees that.
    #[test]
    fn consecutive_paragraphs_come_from_different_templates() {
        let n = PARAGRAPHS.len();
        for i in 0..(n * 6) {
            let a = (i + (i / n)) % n;
            let b = (i + 1 + ((i + 1) / n)) % n;
            assert_ne!(a, b, "template {a} repeated back to back at step {i}");
        }
    }
}
