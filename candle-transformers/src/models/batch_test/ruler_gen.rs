//! RULER long-context benchmark — task generation and in-process evaluation.
//!
//! Implements the four core RULER task types as pure-Rust data generators,
//! plus a generic batched inference runner so the tasks can be executed
//! inside `#[test]` functions that already have a loaded model.
//!
//! # Tasks
//! | Name            | Abbreviated | What it tests                          |
//! |-----------------|-------------|----------------------------------------|
//! | `niah_single_1` | NIAH-S1     | Retrieve one needle from haystack      |
//! | `niah_multikey_2`| NIAH-MK2   | Retrieve two needles from haystack     |
//! | `vt`            | VT          | Trace a 4-link variable chain          |
//! | `cwe`           | CWE         | Identify the 10 most-frequent words    |
//!
//! # Prompt format
//! All prompts target Qwen3 ChatML with thinking suppressed:
//! ```text
//! <|im_start|>system
//! You are a helpful assistant.<|im_end|>
//! <|im_start|>user
//! /no_think
//! {body}
//! Question: {question}
//! Answer with the exact value only.<|im_end|>
//! <|im_start|>assistant
//! <think>
//!
//! </think>
//!
//! ```

use candle::{Device, Result, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokenizers::Tokenizer;

use crate::models::batched_inference::{BatchedConfig, InferenceMode, ManagedBatchedModel};

/// Where to load samples from for [`run_ruler_benchmark`].
pub enum RulerDataSource<'a> {
    /// Generate samples procedurally (deterministic, no external files).
    Generated,
    /// Load from a directory of JSONL files produced by the canonical RULER
    /// data-generation script.  Expected filename pattern:
    /// `<task>_<ctx_len>.jsonl`  e.g. `niah_single_1_4096.jsonl`
    Jsonl(&'a std::path::Path),
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Prefill chunk size — must match KV cache CHUNK_SIZE (32).
const PREFILL_STEP: usize = 32;

/// Qwen3 EOS token IDs.
pub const QWEN3_EOS_IDS: &[u32] = &[151645, 151643]; // <|im_end|>, <|endoftext|>

// ChatML prompt fragments (Qwen3).
const SYS_START: &str = "<|im_start|>system\n";
const SYS_END: &str = "<|im_end|>\n";
const USER_START: &str = "<|im_start|>user\n";
const USER_END: &str = "<|im_end|>\n";
const ASST_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Pool of unusual color/dye names used as "common words" in CWE tasks.
/// These are chosen to be absent from the filler corpus.
const CWE_WORD_POOL: &[&str] = &[
    "crimson",
    "azure",
    "violet",
    "amber",
    "scarlet",
    "cobalt",
    "magenta",
    "tawny",
    "ochre",
    "sienna",
    "umber",
    "taupe",
    "mauve",
    "cerise",
    "fuchsia",
    "vermillion",
    "periwinkle",
    "chartreuse",
    "gamboge",
    "cinnabar",
    "carmine",
    "saffron",
    "turquoise",
    "verdigris",
    "puce",
    "russet",
    "bisque",
    "ecru",
    "alabaster",
    "celadon",
];

/// Varied public-domain-style prose (~750 words, ~1050 Qwen3 tokens).
/// Tiled to fill any context length. Deliberately avoids words in CWE_WORD_POOL
/// and needle number sequences.
const FILLER_CORPUS: &str = "The solar system formed approximately 4.6 billion years ago \
from a dense region of a molecular cloud. The gravitational collapse of this region caused it to \
spin and flatten into a protoplanetary disk, from which the Sun and planets eventually formed. \
The four inner planets—Mercury, Venus, Earth, and Mars—have solid rocky surfaces, while the four \
outer planets—Jupiter, Saturn, Uranus, and Neptune—are composed primarily of gas and ice.\n\n\
The ocean covers more than 70 percent of Earth's surface and contains about 97 percent of all \
the water on the planet. Deep ocean currents, driven by differences in temperature and salinity, \
circulate water around the globe over centuries. The ocean absorbs heat from the Sun and releases \
it slowly, moderating temperatures in coastal regions. Coral reefs support an estimated \
25 percent of all marine species despite covering less than one percent of the ocean floor.\n\n\
Languages are used by humans to communicate thoughts, feelings, and information. \
There are approximately 7,000 languages spoken in the world today, but more than half of the \
world's population speaks one of just 23 languages as their first language. Languages evolve \
continuously, with new words being added and old ones becoming obsolete. Writing systems, \
developed independently in several ancient civilizations, allowed language to be recorded and \
transmitted across generations.\n\n\
Mountain ranges are formed by the movement of tectonic plates, volcanic activity, or erosion. \
The Himalayas, which include Mount Everest, the highest peak on Earth at 8,849 meters, were \
formed by the collision of the Indian and Eurasian tectonic plates beginning about \
50 million years ago. Mountains affect local climate by forcing air masses to rise, cool, \
and deposit precipitation on their windward slopes.\n\n\
The human brain contains approximately 86 billion neurons, each connected to thousands of \
others through synapses. Neural signals travel as electrochemical impulses along axons and \
across synapses. The cerebral cortex, the outer layer of the brain, is responsible for higher \
cognitive functions including reasoning, language, and consciousness. Sleep is essential for \
brain function, allowing neural connections to consolidate memories.\n\n\
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy \
into chemical energy stored as glucose. In the chloroplasts of plant cells, chlorophyll absorbs \
red and blue light while reflecting green light, giving plants their characteristic color. \
The light-dependent reactions produce oxygen as a byproduct, which is released into the \
atmosphere. Over geological time, photosynthesis has transformed Earth's atmosphere.\n\n\
Trade routes connecting civilizations around the ancient world facilitated the exchange of \
goods, ideas, and technologies. The Silk Road, a network of overland routes stretching from \
China to the Mediterranean, was used from around 200 BCE to the 15th century CE. Merchants \
carried silk, spices, glassware, and other commodities across thousands of kilometers. Along \
with physical goods, religions, art styles, and diseases spread through these trading networks.\n\n\
Rivers are a vital component of the freshwater cycle, transporting water from elevated terrain \
to the sea. The Amazon River in South America carries the largest volume of water of any river \
in the world, discharging about 20 percent of all fresh water that flows into the world's \
oceans. Rivers shape the landscape through erosion and deposition, forming features such as \
canyons, alluvial fans, and deltas.\n\n\
The periodic table of elements, first organized by Dmitri Mendeleev in 1869, arranges the \
known chemical elements by increasing atomic number and groups them by similar chemical \
properties. Elements in the same group share the same number of valence electrons, which \
determines how they bond with other elements. Metals make up the majority of elements and \
are generally shiny, malleable, and good conductors of heat and electricity.\n\n\
Architecture has evolved significantly over thousands of years, reflecting advances in \
materials, construction techniques, and cultural values. Ancient Egyptian architects built \
massive stone monuments such as the pyramids, which have endured for millennia. The Romans \
developed concrete and the arch, enabling the construction of large-scale infrastructure \
like aqueducts and amphitheaters. Modern architects use steel, reinforced concrete, and \
glass to create skyscrapers and other complex structures that were unimaginable to their \
predecessors.";

// ── Public types ──────────────────────────────────────────────────────────────

/// A single RULER evaluation sample: the prompt and its expected answer(s).
#[derive(Clone, Debug)]
pub struct RulerSample {
    /// The full prompt string ready to be tokenized and fed to the model.
    pub input: String,
    /// Expected answer strings.  Scoring checks that all appear in the prediction.
    pub outputs: Vec<String>,
}

/// RULER task variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RulerTask {
    /// Single needle-in-a-haystack — one magic number.
    NiahSingle1,
    /// Multi-key needle — two independent magic numbers.
    NiahMultiKey2,
    /// Variable tracing — follow a 4-link assignment chain.
    Vt,
    /// Common word extraction — identify the 10 most frequent words.
    Cwe,
}

impl RulerTask {
    pub fn name(self) -> &'static str {
        match self {
            Self::NiahSingle1 => "niah_single_1",
            Self::NiahMultiKey2 => "niah_multikey_2",
            Self::Vt => "vt",
            Self::Cwe => "cwe",
        }
    }
}

impl std::fmt::Display for RulerTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── Internal filler helpers ───────────────────────────────────────────────────

/// Tokenize `FILLER_CORPUS` and tile to approximately `target_len` tokens.
/// Returns a string that decodes to approximately that many tokens when
/// re-tokenized as part of a longer prompt (±2%).
fn make_filler(tokenizer: &Tokenizer, target_len: usize) -> String {
    let corpus_tokens: Vec<u32> = tokenizer
        .encode(FILLER_CORPUS, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let n = corpus_tokens.len();
    if n == 0 || target_len == 0 {
        return String::new();
    }

    // Tile to slightly under target (leave room for BPE-boundary expansion).
    let safe = target_len.saturating_sub(target_len / 20);
    let mut tiled: Vec<u32> = Vec::with_capacity(safe + n);
    while tiled.len() < safe {
        let need = safe - tiled.len();
        tiled.extend_from_slice(&corpus_tokens[..need.min(n)]);
    }

    tokenizer.decode(&tiled, true).unwrap_or_default()
}

/// Count tokens for a string using the given tokenizer.
fn token_count(tokenizer: &Tokenizer, text: &str) -> usize {
    tokenizer.encode(text, false).map(|e| e.len()).unwrap_or(0)
}

// ── Prompt assembly ───────────────────────────────────────────────────────────

/// Build a complete Qwen3 ChatML prompt with thinking suppressed.
fn build_prompt(body: &str, question: &str) -> String {
    format!(
        "{SYS_START}You are a helpful assistant.{SYS_END}\
         {USER_START}/no_think\n\
         {body}\n\n\
         Question: {question}\n\
         Answer with the exact value only.{USER_END}\
         {ASST_PREFIX}"
    )
}

// ── Task generators ───────────────────────────────────────────────────────────

/// Generate `n` NIAH-single-1 samples targeting `context_len` tokens each.
fn generate_niah_single(
    tokenizer: &Tokenizer,
    context_len: usize,
    n: usize,
    rng: &mut StdRng,
) -> Vec<RulerSample> {
    // Pre-tokenize the question template to estimate overhead.
    let question_template = "What is the special magic number for CITY?";
    let needle_template = "\n\nRemember: The special magic number for CITY is: NUMBER.\n\n";
    let overhead = token_count(tokenizer, &build_prompt("", question_template))
        + token_count(tokenizer, needle_template)
        + 10; // buffer

    let filler_tokens = context_len.saturating_sub(overhead);

    (0..n)
        .map(|_| {
            let number: u64 = rng.random_range(10_000..99_999);
            let city_idx = rng.random_range(0..CITY_NAMES.len());
            let city = CITY_NAMES[city_idx];

            let filler = make_filler(tokenizer, filler_tokens);
            // Place needle at random depth in filler.
            let depth = rng.random_range(10usize..90);
            let raw_split = filler.len() * depth / 100;
            let split = next_char_boundary(&filler, raw_split);
            let needle =
                format!("\n\nRemember: The special magic number for {city} is: {number}.\n\n");
            let body = format!("{}{}{}", &filler[..split], needle, &filler[split..]);
            let question = format!("What is the special magic number for {city}?");
            RulerSample {
                input: build_prompt(&body, &question),
                outputs: vec![number.to_string()],
            }
        })
        .collect()
}

/// Generate `n` NIAH-multikey-2 samples (2 independent needles).
fn generate_niah_multikey2(
    tokenizer: &Tokenizer,
    context_len: usize,
    n: usize,
    rng: &mut StdRng,
) -> Vec<RulerSample> {
    let overhead = token_count(
        tokenizer,
        &build_prompt("", "What are the magic numbers for CITY1 and CITY2?"),
    ) + 60;

    let filler_tokens = context_len.saturating_sub(overhead);

    (0..n)
        .map(|_| {
            let num1: u64 = rng.random_range(10_000..99_999);
            let num2: u64 = rng.random_range(10_000..99_999);
            let city1 = CITY_NAMES[rng.random_range(0..CITY_NAMES.len() / 2)];
            let city2 = CITY_NAMES[rng.random_range(CITY_NAMES.len() / 2..CITY_NAMES.len())];

            let filler = make_filler(tokenizer, filler_tokens);
            let len = filler.len();

            // Needle 1 at ~33%, needle 2 at ~67%.
            let split1 = next_char_boundary(&filler, len / 3);
            let split2 = next_char_boundary(&filler, 2 * len / 3);
            let needle1 =
                format!("\n\nRemember: The special magic number for {city1} is: {num1}.\n\n");
            let needle2 =
                format!("\n\nRemember: The special magic number for {city2} is: {num2}.\n\n");
            let body = format!(
                "{}{}{}{}{}",
                &filler[..split1],
                needle1,
                &filler[split1..split2],
                needle2,
                &filler[split2..]
            );
            let question = format!(
                "What are the special magic numbers for {city1} and {city2}? \
                 List both numbers, one per line."
            );
            RulerSample {
                input: build_prompt(&body, &question),
                outputs: vec![num1.to_string(), num2.to_string()],
            }
        })
        .collect()
}

/// Variable names used for VT (tracing A → B → C → D → value).
const VT_VARS: &[&str] = &["VAR_A", "VAR_B", "VAR_C", "VAR_D"];

/// Generate `n` variable-tracing samples: A=B, B=C, C=D, D=<number>.
/// Assignment lines are distributed throughout the haystack.
fn generate_vt(
    tokenizer: &Tokenizer,
    context_len: usize,
    n: usize,
    rng: &mut StdRng,
) -> Vec<RulerSample> {
    let overhead = token_count(tokenizer, &build_prompt("", "What is the value of VAR_A?")) + 80;
    let filler_tokens = context_len.saturating_sub(overhead);

    (0..n)
        .map(|_| {
            let value: u64 = rng.random_range(10_000..99_999);

            // Build assignment lines: A=B, B=C, C=D, D=value
            let chain_len = VT_VARS.len(); // 4
            let mut assignments = Vec::with_capacity(chain_len);
            for i in 0..chain_len - 1 {
                assignments.push(format!(
                    "\n[assignment] {} = {}\n",
                    VT_VARS[i],
                    VT_VARS[i + 1]
                ));
            }
            assignments.push(format!(
                "\n[assignment] {} = {}\n",
                VT_VARS[chain_len - 1],
                value
            ));

            // Shuffle the assignment order so the chain is non-linear in position.
            fisher_yates_shuffle(&mut assignments, rng);

            let filler = make_filler(tokenizer, filler_tokens);
            let len = filler.len();

            // Distribute 4 assignment lines at quartile positions in the filler.
            let positions: [usize; 4] = [len / 8, 3 * len / 8, 5 * len / 8, 7 * len / 8];
            let mut splits: Vec<usize> = positions
                .iter()
                .map(|&p| next_char_boundary(&filler, p.min(len)))
                .collect();
            // Ensure strictly monotone (for very short fillers the positions may coincide).
            for i in 1..splits.len() {
                if splits[i] <= splits[i - 1] {
                    splits[i] = (splits[i - 1] + 1).min(len);
                }
            }

            // Assemble: filler_chunk assignment filler_chunk ...
            let mut body = String::new();
            let mut prev = 0usize;
            for (i, &split) in splits.iter().enumerate() {
                let safe = split.min(len);
                body.push_str(&filler[prev..safe]);
                body.push_str(&assignments[i]);
                prev = safe;
            }
            body.push_str(&filler[prev..]);

            let question = format!(
                "Starting from {}, trace through the variable assignments above \
                 to find its final value. What number does {} equal?",
                VT_VARS[0], VT_VARS[0]
            );
            RulerSample {
                input: build_prompt(&body, &question),
                outputs: vec![value.to_string()],
            }
        })
        .collect()
}

/// Generate `n` common-word-extraction samples.
/// K=10 "common" words appear `freq` times each; ~2× more unique "rare" words appear once.
fn generate_cwe(
    tokenizer: &Tokenizer,
    context_len: usize,
    n: usize,
    rng: &mut StdRng,
) -> Vec<RulerSample> {
    const K: usize = 10;

    // Frequency of each common word — scale with context so common words stay dominant.
    let freq = (context_len / 300).max(8);

    // Approximate tokens per word in a comma-separated list (~2 tokens each including ", ").
    let overhead = token_count(
        tokenizer,
        &build_prompt(
            "",
            "What are the 10 most frequently appearing words? List as comma-separated words.",
        ),
    ) + 20;
    let word_tokens = context_len.saturating_sub(overhead);

    // Tokens per entry ~2.2; total entries to generate:
    let total_entries = word_tokens / 2;
    let common_entries = K * freq;
    let rare_entries = total_entries.saturating_sub(common_entries);

    (0..n)
        .map(|_| {
            // Pick K random common words from the pool (without replacement).
            let mut pool_indices: Vec<usize> = (0..CWE_WORD_POOL.len()).collect();
            fisher_yates_shuffle(&mut pool_indices, rng);
            let common_words: Vec<&str> = pool_indices[..K]
                .iter()
                .map(|&i| CWE_WORD_POOL[i])
                .collect();

            // Build the full word list: K×freq common + rare unique entries.
            let mut word_list: Vec<String> = Vec::with_capacity(total_entries);
            for &w in &common_words {
                for _ in 0..freq {
                    word_list.push(w.to_string());
                }
            }
            for i in 0..rare_entries {
                word_list.push(format!("item_{i:04}"));
            }
            fisher_yates_shuffle(&mut word_list, rng);

            let body = format!("Word list:\n{}", word_list.join(", "));
            let question = "What are the 10 most frequently appearing words in the above \
                            word list? List them as comma-separated words, any order.";
            RulerSample {
                input: build_prompt(&body, question),
                outputs: common_words.iter().map(|s| s.to_string()).collect(),
            }
        })
        .collect()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Generate `n_samples` RULER samples for the given task and context length.
///
/// Token counts are approximate (within ~5% of `context_len`).
pub fn generate_ruler_samples(
    tokenizer: &Tokenizer,
    task: RulerTask,
    context_len: usize,
    n_samples: usize,
    seed: u64,
) -> Vec<RulerSample> {
    let mut rng = StdRng::seed_from_u64(seed);
    match task {
        RulerTask::NiahSingle1 => generate_niah_single(tokenizer, context_len, n_samples, &mut rng),
        RulerTask::NiahMultiKey2 => {
            generate_niah_multikey2(tokenizer, context_len, n_samples, &mut rng)
        }
        RulerTask::Vt => generate_vt(tokenizer, context_len, n_samples, &mut rng),
        RulerTask::Cwe => generate_cwe(tokenizer, context_len, n_samples, &mut rng),
    }
}

/// Score a single prediction against the expected outputs.
///
/// - NIAH / VT: all expected strings must appear verbatim in `pred` (case-insensitive).
/// - CWE: at least half the expected words must appear in `pred`.
pub fn score_ruler_sample(task: RulerTask, pred: &str, outputs: &[String]) -> bool {
    let pred_lower = pred.to_lowercase();
    match task {
        RulerTask::NiahSingle1 | RulerTask::NiahMultiKey2 | RulerTask::Vt => {
            outputs.iter().all(|o| pred_lower.contains(o.as_str()))
        }
        RulerTask::Cwe => {
            let hits = outputs
                .iter()
                .filter(|o| pred_lower.contains(o.as_str()))
                .count();
            hits * 2 >= outputs.len()
        }
    }
}

// ── Batched inference runner ──────────────────────────────────────────────────

/// Run batched prefill + greedy decode for a slice of `RulerSample`s.
///
/// Uses a single `BatchedInferenceSession` for all samples in the slice,
/// exploiting the paged-decode kernel's batch parallelism.
///
/// # Arguments
/// * `model` — any model implementing `ManagedBatchedModel`
/// * `tokenizer` — Qwen3 tokenizer
/// * `samples` — the samples to evaluate
/// * `mode` — `None` for F16 (no compression), `Some(m)` for quantized KV
/// * `max_gen_tokens` — maximum decode steps per sequence
/// * `eos_ids` — token IDs that terminate generation
/// * `device` — GPU device
///
/// # Returns
/// One prediction string per sample (in the same order).
pub fn run_ruler_eval<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    samples: &[RulerSample],
    mode: Option<InferenceMode>,
    max_gen_tokens: usize,
    eos_ids: &[u32],
    device: &Device,
    timings_out: Option<&mut (f64, f64)>,
) -> Result<Vec<String>> {
    let n = samples.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Tokenize all prompts.
    let prompt_tokens: Vec<Vec<u32>> = samples
        .iter()
        .map(|s| {
            tokenizer
                .encode(s.input.as_str(), false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| candle::Error::Msg(e.to_string()))
        })
        .collect::<Result<_>>()?;

    let max_prompt_len = prompt_tokens.iter().map(|t| t.len()).max().unwrap_or(0);

    // Build session.
    let session_config = match mode {
        Some(m) => BatchedConfig {
            k_format: m.k_format(),
            v_format: m.v_format(),
            compression_level: m.compression_level(),
            ..BatchedConfig::default()
        },
        None => BatchedConfig::default(),
    };
    let mut session = model.create_batched_session(session_config)?;

    let seq_indices: Vec<usize> = (0..n)
        .map(|_| session.create_sequence())
        .collect::<Result<_>>()?;

    // ── Batched prefill ───────────────────────────────────────────────────────
    let t_prefill_start = std::time::Instant::now();
    let mut last_logits: Vec<Option<Tensor>> = vec![None; n];
    let mut offset = 0usize;

    while offset < max_prompt_len {
        let chunk_end_max = (offset + PREFILL_STEP).min(max_prompt_len);

        // Group sequences that still have real tokens at this offset, keyed by chunk length.
        let mut by_chunk_len: std::collections::BTreeMap<usize, (Vec<usize>, Vec<Tensor>)> =
            std::collections::BTreeMap::new();
        for (i, tokens) in prompt_tokens.iter().enumerate() {
            if offset >= tokens.len() {
                continue; // fully prefilled
            }
            let end = chunk_end_max.min(tokens.len());
            let chunk_len = end - offset;
            // forward_batched reads input_len from dims()[1], so inputs must be 2D [1, seq_len].
            let tensor = Tensor::new(&tokens[offset..end], device)?.unsqueeze(0)?;
            let e = by_chunk_len
                .entry(chunk_len)
                .or_insert_with(|| (vec![], vec![]));
            e.0.push(seq_indices[i]);
            e.1.push(tensor);
        }

        for (chunk_len, (group_seqs, group_inputs)) in by_chunk_len {
            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &group_seqs,
                    &group_inputs,
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits
                .unwrap_or_default();
            for (&seq_idx, logits) in group_seqs.iter().zip(logits_vec.into_iter()) {
                let orig_i = seq_indices
                    .iter()
                    .position(|&s| s == seq_idx)
                    .expect("seq_idx not found");
                session.advance_sequence(seq_idx, chunk_len)?;
                last_logits[orig_i] = Some(logits);
            }
        }

        offset = chunk_end_max;
    }

    // ── Batched decode ────────────────────────────────────────────────────────
    let prefill_elapsed = t_prefill_start.elapsed().as_secs_f64();
    let t_decode_start = std::time::Instant::now();
    let mut generated: Vec<Vec<u32>> = vec![Vec::with_capacity(max_gen_tokens); n];
    let mut done: Vec<bool> = vec![false; n];

    let mut current_tokens: Vec<u32> = last_logits
        .iter()
        .map(|lo| {
            lo.as_ref()
                .map(|l| argmax(l))
                .unwrap_or(Err(candle::Error::Msg("no prefill logits".into())))
        })
        .collect::<Result<_>>()?;

    for _step in 0..max_gen_tokens {
        // Mark EOS.
        for i in 0..n {
            if !done[i] && eos_ids.contains(&current_tokens[i]) {
                done[i] = true;
            }
        }
        // Push non-EOS tokens.
        for i in 0..n {
            if !done[i] {
                generated[i].push(current_tokens[i]);
            }
        }
        if done.iter().all(|&d| d) {
            break;
        }

        let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
        let active_seqs: Vec<usize> = active.iter().map(|&i| seq_indices[i]).collect();
        let active_inputs: Vec<Tensor> = active
            .iter()
            .map(|&i| Tensor::new(&[current_tokens[i]], device)?.unsqueeze(0))
            .collect::<Result<_>>()?;

        let nl = model.num_layers();
        let logits_vec = model
            .forward_wave(
                &mut session,
                &active_seqs,
                &active_inputs,
                &[],
                &[],
                &[],
                &[],
                0,
                nl,
                None,
            )?
            .logits
            .unwrap_or_default();
        for &seq_idx in &active_seqs {
            session.advance_sequence(seq_idx, 1)?;
        }
        for (orig_i, logits) in active.iter().zip(logits_vec.iter()) {
            current_tokens[*orig_i] = argmax(logits)?;
        }
    }

    let decode_elapsed = t_decode_start.elapsed().as_secs_f64();
    if let Some(out) = timings_out {
        *out = (prefill_elapsed, decode_elapsed);
    }

    // Release sequences.
    for &idx in &seq_indices {
        session.free_sequence(idx)?;
    }
    session.compact()?;

    // Decode token IDs to strings.
    prompt_tokens
        .iter()
        .enumerate()
        .map(|(i, _)| {
            tokenizer
                .decode(&generated[i], true)
                .map(|s| s.trim().to_string())
                .map_err(|e| candle::Error::Msg(e.to_string()))
        })
        .collect()
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()
}

/// Find the next valid UTF-8 char boundary at or after `pos` in `s`.
/// Stable alternative to the unstable `str::ceil_char_boundary`.
fn next_char_boundary(s: &str, pos: usize) -> usize {
    let pos = pos.min(s.len());
    let mut p = pos;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// In-place Fisher-Yates shuffle using the provided RNG.
fn fisher_yates_shuffle<T>(slice: &mut Vec<T>, rng: &mut StdRng) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// City names used as needle identifiers in NIAH tasks.
const CITY_NAMES: &[&str] = &[
    "Amsterdam",
    "Barcelona",
    "Cairo",
    "Dubai",
    "Edinburgh",
    "Frankfurt",
    "Geneva",
    "Helsinki",
    "Istanbul",
    "Jakarta",
    "Kyoto",
    "Lisbon",
    "Montreal",
    "Nairobi",
    "Oslo",
    "Prague",
    "Quebec",
    "Reykjavik",
    "Seoul",
    "Tokyo",
    "Utrecht",
    "Vienna",
    "Warsaw",
    "Xiamen",
    "Yokohama",
    "Zagreb",
    "Bogota",
    "Colombo",
    "Dhaka",
    "Erevan",
];

// ── Benchmark orchestrator ────────────────────────────────────────────────────

/// Configuration for [`run_ruler_benchmark`].
pub struct RulerBenchConfig<'a> {
    /// Model name used in printed output.
    pub model_name: &'a str,
    /// EOS token IDs (e.g. [`QWEN3_EOS_IDS`]).
    pub eos_ids: &'a [u32],
    /// Maximum total KV tokens across all concurrent sequences.
    /// Batch size is capped at `token_budget / context_len`.
    pub token_budget: usize,
    /// Maximum decode tokens per sequence.
    pub max_gen_tokens: usize,
    /// Compression modes to sweep: `(mode, label)`.
    /// `None` = F16 lossless baseline.
    pub modes: &'a [(Option<InferenceMode>, &'a str)],
    /// `(context_len, n_samples)` pairs.
    /// Keep `context_len * max_batch ≤ token_budget`.
    pub lengths_samples: &'a [(usize, usize)],
    /// Tasks to evaluate.
    pub tasks: &'a [RulerTask],
    /// Where to load evaluation samples from.
    /// Defaults to [`RulerDataSource::Generated`] when using `..Default::default()`.
    pub data_source: RulerDataSource<'a>,
}

impl Default for RulerBenchConfig<'static> {
    /// Paper-run defaults: 4 tasks × 4 lengths × 4 compression modes.
    /// Sample counts scale inversely with context length (base=20 @ 4K).
    fn default() -> Self {
        Self {
            model_name: "unknown",
            eos_ids: QWEN3_EOS_IDS,
            token_budget: 32_768,
            max_gen_tokens: 50,
            modes: &[
                (None, "F16 (lossless)"),
                (Some(InferenceMode::Q4_0), "Q4_0 (4.5 BPE)"),
                (Some(InferenceMode::C5), "C5  (PalQuant ~4.4 BPE)"),
                (Some(InferenceMode::Q3_0), "Q3_0 (3.5 BPE)"),
                (Some(InferenceMode::C8), "C8  (PalQuant ~3.3 BPE)"),
            ],
            lengths_samples: &[(4_096, 20), (8_192, 10), (16_384, 5)],
            tasks: &[
                RulerTask::NiahSingle1,
                RulerTask::NiahMultiKey2,
                RulerTask::Vt,
                RulerTask::Cwe,
            ],
            data_source: RulerDataSource::Generated,
        }
    }
}

// ── JSONL loader ──────────────────────────────────────────────────────────────

/// Load RULER samples from a canonical JSONL file.
///
/// Expected line format (RULER data-gen output):
/// ```json
/// {"index": 0, "input": "<prompt>", "outputs": ["42"]}
/// ```
///
/// Lines missing `input` or `outputs` are silently skipped.
pub fn load_ruler_samples_jsonl(path: &std::path::Path) -> Result<Vec<RulerSample>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| candle::Error::Msg(format!("JSONL read {:?}: {}", path, e)))?;
    let mut samples = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Minimal JSON parse — avoid pulling in serde_json as a dep.
        // We only need "input" (string) and "outputs" (array of strings).
        let input = extract_json_str(line, "input").ok_or_else(|| {
            candle::Error::Msg(format!(
                "JSONL {:?} line {}: missing \"input\" field",
                path,
                lineno + 1
            ))
        })?;
        let outputs = extract_json_str_array(line, "outputs").ok_or_else(|| {
            candle::Error::Msg(format!(
                "JSONL {:?} line {}: missing \"outputs\" field",
                path,
                lineno + 1
            ))
        })?;
        samples.push(RulerSample { input, outputs });
    }
    Ok(samples)
}

/// Extract a single JSON string value by key from a flat JSON object line.
/// Returns `None` if not found. Handles basic `\"` escape sequences.
fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..];
    let mut value = String::new();
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            match ch {
                'n' => value.push('\n'),
                't' => value.push('\t'),
                'r' => value.push('\r'),
                other => value.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(value);
        } else {
            value.push(ch);
        }
    }
    None
}

/// Extract a JSON array of strings `["a","b",...]` by key.
fn extract_json_str_array(json: &str, key: &str) -> Option<Vec<String>> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if !rest.starts_with('[') {
        return None;
    }
    let inner = &rest[1..];
    let end = inner.find(']')?;
    let items_str = &inner[..end];
    // Split on `","` boundaries (naive but sufficient for RULER outputs which
    // are simple identifiers — no embedded commas or brackets in values).
    let mut result = Vec::new();
    let mut remaining = items_str;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        if remaining.starts_with('"') {
            remaining = &remaining[1..];
            let close = remaining.find('"').unwrap_or(remaining.len());
            result.push(remaining[..close].replace("\\\"", "\""));
            remaining = &remaining[close..];
            if remaining.starts_with('"') {
                remaining = &remaining[1..];
            }
        }
        // Skip past comma
        if let Some(comma) = remaining.find(',') {
            remaining = &remaining[comma + 1..];
        } else {
            break;
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Build the JSONL filename for a (task, ctx_len) pair, e.g. `niah_single_1_4096.jsonl`.
pub fn ruler_jsonl_filename(task: RulerTask, ctx_len: usize) -> String {
    format!("{task}_{ctx_len}.jsonl")
}

/// Phase 1: parallelism sweep.
///
/// Returns the estimated KV-cache compression ratio (relative to F16) for a
/// given inference mode.  Used to scale the VRAM-safe batch ceiling when modes
/// other than F16 are evaluated in Phase 2.
fn kv_compression_ratio(mode: Option<InferenceMode>) -> f64 {
    match mode {
        None => 1.0, // F16 baseline
        Some(m) => match m {
            InferenceMode::Q8_0 | InferenceMode::Q8_1 | InferenceMode::Q8_KS => 2.0,
            InferenceMode::Q8_Q4 => 2.67,
            InferenceMode::Q8_Q4KS | InferenceMode::Q8_Q8KS => 2.0,
            InferenceMode::Q4_0 | InferenceMode::Q4_1 | InferenceMode::Q4_KS => 4.0,
            InferenceMode::Q3_0 => 5.3,
            InferenceMode::Q2_0 => 8.0,
            InferenceMode::C5 => 3.7,
            InferenceMode::C8 => 4.8,
            InferenceMode::C9 => 5.4,
            InferenceMode::C10 => 7.4,
            InferenceMode::C0 => 1.1,
            InferenceMode::C1 => 1.3,
            InferenceMode::C2 => 1.7,
            InferenceMode::C3 => 2.2,
            InferenceMode::C4 => 2.9,
            InferenceMode::C6 => 4.1,
            InferenceMode::C7 => 4.5,
            _ => 1.0, // conservative fallback for unknown modes
        },
    }
}

/// Runs NIAH-S1 at each context length with `mode=None` (F16), doubling the
/// batch size from 1 until `batch * ctx_len > token_budget` or throughput
/// plateaus.  Returns the F16 throughput-optimal batch, then scales it by each
/// mode's compression ratio to produce per-mode batch ceilings for Phase 2.
pub fn sweep_parallelism<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    cfg: &RulerBenchConfig<'_>,
) -> Result<Vec<(usize, Vec<usize>)>> {
    println!("\n=== Phase 1: Parallelism sweep (NIAH-S1, F16 KV) ===");
    println!(
        "{:<10} {:>8} {:>14} {:>14} {:>10}",
        "Ctx", "Batch", "Prefill t/s", "Decode t/s", "Selected"
    );

    // Safety factor applied when scaling the F16 batch ceiling by compression ratio.
    // Accounts for imprecision in ratio estimates and activation memory overhead.
    // 0.63 = 0.90 baseline × 0.70 (extra 30% headroom requested).
    const SAFETY: f64 = 0.63;

    let mut best_per_length: Vec<(usize, Vec<usize>)> = Vec::new(); // (ctx_len, [batch_per_mode])

    for &(ctx_len, _) in cfg.lengths_samples {
        // Generate a fixed set of samples for the sweep (seed=0).
        // We need enough samples to fill the largest batch we'll try.
        let max_batch = cfg.token_budget / ctx_len;
        let n_sweep = max_batch.max(1);
        let samples =
            generate_ruler_samples(tokenizer, RulerTask::NiahSingle1, ctx_len, n_sweep, 0);

        let mut best_batch = 1usize;
        let mut best_decode_tps = 0.0f64;

        let mut batch = 1usize;
        loop {
            if batch > samples.len() {
                break;
            }
            let batch_samples = &samples[..batch];

            let mut timings = (0.0f64, 0.0f64);
            let preds = run_ruler_eval(
                model,
                tokenizer,
                batch_samples,
                None,
                cfg.max_gen_tokens,
                cfg.eos_ids,
                model.device(),
                Some(&mut timings),
            )?;
            let _ = preds; // discard predictions for sweep

            let (prefill_secs, decode_secs) = timings;

            // Prompt tokens for this batch.
            let prompt_toks: usize = batch_samples
                .iter()
                .map(|s| {
                    tokenizer
                        .encode(s.input.as_str(), false)
                        .map(|e| e.len())
                        .unwrap_or(0)
                })
                .sum();
            let decode_toks = batch * cfg.max_gen_tokens;

            let prefill_tps = if prefill_secs > 0.0 {
                prompt_toks as f64 / prefill_secs
            } else {
                0.0
            };
            let decode_tps = if decode_secs > 0.0 {
                decode_toks as f64 / decode_secs
            } else {
                0.0
            };

            let selected = if decode_tps >= best_decode_tps * 0.97 {
                // Still improving (or within 3% noise) — record as best.
                best_decode_tps = decode_tps;
                best_batch = batch;
                "  <--"
            } else {
                "  plateau"
            };

            println!(
                "{:<10} {:>8} {:>14.0} {:>14.0}{}",
                format!("{}K", ctx_len / 1024),
                batch,
                prefill_tps,
                decode_tps,
                selected,
            );

            // Next batch size: double, but cap at token_budget / ctx_len.
            let next = (batch * 2).min(cfg.token_budget / ctx_len);
            if next == batch || decode_tps < best_decode_tps * 0.90 {
                break; // plateau or cap reached
            }
            batch = next;
        }

        println!("  → Best F16 batch for {}K: {}", ctx_len / 1024, best_batch);

        // Scale to per-mode batch using compression ratio × safety factor.
        // token_budget / ctx_len is the F16 hard ceiling; best_batch may be lower.
        let f16_ceiling = cfg.token_budget / ctx_len;
        println!(
            "  → Projected batches for {}K (safety={:.0}%):",
            ctx_len / 1024,
            SAFETY * 100.0
        );
        let mode_batches: Vec<usize> = cfg
            .modes
            .iter()
            .map(|&(mode, label)| {
                let ratio = kv_compression_ratio(mode);
                // Scale from the F16 ceiling (not just best_batch) so compressed
                // modes are not penalised if F16 plateaued before hitting the wall.
                let b = ((f16_ceiling as f64 * ratio * SAFETY).floor() as usize).max(1);
                println!("      {:<24}  ratio={:.1}×  →  batch {}", label, ratio, b);
                b
            })
            .collect();

        best_per_length.push((ctx_len, mode_batches));
    }

    Ok(best_per_length)
}

/// Phase 2: full RULER benchmark.
///
/// Uses the best batch sizes from the sweep.  Evaluates all `cfg.modes ×
/// cfg.lengths_samples × cfg.tasks` and prints an accuracy table.
pub fn run_ruler_benchmark<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    cfg: &RulerBenchConfig<'_>,
) -> Result<()> {
    let best_batches = sweep_parallelism(model, tokenizer, cfg)?;

    let n_modes = cfg.modes.len();
    let n_tasks = cfg.tasks.len();
    let n_lengths = cfg.lengths_samples.len();

    // scores[mode][length][task]
    let mut scores = vec![vec![vec![f64::NAN; n_tasks]; n_lengths]; n_modes];

    println!(
        "\n=== Phase 2: Full RULER benchmark — {} ===",
        cfg.model_name
    );

    for (li, &(ctx_len, n_samples)) in cfg.lengths_samples.iter().enumerate() {
        let mode_batches: &[usize] = best_batches
            .iter()
            .find(|(c, _)| *c == ctx_len)
            .map(|(_, mb)| mb.as_slice())
            .unwrap_or(&[]);

        println!(
            "\n─── Context {}K  ({} samples/task) ───",
            ctx_len / 1024,
            n_samples
        );

        for (ti, &task) in cfg.tasks.iter().enumerate() {
            // Load or generate data once; reuse across compression modes.
            let all_samples = match &cfg.data_source {
                RulerDataSource::Generated => {
                    print!("  Generating {} … ", task);
                    let seed = (li as u64) * 1000 + (ti as u64);
                    generate_ruler_samples(tokenizer, task, ctx_len, n_samples, seed)
                }
                RulerDataSource::Jsonl(dir) => {
                    let filename = ruler_jsonl_filename(task, ctx_len);
                    let path = dir.join(&filename);
                    print!("  Loading {} … ", path.display());
                    let mut v = load_ruler_samples_jsonl(&path)?;
                    v.truncate(n_samples);
                    v
                }
            };
            let prompt_tok_count = all_samples
                .first()
                .map(|s| {
                    tokenizer
                        .encode(s.input.as_str(), false)
                        .map(|e| e.len())
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            println!(
                "{} samples, prompt ≈ {} tok",
                all_samples.len(),
                prompt_tok_count
            );

            for (mi, &(mode, mode_label)) in cfg.modes.iter().enumerate() {
                let mode_batch = mode_batches.get(mi).copied().unwrap_or(1).max(1);
                print!("    [{mode_label}] (b={mode_batch}) ");
                #[allow(unused_imports)]
                use std::io::Write as _;
                std::io::stdout().flush().ok();

                // Run in mini-batches of `mode_batch`.
                let mut correct = 0usize;
                let t0 = std::time::Instant::now();
                let mut i = 0;
                while i < all_samples.len() {
                    let end = (i + mode_batch).min(all_samples.len());
                    let batch = &all_samples[i..end];
                    let preds = run_ruler_eval(
                        model,
                        tokenizer,
                        batch,
                        mode,
                        cfg.max_gen_tokens,
                        cfg.eos_ids,
                        model.device(),
                        None,
                    )?;
                    for (pred, sample) in preds.iter().zip(batch.iter()) {
                        if score_ruler_sample(task, pred, &sample.outputs) {
                            correct += 1;
                        }
                    }
                    i = end;
                }
                let elapsed = t0.elapsed().as_secs_f64();
                let acc = 100.0 * correct as f64 / all_samples.len() as f64;
                scores[mi][li][ti] = acc;
                println!(
                    "{correct}/{} = {acc:.1}%  ({elapsed:.1}s)",
                    all_samples.len()
                );
            }
        }
    }

    // ── Summary table ─────────────────────────────────────────────────────────
    println!("\n=== RULER Accuracy (%) — {} ===", cfg.model_name);
    let col_w = 14usize;
    let lbl_w = 26usize;

    let header_cells: Vec<String> = cfg
        .lengths_samples
        .iter()
        .flat_map(|&(ctx_len, n)| {
            cfg.tasks
                .iter()
                .map(move |&t| format!("{}@{}K(n={})", &t.name()[..4], ctx_len / 1024, n))
        })
        .collect();

    print!("{:<lbl_w$}", "Compression");
    for h in &header_cells {
        print!("{h:>col_w$}");
    }
    println!();
    println!("{}", "-".repeat(lbl_w + col_w * header_cells.len()));

    for (mi, &(_, mode_label)) in cfg.modes.iter().enumerate() {
        print!("{:<lbl_w$}", mode_label);
        for li in 0..n_lengths {
            for ti in 0..n_tasks {
                let s = scores[mi][li][ti];
                if s.is_nan() {
                    print!("{:>col_w$}", "—");
                } else {
                    print!("{s:>col_w$.1}");
                }
            }
        }
        println!();
    }

    // Per-mode averages.
    println!("{}", "-".repeat(lbl_w + col_w * header_cells.len()));
    for (mi, &(_, mode_label)) in cfg.modes.iter().enumerate() {
        let vals: Vec<f64> = {
            let mut v = Vec::new();
            for li in 0..n_lengths {
                for ti in 0..n_tasks {
                    v.push(scores[mi][li][ti]);
                }
            }
            v
        };
        let vals: Vec<f64> = vals.into_iter().filter(|s| !s.is_nan()).collect();
        let avg = if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        print!("{:<lbl_w$}", mode_label);
        for _ in 0..header_cells.len().saturating_sub(1) {
            print!("{:>col_w$}", "");
        }
        if avg.is_nan() {
            println!("{:>col_w$}  ← avg", "—");
        } else {
            println!("{avg:>col_w$.1}  ← avg");
        }
    }

    Ok(())
}

// ── Streaming benchmark (Mode 1) ──────────────────────────────────────────────

/// Parse a KV-compression mode label (e.g. "F16", "C8", "Q8_0") into an
/// `Option<InferenceMode>` plus a canonical label string.
///
/// Returns `Ok((None, "F16"))` for F16 (lossless).
/// Returns `Err` for unrecognised labels.
pub fn parse_ruler_mode(s: &str) -> Result<(Option<InferenceMode>, String)> {
    let upper = s.to_uppercase();
    let mode = match upper.as_str() {
        "F16" | "NONE" => None,
        "Q8_0" => Some(InferenceMode::Q8_0),
        "Q8_Q4" | "Q8/Q4" => Some(InferenceMode::Q8_Q4),
        "Q4_0" => Some(InferenceMode::Q4_0),
        "Q3_0" => Some(InferenceMode::Q3_0),
        "C5" => Some(InferenceMode::C5),
        "C8" => Some(InferenceMode::C8),
        "C9" => Some(InferenceMode::C9),
        "C10" => Some(InferenceMode::C10),
        _ => return Err(candle::Error::Msg(format!("Unknown RULER mode: '{s}'"))),
    };
    Ok((mode, upper))
}

/// Continuous append-mode RULER benchmark (Mode 1).
///
/// Runs until the process is killed.  After every batch one JSONL line per
/// sample is appended to `log_path` and the file is flushed.
///
/// Each output line:
/// ```json
/// {"quant":"C8","ctx":4096,"task":"niah_single_1","sample":7,"correct":true,"elapsed":1.23}
/// ```
///
/// Slots are (task × ctx_len) pairs cycled in a flat round-robin — each slot
/// advances independently.  If a slot fails (e.g. OOM) it backs off
/// exponentially (1 → 2 → 4 → … → 30 s) before retrying, while other slots
/// continue uninterrupted.
pub fn run_ruler_continuous<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    mode: Option<InferenceMode>,
    mode_label: &str,
    log_path: &std::path::Path,
    tasks: &[RulerTask],
    ctx_config: &[(usize, usize)], // (ctx_len, n_concurrent)
    max_gen_tokens: usize,
    eos_ids: &[u32],
) -> Result<()> {
    use std::io::Write as _;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| candle::Error::Msg(format!("open {:?}: {e}", log_path)))?;
    let mut writer = std::io::BufWriter::new(file);

    println!(
        "\n=== RULER stream: mode={mode_label}  log={} ===",
        log_path.display()
    );
    println!(
        "Ctx batches: {}",
        ctx_config
            .iter()
            .map(|(c, n)| format!("{}K×{n}", c / 1024))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Tasks: {}",
        tasks
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Press Ctrl+C to stop — results flushed after each batch.\n");

    // Per-slot state
    struct Slot {
        task: RulerTask,
        ctx_len: usize,
        n_concurrent: usize,
        batch_idx: u64, // monotonic; used as seed and for sample numbering
        backoff_secs: f64,
        next_run_at: std::time::Instant, // schedule: run the most-overdue slot first
    }

    let now = std::time::Instant::now();
    let mut slots: Vec<Slot> = tasks
        .iter()
        .flat_map(|&task| {
            ctx_config.iter().map(move |&(ctx_len, n_concurrent)| Slot {
                task,
                ctx_len,
                n_concurrent,
                batch_idx: 0,
                backoff_secs: 0.0,
                next_run_at: now,
            })
        })
        .collect();

    let device = model.device();

    // Simple xorshift64 RNG — no external dep needed.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let mut rng: u64 = if seed == 0 { 0xdeadbeefcafe1234 } else { seed };
    let mut xorshift = move || -> u64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    loop {
        // Among all slots that are "ready" (have the minimum next_run_at),
        // pick one at random so every task type gets fair coverage.
        let min_t = slots.iter().map(|s| s.next_run_at).min().unwrap();
        let candidates: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.next_run_at == min_t)
            .map(|(i, _)| i)
            .collect();
        let slot_idx = candidates[xorshift() as usize % candidates.len()];
        let slot = &mut slots[slot_idx];

        // Exponential backoff: sleep if the slot failed last time.
        if slot.backoff_secs > 0.0 {
            let sleep_ms = (slot.backoff_secs * 1000.0) as u64;
            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
        }

        let seed = slot.batch_idx;
        let sample_base = slot.batch_idx * slot.n_concurrent as u64;
        let samples =
            generate_ruler_samples(tokenizer, slot.task, slot.ctx_len, slot.n_concurrent, seed);

        let t0 = std::time::Instant::now();
        let preds = match run_ruler_eval(
            model,
            tokenizer,
            &samples,
            mode,
            max_gen_tokens,
            eos_ids,
            device,
            None,
        ) {
            Ok(p) => {
                // Reset backoff on success.
                slot.backoff_secs = 0.0;
                p
            }
            Err(e) => {
                // Exponential backoff: 1 → 2 → 4 → … capped at 30 s.
                slot.backoff_secs = if slot.backoff_secs < 1.0 {
                    1.0
                } else {
                    (slot.backoff_secs * 2.0).min(30.0)
                };
                eprintln!(
                    "[WARN] batch failed ({mode_label} {}K {}) — backoff {:.0}s: {e}",
                    slot.ctx_len / 1024,
                    slot.task.name(),
                    slot.backoff_secs,
                );
                // On error: push next_run_at forward by backoff so we don't spin.
                slot.next_run_at = std::time::Instant::now()
                    + std::time::Duration::from_secs_f64(slot.backoff_secs);
                continue;
            }
        };
        let elapsed = t0.elapsed().as_secs_f64();
        let per_sample = elapsed / slot.n_concurrent.max(1) as f64;

        // Schedule next run for this slot at "now" so it is eligible
        // immediately. Slower slots will accumulate larger next_run_at
        // values naturally, letting faster slots run more often.
        slot.next_run_at = std::time::Instant::now();

        let mut batch_correct = 0usize;
        for (i, (pred, sample)) in preds.iter().zip(samples.iter()).enumerate() {
            let correct = score_ruler_sample(slot.task, pred, &sample.outputs);
            if correct {
                batch_correct += 1;
            }
            let line = format!(
                    "{{\"quant\":\"{mode_label}\",\"ctx\":{},\"task\":\"{}\",\"sample\":{},\"correct\":{correct},\"elapsed\":{per_sample:.3}}}\n",
                    slot.ctx_len,
                    slot.task.name(),
                    sample_base + i as u64,
                );
            writer
                .write_all(line.as_bytes())
                .map_err(|e| candle::Error::Msg(e.to_string()))?;
        }
        writer
            .flush()
            .map_err(|e| candle::Error::Msg(e.to_string()))?;

        println!(
            "[{mode_label} {}K {}  batch={}]  {batch_correct}/{} correct  {elapsed:.1}s",
            slot.ctx_len / 1024,
            slot.task.name(),
            slot.batch_idx,
            slot.n_concurrent,
        );

        slot.batch_idx += 1;
    }
}

// ── Report (Mode 2) ────────────────────────────────────────────────────────────

/// Read a JSONL log produced by [`run_ruler_continuous`] and print an accuracy
/// table grouped by (quant × ctx_len × task).
pub fn print_ruler_report(log_path: &std::path::Path) -> Result<()> {
    use std::collections::{BTreeMap, BTreeSet};

    let text = std::fs::read_to_string(log_path)
        .map_err(|e| candle::Error::Msg(format!("read {:?}: {e}", log_path)))?;

    // (quant, ctx_len, task) → (correct, scored_total, errored)
    // scored_total = success + failed (excludes errored)
    let mut data: BTreeMap<(String, u64, String), (usize, usize, usize)> = BTreeMap::new();
    let mut skipped = 0usize;
    let mut total_errors = 0usize;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Prefer new "status" field; fall back to legacy "correct" bool.
        let status = extract_json_str(line, "status");
        let legacy_correct = extract_json_bool(line, "correct");
        let resolved: Option<&'static str> = match (status.as_deref(), legacy_correct) {
            (Some("success"), _) => Some("success"),
            (Some("failed"), _) => Some("failed"),
            (Some("error"), _) => Some("error"),
            (None, Some(true)) => Some("success"),
            (None, Some(false)) => Some("failed"),
            _ => None,
        };
        match (
            extract_json_str(line, "quant"),
            extract_json_u64(line, "ctx"),
            extract_json_str(line, "task"),
            resolved,
        ) {
            (Some(q), Some(c), Some(t), Some(st)) => {
                let e = data.entry((q, c, t)).or_insert((0, 0, 0));
                match st {
                    "success" => {
                        e.0 += 1;
                        e.1 += 1;
                    }
                    "failed" => {
                        e.1 += 1;
                    }
                    "error" => {
                        e.2 += 1;
                        total_errors += 1;
                    }
                    _ => {}
                }
            }
            _ => skipped += 1,
        }
    }

    if data.is_empty() {
        println!("No data found in {:?}", log_path);
        return Ok(());
    }
    if skipped > 0 {
        println!("(skipped {skipped} malformed lines)");
    }
    if total_errors > 0 {
        println!("(excluded {total_errors} errored runs from accuracy totals)");
    }

    let quants: Vec<String> = data
        .keys()
        .map(|(q, _, _)| q.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let ctxs: Vec<u64> = data
        .keys()
        .map(|(_, c, _)| *c)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let task_names: Vec<String> = data
        .keys()
        .map(|(_, _, t)| t.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let col_w = 12usize;
    let lbl_w = 10usize;
    let n_cols = ctxs.len() * task_names.len();

    println!("\n=== RULER Report: {} ===", log_path.display());
    print!("{:<lbl_w$}", "Mode");
    for ctx in &ctxs {
        for task in &task_names {
            let abbr = &task[..task.len().min(4)];
            print!("{:>col_w$}", format!("{}@{}K", abbr, ctx / 1024));
        }
    }
    println!("  {:>8}  {:>6}", "Avg%", "Err");
    println!("{}", "─".repeat(lbl_w + col_w * n_cols + 18));

    for quant in &quants {
        print!("{:<lbl_w$}", quant);
        let (mut tot_c, mut tot_n, mut tot_err) = (0usize, 0usize, 0usize);
        for ctx in &ctxs {
            for task in &task_names {
                match data.get(&(quant.clone(), *ctx, task.clone())) {
                    Some((c, n, err)) => {
                        tot_c += c;
                        tot_n += n;
                        tot_err += err;
                        let cell = if *n > 0 {
                            format!("{:.0}%({})", 100.0 * *c as f64 / *n as f64, n)
                        } else {
                            // No scored runs — only errors (or nothing); show err count.
                            format!("E{}", err)
                        };
                        print!("{:>col_w$}", cell);
                    }
                    None => print!("{:>col_w$}", "—"),
                }
            }
        }
        let avg = if tot_n > 0 {
            format!("{:.1}", 100.0 * tot_c as f64 / tot_n as f64)
        } else {
            "—".to_string()
        };
        println!("  {:>8}  {:>6}", avg, tot_err);
    }

    Ok(())
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}
