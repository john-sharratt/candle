//! Generates the RULER task samples procedurally, placing the needles, assignments
//! or words in filler text sized to the requested context length.
//!
//! Each task places its target at a position in the filler: the single needle at a
//! random depth, the two multi-key needles near one third and two thirds, the
//! variable-tracing chain at the quartiles in shuffled order so it is not linear in
//! position, and the common-word list with its frequent words scaled to the context
//! so they stay dominant. Token counts are approximate, within about five percent of
//! the requested length. Positions are rounded to UTF-8 character boundaries so
//! every sample stays valid text. Generation is deterministic for a given seed.

use super::*;

// ── Task generators ───────────────────────────────────────────────────────────

/// Generate `n` NIAH-single-1 samples targeting `context_len` tokens each.
pub(super) fn generate_niah_single(
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
pub(super) fn generate_niah_multikey2(
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
pub(super) const VT_VARS: &[&str] = &["VAR_A", "VAR_B", "VAR_C", "VAR_D"];

/// Generate `n` variable-tracing samples: A=B, B=C, C=D, D=<number>.
/// Assignment lines are distributed throughout the haystack.
pub(super) fn generate_vt(
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
pub(super) fn generate_cwe(
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

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Find the next valid UTF-8 char boundary at or after `pos` in `s`.
/// Stable alternative to the unstable `str::ceil_char_boundary`.
pub(super) fn next_char_boundary(s: &str, pos: usize) -> usize {
    let pos = pos.min(s.len());
    let mut p = pos;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// In-place Fisher-Yates shuffle using the provided RNG.
pub(super) fn fisher_yates_shuffle<T>(slice: &mut Vec<T>, rng: &mut StdRng) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// City names used as needle identifiers in NIAH tasks.
pub(super) const CITY_NAMES: &[&str] = &[
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
