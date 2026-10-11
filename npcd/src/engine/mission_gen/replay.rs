//! The table's reading, replayed on the real model under the sampling the
//! daemon decodes it with and with each repetition control lifted in turn.
//!
//! **What a degenerate field is made of is a sampling question, and it is
//! answered by decoding.** Live readings ended "Keeper encrypted, rejected and
//! explained. The.Keeper encrypted… Kepper encrypted" and "datethe is datethe":
//! a loop whose words are spelled a new way each time round. This runs the
//! same reading on the same drafts under the daemon's sampler and prints every
//! answer whole — how long it thought, how its thinking ended, whether it named
//! its faults, and what the reading's own check made of the call.
//!
//! Ignored by default: it loads the checkpoint onto the GPU (stop the daemon
//! first) and reads the mind at `D:/prog/mind`.
//! `cargo test --release -p npcd --lib replay -- --ignored --nocapture`

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle::Device;
use candle_conversation::SequenceConfig;
use futures::future::join_all;

use super::answer::Fields;
use super::config::{Config, READING_CONTEXT};
use super::corpus::Corpus;
use super::material;
use super::reading;
use super::run::{reading_prompt, READING_TOKENS, THINK};
use crate::engine::journal::tools::arguments;
use crate::engine::mind::Minds;
use crate::engine::prose::{decode_call_thought, CallAsk};
use crate::model;

const MIND: &str = "D:/prog/mind";
const WORLD: &str = "battle-cities";

/// Drafts the live table read: a life event whose `checked` looped, and
/// stories of the length the table reads most.
const DRAFTS: &[&str] = &[
    "layers/life/keeper/2750-09-12 The Silence of the Zenling Plague.md",
    "layers/stories/the-price-of-sigma-7.md",
    "layers/stories/the-unmappable.md",
    "layers/stories/the-breach-at-vault-sigma-7.md",
];

const SEEDS: &[u64] = &[11, 29, 47, 83, 101, 131, 157, 193];

/// The samplers replayed: the daemon's own.
///
/// The presence penalty is what holds this checkpoint out of a loop: lifted,
/// readings went round until their cap and five of eight were refused (16
/// readings, 2026-10-10). DRY, measured beside it, changed nothing but the
/// spelling of a loop and is no longer on any checkpoint's sampler.
fn samplers(base: &SequenceConfig) -> Vec<(&'static str, SequenceConfig)> {
    vec![("as deployed", base.clone())]
}

/// Whether a reading's call names any fault.
fn names_faults(raw: &str) -> bool {
    arguments(raw).is_some_and(|(_, args)| !reading::faults_of(&args).is_empty())
}

/// The reasoning block of a turn as written and the call after it.
fn thought_and_call(whole: &str) -> (&str, &str) {
    match whole.rfind("</think>") {
        Some(at) => (&whole[..at], whole[at + "</think>".len()..].trim_start()),
        None => ("", whole),
    }
}

/// Each field of a reading and its length in words, or what kept it from
/// being a call.
fn fields(raw: &str) -> String {
    match arguments(raw) {
        Some((_, args)) => ["event", "checked", "faults", "verdict"]
            .iter()
            .map(|&k| {
                let v = match k {
                    "faults" => reading::faults_of(&args),
                    _ => Fields(&args).get(k),
                };
                format!("{k}={}", v.split_whitespace().count())
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => "not a call".to_string(),
    }
}

#[test]
#[ignore = "loads the checkpoint on the GPU and reads the live mind at D:/prog/mind"]
fn the_reading_replayed_on_the_real_model() {
    let root = Path::new(MIND);
    let config = Config::load(root)
        .expect("missions.yaml parses")
        .expect("the mind has a missions.yaml");
    let corpus = Corpus::read(root, WORLD);

    let workspace = tempfile::tempdir().expect("a workspace for the substrate");
    let device = Device::new_cuda(0).expect("a CUDA device");
    let mut builder = model::model().builder().workspace_path(workspace.path());
    let conversation = builder.conversation_config();
    let engine = Arc::new(Mutex::new(
        builder.engine(&device).expect("the engine loads"),
    ));
    // The daemon's own config: `Minds::new` resolves the reasoning block's
    // close tokens and budget. Taken from the builder alone, the block's
    // close never fired and the readings thought past their budget.
    let base = Minds::new(Arc::clone(&engine), conversation).base_config();

    // Every section a reading may be shown.
    let every: Vec<String> = READING_CONTEXT.iter().map(|s| s.to_string()).collect();
    let mut asks: Vec<(&str, SequenceConfig, &str, u64, String)> = Vec::new();
    for (name, sampler) in samplers(&base) {
        for &draft in DRAFTS {
            let material =
                material::draft(&corpus, draft, None, &every).expect("the draft is on disk");
            let prompt = reading_prompt(&material, None, &config.reading);
            for &seed in SEEDS {
                asks.push((name, sampler.clone(), draft, seed, prompt.clone()));
            }
        }
    }

    let started = Instant::now();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let raws = rt.block_on(join_all(asks.iter().map(
        |(_, sampler, _, seed, prompt)| {
            let engine = Arc::clone(&engine);
            let system = config.reader.clone();
            async move {
                let calls = reading::specs();
                let ask = CallAsk {
                    system: &system,
                    prompt,
                    calls: &calls,
                    max_tokens: READING_TOKENS,
                    temperature: None,
                    seed: *seed,
                    think: THINK,
                };
                decode_call_thought(&engine, sampler, &ask)
                    .await
                    .unwrap_or_else(|e| format!("decode failed: {e:#}"))
            }
        },
    )));
    println!(
        "{} readings in {:.0} s",
        raws.len(),
        started.elapsed().as_secs_f64()
    );

    let tokens = |text: &str| -> usize {
        engine
            .lock()
            .unwrap()
            .tokenizer()
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
    };
    let (graceful, _) = THINK.eot_budget();
    let mut rows: Vec<(&str, bool, bool, bool)> = Vec::new();
    for ((name, _, draft, seed, _), whole) in asks.iter().zip(&raws) {
        let (thought, raw) = thought_and_call(whole);
        let thought_tokens = tokens(thought);
        let cut = thought_tokens >= graceful as usize;
        let taken = reading::check(raw, draft, &corpus, true);
        let verdict = match &taken {
            Ok(r) => format!("taken ({:?})", r.verdict),
            Err(fault) => format!("refused: {}", fault.chars().take(160).collect::<String>()),
        };
        let ending: String = {
            let t = thought.trim_end();
            let from = t.char_indices().rev().nth(300).map_or(0, |(i, _)| i);
            t[from..].to_string()
        };
        println!(
            "\n===== {name} | {draft} | seed {seed}\nthought {thought_tokens} tokens{} | {} | \
             {verdict}\n--- thought ends: …{ending}\n--- call:\n{raw}",
            if cut { " (at the budget)" } else { "" },
            fields(raw)
        );
        rows.push((name, cut, taken.is_ok(), names_faults(raw)));
    }
    println!("\n===== summary");
    for (name, _) in samplers(&base) {
        let mine: Vec<_> = rows.iter().filter(|(n, ..)| *n == name).collect();
        let count = |cut: bool| mine.iter().filter(|(_, c, ..)| *c == cut).count();
        let taken = |cut: bool| mine.iter().filter(|(_, c, t, _)| *c == cut && *t).count();
        let faulted = mine.iter().filter(|(.., f)| *f).count();
        println!(
            "{name}: thought closed by the model {} of {} taken; cut at the budget {} of {} \
             taken; faults named in {faulted} of {}",
            taken(false),
            count(false),
            taken(true),
            count(true),
            mine.len()
        );
    }
}
