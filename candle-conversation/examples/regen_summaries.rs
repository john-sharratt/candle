//! `regen_summaries` — load the model and regenerate a layer's summary forest
//! over a fixture conversation, printing every source↔summary pair so the
//! compression prompts can be debugged and iterated.
//!
//! It boots a real `ConversationEngine` (model on CUDA) with the production
//! zend projection template, ingests a fixed set of dialogue turns into the
//! chosen layer, lets the async summariser absorb them into the immutable
//! ternary forest (`docs/immutable_summary_forest.md`) using *the current*
//! per-layer `summary` prompts, then walks the forest and prints, for each
//! node:
//!   - SoT leaf: the source turn (user + assistant) and the regenerated
//!     compressed question / answer halves;
//!   - SoS node: its children and the regenerated higher-level summary.
//!
//! Edit `zend/src/prompts/projection.yaml` and re-run to see the effect.
//!
//! ```bash
//! cargo run -p candle-conversation --example regen_summaries \
//!     --features cuda,hub --release -- --layer dialogue
//! ```

use std::time::{Duration, Instant};

use candle_conversation::projection::{Builder, TimelineId, TurnIndex};
use candle_conversation::summary_tree::TurnKind;
use candle_conversation::{models::Model, SamplingConfig};

const MODEL: Model = Model::Qwen3_30B_A3B_Q4;
const ZEND_YAML: &str = include_str!("../../zend/src/prompts/projection.yaml");

/// Pick the fixture matching the layer's content kind: repository index
/// listings for the structural layers, conversational turns otherwise.
fn fixture_for(layer: &str) -> Vec<(&'static str, &'static str)> {
    match layer {
        "repo_map" | "code_reading" => repo_fixture(),
        _ => dialogue_fixture(),
    }
}

/// Repository-index turns, mirroring what `zend::repo_scan` prefills: a header
/// naming the scope + a file listing with line counts / languages. A faithful
/// repo_map summary keeps the directory/file *names* (the structural skeleton)
/// and drops the line counts and incidental detail — and invents nothing.
fn repo_fixture() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "Repository index — workspace root:",
            "(workspace root)\n  - CHANGELOG.md (113 lines, Markdown)\n  - CLAUDE.md (194 lines, \
             Markdown)\n  - Cargo.toml (125 lines, TOML, workspace: 11 members)\n  - README.md \
             (171 lines, Markdown)\n  - candle-core/ (crate)\n  - candle-nn/ (crate)\n  - \
             candle-transformers/ (crate)\n  - candle-kernels/ (crate)\n  - candle-conversation/ \
             (crate)\n  - candle-examples/ (crate)\n",
        ),
        (
            "Repository index — `candle-core/src`:",
            "candle-core/src/\n  - tensor.rs (2104 lines, Rust, Tensor ops)\n  - device.rs (412 \
             lines, Rust, Device enum)\n  - dtype.rs (288 lines, Rust)\n  - shape.rs (520 lines, \
             Rust)\n  - cpu_backend/ (module)\n  - cuda_backend/ (module)\n  - metal_backend/ \
             (module)\n  - quantized/ (module)\n",
        ),
        (
            "Repository index — `candle-nn/src/kv_cache`:",
            "candle-nn/src/kv_cache/\n  - mod.rs (640 lines, Rust, KvFormat/QuantFormat)\n  - \
             cache.rs (380 lines, Rust, KvCache)\n  - rotating.rs (510 lines, Rust, \
             RotatingKvCache)\n  - arena_table.rs (720 lines, Rust)\n  - chunked/ (module)\n",
        ),
        (
            "Repository index — `candle-conversation/src/projection`:",
            "candle-conversation/src/projection/\n  - mod.rs (240 lines, Rust)\n  - schema.rs (900 \
             lines, Rust, LayerSchema/TurnSummary)\n  - yaml.rs (760 lines, Rust)\n  - builder.rs \
             (1020 lines, Rust)\n  - project.rs (880 lines, Rust)\n  - resolver.rs (1100 lines, \
             Rust)\n",
        ),
        (
            "Repository index — `candle-transformers/src/models`:",
            "candle-transformers/src/models/\n  - qwen3.rs (1240 lines, Rust, Qwen3 MoE)\n  - \
             llama.rs (980 lines, Rust)\n  - quantized_qwen3.rs (820 lines, Rust)\n  - mod.rs (60 \
             lines, Rust)\n",
        ),
        (
            "Repository index — `candle-kernels/src`:",
            "candle-kernels/src/\n  - lib.rs (140 lines, Rust, FFI bindings)\n  - build.rs (320 \
             lines, Rust)\n  - paged-decode/ (CUDA kernels)\n  - paged-prefill/ (CUDA kernels)\n  \
             - quantized/ (CUDA kernels)\n",
        ),
    ]
}

/// Fixture dialogue. Concrete, checkable facts (names, numbers) so faithfulness
/// is visible: a good summary keeps exactly these and invents nothing; the
/// known failure modes are (a) the answer-half embellishing with facts not
/// present and (b) a thin/meta source being fabricated into a whole essay.
fn dialogue_fixture() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "Give me a tour of the codebase — main crates, key files, and how everything connects.",
            "This workspace is `candle`, a Rust ML library. The top-level `Cargo.toml` is a \
             workspace with 11 members; the real work lives in the `candle-*` crates. \
             `candle-core` holds `Tensor`, `Device`, and `DType`. `candle-nn` has the neural \
             layers and the KV-cache system. `candle-transformers` has the batched-inference \
             models. `candle-kernels` holds the CUDA kernels. Runnable demos live in \
             `candle-examples`.",
        ),
        (
            "What question did I ask at the start of this conversation?",
            "You asked for a tour of the codebase — its main crates, key files, and how \
             everything connects.",
        ),
        (
            "We're planning a 9-day trip to Kyoto in April with a 3200 dollar budget for two \
             people. What should we prioritise?",
            "Prioritise Fushimi Inari at dawn and the Arashiyama bamboo grove. On 3200 dollars \
             across 9 days, budget about 180 dollars a day after flights, and book a ryokan for \
             two of the nights.",
        ),
        (
            "The auth service returns 401 for valid access tokens after the 2pm deploy, but \
             refresh tokens still work. Any idea?",
            "A 401 on valid access tokens while refresh tokens still work points to clock skew \
             or a rotated signing key. Check whether the 2pm deploy changed the JWT issuer or \
             the public-key kid the verifier loads.",
        ),
        (
            "Tonkotsu ramen this weekend — how long does the pork-bone broth actually need?",
            "Tonkotsu broth needs a hard rolling boil for 10 to 12 hours to emulsify the \
             collagen into that milky white. Blanch the bones first, then top up water as it \
             reduces.",
        ),
        (
            "Our Q3 marketing budget shows 48k but finance approved only 40k. Where do we cut \
             the 8k?",
            "Cut about 6k from the paid-social experiment line and trim 2k from conference \
             travel. Keep the content retainer intact — it drives the organic pipeline finance \
             measures you on.",
        ),
    ]
}

fn arg_value(flag: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == flag {
            return w[1].clone();
        }
    }
    default.to_string()
}

fn main() -> candle_conversation::Result<()> {
    let layer_name = arg_value("--layer", "dialogue");

    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("regen_summaries needs a CUDA device (it loads a real model). Aborting.");
            return Ok(());
        }
    };

    let tmp = tempfile::tempdir().expect("tempdir");
    let mut builder = MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(64)
        .max_concurrent(4)
        .workspace_path(tmp.path());
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();

    eprintln!("=== Loading {MODEL} (zend schema, layer='{layer_name}') ===");
    let t0 = Instant::now();
    let engine = builder.engine(&device)?;
    eprintln!("    loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut proj_builder = Builder::from_yaml_with_vars_and_dialect(
        ZEND_YAML,
        &[("workspace", "test-project")],
        Some(&config.dialect),
    )
    .expect("zend projection.yaml must parse");
    {
        let tokenizer = engine.tokenizer();
        proj_builder
            .tokenize_templates::<String, _>(|s| {
                tokenizer
                    .encode(s, false)
                    .map(|enc| enc.get_ids().to_vec())
                    .map_err(|e| e.to_string())
            })
            .expect("tokenize templates");
    }
    let layer = proj_builder
        .id_for_layer(&layer_name)
        .unwrap_or_else(|| panic!("zend schema must declare a '{layer_name}' layer"));
    // Use the target layer's own first group (dialogue → primary_conversation,
    // repo_map → structure, …) rather than hardcoding the dialogue group.
    let group = proj_builder
        .schema()
        .layers
        .iter()
        .find(|l| l.id == layer)
        .and_then(|l| l.groups.first())
        .map(|g| g.id)
        .unwrap_or_else(|| panic!("layer '{layer_name}' declares no group"));

    let mut conv = engine.new_conversation_with_projection(
        &system_prompt,
        proj_builder,
        layer,
        group,
        config,
    )?;
    let timeline: TimelineId = conv.timeline_id();

    let turns = fixture_for(&layer_name);
    eprintln!("Ingesting {} turns...", turns.len());
    for (user, assistant) in &turns {
        conv.insert_turn(user, assistant)?;
    }

    // Wait until every ingested turn has its SoT leaf. `pending_summary_len`
    // hits 0 the moment the summariser *pops* the queue — well before the slow
    // per-turn compressions finish — so block on the leaf count instead, or we
    // print a half-built forest.
    eprintln!(
        "Waiting for the summariser to compress all {} turns...",
        turns.len()
    );
    let deadline = Instant::now() + Duration::from_secs(1800);
    loop {
        let leaves = count_sot_leaves(&engine, timeline);
        if engine.pending_summary_len(timeline) == 0 && leaves >= turns.len() {
            break;
        }
        if Instant::now() >= deadline {
            eprintln!(
                "WARNING: only {leaves}/{} turns compressed before timeout",
                turns.len()
            );
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    // Let any trailing carry-merge land its tree meta.
    std::thread::sleep(Duration::from_millis(750));

    print_forest(&engine, timeline, &turns);
    Ok(())
}

/// Walk the forest and print every regenerated summary against its source.
/// Source text comes from `fixture` (ground truth) rather than the substrate:
/// `assistant_text_of` returns only a stub for prefill-only turns, and we want
/// the full original to judge faithfulness against.
fn print_forest(
    engine: &candle_conversation::ConversationEngine,
    timeline: TimelineId,
    fixture: &[(&str, &str)],
) {
    let conversation = engine.conversation();
    let guard = conversation.read();
    let count = guard.turn_count(timeline);

    // Collect summary nodes, ordered: SoT leaves (by the normal turn they cover)
    // first, then SoS by ascending level.
    let mut leaves: Vec<(u32, u32)> = Vec::new(); // (normal_idx, sot_idx)
    let mut internals: Vec<(u8, u32, Vec<u32>)> = Vec::new(); // (level, idx, children)
    for i in 0..count {
        let idx = TurnIndex(i);
        let Some(meta) = guard.tree_meta_of(timeline, idx) else {
            continue;
        };
        match meta.kind {
            TurnKind::SummaryOfTurns => {
                let normal = meta.children.first().map(|c| c.0).unwrap_or(i);
                leaves.push((normal, i));
            }
            TurnKind::SummaryOfSummaries => {
                internals.push((
                    meta.tree_height,
                    i,
                    meta.children.iter().map(|c| c.0).collect(),
                ));
            }
            TurnKind::Normal => {}
        }
    }
    leaves.sort_unstable();
    internals.sort_unstable();

    // Normal turns are interleaved with summary turns, so their substrate
    // indices aren't fixture order. Map each Normal turn to its fixture entry by
    // chronological rank (they're inserted in fixture order).
    let mut normal_indices: Vec<u32> = (0..count)
        .filter(|i| {
            matches!(
                guard.tree_meta_of(timeline, TurnIndex(*i)).map(|m| m.kind),
                Some(TurnKind::Normal)
            )
        })
        .collect();
    normal_indices.sort_unstable();
    let rank_of: std::collections::HashMap<u32, usize> = normal_indices
        .iter()
        .enumerate()
        .map(|(r, idx)| (*idx, r))
        .collect();

    println!(
        "\n================ REGENERATED SUMMARY FOREST (timeline {}) ================",
        timeline.raw()
    );
    println!(
        "{} SoT leaves, {} SoS internals\n",
        leaves.len(),
        internals.len()
    );

    println!("---------------- SoT leaves (compress one turn each) ----------------");
    for (normal, sot) in &leaves {
        let s = TurnIndex(*sot);
        let (src_user, src_reply) = rank_of
            .get(normal)
            .and_then(|r| fixture.get(*r))
            .copied()
            .unwrap_or(("<?>", "<?>"));
        println!("\n┌─ SoT #{sot}  compresses normal turn #{normal}");
        println!("│  SOURCE user : {}", oneline(src_user));
        println!("│  SOURCE reply: {}", oneline(src_reply));
        println!("│  ── regenerated ──");
        println!(
            "│  SUMMARY Q   : {}",
            oneline(&guard.user_text_of(timeline, s))
        );
        println!(
            "│  SUMMARY A   : {}",
            oneline(&guard.assistant_text_of(timeline, s))
        );
        println!("└─");
    }

    if !internals.is_empty() {
        println!(
            "\n---------------- SoS internals (compress {} children) ----------------",
            candle_conversation::summary_tree::MERGE_FANOUT
        );
        for (level, idx, children) in &internals {
            let s = TurnIndex(*idx);
            println!("\n┌─ SoS #{idx}  level={level}  children={children:?}");
            println!(
                "│  SUMMARY Q   : {}",
                oneline(&guard.user_text_of(timeline, s))
            );
            println!(
                "│  SUMMARY A   : {}",
                oneline(&guard.assistant_text_of(timeline, s))
            );
            println!("└─");
        }
    }
    println!("\n==========================================================================");
}

/// Count `SummaryOfTurns` leaves currently in the timeline's forest.
fn count_sot_leaves(
    engine: &candle_conversation::ConversationEngine,
    timeline: TimelineId,
) -> usize {
    let conversation = engine.conversation();
    let guard = conversation.read();
    let count = guard.turn_count(timeline);
    (0..count)
        .filter(|i| {
            matches!(
                guard.tree_meta_of(timeline, TurnIndex(*i)).map(|m| m.kind),
                Some(TurnKind::SummaryOfTurns)
            )
        })
        .count()
}

/// Collapse whitespace/newlines so each field prints on one line.
fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
