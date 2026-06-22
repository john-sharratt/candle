//! Model-loading integration test for the compression summary tree.
//!
//! Loads the dev model (`Qwen3-30B-A3B Q4_K_M`, ~16 GB with expert LRU), submits
//! 32 distinctive prefill-only turns, waits for the auto-spawned summariser thread
//! to absorb them into the AVL summary tree, then reads back every
//! `SummaryOfTurns` / `SummaryOfSummaries` node and decodes its compressed
//! text. The assertions are deliberately lenient — the point is to *examine*
//! the compressed output (printed with `--nocapture`), not to over-constrain
//! it.
//!
//! Run with:
//! ```bash
//! cargo test -p candle-conversation --features hub \
//!     --test compression_integration -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use candle_conversation::{
    models::Model, projection::TimelineId, summary_tree::TurnKind, SamplingConfig,
};

const COMP_MODEL: Model = Model::Qwen3_30B_A3B_Q4;

/// Six rotating mini-topics so the absorbed turns carry real, distinctive
/// content the compressor must preserve (names, numbers, decisions).
fn topic_exchange(i: usize) -> (String, String) {
    let topic = i % 6;
    let (user, assistant) = match topic {
        0 => (
            "We're planning a trip to Kyoto in April for 9 days. \
             Budget is 3200 dollars for two people. What should we prioritise?",
            "For Kyoto in April, prioritise Fushimi Inari at dawn and Arashiyama \
             bamboo grove. With 3200 dollars for 9 days, budget about 180 dollars \
             a day after the flights, and book a ryokan for two of the nights.",
        ),
        1 => (
            "The auth service is returning 401 for valid tokens after the deploy at 2pm. \
             Refresh tokens still work. Any idea?",
            "A 401 on valid access tokens but working refresh tokens points at a clock \
             skew or a rotated signing key. Check that the 2pm deploy didn't change the \
             JWT issuer or the public key kid the verifier loads.",
        ),
        2 => (
            "I want to make tonkotsu ramen from scratch this weekend. \
             How long does the pork-bone broth actually need?",
            "Tonkotsu broth needs a hard rolling boil for 10 to 12 hours to emulsify the \
             collagen into that milky white. Blanch the bones first, then top up water as \
             it reduces. The tare and chashu you can do in parallel on day two.",
        ),
        3 => (
            "Our Q3 budget spreadsheet shows marketing at 48k but finance approved only 40k. \
             Where do we cut the 8k?",
            "Cut the 8k from the paid-social experiment line (about 6k) and trim the \
             conference travel by 2k. Keep the content retainer intact since it drives the \
             organic pipeline that finance is measuring you on.",
        ),
        4 => (
            "My dog Pixel, a 3-year-old border collie, keeps chewing the door frame when \
             I leave. How do I stop it?",
            "Border collies like Pixel chew from under-stimulation, not spite. Give Pixel a \
             frozen stuffed Kong as you leave and a 30-minute fetch session beforehand. The \
             door-frame chewing should drop within two weeks of real exercise.",
        ),
        _ => (
            "I'm getting 'cannot borrow `self.cache` as mutable more than once' in my Rust \
             loop. How do I restructure it?",
            "That borrow-check error means you're holding a mutable borrow of self.cache \
             across loop iterations. Collect the keys you need into a Vec first, drop that \
             borrow, then mutate self.cache in a second pass over the owned keys.",
        ),
    };
    (format!("Turn {i}: {user}"), assistant.to_string())
}

#[test]
#[ignore = "loads a real model on CUDA"]
fn compression_tree_over_32_turns() -> candle_conversation::Result<()> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!(
                "compression_tree_over_32_turns: no CUDA device available — skipping (this test \
                 loads a real model on the GPU)."
            );
            return Ok(());
        }
    };

    let tmp = tempfile::tempdir().expect("tempdir");

    // Build the builder, then pull the system prompt + conversation config out
    // BEFORE `.engine()` borrows it mutably to load the model. The summariser
    // thread is auto-spawned by the engine — we do not spawn it ourselves.
    let mut builder = COMP_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(64)
        .max_concurrent(4)
        .workspace_path(tmp.path());
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();

    eprintln!("\n=== Loading {COMP_MODEL} ===");
    let load_start = Instant::now();
    let engine = builder.engine(&device)?;
    eprintln!("    Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let mut conv = engine.new_conversation(&system_prompt, config)?;
    let timeline: TimelineId = conv.timeline_id();

    // Submit 32 distinctive prefill-only turns.
    for i in 0..32usize {
        let (user, assistant) = topic_exchange(i);
        conv.insert_turn(&user, &assistant)?;
    }
    eprintln!("Submitted 32 turns; waiting for the summariser to reach idle...");

    // Wait for the auto-spawned summariser to drain the pending queue. The
    // ternary carry builds every internal node synchronously as leaves land, so
    // once nothing is pending the forest is whole (no separate sweep/backlog).
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut reached_idle = false;
    loop {
        let pending = engine.pending_summary_len(timeline);
        if pending == 0 {
            reached_idle = true;
            break;
        }
        if Instant::now() >= deadline {
            eprintln!("summariser did not reach idle within 600s: pending={pending}");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(reached_idle, "summariser must reach idle (pending=0)");
    eprintln!("Summariser reached idle.\n");

    // Read back the forest and dump every compressed node.
    let conversation = engine.conversation();
    let tok = engine.tokenizer();
    let guard = conversation.read();

    let peaks = guard.peaks_of(timeline);
    assert!(
        !peaks.is_empty(),
        "a summary forest must exist after 32 turns"
    );
    let (root, root_height) = peaks.iter().max_by_key(|(_, l)| *l).copied().unwrap();
    eprintln!(
        "Tallest peak = {root:?}, level = {root_height}, peaks = {}, turn_count = {}\n",
        peaks.len(),
        guard.turn_count(timeline)
    );

    let mut n_sot = 0usize;
    let mut n_sos = 0usize;

    let count = guard.turn_count(timeline);
    for i in 0..count {
        let idx = candle_conversation::projection::TurnIndex(i);
        let meta = match guard.tree_meta_of(timeline, idx) {
            Some(m) => m.clone(),
            None => continue,
        };
        let is_summary = matches!(
            meta.kind,
            TurnKind::SummaryOfTurns | TurnKind::SummaryOfSummaries
        );
        if !is_summary {
            continue;
        }
        match meta.kind {
            TurnKind::SummaryOfTurns => n_sot += 1,
            TurnKind::SummaryOfSummaries => n_sos += 1,
            TurnKind::Normal => {}
        }

        let ids = guard.assistant_token_ids_of(timeline, idx).to_vec();
        let text = tok
            .decode(&ids, true)
            .unwrap_or_else(|e| format!("<decode error: {e}>"));
        // `assistant_text_of` is the generated-only text (no leaked prompt
        // tail) — the actual compressed rewrite.
        let generated = guard.assistant_text_of(timeline, idx);
        eprintln!(
            "── node {idx:?}  kind={:?}  height={}  children={:?}  tokens={}",
            meta.kind,
            meta.tree_height,
            meta.children,
            ids.len()
        );
        eprintln!("   [generated] {}", generated.trim());
        eprintln!("   [token_ids] {}\n", text.trim());

        assert!(
            !ids.is_empty(),
            "compressed node {idx:?} must carry decoded tokens"
        );
    }

    eprintln!("==== Totals: {n_sot} SummaryOfTurns, {n_sos} SummaryOfSummaries ====\n");

    assert!(n_sot > 0, "at least one SummaryOfTurns leaf must form");

    drop(guard);
    engine.shutdown()?;
    Ok(())
}

/// Distinct whitespace-split tokens — a degenerate loop ("2022\n2022\n…" /
/// "assistant\nassistant\n…") collapses to 1–2; a real compression has dozens.
fn distinct_word_count(s: &str) -> usize {
    use std::collections::HashSet;
    s.split_whitespace().collect::<HashSet<_>>().len()
}

/// Reproduction: drive the **actual zend projection template** (not the plain
/// synthetic fallback the test above uses) through a real compression decode.
///
/// This is the daemon's path — the per-layer `turns`/`summaries` ×
/// `question`/`answer` prompts only ever got parse-validated, never decoded. In
/// the live daemon the zend compression collapsed into degenerate loops
/// (`2022\n2022\n…`), so this asserts every summary node carries a real
/// (non-degenerate) compression.
#[test]
#[ignore = "loads a real model on CUDA"]
fn compression_over_zend_schema() -> candle_conversation::Result<()> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("compression_over_zend_schema: no CUDA device — skipping.");
            return Ok(());
        }
    };

    let tmp = tempfile::tempdir().expect("tempdir");
    let mut builder = COMP_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(64)
        .max_concurrent(4)
        .workspace_path(tmp.path());
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();

    eprintln!("\n=== Loading {COMP_MODEL} (zend schema) ===");
    let engine = builder.engine(&device)?;

    // The real daemon projection template — same `from_yaml_with_vars_and_dialect`
    // call `zend::session::build_projection_builder` makes. Tools are left out
    // (the dialogue compression path doesn't depend on the tool catalog).
    const ZEND_YAML: &str = include_str!("../../zend/src/prompts/projection.yaml");
    let mut proj_builder =
        candle_conversation::projection::Builder::from_yaml_with_vars_and_dialect(
            ZEND_YAML,
            &[("workspace", "test-project")],
            Some(&config.dialect),
        )
        .expect("zend projection.yaml must parse");
    // `kind: template` items (system_open/close, no_think_prefix, …) must be
    // pre-tokenised before projection — same call zend's session setup makes.
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
    let dialogue_layer = proj_builder
        .id_for_layer("dialogue")
        .expect("zend schema must declare a 'dialogue' layer");
    let primary_group = proj_builder
        .id_for_group("primary_conversation")
        .expect("zend schema must declare a 'primary_conversation' group");

    // Install tool sections into the dialogue layer's `tools` collection —
    // exactly what the daemon's `install_tool_catalog` does. Runtime section-id
    // allocation must not collide with the compression-prompt section ids (which
    // live in `layer.summary`, not `system_prompt.items`), or `ensure_summary_-
    // section` injects a tool's JSON as the compression prompt and the decode
    // degenerates. This is the conditions the live daemon hit.
    let tools_coll = proj_builder
        .id_for_collection_in(dialogue_layer, "tools")
        .expect("dialogue layer must declare a 'tools' collection");
    let mut tool_ids: Vec<u32> = Vec::new();
    for t in 0..8 {
        let id = proj_builder
            .add_section_to_collection(
                dialogue_layer,
                tools_coll,
                format!("tool_{t}"),
                format!(
                    "{{\"name\":\"tool_{t}\",\"description\":\"does thing {t}\",\"parameters\":{{}}}}"
                ),
                50.0,
            )
            .expect("add tool section");
        tool_ids.push(id.raw());
    }
    // Diagnosis: are the compression-prompt section ids disjoint from the tool
    // section ids? A collision means a tool's JSON gets injected as the
    // compression prompt.
    {
        let sch = proj_builder.schema();
        let dl = sch
            .layers
            .iter()
            .find(|l| l.id == dialogue_layer)
            .expect("dialogue layer in schema");
        let mut comp_ids = vec![
            dl.summary.turns.user.system_prompt.id.raw(),
            dl.summary.turns.assistant.system_prompt.id.raw(),
        ];
        if let Some(s) = &dl.summary.summaries {
            comp_ids.push(s.user.system_prompt.id.raw());
            comp_ids.push(s.assistant.system_prompt.id.raw());
        }
        eprintln!("compression-prompt section ids: {comp_ids:?}");
        eprintln!("tool section ids:               {tool_ids:?}");
        let collide: Vec<u32> = comp_ids
            .iter()
            .copied()
            .filter(|c| tool_ids.contains(c))
            .collect();
        assert!(
            collide.is_empty(),
            "compression-prompt section ids {comp_ids:?} collide with runtime-added tool \
             section ids {tool_ids:?} (collision: {collide:?}) — ensure_summary_section would \
             inject a tool's JSON as the compression prompt, corrupting the compression decode"
        );
    }

    let mut conv = engine.new_conversation_with_projection(
        &system_prompt,
        proj_builder,
        dialogue_layer,
        primary_group,
        config,
    )?;
    let timeline: TimelineId = conv.timeline_id();

    // 8 distinctive turns. Turn 0 is long (~1k tokens) so its content prefill
    // chunks, matching the 1240-token live turn.
    for i in 0..8usize {
        let (user, assistant) = topic_exchange(i);
        let assistant = if i == 0 {
            // Repeat the distinctive bodies to ~1k tokens of varied content.
            let mut long = String::new();
            for j in 0..6 {
                long.push_str(&topic_exchange(j).1);
                long.push(' ');
            }
            long
        } else {
            assistant
        };
        conv.insert_turn(&user, &assistant)?;
    }
    eprintln!("Submitted 8 turns (zend schema); waiting for summariser idle...");

    let deadline = Instant::now() + Duration::from_secs(600);
    let mut reached_idle = false;
    loop {
        if engine.pending_summary_len(timeline) == 0 {
            reached_idle = true;
            break;
        }
        if Instant::now() >= deadline {
            eprintln!("summariser did not reach idle within 600s");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(reached_idle, "summariser must reach idle");

    let conversation = engine.conversation();
    let guard = conversation.read();
    let count = guard.turn_count(timeline);
    let mut nodes = 0usize;
    let mut degenerate = 0usize;
    for i in 0..count {
        let idx = candle_conversation::projection::TurnIndex(i);
        let meta = match guard.tree_meta_of(timeline, idx) {
            Some(m) => m.clone(),
            None => continue,
        };
        if !matches!(
            meta.kind,
            TurnKind::SummaryOfTurns | TurnKind::SummaryOfSummaries
        ) {
            continue;
        }
        nodes += 1;
        let generated = guard.assistant_text_of(timeline, idx);
        let distinct = distinct_word_count(&generated);
        let is_degenerate = distinct < 8;
        if is_degenerate {
            degenerate += 1;
        }
        eprintln!(
            "── {idx:?} kind={:?} distinct_words={distinct}{}",
            meta.kind,
            if is_degenerate {
                "   <<< DEGENERATE"
            } else {
                ""
            }
        );
        eprintln!(
            "   [generated] {}",
            generated.trim().chars().take(220).collect::<String>()
        );
    }
    drop(guard);
    engine.shutdown()?;

    assert!(nodes > 0, "summary nodes must form");
    assert_eq!(
        degenerate, 0,
        "{degenerate}/{nodes} compressed nodes degenerated into loops on the zend schema — \
         the compression decode context is corrupted"
    );
    Ok(())
}
