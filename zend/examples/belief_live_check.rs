//! Replicate the DAEMON's live belief scan offline: build the real projection
//! (yaml + tool catalog), open the substrate exactly as the engine does, and call
//! the *live* `Conversation::score_belief_collections` on a real stored probe.
//! If this prints all-zero scores, the live path is broken independently of the
//! GPU decode probe.
//!
//! ```text
//! cargo run -p zend --example belief_live_check --release -- <probe_stream_id_hex>
//! ```

use std::path::PathBuf;

use candle_conversation::models::Dialect;
use candle_conversation::persistence::streams::StreamId;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{Builder, Conversation, SystemPromptItem};
use candle_conversation::provenance::decode_wide_sigs;
use candle_conversation::substrate::Substrate;

const YAML: &str = include_str!("../src/prompts/projection.yaml");

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(probe_hex) = args.get(1) else {
        anyhow::bail!(
            "usage: belief_live_check <probe_stream_id_hex>\n\
             (pick a stream id from `substrate_inspect streams` — the turn whose \
             stored wide-Q sig should be scored against the live gallery)"
        );
    };
    let probe_id = StreamId(u64::from_str_radix(probe_hex.trim_start_matches("0x"), 16)?);

    // 1. Build the projection EXACTLY as the daemon does (build_projection_builder
    //    + install_tool_catalog).
    let dialect = Dialect::chat_ml();
    let mut builder =
        Builder::from_yaml_with_vars_and_dialect(YAML, &[("workspace", "candle")], Some(&dialect))
            .expect("yaml parse");
    let tool_sections = zend::tools::install_tool_catalog(&mut builder)?;
    println!("installed {} tool sections", tool_sections.len());

    // 2. Open the substrate exactly as the engine does.
    let mut substrate = Substrate::new();
    let persistence =
        SubstratePersistence::open_in_with_substrate(&PathBuf::from("."), &mut substrate)?;
    let conv = Conversation::from_parts(substrate, persistence);

    // 3. Read the probe turn's stored wide-Q sig.
    let mut probe = {
        let sub = conv.read();
        let blob = sub
            .stream_of(probe_id)
            .and_then(|e| e.wide_q_sigs.clone())
            .expect("probe has no wide_q_sigs");
        decode_wide_sigs(&blob).expect("probe decodes")
    };
    // Optional: truncate to the FIRST N windows to mimic an early live reproject
    // probe (`FIRST=N`), which is where the daemon mis-ranks.
    if let Ok(n) = std::env::var("FIRST").unwrap_or_default().parse::<usize>() {
        probe.truncate(n);
    }
    println!("probe: {} wide-Q windows", probe.len());

    // 4. Report the shared system prompt's tools collection state.
    let sp = &builder.schema().system_prompt;
    for item in &sp.items {
        if let SystemPromptItem::Collection(coll) = item {
            println!(
                "collection {:?}: {} sections, policy.tags={:?}",
                coll.name,
                coll.sections.len(),
                coll.policy.tags
            );
        }
    }

    // 5. The LIVE call the reproject uses. `observe: false` — this is a
    // read-only diagnostic, and folding hit levels into the normalization band
    // is the once-per-turn seal scan's job; no arena, so the CPU scan runs (same
    // ranking as the GPU gallery path).
    let scores = conv.score_belief_collections(sp, &probe, None, false, None);

    // 6. Report the top scored tools.
    let mut ranked: Vec<(String, f32)> = Vec::new();
    for item in &sp.items {
        if let SystemPromptItem::Collection(coll) = item {
            for s in &coll.sections {
                ranked.push((s.name.clone(), scores.section(s.id)));
            }
        }
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let nonzero = ranked.iter().filter(|(_, s)| *s != 0.0).count();
    println!(
        "\nscore_belief_collections → {nonzero} non-zero of {} sections",
        ranked.len()
    );
    for (name, score) in ranked.iter().take(10) {
        println!("  {name:30} {score:10.3}");
    }
    if nonzero == 0 {
        println!(
            "\n>>> ALL ZERO — the live belief scan produced no scores. BUG CONFIRMED offline."
        );
    } else {
        println!("\n>>> live scan produced scores — the offline path is healthy.");
    }
    Ok(())
}
