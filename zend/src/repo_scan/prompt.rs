//! The shared system prompt text that the ingest layers' in-memory tree formats
//! each turn with.
//!
//! The text is not what the model attends to. A conversation's prompt K/V is the
//! schema's content-addressed sections, laid onto its slot by its first turn, and
//! an append-only ingest never re-projects, so that frame is its prompt for life.
//! The string walks the schema at the same default branch as that frame, so it
//! mirrors the K/V it sits beside. Collection nodes are skipped, since their
//! members are chosen at projection, and the walk continues through the nodes
//! below them as ordinary content.

use candle_conversation::projection;

use super::*;

/// The shared system prompt every ingest layer frames on, wrapped in the
/// engine's dialect markers. Used by `repo_map`, `code_reading` and the raw
/// layers so all three read from one place.
///
/// **This text does not become K/V, and that is the point.** A conversation's
/// prompt K/V is the schema's own content-addressed sections, laid onto its
/// empty slot by its first turn from its frame (`ProjectionInputs::frame`, the
/// birth prefix `Sequence::new_with_projection` builds) — the same stored
/// sections a chat conversation injects. An append-only ingest never
/// re-projects (`skip_projection`), so that frame is its prompt for the
/// conversation's whole life and no runtime selection can alter it. What this
/// string is for is the in-memory tree's per-turn formatting; it walks the
/// schema at the SAME default branch the frame uses, so the text mirrors the
/// K/V rather than describing a prompt that never existed. Changing what an ingest is framed by means changing the schema's
/// defaults.
pub(crate) fn shared_system_prompt(
    builder: &projection::Builder,
    layer_name: &str,
    config: &SequenceConfig,
) -> String {
    debug_assert!(
        builder.schema().layers.iter().any(|l| l.name == layer_name),
        "projection schema missing '{layer_name}' layer"
    );
    config
        .dialect
        .format_system_prompt(&shared_prompt_body(builder))
}

/// The unwrapped body of [`shared_system_prompt`] — the schema walk at the
/// default branch, with no dialect framing. Split out so the assembled prompt
/// can be asserted directly.
pub(crate) fn shared_prompt_body(builder: &projection::Builder) -> String {
    use projection::SystemPromptItem;
    let mut body = String::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => body.push_str(&s.content),
            SystemPromptItem::SectionTree(tree) => {
                for node in &tree.nodes {
                    // A collection node has no options of its own; its members
                    // are provenance-selected and live-prefilled at projection.
                    // Skip it and keep walking — the nodes below it are ordinary
                    // content, and stopping here (as the dialogue prelude does,
                    // its contract being "the text before the tools") would drop
                    // them.
                    if node.collection.is_some() {
                        continue;
                    }
                    // Structural markers (`<tools>` …) are generated at
                    // projection, never part of the static prelude.
                    if node.glue.is_some() {
                        continue;
                    }
                    if let Some(option) = node.options.get(node.chosen(&tree.default_selection)) {
                        body.push_str(&option.content);
                    }
                }
            }
            // Same reasoning as the in-tree collection node above.
            SystemPromptItem::Collection(_) => continue,
        }
    }
    body
}
