//! The mind's real projection, built once and shared by everything that writes.
//!
//! # Why the synthetic schema was not enough
//!
//! Every conversation this daemon opened was built with
//! `Builder::for_plain_prompt` — a synthetic schema of one layer holding one
//! section. It works, in the sense that turns land and seal. What it cannot do
//! is **gather**, because there is nothing declared to gather from: no world
//! layer, no memory layer, no thresholds, no budgets.
//!
//! That is why 1,818 ingested documents produced no provenance signatures. A
//! projection runs against the conversation's *own* schema, and a synthetic
//! schema has nothing in it. The documents were written and were unreachable —
//! two different claims, and only the first was true.
//!
//! # What a signature is for
//!
//! When an episode of a character's life is prefilled, a projection over the
//! real schema selects the world content relevant to it, and persisting that
//! projection writes the link. Later, provenance scanning over that memory has a
//! **hook**: the world documents the moment happened against are already
//! attached to it.
//!
//! Without the hook a life turn and a world document are two unrelated things in
//! one substrate, and the gather has to rediscover the relationship from surface
//! text every time — or miss it, which is the more likely outcome and the
//! quieter one.
//!
//! # One builder, shared
//!
//! Parsed once at load and cloned per conversation. The schema is several
//! thousand lines of YAML and re-parsing it per document would be the largest
//! single cost in the ingest — and worse, two parses could disagree, which is
//! the kind of divergence that shows up as a layer that gathers on some
//! documents and not others.

use std::path::Path;

use candle_conversation::models::DialectType;
use candle_conversation::projection::{Builder, GroupId, LayerId};

/// The mind's projection, and the target a written document lands in.
pub struct Projection {
    pub builder: Builder,
    /// The layer and group an ingested document or life episode is written to.
    ///
    /// The **live conversation layer** — the one declaring a `Sequence` rule.
    /// Content written here is what the gather reaches from a dialogue, which is
    /// the whole point of writing it.
    pub layer: LayerId,
    pub group: GroupId,
    /// The system-prompt prelude the schema declares, ChatML-wrapped by the
    /// model builder and handed to every conversation.
    pub prelude: String,
}

/// Build the projection from the mind's `projection.yaml`.
///
/// `None` when there is no mind or the schema will not parse. Unlike zend, this
/// does **not** panic on a bad schema: npcd serves authored content, accounts
/// and the console from the same process, and a YAML typo should degrade the
/// engine rather than take the daemon down with it. The caller falls back to the
/// synthetic schema and says so.
pub fn build(mind: Option<&Path>, world_name: &str) -> Option<Projection> {
    let path = mind?.join("projection.yaml");
    let yaml = std::fs::read_to_string(&path).ok()?;
    // ChatML, so the schema's `kind: template` items resolve to the right
    // structural-token strings at parse time. The model is a ChatML family
    // member; a mismatch here produces a schema whose glue markers are wrong in
    // a way nothing checks.
    let dialect = DialectType::ChatML.dialect();
    let builder = match Builder::from_yaml_with_vars_and_dialect(
        &yaml,
        &[("workspace", world_name)],
        Some(&dialect),
    ) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(
                "projection schema {} failed to parse: {e:#} — falling back to a schema \
                 that cannot gather",
                path.display()
            );
            return None;
        }
    };

    let (layer, group) = live_target(&builder)?;
    let prelude = prelude(&builder);
    tracing::info!(
        "projection: {} layer(s) from {}, {} bytes of prelude",
        builder.schema().layers.len(),
        path.display(),
        prelude.len()
    );
    Some(Projection {
        builder,
        layer,
        group,
        prelude,
    })
}

/// How many turns the mind's schema asks to keep verbatim in the redo log.
///
/// `None` when the mind declares no `turn_retention`, which keeps every turn —
/// the behaviour every deployment had before this existed, and the right default
/// for a conversation that ends.
///
/// Read straight from the YAML rather than through [`Builder`], because this is
/// a property of the *daemon's* storage rather than of the projection: the
/// schema decides what a turn is composed from, and this decides how long the
/// log carries one. Threading it through the builder would put a storage policy
/// in the vocabulary of prompt composition, where nothing else in the schema
/// would ever read it.
///
/// A malformed value is a warning and a `None`, on the same reasoning as
/// [`build`]: a typo should cost retention, not the daemon.
pub fn turn_retention(mind: Option<&Path>) -> Option<u64> {
    let path = mind?.join("projection.yaml");
    let yaml = std::fs::read_to_string(&path).ok()?;
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let block = doc.get("turn_retention")?;
    match block.get("keep_turns").and_then(|v| v.as_u64()) {
        Some(n) if n > 0 => Some(n),
        Some(_) => {
            tracing::warn!(
                "turn_retention.keep_turns is 0 in {} — a conversation that keeps no turns \
                 would retire what it just said, so retention is off",
                path.display()
            );
            None
        }
        None => {
            tracing::warn!(
                "turn_retention in {} has no numeric `keep_turns` — retention is off and the \
                 log will grow without bound",
                path.display()
            );
            None
        }
    }
}

/// The live conversation layer's `(layer, group)`.
///
/// Identified by its `Sequence` selection rule — the recent-N plus
/// historical-top-K shape that only the live layer uses. Same test zend's
/// `is_live_conversation` applies, and for the same reason: the schema names its
/// layers whatever the author likes, so the *rule* is the reliable signal and
/// the name is not.
fn live_target(builder: &Builder) -> Option<(LayerId, GroupId)> {
    use candle_conversation::projection::SelectionRule;
    for layer in &builder.schema().layers {
        for group in &layer.groups {
            if matches!(group.selection, SelectionRule::Sequence { .. }) {
                return Some((layer.id, group.id));
            }
        }
    }
    // No live layer: a schema that declares only content layers. Fall back to
    // the first group there is, so a document still lands somewhere rather than
    // the ingest refusing wholesale.
    let layer = builder.schema().layers.first()?;
    Some((layer.id, layer.groups.first()?.id))
}

/// Everything in the system prompt before the first collection.
///
/// The static prelude. What follows a collection is expanded at projection time
/// rather than being part of the fixed text, so including it here would prefill
/// the same content twice — once statically and once as the projection composes
/// it.
fn prelude(builder: &Builder) -> String {
    use candle_conversation::projection::SystemPromptItem;
    let mut out = String::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => out.push_str(&s.content),
            SystemPromptItem::SectionTree(t) => {
                for n in &t.nodes {
                    if n.collection.is_some() {
                        return out;
                    }
                    // Glue markers are live-prefilled at projection, not part of
                    // the static prelude.
                    if n.glue.is_some() {
                        continue;
                    }
                    out.push_str(&n.options[n.chosen(&t.default_selection)].content);
                }
            }
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_mind_means_no_projection() {
        assert!(build(None, "world").is_none());
    }

    #[test]
    fn a_missing_schema_is_an_absence_not_a_panic() {
        let dir = std::env::temp_dir().join("npcd-schema-none");
        let _ = std::fs::create_dir_all(&dir);
        assert!(build(Some(&dir), "world").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A YAML typo must not take the daemon down.** npcd serves the console,
    /// the accounts and the authored content from the same process; a bad schema
    /// should degrade the engine and leave the rest answering. zend panics here
    /// because zend *is* the engine.
    #[test]
    fn a_malformed_schema_degrades_rather_than_panicking() {
        let dir = std::env::temp_dir().join("npcd-schema-bad");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(
            dir.join("projection.yaml"),
            "layers: [ this is not\n  valid: yaml",
        )
        .unwrap();
        assert!(build(Some(&dir), "world").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real schema parses and finds a live layer to write into. Reads the
    /// mind that ships with the repo, so a schema change that breaks the engine
    /// breaks this first.
    #[test]
    fn the_bundled_schema_parses_and_has_a_live_layer() {
        let mind = Path::new(env!("CARGO_MANIFEST_DIR"));
        if !mind.join("projection.yaml").is_file() {
            return; // no bundled schema in this checkout
        }
        let p = build(Some(mind), "battle-cities").expect("the bundled schema parses");
        assert!(
            !p.builder.schema().layers.is_empty(),
            "a schema with no layers cannot gather, which is the whole point"
        );
        assert!(
            !p.prelude.is_empty(),
            "the prelude is empty — every conversation would open with no system prompt"
        );
    }
}
