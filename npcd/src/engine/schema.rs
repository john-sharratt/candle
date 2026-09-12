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

use crate::engine::dreams;

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

    // Nobody's dreams until a conversation says whose — see [`dreams::close`].
    // Here, the one place the schema is built, so every conversation opened on
    // it — an ingest, a life episode, a probe — starts closed.
    let mut builder = builder;
    dreams::close(&mut builder);

    let (layer, group) = live_target(&builder)?;
    let prelude = prelude(&builder);
    let faults = boundary_faults(&builder);
    if !faults.is_empty() {
        tracing::warn!(
            "projection {}: {} authored section(s) do not end in exactly one blank line, so the \
             model reads them run into what follows or with a gap after: {}",
            path.display(),
            faults.len(),
            faults.join(", ")
        );
    }
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

/// The user turns a reflection sends, as the mind authors them.
///
/// **Content, not code.** The exact wording of these decides what a character
/// produces — every clause in them was arrived at by running the thing and
/// watching it fail without it — so they live in the mind beside the
/// `stance: reflecting` branch they run under, and are edited there.
///
/// They were briefly held as `const`s in [`crate::engine::reflect`] *and*
/// authored here, which is one prompt in two places: the two diverged within a
/// day, and the copy the engine was sending was not the copy anybody was
/// reading. There is now one of each.
pub struct ReflectionTurns {
    /// Asked first. Its answer goes back to the character.
    pub question_one: String,
    /// Asked second. Produces the dream brief and never reaches the character.
    /// Carries [`DOMAIN_SLOT`] and [`AXES_SLOT`], both filled by the engine.
    pub question_two: String,
    /// Sent only when the brief fails its structural checks.
    pub retry: String,
    /// The repair pass, in order, sent after the brief parses.
    ///
    /// **Not conditional on a fault.** These rewrite a brief that already passed
    /// every structural check, because the failure they address is a brief that
    /// is well-formed and is not a dream — which no cheap check can see. Empty
    /// when the mind authors no `repair` list, which leaves the old
    /// one-retry-and-no-more behaviour exactly as it was.
    ///
    /// Order is the author's and is load-bearing; see the block's own comment in
    /// `projection.yaml`.
    pub repair: Vec<String>,
}

/// Where the rotated behaviour-space cell goes in `question_two`.
///
/// **`[[...]]` and not `{...}`, because single braces belong to the schema
/// builder.** [`Builder::from_yaml_with_vars_and_dialect`] runs a `{ident}`
/// template pass over the whole file before parsing it and fails on any name it
/// was not given. A slot spelled `{domain}` — or `{{domain}}`, which contains one
/// — therefore does not fail the reflection. It fails **the entire projection**,
/// and [`build`] then hands back nothing, so every character in the daemon thinks
/// under a fallback schema that cannot gather. One ERROR line at startup is the
/// only symptom, and everything downstream keeps answering.
///
/// That is not hypothetical: it is what `{{domain}}` did here, and it stood for a
/// day of live runs whose gathers were all empty.
pub const DOMAIN_SLOT: &str = "[[domain]]";

/// Where the diversity steer goes in `question_two`. Same spelling rule as
/// [`DOMAIN_SLOT`], for the same reason.
pub const AXES_SLOT: &str = "[[sampled_axes]]";

/// The placeholders [`ReflectionTurns::question_two`] must carry.
///
/// Checked rather than assumed, because a missing one fails *silently* in the
/// worst way available: without [`DOMAIN_SLOT`] the rotation through the
/// behaviour space still runs, still reports which cell it chose, and has no
/// effect on what is generated — so the coverage the whole design rests on would
/// be a field in the response and nothing else.
const QUESTION_TWO_SLOTS: [&str; 2] = [DOMAIN_SLOT, AXES_SLOT];

/// Read the reflection's turns out of the mind's `projection.yaml`.
///
/// `None` when there is no mind, no `reflection` block, or a turn is missing or
/// malformed — and reflection is then unavailable rather than falling back to
/// something built in, because a built-in copy is the divergence
/// [`ReflectionTurns`] exists to have removed. The route says so.
///
/// Read straight from the YAML rather than through [`Builder`] for the same
/// reason as [`turn_retention`]: these are user turns the engine sends, not
/// system-prompt composition, and nothing in the schema's own vocabulary would
/// ever read them.
pub fn reflection(mind: Option<&Path>) -> Option<ReflectionTurns> {
    let path = mind?.join("projection.yaml");
    let yaml = std::fs::read_to_string(&path).ok()?;
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let block = doc.get("reflection")?;

    let turn = |key: &str| -> Option<String> {
        let text = block.get(key).and_then(|v| v.as_str()).unwrap_or_default();
        let text = text.trim();
        if text.is_empty() {
            tracing::error!(
                "reflection.{key} is missing or empty in {} — reflection is unavailable until it \
                 is authored",
                path.display()
            );
            return None;
        }
        Some(text.to_string())
    };

    let question_one = turn("question_one")?;
    let question_two = turn("question_two")?;
    let retry = turn("question_two_retry")?;

    for slot in QUESTION_TWO_SLOTS {
        if !question_two.contains(slot) {
            tracing::error!(
                "reflection.question_two in {} carries no `{slot}` — the engine fills it, and \
                 without it the value is computed and discarded",
                path.display()
            );
            return None;
        }
    }

    // A list of prose turns. A malformed entry is dropped rather than failing the
    // block: the repair pass degrades one question at a time, and a reflection
    // with two good repairs is worth more than none.
    let repair: Vec<String> = block
        .get("repair")
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    tracing::info!(
        "reflection: two authored turns, a retry and {} repair question(s) from {}",
        repair.len(),
        path.display()
    );
    Some(ReflectionTurns {
        question_one,
        question_two,
        retry,
        repair,
    })
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
    // **Exactly what the model reads, in the order it reads it.**
    //
    // Each piece is the content the projection hands the model for it, byte for
    // byte — the dialect header included, and glue markers, which are prefilled
    // live rather than sealed but are read all the same. The model is given each
    // section's content as written, with nothing between one and the next.
    //
    // This used to trim every piece and join them with a blank line, which fixed
    // the layout here and nowhere else: the copy showed a blank line after
    // `<|im_start|>system` that the model never saw, and hid two selector
    // options running together on consecutive lines, which it did. A copy that
    // tidies what it reports is a copy that cannot show the fault it tidied.
    // Layout is the mind's to get right; [`boundary_faults`] says when it is not.
    let mut out = String::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => out.push_str(&s.content),
            SystemPromptItem::SectionTree(t) => {
                for n in &t.nodes {
                    if n.collection.is_some() {
                        return out;
                    }
                    let option = if n.glue.is_some() {
                        0
                    } else {
                        n.chosen(&t.default_selection)
                    };
                    out.push_str(&n.options[option].content);
                }
            }
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}

/// Everything a conversation opened under this schema is framed by, as one
/// string to fingerprint — see `mind::frame_fingerprint`.
///
/// **The authored YAML and every member installed into it.** [`prelude`] is
/// only the text before the first collection, and a fingerprint of that missed
/// everything a character reads after it: the acts, the frame for acting, who
/// everybody is. The acts and the call format changed under a cast whose
/// conversations went on being rejoined, each of them reading a history written
/// for a prompt it no longer had.
///
/// Members are listed by collection and name in declaration order, so the same
/// schema installed the same way is the same string on every start.
pub fn frame(builder: &Builder) -> String {
    use candle_conversation::projection::SystemPromptItem;
    let mut out = builder.source_yaml().unwrap_or_default().to_string();
    let collections = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .flat_map(|item| match item {
            SystemPromptItem::Collection(c) => vec![c],
            SystemPromptItem::SectionTree(t) => t
                .nodes
                .iter()
                .filter_map(|n| n.collection.as_ref())
                .map(|tc| &tc.collection)
                .collect(),
            SystemPromptItem::Section(_) => Vec::new(),
        });
    for c in collections {
        for s in &c.sections {
            // NUL-separated: no authored text contains one, so two different
            // installs cannot run together into the same string.
            out.push_str(&format!("\0{}\0{}\0{}", c.name, s.name, s.content));
        }
    }
    out
}

/// Authored sections whose content does not end in exactly one blank line.
///
/// **The model reads each section's content verbatim, with nothing between one
/// and the next**, so the line feeds a section ends with are the only
/// separation it gets — and a YAML `|+` block ends with however many the author
/// happened to leave before the next key. Eight selector options had no blank
/// line after them, so the frame the model read ran `…what you are already
/// certain of.` straight into `Do what the moment actually calls for:` on the
/// next line, as one paragraph, in every conversation the daemon held.
///
/// Templates and glue are structural text and carry their own newlines;
/// collection members are separated by their collection's glue. What is left is
/// prose somebody wrote, and that is what this checks. Logged rather than
/// refused: a layout slip costs a paragraph break, not the schema.
fn boundary_faults(builder: &Builder) -> Vec<String> {
    use candle_conversation::projection::SystemPromptItem;
    let ends_right = |c: &str| c.is_empty() || (c.ends_with("\n\n") && !c.ends_with("\n\n\n"));
    let mut faults = Vec::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) if !s.is_template => {
                if !ends_right(&s.content) {
                    faults.push(s.name.clone());
                }
            }
            SystemPromptItem::SectionTree(t) => {
                for n in t
                    .nodes
                    .iter()
                    .filter(|n| n.glue.is_none() && n.collection.is_none())
                {
                    for o in n.options.iter().filter(|o| !ends_right(&o.content)) {
                        faults.push(match n.options.len() {
                            1 => n.name.clone(),
                            _ => format!("{}.{}", n.name, o.id),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    faults
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_mind_means_no_projection() {
        assert!(build(None, "world").is_none());
    }

    /// **The frame covers what is installed after the YAML is read**, which is
    /// the part the old fingerprint — the prelude — could not see. The same
    /// install is the same frame, so a restart that changed nothing rejoins; an
    /// act whose entry changed is a different frame, so it does not.
    #[test]
    fn the_frame_covers_what_is_installed_after_the_yaml() {
        let yaml = "system_prompt:\n  items:\n    - kind: section\n      id: frame\n      \
                    content: hello\n    - kind: collection\n      name: tools\n      \
                    selection: { kind: always_visible }\n      sections: []\nlayers: []\n";
        let installed = |entry: &str| {
            let mut b = Builder::from_yaml(yaml).unwrap();
            let cid = b.id_for_system_collection("tools").unwrap();
            b.add_section_to_collection(cid, "tools/say", entry, 100.0)
                .unwrap();
            frame(&b)
        };
        assert_eq!(installed("say(intent)"), installed("say(intent)"));
        assert_ne!(installed("say(intent)"), installed("say(intent, manner?)"));
        assert_ne!(
            frame(&Builder::from_yaml(yaml).unwrap()),
            installed("say(intent)")
        );
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

    #[test]
    fn no_mind_means_no_reflection_turns() {
        assert!(reflection(None).is_none());
    }

    /// **The regression this guards cost a day of live runs.** The engine's slots
    /// sit inside the same YAML the builder templates, so a slot spelled in the
    /// builder's `{ident}` syntax does not break the reflection — it breaks the
    /// whole projection, and every character in the daemon then thinks under a
    /// fallback schema that cannot gather, with one ERROR line as the only sign.
    ///
    /// Asserted against the real builder rather than by inspecting the strings,
    /// because what counts as its syntax is its business and may widen.
    #[test]
    fn the_reflection_slots_survive_the_builders_template_pass() {
        let yaml = format!(
            "reflection:\n  question_one: what comes to you?\n  question_two: |+\n    \
             from {DOMAIN_SLOT}\n    {AXES_SLOT}\n  question_two_retry: again\n\
             system_prompt:\n  items:\n    - kind: section\n      id: frame\n      content: hello\n\
             layers: []\n"
        );
        let dialect = DialectType::ChatML.dialect();
        let built =
            Builder::from_yaml_with_vars_and_dialect(&yaml, &[("workspace", "w")], Some(&dialect));
        assert!(
            built.is_ok(),
            "the reflection slots collide with the schema builder's own template syntax, which \
             fails the whole projection and not just the reflection: {:?}",
            built.err()
        );
    }

    /// **The silent failure this reader exists to make loud.** A `question_two`
    /// the engine cannot fill still runs, still reports the domain it rotated to,
    /// and generates against none of it — so the block is refused instead.
    #[test]
    fn a_question_two_that_cannot_be_filled_is_refused() {
        let dir = std::env::temp_dir().join("npcd-reflection-slots");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(
            dir.join("projection.yaml"),
            format!(
                "reflection:\n  question_one: what comes to you?\n  \
                 question_two: describe a dream from {DOMAIN_SLOT}\n  question_two_retry: again\n"
            ),
        )
        .unwrap();
        assert!(
            reflection(Some(&dir)).is_none(),
            "{AXES_SLOT} is absent, so the diversity steer would be computed and discarded"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_authored_reflection_block_is_read_whole() {
        let dir = std::env::temp_dir().join("npcd-reflection-ok");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(
            dir.join("projection.yaml"),
            format!(
                "reflection:\n  question_one: |+\n    what comes to you?\n  question_two: |+\n    \
                 from {DOMAIN_SLOT}\n    {AXES_SLOT}\n  question_two_retry: |+\n    again\n"
            ),
        )
        .unwrap();
        let t = reflection(Some(&dir)).expect("a complete block is read");
        assert_eq!(t.question_one, "what comes to you?");
        assert!(t.question_two.contains(DOMAIN_SLOT));
        assert_eq!(t.retry, "again");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Reading the schema bundled beside this crate answers rather than
    /// panicking, whatever it holds. The fixture is an older copy of the mind's
    /// schema and may predate the `reflection` block, which is exactly the case a
    /// deployment hits after a mind is rolled back.
    #[test]
    fn reading_the_bundled_schema_does_not_panic() {
        let mind = Path::new(env!("CARGO_MANIFEST_DIR"));
        if !mind.join("projection.yaml").is_file() {
            return; // no bundled schema in this checkout
        }
        // Whatever the fixture says, the reader must answer rather than panic.
        let _ = reflection(Some(mind));
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

    /// A schema holding exactly `items`, built the way [`build`] builds one.
    fn schema(items: &str) -> Builder {
        let yaml = format!("system_prompt:\n  items:\n{items}layers: []\n");
        let dialect = DialectType::ChatML.dialect();
        Builder::from_yaml_with_vars_and_dialect(&yaml, &[("workspace", "w")], Some(&dialect))
            .expect("the fixture schema parses")
    }

    /// **A `|+` block ends with whatever the author left before the next key.**
    ///
    /// With no blank line after it the block keeps one newline, and the model
    /// reads it run straight into the next section; with one blank line it keeps
    /// two. Asserted through the real parser because the whole fix rests on YAML
    /// behaving this way, and a guess about chomping is exactly what failed.
    #[test]
    fn a_block_with_no_blank_line_after_it_is_named() {
        let b = schema(
            "    - kind: section\n      id: tight\n      content: |+\n        one line\n\
             \x20   - kind: section\n      id: spaced\n      content: |+\n        one line\n\n",
        );
        assert_eq!(boundary_faults(&b), vec!["tight".to_string()]);
    }

    /// The case that shipped: selector options, where every option but the last
    /// is followed directly by the next `- id:`.
    #[test]
    fn a_selector_option_with_no_blank_line_after_it_is_named() {
        let b = schema(
            "    - kind: section_tree\n      nodes:\n        - kind: selector\n          id: dial\n\
             \x20         default: a\n          options:\n            - id: a\n\
             \x20             content: |+\n                first\n            - id: b\n\
             \x20             content: |+\n                second\n\n",
        );
        assert_eq!(boundary_faults(&b), vec!["dial.a".to_string()]);
    }

    /// Two blank lines is a gap, not a separation.
    #[test]
    fn a_block_ending_in_two_blank_lines_is_named() {
        let b =
            schema("    - kind: section\n      id: gap\n      content: |+\n        one line\n\n\n");
        assert_eq!(boundary_faults(&b), vec!["gap".to_string()]);
    }

    /// **The copy is what the model reads, byte for byte.** The header is the
    /// template's own text — one newline, no blank line after it — and a section
    /// arrives exactly as written. This used to trim and re-join every piece, so
    /// the copy showed a blank line after `<|im_start|>system` that the model
    /// never saw.
    #[test]
    fn the_prelude_is_the_raw_composition_and_templates_are_not_checked() {
        let b = schema(
            "    - kind: template\n      id: system_open\n      dialect: system_start\n\
             \x20   - kind: section\n      id: frame\n      content: |+\n        hello\n\n",
        );
        assert_eq!(prelude(&b), "<|im_start|>system\nhello\n\n");
        assert!(
            boundary_faults(&b).is_empty(),
            "the header is structural text"
        );
    }
}
