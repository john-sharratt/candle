//! The journal as the dialogue's system prompt carries it.
//!
//! The schema declares a `journal` collection with no members of its own, and a
//! `journal_intro` that emits only while the collection does. The sections are
//! the conversation's: one per entry and one for the open items, submitted to
//! its own timeline ([`candle_conversation::Sequence::submit_section`]), sealed
//! and persisted with it, invisible to every other conversation and tombstoned
//! with it.
//!
//! The collection selects its members by name, and a round pins exactly the
//! sections that have sealed. A conversation that has written nothing — or
//! whose sections are all still sealing — pins none, so neither the collection
//! nor its intro emits: the prompt shows the journal as it last sealed, never a
//! heading with nothing under it. A new entry joins the prompt from the first
//! round after its section seals; until then the turns it was written from are
//! still in the conversation.

use candle_conversation::projection::{Builder, SelectionRule};
use candle_conversation::{SelectionState, Want};
use journal_core::section::{JournalPrompt, JOURNAL};

/// Scope the schema's `journal` collection to the members a conversation
/// names. `false` when the schema declares no journal: the dialogue then
/// journals nothing, which is not an error.
pub fn install(builder: &mut Builder) -> anyhow::Result<bool> {
    if builder.id_for_system_collection(JOURNAL).is_none() {
        return Ok(false);
    }
    builder
        .set_collection_selection(
            JOURNAL,
            SelectionRule::Named {
                selector: JOURNAL.to_string(),
            },
        )
        .map_err(|e| anyhow::anyhow!("scoping `{JOURNAL}` to its members: {e}"))?;
    Ok(true)
}

/// The sections `prompt` should be carried as.
pub fn wants(prompt: &JournalPrompt) -> Vec<Want<'_>> {
    prompt
        .sections
        .iter()
        .map(|s| Want {
            name: &s.name,
            text: &s.text,
        })
        .collect()
}

/// Pin the journal sections that have `sealed` for a round to read.
pub fn select(selection: &mut SelectionState, sealed: &[String]) {
    selection.select_all(JOURNAL, sealed.iter().map(String::as_str));
}

#[cfg(test)]
mod tests {
    use candle_conversation::models::Dialect;
    use candle_conversation::projection::SystemPromptItem;
    use journal_core::entry::Item;
    use journal_core::section::{Kept, OPEN_SECTION};

    use super::*;
    use crate::journal::style::CODING;
    use crate::session::PROJECTION_SCHEMA_TEMPLATE;

    fn schema() -> Builder {
        Builder::from_yaml_with_vars_and_dialect(
            PROJECTION_SCHEMA_TEMPLATE,
            &[("workspace", "demo")],
            Some(&Dialect::chat_ml()),
        )
        .unwrap()
    }

    /// The journal has no member of its own: it holds nothing but each
    /// conversation's sections.
    #[test]
    fn the_dialogue_schema_declares_a_memberless_journal_scoped_by_name() {
        let mut b = schema();
        assert!(install(&mut b).unwrap());
        assert!(b.id_for_system_section("journal_intro").is_some());
        assert!(
            b.schema().system_prompt.items.iter().any(|item| matches!(
                item,
                SystemPromptItem::Collection(c) if c.name == JOURNAL && c.sections.is_empty()
            )),
            "no member of its own"
        );
    }

    #[test]
    fn a_schema_with_no_journal_installs_nothing() {
        let yaml = PROJECTION_SCHEMA_TEMPLATE
            .replace("name: journal", "name: notes")
            .replace("depends_on: journal", "depends_on: notes");
        assert_ne!(yaml, PROJECTION_SCHEMA_TEMPLATE);
        let mut b = Builder::from_yaml_with_vars_and_dialect(
            &yaml,
            &[("workspace", "demo")],
            Some(&Dialect::chat_ml()),
        )
        .unwrap();
        assert!(!install(&mut b).unwrap());
    }

    /// A round pins exactly what has sealed; with nothing sealed it pins
    /// nothing, and the journal does not emit.
    #[test]
    fn a_round_pins_exactly_the_sealed_sections() {
        let mut sel = SelectionState::default();
        select(&mut sel, &[]);
        assert!(sel.members(JOURNAL).is_empty());
        let sealed = ["journal/3".to_string(), OPEN_SECTION.to_string()];
        select(&mut sel, &sealed);
        assert_eq!(sel.members(JOURNAL), ["journal/3", "journal/open"]);
        select(&mut sel, &[]);
        assert!(sel.members(JOURNAL).is_empty(), "replaced, not added to");
    }

    #[test]
    fn the_wanted_sections_are_the_prompts_in_order() {
        let mut kept = Kept {
            entries: vec![],
            open: vec![],
            next_item: 1,
        };
        assert!(wants(&JournalPrompt::of(&kept, &CODING)).is_empty());
        kept.open = vec![Item {
            id: 1,
            text: "rerun".into(),
        }];
        let prompt = JournalPrompt::of(&kept, &CODING);
        let w = wants(&prompt);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].name, OPEN_SECTION);
        assert_eq!(w[0].text, "Still open:\n#1 rerun");
    }
}
