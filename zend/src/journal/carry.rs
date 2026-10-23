//! The journal as the dialogue's system prompt carries it.
//!
//! The schema declares a `journal` collection with no members of its own, and a
//! `journal_intro` that emits only while the collection does. The sections are
//! the conversation's: one per entry and one for the open items, submitted to
//! its own timeline ([`candle_conversation::Sequence::submit_section`]), sealed
//! and persisted with it, invisible to every other conversation and tombstoned
//! with it.
//!
//! The collection selects its members by name. A conversation that has written
//! nothing pins none, so neither the collection nor its intro emits, and every
//! other layer's prompt reads exactly as it would with no journal declared. One
//! that has written something pins the sections that have sealed, or the
//! [`STAND_IN`] member while none has.

use candle_conversation::projection::{Builder, SelectionRule};
use candle_conversation::{SelectionState, Want};
use journal_core::section::{JournalPrompt, JOURNAL};

/// The schema member read under the journal's intro while the conversation's
/// own sections are still sealing.
pub const STAND_IN: &str = "journal/carrying";

const STAND_IN_TEXT: &str = "Entries you wrote down earlier in this session, which are being \
brought back to you.";

/// Scope the schema's `journal` collection to the members a conversation names,
/// and give it its stand-in. `false` when the schema declares no journal: the
/// dialogue then journals nothing, which is not an error.
pub fn install(builder: &mut Builder) -> anyhow::Result<bool> {
    let Some(collection) = builder.id_for_system_collection(JOURNAL) else {
        return Ok(false);
    };
    builder
        .set_collection_selection(
            JOURNAL,
            SelectionRule::Named {
                selector: JOURNAL.to_string(),
            },
        )
        .map_err(|e| anyhow::anyhow!("scoping `{JOURNAL}` to its members: {e}"))?;
    builder
        .add_section_to_collection(collection, STAND_IN.to_string(), STAND_IN_TEXT, 100.0)
        .map_err(|e| anyhow::anyhow!("installing `{STAND_IN}`: {e}"))?;
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

/// Pin what a turn reads of the journal: nothing while nothing is written, the
/// sections that have `sealed` once something is, or the stand-in until one has.
pub fn select(selection: &mut SelectionState, written: bool, sealed: &[String]) {
    if !written {
        return;
    }
    if sealed.is_empty() {
        selection.select(JOURNAL, STAND_IN.to_string());
    } else {
        selection.select_all(JOURNAL, sealed.iter().map(String::as_str));
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::models::Dialect;
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

    #[test]
    fn the_dialogue_schema_declares_a_journal_scoped_by_name_with_its_stand_in() {
        let mut b = schema();
        assert!(install(&mut b).unwrap());
        assert!(b.id_for_system_section(STAND_IN).is_some());
        assert!(b.id_for_system_section("journal_intro").is_some());
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
        assert!(b.id_for_system_section(STAND_IN).is_none());
    }

    #[test]
    fn nothing_written_pins_nothing() {
        let mut sel = SelectionState::default();
        select(&mut sel, false, &[]);
        assert!(sel.members(JOURNAL).is_empty());
    }

    #[test]
    fn a_written_journal_reads_its_stand_in_until_a_section_seals() {
        let mut sel = SelectionState::default();
        select(&mut sel, true, &[]);
        assert_eq!(sel.members(JOURNAL), [STAND_IN]);
        let sealed = ["journal/3".to_string(), OPEN_SECTION.to_string()];
        select(&mut sel, true, &sealed);
        assert_eq!(sel.members(JOURNAL), ["journal/3", "journal/open"]);
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
