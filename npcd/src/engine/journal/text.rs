//! The turns behind a character's journal entries, kept in the substrate for
//! its `journal_read` and `journal_grep` (`docs/zen_moon_mind.md` §16) —
//! `host::journal_text`, owned by the character, since its journal outlives
//! each conversation it is carried in, and read in its own style.

use candle_conversation::persistence::SharedSubstrate;
use host::journal_text::JournalText;

use crate::engine::journal::style::NPC;

/// The owner a character's journal text is kept under.
pub fn owner(npc_id: u64) -> String {
    format!("npc-{npc_id}")
}

/// Character `npc_id`'s journal text.
pub fn of(shared: SharedSubstrate, npc_id: u64) -> JournalText {
    JournalText::of(shared, &owner(npc_id), &NPC)
}

#[cfg(test)]
mod tests {
    use journal_core::transcript::{Filed, Miss, Part};
    use tempfile::TempDir;

    use super::*;
    use crate::npcs::Npcs;

    /// **A character's turns are its own**: filed under its entry, they read
    /// back for it and for no other character.
    #[test]
    fn a_characters_turns_are_its_own() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let zen = of(shared.clone(), 7);
        zen.file(&Filed {
            entry: 3,
            parts: vec![Part {
                id: 1,
                at_ms: 0,
                text: "John asked about the weather in Sydney.".into(),
            }],
        })
        .unwrap();
        let page = zen.read(3, 1, 10).unwrap();
        assert_eq!(page.total, 1);
        assert!(
            page.text
                .ends_with("John asked about the weather in Sydney."),
            "{}",
            page.text
        );
        assert_eq!(of(shared, 70).read(3, 1, 10), Err(Miss::Entry(3)));
    }
}
