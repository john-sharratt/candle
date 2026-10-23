//! Which conversation holds a character's journal.
//!
//! `docs/journal.md` §6. A character's journal is a collection of sections of its
//! **own conversation** ([`journal_core::section`]), read under `journal_intro`
//! while a character that has written nothing reads `journal_none`. A character
//! starts a new conversation with each mission, so the conversation holding its
//! journal's record is found by whose it is ([`META_OF`]), not by name.

/// Conversation metadata: whose journal a conversation holds.
pub const META_OF: &str = "journal.of";

#[cfg(test)]
mod tests {
    use journal_core::section::JOURNAL as CORE_JOURNAL;

    use crate::engine::identity::JOURNAL;

    /// The collection the projection reads the journal under is the one the core
    /// names its sections for.
    #[test]
    fn the_projection_collection_is_the_cores() {
        assert_eq!(JOURNAL, CORE_JOURNAL);
    }
}
