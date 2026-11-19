//! Which of a conversation's journal sections are shared
//! (`docs/shared_journal_sections.md` §2).
//!
//! A section is shared when its text is no conversation's own — an ancestor's
//! journal, a memory topic, the entries' heading, the opening, the close with
//! nothing open — **and so is every section before it**: a chained section's
//! address includes what it continues, so one after an owned section is that
//! conversation's alone whatever its text. The leading run of shared sections
//! is the chain's shared run; its last keeps the state it ends on when a
//! section of the conversation's own continues it.

use candle_conversation::Sharing;

/// The sharing of each section of a chain whose sections are `shareable` as
/// marked, in chain order. The run's last section is [`Sharing::SharedEnd`]
/// when an owned section follows it in this chain, or when `followed` says one
/// will in the conversations the run is sealed for.
pub fn chain_sharing(shareable: &[bool], followed: bool) -> Vec<Sharing> {
    let run = shareable.iter().take_while(|&&s| s).count();
    let ends = run > 0 && (run < shareable.len() || followed);
    (0..shareable.len())
        .map(|i| match i {
            i if i >= run => Sharing::Owned,
            i if ends && i + 1 == run => Sharing::SharedEnd,
            _ => Sharing::Shared,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharing::{Owned, Shared, SharedEnd};

    /// A thread conversation's opening: lineage, history, both opening entries
    /// and the close are shared, the reader's line owned, and the close ends the
    /// run.
    #[test]
    fn a_new_thread_conversation_shares_all_but_its_own_line() {
        assert_eq!(
            chain_sharing(&[true, true, true, true, true, false], false),
            [Shared, Shared, Shared, Shared, SharedEnd, Owned]
        );
    }

    /// A drafted entry makes everything after it the conversation's, the close
    /// among them, whatever its text.
    #[test]
    fn everything_after_an_owned_section_is_owned() {
        assert_eq!(
            chain_sharing(&[true, true, false, true, false], false),
            [Shared, SharedEnd, Owned, Owned, Owned]
        );
    }

    /// A threadless conversation's journal is shared whole and keeps no end
    /// state — nothing of its own follows — unless the run is sealed for
    /// conversations that continue it.
    #[test]
    fn a_wholly_shared_chain_keeps_its_end_only_when_followed() {
        assert_eq!(
            chain_sharing(&[true, true, true], false),
            [Shared, Shared, Shared]
        );
        assert_eq!(
            chain_sharing(&[true, true, true], true),
            [Shared, Shared, SharedEnd]
        );
    }

    #[test]
    fn a_chain_owned_from_its_start_shares_nothing() {
        assert_eq!(chain_sharing(&[false, true], true), [Owned, Owned]);
        assert!(chain_sharing(&[], true).is_empty());
    }
}
