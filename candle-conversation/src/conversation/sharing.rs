//! Whose a submitted section is: the conversation's own, or shared by every
//! conversation that would compute exactly it
//! (`docs/shared_journal_sections.md`).

/// The `debug_name` prefix a shared section's stream is declared under — what
/// tells a shared stream apart from an owned one (`conv/<owner>/…`).
pub const SHARED_SECTION_NAME_PREFIX: &str = "shared/";

/// Whose a section of a chained collection is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sharing {
    /// The conversation's own: addressed under its owner, ended with it.
    #[default]
    Owned,
    /// No conversation's own: addressed by content alone, restored by every
    /// conversation that would seal exactly it, and never ended by one leaving.
    Shared,
    /// [`Self::Shared`], and the last of its shared run: its seal also keeps the
    /// state it ends on, for the owned section a chain continues it with.
    SharedEnd,
}

impl Sharing {
    /// Whether the section is addressed by content alone.
    pub fn is_shared(self) -> bool {
        !matches!(self, Sharing::Owned)
    }

    /// Whether the section's seal keeps the state it ends on.
    pub fn keeps_end_state(self) -> bool {
        matches!(self, Sharing::SharedEnd)
    }
}

/// The `debug_name` a shared section named `name` is declared under.
pub fn shared_stream_name(name: &str) -> String {
    format!("{SHARED_SECTION_NAME_PREFIX}{name}")
}

/// Whether a stream declared under `debug_name` is a shared section's.
pub fn is_shared_stream_name(debug_name: &str) -> bool {
    debug_name.starts_with(SHARED_SECTION_NAME_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_shared_kinds_are_shared_and_only_the_run_end_keeps_state() {
        assert_eq!(
            [Sharing::Owned, Sharing::Shared, Sharing::SharedEnd]
                .map(|s| (s.is_shared(), s.keeps_end_state())),
            [(false, false), (true, false), (true, true)]
        );
        assert_eq!(Sharing::default(), Sharing::Owned);
    }

    #[test]
    fn a_shared_stream_is_told_apart_by_its_name() {
        assert_eq!(shared_stream_name("journal/1"), "shared/journal/1");
        assert!(is_shared_stream_name("shared/lineage/3"));
        assert!(!is_shared_stream_name("conv/77/lineage/3"));
        assert!(!is_shared_stream_name("tool_catalog"));
    }
}
