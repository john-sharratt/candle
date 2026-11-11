//! A journal entry's file as a reader of the repository is shown it: the file
//! with its `by` and `channel` fields left out. A file read then names the
//! person a line is from (`account`, `speaker`) and never the app it was
//! written through — as every other view of a line does ([`super::who`]).

use crate::line::LineId;

/// Whether the file at `path` is a journal entry's: `<line id>.yaml`.
pub fn is_entry(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.strip_suffix(".yaml")
        .is_some_and(|id| LineId::parse(id).is_ok())
}

/// `text`, an entry file's, without its `by:` and `channel:` fields. An entry
/// is written one field to a line at the left margin (`Line::render`), so a
/// field is a line that begins with its name; a folded text's lines are
/// indented, and every one of them is kept.
pub fn shown(text: &str) -> String {
    text.split_inclusive('\n')
        .filter(|l| !(l.starts_with("by:") || l.starts_with("channel:")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_file_is_named_by_its_line_id() {
        assert!(is_entry("groups/battle-cities/2026-10-31-0039-4b84.yaml"));
        assert!(is_entry("2026-10-31-0039-4b84.yaml"));
        for not in [
            "groups/zen/journal.yaml",
            "groups/zen/2026-10-31-0039-4b84.yml",
            "groups/zen/2026-10-31-0039-4B84.yaml",
            "groups/zen/conversations/2026-11-08-0308-claude-code",
            "src/2026-10-31-0039-4b84.yaml.rs",
        ] {
            assert!(!is_entry(not), "{not}");
        }
    }

    /// **The writer and the channel go; every other field and the whole text
    /// stay**, folded lines and all, byte for byte.
    #[test]
    fn an_entry_is_shown_without_its_writer_or_channel() {
        assert_eq!(
            shown(
                "at: 2026-10-30T09:12:40Z\nkind: note\nby: claude-code\naccount: john\n\
                 text: >-\n  The candle fork powers two products: Zen Code, by: John,\n  \
                 and Battle Cities.\n"
            ),
            "at: 2026-10-30T09:12:40Z\nkind: note\naccount: john\n\
             text: >-\n  The candle fork powers two products: Zen Code, by: John,\n  \
             and Battle Cities.\n"
        );
        assert_eq!(
            shown(
                "at: 2026-10-30T09:12:40Z\r\nkind: episode\r\nby: server\r\naccount: john\r\n\
                 channel: grok\r\ntext: by: hand\r\n"
            ),
            "at: 2026-10-30T09:12:40Z\r\nkind: episode\r\naccount: john\r\ntext: by: hand\r\n"
        );
        assert_eq!(
            shown("at: 2026-11-10T09:12:40Z\nkind: episode\nby: npcd\ntext: x"),
            "at: 2026-11-10T09:12:40Z\nkind: episode\ntext: x"
        );
    }
}
