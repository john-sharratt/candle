//! The names the knowledge base's members are registered under
//! (`docs/provenance_system_collection.md` §12 D15): the ingest seals each
//! member under one, and the projection finds the member's section by it.

use super::key::LineSpan;

/// The heading member of the file at `path`, blob `blob`.
pub fn heading_name(path: &str, blob: &str) -> String {
    format!("{path}@{blob}#heading")
}

/// The chunk member holding `lines` of the file at `path`, blob `blob`.
pub fn chunk_name(path: &str, blob: &str, lines: LineSpan) -> String {
    format!("{path}@{blob}#{}-{}", lines.first, lines.last)
}

/// The shared gap piece.
pub const GAP_NAME: &str = "pieces#gap";
/// The shared closing piece of a file another file follows.
pub const CLOSING_JOIN_NAME: &str = "pieces#closing_join";
/// The shared closing piece of the last file.
pub const CLOSING_LAST_NAME: &str = "pieces#closing_last";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_member_is_named_by_its_file_and_its_lines() {
        assert_eq!(
            heading_name("candle/a.rs", "ab12"),
            "candle/a.rs@ab12#heading"
        );
        assert_eq!(
            chunk_name(
                "candle/a.rs",
                "ab12",
                LineSpan {
                    first: 201,
                    last: 400
                }
            ),
            "candle/a.rs@ab12#201-400"
        );
    }
}
