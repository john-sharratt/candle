//! The names the knowledge base's members are registered under
//! (`docs/provenance_system_collection.md` §12 D15): the ingest seals each
//! section under one, and the projection finds the member's section by it.

use super::key::LineSpan;

/// The section member holding `lines` of the file at `path`, blob `blob`.
pub fn chunk_name(path: &str, blob: &str, lines: LineSpan) -> String {
    format!("{path}@{blob}#{}-{}", lines.first, lines.last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_member_is_named_by_its_file_and_its_lines() {
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
