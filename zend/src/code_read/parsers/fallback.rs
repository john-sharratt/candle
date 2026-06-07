//! Fixed-window fallback carver.
//!
//! Used when no language-specific parser fires (or when one fails to
//! parse a file).  Splits the source into 100-line windows labelled
//! `chunk N`.

use crate::code_read::types::{ChunkKind, Scope};

pub const WINDOW_LINES: u32 = 100;

pub fn carve(source: &[u8]) -> Vec<Scope> {
    let text = match std::str::from_utf8(source) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let total = text.lines().count() as u32;
    if total == 0 {
        return Vec::new();
    }

    let mut scopes = Vec::new();
    let mut start = 1u32;
    let mut part = 1usize;
    while start <= total {
        let end = (start + WINDOW_LINES - 1).min(total);
        scopes.push(Scope {
            path: vec![format!("chunk {part}")],
            kind: ChunkKind::Fallback,
            start_line: start,
            end_line: end,
        });
        start = end + 1;
        part += 1;
    }
    scopes
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::WINDOW_LINES;
    use crate::code_read::test_util::{
        assert_content_preserved, assert_full_coverage, verify_coverage_only,
    };
    use crate::repo_scan::Language;

    fn verify_cov(src: &str) -> Vec<crate::code_read::types::Scope> {
        verify_coverage_only(src, Language::PlainText, false)
    }

    #[test]
    fn empty_input_emits_nothing() {
        assert!(super::carve(b"").is_empty());
    }

    #[test]
    fn small_file_emits_single_chunk() {
        let scopes = verify_cov("a\nb\nc\n");
        assert!(scopes.iter().any(|s| s.start_line == 1));
    }

    #[test]
    fn large_file_splits_into_windows() {
        let src: String = (0..250).map(|_| "x\n").collect();
        let scopes = super::carve(src.as_bytes());
        assert_eq!(scopes.len(), 3); // 100 + 100 + 50
        assert_eq!(scopes[0].start_line, 1);
        assert_eq!(scopes[0].end_line, 100);
        assert_eq!(scopes[1].start_line, 101);
        assert_eq!(scopes[1].end_line, 200);
        assert_eq!(scopes[2].start_line, 201);
        assert_eq!(scopes[2].end_line, 250);
        assert_content_preserved(src.as_bytes(), &scopes);
        assert_full_coverage(src.as_bytes(), &scopes);
    }

    #[test]
    fn exactly_window_size_file_is_single_chunk() {
        let src: String = (0..WINDOW_LINES).map(|_| "x\n").collect();
        let scopes = super::carve(src.as_bytes());
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].end_line, WINDOW_LINES);
        assert_full_coverage(src.as_bytes(), &scopes);
    }

    #[test]
    fn window_size_plus_one_splits_into_two_chunks() {
        let src: String = (0..WINDOW_LINES + 1).map(|_| "x\n").collect();
        let scopes = super::carve(src.as_bytes());
        assert_eq!(scopes.len(), 2);
        assert_eq!(scopes[0].end_line, WINDOW_LINES);
        assert_eq!(scopes[1].start_line, WINDOW_LINES + 1);
        assert_eq!(scopes[1].end_line, WINDOW_LINES + 1);
        assert_full_coverage(src.as_bytes(), &scopes);
    }

    #[test]
    fn fallback_covers_all_lines() {
        let src: String = (0..555).map(|_| "y\n").collect();
        let scopes = super::carve(src.as_bytes());
        assert_full_coverage(src.as_bytes(), &scopes);
        assert_content_preserved(src.as_bytes(), &scopes);
    }

    #[test]
    fn handles_invalid_utf8_without_panic() {
        let scopes = super::carve(&[0xff, 0xfe, 0xfd, b'\n']);
        // Invalid UTF-8 returns empty rather than panicking.
        assert!(scopes.is_empty());
    }

    #[test]
    fn handles_crlf_line_endings() {
        // Each `\r\n` is one logical line.
        let src: Vec<u8> = (0..150).flat_map(|_| b"x\r\n".to_vec()).collect();
        let scopes = super::carve(&src);
        assert_eq!(scopes.len(), 2);
        assert_full_coverage(&src, &scopes);
    }

    #[test]
    fn single_line_no_trailing_newline_is_one_chunk() {
        let src = b"hello";
        let scopes = super::carve(src);
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].end_line, 1);
        assert_full_coverage(src, &scopes);
    }
}
