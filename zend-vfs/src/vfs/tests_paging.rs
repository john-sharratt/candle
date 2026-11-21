//! Tests for page arithmetic.
//!
//! Each test line carries its own line number, so a page's content can be checked
//! without re-deriving it. The tests pin a 650-line file at 200-line strides, the
//! clamp that reads an over-shot request as the tail, the empty file as page 0 of
//! 0, and a missing path giving no page. An oversize or non-UTF-8 workspace file
//! refuses to page, and the upper layer pages the same way the workspace does.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_upper::tests::{put, store_with_tree};

    // ── read_page ────────────────────────────────────────────────────────────

    /// `n` lines, each its own line number so a page's content is checkable
    /// without re-deriving it — line `k` (1-based) is the string `k`.
    pub(in super::super) fn numbered_lines(n: u32) -> String {
        (1..=n)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn a_short_file_is_entirely_page_0() {
        let (_dir, s) = store_with_tree();
        let p = s.read_page("src/main.rs", 0).unwrap().unwrap();
        assert_eq!(p.page, 0);
        assert_eq!(p.total_pages, 1);
        assert_eq!(p.start_line, 1);
        assert_eq!(p.end_line, 1);
        assert_eq!(p.total_lines, 1);
        assert_eq!(p.body, "fn main() {}");
    }

    #[test]
    fn a_file_of_exactly_650_lines_pages_at_200_line_strides() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "big.txt", &numbered_lines(650));

        let p0 = s.read_page("big.txt", 0).unwrap().unwrap();
        assert_eq!(p0.page, 0);
        assert_eq!(p0.total_pages, 4);
        assert_eq!(p0.start_line, 1);
        assert_eq!(p0.end_line, 200);
        assert_eq!(p0.total_lines, 650);
        assert_eq!(p0.body.lines().next(), Some("1"));
        assert_eq!(p0.body.lines().last(), Some("200"));

        let p1 = s.read_page("big.txt", 1).unwrap().unwrap();
        assert_eq!(p1.page, 1);
        assert_eq!(p1.start_line, 201);
        assert_eq!(p1.end_line, 400);
        assert_eq!(p1.body.lines().next(), Some("201"));
        assert_eq!(p1.body.lines().last(), Some("400"));

        let p3 = s.read_page("big.txt", 3).unwrap().unwrap();
        assert_eq!(p3.page, 3);
        assert_eq!(p3.start_line, 601);
        assert_eq!(p3.end_line, 650);
        assert_eq!(p3.body.lines().next(), Some("601"));
        assert_eq!(p3.body.lines().last(), Some("650"));
    }

    /// A page past the end clamps to the last one — the same "over-shoot reads
    /// as the tail, not an error" rule `file_list`'s own paging applies.
    #[test]
    fn a_page_past_the_end_clamps_to_the_last_page() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "big.txt", &numbered_lines(650));
        let clamped = s.read_page("big.txt", 99).unwrap().unwrap();
        let last = s.read_page("big.txt", 3).unwrap().unwrap();
        assert_eq!(clamped.page, last.page);
        assert_eq!(clamped.start_line, last.start_line);
        assert_eq!(clamped.end_line, last.end_line);
        assert_eq!(clamped.body, last.body);
    }

    #[test]
    fn an_empty_file_reports_page_0_of_0() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "empty.txt", "");
        let p = s.read_page("empty.txt", 0).unwrap().unwrap();
        assert_eq!(p.page, 0);
        assert_eq!(p.start_line, 1);
        assert_eq!(p.end_line, 0);
        assert_eq!(p.total_lines, 0);
        assert_eq!(p.total_pages, 0);
        assert_eq!(p.body, "");
    }

    #[test]
    fn read_page_of_a_missing_path_is_none() {
        let (_dir, s) = store_with_tree();
        assert!(s.read_page("nope.rs", 0).unwrap().is_none());
    }

    /// The upper layer pages from the same in-memory string via a cursor —
    /// no second copy, but the same page math as the workspace layer.
    #[test]
    fn upper_layer_files_page_too() {
        let s = VfsStore::new();
        s.write("big.txt", numbered_lines(650)).unwrap();
        let p1 = s.read_page("big.txt", 1).unwrap().unwrap();
        assert_eq!(p1.start_line, 201);
        assert_eq!(p1.end_line, 400);
        assert_eq!(p1.total_pages, 4);
        assert_eq!(p1.body.lines().next(), Some("201"));
    }

    #[test]
    fn oversize_lower_file_refuses_to_page() {
        let (dir, s) = store_with_tree();
        let big = (MAX_LOWER_FILE_BYTES + 1) as usize;
        std::fs::write(dir.path().join("huge.txt"), vec![b'a'; big]).unwrap();
        let err = s.read_page("huge.txt", 0).unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");
    }

    #[test]
    fn non_utf8_lower_file_is_unreadable() {
        let (dir, s) = store_with_tree();
        std::fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00]).unwrap();
        let err = s.read("blob.bin").unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");
    }
}
