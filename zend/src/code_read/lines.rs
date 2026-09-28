//! A file's line count, as `file_read` reports it.

/// How many lines `file_read` reports for `bytes`: one per `\n`, and one
/// more for a last line that has none.
pub fn line_count(bytes: &[u8]) -> usize {
    let ended = bytes.iter().filter(|b| **b == b'\n').count();
    ended + usize::from(bytes.last().is_some_and(|b| *b != b'\n'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counts `file_read`'s paging gives the same bytes.
    #[test]
    fn lines_are_counted_as_file_read_counts_them() {
        assert_eq!(line_count(b""), 0);
        assert_eq!(line_count(b"\n"), 1);
        assert_eq!(line_count(b"a"), 1);
        assert_eq!(line_count(b"a\n"), 1);
        assert_eq!(line_count(b"a\nb"), 2);
        assert_eq!(line_count(b"a\r\nb\r\n"), 2);
        assert_eq!(line_count(b"a\n\n"), 2);
    }
}
