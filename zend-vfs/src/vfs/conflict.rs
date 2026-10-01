//! The conflict markers a merge leaves in a file.

/// The line that opens a conflict's own side.
const OPENS: &str = "<<<<<<< ";
/// The line that closes a conflict's other side.
const CLOSES: &str = ">>>>>>> ";

/// Whether `text` still holds a conflict a merge marked: a line opening one
/// and a line closing one.
pub fn has_markers(text: &str) -> bool {
    text.lines().any(|l| l.starts_with(OPENS)) && text.lines().any(|l| l.starts_with(CLOSES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marked_conflict_is_found_and_a_settled_file_is_not() {
        assert!(has_markers(
            "a\n<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> origin/main\nc\n"
        ));
        assert!(!has_markers("a\nmine\ntheirs\nc\n"));
        assert!(
            !has_markers("<<<<<<< only an opening line\n"),
            "one line alone is not a conflict"
        );
        assert!(!has_markers("  <<<<<<< indented\n  >>>>>>> indented\n"));
    }
}
