//! How much of a page fits in what a reader takes in one answer. A page
//! counts lines, and a line holds up to 2,000 characters, so a page of
//! them can be longer than the answer that carries it; a page cut by its
//! carrier loses its last line, which is where it says how to read on. Each
//! page is cut here instead, to the lines that fit, and still names where
//! the next one begins.

/// What a page's closing line — how many more, and the cursor to page on
/// with — is kept room for.
pub const CURSOR_ROOM: usize = 400;

/// Characters `text` takes, with the line break that follows it.
pub fn taken(text: &str) -> usize {
    text.chars().count() + 1
}

/// How many of `lines`, from the first, fit in `room` characters, each with
/// its line break: at least one when there is any, so a page always moves.
pub fn fitting<S: AsRef<str>>(lines: &[S], room: usize) -> usize {
    let mut used = 0;
    let mut n = 0;
    for l in lines {
        used += taken(l.as_ref());
        if used > room && n > 0 {
            break;
        }
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_fit_from_the_first_and_a_page_always_moves() {
        let lines = ["abcd", "efgh", "ij"];
        assert_eq!(fitting(&lines, 15), 3, "5 + 5 + 3");
        assert_eq!(fitting(&lines, 12), 2);
        assert_eq!(fitting(&lines, 10), 2);
        assert_eq!(fitting(&lines, 9), 1);
        assert_eq!(fitting(&lines, 0), 1, "one line, however long");
        assert_eq!(fitting::<&str>(&[], 0), 0);
        assert_eq!(taken("é"), 2, "characters, not bytes");
    }
}
