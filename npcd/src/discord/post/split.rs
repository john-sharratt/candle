//! A message cut into posts Discord will take (`docs/npcd_discord.md` §5.1):
//! at most [`LIMIT`] characters each, cut after a sentence where one ends in
//! reach, else at a space, else where the limit falls.

/// The most characters Discord takes in one message.
pub const LIMIT: usize = 2000;

/// `text` as posts of at most `limit` characters, in order, none empty.
pub fn split(text: &str, limit: usize) -> Vec<String> {
    let mut posts = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        let Some((limit_at, past)) = rest.char_indices().nth(limit) else {
            posts.push(rest.to_string());
            break;
        };
        // One character past the limit is looked at too, so a break that
        // falls exactly at the limit is found.
        let seen = &rest[..limit_at + past.len_utf8()];
        let cut = sentence_end(seen, limit_at)
            .or_else(|| seen.rfind(char::is_whitespace))
            .filter(|&at| at > 0)
            .unwrap_or(limit_at);
        posts.push(rest[..cut].trim_end().to_string());
        rest = rest[cut..].trim_start();
    }
    posts
}

/// Where the last sentence that ends in `seen` at or before byte `max`
/// ends: just past its `.`, `!` or `?` where a space follows, or at a line
/// break.
fn sentence_end(seen: &str, max: usize) -> Option<usize> {
    let mut found = None;
    let mut chars = seen.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        let next = chars.peek().map(|(_, n)| *n);
        let end = match c {
            '\n' => Some(at),
            '.' | '!' | '?' if next.is_some_and(char::is_whitespace) => Some(at + 1),
            _ => None,
        };
        if let Some(end) = end.filter(|&end| end <= max) {
            found = Some(end);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_message_is_one_post() {
        assert_eq!(split("  hello there  ", LIMIT), ["hello there"]);
        assert_eq!(split("", LIMIT), Vec::<String>::new());
        assert_eq!(split(&"a".repeat(LIMIT), LIMIT), ["a".repeat(LIMIT)]);
    }

    /// **Cut after the last sentence that fits**, the space after it not
    /// carried into either post.
    #[test]
    fn a_long_message_is_cut_after_a_sentence() {
        assert_eq!(
            split("One two. Three four! Five six seven.", 22),
            ["One two. Three four!", "Five six seven."]
        );
        assert_eq!(
            split("First line\nsecond line that runs on", 20),
            ["First line", "second line that", "runs on"]
        );
    }

    /// No sentence ends in reach: at a space; no space either: where the
    /// limit falls.
    #[test]
    fn with_no_sentence_in_reach_it_cuts_at_a_space_then_anywhere() {
        assert_eq!(split("aaaa bbbb cccc", 9), ["aaaa bbbb", "cccc"]);
        assert_eq!(split("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        // `e.g.` with no space after it is not the end of a sentence.
        assert_eq!(split("x e.g.y zzzz", 8), ["x e.g.y", "zzzz"]);
    }

    /// **Characters, not bytes**: a post of wide characters is cut at the
    /// limit's count of them, never inside one.
    #[test]
    fn the_limit_counts_characters() {
        let wide = "日".repeat(LIMIT + 5);
        let posts = split(&wide, LIMIT);
        assert_eq!(
            posts.iter().map(|p| p.chars().count()).collect::<Vec<_>>(),
            [LIMIT, 5]
        );
        let long = format!("{}. {}", "a".repeat(1500), "b".repeat(1000));
        let posts = split(&long, LIMIT);
        assert_eq!(posts, [format!("{}.", "a".repeat(1500)), "b".repeat(1000)]);
    }
}
