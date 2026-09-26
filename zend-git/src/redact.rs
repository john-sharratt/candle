//! Credentials embedded in URLs, removed from anything the layer reports.
//!
//! A remote URL may carry a user and token (`https://user:ghp_x@host/repo`).
//! Git echoes remote URLs in its errors, and `remotes()` returns them; either
//! could carry the token into a log, an error shown to a user, or a tool
//! result the model reads. Every such string passes through [`redact_urls`].

/// `text` with the user-info part of every `scheme://user[:secret]@host`
/// replaced by `***`.
pub(crate) fn redact_urls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("://") {
        let (head, tail) = rest.split_at(at + 3);
        out.push_str(head);
        // The authority ends at the first `/`, whitespace or quote; user-info
        // is whatever precedes its last `@`.
        let end = tail
            .find(|c: char| c == '/' || c.is_whitespace() || c == '\'' || c == '"')
            .unwrap_or(tail.len());
        match tail[..end].rfind('@') {
            Some(user_end) => {
                out.push_str("***");
                out.push_str(&tail[user_end..end]);
            }
            None => out.push_str(&tail[..end]),
        }
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_removed_and_everything_else_kept() {
        for (input, expected) in [
            (
                "fatal: unable to access 'https://user:ghp_secret@github.com/x/y.git/': 403",
                "fatal: unable to access 'https://***@github.com/x/y.git/': 403",
            ),
            (
                "https://ghp_token@github.com/x and https://a:b@c.example/d",
                "https://***@github.com/x and https://***@c.example/d",
            ),
            ("https://github.com/x/y.git", "https://github.com/x/y.git"),
            (
                "ssh://git@github.com/x/y.git",
                "ssh://***@github.com/x/y.git",
            ),
            ("git@github.com:x/y.git", "git@github.com:x/y.git"),
            ("no url here", "no url here"),
            ("https://user:p@ss@host/x", "https://***@host/x"),
        ] {
            assert_eq!(redact_urls(input), expected, "{input}");
        }
    }
}
