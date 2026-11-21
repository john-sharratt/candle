//! Normalising a caller's path to the key both layers store it under.
//!
//! Every spelling of a path normalises to one canonical key before use, so the
//! two layers and every operation address an entry the same way. A workspace can
//! hold names that no Windows folder opens as themselves, such as a colon or an
//! 8.3 short-name tail; listings apply the same name rules as reads, so a file
//! the model is shown can always be read. Prefix matching follows plain string
//! semantics, so `src/ma` selects `src/main.rs`, and an empty prefix matches
//! everything.

use super::*;

impl VfsStore {
    // ── Path rules ───────────────────────────────────────────────────────────

    /// Whether every segment of `norm` is a name a folder opens as itself — no
    /// `:`, no trailing `.` or space, no 8.3 short-name tail.
    ///
    /// The workspace walks apply it too, so a listing never shows a file the
    /// store cannot open. A Linux workspace can hold `logs/12:00.txt` or
    /// `notes~2.md`; listed but refused by every read, the model was shown a
    /// file it could name and never read.
    pub(crate) fn addressable(norm: &str) -> bool {
        !norm
            .split('/')
            .any(|s| s.contains(':') || s.ends_with('.') || s.ends_with(' ') || is_short_name(s))
    }

    /// `true` when `key` is under `prefix`. Plain string-prefix semantics, as
    /// the search tools' `prefix` parameter documents — so `src/` and `src`
    /// and even the partial `src/ma` all select `src/main.rs`. An empty
    /// prefix matches everything.
    pub(super) fn matches_prefix(key: &str, prefix: &str) -> bool {
        prefix.is_empty() || key.starts_with(prefix)
    }

    /// Canonical overlay key for a caller-supplied path. See the module docs.
    pub fn normalize(path: &str) -> String {
        let path = path.trim_start_matches('/');
        let mut parts: Vec<&str> = Vec::new();
        for segment in path.split(['/', '\\']) {
            match segment {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                s => parts.push(s),
            }
        }
        parts.join("/")
    }
}

/// Whether `segment` has the shape of a Windows 8.3 short name's tilde tail —
/// a `~` followed by a digit (`SECRET~1`, `PROGRA~2.TXT`).
pub(super) fn is_short_name(segment: &str) -> bool {
    segment
        .as_bytes()
        .windows(2)
        .any(|w| w[0] == b'~' && w[1].is_ascii_digit())
}
