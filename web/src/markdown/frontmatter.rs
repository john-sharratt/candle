//! YAML front matter, for content that carries its own metadata.
//!
//! A blog post owns its title, date and summary — they belong in the file, so
//! writing a post is creating one file and nothing else. Papers are the other
//! case: they are design documents living in `docs/`, they carry no front
//! matter, and adding some would put publication metadata into a working
//! document. Their metadata lives in the site's manifest instead, which is also
//! what decides that a document is published at all.

use serde::Deserialize;

/// One post's metadata, from the `---` block at the top of the file.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Post {
    pub title: String,
    /// `YYYY-MM-DD`. A string rather than a date type because it is only ever
    /// displayed and sorted, and both work lexically on this format.
    pub date: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Keeps a post out of the index while it is being written. It stays
    /// readable at its own URL, which is how you show someone a draft.
    #[serde(default)]
    pub draft: bool,
    /// Editorial position in the index, lowest first. Posts carrying one lead
    /// the list in that order; everything else follows by date, newest first.
    ///
    /// Separate from `date` because the two answer different questions. `date`
    /// is when the piece was written and has to stay honest — several of these
    /// were published years before the site existed. `feature` is which piece a
    /// first-time reader should meet first, which is a choice about the reader
    /// and not about the calendar.
    #[serde(default)]
    pub feature: Option<u32>,
    /// The accent this post is written in — headings, links, callouts, its row
    /// on the index. A closed set rather than a colour value, so a post cannot
    /// invent a colour that is outside the theme or unreadable against it.
    #[serde(default)]
    pub tint: Tint,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Tint {
    /// The site's own accent.
    #[default]
    Accent,
    Ok,
    Info,
    Violet,
    Warn,
    Crit,
}

impl Tint {
    /// The CSS class that rebinds `--post` for this article.
    pub fn class(self) -> &'static str {
        match self {
            Tint::Accent => "tint-accent",
            Tint::Ok => "tint-ok",
            Tint::Info => "tint-info",
            Tint::Violet => "tint-violet",
            Tint::Warn => "tint-warn",
            Tint::Crit => "tint-crit",
        }
    }
}

/// Split a leading `---` block off the body.
///
/// Returns `(None, whole_input)` when there is no front matter, so a plain
/// markdown file passes through untouched.
pub fn split(text: &str) -> (Option<&str>, &str) {
    let body = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"));
    let Some(body) = body else {
        return (None, text);
    };

    // The terminator must be a line of its own, or a `---` horizontal rule in
    // the prose would truncate the document at the first one.
    for (idx, line) in line_offsets(body) {
        if line.trim_end() == "---" {
            let rest = &body[idx + line.len()..];
            let rest = rest
                .strip_prefix('\n')
                .or_else(|| rest.strip_prefix("\r\n"))
                .unwrap_or(rest);
            return (Some(&body[..idx]), rest);
        }
    }
    // An unterminated block is a mistake in the file, not a document whose
    // first half is metadata: treat the whole thing as body so the author sees
    // their `---` on the page rather than losing everything after it.
    (None, text)
}

fn line_offsets(s: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut at = 0;
    std::iter::from_fn(move || {
        if at >= s.len() {
            return None;
        }
        let rest = &s[at..];
        let len = rest.find('\n').map(|i| i + 1).unwrap_or(rest.len());
        let item = (at, &rest[..len]);
        at += len;
        Some(item)
    })
}

/// Parse a post's front matter. `Err` names the file's problem, not serde's.
pub fn post(text: &str) -> anyhow::Result<(Post, &str)> {
    let (fm, body) = split(text);
    let fm = fm.ok_or_else(|| {
        anyhow::anyhow!("no front matter — a post needs a `---` block with at least title and date")
    })?;
    let meta: Post = serde_yaml::from_str(fm)?;
    Ok((meta, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_splits_off_the_body() {
        let (fm, body) = split("---\ntitle: Hi\n---\n# Heading\n");
        assert_eq!(fm, Some("title: Hi\n"));
        assert_eq!(body, "# Heading\n");
    }

    #[test]
    fn a_plain_file_is_all_body() {
        let (fm, body) = split("# Heading\n\ntext\n");
        assert!(fm.is_none());
        assert_eq!(body, "# Heading\n\ntext\n");
    }

    #[test]
    fn a_horizontal_rule_in_the_prose_does_not_terminate_early() {
        let src = "---\ntitle: Hi\ndate: 2026-08-25\n---\nintro\n\n---\n\nmore\n";
        let (fm, body) = split(src);
        assert_eq!(fm, Some("title: Hi\ndate: 2026-08-25\n"));
        assert!(
            body.contains("more"),
            "the rule truncated the body: {body:?}"
        );
    }

    #[test]
    fn an_unterminated_block_keeps_the_whole_document() {
        // Losing the post silently would be much worse than showing the dashes.
        let src = "---\ntitle: Hi\nbody goes on forever\n";
        let (fm, body) = split(src);
        assert!(fm.is_none());
        assert_eq!(body, src);
    }

    #[test]
    fn a_post_parses_its_metadata() {
        let (meta, body) =
            post("---\ntitle: One Card\ndate: 2026-08-25\nsummary: s\ntags: [a, b]\n---\nbody\n")
                .unwrap();
        assert_eq!(meta.title, "One Card");
        assert_eq!(meta.tags, ["a", "b"]);
        assert!(!meta.draft);
        assert_eq!(
            meta.tint,
            Tint::Accent,
            "a post with no tint uses the site accent"
        );
        assert_eq!(body, "body\n");
    }

    #[test]
    fn a_tint_is_a_closed_set() {
        let src = |t| format!("---\ntitle: t\ndate: 2026-08-25\ntint: {t}\n---\nx");
        assert_eq!(post(&src("violet")).unwrap().0.tint, Tint::Violet);
        assert_eq!(post(&src("ok")).unwrap().0.tint, Tint::Ok);
        // A colour outside the theme is a mistake in the file, not a new colour.
        assert!(post(&src("hotpink")).is_err());
        assert!(post(&src("#ff00ff")).is_err());
    }

    #[test]
    fn a_typo_in_the_front_matter_is_an_error_rather_than_a_default() {
        // `sumary:` silently becoming an empty summary is exactly the kind of
        // thing nobody notices until the index looks wrong.
        assert!(post("---\ntitle: t\ndate: 2026-08-25\nsumary: s\n---\nx").is_err());
        assert!(post("---\nsummary: s\n---\nx").is_err()); // no title, no date
    }
}
