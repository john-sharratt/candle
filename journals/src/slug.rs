//! A name in the tree (§2.2): a person, a group, a bot, or a journal below the
//! fixed levels. Kebab-case, at most 40 characters, optionally numbered
//! `.2`, `.3` … when a name recurs among siblings.

use std::collections::BTreeSet;
use std::fmt;

/// The longest a name may be.
pub const MAX_SLUG: usize = 40;

/// Names no journal may take: a journal's folder also holds its `archive/`.
pub const RESERVED: &[&str] = &["archive"];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slug(String);

impl Slug {
    /// `s` exactly, or why it is not a name.
    pub fn parse(s: &str) -> Result<Slug, String> {
        let bad = |why: &str| format!("`{s}` is not a name: {why}");
        if s.is_empty() {
            return Err(bad("it is empty"));
        }
        if s.chars().count() > MAX_SLUG {
            return Err(bad("it is longer than 40 characters"));
        }
        let (stem, number) = match s.rsplit_once('.') {
            Some((stem, n)) => (stem, Some(n)),
            None => (s, None),
        };
        if let Some(n) = number {
            let ok = !n.is_empty()
                && n.bytes().all(|c| c.is_ascii_digit())
                && !n.starts_with('0')
                && n.parse::<u32>().is_ok_and(|n| n >= 2);
            if !ok {
                return Err(bad("a number after `.` is 2 or more"));
            }
        }
        if stem.is_empty()
            || stem.starts_with('-')
            || stem.ends_with('-')
            || stem.contains("--")
            || !stem
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        {
            return Err(bad(
                "use lowercase letters, digits and single hyphens between them",
            ));
        }
        Ok(Slug(s.to_string()))
    }

    /// The nearest name to free text: lowercase, every run of other
    /// characters one hyphen, cut to 40. `None` when nothing is left.
    pub fn slugify(text: &str) -> Option<Slug> {
        let mut out = String::new();
        let mut gap = false;
        for c in text.chars().flat_map(char::to_lowercase) {
            if c.is_ascii_lowercase() || c.is_ascii_digit() {
                if gap && !out.is_empty() {
                    out.push('-');
                }
                gap = false;
                out.push(c);
            } else {
                gap = true;
            }
            if out.len() >= MAX_SLUG {
                break;
            }
        }
        out.truncate(MAX_SLUG);
        let trimmed = out.trim_end_matches('-');
        Slug::parse(trimmed).ok()
    }

    /// `self` if no sibling has it, else the first of `self.2`, `self.3` …
    /// that is free, cut so the whole stays within 40 characters.
    pub fn numbered(&self, taken: &BTreeSet<Slug>) -> Slug {
        if !taken.contains(self) {
            return self.clone();
        }
        let stem = self.stem();
        for n in 2u32.. {
            let suffix = format!(".{n}");
            let room = MAX_SLUG - suffix.len();
            let cut = stem[..stem.len().min(room)].trim_end_matches('-');
            let candidate = Slug(format!("{cut}{suffix}"));
            if !taken.contains(&candidate) {
                return candidate;
            }
        }
        unreachable!("u32 names run out only after four billion siblings")
    }

    /// The name without its number.
    pub fn stem(&self) -> &str {
        match self.0.rsplit_once('.') {
            Some((stem, _)) => stem,
            None => &self.0,
        }
    }

    pub fn is_reserved(&self) -> bool {
        RESERVED.contains(&self.0.as_str())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Slug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::slug;

    #[test]
    fn names_are_kebab_case_with_an_optional_number() {
        for ok in [
            "candle",
            "house-build",
            "2026-10-30-2105",
            "c3f1a07e",
            "verify.2",
            "verify.10",
            "a",
            "a-b-c",
            &"x".repeat(40),
        ] {
            assert_eq!(Slug::parse(ok).map(|s| s.to_string()), Ok(ok.to_string()));
        }
    }

    #[test]
    fn anything_else_is_refused_with_its_reason() {
        for bad in [
            "",
            "Candle",
            "house_build",
            "house build",
            "-a",
            "a-",
            "a--b",
            "a.1",
            "a.0",
            "a.02",
            "a.",
            ".2",
            "a.b",
            "a.2.3",
            "café",
            "a/b",
            "..",
            &"x".repeat(41),
        ] {
            assert!(Slug::parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            Slug::parse("A").unwrap_err(),
            "`A` is not a name: use lowercase letters, digits and single hyphens between them"
        );
    }

    #[test]
    fn free_text_becomes_the_nearest_name() {
        let s = |t: &str| Slug::slugify(t).map(|s| s.to_string());
        assert_eq!(s("House Build!"), Some("house-build".into()));
        assert_eq!(s("  --Permits & Fees--  "), Some("permits-fees".into()));
        assert_eq!(s("Đà Nẵng trip"), Some("n-ng-trip".into()));
        assert_eq!(s("!!!"), None);
        assert_eq!(s(""), None);
        let long = s(&"word ".repeat(20)).unwrap();
        assert!(long.len() <= 40 && !long.ends_with('-'), "{long}");
    }

    #[test]
    fn a_recurring_name_takes_the_next_free_number() {
        let taken: BTreeSet<Slug> = ["verify", "verify.2", "verify.4"]
            .into_iter()
            .map(slug)
            .collect();
        assert_eq!(slug("verify").numbered(&taken), slug("verify.3"));
        assert_eq!(slug("review").numbered(&taken), slug("review"));
        assert_eq!(slug("verify.2").numbered(&taken), slug("verify.3"));
        // A full-length name is cut to make room for its number…
        let full = slug(&format!("{}-b", "a".repeat(38)));
        let numbered = full.numbered(&BTreeSet::from([full.clone()]));
        assert_eq!(numbered.to_string(), format!("{}.2", "a".repeat(38)));
        // …and never left ending in a hyphen by the cut.
        let full = slug(&format!("{}-bc", "a".repeat(37)));
        let numbered = full.numbered(&BTreeSet::from([full.clone()]));
        assert_eq!(numbered.to_string(), format!("{}.2", "a".repeat(37)));
        assert!(Slug::parse(numbered.as_str()).is_ok());
    }

    #[test]
    fn archive_is_reserved() {
        assert!(slug("archive").is_reserved());
        assert!(!slug("archives").is_reserved());
    }
}
