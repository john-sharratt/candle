//! The paths an ingest layer never reads (`--ignore`, `motherd.yaml`'s
//! `ignore:`; `docs/zend_branch_ingest.md` §5).
//!
//! A pattern is a workspace key, `<repo>/<path>`, in which `*` matches any run
//! of characters, `/` included: `candle/docs/results/dependency_battery_*`.
//! An ignored path is refused by [`IngestScope::admits`] exactly as a path
//! past the depth bound or off the extension allowlist is, so it is neither
//! read nor retained, and the pass tombstones whatever reading of it the
//! substrate already holds.
//!
//! [`IngestScope::admits`]: super::filter::IngestScope::admits

use std::sync::Arc;

/// A set of ignore patterns. Empty ignores nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Ignore(Arc<[String]>);

impl Ignore {
    /// The patterns, each trimmed of surrounding `/`; empty ones dropped.
    pub fn new<S: AsRef<str>>(patterns: &[S]) -> Ignore {
        let kept: Vec<String> = patterns
            .iter()
            .map(|p| p.as_ref().trim_matches('/').to_string())
            .filter(|p| !p.is_empty())
            .collect();
        Ignore(Arc::from(kept))
    }

    /// Whether `key`, a workspace key `<repo>/<path>`, is ignored.
    pub fn matches(&self, key: &str) -> bool {
        self.0.iter().any(|pattern| glob(pattern, key))
    }

    /// The patterns, as given.
    pub fn patterns(&self) -> &[String] {
        &self.0
    }
}

/// Whether `key` matches `pattern`, where `*` matches any run of characters,
/// `/` included.
fn glob(pattern: &str, key: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == key,
        Some((head, rest)) => {
            let Some(after) = key.strip_prefix(head) else {
                return false;
            };
            (0..=after.len())
                .filter(|&i| after.is_char_boundary(i))
                .any(|i| glob(rest, &after[i..]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A pattern names the files it covers and no others**: a literal key,
    /// a trailing `*` over a folder at any depth, and a `*` inside a name.
    #[test]
    fn patterns_match_the_keys_they_name() {
        let ignore = Ignore::new(&[
            "candle/docs/results/dependency_battery_*",
            "battle-cities/docs/lore/*",
            "yenbot/README.md",
        ]);
        assert!(ignore.matches("candle/docs/results/dependency_battery_2026-11-10.md"));
        assert!(ignore.matches("battle-cities/docs/lore/skills/fire/ember.md"));
        assert!(ignore.matches("yenbot/README.md"));
        assert!(!ignore.matches("candle/docs/results/theorem_depth_tests_2026-11-10.md"));
        assert!(!ignore.matches("battle-cities/docs/design.md"));
        assert!(!ignore.matches("yenbot/README.md.bak"));
        assert!(!ignore.matches("other/candle/docs/results/dependency_battery_x.md"));
    }

    /// Surrounding `/` and empty patterns change nothing; no pattern ignores
    /// nothing.
    #[test]
    fn empty_and_slashed_patterns_are_normalised() {
        let ignore = Ignore::new(&["/candle/docs/*/", "", "/"]);
        assert_eq!(ignore.patterns(), ["candle/docs/*"]);
        assert!(ignore.matches("candle/docs/a.md"));
        assert!(!Ignore::default().matches("candle/docs/a.md"));
    }
}
