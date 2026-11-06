//! Topics are strings with `/` segments. A pattern admits a topic equal to it,
//! or — ending in `*` — every topic that starts with what comes before the
//! `*`: `gate/yenbot/result/*` admits `gate/yenbot/result/01JD`.

/// Whether `pattern` admits `topic`.
pub fn admits(pattern: &str, topic: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => topic.starts_with(prefix),
        None => pattern == topic,
    }
}

/// Whether `topic` is one a publisher may name: not empty, no wildcard, no
/// empty segment.
pub fn valid(topic: &str) -> bool {
    !topic.is_empty() && !topic.contains('*') && topic.split('/').all(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pattern_admits_its_topic_or_what_is_under_its_star() {
        assert!(admits("repo/yenbot", "repo/yenbot"));
        assert!(!admits("repo/yenbot", "repo/yenbot2"));
        assert!(!admits("repo/yenbot", "repo/yenbot/x"));
        assert!(admits("gate/yenbot/result/*", "gate/yenbot/result/01JD"));
        assert!(!admits("gate/yenbot/result/*", "gate/yenbot"));
        assert!(admits("*", "anything/at/all"));
    }

    #[test]
    fn a_published_topic_names_one_place() {
        assert!(valid("repo/yenbot"));
        assert!(!valid(""));
        assert!(!valid("repo/*"));
        assert!(!valid("repo//x"));
        assert!(!valid("/repo"));
    }
}
