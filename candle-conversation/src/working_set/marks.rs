//! Working-set marks on a dialogue's turns are not gather scope.
//!
//! A turn's tags say which tag-scoped gallery it belongs to, and "no tags" is
//! what makes a turn ordinary dialogue. Turns sealed while the working set
//! locked what the fast path served also carry `working_set:` marks, which
//! name no gallery. Every reader of tags as scope goes through [`gather_tags`]
//! or [`is_dialogue`], so a dialogue turn that carries only marks stays
//! dialogue.

/// Every mark starts with this.
const MARK_PREFIX: &str = "working_set:";

/// Whether `tag` is a working-set mark rather than a gather scope.
pub fn is_mark(tag: &str) -> bool {
    tag.starts_with(MARK_PREFIX)
}

/// `tags` without the working-set marks — the turn's gather scope.
pub fn gather_tags(tags: &[String]) -> impl Iterator<Item = &String> {
    tags.iter().filter(|t| !is_mark(t))
}

/// Whether a turn with `tags` is ordinary dialogue: it belongs to no gather
/// scope.
pub fn is_dialogue(tags: &[String]) -> bool {
    gather_tags(tags).next().is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// A dialogue turn carrying only marks is still dialogue; a gallery turn
    /// with a mark keeps its scope.
    #[test]
    fn marks_are_not_gather_scope() {
        assert!(is_dialogue(&[]));
        assert!(is_dialogue(&tags(&[
            "working_set:release",
            "working_set:lock:3"
        ])));
        assert!(!is_dialogue(&tags(&["tool", "working_set:release"])));
        let turn = tags(&["working_set:lock:3", "code"]);
        let scope: Vec<&String> = gather_tags(&turn).collect();
        assert_eq!(scope, vec!["code"]);
    }
}
