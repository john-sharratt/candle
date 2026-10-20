//! The StoryRewrite workload, in one place for every harness that runs it.
//!
//! The batched forward gates measure their headline throughput on it: each
//! session is asked to copy a ~550-token story with its protagonist renamed,
//! greedily. A harness anywhere else — the conversation engine's probe, the
//! cluster load test sent over HTTP — measures the *same* work only if it sends
//! the same bytes and calls the same output correct. So the prompt bodies
//! ([`fixtures`]), the comparison form ([`normalize`]) and the verdict
//! ([`judge`]) live in this crate, which depends on nothing: an HTTP client can
//! use it without building a model.

pub mod fixtures;
pub mod normalize;

use normalize::normalize_story;

/// Characters a correct rewrite may differ from the expected one by, past
/// the common prefix: the gate's own allowance for a trailing token.
pub const TOLERANCE: usize = 5;

/// Fewer comparable characters than this is a failed rewrite, not a vacuous
/// pass: with almost nothing decoded the tolerance would accept anything.
pub const MIN_COMPARED: usize = 16;

/// How a reply compared with its expected rewrite: the characters they share
/// from the start, of those both have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verdict {
    pub common: usize,
    pub compared: usize,
}

impl Verdict {
    /// Whether the reply is the rewrite: at least [`MIN_COMPARED`]
    /// characters compared, and all but [`TOLERANCE`] of them shared.
    pub fn passed(self) -> bool {
        self.compared >= MIN_COMPARED && self.common >= self.compared.saturating_sub(TOLERANCE)
    }
}

/// `reply` against `expected`, both in [`normalize_story`]'s comparison form.
pub fn judge(reply: &str, expected: &str) -> Verdict {
    let got: Vec<char> = normalize_story(reply.trim()).chars().collect();
    let want: Vec<char> = normalize_story(expected.trim()).chars().collect();
    Verdict {
        common: got.iter().zip(&want).take_while(|(a, b)| a == b).count(),
        compared: got.len().min(want.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expected rewrite passes itself; a reply cut short passes on what it
    /// has; a wrong name fails; a near-empty reply fails.
    #[test]
    fn a_rewrite_is_judged_on_its_common_prefix() {
        let story = fixtures::story_prompt();
        let want = fixtures::story_rewrite_expected(&story, "Emily");
        assert!(judge(&want, &want).passed());
        let half: String = want.chars().take(want.chars().count() / 2).collect();
        assert!(judge(&half, &want).passed(), "a reply cut at max_tokens");
        let wrong = fixtures::story_rewrite_expected(&story, "Benjamin");
        assert!(!judge(&wrong, &want).passed(), "another session's name");
        assert!(!judge("The", &want).passed(), "nothing decoded");
        let v = judge("The Backyard Astronaut", "The Backyard Astronomer");
        assert_eq!((v.common, v.compared), (19, 22));
        assert!(v.passed());
    }

    /// The fixtures are the gate's: the story names Marcus for the rename, the
    /// system prompt carries the placeholder, and every session has a name.
    #[test]
    fn the_fixtures_are_the_gates() {
        assert!(fixtures::story_prompt().contains("Marcus"));
        assert!(!fixtures::story_prompt().contains('\r'));
        assert!(fixtures::system_prompt().contains("{INSERT_NAME}"));
        let names = fixtures::session_names();
        assert_eq!(names.len(), 99);
        assert!(names.iter().all(|n| !n.is_empty() && n.trim() == n));
    }
}
