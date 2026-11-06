//! Whether a turn has written an answer: what a tail follows
//! (`docs/stencil_tree.md` §9.3).
//!
//! A tail writes on from an answer's end, so an end token with nothing before
//! it but the reasoning is not one. A reviewer reasoned until its think block
//! was closed for it and then sampled the end straight away; the `Steps:` tail
//! wrote its section onto the empty answer, and an answer that says nothing
//! passed as one that accounts for every step.

/// The most tokens an answer that is only whitespace is read for: past this
/// many, the turn has written an answer.
const PROBE: usize = 16;

/// Whether the tokens after the reasoning hold an answer: anything but
/// whitespace. `decode` gives a token run's text; it is asked only of a run of
/// at most [`PROBE`] tokens, once, at the turn's end.
pub(super) fn answered(after: &[u32], decode: impl FnOnce(&[u32]) -> String) -> bool {
    after.len() > PROBE || decode(after).chars().any(|c| !c.is_whitespace())
}

/// The tokens of `generated` after the reasoning that closed at `close_at`
/// (its `</think>`); all of them when there was none.
pub(super) fn after_reasoning(generated: &[u32], close_at: Option<u32>) -> &[u32] {
    match close_at {
        Some(at) => generated.get(at as usize + 1..).unwrap_or(&[]),
        None => generated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|&t| match t {
                1 => "\n\n",
                2 => " ",
                _ => "x",
            })
            .collect()
    }

    /// **An answer is anything but whitespace after the reasoning**: blank
    /// lines alone are none, one word is one, and a run past the probe is one
    /// without being read.
    #[test]
    fn an_answer_is_anything_but_whitespace() {
        assert!(!answered(&[], text));
        assert!(!answered(&[1, 2, 1], text));
        assert!(answered(&[1, 7], text));
        assert!(answered(&[1; PROBE + 1], |_| unreachable!("not read")));
    }

    /// **The answer starts after the think close**, and is the whole turn
    /// when it did not reason.
    #[test]
    fn the_answer_starts_after_the_close() {
        let generated = [5, 5, 9, 1, 7];
        assert_eq!(after_reasoning(&generated, Some(2)), [1, 7]);
        assert_eq!(after_reasoning(&generated, Some(4)), [] as [u32; 0]);
        assert_eq!(after_reasoning(&generated, None), generated);
    }
}
