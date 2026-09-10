//! Keep a collapsed `<think></think>` off the wire.
//!
//! The engine strips empty think blocks from the text it **stores**
//! (`strip_empty_think_blocks`, in `build_turn_layout`), and deliberately keeps
//! non-empty reasoning so a dialogue `Thinking` segment has something to show.
//! The **stream** is a separate assembly — raw token events decoded
//! incrementally — and nothing filtered it, so an empty block went out verbatim
//! and rendered as leaked markup.
//!
//! It became the common case with thinking-span projection: every turn but the
//! most recent has its reasoning windowed out of the K/V, so the model reads a
//! history with no visible reasoning and answers with a collapsed block of its
//! own. Measured on a six-turn conversation, every single turn came back as
//! `<think>\n\n</think>\n\n` followed by the answer.
//!
//! # Why this holds rather than filters
//!
//! A think block cannot be classified until its `</think>` arrives, and holding
//! output until then would stall a genuine reasoning turn for its entire
//! reasoning. So the hold is bounded by *content*, not by the close marker: it
//! lasts only while the block is still whitespace-only. The first non-blank
//! token inside it resolves the gate open forever, and streaming proceeds
//! normally — a thinking turn is never held waiting for its own close.

/// What the stream may do with the text decoded so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThinkGate {
    /// A think block is open and still blank. Emit nothing yet; at most a few
    /// whitespace tokens are ever withheld.
    Hold,
    /// Stream from this byte offset. Non-zero exactly when an empty block was
    /// resolved and skipped, together with the blank run after it, so the answer
    /// does not open on a gap.
    Open(usize),
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Classify the leading think block in `text`, if there is one.
///
/// Call until it returns [`ThinkGate::Open`]; the answer never changes back, so
/// the caller resolves once per turn and then streams unconditionally.
pub(crate) fn gate_leading_think(text: &str) -> ThinkGate {
    let Some(open) = text.find(OPEN) else {
        // No complete marker yet, and three cases hide here. Blank text could
        // still become one. So could a PARTIAL marker — `<`, `<th` — which
        // arrives whenever the tokenizer splits `<think>`; releasing those bytes
        // is what would leak the markup this gate exists to hide, so a prefix of
        // the marker holds too. Anything else is an answer that never opened a
        // block and must not be withheld.
        let head = text.trim_start();
        return if head.is_empty() || OPEN.starts_with(head) {
            ThinkGate::Hold
        } else {
            ThinkGate::Open(0)
        };
    };
    // **Only a LEADING block is the gate's business.** `find` locates the first
    // marker anywhere, so a block that opens *after* real text — a steering tree
    // replaying one mid-answer — would otherwise be skipped along with every
    // byte before it, silently deleting the answer's opening words. Anything
    // non-blank ahead of the marker means the turn has already started
    // streaming, and the gate has nothing left to decide.
    if !text[..open].trim().is_empty() {
        return ThinkGate::Open(0);
    }
    let inner_at = open + OPEN.len();
    let Some(rel) = text[inner_at..].find(CLOSE) else {
        // Open and unclosed: real reasoning the moment it is not just blank.
        return if text[inner_at..].trim().is_empty() {
            ThinkGate::Hold
        } else {
            ThinkGate::Open(0)
        };
    };
    if text[inner_at..inner_at + rel].trim().is_empty() {
        let after = inner_at + rel + CLOSE.len();
        let tail = text[after..].trim_start();
        if tail.is_empty() {
            // **The block has closed but its trailing gap has not arrived yet.**
            //
            // Resolving here would release the gate at the exact byte after
            // `</think>`, and the `\n\n` the template puts between the block and
            // the answer would then stream verbatim — a leading blank gap, which
            // is most of what the skip existed to remove. The caller feeds this
            // one token at a time, so "nothing after it" is the normal state for
            // a token or two, not the end of the turn.
            //
            // Still bounded by content: the first non-blank byte resolves it.
            // A turn whose answer is entirely whitespace holds to the end and
            // emits nothing, which is the right rendering of that turn anyway.
            return ThinkGate::Hold;
        }
        ThinkGate::Open(text.len() - tail.len())
    } else {
        // A real block that has already closed — stream it whole, including the
        // markers, which is what a client rendering a reasoning pane expects.
        ThinkGate::Open(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{gate_leading_think, ThinkGate};

    /// The case this exists for: a collapsed block is skipped along with the
    /// blank run after it, so the client's first token is the answer.
    #[test]
    fn an_empty_block_is_skipped_with_the_gap_after_it() {
        let s = "<think>\n\n</think>\n\nParis";
        let ThinkGate::Open(at) = gate_leading_think(s) else {
            panic!("a closed empty block resolves")
        };
        assert_eq!(&s[at..], "Paris");
    }

    /// `<think></think>` with nothing between the tags at all.
    #[test]
    fn a_block_with_no_inner_text_is_skipped() {
        let s = "<think></think>4";
        let ThinkGate::Open(at) = gate_leading_think(s) else {
            panic!("resolves")
        };
        assert_eq!(&s[at..], "4");
    }

    /// **Real reasoning is never withheld waiting for its close.** The gate
    /// opens on the first non-blank token inside the block, long before
    /// `</think>` arrives — otherwise a thinking turn would stream nothing until
    /// it had finished thinking.
    #[test]
    fn real_reasoning_opens_the_gate_before_it_closes() {
        assert_eq!(gate_leading_think("<think>\n\nThe user"), ThinkGate::Open(0));
        assert_eq!(gate_leading_think("<think>x"), ThinkGate::Open(0));
    }

    /// A closed non-empty block streams whole, markers included — a client
    /// rendering a reasoning pane needs them.
    #[test]
    fn a_real_closed_block_streams_from_the_start() {
        assert_eq!(
            gate_leading_think("<think>reasoning</think>Answer"),
            ThinkGate::Open(0)
        );
    }

    /// The hold is bounded by content: only a partial open marker or the blank
    /// inside one is ever withheld.
    #[test]
    fn only_a_blank_opening_block_is_held() {
        assert_eq!(gate_leading_think(""), ThinkGate::Hold);
        assert_eq!(gate_leading_think("<th"), ThinkGate::Hold);
        assert_eq!(gate_leading_think("<think>"), ThinkGate::Hold);
        assert_eq!(gate_leading_think("<think>\n\n"), ThinkGate::Hold);
    }

    /// A turn that opens straight into its answer is never held.
    #[test]
    fn an_answer_with_no_block_streams_immediately() {
        assert_eq!(gate_leading_think("Paris"), ThinkGate::Open(0));
    }

    /// Only the LEADING block is gated. A later `<think>` in the answer body —
    /// which a steering tree can replay — is ordinary text by then, because the
    /// gate has already resolved and is not consulted again.
    #[test]
    fn a_later_block_is_not_the_gates_business() {
        let s = "Answer <think></think> more";
        assert_eq!(gate_leading_think(s), ThinkGate::Open(0));
    }

    /// **Fed the way the stream actually feeds it: one token at a time.**
    ///
    /// The caller accumulates and re-asks after every token, so what matters is
    /// not one verdict on a whole string but the sequence of them — and that the
    /// bytes finally released are exactly the answer, with the collapsed block
    /// and the blank run after it gone, and nothing of the answer lost. The
    /// marker arrives split (`<`, `th`, `ink>`), which is what a tokenizer does
    /// to it and what an earlier version of this gate leaked.
    #[test]
    fn token_by_token_releases_exactly_the_answer() {
        for tokens in [
            vec!["<", "th", "ink>", "\n\n", "</think>", "\n\n", "Paris", "."],
            vec!["<think>", "</think>", "Paris."],
            vec!["<think>\n\n</think>\n\nParis."],
        ] {
            let mut held = String::new();
            let mut open = false;
            let mut out = String::new();
            for t in &tokens {
                if open {
                    out.push_str(t);
                    continue;
                }
                held.push_str(t);
                if let ThinkGate::Open(at) = gate_leading_think(&held) {
                    open = true;
                    out.push_str(&held[at..]);
                }
            }
            assert!(open, "the gate must resolve for {tokens:?}");
            assert_eq!(out, "Paris.", "wrong bytes released for {tokens:?}");
        }
    }

    /// The mirror of the above: a turn that reasons for real must not be
    /// withheld, and must lose nothing. The gate opens inside the block, so the
    /// markers and the reasoning both reach the client.
    #[test]
    fn a_reasoning_turn_streams_every_byte() {
        let tokens = ["<think>", "\n\n", "The", " user", "</think>", "Hi"];
        let mut held = String::new();
        let mut open = false;
        let mut out = String::new();
        for t in tokens {
            if open {
                out.push_str(t);
                continue;
            }
            held.push_str(t);
            if let ThinkGate::Open(at) = gate_leading_think(&held) {
                open = true;
                out.push_str(&held[at..]);
            }
        }
        assert_eq!(out, "<think>\n\nThe user</think>Hi");
    }
}
