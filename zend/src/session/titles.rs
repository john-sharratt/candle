//! A conversation's sidebar title: a fast one from its first message, then a
//! more accurate one every time a request turn finishes, so the title keeps up
//! with what the conversation is about now.
//!
//! Both are decoded by the background titler conversation, off the request
//! path. A re-title waits in [`PendingRetitles`] until the titler reaches it:
//! a conversation has at most one queued, holding its latest exchange, so a
//! burst of short turns asks for one title, not one each, and the titler's
//! queue never fills with stale work.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::think_strip::strip_think_blocks;

// The titler uses this plain system prompt (not the projection schema), so the
// dialogue `no_think` *section* never applies. Suppression is the `<think>`
// trigger bound to `ThinkMode::Off` in `generate_one_title`, which is structural
// and cannot be ignored — see there for why the soft switch is not enough.
//
// No `/no_think` here. It read as a system-side switch and was never one: the
// marker only does anything on a dialect that declares it
// (`Dialect::thinking_suppression`), this family declares the empty string, and
// on the families that do declare it, it is honoured from the *user* turn rather
// than the system prompt. It was prose to every model that ever saw it.
pub(super) const TITLER_SYSTEM_PROMPT: &str = "You write short conversation titles. \
You are given a conversation's first message, or its current title and its latest \
exchange. Reply with a 3-6 word title that captures what the conversation is about now. \
Keep the current title when it still fits. No quotes, no period, no preamble — just the title.";

/// Tokens of a first message the first title reads, from its start and end.
pub(super) const FIRST_HEAD_TOKENS: usize = 50;
pub(super) const FIRST_TAIL_TOKENS: usize = 50;

/// Tokens of the latest request a re-title reads, from its start and end.
const REQUEST_HEAD_TOKENS: usize = 40;
const REQUEST_TAIL_TOKENS: usize = 24;

/// Tokens of the answer a re-title reads, from its start: what it set out to
/// do is the part that names the topic.
const ANSWER_HEAD_TOKENS: usize = 48;

/// A conversation's latest finished exchange, as a re-title reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Finished {
    pub request: String,
    pub answer: String,
}

/// The re-titles queued and not yet started, one per conversation, each
/// holding that conversation's latest exchange. See the module documentation.
#[derive(Debug, Default)]
pub(super) struct PendingRetitles(Mutex<HashMap<TimelineId, Finished>>);

impl PendingRetitles {
    /// Hold `exchange` as `timeline`'s re-title. `true` when none was queued —
    /// the caller sends the titler a job for it; `false` when one was, which
    /// now reads this exchange instead of the one it was queued with.
    pub fn offer(&self, timeline: TimelineId, exchange: Finished) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(timeline, exchange)
            .is_none()
    }

    /// The exchange `timeline`'s queued re-title reads, taken as it starts —
    /// a turn that finishes after this queues a fresh one.
    pub fn take(&self, timeline: TimelineId) -> Option<Finished> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&timeline)
    }
}

/// What a re-title prefills: the current title, when the conversation has
/// one, the latest request's head and tail, and the start of the answer, the
/// answer's reasoning left out. `cut(text, head, tail)` bounds a text to its
/// first `head` and last `tail` tokens.
pub(super) fn retitle_prompt(
    current: Option<&str>,
    exchange: &Finished,
    cut: &dyn Fn(&str, usize, usize) -> String,
) -> String {
    let answer = strip_think_blocks(&exchange.answer);
    let mut out = String::new();
    if let Some(title) = current.filter(|t| !t.is_empty()) {
        out.push_str(&format!("Current title: {title}\n"));
    }
    out.push_str(&format!(
        "Latest request: {}\n",
        cut(
            exchange.request.trim(),
            REQUEST_HEAD_TOKENS,
            REQUEST_TAIL_TOKENS
        )
    ));
    out.push_str(&format!(
        "Answer begins: {}",
        cut(answer.trim(), ANSWER_HEAD_TOKENS, 0)
    ));
    out
}

/// Bound `text` to its first `head` and last `tail` tokens: verbatim when it
/// is no longer than that, otherwise the two windows joined by `" … "` (the
/// head alone, with a trailing `" …"`, when `tail` is 0). Bounds the titler's
/// prefill however much was pasted.
pub(super) fn head_tail_truncate(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    head: usize,
    tail: usize,
) -> String {
    let Ok(encoded) = tokenizer.encode(text, false) else {
        return text.to_string();
    };
    let ids = encoded.get_ids();
    if ids.len() <= head + tail {
        return text.to_string();
    }
    let head_text = tokenizer.decode(&ids[..head], false).unwrap_or_default();
    if tail == 0 {
        return format!("{head_text} …");
    }
    let tail_text = tokenizer
        .decode(&ids[ids.len() - tail..], false)
        .unwrap_or_default();
    format!("{head_text} … {tail_text}")
}

/// Clean a model-generated title: strip any think block, wrapping quotes,
/// trailing punctuation, and a leading "Title:" the model might add.
pub(super) fn clean_title(raw: &str) -> String {
    // The titler's think block is closed structurally, but the model still
    // emits the empty `<think></think>` (and may reason despite it) — strip any
    // think block so it never lands in the sidebar label.
    let stripped = strip_think_blocks(raw);
    let s = stripped.trim();
    let s = s.strip_prefix("Title:").unwrap_or(s).trim();
    let s = s
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or(s);
    let s = s
        .strip_prefix('\'')
        .and_then(|r| r.strip_suffix('\''))
        .unwrap_or(s);
    let s = s.trim_end_matches(['.', '!', '?']);
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn exchange(request: &str, answer: &str) -> Finished {
        Finished {
            request: request.to_string(),
            answer: answer.to_string(),
        }
    }

    /// Words stand for tokens: the first `head` and last `tail` of them.
    fn words(text: &str, head: usize, tail: usize) -> String {
        let w: Vec<&str> = text.split_whitespace().collect();
        if w.len() <= head + tail {
            return text.to_string();
        }
        match tail {
            0 => format!("{} …", w[..head].join(" ")),
            _ => format!(
                "{} … {}",
                w[..head].join(" "),
                w[w.len() - tail..].join(" ")
            ),
        }
    }

    /// **A second re-title for the same conversation replaces the queued
    /// one** and asks for no second job; another conversation's is its own;
    /// once taken, the next finished turn queues afresh.
    #[test]
    fn a_second_retitle_replaces_the_queued_one() {
        let pending = PendingRetitles::default();
        assert!(pending.offer(tl(1), exchange("first", "a")));
        assert!(
            !pending.offer(tl(1), exchange("second", "b")),
            "no second job"
        );
        assert!(pending.offer(tl(2), exchange("other", "c")));
        assert_eq!(pending.take(tl(1)), Some(exchange("second", "b")));
        assert_eq!(pending.take(tl(1)), None, "taken as it starts");
        assert!(
            pending.offer(tl(1), exchange("third", "d")),
            "queued afresh"
        );
        assert_eq!(pending.take(tl(2)), Some(exchange("other", "c")));
    }

    /// **A re-title reads the current title, the request and the start of
    /// the answer**, its reasoning left out — the exact prefill.
    #[test]
    fn a_retitle_reads_the_title_the_request_and_the_answer() {
        let ex = exchange(
            "  why does the seal scan hold the wave thread?  ",
            "<think>Let me look.</think>\n\nThe journal gallery walks every turn.",
        );
        assert_eq!(
            retitle_prompt(Some("Gallery Index Cache"), &ex, &words),
            "Current title: Gallery Index Cache\n\
             Latest request: why does the seal scan hold the wave thread?\n\
             Answer begins: The journal gallery walks every turn."
        );
        assert_eq!(
            retitle_prompt(None, &ex, &words),
            "Latest request: why does the seal scan hold the wave thread?\n\
             Answer begins: The journal gallery walks every turn.",
            "no title yet, no title line"
        );
    }

    /// **A long exchange is bounded**: the request to its head and tail, the
    /// answer to its head.
    #[test]
    fn a_long_exchange_is_bounded() {
        let request: Vec<String> = (0..100).map(|i| format!("r{i}")).collect();
        let answer: Vec<String> = (0..100).map(|i| format!("a{i}")).collect();
        let ex = exchange(&request.join(" "), &answer.join(" "));
        let prompt = retitle_prompt(Some("T"), &ex, &words);
        let want = format!(
            "Current title: T\nLatest request: {} … {}\nAnswer begins: {} …",
            request[..40].join(" "),
            request[76..].join(" "),
            answer[..48].join(" "),
        );
        assert_eq!(prompt, want);
    }

    #[test]
    fn strips_empty_no_think_block() {
        // The model still emits an empty think block before the label.
        assert_eq!(
            clean_title("<think>\n\n</think>\n\nUnbounded Context Engine"),
            "Unbounded Context Engine"
        );
    }

    #[test]
    fn strips_reasoned_think_block() {
        assert_eq!(
            clean_title("<think>The user is asking about X, so a good title is…</think>\nProjection Bar Rendering"),
            "Projection Bar Rendering"
        );
    }

    #[test]
    fn keeps_existing_cleanup() {
        // Think-strip composes with the prefix/quote/punctuation trimming.
        assert_eq!(
            clean_title("<think></think>Title: \"KV Cache Tiering.\""),
            "KV Cache Tiering"
        );
        assert_eq!(clean_title("Plain Title"), "Plain Title");
    }
}
