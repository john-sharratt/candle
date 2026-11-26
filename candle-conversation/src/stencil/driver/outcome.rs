//! Decides how a sampled token is committed: as written, rewritten into the bytes
//! that close or repair its span, or dropped.
//!
//! A free-text span can end inside a token, as when a model merges a value's close
//! with the next delimiter. The token is then committed as the part that belongs to
//! the value, and the session moves on as if that were what was written. A token
//! that cannot be committed as written is repaired, and a token that contributes
//! nothing is dropped. A close token that ends a span is dropped, so the steering
//! can supply its continuation in its place. The simulator shares this decision, so
//! a test judges exactly what the decode loop commits.

use super::*;

/// What [`StencilDriver::accept`] decided about the sampled token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Healed {
    /// The token was accepted as-is — commit it normally.
    No,
    /// Commit the re-tokenized `bytes` in place of the sampled token.
    ///
    /// A free-text span ended *inside* this token (the model merged the value's
    /// close with the next node's delimiter — `",` — and `bytes` is the part
    /// that belongs to the value); or the token could not be committed as
    /// written, and `bytes` is its repair — a character escaped, a malformed
    /// value completed, an EOS inside a value replaced by the text that closes
    /// it, a value at its hard limit completed after the token. The session has
    /// already moved on as if the model had written `bytes`. Never empty: a
    /// token that contributes nothing is [`Healed::Drop`].
    Rewrite { bytes: Vec<u8> },
    /// Drop this sampled token entirely: it must NOT be committed to the
    /// sequence's KV.  A token-closed free-text span with `suppress_close` ended
    /// on this exact token (the close token, e.g. `</think>`), and the span
    /// suppresses it so the steering can retry — the session cursor has already
    /// advanced, so the next [`step`](StencilDriver::step) yields the
    /// continuation [`Prefill`](StepMask::Prefill) that prefills in its place.
    /// (The decode-loop side that skips the commit lands separately in
    /// `scheduler/decode.rs`; the stencil core only produces this signal.)
    Drop,
}

/// What the decode loop should do for this sequence next.
#[derive(Debug, Clone, PartialEq)]
pub enum StepMask {
    /// Inject this `Static` run (or the bail closing run) into the sequence's KV.
    /// The session cursor has already advanced past the run, so the next
    /// [`step`](StencilDriver::step) yields the node after it.
    Prefill(Vec<TokenId>),
    /// Mask the sampler to this allowed set (a branch frontier).
    Branch(AllowedSet),
    /// Free decode within a span (normal decode — EOS and any close token are
    /// intercepted by the session).  `close_boost` is the soft close-token logit
    /// ramp; `opens_value` marks the first step of a call's value, which bans
    /// the end-of-turn tokens and lifts the repetition penalties for this one
    /// step (see `StencilSession::free_decode_action`); `barred` are tokens this
    /// step may not draw — those that would close a required value blank.
    Free {
        close_boost: f32,
        opens_value: bool,
        barred: Option<Arc<[TokenId]>>,
    },
    /// The stencil finished — resume unconstrained decoding.
    Done,
}

/// How a sampled token is committed, given what the session observed of it and
/// the rewrite it left.
///
/// Shared by [`StencilDriver::accept`] and the simulator, so the simulator
/// commits exactly what the decode loop does — a test that judged the session
/// on committing whole tokens would pass output the decode loop never produces.
pub(in super::super) fn healing(
    observe: Observe,
    rewrite: Option<Vec<u8>>,
    bytes: &[u8],
) -> Healed {
    match (observe, rewrite) {
        (_, Some(bytes)) if bytes.is_empty() => Healed::Drop,
        (_, Some(bytes)) => Healed::Rewrite { bytes },
        (Observe::SpanClosed { leftover }, None) if leftover > 0 && leftover < bytes.len() => {
            Healed::Rewrite {
                bytes: bytes[..bytes.len() - leftover].to_vec(),
            }
        }
        (Observe::TokenClosedDrop | Observe::DelimiterDropped, None) => Healed::Drop,
        _ => Healed::No,
    }
}
