//! When a turn placed on a member would finish, by the rates it advertises
//! (`docs/cooperative_admission.md` §2.1).

use super::{capacity_of, TurnSize};
use crate::frame::{Capacity, PLACEMENT_WIDE};
use crate::gateway::registry::{CardLoad, MemberView};

/// When a turn placed on `m` now would finish, in microseconds from now, by
/// what it says a turn costs it (`docs/cooperative_admission.md` §2.1, §3
/// "earliest advertised finish"): the prefill already queued there, the
/// turns handed to its card and not yet in its queue, and this one, at the
/// prefill rate those turns and this one leave it ([`token_ns_with`]); then the
/// reply at the decode step they give it ([`step_with_us`]), over as many steps
/// as the reply takes at the tokens a step commits there ([`reply_steps`]) —
/// each turn of `size`. `card` is what the gateway has placed on `m`'s card,
/// every app's units together (§4.2). `None` until it has learned both rates —
/// a member is never placed by figures it has not measured.
pub(crate) fn finish_us(m: &MemberView, card: CardLoad, size: TurnSize) -> Option<u64> {
    let c = capacity_of(m);
    if c.prefill_tok_s == 0 || c.decode_step_next_us == 0 {
        return None;
    }
    // A request handed since the report and already answered brought no
    // prompt that is still waiting: only those still in flight can be.
    let pending = card.pending();
    let tokens = u64::from(c.prefill_backlog) + (u64::from(pending) + 1) * size.prompt;
    let prefill_us = tokens * token_ns_with(c, pending + 1) / 1000;
    Some(prefill_us + reply_steps(c, size.reply) * step_with_us(c, pending + 1))
}

/// The steps a reply of `reply` tokens takes on `c`: one a token, or fewer
/// where it speculates (`Capacity::tokens_per_step_milli`). **Per card, not
/// pooled**: what a step commits is the card's, where the turn's size is the
/// turn's. Priced at a token a step, a card committing 2.6 looked 2.6 times
/// slower than it decoded, and took fewer turns the more speculation helped it.
fn reply_steps(c: &Capacity, reply: u64) -> u64 {
    match u64::from(c.tokens_per_step_milli) {
        0 => reply,
        milli => (reply * 1000).div_ceil(milli.max(1000)),
    }
}

/// The time one prompt token takes with `extra` rows more than `c` reports
/// decoding, in nanoseconds: its prefill rate at 1 and its wide rate at
/// [`PLACEMENT_WIDE`], as time per token on the line through them, extended
/// past the wide point. Time, not rate, is what extends: a falling rate
/// reaches zero, a growing time per token does not. A wide rate it has not
/// learned, or one no slower, is flat.
pub(crate) fn token_ns_with(c: &Capacity, extra: u32) -> u64 {
    let next = 1_000_000_000 / u64::from(c.prefill_tok_s);
    if c.prefill_tok_s_wide == 0 || c.prefill_tok_s_wide >= c.prefill_tok_s {
        return next;
    }
    let wide = 1_000_000_000 / u64::from(c.prefill_tok_s_wide);
    let beyond = u64::from(extra.saturating_sub(1));
    next + (wide - next) * beyond / u64::from(PLACEMENT_WIDE - 1)
}

/// The decode step with `extra` rows more than `c` reports decoding, in
/// microseconds: its one-more-row step at 1 and its wide step at
/// [`PLACEMENT_WIDE`], on the line through them, extended past the wide
/// point. A wide step it has not learned, or one no slower, is flat.
pub(crate) fn step_with_us(c: &Capacity, extra: u32) -> u64 {
    let next = u64::from(c.decode_step_next_us);
    let wide = u64::from(c.decode_step_wide_us);
    if wide <= next {
        return next;
    }
    let beyond = u64::from(extra.saturating_sub(1));
    next + (wide - next) * beyond / u64::from(PLACEMENT_WIDE - 1)
}
