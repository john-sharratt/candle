//! Which of the members a request may go to receives it: the placement policy
//! (§5.3, `docs/cooperative_admission.md` §4). The registry decides who is
//! eligible — ready, holding the key's state, within a draw's reach — and
//! hands the pool here, with what is placed on each card; this decides among
//! them, or says that none has room.

mod best;
mod draw;
mod finish;
mod reach;
mod room;
mod turn_size;

#[cfg(test)]
pub(crate) use finish::token_ns_with;
pub(crate) use finish::{finish_us, step_with_us};
pub(crate) use reach::{guest_reach, within_reach};
pub(crate) use room::{has_room, held, width};
pub use turn_size::TurnSize;
#[cfg(test)]
pub use turn_size::{NOMINAL_PROMPT_TOKENS, NOMINAL_REPLY_TOKENS};

use best::best_of;
use draw::draw_of;

use super::registry::{CardKey, CardLoad, Cards, Fences, MemberView};
use crate::frame::Capacity;

/// What a placement weighs beside the members themselves: what the gateway has
/// placed on each card, and the share of each card the fence holds back.
pub(crate) struct Weigh<'a> {
    pub cards: &'a Cards,
    pub fences: &'a Fences,
}

impl Weigh<'_> {
    /// What is placed on `m`'s card.
    pub fn card(&self, m: &MemberView) -> CardLoad {
        self.cards.get(&CardKey::of(m)).copied().unwrap_or_default()
    }

    /// The thousandths of `m`'s card the fence holds back.
    pub fn fence(&self, m: &MemberView) -> u32 {
        self.fences.get(&CardKey::of(m)).copied().unwrap_or(0)
    }
}

/// Where a placement put a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placed {
    To(u64),
    /// Every eligible card is full: the unit waits at the gateway (§4.3).
    Full,
    /// Nobody is eligible.
    Nobody,
}

/// The member of `pool` a request goes to: a draw by [`draw_of`], anything
/// else by [`best_of`].
pub(crate) fn place(pool: Vec<&MemberView>, draw: Option<u64>, weigh: &Weigh<'_>) -> Placed {
    if pool.is_empty() {
        return Placed::Nobody;
    }
    match draw {
        Some(_) => draw_of(pool, weigh).map_or(Placed::Nobody, Placed::To),
        None => best_of(pool, weigh),
    }
}

fn capacity_of(m: &MemberView) -> &Capacity {
    m.capacity.as_ref().expect("a ready member sent capacity")
}
