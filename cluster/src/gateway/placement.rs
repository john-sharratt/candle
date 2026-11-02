//! Which of the members a request may go to receives it: the static placement
//! of `docs/static_placement.md`. The registry decides who is eligible — ready,
//! holding the key's state, within a draw's reach — and hands the pool here,
//! with what is placed on each card and the machine table; this decides among
//! them, or says that none has room.

mod conversation;
mod draw;
mod least;
mod reach;
mod room;

pub(crate) use reach::{guest_reach, within_reach};
pub(crate) use room::has_room;

use std::cell::Cell;
use std::collections::BTreeSet;

use conversation::conversation_of;
use draw::draw_of;
pub(crate) use draw::draw_target;
use least::least_of;

use super::registry::{CardKey, CardLoad, Cards, MemberView, MODEL_APPS};
use crate::machines::MachineTable;

/// What a placement weighs beside the members themselves: what the gateway has
/// placed on each card, what each machine takes, the cards held for a waiting
/// draw, and where the round robin stands.
pub(crate) struct Weigh<'a> {
    pub cards: &'a Cards,
    pub machines: &'a MachineTable,
    /// Cards held for a draw that waits for their turns to end
    /// (`docs/static_placement.md` §4): they take no new unit but a draw.
    pub reserved: &'a BTreeSet<CardKey>,
    /// The machine the round robin last gave a unit to, as `(rank, node)`
    /// (§3.2).
    pub cursor: &'a Cell<Option<(u32, u64)>>,
}

impl Weigh<'_> {
    /// What is placed on `m`'s card.
    pub fn card(&self, m: &MemberView) -> CardLoad {
        self.cards.get(&CardKey::of(m)).copied().unwrap_or_default()
    }

    /// Whether `m`'s card is held for a waiting draw.
    pub fn reserved(&self, m: &MemberView) -> bool {
        self.reserved.contains(&CardKey::of(m))
    }
}

/// Where a placement put a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placed {
    To(u64),
    /// Every eligible card is full: the unit waits at the gateway (§3.4).
    Full,
    /// Nobody is eligible.
    Nobody,
}

/// The member of `pool`, members of `app`, a request goes to: a draw by
/// [`draw_of`], any other unit of the model's apps by [`conversation_of`], and
/// a unit of an app that runs no model by [`least_of`].
pub(crate) fn place(
    app: &str,
    pool: Vec<&MemberView>,
    draw: Option<u64>,
    weigh: &Weigh<'_>,
) -> Placed {
    if pool.is_empty() {
        return Placed::Nobody;
    }
    if !MODEL_APPS.contains(&app) {
        return least_of(pool);
    }
    match draw {
        Some(_) => draw_of(pool, weigh),
        None => conversation_of(pool, weigh),
    }
}
