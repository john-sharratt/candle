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
/// draw, and where the app's round robin stands.
pub(crate) struct Weigh<'a> {
    pub cards: &'a Cards,
    pub machines: &'a MachineTable,
    /// Cards held for a draw that waits for their turns to end
    /// (`docs/static_placement.md` §4): they take no new unit but a draw.
    pub reserved: &'a BTreeSet<CardKey>,
    /// The machine this app's round robin last gave a unit to, as
    /// `(rank, node)` (§3.2).
    pub cursor: &'a Cell<Option<(u32, u64)>>,
    /// The draw being placed has waited its bound for a card's turns to end
    /// (`RESERVE_FOR`): it takes its card as it stands.
    pub draw_overdue: bool,
}

/// Of `members`, those not lagging on replication — or all of them when every
/// one is (`docs/static_placement.md` §3.2). Applied to the members that have
/// room, so a lagging one with room still takes a unit every other is too full
/// for, and after the holders of a released key are found, so a lagging member
/// that alone holds a conversation still takes it. On 2026-11-02 a new
/// conversation placed on a lagging WhiteWorld stayed there, turn after turn,
/// because no other member ever held it complete.
fn prefer_current(members: Vec<&MemberView>) -> Vec<&MemberView> {
    if members.iter().all(|m| m.lagging) {
        return members;
    }
    members.into_iter().filter(|m| !m.lagging).collect()
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
    /// No machine of the pool takes the unit at all — named nowhere in the
    /// table, or without the key the unit needs. The registry refuses it only
    /// when no machine of the whole table takes it (`Registry::unplaced`).
    Unplaced,
    /// Nobody is eligible.
    Nobody,
}

impl Placed {
    /// Whether the unit found no member at all — nobody eligible, or nobody
    /// whose machine takes it — so a caller with a wider pool to try tries it.
    pub fn missed(self) -> bool {
        matches!(self, Placed::Nobody | Placed::Unplaced)
    }
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
