//! Rule 3 (`docs/static_placement.md` §3.2, §4): whether a member's card has
//! room for one more unit — by what its machine takes, never by what it
//! measures.

use super::Weigh;
use crate::gateway::registry::{MemberView, MODEL_APPS};

/// Whether `m`'s card has room for one more unit, a draw when `draw`.
///
/// - **A draw** needs a card that is not drawing — a card draws one picture
///   at a time — on a machine that draws at all.
/// - **Anything else** needs a card that is neither drawing (a draw holds the
///   card's admission while it runs) nor held for a waiting draw, on a machine
///   that takes conversations, below its `max` units of work.
/// - **A member of an app that runs no model** has room while its card is not
///   drawing: the table places the model's work, not an agent's.
pub(crate) fn has_room(m: &MemberView, weigh: &Weigh<'_>, draw: bool) -> bool {
    let card = weigh.card(m);
    if !MODEL_APPS.contains(&m.app.as_str()) {
        return card.drawing == 0;
    }
    if draw {
        return card.drawing == 0 && weigh.machines.pictures(&m.host).is_some();
    }
    if card.drawing > 0 || weigh.reserved(m) {
        return false;
    }
    weigh
        .machines
        .conversations(&m.host)
        .is_some_and(|c| card.work() < c.max())
}
