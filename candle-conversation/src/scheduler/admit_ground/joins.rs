//! The rule for when a turn may join a forward whose rows are already spent.
//!
//! A forward fills its rows in order, so a turn that joins a full forward
//! advances by nothing until the turns ahead of it finish, while it has already
//! claimed its store and scratch. A full forward therefore takes no more turns.
//! A short section below the live bands may still join one, since a
//! conversation's short frame should not wait for rows, but a file's bulk ingest
//! waits. A closed forward has no rows left whatever its width, so a turn the
//! rate refused is not offered the same rows again and comes back only as a
//! join.

use super::*;

/// The first position of a `rows`-wide chunk ending at the last token of a
/// `turn`-token prefill that starts at `depth` — the deepest chunk the turn
/// runs at that width. A turn no wider than one chunk is its own deepest.
pub(super) fn deepest_chunk_offset(depth: usize, turn: usize, rows: usize) -> usize {
    depth + turn.saturating_sub(rows)
}

/// The most tokens a section below the live bands may hold and still join a
/// full forward: a conversation's frame, not a file. A 60-token `Low` frame
/// waited 201 s for rows on BlackWorld (2026-10-31) with a turn behind it.
pub(super) const SHORT_SECTION_TOKENS: usize = 256;

/// Whether a section of `tokens` at `prio` may join a forward whose rows are
/// spent: every live band's, and a short one's whatever its band. A file's
/// section below them waits for rows, as bulk ingest does.
pub(in super::super) fn joins_a_full_forward(prio: DecodePriority, tokens: usize) -> bool {
    prio.is_live() || tokens <= SHORT_SECTION_TOKENS
}

/// Rows a forward `chunk_rows` wide still has for an offer once `taken` are
/// spent, or `None` when it is full.
///
/// **A full forward takes no more turns.** Its rows are filled in order
/// (`gang::gang_shares`), so a turn joining a full forward advances by
/// nothing until the ones ahead finish — yet it claims its store and scratch
/// at admission, and holds them while it waits. Joined while every member could
/// still advance a chunk, BlackWorld's code_read ingest (2026-10-27) held 13
/// prefills' recurrent stores at once against one forward's rows. It is offered
/// again at the next pass, once rows have come free.
///
/// **A closed forward has no rows left, whatever its width.** Closing ends the
/// rows at what the pass admitted, and the cohort composes at no less than a
/// chunk (`Scheduler::set_rate_width`), so a pass closed below a chunk is
/// still a chunk wide. Read as room, that chunk re-offered the turn the rate
/// had just refused with the same rows, the rate refused it again, and the
/// pass never ended: on BlackWorld (d919bd410, 2026-10-20) one 4-row `Normal`
/// prompt was offered and refused every 1.4 ms for four hours, and no wave ran
/// to step the seventeen decodes the refusal was protecting. Closed with no
/// member in it, the turn comes back as a join and advances by that chunk.
pub(super) fn room_left(
    chunk_rows: usize,
    taken: usize,
    closed: bool,
    members: usize,
) -> Option<usize> {
    let room = if closed {
        0
    } else {
        chunk_rows.saturating_sub(taken)
    };
    if room == 0 && members > 0 {
        return None;
    }
    Some(room)
}
