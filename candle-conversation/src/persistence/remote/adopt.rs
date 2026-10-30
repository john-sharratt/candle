//! Adopting another member's turn once all of it has arrived
//! (`docs/replicated_substrate.md` §6.5).
//!
//! The turn goes in the slot its stamp gives it in the timeline's order, its
//! residence created cold: the cold references point at the local copies of
//! its `Chunk` records, and its K/V loads on first use like any turn a reload
//! restored. A position below it that holds nothing yet is a gap, filled when
//! its own turn completes.
//!
//! In two steps: the plan reads what the turn needs off the log, and changes
//! nothing; the placing is a substrate edit, which may run more than once.

use std::collections::BTreeSet;

use super::super::resume::{recover_turn_cold_refs, recover_turn_meta};
use super::super::streams::TurnDecl;
use super::super::{Result, SubstratePersistence};
use super::completeness::turn_is_complete;
use crate::projection::{GroupId, LayerId, TimelineId, TurnIndex};
use crate::substrate::{RestoredTurn, StreamKey, Substrate};
use crate::token_buffer::TokenBuffer;
use crate::turn_layout::TurnLayout;

/// One turn to adopt, read off the log and ready to place.
pub(super) struct Adoption {
    timeline: TimelineId,
    target: Option<(LayerId, GroupId)>,
    index: TurnIndex,
    turn: RestoredTurn,
}

impl Adoption {
    /// Register the turn's timeline and place the turn — `None` when a turn
    /// already holds its position. Edits only `substrate`, so it can run again
    /// on a newer version.
    pub(super) fn place(&self, substrate: &mut Substrate) -> Option<(TimelineId, TurnIndex)> {
        if let Some((layer, group)) = self.target {
            substrate.register_timeline(self.timeline, layer, group);
        }
        substrate
            .place_turn(self.timeline, self.index, self.turn.clone())
            .then_some((self.timeline, self.index))
    }
}

/// Plan the adoption of every turn of `keys` that is another member's,
/// complete, live and not held here yet.
pub(super) fn plan_adoptions(
    p: &mut SubstratePersistence,
    substrate: &Substrate,
    keys: &BTreeSet<StreamKey>,
    n_layers: usize,
) -> Result<Vec<Adoption>> {
    let mut adopted = Vec::new();
    for &key in keys {
        let Some((timeline, decl)) = adoptable(substrate, key, n_layers) else {
            continue;
        };
        let cold = match recover_turn_cold_refs(substrate, &decl, n_layers) {
            Ok(cold) => cold,
            Err(e) => {
                tracing::warn!(
                    timeline_id = decl.timeline_id,
                    turn_index = decl.turn_index,
                    writer = format!("{:x}", key.writer),
                    "another member's turn holds chunks outside its declared grid, so it stays \
                     pending: {e}"
                );
                continue;
            }
        };
        // The turn's `Tokens` may have arrived in this very batch, still staged.
        p.flush_staged()?;
        let meta = recover_turn_meta(p, substrate, &decl)?;
        let target = LayerId::from_raw(decl.layer_id).zip(GroupId::from_raw(decl.group_id));
        adopted.push(Adoption {
            timeline,
            target,
            index: TurnIndex(decl.turn_index),
            turn: RestoredTurn {
                layout: TurnLayout::new(decl.segments),
                token_ids: TokenBuffer::from(meta.token_ids),
                token_count: meta.token_count,
                cold,
                block_start: decl.block_start,
                block_end: decl.block_end,
            },
        });
    }
    Ok(adopted)
}

/// `key`'s turn, with its declaration's index set to the position it holds,
/// when it is another member's turn, complete, live, and not held here.
fn adoptable(
    substrate: &Substrate,
    key: StreamKey,
    n_layers: usize,
) -> Option<(TimelineId, TurnDecl)> {
    let turn = substrate.turn_at(key)?;
    let timeline = TimelineId::from_raw(turn.decl.timeline_id)?;
    if substrate.is_timeline_dead(timeline.raw())
        || substrate.is_turn_tombstoned(timeline, turn.index.0)
        || substrate.holds_turn(timeline, turn.index)
    {
        return None;
    }
    let distilled = substrate.distill_mode(timeline).is_some();
    if !turn_is_complete(turn.decl, turn.runtime, n_layers, distilled) {
        return None;
    }
    Some((
        timeline,
        TurnDecl {
            turn_index: turn.index.0,
            ..turn.decl.clone()
        },
    ))
}
