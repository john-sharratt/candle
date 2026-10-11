//! Adopting another member's turn once all of it has arrived
//! (`docs/replicated_substrate.md` §6.5).
//!
//! The turn goes in the slot its stamp gives it in the timeline's order, its
//! residence created cold: the cold references point at the local copies of
//! its `Chunk` records, and its K/V loads on first use like any turn a reload
//! restored. A position below it that holds nothing yet is a gap, filled when
//! its own turn completes.

use std::collections::BTreeSet;

use super::super::resume::{recover_turn_cold_refs, recover_turn_meta};
use super::super::streams::TurnDecl;
use super::super::{Result, SubstratePersistence};
use super::completeness::turn_is_complete;
use crate::projection::{GroupId, LayerId, TimelineId, TurnIndex};
use crate::substrate::{RestoredTurn, StreamKey, Substrate};
use crate::token_buffer::TokenBuffer;
use crate::turn_layout::TurnLayout;

/// Adopt every turn of `keys` that is another member's, complete, live and
/// not held here yet. Returns where each adopted turn was placed.
pub(super) fn adopt_complete_turns(
    p: &mut SubstratePersistence,
    substrate: &mut Substrate,
    keys: &BTreeSet<StreamKey>,
    n_layers: usize,
) -> Result<Vec<(TimelineId, TurnIndex)>> {
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
        if let (Some(layer), Some(group)) = (
            LayerId::from_raw(decl.layer_id),
            GroupId::from_raw(decl.group_id),
        ) {
            substrate.register_timeline(timeline, layer, group);
        }
        let index = TurnIndex(decl.turn_index);
        let turn = RestoredTurn {
            layout: TurnLayout::new(decl.segments),
            token_ids: TokenBuffer::from(meta.token_ids),
            token_count: meta.token_count,
            cold,
            block_start: decl.block_start,
            block_end: decl.block_end,
        };
        if substrate.place_turn(timeline, index, turn) {
            adopted.push((timeline, index));
        }
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
