//! The recurrent state at a point in the system prompt.
//!
//! A section is sealed by borrowing its prefix's K/V and prefilling only its own
//! tokens. Attention reads the borrowed prefix; a recurrence cannot borrow, so a
//! section that starts its recurrence from zero is computed — every layer the
//! recurrence feeds, its K/V included — as if nothing came before it. A section
//! sealed from the state its prefix leaves behind is computed as it will be
//! read.
//!
//! State does not compose (`state([A][B])` is no function of `state([A])` and
//! `state([B])`), so a state is valid for one ordered prefix and is keyed by
//! that prefix's content chain — the key the sealed K/V is addressed by, so the
//! two go stale together. [`state_after`] keeps the state after a run of sealed
//! sections in the substrate's checkpoint store, where every conversation on
//! the same prompt finds it; [`continue_state`] runs a short continuation from
//! a state without keeping it, for a chain of a conversation's own sections
//! whose states belong to that conversation alone.

use std::sync::{Arc, Mutex};

use flume::{Sender, WeakSender};

use crate::error::ConversationError;
use crate::persistence::content_hash::{ContentChain, ContentHash};
use crate::persistence::record::{BranchCheckpointPayload, SnapshotLayer};
use crate::projection::{Conversation, SectionId};
use crate::scheduler::exported_state::{ExportedState, SharedState};
use crate::scheduler::SchedulerRequest;
use crate::token_buffer::TokenBuffer;
use candle_transformers::models::delta_net::ExportedLayerState;

/// The state a stored checkpoint describes, as a slot is given it.
pub(crate) fn shared(payload: &BranchCheckpointPayload) -> SharedState {
    let layers: Arc<[ExportedLayerState]> = payload
        .layers
        .iter()
        .map(|l| ExportedLayerState {
            layer_index: l.layer_index,
            n_v_heads: l.n_v_heads,
            d_v: l.d_v,
            d_k: l.d_k,
            state: l.state.clone(),
            conv_channels: l.conv_channels,
            conv_tail_cols: l.conv_tail_cols,
            conv_tail: l.conv_tail.clone(),
        })
        .collect();
    SharedState {
        schedule_hash: payload.schedule_hash,
        layers,
        aux: payload.aux.clone().into(),
    }
}

/// Run `tokens` through the model from `start` (from zero with `None`) and
/// return the state they leave. `Ok(None)` when the model carries no recurrent
/// state, so a caller asks unconditionally.
pub(crate) fn run_state(
    scheduler_tx: &Sender<SchedulerRequest>,
    start: Option<SharedState>,
    tokens: Vec<u32>,
) -> crate::Result<Option<ExportedState>> {
    let (tx, rx) = flume::bounded(1);
    scheduler_tx
        .send(SchedulerRequest::BranchCheckpointPass {
            start,
            tokens: TokenBuffer::from(tokens),
            response_tx: tx,
        })
        .map_err(|_| ConversationError::SchedulerGone)?;
    rx.recv().map_err(|_| ConversationError::SchedulerGone)?
}

/// The checkpoint for the prompt prefix `prefix`, whose ordered tokens are
/// `tokens`: the stored one, or one computed now and stored — and which of the
/// two. `Ok(None)` when the model carries no recurrent state.
///
/// A computed payload is handed back in memory and memoised as well as
/// enqueued, because the durable write is fire-and-forget: a read issued
/// straight after it finds nothing.
pub(crate) fn checkpoint(
    scheduler_tx: &Sender<SchedulerRequest>,
    substrate: &Conversation,
    prefix: ContentHash,
    tokens: Vec<u32>,
) -> crate::Result<Option<(Arc<BranchCheckpointPayload>, bool)>> {
    match substrate.read_branch_checkpoint(prefix) {
        Ok(Some(p)) => return Ok(Some((p, false))),
        Ok(None) => {}
        // Present but unreadable: computed again over it.
        Err(e) => tracing::warn!("checkpoint unreadable, recomputing: {e}"),
    }
    let Some(state) = run_state(scheduler_tx, None, tokens)? else {
        return Ok(None);
    };
    let payload = Arc::new(BranchCheckpointPayload {
        prefix_hash: prefix,
        schedule_hash: state.schedule_hash,
        layers: state.layers.into_iter().map(SnapshotLayer::from).collect(),
        aux: state.aux,
    });
    substrate.enqueue_branch_checkpoint(prefix, payload.clone());
    substrate.memo_branch_checkpoint(prefix, &payload);
    Ok(Some((payload, true)))
}

/// The ordered tokens of `sections` and the content chain they make, leaving
/// out any with no tokens.
pub(crate) fn chain_of(
    substrate: &Conversation,
    sections: &[SectionId],
) -> (ContentHash, Vec<u32>) {
    let view = substrate.read();
    let mut chain = ContentChain::new();
    let mut tokens = Vec::new();
    for &id in sections {
        let section = view.section_tokens_of(id);
        if section.is_empty() {
            continue;
        }
        chain.push_section(&section);
        tokens.extend_from_slice(&section);
    }
    (chain.prefix(), tokens)
}

/// How many states [`state_after`] holds decoded in this process. One per
/// collection prefix a seal asks after — a handful per prompt — each the size
/// of a checkpoint.
const HELD_STATES: usize = 8;

/// One held state: the engine whose model computed it, the prompt prefix it
/// follows, and the state.
///
/// Keyed by the engine as well as the prefix because a prefix hashes tokens
/// alone: two engines in one process on models sharing a tokenizer
/// (`docs/parasite_model.md` §12) see the same prefix and must never be handed
/// each other's recurrent state. The engine is its scheduler's channel, held
/// weakly so a cached state never keeps an engine's channel open.
struct Held {
    engine: WeakSender<SchedulerRequest>,
    prefix: ContentHash,
    state: SharedState,
}

impl Held {
    fn is(&self, engine: &Sender<SchedulerRequest>, prefix: ContentHash) -> bool {
        self.prefix == prefix
            && self
                .engine
                .upgrade()
                .is_some_and(|tx| tx.same_channel(engine))
    }
}

/// The states [`state_after`] last served, newest last. The checkpoint store
/// memoises one record and re-reads any other off the log — a ~63 MiB decode
/// under the persistence lock — while a journal asks after its collection's
/// prefix every time an entry seals.
static HELD: Mutex<Vec<Held>> = Mutex::new(Vec::new());

/// `state` as the one `engine` holds under `prefix`, newest, past the cap
/// dropping the oldest. A state whose engine is gone is dropped too.
fn hold(engine: &Sender<SchedulerRequest>, prefix: ContentHash, state: SharedState) {
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    held.retain(|h| !h.is(engine, prefix) && h.engine.upgrade().is_some());
    held.push(Held {
        engine: engine.downgrade(),
        prefix,
        state,
    });
    if held.len() > HELD_STATES {
        held.remove(0);
    }
}

fn held(engine: &Sender<SchedulerRequest>, prefix: ContentHash) -> Option<SharedState> {
    let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    held.iter()
        .find(|h| h.is(engine, prefix))
        .map(|h| h.state.clone())
}

/// The state the checkpoint for `prefix` describes, shared with every slot
/// `engine` gives the same prompt branch: the held one, or one materialised
/// from `payload` and held. Materialised per install, every conversation opened
/// on a branch copied the whole state (~112 MiB on Qwen4Exp) and freed it again
/// when its slot's snapshot was dropped — ~5 ms on the scheduler thread, ~400
/// times in one calibration burst (BlackWorld, 2026-11-16).
pub(crate) fn shared_held(
    engine: &Sender<SchedulerRequest>,
    prefix: ContentHash,
    payload: &BranchCheckpointPayload,
) -> SharedState {
    if let Some(state) = held(engine, prefix) {
        return state;
    }
    let state = shared(payload);
    hold(engine, prefix, state.clone());
    state
}

/// The state after the sealed `sections`, in order — `Ok(None)` before any
/// token, where the state is zero, and for a model with no recurrent state.
pub(crate) fn state_after(
    scheduler_tx: &Sender<SchedulerRequest>,
    substrate: &Conversation,
    sections: &[SectionId],
) -> crate::Result<Option<SharedState>> {
    let (prefix, tokens) = chain_of(substrate, sections);
    if tokens.is_empty() {
        return Ok(None);
    }
    if let Some(state) = held(scheduler_tx, prefix) {
        return Ok(Some(state));
    }
    let state = checkpoint(scheduler_tx, substrate, prefix, tokens)?.map(|(p, _)| shared(&p));
    if let Some(state) = &state {
        hold(scheduler_tx, prefix, state.clone());
    }
    Ok(state)
}

/// The state after `start` and then the sealed `sections`, computed now and
/// not kept. `start` itself when `sections` hold no tokens.
pub(crate) fn continue_state(
    scheduler_tx: &Sender<SchedulerRequest>,
    substrate: &Conversation,
    start: Option<SharedState>,
    sections: &[SectionId],
) -> crate::Result<Option<SharedState>> {
    let (_, tokens) = chain_of(substrate, sections);
    if tokens.is_empty() {
        return Ok(start);
    }
    Ok(run_state(scheduler_tx, start, tokens)?.map(SharedState::from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::hash_bytes;

    fn payload(prefix: ContentHash, fill: u8) -> BranchCheckpointPayload {
        BranchCheckpointPayload {
            prefix_hash: prefix,
            schedule_hash: 7,
            layers: vec![SnapshotLayer {
                layer_index: 0,
                n_v_heads: 1,
                d_v: 1,
                d_k: 1,
                state: vec![fill; 4],
                conv_channels: 1,
                conv_tail_cols: 1,
                conv_tail: vec![fill; 4],
            }],
            aux: Vec::new(),
        }
    }

    /// **Every install on one prompt branch shares one copy of its state**:
    /// the second asks for the rows the first materialised, not a copy of the
    /// payload; another branch gets its own.
    #[test]
    fn one_branchs_installs_share_its_rows() {
        let (engine, _rx) = flume::unbounded();
        let a = hash_bytes(b"prompt branch a, for one_branchs_installs_share_its_rows");
        let b = hash_bytes(b"prompt branch b, for one_branchs_installs_share_its_rows");
        let first = shared_held(&engine, a, &payload(a, 1));
        let second = shared_held(&engine, a, &payload(a, 9));
        assert!(Arc::ptr_eq(&first.layers, &second.layers));
        assert_eq!(
            second.layers[0].state,
            vec![1u8; 4],
            "the held rows, not the new payload's"
        );
        let other = shared_held(&engine, b, &payload(b, 3));
        assert!(!Arc::ptr_eq(&first.layers, &other.layers));
        assert_eq!(other.layers[0].state, vec![3u8; 4]);
    }

    /// **Two engines on one prompt never share a state.** The prefix hashes
    /// tokens alone, so a second engine whose model shares the tokenizer sees
    /// the same prefix — and gets its own model's state, not the first's.
    #[test]
    fn two_engines_on_one_prefix_hold_their_own_states() {
        let (first_engine, _first_rx) = flume::unbounded();
        let (second_engine, _second_rx) = flume::unbounded();
        let p = hash_bytes(b"one prompt, for two_engines_on_one_prefix_hold_their_own_states");
        let first = shared_held(&first_engine, p, &payload(p, 1));
        let second = shared_held(&second_engine, p, &payload(p, 2));
        assert!(!Arc::ptr_eq(&first.layers, &second.layers));
        assert_eq!(second.layers[0].state, vec![2u8; 4]);
        let again = shared_held(&first_engine, p, &payload(p, 9));
        assert!(Arc::ptr_eq(&first.layers, &again.layers));
    }

    /// A state held for an engine that is gone is never served, and holds no
    /// channel open.
    #[test]
    fn a_gone_engines_state_is_not_served() {
        let p = hash_bytes(b"one prompt, for a_gone_engines_state_is_not_served");
        let (gone, gone_rx) = flume::unbounded();
        shared_held(&gone, p, &payload(p, 4));
        drop(gone_rx);
        let weak = gone.downgrade();
        drop(gone);
        assert!(weak.upgrade().is_none(), "the held state kept the channel");
        let (engine, _rx) = flume::unbounded();
        let fresh = shared_held(&engine, p, &payload(p, 5));
        assert_eq!(fresh.layers[0].state, vec![5u8; 4]);
    }
}
