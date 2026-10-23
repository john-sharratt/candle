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

use flume::Sender;

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

/// The states [`state_after`] last served, newest last. The checkpoint store
/// memoises one record and re-reads any other off the log — a ~63 MiB decode
/// under the persistence lock — while a journal asks after its collection's
/// prefix every time an entry seals.
static HELD: Mutex<Vec<(ContentHash, SharedState)>> = Mutex::new(Vec::new());

/// `state` as the one held under `prefix`, newest, past the cap dropping the
/// oldest.
fn hold(prefix: ContentHash, state: SharedState) {
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    held.retain(|(p, _)| *p != prefix);
    held.push((prefix, state));
    if held.len() > HELD_STATES {
        held.remove(0);
    }
}

fn held(prefix: ContentHash) -> Option<SharedState> {
    let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    held.iter()
        .find(|(p, _)| *p == prefix)
        .map(|(_, s)| s.clone())
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
    if let Some(state) = held(prefix) {
        return Ok(Some(state));
    }
    let state = checkpoint(scheduler_tx, substrate, prefix, tokens)?.map(|(p, _)| shared(&p));
    if let Some(state) = &state {
        hold(prefix, state.clone());
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
    Ok(run_state(scheduler_tx, start, tokens)?.map(|s| SharedState::from(&s)))
}
