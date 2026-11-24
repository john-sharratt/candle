//! The types a turn streams to its client, and how a turn begins and resumes.
//!
//! `StreamItem` is what a turn yields: token text, progress for the prefill and
//! the reasoning, tool lifecycle notices, a projection event per decode, the
//! finished turn's usage, and errors. Errors are sent as their own items so they
//! are not folded into a reasoning block. A tool notice is sent as running before
//! its batch executes and as done once results are back. `TurnStart` says whether
//! a turn begins from a message or resumes a sealed call turn, and `Resume` says
//! whether that call's tools ran before a restart.

use super::*;

// The schema-driven ingest registry lives in [`crate::ingest`]; a raw layer's
// live state ([`IngestConv`]) is keyed by layer name in
// `InferenceState::ingest_convs`. The branch layers (`repo_map`,
// `code_reading`) keep no live state: each pass reads what they hold from the
// substrate ([`crate::branch_ingest`]).

// The number of tools surfaced into the system prompt is governed by
// the `selection: { kind: top_k, k: N }` rule on the `tools` collection
// in `prompts/projection.yaml`.  Tune K there.

// ── Stream item ───────────────────────────────────────────────────────────────

/// A tool-execution lifecycle notice for the GUI.  Tool calls run *between*
/// streamed turns (the model emits the call, the daemon executes it, then a
/// `<tool_response>` turn continues), a gap during which the client otherwise
/// shows a static "no result" with no sign of activity.  Emitted with
/// `phase = "running"` just before the batch executes and `phase = "done"` once
/// results are back, so the GUI can show a spinner on the in-flight tool cards.
#[derive(Clone, serde::Serialize)]
pub struct ToolStatusOut {
    /// `"running"` while the batch executes, `"done"` once results are back.
    pub phase: &'static str,
    /// The tool names in this batch, in call order.
    pub tools: Vec<String>,
    /// Each tool's JSON response, in the same order as `tools` — populated only
    /// on the `"done"` notice (empty/omitted while running).  Lets the GUI
    /// resolve the in-flight cards immediately, before the post-stream hydrate.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<Value>,
    /// Each result's length in tokens as it will sit in the context, in the
    /// same order — encoded as the next turn will encode it, so it is the length
    /// the reloaded history shows for that result. `"done"` notice only.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<usize>,
}

/// Items yielded by [`ZendSession::submit`].
#[derive(Clone)]
pub enum StreamItem {
    Status(String),
    Token(String),
    /// A projection event emitted once per decode (at seal): the materialized
    /// context's composition + decode throughput, driving the GUI timeline dots
    /// and the projection popover (docs/zend_ui_redesign.md §2.3).
    Projection(ProjectionEventOut),
    /// A tool-execution lifecycle notice (running / done) for the in-flight
    /// tool cards.  Display-only: never part of the collected completion body.
    Tool(ToolStatusOut),
    /// How far the prefill of the turn's input has got, in tokens. A tool
    /// round's results can run to tens of thousands of tokens, and this is the
    /// only sign of progress the GUI has before the answer starts. Display-only.
    Prefill {
        done: usize,
        total: usize,
    },
    /// The turn's reasoning so far, in generated tokens — sent while its
    /// `<think>` block is open, and once more (`done`) with the total when it
    /// closes. Counted as the turn records it (see `think_progress`), so the
    /// total is what a reloaded history shows. Display-only.
    Think {
        tokens: usize,
        done: bool,
    },
    /// A finished turn: its token counts, which become the reply's `usage`, and
    /// how it ended, which becomes its `finish_reason`. Sent once per turn; a
    /// reply the daemon answers over several turns sends one for each and ends
    /// the way its last turn did.
    TurnEnd {
        usage: Usage,
        finish: FinishReason,
    },
    /// The turn failed: what the engine reported, worded for the user. Its own
    /// item rather than text, because text lands wherever the reply has got to
    /// — inside an open `<think>` block, where the GUI folds it away — and an
    /// error is not part of the reply. A streamed reply sends it in-band; a
    /// collected one answers `500` with it, since a body-reading client has the
    /// status to go on.
    Error(String),
    /// The repeat guard or the round cap closed the turn's tool loop, so the
    /// turn that follows answers without tools — sent once, before it, naming
    /// which ([`CLOSED_BY_REPEATS`], [`CLOSED_BY_CAP`]). A task `/mcp` started
    /// reads it as stuck (`crate::mcp::workers`); the chat stream shows nothing
    /// for it.
    Closing(&'static str),
    /// A worker's turn stopped at a round boundary to move to a markedly
    /// faster member (`crate::worker_hop`): its call turn `call_turn` is
    /// sealed, its tools not run, and the member `to` is to run them once it
    /// holds the turn's commit (`ZendSession::committed_through`). Sent once,
    /// last; only a worker's turn sends it.
    Handed {
        to: u64,
        call_turn: u32,
        /// The round it stopped at, so the round cap counts on where it
        /// moves.
        iteration: usize,
    },
    /// The member is leaving its clusters: the turn stopped where it stood,
    /// nothing of its round sealed, its entries kept and its conversation
    /// marked for the member that takes it (`docs/turn_handover.md` §4). Sent
    /// once, last.
    Left,
}

/// What a client streaming a turn is told when the member leaves under it.
pub const LEFT_STATUS: &str = "this member is restarting: the reply continues on another member";

/// A sealed call turn whose round a turn finishes (`TurnStart::resume`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resume {
    pub call_turn: u32,
    /// How its calls run: `Resumed` after a restart, which may have run some
    /// of them; `Live` for a round another member handed here before running
    /// any.
    pub dispatch: Dispatch,
    /// The round the call turn was: the turn goes on from the next.
    pub iteration: usize,
}

/// How a turn begins (`run_inference_stream`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnStart {
    /// A sealed call turn whose round this turn finishes; `None` for a turn
    /// that begins from its message.
    pub resume: Option<Resume>,
    /// A worker's turn, which moves to a markedly faster member at a round
    /// boundary (`crate::worker_hop`). Nothing else may: a chat client reads
    /// its own turn, and a turn the boot scan finishes has nobody to hand to.
    pub may_hand: bool,
    /// The images the message carries, read before its first round.
    pub images: Vec<ImageRef>,
}

/// [`StreamItem::Closing`]'s reason when the repeat guard closed the loop.
pub const CLOSED_BY_REPEATS: &str = "repeats";

/// [`StreamItem::Closing`]'s reason when the round cap closed it.
pub const CLOSED_BY_CAP: &str = "round_cap";

/// [`StreamItem::Closing`]'s reason when Claude stopped the worker
/// (`job_cancel` on a task, [`stops`]).
pub const CLOSED_BY_CLAUDE: &str = "stopped";

/// Forward a turn's items to the request that started it, recording each one
/// in `live` for a client that reconnects.
///
/// Drains `ts` to the end whether or not the request is still there: the turn
/// runs to completion either way, and a client that comes back mid-turn follows
/// it from `live`. An error item is recorded as the turn error the request
/// showed for it, so a follower sees the same failure. A turn no request
/// started — one resumed after a restart — passes no `tx` and is followed from
/// `live` alone.
pub(super) async fn relay_live(
    ts: &mut Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>>,
    tx: Option<&mpsc::Sender<anyhow::Result<StreamItem>>>,
    live: &LiveTurn,
    conv_id: &str,
) {
    let mut listening = tx.is_some();
    while let Some(item) = ts.next().await {
        live.push(match &item {
            Ok(i) => i.clone(),
            Err(e) => StreamItem::Error(e.to_string()),
        });
        let Some(tx) = tx.filter(|_| listening) else {
            continue;
        };
        if tx.send(item).await.is_err() {
            listening = false;
            tracing::info!(
                conv_id,
                "client went away mid-stream — the turn runs to completion and a \
                 reconnecting client follows it live"
            );
        }
    }
}

/// Process-global monotonic id for projection events, so dot ids stay unique
/// across conversations and requests.
pub(super) static PROJ_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

// ── Inference state ───────────────────────────────────────────────────────────

pub(super) struct ConvState {
    pub(super) conv: Sequence,
    /// The identity this conversation speaks as (a sub-folder of `identities/`),
    /// or `None` for the default / no identity. Seeded from the substrate on
    /// fork, overridden when a request carries an explicit `identity`, and used
    /// each turn to scope the projection's identity collection.
    pub(super) identity: Option<String>,
    /// This conversation's own file stores: what its `file_*` calls write,
    /// edit and delete, kept as deltas over each repository and seen by no other
    /// conversation. Every round of its tools runs against these
    /// ([`ToolHost::context_for`]); each round's changes are saved with the
    /// conversation and restored when this state is built again
    /// ([`conv_overlay`]).
    pub(super) files: Arc<RepoFiles>,
    /// What the substrate holds of `files`, as events on the conversation's
    /// timeline — what each save writes the difference against. Unread until
    /// the conversation's events have been read back, and while unread
    /// nothing is saved.
    pub(super) mirror: Arc<Mutex<Mirror>>,
    /// The journal sections this conversation holds, reconciled with its
    /// journal at the head of every round (`docs/zend_journal.md`).
    pub(super) journal: Carried,
    /// The files its calls have read or edited: what each round's calls to
    /// them are held to (`held_values`, `docs/stencil_tree.md` §9.2).
    pub(super) seen: FilesSeen,
}

/// The most tool rounds one turn runs before the daemon stops feeding it.
///
/// The bound matters because nothing else provides one — a client that closes
/// its tab no longer ends the turn (that is deliberate: the work finishes and
/// seals so a reload finds it done), so a model calling a tool every round
/// would otherwise decode, seal and dispatch forever, holding the
/// conversation's lock and the card with nobody waiting on the result.
///
/// **Reaching it closes the loop; it never just stops.** A long job reaches it
/// too: measured 2026-10-24, a turn revising a 1,400-line paper made ~60
/// edits over 31 rounds, every one landing, and the turn ended on its last
/// call's results with nothing said to the user. The round before the cap is
/// answered by a closing turn ([`closes_at_cap`]), as the repeat guard's is:
/// no tool call, a bounded answer saying what was done and what remains.
///
/// **128, not 32.** A worker's restructure of the journals — dozens of reads,
/// edits and deletes, a preview, a push — met the cap of 32 at its thirtieth
/// round on 2026-10-31 and sat stuck waiting to be told to go on. The bound is
/// a guard against a turn that never ends, not a budget for one that is
/// working; the repeat guard still closes a loop that only repeats itself.
pub(super) const MAX_TOOL_ROUNDS: usize = 128;

/// Whether the results of round `iteration` are the last a turn may answer
/// under [`MAX_TOOL_ROUNDS`] — so the turn answering them must close the loop.
pub(super) fn closes_at_cap(iteration: usize) -> bool {
    iteration + 2 >= MAX_TOOL_ROUNDS
}

/// The hard cap on one assistant turn, in generated tokens — the scheduler's
/// `max_tokens` for every turn that does not name its own.
///
/// A prose answer never gets near it: the think and answer budgets
/// ([`think_budget::steer`]) end one long before. It exists for the turn those
/// budgets stand down for — one writing a tool call, whose string values may
/// each run to
/// [`MAX_STRING_VALUE_TOKENS`](candle_conversation::stencil::MAX_STRING_VALUE_TOKENS)
/// so a whole file can be written. So it has to hold the longest think block and
/// one full value with room to spare, which the engine's default of 16k does
/// not.
pub(crate) const MAX_TURN_TOKENS: usize = 65_536;
