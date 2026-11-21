//! The wording, deadlines and retry intervals of a task's prompts, waits and
//! handings between members.
//!
//! The task prompt ends by telling the worker how to ask Claude rather than guess,
//! and a resumed or sent-back task is told why first. A wait says what a running
//! worker has done again at a fixed interval, so the line stays current. The
//! deadlines bound how long a draft, a lease or a handed turn may take. The
//! handing waits are kept under the beat's own limit, since the beat is stopped
//! while a handing runs.

use super::*;

/// How long a task's draft may take before its earlier entries are taken.
pub(super) const DRAFT_DEADLINE: Duration = Duration::from_secs(15 * 60);

/// How often a draft's progress is read.
pub(super) const DRAFT_POLL: Duration = Duration::from_secs(2);

/// The least time between two writes of a running worker's record.
pub(super) const STEP_GAP: Duration = Duration::from_secs(5);

/// How often a wait says again what a worker still running has done, with
/// its time brought up to date, when it has taken no new step — inside the
/// stream's own repeat (`http::PROGRESS_EVERY`), so the line the person sees
/// is a worker's, with its time current.
pub(super) const PROGRESS_AGAIN: Duration = Duration::from_secs(10);

/// How much of a prompt the thread's journal quotes.
pub(super) const PROMPT_QUOTE: usize = 160;

/// The line a worker's question block opens with.
pub const QUESTION_MARK: &str = "QUESTION:";

/// What a task sent back by its answer's check is told first: who sent it
/// back. Unheaded, a worker took the check's words for the user's, and its
/// journal credited the user with asking for them
/// (`docs/zen_coding_challenge_fixes.md` Cause 7).
pub const SENT_BACK: &str = "From zend, which checked your answer — not the user:\n\n";

/// What every task's prompt ends with: how to ask Claude instead of guessing.
/// Asking ends the turn — the worker's last words are the question — so
/// nothing waits mid-turn, a restart finds a finished turn, and the answer is
/// simply the worker's next turn.
pub const ASKING: &str =
    "\n\n---\nIf you cannot go on without a decision that is not yours to make \
— the task is ambiguous, two approaches trade off and the choice is a design decision, the same \
failure keeps coming back — do not guess and do not stop. Ask: end your final response with this \
block and nothing after it, and the answer comes back as your next message.\n\
QUESTION: what you need decided or explained, in a sentence or two\n\
OPTIONS: each choice you see, and what it leads to\n\
TRIED: what you already did, and what happened\n\
POINTERS: the files and lines, commands or turns the question rests on\n\
Never ask what you can settle yourself by reading, running or searching.";

/// What a task cut off by a restart is told first.
pub(super) const RESUMED: &str =
    "Your work was cut off, so your last turn may not have finished. Your \
    conversation and your journal are intact; changes you had not committed may be gone. Check \
    where the work stands (`git_status`, reading what you wrote) before repeating anything, then \
    finish the task below. It is your task, whatever else you remember.";

/// What Zen's side of a chat, cut off, is told first: a chat worker's prompt is
/// the message it was answering, not a task.
pub(super) const CHAT_RESUMED: &str =
    "Your answer was cut off by a restart, so it may not have finished. \
    Answer the message below again, whole, as your next message in the chat.";

/// How many of a cut-off task's journal entries it is reminded of.
pub(super) const RESUMED_ENTRIES: usize = 8;

/// How much of each it is reminded of.
pub(super) const RESUMED_ENTRY_QUOTE: usize = 300;

/// How long a member starting a worker's turns waits for their lease: long
/// enough for one a restart left behind to run out ([`ingest_lease`]'s
/// `INGEST_TTL`).
pub(super) const LEASE_WAIT: Duration = Duration::from_secs(150);

/// How often it asks again.
pub(super) const LEASE_LOOK: Duration = Duration::from_secs(2);

/// How long a handed turn's commit may take to land before the member keeps
/// the round itself.
pub(super) const HAND_COMMIT_WAIT: Duration = Duration::from_secs(20);

/// How long the member a turn is handed to may take to answer each of the
/// handing's two asks — it waits to hold the turn's records first
/// ([`CONTINUE_HOLD_WAIT`]), and the second ask carries the records it lacked.
/// Both, with the commit's wait, [`TAKEN_WAIT`] and a beat's interval, are
/// under the beat's own limit (`threads::BEAT_LOST_MS`): the beat is stopped
/// while the handing runs.
pub(super) const HAND_ANSWER_WAIT: Duration = Duration::from_secs(25);

/// How long the member taking a handed worker waits to hold its records.
pub const CONTINUE_HOLD_WAIT: Duration = Duration::from_secs(10);

/// The error a member taking a worker answers when it lacks the handing
/// member's records, with the ranges it lacks as `missing`.
pub const MISSING: &str = "missing";

/// The route a worker is handed on (`crate::api::worker_continue`).
pub const CONTINUE_PATH: &str = "/v1/mcp/worker/continue";

/// How long a member whose handing failed waits for the record to say the
/// peer took it after all — its write may still be on its way here.
pub(super) const TAKEN_WAIT: Duration = Duration::from_secs(10);
