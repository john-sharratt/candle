use crate::sequence_handle::SequenceId;
use thiserror::Error;

/// What a [`ConversationError::TurnStalled`] says, which a caller holding only
/// its text finds with [`is_stall_message`].
pub const TURN_STALLED: &str = "the turn made no progress for";

/// Whether `msg` — an error as text, its chain included — is a stalled turn's:
/// not a failure of the work the turn was for, which a member with room does.
pub fn is_stall_message(msg: &str) -> bool {
    msg.contains(TURN_STALLED)
}

/// Errors produced by the conversation engine.
#[derive(Debug, Error)]
pub enum ConversationError {
    #[error("sequence {sequence_id} already has a turn in flight")]
    TurnInFlight {
        /// The sequence that already has an active turn.
        sequence_id: SequenceId,
    },

    #[error("model error: {0}")]
    Model(#[from] candle::Error),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("scheduler has shut down")]
    SchedulerGone,

    /// A graceful shutdown asked the in-flight ingest to stop while this turn's
    /// summary decode was still running (see [`crate::CancelToken::request`]).
    /// The wait was abandoned and the turn handle dropped so the scheduler stops
    /// decode at its next step. This is a cooperative stop, not a failure — the
    /// ingest caller must treat it as "cancelled", NOT count it toward any
    /// decode-failure budget.
    #[error("ingest cancelled during summary decode")]
    IngestCancelled,

    /// A background turn produced no token and ran no prefill chunk for
    /// `secs` — never admitted, held behind other work the whole time, or lost
    /// — and its wait gave it up (`handle::STALL`). The caller drops its
    /// handle, which withdraws the turn, and hands back whatever it held for
    /// the work, so a member with room can do it.
    #[error("the turn made no progress for {secs} s: it was not admitted, or was lost")]
    TurnStalled { secs: u64 },

    #[error("cold store I/O error: {0}")]
    StoreIo(#[from] std::io::Error),

    #[error("sequence {0} not found")]
    SequenceNotFound(usize),

    #[error("internal channel error: {0}")]
    Channel(String),

    #[error("log serialization error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("download error: {0}")]
    Download(String),

    /// Recording `parent` as `child`'s fork parent would close a cycle:
    /// `parent` is `child` or one of its descendants.
    #[error("timeline {child} cannot fork from {parent}: it is on {parent}'s own lineage")]
    ForkCycle {
        /// The timeline whose parent was being recorded.
        child: u64,
        /// The parent refused.
        parent: u64,
    },

    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stalled turn's text is found by its marker, however far down an
    /// error chain it sits.
    #[test]
    fn a_stalled_turn_is_found_by_its_text() {
        let stalled = ConversationError::TurnStalled { secs: 60 }.to_string();
        assert!(stalled.starts_with(TURN_STALLED), "{stalled}");
        assert!(is_stall_message(&format!("decode: {stalled}")));
        assert!(!is_stall_message(
            &ConversationError::SchedulerGone.to_string()
        ));
    }
}
