use crate::sequence_handle::SequenceId;
use thiserror::Error;

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

    #[error("{0}")]
    Other(String),
}
