//! What a raw turn decoded: its ids, the text it finished with, and why it
//! stopped.

use crate::FinishReason;

/// One raw turn's decode: what `ConversationEngine::infer_raw_token_ids`
/// answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawDecode {
    /// The ids decoded, the end-of-sequence token that ended it excluded.
    pub tokens: Vec<u32>,
    /// The text the turn finished with, when it finished.
    pub text: Option<String>,
    /// Why it stopped; `None` when it never finished.
    pub finish: Option<FinishReason>,
}
