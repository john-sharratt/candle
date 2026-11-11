//! What a private conversation holds while its diaries are open: the kek its
//! passphrase derived, each diary it unwrapped — its own key and its topic —
//! and when it was last used. In memory only and never written; its kek and
//! keys are zeroized when it goes (`docs/private_mode.md` §5.3–§5.4).

use std::collections::BTreeMap;

use journals::diaries::crypto::{Dek, Kek};
use journals::diaries::layout::{DiaryId, SALT_LEN};

/// One diary, open.
pub struct Opened {
    pub dek: Dek,
    pub topic: String,
}

/// One conversation's open passphrase and diaries.
pub struct Unlocked {
    pub kek: Kek,
    /// The salt `kek` was derived with while the repository has none yet:
    /// the first diary created writes it.
    pub pending_salt: Option<[u8; SALT_LEN]>,
    pub open: BTreeMap<DiaryId, Opened>,
    pub used_ms: u64,
}

impl Unlocked {
    /// The open diary on `topic`.
    pub fn find(&self, topic: &str) -> Option<(DiaryId, &Opened)> {
        self.open
            .iter()
            .find(|(_, o)| o.topic == topic)
            .map(|(id, o)| (*id, o))
    }
}
