//! The kinds of segment a turn is made of, and whether each one occupies K/V.
//!
//! A segment is real when it has a K/V span and ethereal when it is only recorded
//! for display or the projection record. Dialect boundary markers are stored as a
//! kind rather than as their bytes, so the marker text is derived from the active
//! dialect at the point of use. Real or ethereal is the one axis that expresses
//! reasoning whose K/V was dropped and glue that is not emitted into the grid.

use super::*;

/// A dialect boundary marker.  The marker text/tokens are derived from the active
/// dialect by this kind — the layout stores only the kind (and, when real, the
/// [`KvSpan`] it occupies), never the marker bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlueKind {
    /// `<|im_start|>system\n`
    SystemStart,
    /// `<|im_start|>user\n` — opens a turn.
    UserStart,
    /// `<|im_start|>assistant\n`
    AssistantStart,
    /// `<|im_end|>\n` — closes a message / a turn.
    ImEnd,
    /// The `/no_think` soft-switch, emitted right after `user_start` on a
    /// thinking-suppressed turn.
    NoThink,
}

/// One ordered part of a turn.  Real segments carry a [`KvSpan`]; ethereal ones
/// carry none and contribute nothing to the K/V grid (their `text` / `kind` is
/// recorded for the projection record and the GUI only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnSegment {
    /// A dialect boundary marker.  `kv: Some` ⇒ REAL (recomputed per projection
    /// from the marker); `None` ⇒ ETHEREAL (recorded, not materialized).
    Glue {
        marker: GlueKind,
        #[serde(default)]
        kv: Option<KvSpan>,
    },
    /// User message body — always real.
    User { text: String, kv: KvSpan },
    /// A `<think>…</think>` reasoning block.  `kv: Some` ⇒ REAL (K/V stored);
    /// `None` ⇒ ETHEREAL (prose kept, K/V dropped).
    Thinking {
        text: String,
        #[serde(default)]
        kv: Option<KvSpan>,
    },
    /// Assistant answer body — always real; `text` optional.
    Assistant {
        #[serde(default)]
        text: Option<String>,
        kv: KvSpan,
    },
}

impl TurnSegment {
    /// The K/V span this segment occupies, or `None` when it is ethereal.
    pub fn kv(&self) -> Option<KvSpan> {
        match self {
            TurnSegment::Glue { kv, .. } | TurnSegment::Thinking { kv, .. } => *kv,
            TurnSegment::User { kv, .. } | TurnSegment::Assistant { kv, .. } => Some(*kv),
        }
    }
    /// True when the segment is materialized into the slot's K/V.
    pub fn is_real(&self) -> bool {
        self.kv().is_some()
    }
}
