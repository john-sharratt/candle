//! Startup turn-integrity classification — completeness beyond the CRC.
//!
//! The per-chunk CRC (and the GPU golden checksum) verify that bytes which
//! WERE written are intact. They cannot see records that were never written at
//! all: the seal path persists a turn's pieces through independent channels
//! (`StreamDecl` + KV chunks on the persistence thread, `Tokens` / `WideQSig`
//! on the fire-and-forget async writer), so a hard kill can land a subset. The
//! reload then restores a turn that is alive but incomplete — selectable yet
//! text-less (no `Tokens` record) or unmaterialisable (no KV chunks).
//!
//! [`classify_turn`] is the pure verdict on one restored turn's completeness.
//! The reconstruct loop applies it with the owning layer's
//! [`CorruptTurnPolicy`](crate::projection::schema::CorruptTurnPolicy):
//! regenerable layers (`DropConversation` — code_read files, repo_map
//! clusters) tombstone the timeline so the background refresh re-ingests it;
//! the dialogue (`DropTurn`) is user history and is never auto-deleted for
//! incompleteness — it restores with a warning and renders/attends through
//! whatever pieces survived.
//!
//! ## Why this cannot thrash
//!
//! The check runs ONLY during startup reconstruct, when no writer exists —
//! the on-disk state is final, so a conversation can never be judged
//! "incomplete" merely because it is still being written. The remaining
//! re-ingest triggers all converge:
//!
//! - a file killed mid-ingest is tombstoned once and re-ingested once at
//!   startup — which the ingest resume would do anyway (its completion hash
//!   was never committed);
//! - distilled timelines (`DistillMode` — tokens/KV shed deliberately) are
//!   exempt, so the calibration corpus is never re-run;
//! - zero-token turns (tombstone placeholders, turns that legitimately hold
//!   no content) are exempt;
//! - workspaces that never persist KV (no chunk stream anywhere) skip the
//!   KV-presence check entirely, so `compression_policy = None` runs are not
//!   condemned wholesale;
//! - a repaired conversation re-ingests with all records present and passes
//!   the next startup's check. A conversation failing EVERY startup would
//!   re-ingest once per startup (bounded, loudly logged) — that is a writer
//!   bug surfacing, not a loop.

use crate::persistence::record::DistillMode;

/// Completeness verdict for one restored turn, most severe applicable state.
///
/// A missing layout is deliberately NOT a violation: with tokens present the
/// text is fully derivable by decode, and with tokens absent the verdict is
/// already `MissingTokens` — layout absence alone degrades nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnIntegrity {
    /// All expected pieces present (or absent by design).
    Ok,
    /// The decl declares a KV block span but no chunk stream exists AND no
    /// `Tokens` record survived — nothing remains to rebuild the turn from,
    /// so it can never materialise into a projection. (KV-less turns WITH
    /// tokens are recoverable: the shutdown drain deliberately leaves a
    /// just-sealed turn's KV behind and the reload re-prefills it from the
    /// persisted token ids — that is the redo log's designed cold path.)
    MissingKv,
    /// No `Tokens` record survived — the turn cannot be decoded to verbatim
    /// text (layout text may still render it).
    MissingTokens,
}

impl TurnIntegrity {
    /// Human-readable reason fragment for tombstone records and logs.
    pub fn describe(self) -> &'static str {
        match self {
            TurnIntegrity::Ok => "complete",
            TurnIntegrity::MissingKv => "KV chunks missing (turn can never materialise)",
            TurnIntegrity::MissingTokens => "Tokens record missing (no verbatim text)",
        }
    }
}

/// Classify one restored turn's completeness.
///
/// * `distill` — the timeline's distillation mode; any mode exempts the turn
///   (its pieces were shed deliberately).
/// * `declares_content` — the decl declares a non-empty KV block span, i.e.
///   the turn CLAIMS sealed content. `false` exempts the turn (tombstone
///   placeholders and ghost turns claim nothing). This is judged from the
///   decl, never from the recovered chunk index — a turn whose chunks were
///   all lost would otherwise report zero tokens and dodge the check.
/// * `has_tokens` — a `Tokens` record was recovered (non-empty ids).
/// * `expects_kv` — this workspace persists KV at all (some chunk stream
///   exists somewhere); `false` under `compression_policy = None`.
/// * `has_kv` — cold chunk refs were recovered for the turn.
pub fn classify_turn(
    distill: Option<DistillMode>,
    declares_content: bool,
    has_tokens: bool,
    expects_kv: bool,
    has_kv: bool,
) -> TurnIntegrity {
    if distill.is_some() || !declares_content {
        return TurnIntegrity::Ok;
    }
    if expects_kv && !has_kv && !has_tokens {
        return TurnIntegrity::MissingKv;
    }
    if !has_tokens {
        return TurnIntegrity::MissingTokens;
    }
    TurnIntegrity::Ok
}

#[cfg(test)]
mod tests {
    use super::{classify_turn, TurnIntegrity};
    use crate::persistence::record::DistillMode;

    #[test]
    fn complete_turn_is_ok() {
        assert_eq!(
            classify_turn(None, true, true, true, true),
            TurnIntegrity::Ok
        );
    }

    #[test]
    fn distilled_timeline_is_exempt_in_both_modes() {
        // ProvenanceOnly sheds tokens + KV; TextOnly sheds sigs. Either way the
        // absence is deliberate — flagging it would re-run the calibration
        // corpus every startup (the thrash case this exemption exists for).
        for mode in [DistillMode::ProvenanceOnly, DistillMode::TextOnly] {
            assert_eq!(
                classify_turn(Some(mode), true, false, true, false),
                TurnIntegrity::Ok
            );
        }
    }

    #[test]
    fn zero_span_placeholder_is_exempt() {
        // Tombstone placeholders and ghost turns declare no KV block span —
        // they claim nothing, so nothing is expected of them.
        assert_eq!(
            classify_turn(None, false, false, true, false),
            TurnIntegrity::Ok
        );
    }

    #[test]
    fn missing_kv_outranks_missing_tokens() {
        assert_eq!(
            classify_turn(None, true, false, true, false),
            TurnIntegrity::MissingKv
        );
    }

    #[test]
    fn kv_less_turn_with_tokens_is_recoverable() {
        // The clean-shutdown shape: the drain deferred the just-sealed turn's
        // pinned hot KV (decl + tokens flushed, chunks never reached cold).
        // The tokens are the rebuild source — reload re-prefills the K/V —
        // so this is NOT damage and must never tombstone the conversation.
        assert_eq!(
            classify_turn(None, true, true, true, false),
            TurnIntegrity::Ok
        );
    }

    #[test]
    fn kv_not_expected_skips_the_kv_check() {
        // `compression_policy = None` workspaces never persist chunks — their
        // absence is not damage.
        assert_eq!(
            classify_turn(None, true, true, false, false),
            TurnIntegrity::Ok
        );
    }

    #[test]
    fn lost_tokens_record_is_flagged() {
        // The observed field failure: decl + KV + sigs landed, the async
        // `Tokens` record did not (hard kill before the writer drained).
        assert_eq!(
            classify_turn(None, true, false, true, true),
            TurnIntegrity::MissingTokens
        );
    }
}
