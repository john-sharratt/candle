//! A conversation's priority band, as the scheduler ranks it.
//!
//! A conversation's band is its target layer's `decode_priority`: the dialogue
//! is `Urgent`, the ingest layers `Low`. Bands share the device rather than take
//! turns with it (`docs/continuous_fair_waves.md` §3): every band's decode rows
//! ride every forward, and a band's prefill creeps at the airtime ratio of the
//! decodes that outrank it — and a `High` prefill, beside a `High` decode, is
//! paced in tokens. Nothing is held out of a forward for its band.
//!
//! Tenants are equal: the band alone ranks, whichever tenant the work belongs
//! to (`docs/shared_model_host.md` §3.6).
//!
//! A conversation may record a band of its own, under
//! [`DecodePriority::METADATA_KEY`], in place of its layer's: a task delegated
//! to a dialogue-layer conversation decodes at `normal` while nobody waits on
//! it, below a live user (`docs/claude_zend_gateway.md` §3.4).

use crate::projection::{Conversation, DecodePriority, TimelineId};

/// A band's rank: higher outranks lower.
pub(super) fn rank(p: DecodePriority) -> usize {
    match p {
        DecodePriority::Low => 0,
        DecodePriority::Normal => 1,
        DecodePriority::High => 2,
        DecodePriority::Urgent => 3,
    }
}

/// The band `timeline` records for itself, when it records one it can be
/// read as.
pub(super) fn recorded_band(
    conversation: &Conversation,
    timeline: TimelineId,
) -> Option<DecodePriority> {
    conversation
        .conversation_metadata_value(timeline, DecodePriority::METADATA_KEY)
        .and_then(|name| DecodePriority::parse(&name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId};
    use DecodePriority::{High, Low, Normal, Urgent};

    #[test]
    fn the_bands_rank_low_normal_high_urgent() {
        assert!(rank(Low) < rank(Normal));
        assert!(rank(Normal) < rank(High));
        assert!(rank(High) < rank(Urgent));
    }

    #[test]
    fn a_band_reads_back_by_its_name_and_nothing_else_does() {
        for band in [Low, Normal, High, Urgent] {
            assert_eq!(DecodePriority::parse(band.name()), Some(band));
        }
        assert_eq!(DecodePriority::parse("soon"), None);
        assert_eq!(DecodePriority::parse("High"), None);
    }

    /// **A band recorded before the conversation exists is its band once it
    /// does**: the record is held until the timeline registers — which a new
    /// conversation does before its first slot is made — and a band that is
    /// not a band's name is none.
    #[test]
    fn a_conversation_records_its_own_band() {
        let conversation = Conversation::new();
        let timeline = TimelineId::from_raw(7).unwrap();
        assert_eq!(recorded_band(&conversation, timeline), None);
        conversation
            .set_conversation_metadata(timeline, DecodePriority::METADATA_KEY, "normal")
            .unwrap();
        conversation.register_timeline(
            timeline,
            LayerId::from_raw(1).unwrap(),
            GroupId::from_raw(1).unwrap(),
        );
        assert_eq!(recorded_band(&conversation, timeline), Some(Normal));
        conversation
            .set_conversation_metadata(timeline, DecodePriority::METADATA_KEY, "soon")
            .unwrap();
        assert_eq!(recorded_band(&conversation, timeline), None);
    }
}
