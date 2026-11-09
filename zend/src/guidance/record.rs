//! Sealing a read as one more calibration of its unit, and keeping each
//! unit's newest few.

use std::time::{SystemTime, UNIX_EPOCH};

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use crate::calibrate::{calibration_meta, seal_calibration, Frame};

/// Conversation metadata on a use-exemplar: the content key of the unit it
/// calibrates. Apart from the drafted calibrations' marker, so a layer's
/// calibration pass neither counts nor retires it.
pub const USE_KEY: &str = "guidance_use";

/// Conversation metadata on a use-exemplar: when it was sealed, in
/// milliseconds since the epoch — what [`MAX_EXEMPLARS`] keeps the newest by.
pub const USED_AT_KEY: &str = "guidance_used_at";

/// The most use-exemplars a unit keeps, the newest. A guide read a thousand
/// times does not fill the gallery, and reasoning that has drifted ages out.
pub const MAX_EXEMPLARS: usize = 16;

/// Seal `case` as a use-exemplar of `unit` — committed under `key`, its
/// subject `subject` — then retire the unit's exemplars past
/// [`MAX_EXEMPLARS`], and every exemplar whose unit is gone. Returns how many
/// were retired.
pub fn record_use(
    engine: &ConversationEngine,
    frame: &Frame<'_>,
    unit: TimelineId,
    key: &str,
    subject: &str,
    case: (String, String),
) -> anyhow::Result<usize> {
    let mut meta = calibration_meta(unit, subject);
    meta.insert(USE_KEY.to_string(), key.to_string());
    meta.insert(USED_AT_KEY.to_string(), now_ms().to_string());
    seal_calibration(engine, frame, &meta, &[case])?;
    let exemplars: Vec<(TimelineId, String, u64)> = engine
        .conversations_with_metadata_key(USE_KEY)
        .into_iter()
        .map(|(timeline, of)| {
            let used_at = engine
                .conversation_metadata_value(timeline, USED_AT_KEY)
                .and_then(|at| at.parse().ok())
                .unwrap_or(0);
            (timeline, of, used_at)
        })
        .collect();
    let conversation = engine.conversation();
    let substrate = conversation.read();
    let standing: Vec<Standing> = exemplars
        .into_iter()
        .map(|(timeline, of, used_at)| Standing {
            timeline,
            used_at,
            of_this_unit: of == key,
            unit_gone: substrate.calibrates(timeline).is_none_or(|unit| {
                substrate.is_tombstoned(unit) || substrate.timeline_target(unit).is_none()
            }),
        })
        .collect();
    drop(substrate);
    let mut retired = 0;
    for timeline in to_retire(standing, MAX_EXEMPLARS) {
        if engine.tombstone_timeline(timeline).is_ok() {
            retired += 1;
        }
    }
    Ok(retired)
}

/// One standing use-exemplar, as [`to_retire`] judges it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Standing {
    timeline: TimelineId,
    used_at: u64,
    /// Whether it calibrates the unit just read.
    of_this_unit: bool,
    /// Whether the unit it calibrates is tombstoned or absent.
    unit_gone: bool,
}

/// The exemplars to retire: every one whose unit is gone, and the unit's own
/// past its newest `keep`. Others' exemplars are judged when they are next
/// read.
fn to_retire(mut standing: Vec<Standing>, keep: usize) -> Vec<TimelineId> {
    let mut out: Vec<TimelineId> = standing
        .iter()
        .filter(|s| s.unit_gone)
        .map(|s| s.timeline)
        .collect();
    standing.retain(|s| s.of_this_unit && !s.unit_gone);
    standing.sort_by(|a, b| b.used_at.cmp(&a.used_at).then(b.timeline.cmp(&a.timeline)));
    out.extend(standing.into_iter().skip(keep).map(|s| s.timeline));
    out
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn standing(n: u64, used_at: u64, of_this_unit: bool, unit_gone: bool) -> Standing {
        Standing {
            timeline: tl(n),
            used_at,
            of_this_unit,
            unit_gone,
        }
    }

    /// **A unit keeps its newest exemplars**; another unit's are left for its
    /// own next read.
    #[test]
    fn a_unit_keeps_its_newest_exemplars() {
        let got = to_retire(
            vec![
                standing(1, 100, true, false),
                standing(2, 300, true, false),
                standing(3, 200, true, false),
                standing(4, 50, false, false),
            ],
            2,
        );
        assert_eq!(got, vec![tl(1)]);
    }

    /// **An exemplar whose unit is gone is retired**, whichever unit it was.
    #[test]
    fn an_exemplar_of_a_unit_that_is_gone_is_retired() {
        let got = to_retire(
            vec![standing(1, 100, false, true), standing(2, 300, true, false)],
            2,
        );
        assert_eq!(got, vec![tl(1)]);
    }

    /// On equal times the later-minted exemplar is the newer.
    #[test]
    fn on_equal_times_the_later_one_is_kept() {
        let got = to_retire(
            vec![standing(5, 100, true, false), standing(6, 100, true, false)],
            1,
        );
        assert_eq!(got, vec![tl(5)]);
    }
}
