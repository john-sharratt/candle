//! What a layer's calibration pass has to do, from what the layer holds and
//! what is already calibrated — the tools' resume filter, for units.
//!
//! A unit's calibration is done once a calibration carrying its current
//! marker is **archived**: the archive is the commit, written only after every
//! question sealed. Anything else carrying one of the layer's markers goes —
//! a calibration of a unit no longer there, of a version since replaced, under
//! a prompt since reworded, one begun and never finished, and a second
//! finished one of the same marker.

use std::collections::{HashMap, HashSet};

use candle_conversation::projection::TimelineId;

/// A unit that stands: its conversation, what it is of, and the marker its
/// calibration carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Standing {
    pub timeline: TimelineId,
    pub subject: String,
    pub marker: String,
}

/// A calibration found carrying one of the layer's markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prior<'a> {
    pub timeline: TimelineId,
    pub marker: &'a str,
    pub archived: bool,
}

/// What a pass does.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// The units to calibrate, in the order they stand.
    pub run: Vec<Standing>,
    /// Calibrations to tombstone.
    pub retire: Vec<TimelineId>,
    /// Units already calibrated.
    pub done: usize,
}

pub fn plan(units: Vec<Standing>, priors: &[Prior<'_>]) -> Plan {
    let current: HashSet<&str> = units.iter().map(|u| u.marker.as_str()).collect();
    let mut kept: HashMap<&str, TimelineId> = HashMap::new();
    let mut retire = Vec::new();
    let mut by_order: Vec<&Prior<'_>> = priors.iter().collect();
    by_order.sort_by_key(|p| p.timeline);
    for p in by_order {
        let keep = p.archived && current.contains(p.marker) && !kept.contains_key(p.marker);
        if keep {
            kept.insert(p.marker, p.timeline);
        } else {
            retire.push(p.timeline);
        }
    }
    let (done, run): (Vec<Standing>, Vec<Standing>) = units
        .into_iter()
        .partition(|u| kept.contains_key(u.marker.as_str()));
    Plan {
        run,
        retire,
        done: done.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn unit(raw: u64, marker: &str) -> Standing {
        Standing {
            timeline: tl(raw),
            subject: format!("s{raw}"),
            marker: marker.into(),
        }
    }

    #[test]
    fn an_archived_current_calibration_is_done_and_the_rest_retire() {
        let units = vec![unit(1, "a"), unit(2, "b"), unit(3, "c")];
        let priors = [
            // a: finished.
            Prior {
                timeline: tl(10),
                marker: "a",
                archived: true,
            },
            // b: begun, never finished.
            Prior {
                timeline: tl(11),
                marker: "b",
                archived: false,
            },
            // a unit no longer standing, or a version replaced.
            Prior {
                timeline: tl(12),
                marker: "gone",
                archived: true,
            },
            // a second finished calibration of a: the first stays.
            Prior {
                timeline: tl(13),
                marker: "a",
                archived: true,
            },
        ];
        let p = plan(units, &priors);
        assert_eq!(p.done, 1);
        assert_eq!(p.run, vec![unit(2, "b"), unit(3, "c")]);
        assert_eq!(p.retire, vec![tl(11), tl(12), tl(13)]);
    }

    #[test]
    fn nothing_calibrated_runs_every_unit() {
        let p = plan(vec![unit(1, "a")], &[]);
        assert_eq!((p.done, p.run.len(), p.retire.len()), (0, 1, 0));
    }
}
