//! What a layer's calibration pass has to do, from what the layer holds and
//! what is already calibrated — the tools' resume filter, for units.
//!
//! A unit's calibration is done once a calibration carrying its current
//! marker is **archived**: the archive is the commit, written only after every
//! question sealed. Anything else carrying one of the layer's markers goes —
//! a calibration of a unit no longer there, of a version since replaced, under
//! a prompt since reworded, one begun and never finished, and a second
//! finished one of the same marker.
//!
//! **A unit that is there keeps its calibration whether or not it is ready.**
//! A unit with a turn still to arrive from another member, or a chain still
//! sealing, cannot be calibrated now — but it is not gone, and its finished
//! calibration stays for when it is whole again.
//!
//! **A calibration is replaced only once its replacement is committed.** A
//! changed prompt, count or budget gives every unit a new marker, and the
//! pass that calibrates them all again runs for hours. A unit's finished
//! calibration under an earlier marker — the same unit, by its content key —
//! stands until a calibration carrying its current marker is archived, and is
//! retired by the pass after; a unit is never left with none in between.

use std::collections::{HashMap, HashSet};

use candle_conversation::projection::TimelineId;

/// A unit of the layer: its conversation, what it is of, the marker its
/// calibration carries, and whether it can be calibrated now — committed,
/// finished, and every turn holding its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub timeline: TimelineId,
    pub subject: String,
    /// The content key it is committed under: the unit, whatever it is asked.
    pub key: String,
    pub marker: String,
    pub ready: bool,
}

/// A calibration found carrying one of the layer's markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prior<'a> {
    pub timeline: TimelineId,
    pub marker: &'a str,
    /// The content key of the unit its marker names, whatever it was asked
    /// ([`super::spec::CalibrateSpec::unit_of`]).
    pub unit: &'a str,
    pub archived: bool,
}

/// What a pass does.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// The ready units to calibrate, in the order they were given.
    pub run: Vec<Unit>,
    /// Calibrations to tombstone.
    pub retire: Vec<TimelineId>,
    /// Units already calibrated.
    pub done: usize,
    /// Units neither calibrated nor ready.
    pub waiting: usize,
}

pub fn plan(units: Vec<Unit>, priors: &[Prior<'_>]) -> Plan {
    let present: HashSet<&str> = units.iter().map(|u| u.marker.as_str()).collect();
    let present_units: HashSet<&str> = units.iter().map(|u| u.key.as_str()).collect();
    let unit_of: HashMap<&str, &str> = units
        .iter()
        .map(|u| (u.marker.as_str(), u.key.as_str()))
        .collect();
    let mut kept: HashMap<&str, TimelineId> = HashMap::new();
    let mut retire = Vec::new();
    let mut by_order: Vec<&Prior<'_>> = priors.iter().collect();
    by_order.sort_by_key(|p| p.timeline);
    // Calibrations of an earlier marker of a present unit: the newest stands
    // until the unit's current one is committed.
    let mut standing: HashMap<&str, TimelineId> = HashMap::new();
    for p in by_order {
        if p.archived && present.contains(p.marker) && !kept.contains_key(p.marker) {
            kept.insert(p.marker, p.timeline);
        } else if p.archived && !present.contains(p.marker) && present_units.contains(p.unit) {
            if let Some(older) = standing.insert(p.unit, p.timeline) {
                retire.push(older);
            }
        } else {
            retire.push(p.timeline);
        }
    }
    let committed: HashSet<&str> = kept
        .keys()
        .filter_map(|m| unit_of.get(m).copied())
        .collect();
    for (unit, tl) in standing {
        if committed.contains(unit) {
            retire.push(tl);
        }
    }
    retire.sort();
    let (done, rest): (Vec<Unit>, Vec<Unit>) = units
        .into_iter()
        .partition(|u| kept.contains_key(u.marker.as_str()));
    let (run, waiting): (Vec<Unit>, Vec<Unit>) = rest.into_iter().partition(|u| u.ready);
    Plan {
        run,
        retire,
        done: done.len(),
        waiting: waiting.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// A unit whose content key is the marker's first letter: `a1` and `a2`
    /// are two markers of unit `a`.
    fn unit(raw: u64, marker: &str) -> Unit {
        Unit {
            timeline: tl(raw),
            subject: format!("s{raw}"),
            key: marker[..1].into(),
            marker: marker.into(),
            ready: true,
        }
    }

    fn prior(raw: u64, marker: &str, archived: bool) -> Prior<'_> {
        Prior {
            timeline: tl(raw),
            marker,
            unit: &marker[..1],
            archived,
        }
    }

    /// **A calibration under an earlier marker stands until its replacement
    /// is committed.** Unit `a` is now asked as `a2`; its `a1` calibration
    /// stands while `a2` is only begun, and the older `a0` goes. Once `a2` is
    /// archived, `a1` retires.
    #[test]
    fn an_earlier_calibration_stands_until_its_replacement_commits() {
        let p = plan(
            vec![unit(1, "a2")],
            &[
                prior(9, "a0", true),
                prior(10, "a1", true),
                prior(11, "a2", false),
            ],
        );
        assert_eq!((p.done, p.run.len()), (0, 1));
        assert_eq!(p.retire, vec![tl(9), tl(11)]);

        let p = plan(
            vec![unit(1, "a2")],
            &[prior(10, "a1", true), prior(12, "a2", true)],
        );
        assert_eq!((p.done, p.run.len()), (1, 0));
        assert_eq!(p.retire, vec![tl(10)]);
    }

    #[test]
    fn an_archived_current_calibration_is_done_and_the_rest_retire() {
        let units = vec![unit(1, "a"), unit(2, "b"), unit(3, "c")];
        let priors = [
            // a: finished.
            prior(10, "a", true),
            // b: begun, never finished.
            prior(11, "b", false),
            // a unit no longer there, or a version replaced.
            prior(12, "zone", true),
            // a second finished calibration of a: the first stays.
            prior(13, "a", true),
        ];
        let p = plan(units, &priors);
        assert_eq!((p.done, p.waiting), (1, 0));
        assert_eq!(p.run, vec![unit(2, "b"), unit(3, "c")]);
        assert_eq!(p.retire, vec![tl(11), tl(12), tl(13)]);
    }

    /// A unit that is there but not ready keeps its finished calibration and
    /// is not run; one with none waits for a pass that finds it ready.
    #[test]
    fn a_unit_not_ready_keeps_its_calibration_and_waits() {
        let mut busy = unit(1, "a");
        busy.ready = false;
        let mut new = unit(2, "b");
        new.ready = false;
        let p = plan(vec![busy, new], &[prior(10, "a", true)]);
        assert_eq!((p.done, p.waiting, p.run.len()), (1, 1, 0));
        assert!(p.retire.is_empty());
    }

    #[test]
    fn nothing_calibrated_runs_every_ready_unit() {
        let p = plan(vec![unit(1, "a")], &[]);
        assert_eq!((p.done, p.run.len(), p.retire.len()), (0, 1, 0));
    }
}
