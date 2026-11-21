//! A belief turn group's scan plans, kept with its assembly.
//!
//! A group's paged segments are a pure function of its assembled files and of
//! how the scan reads them — who its turns are held for, the fold groups and
//! keeps of its content and calibration competitions, how its cases are
//! drawn and the vote laws (`belief_files::scan_file_scans`). An assembly's
//! files never change (`group_assembly`), so the index key those segments
//! give ([`ScanPlan`]) is worked out at the assembly's first scan under each
//! reading and kept, and every later scan hands the arena the key instead of
//! building and fingerprinting ~23,000 windows to find the same one.

use std::sync::{Arc, Mutex};

use super::adaptive::ScanCases;
use crate::provenance::gallery_arena::{GroupSet, Holder, Keep, ScanPlan};
use crate::provenance::VoteLaw;

/// How a scan reads an assembly's files: everything besides the files that
/// its segments are built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlanParams {
    pub holder: Holder,
    pub groups: GroupSet,
    pub calibration_groups: GroupSet,
    pub cases: ScanCases,
    pub keep: Keep,
    pub calibration_keep: Keep,
    pub law: VoteLaw,
    pub calibration_law: VoteLaw,
}

/// The plans kept for one assembly, by [`PlanParams`].
#[derive(Debug, Default)]
pub(crate) struct ScanPlans(Mutex<Vec<(PlanParams, Arc<ScanPlan>)>>);

/// Readings kept per assembly at most: a group is read one or two ways.
const MAX_PLANS: usize = 8;

impl ScanPlans {
    /// The plan kept for `params`.
    pub fn get(&self, params: &PlanParams) -> Option<Arc<ScanPlan>> {
        let plans = self.0.lock().unwrap_or_else(|e| e.into_inner());
        plans
            .iter()
            .find(|(p, _)| p == params)
            .map(|(_, plan)| Arc::clone(plan))
    }

    /// Keep `plan` for `params`; past [`MAX_PLANS`], the oldest goes.
    pub fn put(&self, params: PlanParams, plan: Arc<ScanPlan>) {
        let mut plans = self.0.lock().unwrap_or_else(|e| e.into_inner());
        plans.retain(|(p, _)| *p != params);
        if plans.len() >= MAX_PLANS {
            plans.remove(0);
        }
        plans.push((params, plan));
    }
}

impl Clone for ScanPlans {
    fn clone(&self) -> Self {
        Self(Mutex::new(
            self.0.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(holder: Holder) -> PlanParams {
        PlanParams {
            holder,
            groups: GroupSet::all(3),
            calibration_groups: GroupSet::all(3),
            cases: ScanCases::default(),
            keep: Keep::default(),
            calibration_keep: Keep::default(),
            law: VoteLaw::SINGLE,
            calibration_law: VoteLaw::SINGLE,
        }
    }

    fn plan(key: u64) -> Arc<ScanPlan> {
        Arc::new(ScanPlan {
            key,
            weighed: vec![GroupSet::all(3)],
            windows: 4,
        })
    }

    /// **A plan is handed back for the reading it was kept under and no
    /// other**; keeping one again replaces it, and the oldest goes past the
    /// cap.
    #[test]
    fn a_plan_serves_only_its_own_reading() {
        let plans = ScanPlans::default();
        assert!(plans.get(&params(Holder::Shared)).is_none());
        plans.put(params(Holder::Shared), plan(7));
        assert_eq!(plans.get(&params(Holder::Shared)).map(|p| p.key), Some(7));
        let other = PlanParams {
            law: VoteLaw {
                top_k: 8,
                power: 0.5,
            },
            ..params(Holder::Shared)
        };
        assert!(plans.get(&other).is_none(), "another vote law");
        plans.put(params(Holder::Shared), plan(9));
        assert_eq!(plans.get(&params(Holder::Shared)).map(|p| p.key), Some(9));
        for k in 0..MAX_PLANS {
            plans.put(
                PlanParams {
                    law: VoteLaw {
                        top_k: k + 2,
                        power: 1.0,
                    },
                    ..params(Holder::Shared)
                },
                plan(k as u64),
            );
        }
        assert!(
            plans.get(&params(Holder::Shared)).is_none(),
            "the oldest went"
        );
    }
}
