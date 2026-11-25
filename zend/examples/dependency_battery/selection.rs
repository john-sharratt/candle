//! Which arms a launch asks, and up to which run: `run --arms` names arms by
//! id, each up to `--runs` or up to a run of its own given as `id:runs`, so
//! one launch can add runs to some arms and first runs to others, all
//! interleaved in one queue (`schedule.rs`).

use anyhow::{anyhow, bail, Context};

use crate::arms::{Arm, ARMS, PROBES};

/// The arms `specs` names, each with the highest run it is asked to: every
/// arm up to `runs` when `specs` names none.
pub fn arm_runs(specs: &[String], runs: u32) -> anyhow::Result<Vec<(Arm, u32)>> {
    if specs.is_empty() {
        return Ok(ARMS.iter().map(|&a| (a, runs)).collect());
    }
    let mut out: Vec<(Arm, u32)> = Vec::with_capacity(specs.len());
    for spec in specs {
        let spec = spec.trim();
        let (id, upto) = match spec.split_once(':') {
            Some((id, n)) => (
                id,
                n.parse()
                    .with_context(|| format!("`{spec}`: `{n}` is not a run number"))?,
            ),
            None => (spec, runs),
        };
        let arm = Arm::of(&id.to_ascii_lowercase()).ok_or_else(|| {
            let ids: Vec<&str> = ARMS.iter().chain(PROBES.iter()).map(|a| a.id()).collect();
            anyhow!("`{id}` is no arm: one of {}", ids.join(", "))
        })?;
        if upto == 0 {
            bail!("`{spec}`: runs are numbered from 1");
        }
        if out.iter().any(|(a, _)| *a == arm) {
            bail!("`{id}` is named twice");
        }
        out.push((arm, upto));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_arm_named_is_every_arm_to_the_runs_given() {
        let all = arm_runs(&[], 3).unwrap();
        assert_eq!(all.len(), ARMS.len());
        assert!(all.iter().all(|(_, r)| *r == 3));
        assert_eq!(all[0], (Arm::A0, 3));
    }

    /// **An arm takes `--runs`, or a run of its own after a colon.**
    #[test]
    fn each_arm_is_asked_to_its_own_highest_run() {
        assert_eq!(
            arm_runs(&specs(&["r1", "T1", "c1:6", " b1:6 "]), 3).unwrap(),
            vec![(Arm::R1, 3), (Arm::T1, 3), (Arm::C1, 6), (Arm::B1, 6)]
        );
    }

    #[test]
    fn a_bad_selection_is_refused() {
        assert!(arm_runs(&specs(&["z9"]), 3).is_err());
        assert!(arm_runs(&specs(&["c1:x"]), 3).is_err());
        assert!(arm_runs(&specs(&["c1:0"]), 3).is_err());
        assert!(arm_runs(&specs(&["c1", "c1:6"]), 3).is_err());
    }
}
