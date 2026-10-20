//! Which build an install failed to put in place, found at start.
//!
//! An install ends with the old process handing the new build to `keep` as a
//! trial. A trial that never proves itself is rolled back: keep restores the
//! stable file and starts it with the flags it always uses, so the restored
//! process is not told the install failed. It can tell anyway. The install
//! plan the old process wrote before the hand-off is still there, and it names
//! the build being installed. A process starting under a plan that names
//! another build is the build that plan replaced, back after its successor
//! failed.
//!
//! Not knowing it, the restored member is offered the same build at its next
//! tick, installs it, fails again, and serves nothing: on WhiteWorld
//! (2026-10-20) every trial of a rolled-out build ran out of the card the
//! process it replaced still held, for a whole load-test run.

use std::path::Path;

use super::super::decisions::build_name;
use super::super::watchdog::{read_plan, PLAN};
use super::Setup;

/// The build an install into `dir` failed to put in place: the one this
/// process was started knowing failed (`--upgrade-failed`), or else the one
/// the plan names when it is not this process's own. `None` when there was no
/// install, or it succeeded.
pub fn failed_install(setup: &Setup, dir: &Path) -> Option<String> {
    if let Some(build) = &setup.failed_from {
        return Some(build.clone());
    }
    let plan = read_plan(&dir.join(PLAN)).ok()?;
    (plan.installing != build_name(&setup.info.build())).then_some(plan.installing)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::launch::LaunchSpec;
    use crate::upgrade::swap::Swapped;
    use crate::upgrade::watchdog::{write_plan, WatchPlan};
    use crate::version::info::{BuildInfo, Opens, ZEND};

    fn setup(t: &TempDir, count: u32, failed_from: Option<&str>) -> Setup {
        Setup {
            info: BuildInfo {
                kind: ZEND.into(),
                count,
                hash: format!("h{count}"),
                local: String::new(),
                platform: "windows-x86_64".into(),
                opens: Opens::default(),
            },
            exe: t.path().join("motherd.exe"),
            upgrade_dir: t.path().join("upgrade"),
            keep_builds: 5,
            failed_from: failed_from.map(String::from),
        }
    }

    /// The plan the hand-off writes before it posts the build to keep.
    fn plan(t: &TempDir, installing: &str) {
        write_plan(&WatchPlan {
            old_pid: 1,
            launch: LaunchSpec {
                exe: t.path().join("motherd.exe"),
                args: Vec::new(),
                cwd: t.path().to_path_buf(),
            },
            swapped: Swapped {
                exe: t.path().join("keep/stage/motherd.exe"),
                kept: t.path().join("upgrade/previous"),
                from_kept: None,
            },
            installing: installing.into(),
            upgrade_dir: t.path().join("upgrade"),
            log: None::<PathBuf>,
            ready_within_ms: 1_000,
            keep_builds: 5,
            protect: Vec::new(),
            leases: Vec::new(),
        })
        .unwrap();
    }

    /// **A build back under a plan naming another build is back after it
    /// failed.** The build the plan installed starting under it is the install
    /// succeeding; no plan is no install; and a failure the process was told
    /// of is the one it reports.
    #[test]
    fn a_process_restored_under_anothers_plan_names_it_failed() {
        let t = TempDir::new().unwrap();
        let dir = t.path().join("upgrade");
        assert_eq!(failed_install(&setup(&t, 12, None), &dir), None, "no plan");

        plan(&t, "13-h13");
        assert_eq!(
            failed_install(&setup(&t, 12, None), &dir),
            Some("13-h13".into()),
            "keep rolled 13 back and restored 12"
        );
        assert_eq!(
            failed_install(&setup(&t, 13, None), &dir),
            None,
            "13 started under its own plan"
        );
        assert_eq!(
            failed_install(&setup(&t, 12, Some("14-h14")), &dir),
            Some("14-h14".into()),
            "the watchdog's flag names it outright"
        );
    }
}
