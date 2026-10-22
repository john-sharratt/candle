//! What a member does about the builds it knows of (§3.5, §3.8–§3.11), decided
//! from the facts alone: which build it installs, under which lease, and which
//! it fetches ahead of that.
//!
//! **Installs**, first match wins:
//!
//! 1. **Admitted below the floor** — the floor's build, at once, with no lease:
//!    an upgrading member serves nothing the app would lose (§3.5). One whose
//!    install of that build already failed here takes instead the newest build
//!    at or above the floor that every app it serves certified and it never
//!    refused, with no lease either. With none, it installs nothing until one
//!    certifies. The refusal is this member's alone — the group never hears of
//!    it — so waiting on the floor it was admitted under would wait until its
//!    socket happened to close, however many builds above it certified.
//! 2. **A forced rollout naming it** — the latest `rollout/<app>/<n>` of any
//!    app it serves whose targets include it, while it is not on that build
//!    yet, under one of the rollout's `parallel` leases (§3.11). Being named
//!    retries a build it once refused, and installs over a local build.
//! 3. **A certified build** — the newest build above its own that every app it
//!    serves has certified (`Pass` is the latest verdict), no app pauses, and
//!    it never refused; under `upgrade/<app>` (§3.9). A local build never
//!    upgrades this way (§3.10).
//!
//! **Fetches** the build it would install, or else the newest build offered
//! above its own that is not rejected or refused — distribution runs while the
//! candidate certifies, so the artifact is there at the deadline.

use std::collections::{BTreeMap, BTreeSet};

use crate::frame::BuildId;
use crate::upgrade::decisions::{build_name, parse_build, Rollout, Settled};
use crate::version::info::BuildInfo;

/// What a member knows when it decides.
#[derive(Debug, Clone, Default)]
pub struct Facts {
    pub own: Option<BuildInfo>,
    /// Its node in each app it serves, in hex as rollouts name it: a motherd
    /// is a node of each substrate.
    pub nodes: BTreeMap<String, String>,
    /// The apps it serves.
    pub apps: Vec<String>,
    /// The floor's build, when it was admitted below it.
    pub upgrading: Option<BuildId>,
    /// Every build offered or held that it has heard of.
    pub known: Vec<BuildId>,
    /// The latest verdict on each build, per app: `(app, count-hash)`.
    pub verdicts: BTreeMap<(String, String), Settled>,
    /// Each app's latest forced rollout, and its number.
    pub rollouts: BTreeMap<String, (u64, Rollout)>,
    /// The apps whose automatic rollout is paused.
    pub paused: BTreeSet<String>,
    /// Builds an install of which failed here: `count-hash`.
    pub refused: BTreeSet<String>,
}

/// Which lease an install is made under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lease {
    /// None: admitted below the floor.
    None,
    /// `upgrade/<app>` for every app it serves.
    Auto,
    /// One of `upgrade/<app>/<i>`, `i < parallel`, of the forced rollout
    /// `rollout/<app>/<n>`.
    Forced { app: String, n: u64, parallel: u32 },
}

/// A build to install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub build: BuildId,
    pub lease: Lease,
}

impl Facts {
    fn rejected(&self, build: &BuildId) -> bool {
        let name = build_name(build);
        self.verdicts
            .iter()
            .any(|((_, b), v)| *b == name && *v == Settled::Fail)
    }

    fn certified(&self, build: &BuildId) -> bool {
        let name = build_name(build);
        !self.apps.is_empty()
            && self
                .apps
                .iter()
                .all(|app| self.verdicts.get(&(app.clone(), name.clone())) == Some(&Settled::Pass))
    }
}

/// The build this member installs now, if any.
pub fn target(f: &Facts) -> Option<Target> {
    let own = f.own.as_ref()?;
    let running = own.build();
    if let Some(floor) = &f.upgrading {
        if f.refused.contains(&build_name(floor)) {
            return f
                .known
                .iter()
                .filter(|b| !b.is_local() && b.count >= floor.count && **b != running)
                .filter(|b| f.certified(b) && !f.refused.contains(&build_name(b)))
                .max_by_key(|b| b.count)
                .map(|b| Target {
                    build: b.clone(),
                    lease: Lease::None,
                });
        }
        if *floor != running {
            return Some(Target {
                build: floor.clone(),
                lease: Lease::None,
            });
        }
    }
    let forced = f
        .rollouts
        .iter()
        .filter(|(app, (_, r))| f.nodes.get(*app).is_some_and(|n| r.targets.contains(n)))
        .filter_map(|(app, (n, r))| {
            let (count, hash) = parse_build(&r.build)?;
            let build = BuildId {
                count,
                hash,
                local: String::new(),
            };
            (build != running).then(|| Target {
                build,
                lease: Lease::Forced {
                    app: app.clone(),
                    n: *n,
                    parallel: r.parallel.max(1),
                },
            })
        })
        .max_by_key(|t| match &t.lease {
            Lease::Forced { n, .. } => *n,
            _ => 0,
        });
    if forced.is_some() {
        return forced;
    }
    if running.is_local() || f.apps.iter().any(|a| f.paused.contains(a)) {
        return None;
    }
    f.known
        .iter()
        .filter(|b| !b.is_local() && b.count > running.count)
        .filter(|b| f.certified(b) && !f.refused.contains(&build_name(b)))
        .max_by_key(|b| b.count)
        .map(|b| Target {
            build: b.clone(),
            lease: Lease::Auto,
        })
}

/// The build this member fetches now, if any: the one it would install, or
/// the newest offered above its own.
pub fn prefetch(f: &Facts) -> Option<BuildId> {
    if let Some(t) = target(f) {
        return Some(t.build);
    }
    let own = f.own.as_ref()?.build();
    if own.is_local() {
        return None;
    }
    f.known
        .iter()
        .filter(|b| !b.is_local() && b.count > own.count)
        .filter(|b| !f.rejected(b) && !f.refused.contains(&build_name(b)))
        .max_by_key(|b| b.count)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::version::info::{Opens, ZEND};

    fn b(count: u32) -> BuildId {
        BuildId {
            count,
            hash: format!("h{count}"),
            local: String::new(),
        }
    }

    fn facts(own: u32, local: &str) -> Facts {
        Facts {
            own: Some(BuildInfo {
                kind: ZEND.into(),
                count: own,
                hash: format!("h{own}"),
                local: local.into(),
                platform: "p".into(),
                opens: Opens::default(),
            }),
            nodes: BTreeMap::from([("zend".into(), "0000000000000007".into())]),
            apps: vec!["zend".into()],
            ..Facts::default()
        }
    }

    fn rollout(build: &str, targets: &[&str], parallel: u32) -> Rollout {
        Rollout {
            build: build.into(),
            from: "0000000000000001".into(),
            targets: targets.iter().map(|t| t.to_string()).collect(),
            parallel,
            artifacts: vec![],
            rejects: vec![],
            by: "ops".into(),
            at_ms: 0,
        }
    }

    /// Below the floor, the floor's build is installed at once, without a
    /// lease — unless an install of it already failed here.
    #[test]
    fn an_upgrading_member_takes_the_floor_without_a_lease() {
        let mut f = facts(10, "");
        f.upgrading = Some(b(12));
        assert_eq!(
            target(&f),
            Some(Target {
                build: b(12),
                lease: Lease::None
            })
        );
        f.refused.insert("12-h12".into());
        assert_eq!(target(&f), None);
        assert_eq!(prefetch(&f), None);
    }

    /// Below a floor whose build failed here, the newest build at or above the
    /// floor that certified and was never refused is installed instead, with
    /// no lease — not nothing for as long as the socket stays open. A build
    /// below the floor, one not certified, or one refused too is not.
    #[test]
    fn a_refused_floor_gives_way_to_a_newer_certified_build() {
        let mut f = facts(10, "");
        f.upgrading = Some(b(12));
        f.refused.insert("12-h12".into());
        f.known = vec![b(11), b(12), b(13), b(14)];
        for n in [11, 12, 13] {
            f.verdicts
                .insert(("zend".into(), format!("{n}-h{n}")), Settled::Pass);
        }
        assert_eq!(
            target(&f),
            Some(Target {
                build: b(13),
                lease: Lease::None
            }),
            "14 has not certified, 11 is below the floor"
        );
        f.verdicts
            .insert(("zend".into(), "14-h14".into()), Settled::Pass);
        assert_eq!(target(&f).map(|t| t.build), Some(b(14)));
        f.refused.insert("14-h14".into());
        assert_eq!(target(&f).map(|t| t.build), Some(b(13)));
        f.refused.insert("13-h13".into());
        assert_eq!(target(&f), None, "nothing above the floor is left to take");
    }

    /// A certified build is installed under `upgrade/<app>`; one not
    /// certified, rejected, refused, paused, or below its own is not — but it
    /// is fetched while it certifies.
    #[test]
    fn only_a_certified_build_is_installed_automatically() {
        let mut f = facts(10, "");
        f.known = vec![b(9), b(12)];
        assert_eq!(target(&f), None);
        assert_eq!(prefetch(&f), Some(b(12)), "fetched while certifying");
        f.verdicts
            .insert(("zend".into(), "12-h12".into()), Settled::Pass);
        assert_eq!(
            target(&f),
            Some(Target {
                build: b(12),
                lease: Lease::Auto
            })
        );
        f.paused.insert("zend".into());
        assert_eq!(target(&f), None);
        f.paused.clear();
        f.refused.insert("12-h12".into());
        assert_eq!(target(&f), None);
        assert_eq!(prefetch(&f), None, "a refused build is not fetched");
        f.refused.clear();
        f.verdicts
            .insert(("zend".into(), "12-h12".into()), Settled::Fail);
        assert_eq!(target(&f), None);
        assert_eq!(prefetch(&f), None, "a rejected build is not fetched");
    }

    /// A motherd installs only what both of its apps certified.
    #[test]
    fn every_app_served_must_certify() {
        let mut f = facts(10, "");
        f.apps = vec!["zend".into(), "npcd".into()];
        f.known = vec![b(12)];
        f.verdicts
            .insert(("zend".into(), "12-h12".into()), Settled::Pass);
        assert_eq!(target(&f), None);
        f.verdicts
            .insert(("npcd".into(), "12-h12".into()), Settled::Pass);
        assert_eq!(target(&f).map(|t| t.build), Some(b(12)));
    }

    /// A local build is never upgraded automatically, nor fetches.
    #[test]
    fn a_local_build_is_left_alone() {
        let mut f = facts(10, "abcd");
        f.known = vec![b(12)];
        f.verdicts
            .insert(("zend".into(), "12-h12".into()), Settled::Pass);
        assert_eq!(target(&f), None);
        assert_eq!(prefetch(&f), None);
    }

    /// A forced rollout naming this member installs its build — down as well
    /// as up, over a local build, and a refused one — under one of its
    /// leases; the latest rollout wins, and one not naming it does nothing.
    #[test]
    fn a_forced_rollout_names_its_targets() {
        let mut f = facts(12, "abcd");
        f.refused.insert("10-h10".into());
        f.rollouts.insert(
            "zend".into(),
            (3, rollout("10-h10", &["0000000000000007"], 2)),
        );
        assert_eq!(
            target(&f),
            Some(Target {
                build: b(10),
                lease: Lease::Forced {
                    app: "zend".into(),
                    n: 3,
                    parallel: 2
                }
            })
        );
        f.rollouts.insert(
            "zend".into(),
            (4, rollout("10-h10", &["0000000000000008"], 1)),
        );
        assert_eq!(target(&f), None);
        // On the build already: done.
        let mut on = facts(10, "");
        on.rollouts.insert(
            "zend".into(),
            (3, rollout("10-h10", &["0000000000000007"], 1)),
        );
        assert_eq!(target(&on), None);
    }
}
