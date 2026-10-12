//! Installing a build (§3.9, §3.11), and what the process it starts finishes.
//!
//! **The install.** Once a target is decided ([`target`]), its artifact is
//! verified or kept here, and the build can open this member's substrate
//! ([`check`]), the member takes the install's leases — `upgrade/<app>` for each
//! app it serves, or one of a forced rollout's `upgrade/<app>/<i>` — and hands
//! the install to its [`Installer`]: the process drains, flushes, releases its
//! substrate, closes its sockets, swaps, starts the watchdog and exits. An
//! upgrading member takes no lease. A build that cannot open the substrate
//! strands the member, and its install is never attempted.
//!
//! **The leases outlive the process.** An upgrade lease runs for [`LEASE_TTL`]
//! and is not renewed: it is held through the restart, so no other member
//! installs while this one is down. The process that starts — the new build, or
//! the old one restored — takes each lease back once it is ready, as the same
//! node, which for Paxos is a renewal, and releases it for the next member. The
//! new build writes `<data>/upgrade/ok` first, which is what the watchdog waits
//! for.

use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::decide::{target, Lease};
use super::served::HeldAs;
use super::Upgrader;
use crate::frame::BuildId;
use crate::member::Link;
use crate::replicate::Phase;
use crate::upgrade::compat::{check, Needs};
use crate::upgrade::decisions::build_name;
use crate::upgrade::kept::name as kept_name;
use crate::upgrade::swap::Source;
use crate::upgrade::watchdog::{read_plan, FAILED, OK, PLAN, READY_WITHIN};
use crate::version::info::Opens;

/// How long an upgrade lease runs: through the swap, the restart and the new
/// build's load, with room to spare.
pub const LEASE_TTL: Duration = Duration::from_secs(READY_WITHIN.as_secs() + 5 * 60);

/// The term a returning process takes a lease back for, to release it.
pub const HANDBACK_TTL: Duration = Duration::from_secs(5);

/// How often a member asks again for a lease another member holds.
pub const LEASE_RETRY: Duration = Duration::from_secs(5);

/// The process side of an install.
pub trait Installer: Send + Sync {
    /// Stop serving — drain, flush, release the substrate, close every
    /// socket — then swap in `job.source`, start the watchdog, and exit. `Err`
    /// when it cannot begin: the process is already stopping.
    fn install(&self, job: InstallJob) -> Result<(), String>;
}

/// An install, as the installer carries it out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallJob {
    pub build: BuildId,
    pub source: Source,
    /// The upgrade leases held for it, which the next process hands back.
    pub leases: Vec<String>,
    /// The builds never pruned after it: a forced rollout's target.
    pub protect: Vec<BuildId>,
}

/// Leases a returning process takes back and releases.
#[derive(Debug, Default)]
pub struct Handback {
    /// Each lease's app and key, and whether it has been released.
    pub keys: Vec<(String, String, bool)>,
    pub read: bool,
    pub ok_written: bool,
    pub last_try: Option<Instant>,
}

/// The app an upgrade lease belongs to: `upgrade/<app>[/<i>]`.
fn app_of(key: &str) -> Option<&str> {
    key.strip_prefix("upgrade/")?.split('/').next()
}

impl Upgrader {
    /// Whether every app it serves is joined and ready.
    fn ready(&self) -> bool {
        let apps = self.apps();
        !apps.is_empty()
            && apps.iter().all(|m| {
                m.upgrading().is_none()
                    && match m.replicator() {
                        Some(r) => {
                            m.socket_link() == Link::Joined
                                && r.caught_up()
                                && r.phase() == Phase::Joined
                        }
                        None => !m.admitted_sockets().is_empty(),
                    }
            })
    }

    /// Write `ok` for the watchdog, and hand the install's leases back.
    pub(super) async fn finish_install(self: &Arc<Self>) {
        let dir = self.setup.upgrade_dir.clone();
        if !self.state().handback.read {
            let plan = read_plan(&dir.join(PLAN)).ok();
            let mut s = self.state();
            s.handback.read = true;
            if let Some(p) = plan {
                s.handback.keys = p
                    .leases
                    .iter()
                    .filter_map(|k| Some((app_of(k)?.to_string(), k.clone(), false)))
                    .collect();
                s.handback.ok_written = p.installing != build_name(&self.setup.info.build());
            } else {
                s.handback.ok_written = true;
            }
        }
        let (done, ok_due) = {
            let s = self.state();
            (
                s.handback.ok_written && s.handback.keys.iter().all(|(_, _, r)| *r),
                !s.handback.ok_written,
            )
        };
        if done || !self.ready() {
            return;
        }
        if ok_due {
            let wrote = fs::create_dir_all(&dir).and_then(|()| fs::write(dir.join(OK), b""));
            match wrote {
                Ok(()) => {
                    let _ = fs::remove_file(dir.join(FAILED));
                    let mut s = self.state();
                    s.handback.ok_written = true;
                    s.failure = None;
                    tracing::warn!(
                        build = build_name(&self.setup.info.build()),
                        "the new build is ready: told the watchdog"
                    );
                }
                Err(e) => tracing::error!("the upgrade's ok was not written: {e}"),
            }
        }
        let due = self
            .state()
            .handback
            .last_try
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(1));
        if !due {
            return;
        }
        self.state().handback.last_try = Some(Instant::now());
        let keys: Vec<(String, String, bool)> = self.state().handback.keys.clone();
        let mut released = Vec::new();
        for (app, key, done) in keys {
            if done {
                continue;
            }
            let Some(r) = self.member(&app).and_then(|m| m.replicator().cloned()) else {
                released.push(key);
                continue;
            };
            if r.holds(key.as_bytes()) {
                r.release_lease(key.as_bytes(), r.own_through());
                released.push(key);
            } else {
                r.acquire(key.as_bytes(), HANDBACK_TTL);
            }
        }
        let all = {
            let mut s = self.state();
            for (_, key, done) in s.handback.keys.iter_mut() {
                if released.contains(key) {
                    *done = true;
                }
            }
            s.handback.keys.iter().all(|(_, _, d)| *d)
        };
        if all {
            let _ = fs::remove_file(dir.join(PLAN));
        }
    }

    /// Where `build` would be installed from, and what it opens: a kept build,
    /// or its verified downloads.
    fn source_for(&self, build: &BuildId) -> Option<(Source, Opens)> {
        let s = self.state();
        let kind = &self.setup.info.kind;
        if let Some(k) = s
            .served
            .iter()
            .find(|i| i.build == *build && i.from == HeldAs::Kept && i.artifact.kind == *kind)
        {
            return Some((
                Source::Kept(kept_name(build)),
                k.info.as_ref().map(|i| i.opens.clone()).unwrap_or_default(),
            ));
        }
        let exe = s.served.iter().find(|i| {
            i.build == *build && i.from == HeldAs::Verified && i.artifact.kind == *kind
        })?;
        let opens = exe.info.as_ref()?.opens.clone();
        Some((Source::Fetched(exe.path.clone()), opens))
    }

    /// What a build must open to replace this one: the log format and model
    /// its own build opens, and every record type its substrates hold.
    pub(super) fn needs(&self) -> Needs {
        Needs {
            log_format: self.setup.info.opens.log_format,
            census: self.state().census.clone(),
            model: self.setup.info.opens.model.clone(),
        }
    }

    pub(super) async fn install_tick(self: &Arc<Self>) {
        let chosen = {
            let s = self.state();
            s.forced.clone().or_else(|| target(&s.facts))
        };
        let Some(t) = chosen else {
            let mut s = self.state();
            if s.stranded.is_none() {
                s.step.clear();
            }
            return;
        };
        let upgrading = matches!(t.lease, Lease::None);
        let Some((source, opens)) = self.source_for(&t.build) else {
            self.state().step = if upgrading { "upgrading" } else { "waiting" }.into();
            return;
        };
        if let Err(reason) = check(&opens, &self.needs()) {
            let mut s = self.state();
            s.stranded = Some(reason);
            s.step = "stranded".into();
            return;
        }
        let leases: Vec<String> = match &t.lease {
            Lease::None => Vec::new(),
            Lease::Auto => self
                .apps()
                .iter()
                .map(|m| format!("upgrade/{}", m.app()))
                .collect(),
            Lease::Forced { app, parallel, .. } => {
                let held = (0..*parallel)
                    .map(|i| format!("upgrade/{app}/{i}"))
                    .find(|k| {
                        self.member(app)
                            .and_then(|m| m.replicator().cloned())
                            .is_some_and(|r| r.holds(k.as_bytes()))
                    });
                // Not holding one yet: ask for one of them at random, so
                // members waiting together spread across the slots.
                let slot = held.unwrap_or_else(|| {
                    format!("upgrade/{app}/{}", rand::random::<u32>() % parallel)
                });
                let mut keys = vec![slot];
                keys.extend(
                    self.apps()
                        .iter()
                        .filter(|m| m.app() != app)
                        .map(|m| format!("upgrade/{}", m.app())),
                );
                keys
            }
        };
        let mut all_held = true;
        for key in &leases {
            let Some(r) = app_of(key)
                .and_then(|a| self.member(a))
                .and_then(|m| m.replicator().cloned())
            else {
                continue;
            };
            if r.holds(key.as_bytes()) {
                continue;
            }
            all_held = false;
            let due = self
                .state()
                .lease_try
                .is_none_or(|at| at.elapsed() >= LEASE_RETRY);
            if due {
                r.acquire(key.as_bytes(), LEASE_TTL);
            }
        }
        if !all_held {
            let mut s = self.state();
            s.step = "waiting".into();
            if s.lease_try.is_none_or(|at| at.elapsed() >= LEASE_RETRY) {
                s.lease_try = Some(Instant::now());
            }
            return;
        }
        let protect: Vec<BuildId> = match &t.lease {
            Lease::Forced { .. } => vec![t.build.clone()],
            _ => Vec::new(),
        };
        {
            let mut s = self.state();
            s.step = "installing".into();
            s.installing = true;
        }
        tracing::warn!(
            build = build_name(&t.build),
            ?leases,
            "installing: draining, then swapping in the new build"
        );
        let job = InstallJob {
            build: t.build,
            source,
            leases,
            protect,
        };
        if let Err(e) = self.installer.install(job) {
            tracing::error!("the install did not start: {e}");
            let mut s = self.state();
            s.installing = false;
            s.step = "waiting".into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lease_names_its_app() {
        assert_eq!(app_of("upgrade/zend"), Some("zend"));
        assert_eq!(app_of("upgrade/npcd/1"), Some("npcd"));
        assert_eq!(app_of("world/w1"), None);
    }
}
