//! What the upgrade commands ask of a member (§12.4): to settle a build by
//! hand, pause or resume an app's rollout, decide a forced rollout — each a
//! Paxos decision this member proposes and waits for — and, for this member
//! alone, to roll back to a kept build, install a verified one now, or prune
//! the builds it keeps.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::{sleep, Instant};

use super::decide::{Lease, Target};
use super::served::HeldAs;
use super::Upgrader;
use crate::clock::now_ms;
use crate::frame::BuildId;
use crate::upgrade::decisions::{
    build_name, certify_prefix, encode, next, parse_build, pause_prefix, rollout_prefix, Pause,
    Rollout, Settled, Verdict,
};
use crate::upgrade::kept::{builds_dir, list, prune, remove};

/// How long a command waits for the group to decide what it proposed.
pub const DECIDE_WAIT: Duration = Duration::from_secs(10);

/// How often it looks.
const LOOK: Duration = Duration::from_millis(100);

/// A `count-hash` as a build at `origin/main`.
pub fn build_of(name: &str) -> Result<BuildId, String> {
    let (count, hash) =
        parse_build(name).ok_or_else(|| format!("{name:?} is not a build: `count-hash`"))?;
    Ok(BuildId {
        count,
        hash,
        local: String::new(),
    })
}

impl Upgrader {
    /// Propose `value` under the next number of `prefix` in `app`, and wait
    /// until the group decided it there; its number. A number another value
    /// took first moves it to the next.
    async fn decide_next(&self, app: &str, prefix: &str, value: Vec<u8>) -> Result<u64, String> {
        let r = self
            .member(app)
            .and_then(|m| m.replicator().cloned())
            .ok_or_else(|| format!("{app} is not joined here"))?;
        let deadline = Instant::now() + DECIDE_WAIT;
        loop {
            let held = r.decisions(prefix).await;
            let n = next(&held, prefix);
            let key = format!("{prefix}{n}");
            if !r.propose(key.as_bytes(), value.clone()) {
                return Err("this member proposes nothing in fault mode".into());
            }
            loop {
                let held = r.decisions(prefix).await;
                if let Some((_, v)) = held.iter().find(|(k, _)| *k == key) {
                    if *v == value {
                        return Ok(n);
                    }
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(format!(
                        "no quorum decided {key} within {} s",
                        DECIDE_WAIT.as_secs()
                    ));
                }
                sleep(LOOK).await;
            }
        }
    }

    /// `certify-anyway` and `reject-build`: settle `build` in `app` by hand.
    pub async fn settle(
        &self,
        app: &str,
        build: &str,
        result: Settled,
        by: &str,
        reason: &str,
    ) -> Result<u64, String> {
        let b = build_of(build)?;
        let verdict = Verdict {
            result,
            by: by.to_string(),
            at_ms: now_ms(),
            reason: reason.to_string(),
            figures: Vec::new(),
        };
        self.decide_next(app, &certify_prefix(app, &b), encode(&verdict))
            .await
    }

    /// `pause-rollout` and `resume-rollout`.
    pub async fn pause(&self, app: &str, paused: bool, by: &str) -> Result<u64, String> {
        let p = Pause {
            paused,
            by: by.to_string(),
            at_ms: now_ms(),
        };
        self.decide_next(app, &pause_prefix(app), encode(&p)).await
    }

    /// `rollout`: decide a forced rollout's confirmed plan, then — for a
    /// downgrade — a rejecting verdict on each build it rolls back from
    /// (§3.11). The rolled-out build itself is accepted.
    pub async fn rollout(&self, app: &str, plan: &Rollout) -> Result<u64, String> {
        build_of(&plan.build)?;
        let n = self
            .decide_next(app, &rollout_prefix(app), encode(plan))
            .await?;
        let why = format!("rolled back from by rollout {n}");
        for rejected in &plan.rejects {
            self.settle(app, rejected, Settled::Fail, &plan.by, &why)
                .await?;
        }
        let rejected_before = self
            .state()
            .facts
            .verdicts
            .get(&(app.to_string(), plan.build.clone()))
            == Some(&Settled::Fail);
        if rejected_before {
            self.settle(
                app,
                &plan.build,
                Settled::Pass,
                &plan.by,
                &format!("rolled out by rollout {n}"),
            )
            .await?;
        }
        Ok(n)
    }

    /// The floor this member's sockets were admitted under, the highest.
    fn floor(&self) -> Option<BuildId> {
        self.apps()
            .iter()
            .flat_map(|m| m.admitted_sockets().into_iter().filter_map(|c| m.floor(c)))
            .max_by_key(|f| f.count)
    }

    /// `rollback`: put this member on `build` — by default the build it ran
    /// before — from its own `builds/`, under `upgrade/<app>`. A build below
    /// the floor is refused: the member would be admitted to upgrade and put
    /// straight back.
    pub fn rollback(self: &Arc<Self>, build: Option<&str>) -> Result<String, String> {
        let kept = list(&builds_dir(&self.setup.exe));
        let k = match build {
            Some(name) => kept
                .iter()
                .find(|k| k.name == name)
                .ok_or_else(|| format!("no kept build {name} here"))?,
            None => kept
                .first()
                .ok_or_else(|| "no build is kept here to roll back to".to_string())?,
        };
        let b = k.json.info.build();
        if let Some(floor) = self.floor().filter(|f| b.count < f.count) {
            return Err(format!(
                "{} is below the floor {}: roll the app back with `rollout`, which moves the floor",
                k.name,
                build_name(&floor)
            ));
        }
        self.state().forced = Some(Target {
            build: b,
            lease: Lease::Auto,
        });
        Ok(k.name.clone())
    }

    /// `upgrade-now`: install the newest verified build above its own at once,
    /// under `upgrade/<app>`.
    pub fn upgrade_now(self: &Arc<Self>) -> Result<String, String> {
        let own = self.setup.info.build();
        let newest = self
            .state()
            .served
            .iter()
            .filter(|i| {
                i.from == HeldAs::Verified
                    && i.artifact.kind == self.setup.info.kind
                    && i.build.count > own.count
            })
            .map(|i| i.build.clone())
            .max_by_key(|b| b.count)
            .ok_or_else(|| "no verified build above this one is held here".to_string())?;
        let name = build_name(&newest);
        self.state().forced = Some(Target {
            build: newest,
            lease: Lease::Auto,
        });
        Ok(name)
    }

    /// `prune-builds`: delete the kept builds `names`, or all but the newest
    /// `keep` — never one a rollout in progress targets.
    pub fn prune_builds(
        &self,
        names: &[String],
        keep: Option<usize>,
    ) -> Result<Vec<String>, String> {
        let builds = builds_dir(&self.setup.exe);
        let protect: Vec<BuildId> = self
            .state()
            .facts
            .rollouts
            .values()
            .filter_map(|(_, r)| build_of(&r.build).ok())
            .collect();
        let gone = match keep {
            Some(k) => prune(&builds, k, &protect).map_err(|e| e.to_string())?,
            None => remove(&builds, names, &protect)?,
        };
        self.state().announce_due = true;
        self.state().scanned = false;
        Ok(gone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_build_is_named_count_hash() {
        assert_eq!(
            build_of("4815-9f8e7d6"),
            Ok(BuildId {
                count: 4815,
                hash: "9f8e7d6".into(),
                local: String::new()
            })
        );
        assert_eq!(
            build_of("latest").unwrap_err(),
            "\"latest\" is not a build: `count-hash`"
        );
    }
}
