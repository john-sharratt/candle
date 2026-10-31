//! What the group has decided about builds, read from the substrate once a
//! second (§3.8, §3.11): the latest verdict on each build per app, each app's
//! latest forced rollout, and whether its automatic rollout is paused — and
//! whether its own build is rejected, which its next `HELLO` says. And
//! what a member does about a verdict on its own build: rejected, it stops
//! taking work — a candidate in fault mode, any other drained — and accepted
//! after all, it takes work again. A rejected build's downloads are removed.
//!
//! **The store is asked with a bound.** Each app's reads run as one task
//! ([`FactsRead`]); a read started this tick is waited for up to the tick's
//! store wait, and one still running from an earlier tick is looked at without
//! waiting. While any app's store has not answered, the facts stay as last
//! read, the step names the store and the call it is blocked on, and the tick
//! acts on nothing the facts decide.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::decide::{target, Lease, Target};
use super::facts_read::{verdicts_prefix, AppFacts, FactsRead};
use super::served::HeldAs;
use super::Upgrader;
use crate::coord::VerdictLine;
use crate::frame::BuildId;
use crate::upgrade::decisions::{
    build_name, latest, parse_build, pause_prefix, rollout_prefix, Pause, Rollout, Settled, Verdict,
};

impl Upgrader {
    /// Every app's store's answers, or the step to show while one has not
    /// answered: a read started now is waited for up to `wait`, one still in
    /// flight from an earlier tick not at all.
    async fn read_facts(&self, wait: Duration) -> Result<BTreeMap<String, AppFacts>, String> {
        let mut held = std::mem::take(&mut self.state().reads);
        let mut reads = Vec::new();
        for m in self.apps() {
            let Some(r) = m.replicator().cloned() else {
                continue;
            };
            match held.remove(m.app()) {
                Some(read) => reads.push((read, false)),
                None => reads.push((FactsRead::start(m.app(), r), true)),
            }
        }
        let deadline = Instant::now() + wait;
        let mut answered = BTreeMap::new();
        let mut waiting = BTreeMap::new();
        let mut blocked = None;
        for (read, fresh) in reads {
            let budget = if fresh {
                deadline.saturating_duration_since(Instant::now())
            } else {
                Duration::ZERO
            };
            let app = read.app().to_string();
            match read.wait(budget).await {
                Ok(facts) => {
                    answered.insert(app, facts);
                }
                Err(read) => {
                    blocked.get_or_insert_with(|| read.blocked_step(wait));
                    waiting.insert(app, read);
                }
            }
        }
        self.state().reads = waiting;
        match blocked {
            Some(step) => Err(step),
            None => Ok(answered),
        }
    }

    /// Read the group's decisions and the census, waiting up to `wait` for
    /// each app's store. `false` when a store has not answered: the facts are
    /// kept as last read and the step says what it is blocked on.
    pub(super) async fn refresh_facts(&self, wait: Duration) -> bool {
        let mut answered = match self.read_facts(wait).await {
            Ok(answered) => answered,
            Err(step) => {
                let mut s = self.state();
                if s.blocked.is_none() {
                    tracing::warn!("{step}: the upgrades act on nothing until it answers");
                }
                s.blocked = Some(step.clone());
                s.step = step;
                return false;
            }
        };
        if let Some(was) = self.state().blocked.take() {
            tracing::warn!("the upgrades' store reads answer again, after: {was}");
        }
        let mut verdicts = BTreeMap::new();
        let mut lines: BTreeMap<String, Vec<VerdictLine>> = BTreeMap::new();
        let mut rollouts = BTreeMap::new();
        let mut paused = BTreeSet::new();
        let mut census: Vec<u8> = Vec::new();
        let own = build_name(&self.setup.info.build());
        for m in self.apps() {
            let app = m.app().to_string();
            let Some(facts) = answered.remove(&app) else {
                continue;
            };
            census.extend(facts.census);
            let prefix = verdicts_prefix(&app);
            let mut by_build: BTreeMap<String, Vec<(String, Vec<u8>)>> = BTreeMap::new();
            for (k, v) in facts.certify {
                let Some((build, _)) = k
                    .strip_prefix(&prefix)
                    .and_then(|rest| rest.rsplit_once('/'))
                else {
                    continue;
                };
                by_build.entry(build.to_string()).or_default().push((k, v));
            }
            for (build, items) in by_build {
                let Some((n, v)) = latest::<Verdict>(&items, &format!("{prefix}{build}/")) else {
                    continue;
                };
                verdicts.insert((app.clone(), build.clone()), v.result);
                lines.entry(app.clone()).or_default().push(VerdictLine {
                    build,
                    n,
                    verdict: v,
                });
            }
            m.set_rejected(verdicts.get(&(app.clone(), own.clone())) == Some(&Settled::Fail));
            let rp = rollout_prefix(&app);
            if let Some(rollout) = latest::<Rollout>(&facts.rollouts, &rp) {
                rollouts.insert(app.clone(), rollout);
            }
            let pp = pause_prefix(&app);
            if let Some((_, p)) = latest::<Pause>(&facts.pauses, &pp) {
                if p.paused {
                    paused.insert(app.clone());
                }
            }
        }
        let upgrading = self.apps().iter().find_map(|m| m.upgrading());
        let rejected: Vec<BuildId> = verdicts
            .iter()
            .filter(|(_, v)| **v == Settled::Fail)
            .filter_map(|((_, b), _)| parse_build(b))
            .map(|(count, hash)| BuildId {
                count,
                hash,
                local: String::new(),
            })
            .filter(|b| *b != self.setup.info.build())
            .collect();
        census.sort_unstable();
        census.dedup();
        {
            let mut s = self.state();
            s.census = census;
            s.facts.verdicts = verdicts;
            s.facts.rollouts = rollouts;
            s.facts.paused = paused;
            s.facts.upgrading = upgrading;
            s.verdicts = lines;
            // Downloads of a rejected build go; a build kept or running stays
            // where it is, the way back to it.
            s.served
                .retain(|i| !(rejected.contains(&i.build) && i.from == HeldAs::Verified));
            if s.fetch
                .as_ref()
                .is_some_and(|f| rejected.contains(&f.build))
            {
                s.fetch = None;
            }
        }
        for b in rejected {
            let parts = self.parts.clone();
            let _ = tokio::task::spawn_blocking(move || parts.remove(&b)).await;
        }
        true
    }

    /// Act on the verdicts on its own build. A member a forced rollout names
    /// takes the rollout's build instead: draining would leave it unable to
    /// take the lease it installs under.
    pub(super) async fn own_verdicts(self: &Arc<Self>) {
        let own = build_name(&self.setup.info.build());
        let rolling = matches!(
            target(&self.state().facts),
            Some(Target {
                lease: Lease::Forced { .. },
                ..
            })
        );
        if rolling {
            return;
        }
        for m in self.apps() {
            let Some(r) = m.replicator().cloned() else {
                continue;
            };
            let app = m.app().to_string();
            let (verdict, reason, candidate, rejected) = {
                let s = self.state();
                let line = s
                    .verdicts
                    .get(&app)
                    .and_then(|l| l.iter().find(|v| v.build == own));
                (
                    line.map(|l| l.verdict.result),
                    line.map(|l| l.verdict.reason.clone()).unwrap_or_default(),
                    s.candidates.contains_key(&app),
                    s.rejected_self.contains(&app),
                )
            };
            match verdict {
                Some(Settled::Fail) if !rejected => {
                    self.state().rejected_self.insert(app.clone());
                    tracing::error!(app, build = own, "this build is rejected: {reason}");
                    if candidate {
                        r.certification_failed(reason).await;
                    } else {
                        r.drain().await;
                    }
                }
                Some(Settled::Pass) if rejected => {
                    self.state().rejected_self.remove(&app);
                    tracing::warn!(
                        app,
                        build = own,
                        "this build is accepted: taking work again"
                    );
                    r.certification_accepted();
                    if r.drained() {
                        if let Err(e) = r.undrain() {
                            tracing::warn!(app, "could not take work again yet: {e}");
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
