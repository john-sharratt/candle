//! A gateway's side of A/B upgrades (§3.5, §3.8, §3.9, §3.11), per app: the
//! pending floor and its grace period, the candidate and its certification,
//! the builds the group rejected, the floor rising once a build is certified or
//! rolled out, and the members a rollout leaves behind.
//!
//! Everything here is in memory, like the floor itself (§3.5): a gateway that
//! restarts learns the group's verdicts and rollouts again from its members'
//! `COORD`. It decides; the hub acts — [`Action`]s are what it sends, raises
//! and closes.
//!
//! | Module | |
//! |---|---|
//! | [`stats`] | per-route live and mirrored samples, the live share and the mirror cap |
//! | [`mirror`] | comparing a mirrored call: `Exact` and `Shape` |
//! | [`verdict`] | the statistics: error test, latency ratios, mismatches, gaps |
//! | [`held`] | the builds members hold, from `BUILD_OFFER` and `BUILD_HAVE` |
//! | [`plan`] | a forced rollout's plan, or why it is refused |

pub mod held;
pub mod mirror;
pub mod plan;
pub mod stats;
pub mod verdict;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

use serde::Serialize;

use self::mirror::Compared;
use self::stats::{Mismatch, Sample, Share, Stats};
use self::verdict::{verdict, Figures, Overall};
use crate::close::{Close, CloseCode};
use crate::coord::UpgradeView;
use crate::dispatch::Mirror;
use crate::frame::{BuildId, MemberState, Verdict, VerdictResult};
use crate::upgrade::decisions::{build_name, parse_build, Settled};

/// The default grace period (`cluster: floor_grace_minutes`).
pub const GRACE_DEFAULT: Duration = Duration::from_secs(30 * 60);
/// The default live share (`cluster: canary_share`), in percent.
pub const CANARY_DEFAULT: u64 = 10;
/// How long after the floor rises a member below it is given to start its
/// install before a rollout counts it as left behind.
pub const ROLLOUT_SETTLE: Duration = Duration::from_secs(120);

/// How a gateway certifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpgradeConfig {
    pub grace: Duration,
    pub canary_pct: u64,
}

impl Default for UpgradeConfig {
    fn default() -> Self {
        Self {
            grace: GRACE_DEFAULT,
            canary_pct: CANARY_DEFAULT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateState {
    /// Admitted, still loading or catching up: its grace has not started.
    Loading,
    Certifying,
    /// This gateway sent its verdict; the group's is awaited.
    Passed,
    PassedWithGaps,
    NoOpinion,
    Failed,
}

/// The member a newer build arrived on.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub node: u64,
    pub build: BuildId,
    pub admitted_ms: u64,
    pub ready_ms: Option<u64>,
    pub state: CandidateState,
    pub stats: Stats,
    pub share: Share,
    /// This gateway's verdict, once sent, and when it was last sent: sent
    /// again until the group decides, so a candidate that missed it — its
    /// socket dropped, its process restarted — still hears it.
    pub sent: Option<(Verdict, u64)>,
}

/// How often a verdict is sent again while the group has not decided.
pub const VERDICT_AGAIN: Duration = Duration::from_secs(10);

/// What the hub does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Send this gateway's verdict to the candidate.
    Verdict {
        app: String,
        node: u64,
        verdict: Verdict,
    },
    /// The app's floor is now `build`, set by `by` — a certification or a
    /// rollout — in either direction.
    Floor {
        app: String,
        build: BuildId,
        by: String,
    },
    /// Close a member left behind by a rollout, or stranded below the floor.
    Close {
        app: String,
        node: u64,
        close: Close,
    },
    /// A coordination decision for the log.
    Note {
        app: String,
        subject: String,
        text: String,
    },
}

/// One member, as the end of a rollout weighs it.
#[derive(Debug, Clone)]
pub struct Below<'a> {
    pub node: u64,
    pub build: &'a BuildId,
    pub state: MemberState,
    pub upgrade: Option<&'a UpgradeView>,
}

#[derive(Debug, Clone, Default)]
struct AppUpgrades {
    pending: Option<Candidate>,
    deadline_ms: Option<u64>,
    /// When the floor last rose by a certification or a rollout.
    raised_ms: Option<u64>,
    /// The last forced rollout applied.
    rollout_n: u64,
    rejected: BTreeSet<(u32, String)>,
    /// The builds the group certified.
    certified: BTreeSet<(u32, String)>,
}

/// The page's view of one app's upgrade (§12.2 Upgrades).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UpgradesView {
    pub app: String,
    pub candidate: Option<CandidateView>,
    /// When the grace period ends.
    pub deadline_ms: Option<u64>,
    /// When the floor last rose by a certification or a rollout, while the
    /// rollout lasts.
    pub rollout_since_ms: Option<u64>,
    pub rejected: Vec<String>,
    pub certified: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CandidateView {
    pub node: String,
    pub build: String,
    pub state: CandidateState,
    pub ready_ms: Option<u64>,
    pub figures: Figures,
    pub mirrored: u64,
    pub mirror_eligible: u64,
    pub mismatches: Vec<Mismatch>,
}

#[derive(Debug, Default)]
pub struct Upgrades {
    cfg: UpgradeConfig,
    apps: HashMap<String, AppUpgrades>,
}

fn key(b: &BuildId) -> (u32, String) {
    (b.count, b.hash.clone())
}

fn note(app: &str, subject: String, text: String) -> Action {
    Action::Note {
        app: app.to_string(),
        subject,
        text,
    }
}

fn hex(node: u64) -> String {
    format!("{node:016x}")
}

impl Upgrades {
    pub fn new(cfg: UpgradeConfig) -> Self {
        Self {
            cfg,
            apps: HashMap::new(),
        }
    }

    pub fn config(&self) -> UpgradeConfig {
        self.cfg
    }

    fn app(&mut self, app: &str) -> &mut AppUpgrades {
        self.apps.entry(app.to_string()).or_default()
    }

    /// Whether the group rejected `build`.
    pub fn rejected(&self, app: &str, build: &BuildId) -> bool {
        self.apps
            .get(app)
            .is_some_and(|a| a.rejected.contains(&key(build)))
    }

    /// `node` was admitted on `build`, against the app's `floor`: above it, not
    /// local and not rejected, it is the app's candidate — unless an even
    /// newer one already is. A newer candidate raises the pending floor and
    /// keeps the deadline (§3.5).
    pub fn admitted(
        &mut self,
        app: &str,
        node: u64,
        build: &BuildId,
        floor: Option<&BuildId>,
        now_ms: u64,
    ) -> Vec<Action> {
        if build.is_local() || app == "agent" || floor.is_none_or(|f| build.count <= f.count) {
            return Vec::new();
        }
        let a = self.app(app);
        if a.rejected.contains(&key(build)) {
            return Vec::new();
        }
        if let Some(p) = &a.pending {
            if p.node == node && p.build == *build {
                return Vec::new();
            }
            if p.build.count >= build.count {
                return Vec::new();
            }
        }
        a.pending = Some(Candidate {
            node,
            build: build.clone(),
            admitted_ms: now_ms,
            ready_ms: None,
            state: CandidateState::Loading,
            stats: Stats::default(),
            share: Share::default(),
            sent: None,
        });
        vec![note(
            app,
            hex(node),
            format!(
                "candidate {}: pending floor {}, loading",
                build_name(build),
                build.count
            ),
        )]
    }

    /// `node`'s state changed: a candidate become ready starts its grace
    /// period and its certification (§3.8).
    pub fn capacity(
        &mut self,
        app: &str,
        node: u64,
        state: MemberState,
        now_ms: u64,
    ) -> Vec<Action> {
        let grace = self.cfg.grace.as_millis() as u64;
        let a = self.app(app);
        let Some(p) = a.pending.as_mut().filter(|p| p.node == node) else {
            return Vec::new();
        };
        if state != MemberState::Ready || p.ready_ms.is_some() {
            return Vec::new();
        }
        p.ready_ms = Some(now_ms);
        p.state = CandidateState::Certifying;
        let deadline = *a.deadline_ms.get_or_insert(now_ms + grace);
        vec![note(
            app,
            hex(node),
            format!(
                "candidate {} ready: certifying until {}",
                build_name(&p.build),
                deadline
            ),
        )]
    }

    /// `node` left: a candidate gone before the group decided leaves its
    /// build uncertified, and the pending floor goes with it.
    pub fn left(&mut self, app: &str, node: u64) -> Vec<Action> {
        let a = self.app(app);
        if a.pending.as_ref().is_some_and(|p| p.node == node) {
            let p = a.pending.take().expect("checked");
            a.deadline_ms = None;
            return vec![note(
                app,
                hex(node),
                format!(
                    "candidate {} left before the group decided: no pending floor",
                    build_name(&p.build)
                ),
            )];
        }
        Vec::new()
    }

    /// The candidate a request may go to: certifying, this gateway's verdict
    /// not yet sent.
    pub fn canary(&self, app: &str) -> Option<u64> {
        self.apps
            .get(app)?
            .pending
            .as_ref()
            .filter(|p| p.state == CandidateState::Certifying)
            .map(|p| p.node)
    }

    /// Whether this pick goes to the candidate: the live share.
    pub fn take_share(&mut self, app: &str) -> bool {
        let pct = self.cfg.canary_pct;
        self.app(app)
            .pending
            .as_mut()
            .is_some_and(|p| p.share.take(pct))
    }

    pub fn record(&mut self, app: &str, sample: Sample) {
        if let Some(p) = self.app(app).pending.as_mut() {
            p.stats.record(sample);
        }
    }

    /// Whether a baseline request on a mirrored route is mirrored.
    pub fn should_mirror(&mut self, app: &str) -> bool {
        self.app(app)
            .pending
            .as_mut()
            .filter(|p| p.state == CandidateState::Certifying)
            .is_some_and(|p| p.stats.should_mirror())
    }

    pub fn mirrored(
        &mut self,
        app: &str,
        route: &str,
        mode: &Mirror,
        compared: &Compared,
        baseline: &[u8],
        candidate: &[u8],
    ) {
        if let Some(p) = self.app(app).pending.as_mut() {
            p.stats.mirrored(route, mode, compared, baseline, candidate);
        }
    }

    /// Judge `app`'s candidate now; `force` when the grace period is ended by
    /// hand (`raise-floor`).
    fn judge(&mut self, app: &str, now_ms: u64, force: bool) -> Vec<Action> {
        let a = self.app(app);
        let deadline = a.deadline_ms;
        let Some(p) = a
            .pending
            .as_mut()
            .filter(|p| p.state == CandidateState::Certifying)
        else {
            return Vec::new();
        };
        let figures = verdict(&p.stats.samples());
        let due = force || deadline.is_some_and(|d| now_ms >= d);
        let (result, state) = match figures.overall {
            Overall::Fail => (VerdictResult::Fail, CandidateState::Failed),
            _ if !due => return Vec::new(),
            Overall::Pass => (VerdictResult::Pass, CandidateState::Passed),
            Overall::PassWithGaps => (VerdictResult::Pass, CandidateState::PassedWithGaps),
            Overall::NoOpinion => (VerdictResult::NoOpinion, CandidateState::NoOpinion),
        };
        p.state = state;
        let text = format!(
            "certification of {}: {:?}{}",
            build_name(&p.build),
            figures.overall,
            figures
                .routes
                .iter()
                .filter(|r| r.outcome == verdict::Outcome::Fail)
                .map(|r| format!("; {}: {}", r.route, r.why))
                .collect::<String>()
        );
        let sent = Verdict {
            build: p.build.clone(),
            result,
            figures: serde_json::to_vec(&figures).expect("figures serialize"),
        };
        p.sent = Some((sent.clone(), now_ms));
        vec![
            Action::Verdict {
                app: app.to_string(),
                node: p.node,
                verdict: sent,
            },
            note(app, hex(p.node), text),
        ]
    }

    /// A verdict sent and not yet decided on goes again every
    /// [`VERDICT_AGAIN`].
    fn again(&mut self, app: &str, now_ms: u64) -> Vec<Action> {
        let every = VERDICT_AGAIN.as_millis() as u64;
        let Some(p) = self.app(app).pending.as_mut() else {
            return Vec::new();
        };
        let node = p.node;
        match &mut p.sent {
            Some((v, at)) if now_ms >= *at + every => {
                *at = now_ms;
                vec![Action::Verdict {
                    app: app.to_string(),
                    node,
                    verdict: v.clone(),
                }]
            }
            _ => Vec::new(),
        }
    }

    /// Once a second: a significant failure ends certification at once; the
    /// deadline ends it with this gateway's verdict, which goes again until the
    /// group decides.
    pub fn tick(&mut self, now_ms: u64) -> Vec<Action> {
        let apps: Vec<String> = self.apps.keys().cloned().collect();
        apps.iter()
            .flat_map(|app| {
                let mut out = self.judge(app, now_ms, false);
                out.extend(self.again(app, now_ms));
                out
            })
            .collect()
    }

    /// `raise-floor`: end the grace period now. The floor rises once the group
    /// certifies the build, or accepts it with `certify-anyway`.
    pub fn raise_floor(&mut self, app: &str, now_ms: u64) -> Result<Vec<Action>, String> {
        match &self.app(app).pending {
            None => return Err(format!("{app} has no pending floor")),
            Some(p) if p.state == CandidateState::Loading => {
                return Err(format!(
                    "{app}'s candidate {} is still loading: its certification has not begun",
                    build_name(&p.build)
                ))
            }
            Some(p) if p.state != CandidateState::Certifying => {
                return Err(format!(
                    "{app}'s candidate {} has its verdict: the group decides now",
                    build_name(&p.build)
                ))
            }
            Some(_) => {}
        }
        Ok(self.judge(app, now_ms, true))
    }

    /// `cancel-floor`: discard the pending floor; the candidate stays at its
    /// build, and nothing rises.
    pub fn cancel_floor(&mut self, app: &str) -> Result<Vec<Action>, String> {
        let a = self.app(app);
        let Some(p) = a.pending.take() else {
            return Err(format!("{app} has no pending floor"));
        };
        a.deadline_ms = None;
        Ok(vec![note(
            app,
            hex(p.node),
            format!("pending floor {} cancelled by hand", build_name(&p.build)),
        )])
    }

    /// A member's `COORD`: the group's verdicts and its latest forced
    /// rollout, which this gateway follows.
    pub fn coord(
        &mut self,
        app: &str,
        view: &UpgradeView,
        floor: Option<&BuildId>,
        now_ms: u64,
    ) -> Vec<Action> {
        let mut out = Vec::new();
        let a = self.app(app);
        for line in &view.verdicts {
            let Some((count, hash)) = parse_build(&line.build) else {
                continue;
            };
            let build = BuildId {
                count,
                hash,
                local: String::new(),
            };
            match line.verdict.result {
                Settled::Fail => {
                    a.certified.remove(&key(&build));
                    if a.rejected.insert(key(&build)) {
                        out.push(note(
                            app,
                            String::new(),
                            format!("build {} rejected: {}", line.build, line.verdict.reason),
                        ));
                    }
                    if a.pending.as_ref().is_some_and(|p| p.build == build) {
                        a.pending = None;
                        a.deadline_ms = None;
                    }
                }
                Settled::Pass => {
                    a.rejected.remove(&key(&build));
                    let newly = a.certified.insert(key(&build));
                    let pending = a.pending.as_ref().is_some_and(|p| p.build == build);
                    if newly && pending {
                        a.pending = None;
                        a.deadline_ms = None;
                        if floor.is_none_or(|f| f.count < build.count) {
                            a.raised_ms = Some(now_ms);
                            out.push(Action::Floor {
                                app: app.to_string(),
                                build: build.clone(),
                                by: format!("certified ({})", line.verdict.by),
                            });
                        }
                    }
                }
            }
        }
        if let Some(r) = view.rollout.as_ref().filter(|r| r.n > a.rollout_n) {
            a.rollout_n = r.n;
            if let Some((count, hash)) = parse_build(&r.decision.build) {
                let build = BuildId {
                    count,
                    hash,
                    local: String::new(),
                };
                if a.pending.as_ref().is_some_and(|p| p.build != build) {
                    a.pending = None;
                    a.deadline_ms = None;
                }
                a.rejected.remove(&key(&build));
                for rejected in &r.decision.rejects {
                    if let Some(b) = parse_build(rejected) {
                        a.rejected.insert(b);
                    }
                }
                a.raised_ms = Some(now_ms);
                out.push(Action::Floor {
                    app: app.to_string(),
                    build,
                    by: format!("rollout {} by {}", r.n, r.decision.by),
                });
            }
        }
        out
    }

    /// The members a rollout leaves behind (§3.9 step 5): once the floor has
    /// risen and [`ROLLOUT_SETTLE`] has passed, each member below it that is
    /// stranded, failed, refuses the build or is fetching nothing of it is
    /// closed 4410 with its reason. One admitted below the floor to upgrade
    /// that is stranded, or whose install of the floor's build failed and was
    /// restored, is closed at once. The rollout ends when none is left below.
    pub fn left_behind(
        &mut self,
        app: &str,
        floor: &BuildId,
        members: &[Below<'_>],
        now_ms: u64,
    ) -> Vec<Action> {
        let settle = ROLLOUT_SETTLE.as_millis() as u64;
        let a = self.app(app);
        let settled = a.raised_ms.is_some_and(|r| now_ms >= r + settle);
        let floor_name = build_name(floor);
        let mut out = Vec::new();
        let mut below = 0;
        for m in members {
            if m.build.is_local() || m.build.count >= floor.count {
                continue;
            }
            below += 1;
            let view = m.upgrade;
            let stranded = view.and_then(|v| v.stranded.clone());
            let reason = if let Some(s) = stranded {
                Some(format!("stranded: {s}"))
            } else if view.is_some_and(|v| v.refused.contains(&floor_name)) {
                Some(format!(
                    "an install of {floor_name} failed here and was restored"
                ))
            } else if view.is_some_and(|v| {
                v.fetching.is_none()
                    && v.step.is_empty()
                    && !v
                        .holds
                        .iter()
                        .any(|h| h.build == floor_name && h.source == "verified")
            }) {
                Some(format!(
                    "it holds no artifact of {floor_name} and fetches none"
                ))
            } else {
                None
            };
            let now = m.state == MemberState::Upgrading
                && view.is_some_and(|v| v.stranded.is_some() || v.refused.contains(&floor_name));
            if let Some(reason) = reason.filter(|_| settled || now) {
                out.push(Action::Close {
                    app: app.to_string(),
                    node: m.node,
                    close: Close::new(
                        CloseCode::Superseded,
                        format!(
                            "build {} is below the {app} floor {floor_name}, and {reason}; rebuild",
                            build_name(m.build)
                        ),
                    ),
                });
            }
        }
        if below == 0 && a.raised_ms.take().is_some() {
            out.push(note(
                app,
                String::new(),
                format!("rollout of {floor_name} ended"),
            ));
        }
        out
    }

    pub fn view(&self, app: &str) -> UpgradesView {
        let a = self.apps.get(app).cloned().unwrap_or_default();
        let names =
            |s: &BTreeSet<(u32, String)>| s.iter().map(|(c, h)| format!("{c}-{h}")).collect();
        UpgradesView {
            app: app.to_string(),
            candidate: a.pending.as_ref().map(|p| {
                let (eligible, mirrored) = p.stats.mirror_counts();
                CandidateView {
                    node: hex(p.node),
                    build: build_name(&p.build),
                    state: p.state,
                    ready_ms: p.ready_ms,
                    figures: verdict(&p.stats.samples()),
                    mirrored,
                    mirror_eligible: eligible,
                    mismatches: p.stats.mismatches.iter().cloned().collect(),
                }
            }),
            deadline_ms: a.deadline_ms,
            rollout_since_ms: a.raised_ms,
            rejected: names(&a.rejected),
            certified: names(&a.certified),
        }
    }

    /// Each app's pending floor, for the Apps block.
    pub fn pending(&self) -> BTreeMap<String, (BuildId, Option<u64>)> {
        self.apps
            .iter()
            .filter_map(|(app, a)| {
                a.pending
                    .as_ref()
                    .map(|p| (app.clone(), (p.build.clone(), a.deadline_ms)))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
