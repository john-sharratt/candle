//! The gateway's certification and rollout decisions for each app, which
//! say when a newer build becomes the app's floor.
//!
//! A member admitted above the app's floor is a candidate: a member below the
//! floor is offered the floor's build at once, and a candidate is judged once a
//! second by certification. Certification samples requests a baseline member
//! answered and mirrors the ones on mirrored routes to the candidate. Setting
//! the floor closes the agents of an app below the new build with `1001`, and
//! they return to take it. A forced rollout is planned from the members this
//! gateway sees, or refused with the reason.

use super::*;

impl Hub {
    /// What admission decided for the upgrades: a member below the floor is
    /// offered the floor's build at once; one above it is the candidate,
    /// unless its build is one the group rejected.
    pub(super) fn on_admitted(
        &self,
        app: &str,
        node: u64,
        hello: &Hello,
        admitted: &Admitted,
        handle: &SessionHandle,
    ) {
        // An npcd member lends by the plan from its first beat
        // (`docs/cooperative_admission.md` §4.4).
        if app == HOSTING_APP {
            if let Some((epoch, weights)) = self.state().planner.current() {
                let _ = handle.control(Frame::Plan(PlanFrame { epoch, weights }));
            }
        }
        let build = &hello.build;
        if admitted.upgrading {
            let floor = admitted.floor.clone().expect("upgrading is below a floor");
            let offer = self.state().held.offer(&floor);
            self.note(
                app,
                hex(node),
                if admitted.behind {
                    format!(
                        "admitted to upgrade: protocol {} is behind {PROTO}, build {} takes {}",
                        hello.proto, build.count, floor.count
                    )
                } else {
                    format!(
                        "admitted to upgrade: build {} is below the floor {}",
                        build.count, floor.count
                    )
                },
            );
            if let Some(offer) = offer {
                let _ = handle.control(Frame::BuildOffer(offer));
            }
            return;
        }
        if hello.rejected {
            self.note(
                app,
                hex(node),
                format!(
                    "admitted on {}, a build the group rejected",
                    build_name(build)
                ),
            );
            return;
        }
        let actions =
            self.state()
                .upgrades
                .admitted(app, node, build, admitted.floor.as_ref(), now_ms());
        self.act(actions);
    }

    /// Carry out what the upgrades decided.
    pub(super) fn act(&self, actions: Vec<Action>) {
        for a in actions {
            match a {
                Action::Verdict { app, node, verdict } => {
                    self.to_member(&app, node, Frame::Verdict(verdict))
                }
                Action::Floor { app, build, by } => self.set_floor(&app, &build, &by),
                Action::Close { app, node, close } => {
                    if let Some(s) = self.state().sockets.get(&(app.clone(), node)) {
                        s.handle.close(close.clone());
                    }
                    self.note(
                        &app,
                        hex(node),
                        format!("closed {}: {}", close.code, close.reason),
                    );
                }
                Action::Note { app, subject, text } => self.note(&app, subject, text),
            }
        }
    }

    /// `app`'s floor is `build` now. An app's agents follow its build when
    /// the build carries a `tokera-agent`: those below it are closed `1001`
    /// and come back to take it (§3.9).
    pub(super) fn set_floor(&self, app: &str, build: &BuildId, by: &str) {
        let agents_follow = {
            let mut s = self.state();
            s.admission.set_floor(app, build, 0, by);
            rolls_out(app) && s.held.has_kind(build, AGENT)
        };
        self.note(
            app,
            String::new(),
            format!("floor {} ({}) by {by}", build.count, build.hash),
        );
        if !agents_follow {
            return;
        }
        let below: Vec<SessionHandle> = {
            let mut s = self.state();
            s.admission
                .set_floor(AGENT_APP, build, 0, &format!("following {app}"));
            s.registry
                .members(AGENT_APP)
                .iter()
                .filter(|m| !m.build.is_local() && m.build.count < build.count)
                .filter_map(|m| s.sockets.get(&(AGENT_APP.to_string(), m.node)))
                .map(|x| x.handle.clone())
                .collect()
        };
        for h in below {
            h.close(Close::new(
                CloseCode::GoingAway,
                format!(
                    "the floor rose to {} ({}): reconnect to take it",
                    build.count, build.hash
                ),
            ));
        }
    }

    /// Once a second: certification's verdicts, and the members a rollout
    /// leaves behind (§3.8, §3.9).
    pub fn upgrade_tick(&self) {
        let now = now_ms();
        let mut actions = self.state().upgrades.tick(now);
        for app in APPS {
            if !rolls_out(app) {
                continue;
            }
            let mut s = self.state();
            let Some(floor) = s.admission.floor(app).cloned() else {
                continue;
            };
            let members: Vec<(u64, BuildId, MemberState, Option<UpgradeView>)> = s
                .registry
                .members(app)
                .iter()
                .map(|m| {
                    let view = s
                        .sockets
                        .get(&(app.to_string(), m.node))
                        .and_then(|x| x.coord.as_ref())
                        .and_then(|c| c.upgrade.clone());
                    (m.node, m.build.clone(), m.state(), view)
                })
                .collect();
            let below: Vec<Below<'_>> = members
                .iter()
                .map(|(node, build, state, view)| Below {
                    node: *node,
                    build,
                    state: *state,
                    upgrade: view.as_ref(),
                })
                .collect();
            actions.extend(s.upgrades.left_behind(app, &floor, &below, now));
        }
        self.act(actions);
    }

    /// Set how this gateway certifies: the grace period and the live share.
    pub fn set_upgrade_config(&self, cfg: UpgradeConfig) {
        self.state().upgrades = Upgrades::new(cfg);
    }

    /// `raise-floor` (§12.4): end `app`'s grace period now.
    pub fn raise_floor(&self, app: &str) -> Result<(), String> {
        if !rolls_out(app) {
            return Err(format!("{app} is listed, not rolled out"));
        }
        let actions = self.state().upgrades.raise_floor(app, now_ms())?;
        self.note(
            app,
            String::new(),
            "raise-floor: the grace period ends now".into(),
        );
        self.act(actions);
        Ok(())
    }

    /// A forced rollout's plan (§3.11), from what this gateway sees of the
    /// app's members; or why it is refused.
    pub fn plan_rollout(&self, ask: &Ask) -> Result<Plan, String> {
        let app = ask.app.as_str();
        if !rolls_out(app) {
            return Err(format!("{app} is listed, not rolled out"));
        }
        let (owned, floor, quorum, mut seen) = {
            let s = self.state();
            let owned: Vec<(u64, String, BuildId, bool, Option<UpgradeView>)> = s
                .registry
                .members(app)
                .iter()
                .map(|m| {
                    let view = s
                        .sockets
                        .get(&(app.to_string(), m.node))
                        .and_then(|x| x.coord.as_ref())
                        .and_then(|c| c.upgrade.clone());
                    (m.node, m.host.clone(), m.build.clone(), m.ready(), view)
                })
                .collect();
            let quorum = app_view(&s, app).quorum.unwrap_or(1);
            let seen: Vec<BuildId> = s
                .upgrades
                .pending()
                .get(app)
                .map(|(b, _)| b.clone())
                .into_iter()
                .collect();
            (owned, s.admission.floor(app).cloned(), quorum, seen)
        };
        seen.extend(owned.iter().map(|(_, _, b, _, _)| b.clone()));
        let members: Vec<PlanMember<'_>> = owned
            .iter()
            .map(|(node, host, build, ready, view)| PlanMember {
                node: *node,
                host,
                build,
                ready: *ready,
                upgrade: view.as_ref(),
            })
            .collect();
        let tiebreak = self
            .tiebreak
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        plan(
            ask,
            &members,
            floor.as_ref(),
            quorum,
            tiebreak,
            &seen,
            now_ms(),
        )
    }

    /// `cancel-floor` (§12.4): discard `app`'s pending floor.
    pub fn cancel_floor(&self, app: &str) -> Result<(), String> {
        if !rolls_out(app) {
            return Err(format!("{app} is listed, not rolled out"));
        }
        let actions = self.state().upgrades.cancel_floor(app)?;
        self.act(actions);
        Ok(())
    }

    /// A request `node` answered for `app` on `route`, for certification.
    pub fn certify_sample(
        &self,
        app: &str,
        route: &str,
        mirror: &Mirror,
        node: u64,
        failed: bool,
        latency_ms: u32,
    ) {
        let mut s = self.state();
        let Some(candidate) = s.upgrades.canary(app) else {
            return;
        };
        s.upgrades.record(
            app,
            Sample {
                route: route.to_string(),
                mirror: mirror.clone(),
                candidate: node == candidate,
                failed,
                latency_ms,
            },
        );
    }

    /// Whether a request the baseline `node` serves on a mirrored route is
    /// mirrored to the candidate, and to which.
    pub fn mirror_to(&self, app: &str, node: u64) -> Option<u64> {
        let mut s = self.state();
        let candidate = s.upgrades.canary(app).filter(|c| *c != node)?;
        s.upgrades.should_mirror(app).then_some(candidate)
    }

    /// A mirrored call compared.
    pub fn mirrored(
        &self,
        app: &str,
        route: &str,
        mode: &Mirror,
        compared: &Compared,
        baseline: &[u8],
        candidate: &[u8],
    ) {
        self.state()
            .upgrades
            .mirrored(app, route, mode, compared, baseline, candidate);
    }
}
