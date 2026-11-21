//! Which members of each app are admitted here, and where each one runs.
//!
//! A member is admitted under its app whether or not it is ready, and a request
//! is placed only on a ready one. While none is ready, a standing member of the
//! app keeps its loading page answering while the configured upstream does not.
//! Queries that decide a placement read the membership under one hold of the
//! hub, so a request already placed is not placed again, and the labels of a
//! set of nodes are read under one lock so each host and build count comes from
//! the same moment.

use super::*;

impl Hub {
    /// Name the hosts whose agents stand by (§11.3).
    pub fn set_standby_agents(&self, hosts: BTreeSet<String>) {
        self.state().registry.set_standby(hosts);
    }

    /// The machine table placement and the plan read from now on
    /// (`docs/static_placement.md` §2).
    pub fn set_machines(&self, machines: MachineTable) {
        self.state().registry.set_machines(machines);
    }

    pub fn clustered(&self, app: &str) -> bool {
        self.state().registry.clustered(app)
    }

    /// Whether any member of `app` is admitted here, ready or not.
    pub fn admitted(&self, app: &str) -> bool {
        !self.state().registry.members(app).is_empty()
    }

    pub fn member(&self, app: &str, node: u64) -> Option<MemberView> {
        self.state().registry.member(app, node).cloned()
    }

    /// Whether `app` has a member admitted here, and which stands by while none
    /// is ready — under one hold of the hub, never a table, so a request placed
    /// already is not placed again.
    pub fn standing(&self, app: &str) -> Standing {
        standing_of(&self.state().registry, app)
    }

    /// Whether `node` is an admitted member of `app` — without copying it.
    pub fn has_member(&self, app: &str, node: u64) -> bool {
        self.state().registry.member(app, node).is_some()
    }

    /// The host `node` of `app` runs on, when it is admitted.
    pub fn member_host(&self, app: &str, node: u64) -> Option<String> {
        self.state()
            .registry
            .member(app, node)
            .map(|m| m.host.clone())
    }

    /// Each of `nodes`' host and build count, in their order, under one lock —
    /// `("", 0)` for a node not admitted.
    pub fn member_labels(&self, app: &str, nodes: &[u64]) -> Vec<(String, u32)> {
        let s = self.state();
        nodes
            .iter()
            .map(|&n| {
                s.registry
                    .member(app, n)
                    .map_or((String::new(), 0), |m| (m.host.clone(), m.build.count))
            })
            .collect()
    }

    /// Every agent of `app` admitted here, by host and then node (§11.7).
    pub fn agents(&self, app: &str) -> Vec<AgentRow> {
        let s = self.state();
        let mut rows: Vec<AgentRow> = s
            .registry
            .members(app)
            .into_iter()
            .map(|m| {
                let connected = s
                    .sockets
                    .get(&(app.to_string(), m.node))
                    .map_or(0, |x| x.connected_ms);
                AgentRow::of(m, connected, s.registry.stands_by(&m.host))
            })
            .collect();
        rows.sort_by(|a, b| (&a.host, &a.node).cmp(&(&b.host, &b.node)));
        rows
    }

    /// A member of `app` admitted and not ready, whose loading page stands in
    /// while the app's configured upstream does not answer (§5.7).
    pub fn standing_by(&self, app: &str) -> Option<u64> {
        self.state().registry.standing_by(app)
    }

    /// A ready member of `app`: where a command the group records is sent.
    pub fn any_ready(&self, app: &str) -> Option<u64> {
        self.state()
            .registry
            .members(app)
            .iter()
            .find(|m| m.ready())
            .map(|m| m.node)
    }

    /// `app`'s floor, model lock and member count, for an aggregated
    /// `/v1/status`'s `group` block (§5.5).
    pub fn group(&self, app: &str) -> AppView {
        app_view(&self.state(), app)
    }

    /// Everything the status page shows (§12.2).
    pub fn status(&self, log_limit: usize) -> HubStatus {
        let now = now_ms();
        // The tie-break's lock is never taken under the state's.
        let tiebreak_leases = self
            .tiebreak
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|tb| tb.armed_leases(tb.now()))
            .unwrap_or_default();
        let s = self.state();
        let mut members = Vec::new();
        for app in APPS {
            for m in s.registry.members(app) {
                let socket = s.sockets.get(&(app.to_string(), m.node));
                let report = socket.map_or(SocketReport::default(), |x| SocketReport {
                    connected_ms: x.connected_ms,
                    clock_offset_ms: x.clock_offset_ms,
                    coord: x.coord.clone(),
                    queues: Queues {
                        live_bytes: x.handle.gauges().live(),
                        catchup_bytes: x.handle.gauges().catchup(),
                        skipped: x.skipped,
                        dropped: x.dropped,
                        pulses_dropped: x.pulses_dropped,
                    },
                });
                members.push(MemberRow::of(
                    m,
                    report,
                    s.last_close.get(&(app.to_string(), m.node)).cloned(),
                ));
            }
        }
        let faults = members.iter().filter_map(FaultRow::of).collect();
        let claims = APPS
            .iter()
            .flat_map(|app| {
                s.registry.claims_table(app).into_iter().map(|c| ClaimView {
                    app: app.to_string(),
                    key: String::from_utf8_lossy(&c.key).into_owned(),
                    node: hex(c.epoch.node),
                    epoch_ms: c.epoch.ms,
                    holder_here: s.registry.member(app, c.epoch.node).is_some(),
                })
            })
            .collect();
        HubStatus {
            build: crate::version::this_build(),
            gateway: self.name.clone(),
            now_ms: now,
            draining: s.draining,
            apps: APPS.iter().map(|a| app_view(&s, a)).collect(),
            faults,
            worlds: worlds(&members),
            disconnected: disconnected(&members),
            members,
            claims,
            refusals: s.refusals.iter().cloned().collect(),
            traffic: self.traffic.view(now),
            upgrades: APPS.iter().map(|a| s.upgrades.view(a)).collect(),
            builds: s.held.view(),
            tiebreak_leases,
            log: self.log.read(None, None, log_limit),
        }
    }
}

pub(super) fn app_view(s: &State, app: &str) -> AppView {
    let mode = s.modes.get(app).copied().unwrap_or_default();
    let quorum = s
        .sockets
        .iter()
        .filter(|((a, _), _)| a == app)
        .filter_map(|(_, x)| x.coord.as_ref())
        .map(|c| (c.paxos.max_members, c.paxos.quorum))
        .max();
    let pending = s.upgrades.pending().remove(app);
    AppView {
        pending_floor: pending.as_ref().map(|(b, _)| b.clone()),
        grace_ends_ms: pending.and_then(|(_, d)| d),
        max_members: quorum.map(|(m, _)| m),
        quorum: quorum.map(|(_, q)| q),
        app: app.to_string(),
        service: is_service(app),
        mode: mode_name(mode.clustered, s.registry.standing_by(app).is_some()),
        since_ms: mode.since_ms,
        floor: s.admission.floor(app).cloned(),
        floor_set_by: s
            .admission
            .floor_set_by(app)
            .map(|(n, h)| format!("{} on {h}", hex(n))),
        model_lock: s.admission.lock(app).map(|m| m.describe.clone()),
        members: s.registry.members(app).len(),
        ready: s.registry.members(app).iter().filter(|m| m.ready()).count(),
    }
}

pub(super) fn describe(how: &Ended) -> String {
    match how {
        Ended::Closed(c) => format!("closed by the member, {c}"),
        Ended::Refused(c) => format!("closed by this gateway, {c}"),
        Ended::Dropped => "the transport dropped".into(),
        Ended::Dead => "no frame for 45 s".into(),
        Ended::Stalled => "this gateway took no frame from it for 45 s".into(),
    }
}
