//! A terminal's acts, enacted where the world runs
//! (`docs/zen_everywhere.md` §5): awaited, because every one of them reaches
//! something real — an agent on the house network, zend's MCP, a machine.
//!
//! | Terminal | Reached through |
//! |---|---|
//! | the lights, a device | a [`BridgeCall`] to an in-network agent, as a room's device acts go (`crate::bridge`) |
//! | the house network | this member's own survey of it, and a `Wake` through an agent |
//! | the cluster | zend's `/v1/mcp`, as Zen's service (`super::zen_task::call_zend`) — read-only tools alone |
//! | a machine | a `Check` sent to that machine's own agent |
//!
//! **What a terminal raises** is told to every mind of the world that takes
//! no orders ([`Runtime::raise_terminal`]); the one source of it here is the
//! house network's watch on handsets ([`run`]).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use agent::client::Agents;
use agent::wire::BridgeCall;
use effector_bridge::work::profile_act;
use futures::future::join_all;
use serde_json::{Map, Value};
use tokio::time::{interval, MissedTickBehavior};

use super::zen_task::call_zend;
use crate::bridge::finding::{ANDROIDTV, WEBOS};
use crate::bridge::{discover, PAIRING_AGENT};
use crate::engine::act::Act;
use crate::engine::event::{EventKind, Salience};
use crate::engine::runtime::{Recorded, Runtime, LANDED, REFUSED};
use crate::engine::terminal::{
    self, check_args, tv_action, watch, welcome, Kind, Machine, Terminal, ALL_LIGHTS, CLUSTER,
    NETWORK,
};
use crate::sim::terminals::Device;
use crate::world::Hosted;

/// How often the house network is surveyed for the handsets that are watched.
pub const WATCH_EVERY: Duration = Duration::from_secs(300);

/// How many presence surveys pass between two readings of the cluster and the
/// machines: every third, so a quarter of an hour.
const WATCH_EVERY_TICKS: u32 = 3;

/// The machine check the watch runs: the one a machine's own list calls its
/// health.
const WATCH_CHECK: &str = "health";

/// The session zend keeps a mind's cluster reads under.
const WATCH_THREAD: &str = "zen-watch";

/// The most of an answer from the cluster a mind is handed.
const ANSWER_CHARS: usize = 6_000;

fn recorded(act: &Act, landed: bool, answer: String) -> Recorded {
    let first = answer.lines().next().unwrap_or_default().to_string();
    Recorded {
        feed: match landed {
            true => format!("{} {LANDED} {first}", act.summary()),
            false => format!("{} {REFUSED} {first}", act.summary()),
        },
        answer,
        landed,
        departing: false,
    }
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// `said`, cut at [`ANSWER_CHARS`] on a character.
fn cut(said: &str) -> String {
    let said = said.trim();
    match said.char_indices().nth(ANSWER_CHARS) {
        Some((at, _)) => format!("{}\n… (cut: there is more)", &said[..at]),
        None => said.to_string(),
    }
}

/// A device's own arguments with what the world keeps for it folded in: the
/// credentials a pairing earned and its hardware address, never a key that is
/// the world's own bookkeeping. An argument the caller set wins.
fn with_device(args: &Map<String, Value>, device: &Device) -> Map<String, Value> {
    let mut merged = args.clone();
    for (name, value) in &device.creds {
        if name.starts_with('_') {
            continue;
        }
        merged
            .entry(name.clone())
            .or_insert_with(|| Value::from(value.clone()));
    }
    if !device.finding.hardware.is_empty() {
        merged
            .entry("mac".to_string())
            .or_insert_with(|| Value::from(device.finding.hardware.clone()));
    }
    merged
}

/// Hand `call` to an in-network agent — to agent `to` when it is one
/// machine's — and read back whether it was done and the line it gave.
async fn bridged(agents: Option<&Agents>, call: BridgeCall, to: Option<u64>) -> (bool, String) {
    let Some(agents) = agents else {
        return (
            false,
            "No agent in the house can be reached right now. Try again in a moment.".into(),
        );
    };
    match agents.bridge_on(&call, to).await {
        Ok((b, _)) => (b.ok, b.line),
        Err(e) => (false, format!("It could not be reached: {e}")),
    }
}

impl Runtime {
    /// Whether `npc` is a mind with no body: one that takes no orders.
    pub(crate) fn is_bodiless(&self, npc: u64) -> bool {
        self.body_of(npc)
            .is_some_and(|(hosted, body)| hosted.sim(|s| s.missions.is_unordered(&body)))
    }

    /// Ask the gateway which machines the house's agents run on, and keep
    /// the answer. Nothing changes when they cannot be asked.
    async fn refresh_machines(&self) {
        let Some(agents) = self.agents() else {
            return;
        };
        let rows = match agents.listing().await {
            Ok(rows) => rows,
            Err(e) => {
                tracing::warn!("the house's machines could not be listed: {e}");
                return;
            }
        };
        let mut machines: Vec<Machine> = rows
            .into_iter()
            .filter_map(|r| {
                let node = u64::from_str_radix(&r.node, 16).ok()?;
                (!r.host.trim().is_empty()).then(|| Machine {
                    host: r.host.trim().to_string(),
                    node,
                    ready: r.state == "ready" && !r.standby,
                    checks: Vec::new(),
                })
            })
            .collect();
        // One terminal a machine: a machine with two agents is its first.
        machines.sort_by_key(|m| (m.host.to_lowercase(), m.node));
        machines.dedup_by(|b, a| a.host.eq_ignore_ascii_case(&b.host));
        self.set_machines(machines);
    }

    /// The terminal `body` is connected to in `hosted`, if it still stands.
    fn connection(&self, hosted: &Hosted, body: &str) -> Option<Terminal> {
        let machines = self.machines();
        let on_cluster = self.standing.member().is_some();
        hosted.sim(|s| terminal::connected(s, &machines, on_cluster, body))
    }

    /// An act made through a terminal for `npc`: one of the terminal acts, or
    /// one of the house's device acts on what it is connected to.
    pub(crate) async fn terminal_act(&self, npc: u64, act: &Act) -> Recorded {
        if self.scheduler.is_suspended(npc) {
            return recorded(
                act,
                false,
                "You are held still: nothing you do reaches anyone.".into(),
            );
        }
        let Some((hosted, body)) = self.body_of(npc) else {
            return recorded(
                act,
                false,
                "You have no place in a world to act from.".into(),
            );
        };
        if !hosted.sim(|s| s.missions.is_unordered(&body)) {
            return recorded(
                act,
                false,
                "Terminals are not yours to connect to: you work what stands in the room with \
                 you."
                    .into(),
            );
        }
        let (landed, answer) = match act.tool {
            "terminal_list" => self.list_terminals(&hosted).await,
            "terminal_connect" => self.connect(&hosted, &body, &act.args).await,
            "terminal_disconnect" => match hosted.with_sim(|s| s.terminals.disconnect(&body)) {
                Some(was) => (true, format!("You let go of {was}.")),
                None => (false, "You are connected to nothing.".into()),
            },
            // The house network is surveyed from wherever the mind is.
            "survey_network" => self.survey_house(&hosted).await,
            _ => match self.connection(&hosted, &body) {
                None => (
                    false,
                    "You are connected to nothing: `terminal_list` says what there is, and \
                     `terminal_connect` takes you to one."
                        .into(),
                ),
                Some(terminal) => {
                    let (landed, answer) = self.work(&hosted, &terminal, act).await;
                    if landed {
                        hosted.with_sim(|s| s.missions.worked_terminal(&body, &terminal.name));
                    }
                    (landed, answer)
                }
            },
        };
        if landed {
            hosted.with_sim(|s| {
                s.missions
                    .observe(&body, answer.lines().next().unwrap_or(""))
            });
        }
        recorded(act, landed, answer)
    }

    async fn list_terminals(&self, hosted: &Hosted) -> (bool, String) {
        self.refresh_machines().await;
        let machines = self.machines();
        let on_cluster = self.standing.member().is_some();
        let lines: Vec<String> = hosted.sim(|s| {
            terminal::listed(s, &machines, on_cluster)
                .iter()
                .map(|t| format!("- {}", t.line(s)))
                .collect()
        });
        (
            true,
            format!(
                "The terminals you can reach, each by the name `terminal_connect` takes:\n{}",
                lines.join("\n")
            ),
        )
    }

    async fn connect(
        &self,
        hosted: &Hosted,
        body: &str,
        args: &Map<String, Value>,
    ) -> (bool, String) {
        let Some(name) = text(args, "terminal") else {
            return (false, "Say which terminal to connect to.".into());
        };
        let on_cluster = self.standing.member().is_some();
        let find = || {
            let machines = self.machines();
            hosted.sim(|s| terminal::named(s, &machines, on_cluster, &name))
        };
        let Some(mut terminal) = find() else {
            return (
                false,
                format!("There is no terminal called {name}: `terminal_list` says what there is."),
            );
        };
        // A machine says what its checks are as it is connected to: the list
        // is the machine's own, and may have been added to since.
        if let Kind::Machine(m) = &terminal.kind {
            let (ok, why, checks) = match self.agents() {
                None => (false, "no agent can be reached".to_string(), Vec::new()),
                Some(agents) => match agents.bridge_on(&BridgeCall::Checks, Some(m.node)).await {
                    Ok((b, _)) => (b.ok, b.line, b.checks),
                    Err(e) => (false, e.to_string(), Vec::new()),
                },
            };
            if !ok {
                return (false, format!("{} does not answer: {why}", m.host));
            }
            let mut machines = self.machines();
            if let Some(kept) = machines.iter_mut().find(|k| k.node == m.node) {
                kept.checks = checks;
            }
            self.set_machines(machines);
            match find() {
                Some(again) => terminal = again,
                None => return (false, format!("{name} is gone.")),
            }
        }
        hosted.with_sim(|s| s.terminals.connect(body, &terminal.name));
        (true, hosted.sim(|s| welcome(&terminal, s)))
    }

    /// Survey the house network for a mind with no body: every device on it
    /// now, each kept as a terminal of `hosted`'s world.
    async fn survey_house(&self, hosted: &Hosted) -> (bool, String) {
        let found = match discover::survey().await {
            Err(e) => return (false, format!("The survey could not be made: {e}")),
            Ok(found) => found,
        };
        let lines: Vec<String> = found.iter().map(|f| f.line()).collect();
        let n = found.len();
        let changes = hosted.with_sim(|s| s.terminals.surveyed(found));
        tracing::info!(
            world = hosted.id(),
            devices = n,
            "house network surveyed by a mind with no body"
        );
        for change in changes {
            self.raise_terminal(hosted, NETWORK, &change.line(), true);
        }
        match n {
            0 => (
                true,
                "The survey found nothing on the house network.".into(),
            ),
            _ => (
                true,
                format!(
                    "The survey found {n} devices on the house network:\n{}",
                    lines.join("\n")
                ),
            ),
        }
    }

    /// One act on the terminal a mind is connected to.
    async fn work(&self, hosted: &Hosted, terminal: &Terminal, act: &Act) -> (bool, String) {
        let agents = self.agents();
        let not_here = || {
            (
                false,
                format!(
                    "{} has no `{}`: you are connected to it, and that belongs to another \
                     terminal.",
                    terminal.name, act.tool
                ),
            )
        };
        match (&terminal.kind, act.tool) {
            // ── the lights ────────────────────────────────────────────────
            (Kind::Lights, tool) if profile_act(tool).is_some_and(|(p, _)| p == "wiz") => {
                let (_, action) = profile_act(tool).expect("just checked");
                let Some(which) = text(&act.args, "device") else {
                    return (false, "Say which light, or all of them.".into());
                };
                let bulbs: Vec<Device> = hosted.sim(|s| {
                    s.terminals
                        .lights()
                        .into_iter()
                        .filter(|b| {
                            which.eq_ignore_ascii_case(ALL_LIGHTS)
                                || b.name.eq_ignore_ascii_case(&which)
                        })
                        .cloned()
                        .collect()
                });
                if bulbs.is_empty() {
                    return (false, format!("There is no light called {which}."));
                }
                // Every bulb at once: turning the house off is one act, and
                // eight bulbs asked one after another is eight waits.
                let asked = bulbs.iter().map(|b| {
                    bridged(
                        agents.as_ref(),
                        BridgeCall::Work {
                            profile: "wiz".to_string(),
                            name: b.name.clone(),
                            address: b.finding.address.clone(),
                            action: action.to_string(),
                            args: act.args.clone(),
                        },
                        None,
                    )
                });
                let answers = join_all(asked).await;
                let did = answers.iter().filter(|(ok, _)| *ok).count();
                let lines: Vec<String> = bulbs
                    .iter()
                    .zip(&answers)
                    .map(|(b, (ok, line))| match ok {
                        true => line.clone(),
                        false => format!("The {} did not take it: {line}", b.name),
                    })
                    .collect();
                match bulbs.len() {
                    1 => (did == 1, lines.join("\n")),
                    n => (
                        did > 0,
                        format!("{did} of {n} lights took it.\n{}", lines.join("\n")),
                    ),
                }
            }
            // ── the house network ─────────────────────────────────────────
            (Kind::Network, "wake_device") => {
                let Some(name) = text(&act.args, "device") else {
                    return (false, "Say which device to wake.".into());
                };
                let mac = hosted.sim(|s| {
                    s.terminals
                        .named(&name)
                        .map(|d| d.finding.hardware.clone())
                        .filter(|m| !m.is_empty())
                });
                match mac {
                    None => (
                        false,
                        format!("There is no device called {name} the house knows how to wake."),
                    ),
                    Some(mac) => {
                        bridged(agents.as_ref(), BridgeCall::Wake { name, mac }, None).await
                    }
                }
            }
            (Kind::Network, "device_rename") => {
                let (Some(device), Some(name)) =
                    (text(&act.args, "device"), text(&act.args, "called"))
                else {
                    return (
                        false,
                        "Say which device, and what it is to be called.".into(),
                    );
                };
                match hosted.with_sim(|s| s.terminals.rename(&device, &name)) {
                    Ok(()) => (true, format!("{device} is called {name} from now on.")),
                    Err(why) => (false, why),
                }
            }
            (Kind::Network, "presence_watch") => {
                let (Some(device), Some(who)) = (text(&act.args, "device"), text(&act.args, "who"))
                else {
                    return (false, "Say which handset, and whose it is.".into());
                };
                match hosted.with_sim(|s| s.terminals.watch(&device, Some(&who))) {
                    true => (
                        true,
                        format!(
                            "{device} is watched as {who}'s: you will be told when it joins the \
                             house network and when it has left."
                        ),
                    ),
                    false => (false, format!("There is no device called {device}.")),
                }
            }
            (Kind::Network, "presence_forget") => {
                let Some(device) = text(&act.args, "device") else {
                    return (false, "Say which handset.".into());
                };
                match hosted.with_sim(|s| s.terminals.watch(&device, None)) {
                    true => (true, format!("{device} is no longer watched.")),
                    false => (false, format!("There is no device called {device}.")),
                }
            }
            // ── the cluster: how it stands, its logs, its jobs ────────────
            (Kind::Cluster, tool) if terminal::acts::CLUSTER_ACTS.contains(&tool) => {
                match terminal::cluster_call(tool, &act.args) {
                    Ok((zend, call)) => self.ask_cluster(zend, call).await,
                    Err(why) => (false, why),
                }
            }
            // ── a machine: its checks ─────────────────────────────────────
            (Kind::Machine(m), "machine_check") => {
                let Some(name) = text(&act.args, "check") else {
                    return (false, "Say which check to run.".into());
                };
                bridged(
                    agents.as_ref(),
                    BridgeCall::Check {
                        name,
                        args: check_args(&act.args),
                    },
                    Some(m.node),
                )
                .await
            }
            // ── a device ──────────────────────────────────────────────────
            (Kind::Device(d), "pair_device" | "pair_code") => {
                self.pair(hosted, d, act, agents.as_ref()).await
            }
            (Kind::Device(d), tool) if tool.starts_with("tv_") => {
                let profile = d.finding.profile.as_str();
                if !matches!(profile, WEBOS | ANDROIDTV) {
                    return not_here();
                }
                if terminal::needs_pairing(d) {
                    return (
                        false,
                        format!(
                            "The {} is not paired yet: begin with `pair_device`.",
                            d.name
                        ),
                    );
                }
                match tv_action(profile, tool, &act.args) {
                    Err(why) => (false, why),
                    Ok((action, args)) => {
                        bridged(
                            agents.as_ref(),
                            BridgeCall::Work {
                                profile: profile.to_string(),
                                name: d.name.clone(),
                                address: d.finding.address.clone(),
                                action: action.to_string(),
                                args: with_device(&args, d),
                            },
                            None,
                        )
                        .await
                    }
                }
            }
            (Kind::Device(d), tool)
                if profile_act(tool).is_some_and(|(p, _)| p == d.finding.profile) =>
            {
                let (profile, action) = profile_act(tool).expect("just checked");
                bridged(
                    agents.as_ref(),
                    BridgeCall::Work {
                        profile: profile.to_string(),
                        name: d.name.clone(),
                        address: d.finding.address.clone(),
                        action: action.to_string(),
                        args: with_device(&act.args, d),
                    },
                    None,
                )
                .await
            }
            _ => not_here(),
        }
    }

    /// One tool of zend's, called as Zen's service — one of those
    /// [`terminal::cluster_call`] names.
    async fn ask_cluster(&self, tool: &str, args: Value) -> (bool, String) {
        let Some(member) = self.standing.member().cloned() else {
            return (
                false,
                "The cluster cannot be reached from here: this house is on none.".into(),
            );
        };
        match call_zend(&member, tool, &args, WATCH_THREAD).await {
            Ok(said) => (true, cut(&said)),
            Err(why) => (false, format!("The cluster did not answer: {}", cut(&why))),
        }
    }

    /// Pair with the television `device`, keeping what it earns with the
    /// device. An LG pairs in one call, once somebody accepts on its screen;
    /// a Google television takes two — the first shows a code and notes which
    /// agent holds the session, the second finishes on that same agent.
    async fn pair(
        &self,
        hosted: &Hosted,
        device: &Device,
        act: &Act,
        agents: Option<&Agents>,
    ) -> (bool, String) {
        let Some(agents) = agents else {
            return (
                false,
                "No agent in the house can pair it right now. Try again in a moment.".into(),
            );
        };
        let profile = device.finding.profile.as_str();
        if !matches!(profile, WEBOS | ANDROIDTV) {
            return (
                false,
                format!("The {} is not a television to pair.", device.name),
            );
        }
        let code = match act.tool {
            "pair_code" => match (profile, text(&act.args, "code")) {
                (ANDROIDTV, Some(code)) => Some(code),
                (ANDROIDTV, None) => {
                    return (
                        false,
                        "Say the six-digit code the television is showing.".into(),
                    )
                }
                _ => {
                    return (
                        false,
                        format!(
                            "The {} pairs in one step with `pair_device` — it needs no code.",
                            device.name
                        ),
                    )
                }
            },
            _ => None,
        };
        let to = code
            .as_ref()
            .and(device.creds.get(PAIRING_AGENT))
            .and_then(|a| a.parse::<u64>().ok());
        let call = BridgeCall::Pair {
            profile: profile.to_string(),
            address: device.finding.address.clone(),
            code,
        };
        let (b, served_by) = match agents.bridge_on(&call, to).await {
            Ok(pair) => pair,
            Err(e) => {
                return (
                    false,
                    format!("The {} could not be reached: {e}", device.name),
                )
            }
        };
        if !b.ok {
            return (false, b.line);
        }
        let key = device
            .finding
            .hardware
            .to_ascii_lowercase()
            .replace('-', ":");
        if b.creds.is_empty() {
            let set: BTreeMap<String, String> = served_by
                .map(|a| BTreeMap::from([(PAIRING_AGENT.to_string(), a.to_string())]))
                .unwrap_or_default();
            hosted.with_sim(|s| s.terminals.keep_creds(&key, &set, &[]));
            return (
                true,
                "The television is showing a six-digit code. Ask somebody there to read it back, \
                 then finish with `pair_code`."
                    .into(),
            );
        }
        hosted.with_sim(|s| {
            s.terminals
                .keep_creds(&key, &b.creds, &[PAIRING_AGENT.to_string()])
        });
        (true, format!("{} Its acts are yours now.", b.line))
    }

    /// Tell every mind of `hosted` that takes no orders what `terminal`
    /// raised: `text`, urgently when the adapter says so.
    pub(crate) fn raise_terminal(&self, hosted: &Hosted, terminal: &str, text: &str, urgent: bool) {
        let salience = match urgent {
            true => Salience::URGENT,
            false => Salience::NORMAL,
        };
        for (npc, body) in self.bodies.in_world(hosted.id()) {
            if !hosted.sim(|s| s.missions.is_unordered(&body)) {
                continue;
            }
            self.tell(
                npc,
                self.world_ms(npc),
                salience,
                EventKind::Terminal {
                    terminal: terminal.to_string(),
                    text: text.to_string(),
                },
            );
        }
        tracing::info!(world = hosted.id(), terminal, "a terminal raised: {text}");
    }

    /// Note, in every world this member writes, the terminals a mind with no
    /// body can reach now — what its view of the world states
    /// (`engine::whole`), so it never has to ask what there is. The machines
    /// are asked after first: an agent that joined since is a terminal.
    pub async fn note_terminals(&self) {
        self.refresh_machines().await;
        let machines = self.machines();
        let on_cluster = self.standing.member().is_some();
        for hosted in self
            .hosted
            .ids()
            .into_iter()
            .filter(|id| self.standing.writes(id))
            .filter_map(|id| self.hosted.get(&id))
        {
            let names: Vec<String> = hosted.sim(|s| {
                terminal::listed(s, &machines, on_cluster)
                    .into_iter()
                    .map(|t| t.name)
                    .collect()
            });
            if hosted.sim(|s| s.terminals.reachable() != names.as_slice()) {
                hosted.with_sim(|s| s.terminals.set_reachable(names));
            }
        }
    }

    /// Keep the watch for every world this member writes that holds a mind
    /// with no body (`engine::terminal::watch`): read the cluster's standing
    /// and run each machine's `health` check, once for them all, and raise
    /// the terminal in each world only for what changed. A reading that
    /// could not be taken raises nothing: the next one will say.
    pub async fn tend_watch(&self) {
        let watched: Vec<Arc<Hosted>> = self
            .hosted
            .ids()
            .into_iter()
            .filter(|id| self.standing.writes(id))
            .filter_map(|id| self.hosted.get(&id))
            .filter(|h| self.has_bodiless_mind(h))
            .collect();
        if watched.is_empty() {
            return;
        }
        if self.standing.member().is_some() {
            if let (true, status) = self
                .ask_cluster("cluster_status", Value::Object(Map::new()))
                .await
            {
                let now = watch::standing(&status);
                for hosted in &watched {
                    let before = hosted.with_sim(|s| s.terminals.read_cluster(now.clone()));
                    if let Some(news) = before.and_then(|b| watch::cluster_news(&b, &now)) {
                        self.raise_terminal(hosted, CLUSTER, &news, false);
                    }
                }
            }
        }
        let agents = self.agents();
        for machine in self.machines().into_iter().filter(|m| m.ready) {
            let call = BridgeCall::Check {
                name: WATCH_CHECK.to_string(),
                args: BTreeMap::new(),
            };
            let (true, report) = bridged(agents.as_ref(), call, Some(machine.node)).await else {
                continue;
            };
            if let Some(news) = watch::machine_news(&report) {
                for hosted in &watched {
                    self.raise_terminal(hosted, &machine.host, news, false);
                }
            }
        }
    }

    /// Whether a mind with no body lives in `hosted`: the one whose terminals
    /// the house's devices are.
    fn has_bodiless_mind(&self, hosted: &Hosted) -> bool {
        self.bodies
            .in_world(hosted.id())
            .iter()
            .any(|(_, body)| hosted.sim(|s| s.missions.is_unordered(body)))
    }

    /// Survey the house network for every world this member writes that
    /// watches a handset or holds a mind with no body, and raise what changed:
    /// who came home, who went out. One survey serves them all — it is one
    /// house. For the mind it is also what puts the house's lights, speakers
    /// and televisions among its terminals: a device nobody has surveyed is a
    /// terminal it does not have, and it was left to the mind to think of
    /// surveying.
    pub async fn tend_presence(&self) {
        let watching: Vec<Arc<Hosted>> = self
            .hosted
            .ids()
            .into_iter()
            .filter(|id| self.standing.writes(id))
            .filter_map(|id| self.hosted.get(&id))
            .filter(|h| h.sim(|s| s.terminals.watches()) || self.has_bodiless_mind(h))
            .collect();
        if watching.is_empty() {
            return;
        }
        let found = match discover::survey().await {
            Ok(found) => found,
            Err(e) => {
                tracing::debug!("the house network could not be surveyed for its handsets: {e}");
                return;
            }
        };
        for hosted in watching {
            // Looked at again: the world may have been handed on while the
            // survey listened.
            if !self.standing.writes(hosted.id()) {
                continue;
            }
            for change in hosted.with_sim(|s| s.terminals.surveyed(found.clone())) {
                self.raise_terminal(&hosted, NETWORK, &change.line(), true);
            }
        }
    }
}

/// Watch the house network's handsets until the process stops.
pub async fn run(rt: Arc<Runtime>) {
    let mut tick = interval(WATCH_EVERY);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut ticks: u32 = 0;
    loop {
        tick.tick().await;
        if rt.stopping() {
            return;
        }
        rt.note_terminals().await;
        // The machines and the cluster change by the hour, not the minute.
        if ticks.is_multiple_of(WATCH_EVERY_TICKS) {
            rt.tend_watch().await;
        }
        ticks = ticks.wrapping_add(1);
        rt.tend_presence().await;
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::bridge::finding::Finding;

    /// **What the world keeps for a device goes with every call to it** — its
    /// key and its hardware address — and the world's own bookkeeping never
    /// does; an argument the caller set wins.
    #[test]
    fn a_devices_keys_and_address_fold_into_the_call() {
        let device = Device {
            finding: Finding {
                address: "192.168.1.60".into(),
                hardware: "14:ae:85:71:eb:44".into(),
                profile: WEBOS.into(),
                ..Finding::default()
            },
            name: "the den TV".into(),
            creds: BTreeMap::from([
                ("key".to_string(), "K".to_string()),
                (PAIRING_AGENT.to_string(), "7".to_string()),
            ]),
            ..Device::default()
        };
        let args: Map<String, Value> = json!({"volume": 12}).as_object().unwrap().clone();
        let merged = with_device(&args, &device);
        assert_eq!(merged["key"], "K");
        assert_eq!(merged["mac"], "14:ae:85:71:eb:44");
        assert_eq!(merged["volume"], 12);
        assert!(merged.get(PAIRING_AGENT).is_none());
        let set: Map<String, Value> = json!({"key": "CALLER"}).as_object().unwrap().clone();
        assert_eq!(with_device(&set, &device)["key"], "CALLER");
    }

    #[test]
    fn a_long_answer_is_cut_on_a_character() {
        assert_eq!(cut("  ready  "), "ready");
        let long = "é".repeat(ANSWER_CHARS + 3);
        assert!(cut(&long).ends_with("(cut: there is more)"));
    }
}
