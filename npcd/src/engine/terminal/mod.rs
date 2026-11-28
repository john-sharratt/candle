//! Terminals: how a mind with no body reaches real things
//! (`docs/zen_everywhere.md` §5).
//!
//! **A terminal is a real thing with an adapter behind it** — the house's
//! lights, a television, the house network, the cluster, a machine. It is not
//! a part of the world's map: the stations a body works are those, and stay.
//! A mind that takes no orders finds its terminals with `terminal_list`,
//! holds **one** connection (`terminal_connect`), and while it is connected
//! has that terminal's verbs — and no other terminal's — in its grammar.
//!
//! | Terminal | Behind it | Verbs |
//! |---|---|---|
//! | [`LIGHTS`] | every WiZ bulb the house network has shown | the light acts, on one bulb or [`ALL_LIGHTS`] |
//! | a device, by its name | a Sonos player, a receiver, a Cast device, a television | that kind's acts; pairing, for a television not yet paired |
//! | [`NETWORK`] | the house network | survey, wake, name a device, watch a handset |
//! | [`CLUSTER`] | zend's MCP, as Zen's service | read the cluster's standing, its services and its logs |
//! | a machine, by its host name | that machine's agent | its checks ([`agent::checks`]) |
//!
//! The cluster and a machine are **read-only**: deploying, restarting and
//! rolling out stay with people.
//!
//! | Module | |
//! |---|---|
//! | [`acts`] | the catalogue's entries |
//! | [`asked_job`] | whether a job on the cluster was asked for |
//!
//! This module is the reading: which terminals there are, what a connection
//! puts in a mind's grammar, and what each verb asks of the thing behind it.
//! The asking itself reaches the network and is the runtime's
//! (`crate::authority::terminals`).

pub mod acts;
pub mod asked_job;
pub mod watch;

use std::collections::BTreeMap;

use agent::checks::CheckLine;
use serde_json::{json, Map, Value};

use crate::bridge::finding::{
    reach, workable, Finding, ANDROIDTV, CAST, RECEIVER, SONOS, WEBOS, WIZ,
};
use crate::bridge::PAIRING_AGENT;
use crate::engine::tools::Within;
use crate::sim::terminals::Device;
use crate::sim::Sim;
pub use acts::{is_mine, TERMINAL_ACTS};

/// The one terminal over every bulb.
pub const LIGHTS: &str = "the lights";

/// The house network's terminal.
pub const NETWORK: &str = "the house network";

/// The cluster's terminal.
pub const CLUSTER: &str = "the cluster";

/// What names every bulb at once, among the lights.
pub const ALL_LIGHTS: &str = "all the lights";

/// A machine an agent runs on, as last heard of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// Its host name: what its terminal is called.
    pub host: String,
    /// Its agent's node, which a check is sent to.
    pub node: u64,
    /// Whether its agent takes work, as the gateway has it.
    pub ready: bool,
    /// Its checks, as it last listed them — empty until it is connected to.
    pub checks: Vec<CheckLine>,
}

/// What stands behind a terminal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Lights,
    Network,
    Cluster,
    Machine(Machine),
    /// A device, by the hardware address it is kept under.
    Device(Device),
}

/// One terminal a mind can reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Terminal {
    /// What it is called: what `terminal_connect` takes.
    pub name: String,
    pub kind: Kind,
}

impl Terminal {
    /// What it is and whether it answers, as `terminal_list` says it.
    pub fn line(&self, sim: &Sim) -> String {
        let what = match &self.kind {
            Kind::Lights => {
                let bulbs = sim.terminals.lights();
                let on = bulbs.iter().filter(|b| b.present).count();
                format!(
                    "every light in the house, {} of them, {} answering",
                    bulbs.len(),
                    on
                )
            }
            Kind::Network => format!(
                "the house's own network, {} devices known on it",
                sim.terminals.devices().len()
            ),
            Kind::Cluster => {
                "the machines the house runs on, as one: its standing, services and logs — to \
                 read, not to change"
                    .to_string()
            }
            Kind::Machine(m) => format!(
                "a machine of the house, to check — its agent {}",
                match m.ready {
                    true => "answers",
                    false => "is not taking work",
                }
            ),
            Kind::Device(d) => format!(
                "{} — {}",
                what_it_is(&d.finding),
                match (d.present, needs_pairing(d)) {
                    (false, _) => "not seen on the network at the last survey",
                    (true, true) => "answers, and is not paired yet",
                    (true, false) => "answers",
                }
            ),
        };
        format!("{}: {what}", self.name)
    }
}

/// What a device is, in a few words.
fn what_it_is(f: &Finding) -> String {
    let made = format!("{} {}", f.maker, f.model).trim().to_string();
    let kind = match f.profile.as_str() {
        SONOS => "a Sonos player",
        RECEIVER => "an AV receiver",
        CAST => "a Cast device",
        WEBOS => "an LG television",
        ANDROIDTV => "a Google television",
        _ => "a device",
    };
    match made.is_empty() {
        true => kind.to_string(),
        false => format!("{kind} ({made})"),
    }
}

/// Whether a device holds a credential the world works it with — any key that
/// is not the world's own bookkeeping.
fn has_key(d: &Device) -> bool {
    d.creds.keys().any(|k| !k.starts_with('_'))
}

/// Whether a device is a television the world has to pair with before it can
/// work it, and has not.
pub fn needs_pairing(d: &Device) -> bool {
    matches!(d.finding.profile.as_str(), WEBOS | ANDROIDTV) && !has_key(d)
}

/// Every terminal a mind in this world can reach: the lights when the house
/// has any, each device the world can work, the house network, the cluster
/// when this member is on one, and each machine an agent runs on. A device
/// whose name another terminal already goes by is listed as `<name> (device)`.
pub fn listed(sim: &Sim, machines: &[Machine], on_cluster: bool) -> Vec<Terminal> {
    let mut out: Vec<Terminal> = Vec::new();
    if !sim.terminals.lights().is_empty() {
        out.push(Terminal {
            name: LIGHTS.to_string(),
            kind: Kind::Lights,
        });
    }
    out.push(Terminal {
        name: NETWORK.to_string(),
        kind: Kind::Network,
    });
    if on_cluster {
        out.push(Terminal {
            name: CLUSTER.to_string(),
            kind: Kind::Cluster,
        });
    }
    for m in machines {
        out.push(Terminal {
            name: m.host.clone(),
            kind: Kind::Machine(m.clone()),
        });
    }
    for d in sim.terminals.devices() {
        // A bulb is worked through the lights, never as a terminal of its own.
        if d.finding.profile == WIZ || !workable(&d.finding.profile) {
            continue;
        }
        let name = match out.iter().any(|t| t.name.eq_ignore_ascii_case(&d.name)) {
            true => format!("{} (device)", d.name),
            false => d.name.clone(),
        };
        out.push(Terminal {
            name,
            kind: Kind::Device(d.clone()),
        });
    }
    out
}

/// The terminal called `name`, among [`listed`].
pub fn named(sim: &Sim, machines: &[Machine], on_cluster: bool, name: &str) -> Option<Terminal> {
    listed(sim, machines, on_cluster)
        .into_iter()
        .find(|t| t.name.eq_ignore_ascii_case(name.trim()))
}

/// The terminal `body` is connected to, if it still stands.
pub fn connected(
    sim: &Sim,
    machines: &[Machine],
    on_cluster: bool,
    body: &str,
) -> Option<Terminal> {
    named(sim, machines, on_cluster, sim.terminals.connected(body)?)
}

/// The acts `terminal` grants while a mind is connected to it, beyond the
/// house's own device acts, which its device sets bring ([`Within::at`]).
pub fn grants_of(terminal: &Terminal) -> Vec<&'static str> {
    match &terminal.kind {
        Kind::Lights => Vec::new(),
        Kind::Network => acts::NETWORK_ACTS.to_vec(),
        Kind::Cluster => acts::CLUSTER_ACTS.to_vec(),
        Kind::Machine(_) => acts::MACHINE_ACTS.to_vec(),
        Kind::Device(d) => match (d.finding.profile.as_str(), needs_pairing(d)) {
            (WEBOS | ANDROIDTV, false) => tv_acts(&d.finding.profile),
            _ => Vec::new(),
        },
    }
}

/// The television acts a television of `profile` can do: every one for an LG,
/// and for a Google television the ones its remote has a button for.
fn tv_acts(profile: &str) -> Vec<&'static str> {
    acts::TV_ACTS
        .iter()
        .copied()
        .filter(|t| tv_action(profile, t, &Map::new()).is_ok())
        .collect()
}

impl Within {
    /// This situation as a mind with no body has it: the terminal acts, the
    /// terminals it could connect to, and — connected — what its terminal
    /// brings: the device sets the house's acts are bound to, and the acts
    /// the terminal grants.
    pub fn at_terminals(
        mut self,
        sim: &Sim,
        machines: &[Machine],
        on_cluster: bool,
        body: &str,
    ) -> Self {
        // Nothing of the room its seat is in: the house's sets are filled by
        // what stands in a room, and a mind with no body stands in none.
        for set in [
            &mut self.links,
            &mut self.sonos,
            &mut self.receiver,
            &mut self.cast,
            &mut self.lights,
            &mut self.pairable,
            &mut self.pairing,
            &mut self.findings,
            &mut self.unlinked,
            &mut self.onboarded,
            &mut self.doors,
        ] {
            set.clear();
        }
        let now = connected(sim, machines, on_cluster, body);
        self.terminals = listed(sim, machines, on_cluster)
            .into_iter()
            .map(|t| t.name)
            .filter(|n| now.as_ref().is_none_or(|c| &c.name != n))
            .collect();
        let grant = |within: &mut Within, name: &str| {
            if !within.grants.iter().any(|g| g == name) {
                within.grants.push(name.to_string());
            }
        };
        for name in acts::REACH_ACTS {
            grant(&mut self, name);
        }
        // **The survey is offered whatever the mind is connected to**: the
        // house network always stands, and finding what is on it is where
        // every device's terminal comes from. Offered only on that terminal,
        // it was two acts away from a mind asked "what is on the network" —
        // and the mind never took the first.
        self.links = vec![NETWORK.to_string()];
        let Some(terminal) = now else {
            return self;
        };
        grant(&mut self, "terminal_disconnect");
        for name in grants_of(&terminal) {
            grant(&mut self, name);
        }
        match &terminal.kind {
            Kind::Lights => {
                self.lights = std::iter::once(ALL_LIGHTS.to_string())
                    .chain(sim.terminals.lights().iter().map(|d| d.name.clone()))
                    .collect();
            }
            Kind::Network => {
                let devices = sim.terminals.devices();
                self.devices = devices.iter().map(|d| d.name.clone()).collect();
                self.onboarded = devices
                    .iter()
                    .filter(|d| !d.finding.hardware.is_empty())
                    .map(|d| d.name.clone())
                    .collect();
            }
            Kind::Machine(m) => {
                self.checks = m.checks.iter().map(|c| c.name.clone()).collect();
            }
            Kind::Cluster => {}
            Kind::Device(d) => {
                let one = vec![d.name.clone()];
                match d.finding.profile.as_str() {
                    SONOS => self.sonos = one,
                    RECEIVER => self.receiver = one,
                    CAST => self.cast = one,
                    WEBOS | ANDROIDTV if needs_pairing(d) => {
                        match d.creds.contains_key(PAIRING_AGENT) {
                            true => self.pairing = one,
                            false => self.pairable = one,
                        }
                    }
                    _ => {}
                }
            }
        }
        self
    }
}

/// What a mind is told as it connects to `terminal`: what it is, and what it
/// can do there.
pub fn welcome(terminal: &Terminal, sim: &Sim) -> String {
    let mut lines = vec![format!("You are connected to {}.", terminal.name)];
    match &terminal.kind {
        Kind::Lights => {
            let bulbs = sim.terminals.lights();
            lines.push(format!(
                "Its lights: {}.",
                bulbs
                    .iter()
                    .map(|b| b.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            lines.push(format!(
                "Work one by its name, or every one at once as `{ALL_LIGHTS}`: `wiz_status`, \
                 `wiz_on`, `wiz_off`, `wiz_dim`, `wiz_white`, `wiz_colour`."
            ));
        }
        Kind::Network => {
            lines.push(format!(
                "`survey_network` from `{NETWORK}` finds every device on it now; `wake_device` \
                 wakes one that is asleep; `device_rename` gives one the name it goes by; \
                 `presence_watch` watches a handset as somebody's, so you are told when they \
                 come home and go out."
            ));
            for d in sim.terminals.devices() {
                let watched = match &d.person {
                    Some(p) => format!(" — watched as {p}'s"),
                    None => String::new(),
                };
                lines.push(format!(
                    "- {} at {}: {}{watched}",
                    d.name,
                    d.finding.address,
                    reach(&d.finding.profile)
                ));
            }
        }
        Kind::Cluster => lines.push(
            "It is read, not changed: `cluster_status` for how every member stands, \
             `cluster_services` for what each machine's keep runs, `cluster_logs` to search \
             what they wrote."
                .to_string(),
        ),
        Kind::Machine(m) => {
            lines.push(
                "Its checks look and change nothing; each report begins with what is different \
                 from the last time it ran. Run one with `machine_check`:"
                    .to_string(),
            );
            for c in &m.checks {
                lines.push(format!("- {}: {}", c.name, c.says));
                for a in &c.args {
                    let may = match a.values.is_empty() {
                        true => "a word or phrase".to_string(),
                        false => a.values.join(", "),
                    };
                    let need = match a.required {
                        true => "needed",
                        false => "optional",
                    };
                    lines.push(format!("    {} ({need}): {may}", a.name));
                }
            }
            if m.checks.is_empty() {
                lines.push("It lists no checks.".to_string());
            }
        }
        Kind::Device(d) => match (d.finding.profile.as_str(), needs_pairing(d)) {
            (WEBOS | ANDROIDTV, true) => lines.push(
                "It is a television the house has never paired with, so it answers nothing \
                 yet. Begin with `pair_device`: an LG shows a box on its screen for somebody \
                 there to accept; a Google television shows a code, which somebody must read \
                 back to you to finish with `pair_code`."
                    .to_string(),
            ),
            (profile @ (WEBOS | ANDROIDTV), false) => lines.push(format!(
                "It is a television. Here: {}.",
                tv_acts(profile)
                    .iter()
                    .map(|a| format!("`{a}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            (profile, _) => lines.push(format!(
                "It is {}. Its acts are the `{profile}_…` ones, on `{}`.",
                what_it_is(&d.finding),
                d.name
            )),
        },
    }
    lines.join("\n")
}

/// The action of a television's profile that television act `tool` stands
/// for, with its arguments — or why a television of that kind cannot do it.
pub fn tv_action(
    profile: &str,
    tool: &str,
    args: &Map<String, Value>,
) -> Result<(&'static str, Map<String, Value>), String> {
    let lg = profile == WEBOS;
    let text = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let with = |pairs: Vec<(&str, Value)>| -> Map<String, Value> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    };
    let no = |what: &str| {
        Err(format!(
            "A Google television's remote has no way to {what}."
        ))
    };
    Ok(match tool {
        "tv_on" => ("on", Map::new()),
        "tv_off" => ("off", Map::new()),
        "tv_louder" => ("louder", Map::new()),
        "tv_quieter" => ("quieter", Map::new()),
        "tv_mute" => ("mute", Map::new()),
        "tv_play" => ("play", Map::new()),
        "tv_pause" => ("pause", Map::new()),
        "tv_stop" => ("stop", Map::new()),
        "tv_status" if lg => ("status", Map::new()),
        "tv_status" => return no("say what it is doing"),
        "tv_unmute" if lg => ("unmute", Map::new()),
        "tv_unmute" => ("mute", Map::new()),
        "tv_volume" if lg => (
            "volume",
            with(vec![(
                "volume",
                args.get("volume").cloned().unwrap_or(Value::Null),
            )]),
        ),
        "tv_volume" => return no("set a volume by number: use `tv_louder` and `tv_quieter`"),
        "tv_input" if lg => ("input", with(vec![("input", json!(text("input")))])),
        "tv_input" => return no("switch inputs"),
        "tv_show" if lg => ("toast", with(vec![("message", json!(text("message")))])),
        "tv_show" => return no("show a message"),
        "tv_launch" => ("launch", with(vec![("app", json!(text("app")))])),
        "tv_youtube" if lg => (
            "launch",
            with(vec![
                ("app", json!("youtube")),
                ("content", json!(text("video").map(|v| youtube_id(&v)))),
            ]),
        ),
        "tv_youtube" => ("youtube", with(vec![("video", json!(text("video")))])),
        "tv_button" if lg => ("button", with(vec![("name", json!(text("button")))])),
        "tv_button" => ("key", with(vec![("key_name", json!(text("button")))])),
        other => return Err(format!("`{other}` is no television act.")),
    })
}

/// The video id in a YouTube link, or the text itself when it is one already.
fn youtube_id(video: &str) -> String {
    let v = video.trim();
    for marker in ["v=", "youtu.be/"] {
        if let Some(at) = v.find(marker) {
            return v[at + marker.len()..]
                .split(['&', '#', '?'])
                .next()
                .unwrap_or_default()
                .to_string();
        }
    }
    v.to_string()
}

/// A check's arguments as its caller wrote them — one `name: value` to a
/// line — as a map. A line with no `:` is skipped, and the machine then says
/// what it did not get.
pub fn check_args(args: &Map<String, Value>) -> BTreeMap<String, String> {
    let Some(text) = args.get("args").and_then(Value::as_str) else {
        return BTreeMap::new();
    };
    text.lines()
        .filter_map(|line| {
            let (k, v) = line.split_once(':')?;
            let (k, v) = (k.trim(), v.trim());
            (!k.is_empty()).then(|| (k.to_string(), v.to_string()))
        })
        .collect()
}

/// How many lines of a log one reading of it brings back.
const LOG_LINES: u32 = 60;

/// The call on zend one act of the cluster's terminal makes: the tool, and
/// its arguments from the act's. `Err` says what the act did not give.
///
/// **One act, one tool, named.** The cluster's terminal reaches what the
/// house's own tools reach — how it stands, its logs, and the jobs run on its
/// build machines, started and seen through — and nothing that decides for
/// the cluster (a rollout, a restart, a deploy) or that writes code: those are
/// a person's, and a Drone's task.
pub fn cluster_call(
    tool: &str,
    args: &Map<String, Value>,
) -> Result<(&'static str, Value), String> {
    let text = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
    };
    let need = |key: &str, what: &str| text(key).ok_or_else(|| format!("Say {what}."));
    Ok(match tool {
        "cluster_status" => ("cluster_status", json!({})),
        "cluster_services" => ("service_status", json!({})),
        "cluster_audit" => ("cluster_audit", json!({})),
        "cluster_jobs" => ("job_list", json!({})),
        "cluster_logs" => {
            let mut call = json!({
                "grep": need("grep", "what to search the logs for")?,
                "source": "cluster",
                "limit": LOG_LINES,
            });
            for key in ["level", "since"] {
                if let Some(v) = text(key) {
                    call[key] = json!(v);
                }
            }
            ("log_grep", call)
        }
        "cluster_log" => (
            "log_read",
            json!({"source": need("source", "which log to read")?, "limit": LOG_LINES}),
        ),
        "cluster_member" => (
            "member_status",
            json!({"node": need("node", "which member, by its node id")?}),
        ),
        "cluster_memory" => (
            "member_memory",
            json!({"node": need("node", "which member, by its node id")?}),
        ),
        "cluster_job" => (
            "job_status",
            json!({"jobs": [need("job", "which job, by its handle")?], "wait_s": 0}),
        ),
        "cluster_job_read" => {
            let page = match text("page") {
                None => 1,
                Some(p) => p
                    .parse::<u32>()
                    .ok()
                    .filter(|n| *n >= 1)
                    .ok_or_else(|| format!("{p} is not a page: pages are numbered from 1."))?,
            };
            (
                "job_read",
                json!({"job": need("job", "which job, by its handle")?, "page": page, "wait_s": 0}),
            )
        }
        "cluster_job_stop" => (
            "job_cancel",
            json!({"job": need("job", "which job, by its handle")?}),
        ),
        "cluster_job_start" => {
            let given = need("args", "the program's arguments")?;
            // One program and its arguments, never a line for a shell: what a
            // shell would join, pipe or redirect is refused here, where the
            // refusal can say so.
            if let Some(bad) = given
                .split_whitespace()
                .find(|w| matches!(*w, "&&" | "||" | "|" | ";" | ">" | ">>" | "<"))
            {
                return Err(format!(
                    "`{bad}` is a shell's: a job runs one program with its arguments, and \
                     nothing joins two."
                ));
            }
            let words: Vec<&str> = given.split_whitespace().collect();
            (
                "repo_job_start",
                json!({
                    "repo": need("repo", "which repository")?,
                    "branch": "main",
                    "program": need("program", "which program to run")?,
                    "args": words,
                }),
            )
        }
        other => return Err(format!("The cluster has no `{other}`.")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::finding::DEVICE;

    /// **Each act of the cluster's terminal is one named tool of zend's**,
    /// with its arguments as raw JSON; nothing a shell would join is passed,
    /// and what an act did not give is said.
    #[test]
    fn each_cluster_act_is_one_named_call() {
        let args = |v: Value| v.as_object().unwrap().clone();
        let call = |tool: &str, v: Value| cluster_call(tool, &args(v));
        assert_eq!(
            call("cluster_status", json!({})),
            Ok(("cluster_status", json!({})))
        );
        assert_eq!(
            call("cluster_services", json!({})),
            Ok(("service_status", json!({})))
        );
        assert_eq!(
            call("cluster_logs", json!({"grep": "panicked", "level": "WARN"})),
            Ok((
                "log_grep",
                json!({"grep": "panicked", "source": "cluster", "limit": 60, "level": "WARN"})
            ))
        );
        assert_eq!(
            call("cluster_member", json!({"node": " 09dd3987369d9cd8 "})),
            Ok(("member_status", json!({"node": "09dd3987369d9cd8"})))
        );
        assert_eq!(
            call(
                "cluster_job",
                json!({"job": "build:76c3a0f4d39bd9db:Be-t_qMsg8U"})
            ),
            Ok((
                "job_status",
                json!({"jobs": ["build:76c3a0f4d39bd9db:Be-t_qMsg8U"], "wait_s": 0})
            ))
        );
        assert_eq!(
            call(
                "cluster_job_read",
                json!({"job": "cluster:76c3a0f4d39bd9db:4"})
            ),
            Ok((
                "job_read",
                json!({"job": "cluster:76c3a0f4d39bd9db:4", "page": 1, "wait_s": 0})
            ))
        );
        assert_eq!(
            call(
                "cluster_job_start",
                json!({"repo": "candle", "program": "cargo", "args": "test -p web  --lib"})
            ),
            Ok((
                "repo_job_start",
                json!({
                    "repo": "candle",
                    "branch": "main",
                    "program": "cargo",
                    "args": ["test", "-p", "web", "--lib"],
                })
            ))
        );
        assert!(call(
            "cluster_job_start",
            json!({"repo": "candle", "program": "cargo", "args": "test && rm -rf ."})
        )
        .unwrap_err()
        .contains("`&&` is a shell's"));
        assert_eq!(
            call("cluster_member", json!({})),
            Err("Say which member, by its node id.".to_string())
        );
        assert!(call("cluster_job_read", json!({"job": "x", "page": "0"})).is_err());
        // Every act the terminal offers has a call, and nothing else does.
        for act in acts::CLUSTER_ACTS {
            let got = call(act, json!({}));
            assert!(
                !matches!(&got, Err(e) if e.contains("has no")),
                "`{act}` is offered and has no call"
            );
        }
        assert!(call("service_deploy", json!({})).is_err());
    }

    fn finding(hardware: &str, name: &str, profile: &str) -> Finding {
        Finding {
            address: format!("192.168.1.{}", hardware.len() + name.len()),
            hardware: hardware.into(),
            names: vec![name.into()],
            profile: profile.into(),
            ..Finding::default()
        }
    }

    fn house() -> Sim {
        let mut s = Sim::new();
        s.terminals.surveyed(vec![
            finding("a8:bb:50:00:00:01", "kitchen light", WIZ),
            finding("a8:bb:50:00:00:02", "hall light", WIZ),
            finding("f0:f6:c1:a0:79:b4", "SonosZP", SONOS),
            finding("14:ae:85:71:eb:44", "the den TV", WEBOS),
            finding("3c:22:fb:00:00:09", "Pixel-8", DEVICE),
        ]);
        s
    }

    fn machines() -> Vec<Machine> {
        vec![Machine {
            host: "WhiteWorld".into(),
            node: 7,
            ready: true,
            checks: vec![CheckLine {
                name: "listeners".into(),
                says: "Every port it listens on.".into(),
                args: Vec::new(),
            }],
        }]
    }

    fn names(ts: &[Terminal]) -> Vec<&str> {
        ts.iter().map(|t| t.name.as_str()).collect()
    }

    /// **The list is the real things there are**: one terminal over every
    /// bulb, the network, the cluster only on one, each machine, and each
    /// device the world can work — a bulb never by itself, a phone never.
    #[test]
    fn the_list_is_the_real_things_there_are() {
        let s = house();
        assert_eq!(
            names(&listed(&s, &machines(), true)),
            [
                LIGHTS,
                NETWORK,
                CLUSTER,
                "WhiteWorld",
                "SonosZP",
                "the den TV"
            ]
        );
        assert_eq!(
            names(&listed(&Sim::new(), &[], false)),
            [NETWORK],
            "a house with nothing surveyed still has its network"
        );
        let lines: Vec<String> = listed(&s, &machines(), true)
            .iter()
            .map(|t| t.line(&s))
            .collect();
        assert_eq!(
            lines[0],
            "the lights: every light in the house, 2 of them, 2 answering"
        );
        assert_eq!(
            lines[5],
            "the den TV: an LG television — answers, and is not paired yet"
        );
    }

    /// **A mind with no connection has the three terminal acts, the house
    /// network to survey, and nothing else of any terminal's**; connected, it
    /// has that terminal's verbs alone, and the terminal it is on is not
    /// offered to connect to again.
    #[test]
    fn a_connection_brings_one_terminals_verbs_and_no_others() {
        let mut s = house();
        let at = |s: &Sim| Within::default().at_terminals(s, &machines(), true, "zen");
        let free = at(&s);
        assert_eq!(free.grants, ["terminal_list", "terminal_connect", "rest"]);
        assert_eq!(free.links, [NETWORK], "the survey is offered unconnected");
        assert_eq!(free.terminals.len(), 6);
        assert!(free.lights.is_empty() && free.sonos.is_empty() && free.checks.is_empty());

        s.terminals.connect("zen", LIGHTS);
        let lit = at(&s);
        assert_eq!(lit.lights, [ALL_LIGHTS, "hall light", "kitchen light"]);
        assert_eq!(
            lit.grants,
            [
                "terminal_list",
                "terminal_connect",
                "rest",
                "terminal_disconnect"
            ]
        );
        assert!(!lit.terminals.iter().any(|t| t == LIGHTS));
        assert!(lit.sonos.is_empty(), "another terminal's device");

        s.terminals.connect("zen", "sonoszp");
        let playing = at(&s);
        assert_eq!(playing.sonos, ["SonosZP"]);
        assert!(playing.lights.is_empty(), "the last connection was let go");

        s.terminals.connect("zen", "WhiteWorld");
        let checking = at(&s);
        assert_eq!(checking.checks, ["listeners"]);
        assert!(checking.grants.iter().any(|g| g == "machine_check"));
        assert!(!checking.grants.iter().any(|g| g == "cluster_logs"));
        assert_eq!(checking.links, [NETWORK], "and on another terminal");

        s.terminals.connect("zen", NETWORK);
        let net = at(&s);
        assert_eq!(net.links, [NETWORK]);
        assert_eq!(net.devices.len(), 5);
        assert!(net.grants.iter().any(|g| g == "presence_watch"));

        // A terminal that is gone is no connection: its verbs go with it.
        s.terminals.connect("zen", "the toaster");
        assert_eq!(at(&s).grants, ["terminal_list", "terminal_connect", "rest"]);
    }

    /// **A television is paired before it is worked**: unpaired it offers the
    /// pairing and none of its acts; begun, the code; paired, its acts.
    #[test]
    fn a_television_is_paired_before_it_is_worked() {
        let mut s = house();
        s.terminals.connect("zen", "the den TV");
        let at = |s: &Sim| Within::default().at_terminals(s, &[], false, "zen");
        let unpaired = at(&s);
        assert_eq!(unpaired.pairable, ["the den TV"]);
        assert!(!unpaired.grants.iter().any(|g| g == "tv_off"));

        let hw = "14:ae:85:71:eb:44";
        s.terminals.keep_creds(
            hw,
            &BTreeMap::from([(PAIRING_AGENT.to_string(), "7".to_string())]),
            &[],
        );
        let begun = at(&s);
        assert_eq!(begun.pairing, ["the den TV"]);
        assert!(begun.pairable.is_empty());

        s.terminals.keep_creds(
            hw,
            &BTreeMap::from([("key".to_string(), "K".to_string())]),
            &[PAIRING_AGENT.to_string()],
        );
        let paired = at(&s);
        assert!(paired.pairable.is_empty() && paired.pairing.is_empty());
        assert!(paired.grants.iter().any(|g| g == "tv_off"));
        assert!(
            paired.grants.iter().any(|g| g == "tv_show"),
            "an LG shows a message"
        );
    }

    /// **A television act is the action its kind of television has for it**,
    /// and one a Google television's remote has no button for is not offered.
    #[test]
    fn a_television_act_is_its_profiles_action() {
        let args = |v: Value| v.as_object().unwrap().clone();
        assert_eq!(
            tv_action(WEBOS, "tv_volume", &args(json!({"volume": 12}))).unwrap(),
            ("volume", args(json!({"volume": 12})))
        );
        assert_eq!(
            tv_action(
                WEBOS,
                "tv_youtube",
                &args(json!({"video": "https://www.youtube.com/watch?v=abc123&t=4"}))
            )
            .unwrap(),
            (
                "launch",
                args(json!({"app": "youtube", "content": "abc123"}))
            )
        );
        assert_eq!(
            tv_action(ANDROIDTV, "tv_button", &args(json!({"button": "back"}))).unwrap(),
            ("key", args(json!({"key_name": "back"})))
        );
        assert!(tv_action(ANDROIDTV, "tv_volume", &Map::new())
            .unwrap_err()
            .contains("`tv_louder`"));
        let google = tv_acts(ANDROIDTV);
        assert!(google.contains(&"tv_off") && google.contains(&"tv_youtube"));
        assert!(!google.contains(&"tv_show") && !google.contains(&"tv_input"));
        assert_eq!(tv_acts(WEBOS).len(), acts::TV_ACTS.len());
    }

    /// **A connection says what it offers**: the lights by name, a machine's
    /// checks with what each reports.
    #[test]
    fn a_connection_says_what_it_offers() {
        let s = house();
        let lights = named(&s, &machines(), true, "The Lights").unwrap();
        let said = welcome(&lights, &s);
        assert!(
            said.starts_with("You are connected to the lights."),
            "{said}"
        );
        assert!(said.contains("hall light, kitchen light"), "{said}");
        let machine = named(&s, &machines(), true, "whiteworld").unwrap();
        let said = welcome(&machine, &s);
        assert!(
            said.contains("- listeners: Every port it listens on."),
            "{said}"
        );
    }

    /// A check's arguments are read one to a line.
    #[test]
    fn a_checks_arguments_are_read_a_line_each() {
        let args: Map<String, Value> =
            json!({"check": "logs", "args": "log: System\n grep : disk error\nnonsense"})
                .as_object()
                .unwrap()
                .clone();
        assert_eq!(
            check_args(&args),
            BTreeMap::from([
                ("grep".to_string(), "disk error".to_string()),
                ("log".to_string(), "System".to_string())
            ])
        );
        assert!(check_args(&Map::new()).is_empty());
    }
}
