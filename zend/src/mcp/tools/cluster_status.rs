//! `cluster_status`: how the cluster stands (`docs/claude_zend_gateway.md`
//! §8.1) — every zend member's own `/v1/status`, gathered by the gateway, and
//! for an Admin the gateway's view: apps, members, upgrades, faults.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::service::status::trying;
use super::service::{keep, KEEP_STATE};
use super::{Answer, Ctx};
use crate::mcp::ops::json_of;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "cluster_status";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "How the cluster stands: each member — a machine running the motherd daemon — with \
            its machine name, node id, build, state, when it started, and how each app it hosts stands; and, \
            for an Admin, the gateway's view: each app's floor (the oldest \
            build it still routes to) and ready members; each member's place in an upgrade, from its own \
            report — fetching a build (how much of it), waiting for its turn, installing, stranded and why, \
            an install that failed and was restored with its last line — and a member left below the floor \
            with nothing under way; candidates, when a raised floor began its rollout and when the grace \
            period ends; and each member in or near fault, with why, its free disk and unsent events. \
            Check it before delegating heavy work, before and after any operation, whenever a task stalls, \
            and while a build rolls out.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }
    })
}

/// How long ago `ms` was, as a person reads it.
pub fn ago(ms: u64, now: u64) -> String {
    let s = now.saturating_sub(ms) / 1000;
    match s {
        0..=89 => format!("{s} s ago"),
        90..=5_399 => format!("{} min ago", s / 60),
        _ => format!("{:.1} h ago", s as f64 / 3600.0),
    }
}

fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("?")
}

/// The build under `key`, as the gateway and every member write one —
/// `{count, hash, local}` — read as `count-hash`, with ` (local)` for a
/// build with uncommitted changes; `None` when there is none.
fn build_of(v: &Value, key: &str) -> Option<String> {
    let b = v.get(key).filter(|b| b.is_object())?;
    let local = b
        .get("local")
        .and_then(Value::as_str)
        .is_some_and(|l| !l.is_empty());
    Some(format!(
        "{}-{}{}",
        b.get("count").and_then(Value::as_u64)?,
        b.get("hash").and_then(Value::as_str)?,
        if local { " (local)" } else { "" }
    ))
}

/// Each app's replication on one member, from its own `/v1/status` body
/// (`cluster.apps.<app>`): a line naming each app's phase and generation, and
/// what is wrong — an app whose phase or link is not `joined` is cut off or
/// rejoining, and what it writes reaches no peer until it rejoins.
fn replication_of(body: &Value) -> (Option<String>, Vec<String>) {
    let Some(apps) = body.pointer("/cluster/apps").and_then(Value::as_object) else {
        return (None, Vec::new());
    };
    let mut said = Vec::new();
    let mut wrong = Vec::new();
    for (app, a) in apps {
        let link = str_of(a, "link");
        let r = a.get("replication").cloned().unwrap_or(Value::Null);
        let phase = str_of(&r, "phase");
        let mut part = format!(
            "{app} {phase} gen {}",
            r.get("generation")
                .map_or_else(|| "?".into(), Value::to_string)
        );
        if link != "joined" {
            part.push_str(&format!(" (link {link})"));
        }
        let unsent = r.get("unsent").and_then(Value::as_u64).unwrap_or(0);
        if unsent > 0 {
            part.push_str(&format!(", {unsent} unsent"));
        }
        let catchups = r
            .get("catchups")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if catchups > 0 {
            part.push_str(&format!(", {catchups} catching up"));
        }
        if phase != "joined" || link != "joined" {
            wrong.push(format!("{app} is {phase}, link {link}"));
        }
        said.push(part);
    }
    ((!said.is_empty()).then(|| said.join(", ")), wrong)
}

/// The members' own statuses, as the gateway's envelope gathered them: any
/// member whose replication is not joined first, as a problem, then each
/// member's line.
pub fn render_members(envelope: &Value, now: u64) -> String {
    let members = envelope
        .get("members")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = format!("Members answering /v1/status: {}", members.len());
    let problems: Vec<String> = members
        .iter()
        .filter(|m| m.get("status").and_then(Value::as_u64) == Some(200))
        .flat_map(|m| {
            let who = format!("{} {}", str_of(m, "host"), str_of(m, "node"));
            let body = m.get("body").cloned().unwrap_or(Value::Null);
            replication_of(&body).1.into_iter().map(move |w| {
                format!("\nproblem: {who}: {w} — what it writes reaches no peer until it rejoins")
            })
        })
        .collect();
    for p in problems {
        out.push_str(&p);
    }
    for m in &members {
        let (host, node) = (str_of(m, "host"), str_of(m, "node"));
        let code = m.get("status").and_then(Value::as_u64).unwrap_or(0);
        let body = m.get("body").cloned().unwrap_or(Value::Null);
        if code != 200 {
            out.push_str(&format!(
                "\n- {host} {node}: no answer ({code} {})",
                m.get("error").and_then(Value::as_str).unwrap_or("")
            ));
            continue;
        }
        // The release build is the cluster block's; the top-level `build` is
        // the UI's reload hash. A member outside any cluster has no block.
        let build = body
            .get("cluster")
            .and_then(|c| build_of(c, "build"))
            .map_or_else(String::new, |b| format!(" build {b}"));
        let mut line = format!("\n- {host} {node}{build}: {}", str_of(&body, "state"));
        if let Some(d) = body
            .get("detail")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
        {
            line.push_str(&format!(" — {d}"));
        }
        if let Some(at) = body.get("started_at_ms").and_then(Value::as_u64) {
            line.push_str(&format!(", started {}", ago(at, now)));
        }
        if let Some(tenants) = body.get("tenants").and_then(Value::as_object) {
            let t: Vec<String> = tenants
                .iter()
                .map(|(name, s)| format!("{name} {}", str_of(s, "state")))
                .collect();
            line.push_str(&format!("; tenants: {}", t.join(", ")));
        }
        if let Some(r) = replication_of(&body).0 {
            line.push_str(&format!("; replication: {r}"));
        }
        out.push_str(&line);
    }
    if let Some(group) = envelope.get("group").filter(|g| !g.is_null()) {
        out.push_str(&format!("\ngroup: {group}"));
    }
    out
}

/// `bytes` as a person reads a size.
fn size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GB", b / GIB)
    } else {
        format!("{:.0} MB", b / MIB)
    }
}

/// The floor's build count of `app` in the gateway's `status`, for an app
/// that has one.
fn floor_count(status: &Value, app: &str) -> Option<u64> {
    status
        .get("apps")?
        .as_array()?
        .iter()
        .find(|a| a.get("app").and_then(Value::as_str) == Some(app))?
        .pointer("/floor/count")?
        .as_u64()
}

/// Where member `m` — a row of the gateway's status — stands in an upgrade,
/// from its own report (`COORD`'s upgrade view): what it is doing, how far a
/// fetch has got, or why it is stuck. A member below its app's `floor` with
/// nothing under way says so: that is a member left behind. `None` for a
/// member with nothing to say.
pub fn upgrade_line(m: &Value, floor: Option<u64>) -> Option<String> {
    let below = floor.is_some_and(|f| {
        m.pointer("/build/count")
            .and_then(Value::as_u64)
            .is_some_and(|c| c < f)
    });
    let up = m.pointer("/coord/upgrade").filter(|u| !u.is_null());
    let mut said: Vec<String> = Vec::new();
    if let Some(up) = up {
        if let Some(step) = up
            .get("step")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            said.push(step.to_string());
        }
        if let Some(f) = up.get("fetching").filter(|f| !f.is_null()) {
            let (got, of) = (
                f.get("bytes").and_then(Value::as_u64).unwrap_or(0),
                f.get("size").and_then(Value::as_u64).unwrap_or(0),
            );
            let resumes = f.get("resumes").and_then(Value::as_u64).unwrap_or(0);
            said.push(format!(
                "fetching {}: {} of {}{}",
                str_of(f, "build"),
                size(got),
                size(of),
                if resumes > 0 {
                    format!(", resumed {resumes}×")
                } else {
                    String::new()
                }
            ));
        }
        if let Some(why) = up.get("stranded").and_then(Value::as_str) {
            said.push(format!("stranded: {why}"));
        }
        if let Some(f) = up.get("failure").filter(|f| !f.is_null()) {
            let last = f
                .get("last_lines")
                .and_then(Value::as_array)
                .and_then(|l| l.last())
                .and_then(Value::as_str)
                .map_or(String::new(), |l| format!(" (last line: {l})"));
            said.push(format!(
                "its install of {} failed and was restored: {}{last}",
                str_of(f, "build"),
                str_of(f, "reason")
            ));
        }
        if up.get("paused").and_then(Value::as_bool) == Some(true) {
            said.push("the app's rollout is paused".into());
        }
    }
    if let Some(mode) = m
        .pointer("/coord/mode")
        .and_then(Value::as_str)
        .filter(|m| !matches!(*m, "" | "joined"))
    {
        said.push(format!("mode {mode}"));
    }
    if said.is_empty() && below {
        return Some(
            "below the floor, with no upgrade under way: it is left behind — read its \
             log for why (`log_grep`, or a worker's `cluster_logs`)"
                .into(),
        );
    }
    if said.is_empty() {
        return None;
    }
    Some(format!(
        "{}{}",
        if below { "below the floor: " } else { "" },
        said.join("; ")
    ))
}

/// One fault row of the gateway's status, as a line.
fn fault_line(f: &Value, now: u64) -> String {
    let who = format!(
        "{} {} {}",
        str_of(f, "app"),
        str_of(f, "host"),
        str_of(f, "node")
    );
    let free = f.get("free").and_then(Value::as_u64).unwrap_or(0);
    let floor = f.get("floor").and_then(Value::as_u64).unwrap_or(0);
    let unsent = f.get("unsent").and_then(Value::as_u64).unwrap_or(0);
    let disk = format!(
        "{} free of a {} floor, {unsent} events unsent",
        size(free),
        size(floor)
    );
    match f.get("fault").filter(|x| !x.is_null()) {
        Some(x) => format!(
            "{who}: in fault since {} — {}; {disk}",
            ago(x.get("since_ms").and_then(Value::as_u64).unwrap_or(0), now),
            str_of(x, "reason")
        ),
        None => format!("{who}: near its disk floor — {disk}"),
    }
}

/// The gateway's own view, for an Admin.
pub fn render_gateway(status: &Value) -> String {
    let now = status.get("now_ms").and_then(Value::as_u64).unwrap_or(0);
    let mut out = format!(
        "gateway {} (build {}){}",
        str_of(status, "gateway"),
        build_of(status, "build").as_deref().unwrap_or("?"),
        if status.get("draining").and_then(Value::as_bool) == Some(true) {
            ", draining"
        } else {
            ""
        }
    );
    for app in status
        .get("apps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        out.push_str(&format!(
            "\n- {}: {}, {} of {} ready",
            str_of(app, "app"),
            str_of(app, "mode"),
            app.get("ready").and_then(Value::as_u64).unwrap_or(0),
            app.get("members").and_then(Value::as_u64).unwrap_or(0),
        ));
        // A service is admitted on any build and never rolled out: it has
        // no floor to name.
        if app.get("service").and_then(Value::as_bool) == Some(true) {
            out.push_str(", a service (no floor, released per machine)");
            continue;
        }
        out.push_str(&format!(
            ", floor {}",
            build_of(app, "floor").as_deref().unwrap_or("none")
        ));
        if let Some(p) = build_of(app, "pending_floor") {
            out.push_str(&format!(", pending floor {p}"));
        }
    }
    for m in status
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        out.push_str(&format!(
            "\n  {} {} {} build {}: {}",
            str_of(m, "app"),
            str_of(m, "host"),
            str_of(m, "node"),
            build_of(m, "build").as_deref().unwrap_or("?"),
            str_of(m, "state"),
        ));
        let flags: Vec<&str> = ["syncing", "lagging", "silent"]
            .into_iter()
            .filter(|key| m.get(*key).and_then(Value::as_bool) == Some(true))
            .collect();
        if !flags.is_empty() {
            out.push_str(&format!(" ({})", flags.join(", ")));
        }
        if let Some(line) = upgrade_line(m, floor_count(status, str_of(m, "app"))) {
            out.push_str(&format!(" — {line}"));
        }
    }
    for u in status
        .get("upgrades")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let app = str_of(u, "app");
        if let Some(c) = u.get("candidate").filter(|c| !c.is_null()) {
            out.push_str(&format!(
                "\nupgrade {app}: candidate {} on {}, {}",
                str_of(c, "build"),
                str_of(c, "node"),
                c.get("state").map_or_else(|| "?".into(), Value::to_string),
            ));
        }
        if let Some(at) = u.get("rollout_since_ms").and_then(Value::as_u64) {
            out.push_str(&format!(
                "\nupgrade {app}: floor raised {}, members below it upgrading",
                ago(at, now)
            ));
        }
        if let Some(at) = u.get("deadline_ms").and_then(Value::as_u64) {
            out.push_str(&format!(
                "\nupgrade {app}: older builds served until the grace period ends, in {}",
                ago(now, at).trim_end_matches(" ago")
            ));
        }
    }
    for f in status
        .get("faults")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        out.push_str(&format!("\nfault: {}", fault_line(f, now)));
    }
    out
}

/// A machine no member answered for, as its keep `state` says it stands: a
/// trial in progress is a member coming up, a running motherd that does not
/// answer is loading or stuck, and a stopped one is down.
pub fn absent_line(host: &str, state: &Value, now: u64) -> String {
    let motherd = state
        .get("services")
        .and_then(Value::as_array)
        .and_then(|s| {
            s.iter()
                .find(|s| s.get("name").and_then(Value::as_str) == Some("motherd"))
        });
    let Some(m) = motherd else {
        return format!("- {host}: not answering, and its keep runs no motherd");
    };
    let build = build_of(m, "build").unwrap_or_else(|| "?".into());
    if let Some(t) = m.pointer("/trials/running").filter(|t| !t.is_null()) {
        return format!(
            "- {host}: not answering — {} (installed {build})",
            trying(t, now)
        );
    }
    match m.get("running").and_then(Value::as_u64) {
        Some(pid) => format!(
            "- {host}: not answering — keep runs motherd {build} as pid {pid}, so it is \
             loading or stuck: read its lines with `log_grep` (a worker's `cluster_logs`)"
        ),
        None => format!("- {host}: not answering — keep has motherd {build} stopped"),
    }
}

/// The machines with an agent but no member in `envelope`, as their keeps
/// say they stand.
async fn absent(call: &OpsCall, envelope: &Value) -> Vec<String> {
    let Some(agents) = call.session.agents() else {
        return Vec::new();
    };
    let Ok(listed) = agents.listing().await else {
        return Vec::new();
    };
    let answered: Vec<String> = envelope
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|m| str_of(m, "host").to_ascii_lowercase())
        .collect();
    let mut hosts: Vec<String> = listed
        .iter()
        .map(|a| a.host.clone())
        .filter(|h| !answered.contains(&h.to_ascii_lowercase()))
        .collect();
    hosts.sort_by_key(|h| h.to_ascii_lowercase());
    hosts.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    let now = now_ms();
    let mut lines = Vec::new();
    for host in hosts {
        lines.push(
            match keep(call, Method::GET, KEEP_STATE, &host, None).await {
                Ok(state) => absent_line(&host, &state, now),
                Err(e) => format!("- {host}: not answering, and its keep could not be read: {e}"),
            },
        );
    }
    lines
}

pub async fn run(call: &OpsCall, _args: &Value) -> Answer {
    let within = Duration::from_secs(15);
    let members = call.up(Method::GET, "/v1/status", None, within);
    // The gateway's own view is asked for beside the members', not after it.
    let gateway = async {
        if !call.admin() {
            return None;
        }
        Some(
            call.up(Method::GET, "/gateway/status.json?log=0", None, within)
                .await
                .and_then(|r| json_of(&r)),
        )
    };
    let (reply, gateway) = tokio::join!(members, gateway);
    let reply = match reply {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    let envelope = match json_of(&reply) {
        Ok(v) => v,
        Err(e) => return Answer::error(format!("/v1/status: {e}")),
    };
    let mut out = render_members(&envelope, now_ms());
    if call.admin() {
        let gone = absent(call, &envelope).await;
        if !gone.is_empty() {
            out.push_str("\nMachines with no member answering:\n");
            out.push_str(&gone.join("\n"));
        }
    }
    if let Some(gateway) = gateway {
        out.push_str("\n\n");
        match gateway {
            Ok(v) => out.push_str(&render_gateway(&v)),
            Err(e) => out.push_str(&format!("the gateway's own view could not be read: {e}")),
        }
    }
    Answer::ok(out)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    #[test]
    fn a_members_line_says_its_state_its_start_and_its_tenants() {
        let envelope = json!({
            "members": [
                {"node": "00000000000000ab", "host": "blackworld", "status": 200, "body": {
                    "state": "ready", "build": "0123456789abcdef", "detail": "",
                    "started_at_ms": NOW - 7_200_000,
                    "tenants": {"npcd": {"state": "ready"}, "zend": {"state": "loading"}},
                    "cluster": {"build": {"count": 5739, "hash": "c544", "local": ""}, "apps": {}}
                }},
                {"node": "00000000000000cd", "host": "whiteworld", "status": 504, "error": "deadline"}
            ],
            "group": {"members": 2}
        });
        assert_eq!(
            render_members(&envelope, NOW),
            "Members answering /v1/status: 2\n\
             - blackworld 00000000000000ab build 5739-c544: ready, started 2.0 h ago; tenants: \
             npcd ready, zend loading\n\
             - whiteworld 00000000000000cd: no answer (504 deadline)\n\
             group: {\"members\":2}"
        );
    }

    /// **A member's build is its cluster block's**, never the top-level
    /// `build`, which is the UI's reload hash; a member outside any cluster
    /// names none.
    #[test]
    fn a_members_build_is_its_cluster_blocks() {
        let envelope = json!({"members": [
            {"node": "01", "host": "bw", "status": 200, "body": {
                "state": "ready", "build": "0123456789abcdef",
                "cluster": {"build": {"count": 4785, "hash": "1c17aadd0", "local": "9f"}, "apps": {}}}},
            {"node": "02", "host": "ww", "status": 200, "body": {
                "state": "ready", "build": "0123456789abcdef", "cluster": null}}
        ]});
        assert_eq!(
            render_members(&envelope, NOW),
            "Members answering /v1/status: 2\n\
             - bw 01 build 4785-1c17aadd0 (local): ready\n\
             - ww 02: ready"
        );
    }

    /// The gateway's status as `/gateway/status.json` writes it: every build
    /// an object, and a candidate's already a name.
    #[test]
    fn the_gateways_view_names_apps_members_and_candidates() {
        let b = |count: u64, hash: &str, local: &str| json!({"count": count, "hash": hash, "local": local});
        let status = json!({
            "gateway": "bw", "build": b(1, "a", ""), "draining": false,
            "apps": [
                {"app": "zend", "service": false, "mode": "clustered", "ready": 2, "members": 2,
                 "floor": b(10, "h", ""), "pending_floor": b(11, "h", "")},
                {"app": "npcd", "service": false, "mode": "dumb", "ready": 0, "members": 0,
                 "floor": null, "pending_floor": null},
                {"app": "mcp", "service": true, "mode": "clustered", "ready": 2, "members": 2,
                 "floor": null, "pending_floor": null}
            ],
            "members": [
                {"app": "zend", "host": "bw", "node": "01", "build": b(10, "h", ""), "local_build": false, "state": "ready"},
                {"app": "mcp", "host": "ww", "node": "03", "build": b(12, "j", "9f"), "local_build": true, "state": "ready"}
            ],
            "upgrades": [{"app": "zend", "candidate": {"node": "02", "build": "11-h", "state": "certifying"}}],
            "faults": []
        });
        assert_eq!(
            render_gateway(&status),
            "gateway bw (build 1-a)\n\
             - zend: clustered, 2 of 2 ready, floor 10-h, pending floor 11-h\n\
             - npcd: dumb, 0 of 0 ready, floor none\n\
             - mcp: clustered, 2 of 2 ready, a service (no floor, released per machine)\n  \
             zend bw 01 build 10-h: ready\n  \
             mcp ww 03 build 12-j (local): ready\n\
             upgrade zend: candidate 11-h on 02, \"certifying\""
        );
    }

    /// **A member says where it is in an upgrade, from its own report**: a
    /// fetch with how far it has got, a stranding with why, a failed install
    /// with its last line; one below the floor with nothing under way is named
    /// left behind, and one at the floor with nothing to say says nothing.
    #[test]
    fn a_member_says_where_it_is_in_an_upgrade() {
        let gib = 1024 * 1024 * 1024u64;
        let member = |count: u64, upgrade: Value| {
            json!({"build": {"count": count, "hash": "h", "local": ""},
                   "coord": {"mode": "joined", "upgrade": upgrade}})
        };
        let fetching = member(
            4350,
            json!({"step": "upgrading", "fetching": {"build": "4381-7b4f4e062", "bytes": 3 * gib / 2,
                   "size": 3 * gib, "resumes": 1}, "stranded": null, "failure": null, "paused": false}),
        );
        assert_eq!(
            upgrade_line(&fetching, Some(4381)).unwrap(),
            "below the floor: upgrading; fetching 4381-7b4f4e062: 1.5 GB of 3.0 GB, resumed 1×"
        );
        let stuck = member(
            4350,
            json!({"step": "stranded", "fetching": null, "stranded": "no member offers 4381 for windows",
                   "failure": {"build": "4376-12a48b0", "reason": "no serving line",
                               "last_lines": ["loading", "panicked at x.rs:1"]}, "paused": true}),
        );
        assert_eq!(
            upgrade_line(&stuck, Some(4381)).unwrap(),
            "below the floor: stranded; stranded: no member offers 4381 for windows; its install of \
             4376-12a48b0 failed and was restored: no serving line (last line: panicked at x.rs:1); \
             the app's rollout is paused"
        );
        let idle = json!({"step": "", "fetching": null, "stranded": null, "failure": null, "paused": false});
        assert!(upgrade_line(&member(4350, idle.clone()), Some(4381))
            .unwrap()
            .starts_with("below the floor, with no upgrade under way"));
        assert_eq!(upgrade_line(&member(4381, idle), Some(4381)), None);
        let no_report = json!({"build": {"count": 1, "hash": "h", "local": ""}, "coord": null});
        assert_eq!(upgrade_line(&no_report, None), None);
    }

    /// **A fault says why and since when; a member near its disk floor says
    /// how near.**
    #[test]
    fn a_fault_says_why_and_how_near() {
        let gib = 1024 * 1024 * 1024u64;
        let fault = json!({"app": "zend", "host": "ww", "node": "01", "free": gib, "floor": 2 * gib,
                           "unsent": 3, "fault": {"reason": "disk full", "since_ms": NOW - 120_000,
                           "claims_released": 1, "voting": true, "freed": 0}});
        assert_eq!(
            fault_line(&fault, NOW),
            "zend ww 01: in fault since 2 min ago — disk full; 1.0 GB free of a 2.0 GB floor, 3 \
             events unsent"
        );
        let near = json!({"app": "npcd", "host": "bw", "node": "02", "free": 3 * gib,
                          "floor": 2 * gib, "unsent": 0, "fault": null});
        assert_eq!(
            fault_line(&near, NOW),
            "npcd bw 02: near its disk floor — 3.0 GB free of a 2.0 GB floor, 0 events unsent"
        );
    }

    /// **The gateway's view carries each member's upgrade and each fault** in
    /// full, and the rollout's timing.
    #[test]
    fn the_gateways_view_carries_upgrades_and_faults() {
        let b = |count: u64| json!({"count": count, "hash": "h", "local": ""});
        let status = json!({
            "gateway": "bw", "build": b(1), "draining": false, "now_ms": NOW,
            "apps": [{"app": "zend", "service": false, "mode": "clustered", "ready": 2, "members": 2,
                      "floor": b(11), "pending_floor": null}],
            "members": [
                {"app": "zend", "host": "ww", "node": "01", "build": b(10), "state": "ready",
                 "syncing": false, "lagging": true, "silent": false,
                 "coord": {"mode": "joined", "upgrade": {"step": "waiting", "fetching": null,
                           "stranded": null, "failure": null, "paused": false}}}
            ],
            "upgrades": [{"app": "zend", "candidate": null, "deadline_ms": NOW + 600_000,
                          "rollout_since_ms": NOW - 120_000, "rejected": [], "certified": ["11-h"]}],
            "faults": [{"app": "zend", "host": "ww", "node": "01", "free": 0, "floor": 0,
                        "unsent": 0, "fault": null}]
        });
        let text = render_gateway(&status);
        assert!(
            text.contains("\n  zend ww 01 build 10-h: ready (lagging) — below the floor: waiting"),
            "{text}"
        );
        assert!(
            text.contains("\nupgrade zend: floor raised 2 min ago, members below it upgrading"),
            "{text}"
        );
        assert!(
            text.contains(
                "\nupgrade zend: older builds served until the grace period ends, in 10 min"
            ),
            "{text}"
        );
        assert!(
            text.contains("\nfault: zend ww 01: near its disk floor"),
            "{text}"
        );
    }

    /// A member's `/v1/status` body as it writes its apps' replication.
    fn member_body(zend_phase: &str, zend_link: &str) -> Value {
        json!({
            "state": "ready", "build": "0123456789abcdef",
            "detail": "", "started_at_ms": NOW - 60_000,
            "cluster": {"build": {"count": 4419, "hash": "6ecd6158", "local": ""}, "apps": {
                "npcd": {"link": "joined", "replication": {"phase": "joined", "generation": 64,
                         "unsent": 0, "catchups": []}},
                "zend": {"link": zend_link, "replication": {"phase": zend_phase, "generation": 64,
                         "unsent": 12, "catchups": [{"purpose": "Admission"}]}}
            }}
        })
    }

    /// **Each member says its apps' replication, and one not joined is a
    /// problem said first** — the state that tells a cut-off member from a
    /// joined one, which nothing else in the status shows.
    #[test]
    fn a_member_says_its_replication_and_a_cut_off_one_is_a_problem() {
        let envelope = json!({"members": [
            {"node": "a0ac88178cb2b57d", "host": "BlackWorld", "status": 200,
             "body": member_body("joined", "joined")},
            {"node": "893959beacadc372", "host": "WhiteWorld", "status": 200,
             "body": member_body("cut_off", "cut_off")}
        ]});
        assert_eq!(
            render_members(&envelope, NOW),
            "Members answering /v1/status: 2\n\
             problem: WhiteWorld 893959beacadc372: zend is cut_off, link cut_off — what it \
             writes reaches no peer until it rejoins\n\
             - BlackWorld a0ac88178cb2b57d build 4419-6ecd6158: ready, started 60 s ago; \
             replication: npcd joined gen 64, zend joined gen 64, 12 unsent, 1 catching up\n\
             - WhiteWorld 893959beacadc372 build 4419-6ecd6158: ready, started 60 s ago; \
             replication: npcd joined gen 64, zend cut_off gen 64 (link cut_off), 12 unsent, \
             1 catching up"
        );
    }

    /// **A machine with no member answering says why, from its keep**: a
    /// member coming up under a trial, a motherd running but not answering,
    /// or one stopped.
    #[test]
    fn a_machine_with_no_member_says_what_its_keep_runs() {
        let build = json!({"count": 4419, "hash": "6ecd6158", "local": ""});
        let state = |running: Value, trial: Value| {
            json!({"services": [{"name": "motherd", "running": running, "build": build,
                                 "trials": {"running": trial, "last": null}}]})
        };
        assert_eq!(
            absent_line(
                "WhiteWorld",
                &state(
                    json!(464),
                    json!({"artifact": "stage/motherd.exe", "started_ms": NOW - 51_000})
                ),
                NOW
            ),
            "- WhiteWorld: not answering — keep is trying it: 51 s in, of a 10-minute window to \
             print that it serves — the window restarts while it is catching up on replication \
             (installed 4419-6ecd6158)"
        );
        assert_eq!(
            absent_line("WhiteWorld", &state(json!(464), Value::Null), NOW),
            "- WhiteWorld: not answering — keep runs motherd 4419-6ecd6158 as pid 464, so it is \
             loading or stuck: read its lines with `log_grep` (a worker's `cluster_logs`)"
        );
        assert_eq!(
            absent_line("WhiteWorld", &state(Value::Null, Value::Null), NOW),
            "- WhiteWorld: not answering — keep has motherd 4419-6ecd6158 stopped"
        );
        assert_eq!(
            absent_line("AlienWorld", &json!({"services": []}), NOW),
            "- AlienWorld: not answering, and its keep runs no motherd"
        );
    }

    #[test]
    fn a_time_reads_as_how_long_ago() {
        assert_eq!(ago(NOW - 30_000, NOW), "30 s ago");
        assert_eq!(ago(NOW - 600_000, NOW), "10 min ago");
        assert_eq!(ago(NOW - 36_000_000, NOW), "10.0 h ago");
    }
}
