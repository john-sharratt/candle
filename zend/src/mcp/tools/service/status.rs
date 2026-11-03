//! `service_status {deploy?, host?, wait_s?}`: where deploys are
//! (`docs/build_deploy_tools.md` §4). With `deploy`, that deploy's stage —
//! building, built, being tried, proven, rolled back — waiting up to `wait_s`
//! for it to end, each step said on the call's progress. Without, every
//! machine's services as its keep reports them, with the deploys building
//! there and any trial running.

use std::time::{Duration, Instant};

use agent::client::Agents;
use axum::http::Method;
use cluster::log_read::{LogAsk, LogRead};
use serde_json::{json, Value};

use super::{built_service, keep, resolve, Deploy, Stage, KEEP_LOG, KEEP_STATE, PROOF_WINDOW_MS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::cluster_status::ago;
use crate::mcp::tools::repo_job::{phase, place, span, Handle, NO_AGENTS};
use crate::mcp::tools::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "service_status";

/// The seconds a call on a deploy waits when it names no `wait_s`.
const DEFAULT_WAIT_S: u64 = 60;

/// How often a waiting call reads the deploy again.
const POLL: Duration = Duration::from_secs(5);

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Say where deploys are. With `deploy` — the handle `service_deploy` answered — that \
            deploy's stage: building (its phase, time and lines), build failed, built and staged, being \
            tried by keep (how far into the ten-minute proof window, and the trial log's last line), or \
            ended — proven and now installed, rolled back with keep's reason, or cancelled — and it waits up \
            to `wait_s` seconds (60 by default; 0 answers at once) for the deploy to end. Waiting is one \
            call that holds until it ends or the wait runs out: call it once with a long `wait_s`, not again \
            and again. Without `deploy`, every machine's services — or `host`'s alone — as its keep reports \
            them: each service's installed build, whether it runs, the trial running and how the last one \
            ended, with the deploy each belongs to, and the deploys still building there. Read it before a \
            deploy to see what each machine runs and that nothing else is deploying there. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deploy": {"type": "string", "minLength": 18, "maxLength": 64, "description": "A deploy's handle, `<node>:<id>`, as `service_deploy` answered it."},
                "host": {"type": "string", "minLength": 1, "maxLength": 32, "description": "Without `deploy`: one machine only, by name. Every machine when left out."},
                "wait_s": {"type": "integer", "minimum": 0, "maximum": 3600, "description": "With `deploy`: seconds to wait for it to end; 60 by default, 0 to answer at once."}
            },
            "additionalProperties": false
        }
    })
}

/// What `trial`, a trial on keep's board, says of how it ended.
fn ended(trial: &Value, now: u64) -> String {
    let when = ago(
        trial.get("ended_ms").and_then(Value::as_u64).unwrap_or(0),
        now,
    );
    let detail = trial
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("")
        .replace('\n', " / ");
    if trial.get("proven").and_then(Value::as_bool) == Some(true) {
        format!("proven {when}")
    } else if trial.get("cancelled").and_then(Value::as_bool) == Some(true) {
        format!("cancelled and rolled back {when} — {detail}")
    } else {
        format!("rolled back {when} — {detail}")
    }
}

/// How far into its proof window `trial`, running on keep's board, is.
fn trying(trial: &Value, now: u64) -> String {
    let started = trial
        .get("started_ms")
        .and_then(Value::as_u64)
        .unwrap_or(now);
    format!(
        "keep is trying it: {} in, of a {}-minute window to print that it serves — the window \
         restarts while it is catching up on replication{}",
        span(now.saturating_sub(started)),
        PROOF_WINDOW_MS / 60_000,
        if trial.get("cancelled").and_then(Value::as_bool) == Some(true) {
            "; cancelled, rolling back"
        } else {
            ""
        }
    )
}

/// A deploy as an answer: where it is, and what to call next.
pub fn render_deploy(d: &Deploy, now: u64, last_trial_line: Option<&str>) -> String {
    let h = &d.handle;
    let (host, service) = (d.host.as_str(), d.service);
    let head = format!("deploy {h}: {service} to {host}");
    match &d.stage {
        Stage::Building => format!(
            "{head} — building: {}, for {}, {} lines.\nThe build's log: `repo_job_log {{job: \
             \"{h}\", mode: \"tail\"}}`.",
            phase(&d.build.phase, h.node),
            span(now.saturating_sub(d.build.started_ms)),
            d.build.lines
        ),
        Stage::BuildFailed(why) => format!(
            "{head} — the build failed ({why}); nothing was tried, and {host} runs what it ran.\nWhy: \
             `repo_job_log {{job: \"{h}\", mode: \"summary\"}}`."
        ),
        Stage::Built {
            artifact,
            stale: false,
        } => format!("{head} — built and staged at {artifact}; keep's trial of it is being asked for."),
        Stage::Built {
            artifact,
            stale: true,
        } => format!(
            "{head} — built and staged at {artifact} {}, but no trial of it was started: the member \
             that was to ask keep for it is gone. Deploy again with `service_deploy {{host: \
             \"{host}\", service: \"{service}\"}}`.",
            ago(d.build.ended_ms.unwrap_or(now), now)
        ),
        Stage::Trying(trial) => {
            let mut out = format!("{head} — {}", trying(trial, now));
            if let Some(line) = last_trial_line {
                out.push_str(&format!("\nits last line: {line}"));
            }
            out.push_str(&format!(
                "\nIts log: `service_log {{host: \"{host}\", service: \"{service}\", mode: \
                 \"tail\"}}`."
            ));
            out
        }
        Stage::Tried(trial) => {
            let how = ended(trial, now);
            let mut out = format!("{head} — {how}.");
            if trial.get("proven").and_then(Value::as_bool) == Some(true) {
                out.push_str(&format!(
                    "\n{host} runs it as {service} from now on.{}",
                    if service == "motherd" {
                        " It joins the cluster as a candidate; the other members take it once it is \
                         certified, which is the user's decision."
                    } else {
                        ""
                    }
                ));
            } else {
                out.push_str(&format!(
                    "\n{host} runs the build it ran before. Why: `service_log {{host: \"{host}\", \
                     service: \"{service}\", mode: \"summary\"}}`."
                ));
            }
            out
        }
    }
}

/// A keep's services, one block each, naming the deploy each trial belongs to.
pub fn render_keep(host: &str, state: &Value, now: u64) -> String {
    let mut out = format!("keep on {host}:");
    for s in state
        .get("services")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = s.get("name").and_then(Value::as_str).unwrap_or("?");
        let build = match s.get("build").filter(|b| !b.is_null()) {
            Some(b) => {
                let local = b.get("local").and_then(Value::as_str).unwrap_or("");
                format!(
                    "{}-{}{}",
                    b.get("count").map_or_else(|| "?".into(), Value::to_string),
                    b.get("hash").and_then(Value::as_str).unwrap_or("?"),
                    if local.is_empty() {
                        String::new()
                    } else {
                        format!(" (local {local})")
                    }
                )
            }
            None => "unreadable".into(),
        };
        let running = match s.get("running").and_then(Value::as_u64) {
            Some(pid) => format!("running as pid {pid}"),
            None => "not running".into(),
        };
        out.push_str(&format!("\n- {name}: build {build}, {running}"));
        let of = |t: &Value| {
            let artifact = t.get("artifact").and_then(Value::as_str).unwrap_or("?");
            match t.get("deploy").and_then(Value::as_str) {
                Some(d) => format!("deploy {d} ({artifact})"),
                None => artifact.to_string(),
            }
        };
        let trials = s.get("trials").cloned().unwrap_or(Value::Null);
        if let Some(t) = trials.get("running").filter(|t| !t.is_null()) {
            out.push_str(&format!("\n  trial of {} — {}", of(t), trying(t, now)));
        }
        if let Some(t) = trials.get("last").filter(|t| !t.is_null()) {
            out.push_str(&format!("\n  last trial, of {}: {}", of(t), ended(t, now)));
        }
    }
    out
}

/// The deploys building on agent `node`, a line each.
async fn building_on(agents: &Agents, node: u64, now: u64) -> Vec<String> {
    let Ok(jobs) = agents.jobs_of(agents.at(node), 40).await else {
        return Vec::new();
    };
    jobs.jobs
        .iter()
        .filter(|j| !j.status.is_finished())
        .filter_map(|j| {
            let service = built_service(&j.command)?;
            Some(format!(
                "  deploy {} of {service} building: {}, for {}",
                Handle::new(node, &j.id),
                phase(&j.phase, node),
                span(now.saturating_sub(j.started_ms))
            ))
        })
        .collect()
}

/// The last line of `service`'s trial log on `host`, numbered.
async fn last_trial_line(call: &OpsCall, host: &str, service: &str) -> Option<String> {
    let body = json!({"service": service, "ask": LogAsk::Tail { lines: 1 }});
    let read = keep(call, Method::POST, KEEP_LOG, host, Some(&body))
        .await
        .ok()?;
    let read: LogRead = serde_json::from_value(read).ok()?;
    read.lines.last().map(|l| format!("{}: {}", l.n, l.text))
}

/// Every machine's keep, or `only`'s, with the deploys building there.
async fn overview(call: &OpsCall, agents: &Agents, only: Option<&str>) -> Answer {
    let listed = match agents.listing().await {
        Ok(l) => l,
        Err(e) => return Answer::error(format!("the agents could not be listed: {e}")),
    };
    let mut hosts: Vec<(String, u64)> = Vec::new();
    for a in &listed {
        let Ok(node) = u64::from_str_radix(&a.node, 16) else {
            continue;
        };
        if only.is_some_and(|o| !o.trim().eq_ignore_ascii_case(&a.host)) {
            continue;
        }
        if !hosts.iter().any(|(h, _)| h.eq_ignore_ascii_case(&a.host)) {
            hosts.push((a.host.clone(), node));
        }
    }
    if hosts.is_empty() {
        return Answer::error(match only {
            Some(h) => {
                format!("no agent is connected on {h}, so its keep is not reached through one")
            }
            None => "no agent is connected".to_string(),
        });
    }
    let now = now_ms();
    let mut blocks = Vec::new();
    for (host, node) in hosts {
        let mut block = match keep(call, Method::GET, KEEP_STATE, &host, None).await {
            Ok(state) => render_keep(&host, &state, now),
            Err(e) => e,
        };
        let building = building_on(agents, node, now).await;
        if !building.is_empty() {
            block.push_str(&format!(
                "\n- deploys building on its agent ({}):\n",
                place(node)
            ));
            block.push_str(&building.join("\n"));
        }
        blocks.push(block);
    }
    Answer::ok(blocks.join("\n"))
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let Some(deploy) = args.get("deploy").and_then(Value::as_str) else {
        let host = string(args, "host").ok();
        return overview(call, &agents, host).await;
    };
    let h = match Handle::parse(deploy) {
        Ok(h) => h,
        Err(e) => return Answer::error(e.replace("not a job", "not a deploy")),
    };
    let (secs, asked) = call.wait_of(args, DEFAULT_WAIT_S);
    let until = Instant::now() + Duration::from_secs(secs);
    let mut d = match resolve(call, &agents, &h, now_ms()).await {
        Ok(d) => d,
        Err(e) => return Answer::error(e),
    };
    let mut said = String::new();
    while !d.stage.is_over() && Instant::now() < until {
        let line = render_deploy(&d, now_ms(), None);
        let first = line.lines().next().unwrap_or_default().to_string();
        if first != said {
            call.say(first.clone());
            said = first;
        }
        tokio::time::sleep(POLL.min(until.saturating_duration_since(Instant::now()))).await;
        d = match resolve(call, &agents, &h, now_ms()).await {
            Ok(d) => d,
            Err(e) => return Answer::error(e),
        };
    }
    let last = match d.stage {
        Stage::Trying(_) => last_trial_line(call, &d.host, d.service).await,
        _ => None,
    };
    let mut out = render_deploy(&d, now_ms(), last.as_deref());
    if !d.stage.is_over() {
        let cut = asked.map_or(String::new(), |a| {
            format!(" (you asked for {a} s; this client allows {secs})")
        });
        out.push_str(&format!(
            "\nStill going after {secs} s{cut}: `service_status {{deploy: \"{h}\", wait_s: {}}}` \
             keeps waiting.",
            call.ceiling.min(600)
        ));
    }
    Answer::ok(out)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use agent::wire::{JobView, Phase};
    use zend_vfs::JobStatus;

    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    fn deploy(stage: Stage) -> Deploy {
        Deploy {
            handle: Handle::new(0xab, "Xe"),
            host: "BlackWorld".into(),
            service: "motherd",
            build: JobView {
                id: "Xe".into(),
                repo: "candle".into(),
                status: JobStatus::Running,
                lines: 812,
                bytes: 0,
                outcome: None,
                pushed: Vec::new(),
                fetched: None,
                listed: None,
                staged: None,
                phase: Phase::Running,
                command: Vec::new(),
                started_ms: NOW - 125_000,
                ended_ms: Some(NOW - 90_000),
            },
            stage,
        }
    }

    /// **Each stage of a deploy reads as where it is and what to call next.**
    #[test]
    fn a_deploy_reads_as_its_stage() {
        assert_eq!(
            render_deploy(&deploy(Stage::Building), NOW, None),
            "deploy 00000000000000ab:Xe: motherd to BlackWorld — building: running, for 2 min 5 s, \
             812 lines.\nThe build's log: `repo_job_log {job: \"00000000000000ab:Xe\", mode: \
             \"tail\"}`."
        );
        let trying =
            json!({"artifact": "a", "deploy": "00000000000000ab:Xe", "started_ms": NOW - 61_000});
        let text = render_deploy(
            &deploy(Stage::Trying(trying)),
            NOW,
            Some("40: syncing 3 records"),
        );
        assert!(
            text.contains("keep is trying it: 1 min 1 s in, of a 10-minute window"),
            "{text}"
        );
        assert!(
            text.contains("\nits last line: 40: syncing 3 records\n"),
            "{text}"
        );
        let proven = json!({"proven": true, "ended_ms": NOW - 30_000, "detail": "proven"});
        let text = render_deploy(&deploy(Stage::Tried(proven)), NOW, None);
        assert!(
            text.starts_with(
                "deploy 00000000000000ab:Xe: motherd to BlackWorld — proven 30 s ago."
            ),
            "{text}"
        );
        assert!(
            text.contains("certified, which is the user's decision"),
            "{text}"
        );
        let back = json!({"proven": false, "cancelled": false, "ended_ms": NOW - 30_000,
                          "detail": "rolled back\nno proof line within the limit"});
        let text = render_deploy(&deploy(Stage::Tried(back)), NOW, None);
        assert!(
            text.contains("rolled back 30 s ago — rolled back / no proof line within the limit."),
            "{text}"
        );
        assert!(text.contains("mode: \"summary\"}`"), "{text}");
        let stale = deploy(Stage::Built {
            artifact: "D:/stage/motherd.exe".into(),
            stale: true,
        });
        assert!(render_deploy(&stale, NOW, None).contains("no trial of it was started"));
    }

    /// **A keep's services name the deploy each trial belongs to**, and a
    /// cancelled one says so.
    #[test]
    fn a_keep_names_the_deploy_of_each_trial() {
        let state = json!({"services": [
            {"name": "motherd", "running": 4242,
             "build": {"count": 5739, "hash": "c544", "local": ""},
             "trials": {"running": null,
                        "last": {"artifact": "stage/motherd.exe", "deploy": "00000000000000ab:Xe",
                                 "proven": false, "cancelled": true,
                                 "detail": "rolled back\nthe trial was cancelled", "ended_ms": NOW - 120_000}}},
            {"name": "web", "running": null, "build": null,
             "trials": {"running": {"artifact": "stage/web.exe", "deploy": null, "started_ms": NOW - 30_000},
                        "last": null}}
        ]});
        let text = render_keep("BlackWorld", &state, NOW);
        assert!(text.starts_with("keep on BlackWorld:\n- motherd: build 5739-c544, running as pid 4242\n  last trial, of deploy 00000000000000ab:Xe (stage/motherd.exe): cancelled and rolled back 2 min ago"), "{text}");
        assert!(text.contains("\n- web: build unreadable, not running\n  trial of stage/web.exe — keep is trying it: 30 s in"), "{text}");
    }
}
