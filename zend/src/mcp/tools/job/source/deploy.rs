//! A deploy, as a job (`docs/jobs.md` §3, §6): its output is its build's log,
//! then a marker line, then the trial's log on keep's machine — the trial's
//! only once keep's board names the trial this deploy's. Each read is split
//! across the two and numbered as one.

use axum::http::Method;
use cluster::log_read::{Line, LogAsk, LogRead, MAX_LINES};
use serde_json::{json, Value};

use super::super::{shifted, JobHandle, Standing};
use super::build::{self, agents, summary_text};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::cluster_status::ago;
use crate::mcp::tools::repo_job::{lost, phase, span, Handle};
use crate::mcp::tools::service::{
    keep, resolve, Deploy, Stage, KEEP_CANCEL, KEEP_LOG, PROOF_WINDOW_MS,
};
use crate::mcp::tools::Answer;
use crate::mcp::workers::now_ms;

/// The line between a deploy's build and its trial.
fn marker(d: &Deploy) -> String {
    format!(
        "── the trial of {} on {} begins: keep's log on that machine ──",
        d.service, d.host
    )
}

/// Whether keep's board names a trial of this deploy, so its log is part of
/// the deploy's output.
fn tried(d: &Deploy) -> bool {
    matches!(d.stage, Stage::Trying(_) | Stage::Tried(_))
}

/// `ask` of the trial's log.
async fn trial(call: &OpsCall, d: &Deploy, ask: &LogAsk) -> Result<LogRead, String> {
    let body = json!({"service": d.service, "ask": ask});
    let v = keep(call, Method::POST, KEEP_LOG, &d.host, Some(&body)).await?;
    serde_json::from_value(v).map_err(|e| format!("keep on {}: {e}", d.host))
}

/// Where deploy `d` stands, in a line.
fn line(job: &JobHandle, d: &Deploy, now: u64) -> String {
    let head = format!("{job}: {} to {}", d.service, d.host);
    match &d.stage {
        Stage::Building => format!(
            "{head} — building: {}, for {}, {} lines",
            phase(&d.build.phase, d.handle.node),
            span(now.saturating_sub(d.build.started_ms)),
            d.build.lines
        ),
        Stage::BuildFailed(why) => format!("{head} — the build failed ({why}); nothing was tried"),
        Stage::Built { stale: false, .. } => {
            format!("{head} — built; keep's trial is being asked for")
        }
        Stage::Built { stale: true, .. } => format!(
            "{head} — built, but no trial was started: the member that was to ask keep for it is \
             gone; deploy again"
        ),
        Stage::Trying(t) => {
            let started = t.get("started_ms").and_then(Value::as_u64).unwrap_or(now);
            format!(
                "{head} — keep is trying it, {} into a {}-minute window{}",
                span(now.saturating_sub(started)),
                PROOF_WINDOW_MS / 60_000,
                if t.get("cancelled").and_then(Value::as_bool) == Some(true) {
                    ", cancelled and rolling back"
                } else {
                    ""
                }
            )
        }
        Stage::Tried(t) => {
            let when = ago(t.get("ended_ms").and_then(Value::as_u64).unwrap_or(0), now);
            if t.get("proven").and_then(Value::as_bool) == Some(true) {
                format!("{head} — proven {when}: {} runs it from now on", d.host)
            } else if t.get("cancelled").and_then(Value::as_bool) == Some(true) {
                format!("{head} — cancelled and rolled back {when}")
            } else {
                format!("{head} — rolled back {when}")
            }
        }
    }
}

pub async fn standing(call: &OpsCall, job: &JobHandle, h: &Handle) -> Result<Standing, String> {
    let agents = agents(call)?;
    let d = resolve(call, &agents, h, now_ms()).await?;
    let built = build::read(call, h, &LogAsk::Summary).await?;
    let mut total = built.total;
    let mut size = built.size;
    let mut detail = String::new();
    match &d.stage {
        Stage::BuildFailed(_) => {
            if let Some(s) = &built.summary {
                detail = summary_text(job, s);
            }
        }
        Stage::Tried(t) => {
            let trial_read = shifted(trial(call, &d, &LogAsk::Summary).await?, total + 1);
            total = trial_read.total;
            size += trial_read.size;
            let why = t
                .get("detail")
                .and_then(Value::as_str)
                .unwrap_or("")
                .replace('\n', " / ");
            detail = format!("\nkeep: {why}");
            if let Some(s) = &trial_read.summary {
                detail.push_str(&summary_text(job, s));
            }
            if t.get("proven").and_then(Value::as_bool) == Some(true) && d.service == "motherd" {
                detail.push_str(
                    "\nA proven motherd joins the cluster as a candidate; the other members take \
                     it once it is certified, which is the user's decision.",
                );
            }
        }
        Stage::Trying(_) => {
            let trial_read = shifted(
                trial(call, &d, &LogAsk::Tail { lines: 1 }).await?,
                total + 1,
            );
            total = trial_read.total;
            size += trial_read.size;
            if let Some(l) = trial_read.lines.last() {
                detail = format!("\nthe trial's last line {}: {}", l.n, l.text);
            }
        }
        Stage::Building | Stage::Built { .. } => {
            if let Some(l) = built.summary.as_ref().and_then(|s| s.last.last()) {
                detail = format!("\nlast line {}: {}", l.n, l.text);
            }
        }
    }
    Ok(Standing {
        ended: d.stage.is_over(),
        line: line(job, &d, now_ms()),
        detail,
        total,
        size,
        marks: Vec::new(),
    })
}

/// Lines `from..from+count` of the whole: the build's part, the marker, then
/// the trial's part.
async fn range(
    call: &OpsCall,
    d: &Deploy,
    h: &Handle,
    from: u64,
    count: u32,
) -> Result<LogRead, String> {
    let to = from + u64::from(count);
    let built = build::read(call, h, &LogAsk::Lines { from, count }).await?;
    let b = built.total;
    let mut lines = built.lines;
    let mut total = b;
    if tried(d) {
        if (from..to).contains(&(b + 1)) {
            lines.push(Line {
                n: b + 1,
                text: marker(d),
            });
        }
        let start = from.max(b + 2);
        if start < to {
            let t = trial(
                call,
                d,
                &LogAsk::Lines {
                    from: start - (b + 1),
                    count: (to - start) as u32,
                },
            )
            .await?;
            let t = shifted(t, b + 1);
            total = t.total;
            lines.extend(t.lines);
        } else {
            let t = trial(call, d, &LogAsk::Tail { lines: 1 }).await?;
            total = b + 1 + t.total;
        }
    }
    Ok(LogRead {
        total,
        lines,
        next: (to <= total).then_some(to),
        ended: d.stage.is_over(),
        ..LogRead::default()
    })
}

pub async fn read(call: &OpsCall, h: &Handle, ask: &LogAsk) -> Result<LogRead, String> {
    let agents = agents(call)?;
    let d = resolve(call, &agents, h, now_ms()).await?;
    match ask {
        LogAsk::Lines { from, count } => range(call, &d, h, *from, *count).await,
        LogAsk::Summary | LogAsk::Tail { .. } if tried(&d) => {
            let b = build::read(call, h, &LogAsk::Tail { lines: 1 })
                .await?
                .total;
            Ok(shifted(trial(call, &d, ask).await?, b + 1))
        }
        LogAsk::Summary | LogAsk::Tail { .. } => build::read(call, h, ask).await,
        LogAsk::Grep { .. } => {
            let built = build::read(call, h, ask).await?;
            if !tried(&d) {
                return Ok(built);
            }
            let b = built.total;
            let t = shifted(trial(call, &d, ask).await?, b + 1);
            let room = (MAX_LINES as usize).saturating_sub(built.lines.len());
            let mut lines = built.lines;
            let kept = t.lines.len().min(room);
            lines.extend(t.lines.into_iter().take(kept));
            Ok(LogRead {
                total: t.total,
                lines,
                more: built.more + t.more,
                next: built.next.or(t.next),
                ..LogRead::default()
            })
        }
    }
}

pub async fn cancel(call: &OpsCall, job: &JobHandle, h: &Handle) -> Answer {
    let agents = match agents(call) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    let d = match resolve(call, &agents, h, now_ms()).await {
        Ok(d) => d,
        Err(e) => return Answer::error(e),
    };
    match &d.stage {
        Stage::Building => match agents.cancel(h.at(&agents), &h.id).await {
            Ok(_) => Answer::ok(format!(
                "Cancelled {job}: its build of {} on {} is stopped, and nothing will be tried.",
                d.service, d.host
            )),
            Err(e) => Answer::error(lost(h, &e)),
        },
        Stage::Trying(_) => {
            let body = json!({"service": d.service});
            match keep(call, Method::POST, KEEP_CANCEL, &d.host, Some(&body)).await {
                Ok(_) => Answer::ok(format!(
                    "Cancelled {job}: keep on {} is stopping the trial of {} and putting the \
                     previous binary back. `job_status {{jobs: [\"{job}\"], wait_s: 60}}` says \
                     when it is done.",
                    d.host, d.service
                )),
                Err(e) => Answer::error(e),
            }
        }
        Stage::Built { stale: false, .. } => Answer::error(format!(
            "{job} has just built, and keep is being asked for its trial: there is nothing to stop \
             for a moment. Call again in a few seconds."
        )),
        _ => Answer::ok(format!(
            "Nothing to cancel: it has ended.\n{}",
            line(job, &d, now_ms())
        )),
    }
}
