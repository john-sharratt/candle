//! The task acts and their tending, where the world runs
//! (`docs/mission_templates.md` §7): a drone's `task_start`, `task_reply` and
//! `task_nudge` become `repo_task_start`, `repo_task_reply` and
//! `repo_task_nudge` on zend's `/v1/mcp`, sent up this member's cluster socket
//! as Zen — the identity `service:zen` (`docs/mcp_file_tools.md` §4), which the
//! member states and the gateway believes from a member's socket, as zend's own
//! calls to npcd are made. Every [`POLL_EVERY`], each carried mission's task is
//! asked how it stands (`job_status`, `wait_s: 0`), off the decode path; what
//! changed is posted on its carrier's handset from [`FROM`], and a task that
//! reports done signs its mission's task step off.
//!
//! **A suspended carrier is held still here too**: its acts are refused, and
//! its task is neither asked after nor posted to until it is let go.

use std::sync::Arc;
use std::time::Duration;

use axum::http::{header, HeaderName, HeaderValue, Method};
use cluster::member::AppMember;
use host::cluster::up::send_up;
use mcp_server::http::VERSION_HEADER;
use mcp_server::protocol::MODERN;
use serde_json::Value;
use tokio::time::{interval, MissedTickBehavior};
use web::auth::forwarded::{PROVIDER, USER};
use web::cluster::service_key::{subject, SERVICE_PROVIDER};

use crate::engine::act::Act;
use crate::engine::runtime::{Recorded, Runtime, LANDED, REFUSED};
use crate::engine::zen_task::request::{
    answer_of, digest, handle_in, news, rpc, standing_of, status_call, thread_for, zend_call,
    Standing, ZEND_MCP,
};
use crate::engine::zen_task::{task_grants, Task};

/// The service a drone's task is Zen's as: `service:zen`.
pub const ZEN_SERVICE: &str = "zen";

/// Who a task's news comes from on the carrier's handset: Zen's coding
/// service, kept apart from the moon's character Zen, whose thread with the
/// carrier is its own.
pub const FROM: &str = "Zen Code";

/// How often each carried mission's task is asked how it stands.
pub const POLL_EVERY: Duration = Duration::from_secs(15);

/// How long one call to zend may take.
const ANSWER_WITHIN: Duration = Duration::from_secs(30);

/// The headers a call to zend carries: Zen's identity as a member states it,
/// no email, and the 2026-07-28 protocol, which keeps no session to open.
pub fn zen_headers() -> Vec<(HeaderName, HeaderValue)> {
    let sub = subject(ZEN_SERVICE);
    [
        (USER, sub.as_str()),
        (PROVIDER, SERVICE_PROVIDER),
        (VERSION_HEADER, MODERN),
        (
            header::ACCEPT.as_str(),
            "application/json, text/event-stream",
        ),
        (header::CONTENT_TYPE.as_str(), "application/json"),
    ]
    .into_iter()
    .filter_map(|(k, v)| {
        Some((
            HeaderName::from_bytes(k.as_bytes()).ok()?,
            HeaderValue::from_str(v).ok()?,
        ))
    })
    .collect()
}

/// `tool` with `args` called on zend as Zen, on behalf of `thread`: the
/// answer's text, or why there is none.
async fn call_zend(
    member: &AppMember,
    tool: &str,
    args: &Value,
    thread: &str,
) -> Result<String, String> {
    let body = rpc(tool, args, thread).to_string().into_bytes();
    let reply = send_up(
        member,
        Method::POST,
        ZEND_MCP,
        &zen_headers(),
        body,
        Some(ANSWER_WITHIN),
    )
    .await
    .map_err(|e| format!("Zen could not be reached: {e:#}"))?;
    answer_of(&reply.body)
}

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

impl Runtime {
    /// A task act made for `npc`: the zend call it stands for, as Zen.
    pub(crate) async fn zen_task_act(&self, npc: u64, act: &Act) -> Recorded {
        if self.scheduler.is_suspended(npc) {
            return recorded(
                act,
                false,
                "You are held still: nothing you do reaches anyone.".into(),
            );
        }
        let Some((hosted, body)) = self.body_of(npc) else {
            return recorded(act, false, "You have no body here to do that with.".into());
        };
        if !task_grants(&hosted, &body).iter().any(|g| g == act.tool) {
            return recorded(
                act,
                false,
                "Nothing you carry has work for Zen's workers: a task is driven only on a \
                 mission that says so."
                    .into(),
            );
        }
        let Some(member) = self.standing.member().cloned() else {
            return recorded(
                act,
                false,
                "Zen cannot be reached from here: this house is on no cluster.".into(),
            );
        };
        let (chapter, handle) = hosted.sim(|s| {
            (
                s.missions.chapter(&body),
                s.missions
                    .active(&body)
                    .and_then(|m| m.task.as_ref().map(|t| t.handle.clone())),
            )
        });
        let thread = thread_for(npc, chapter);
        let (tool, args) = match zend_call(act.tool, &act.args, &thread, handle.as_deref()) {
            Ok(call) => call,
            Err(why) => return recorded(act, false, why),
        };
        let text = match call_zend(&member, tool, &args, &thread).await {
            Ok(text) => text,
            Err(why) => return recorded(act, false, format!("Zen refused it: {why}")),
        };
        if act.tool == "task_start" {
            let Some(handle) = handle_in(&text, &thread) else {
                return recorded(act, false, format!("Zen did not start it: {text}"));
            };
            hosted.with_sim(|s| {
                s.missions.set_task(&body, Task { handle, heard: 0 });
            });
        }
        let line = text.lines().next().unwrap_or_default().to_string();
        hosted.with_sim(|s| s.missions.observe(&body, &line));
        recorded(act, true, text)
    }

    /// Ask after every carried mission's task in the worlds this member
    /// writes, and pass on what changed.
    pub async fn tend_zen_tasks(&self, member: &AppMember) {
        for world in self.hosted.ids() {
            if !self.standing.writes(&world) {
                continue;
            }
            let Some(hosted) = self.hosted.get(&world) else {
                continue;
            };
            for (npc, body) in self.bodies.in_world(&world) {
                if self.scheduler.is_suspended(npc) {
                    continue;
                }
                let carried = hosted.sim(|s| {
                    s.missions.active(&body).and_then(|m| {
                        m.task
                            .as_ref()
                            .map(|t| (t.handle.clone(), s.missions.chapter(&body)))
                    })
                });
                let Some((handle, chapter)) = carried else {
                    continue;
                };
                let thread = thread_for(npc, chapter);
                let (tool, args) = status_call(&handle);
                let text = match call_zend(member, tool, &args, &thread).await {
                    Ok(text) => text,
                    Err(why) => {
                        tracing::warn!(world, npc, %handle, "a mission's task could not be asked after: {why}");
                        continue;
                    }
                };
                let standing = standing_of(&text, &handle);
                hosted.with_sim(|s| {
                    if s.missions.heard(&body, digest(&text)) {
                        let id = s.threads.reach(&body, FROM);
                        if let Err(why) = s.threads.send_on(&id, FROM, &news(&text), None) {
                            tracing::warn!(npc, "a task's news could not be posted: {why}");
                        }
                    }
                    match standing {
                        Standing::Done => {
                            if s.missions.task_done(&body) {
                                s.missions.observe(&body, format!("{handle} reported done"));
                            }
                        }
                        Standing::Failed => {
                            s.missions.observe(&body, format!("{handle} failed"));
                        }
                        Standing::Running | Standing::Asking | Standing::Stuck => {}
                    }
                });
            }
        }
    }
}

/// Tend the carried missions' tasks for `member` until the process stops.
pub async fn run(rt: Arc<Runtime>, member: Arc<AppMember>) {
    let mut tick = interval(POLL_EVERY);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        if rt.stopping() {
            return;
        }
        rt.tend_zen_tasks(&member).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A drone calls zend as Zen**: `service:zen` from the service-key
    /// provider, no email, in the protocol that keeps no session.
    #[test]
    fn a_drone_calls_as_zen() {
        let h = zen_headers();
        let get = |name: &str| {
            h.iter()
                .find(|(k, _)| k.as_str() == name)
                .map(|(_, v)| v.to_str().unwrap().to_string())
        };
        assert_eq!(get("x-tokera-user").as_deref(), Some("service:zen"));
        assert_eq!(get("x-tokera-provider").as_deref(), Some("service-key"));
        assert_eq!(get("x-tokera-email"), None);
        assert_eq!(get("mcp-protocol-version").as_deref(), Some(MODERN));
    }
}
