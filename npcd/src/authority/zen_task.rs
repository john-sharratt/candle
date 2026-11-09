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

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{header, HeaderName, HeaderValue, Method};
use cluster::member::AppMember;
use futures::stream::{self, StreamExt};
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
    answer_of, digest, handle_in, news, news_of, rpc, standing_of, status_call, thread_for,
    zend_call, Standing, ZEND_MCP,
};
use crate::engine::zen_task::{task_grants, Task};
use crate::sim::Sim;

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
    ///
    /// **The tasks are asked after together**, [`AT_ONCE`] at a time: one call
    /// may take [`ANSWER_WITHIN`], and asked one after another a single slow
    /// zend held every other task's news behind it. Each answer is taken in
    /// as it comes — news is posted in the order the answers land — after
    /// looking again whether this member still writes the world and the
    /// carrier is still let go: either may have changed while it was asked.
    pub async fn tend_zen_tasks(&self, member: &AppMember) {
        let asks = self.carried_tasks();
        tend(
            asks,
            AT_ONCE,
            |ask: &Carried| {
                let (tool, args) = status_call(&ask.handle);
                let thread = ask.thread.clone();
                async move { call_zend(member, tool, &args, &thread).await }
            },
            |ask, answer| match answer {
                Ok(text) => self.take_news(&ask, &text),
                Err(why) => tracing::warn!(
                    world = ask.world,
                    npc = ask.npc,
                    handle = %ask.handle,
                    "a mission's task could not be asked after: {why}"
                ),
            },
        )
        .await;
    }

    /// Every task carried in a world this member writes, by a carrier let go.
    fn carried_tasks(&self) -> Vec<Carried> {
        let mut asks = Vec::new();
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
                if let Some((handle, chapter)) = carried {
                    asks.push(Carried {
                        world: world.clone(),
                        npc,
                        body,
                        thread: thread_for(npc, chapter),
                        handle,
                    });
                }
            }
        }
        asks
    }

    /// Take in what `ask`'s task said, `text`, if this member still writes its
    /// world and its carrier is still let go.
    fn take_news(&self, ask: &Carried, text: &str) {
        if !self.standing.writes(&ask.world) || self.scheduler.is_suspended(ask.npc) {
            return;
        }
        let Some(hosted) = self.hosted.get(&ask.world) else {
            return;
        };
        hosted.with_sim(|s| post_news(s, &ask.body, ask.npc, &ask.handle, text));
    }
}

/// How many carried tasks are asked after at once.
const AT_ONCE: usize = 4;

/// A task carried by a mission, to be asked after.
struct Carried {
    world: String,
    npc: u64,
    body: String,
    thread: String,
    handle: String,
}

/// Ask after each of `asks` through `call`, at most `at_once` in flight, and
/// hand each answer to `apply` as it lands.
async fn tend<A, T, Fut>(
    asks: Vec<A>,
    at_once: usize,
    call: impl Fn(&A) -> Fut,
    mut apply: impl FnMut(A, T),
) where
    Fut: Future<Output = T>,
{
    let mut answers = stream::iter(asks.into_iter().map(|ask| {
        let asked = call(&ask);
        async move { (ask, asked.await) }
    }))
    .buffer_unordered(at_once.max(1));
    while let Some((ask, answer)) = answers.next().await {
        apply(ask, answer);
    }
}

/// What a task's `text` says, taken into `s` for its carrier `body`: its
/// step signed off when it reports done, and news of it posted on the
/// carrier's handset from [`FROM`] once — and only news, never the counters
/// of its running turn, which change on every look.
fn post_news(s: &mut Sim, body: &str, npc: u64, handle: &str, text: &str) {
    let standing = standing_of(text, handle);
    let said = news_of(text, handle);
    let said_digest = digest(&said);
    if standing == Standing::Done && s.missions.task_done(body) {
        s.missions.observe(body, format!("{handle} reported done"));
    }
    if !s.missions.is_news(body, said_digest) {
        return;
    }
    let id = s.threads.reach(body, FROM);
    // Noted as heard only once it is posted: a post that fails is tried again
    // on the next look.
    if let Err(why) = s.threads.send_on(&id, FROM, &news(&said), None) {
        tracing::warn!(npc, "a task's news could not be posted: {why}");
        return;
    }
    s.missions.heard(body, said_digest);
    match standing {
        Standing::Failed => {
            s.missions.observe(body, format!("{handle} failed"));
        }
        Standing::Stuck => s.missions.observe(
            body,
            format!("{handle} is stuck: it waits for `task_reply` to say how to go on"),
        ),
        Standing::Asking => s.missions.observe(
            body,
            format!("{handle} asks a question: answer it with `task_reply`"),
        ),
        Standing::Running | Standing::Done => {}
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
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use npc_map::world::Where;
    use tempfile::TempDir;
    use tokio::time::{sleep, Instant};

    use super::*;
    use crate::authority::Role;
    use crate::engine::mission::{Mission, Origin, Todo};
    use crate::engine::zen_task::TASK_STEP;
    use crate::mind::Mind;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";
    const HANDLE: &str = "task:mission-1-0/fix-door";

    /// **Eight tasks are asked after four at a time**: each call taking 30 s,
    /// half land at 30 s and half at 60 s, and never more than four are in
    /// flight at once.
    #[tokio::test(start_paused = true)]
    async fn tasks_are_asked_after_four_at_a_time() {
        let in_flight = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let started = Instant::now();
        let mut landed = Vec::new();
        tend(
            (0..8).collect::<Vec<u32>>(),
            4,
            |_| async {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                sleep(Duration::from_secs(30)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
            },
            |ask, ()| landed.push((ask, started.elapsed().as_secs())),
        )
        .await;
        assert_eq!(most.load(Ordering::SeqCst), 4);
        let at = |secs| landed.iter().filter(|(_, t)| *t == secs).count();
        assert_eq!((at(30), at(60), landed.len()), (4, 4, 8));
    }

    fn coding() -> Mission {
        Mission::new(
            "Fix the door sensor's build.",
            vec![Todo::new("start the task"), Todo::new(TASK_STEP)],
            Origin::Standing {
                id: "fix-door".into(),
                room: "vault-casting/band-one".into(),
            },
        )
        .granting(vec!["task_start".into(), "task_reply".into()])
    }

    /// A daemon hosting the vault, its drone carrying a task.
    fn carrying() -> (TempDir, Arc<Runtime>) {
        let dir = TempDir::new().unwrap();
        let rt = Runtime::new(Mind::new(None), dir.path()).unwrap();
        let hosted = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        hosted.with(|w| {
            w.enter("drone", "Wick", Where::new("vault-casting", "green-room"))
                .unwrap()
        });
        hosted.with_sim(|s| {
            s.missions.assign("drone", coding());
            s.missions.set_task(
                "drone",
                Task {
                    handle: HANDLE.into(),
                    heard: 0,
                },
            );
        });
        (dir, rt)
    }

    fn ask() -> Carried {
        Carried {
            world: WORLD.into(),
            npc: 7,
            body: "drone".into(),
            thread: "t".into(),
            handle: HANDLE.into(),
        }
    }

    /// The lines on the carrier's thread from Zen Code.
    fn posted(rt: &Runtime) -> usize {
        rt.hosted.get(WORLD).unwrap().sim(|s| {
            s.threads
                .of("drone")
                .iter()
                .flat_map(|t| t.messages.iter())
                .filter(|m| m.from == FROM)
                .count()
        })
    }

    /// **An answer posts one line on the carrier's thread, once**; none when
    /// the world has turned replica while it was asked, or the carrier was
    /// held still.
    #[tokio::test]
    async fn an_answer_is_posted_once_where_the_world_is_still_written() {
        let said = format!("{HANDLE}: question — which door?");
        let (_dir, rt) = carrying();
        rt.take_news(&ask(), &said);
        rt.take_news(&ask(), &said);
        assert_eq!(posted(&rt), 1, "once");

        let (_dir, rt) = carrying();
        rt.standing.set_role(WORLD, Role::Replica, 0);
        rt.take_news(&ask(), &said);
        assert_eq!(posted(&rt), 0, "the world turned replica");

        let (_dir, rt) = carrying();
        rt.scheduler.wake(7, 0, 0);
        rt.scheduler.suspend(7, true);
        rt.take_news(&ask(), &said);
        assert_eq!(posted(&rt), 0, "the carrier is held still");
    }

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
