//! A command's job (`docs/replicated_substrate.md` §12.4): the command answers
//! at once with its id, and its progress streams on
//! `GET /v1/cluster/command/job/:id` as NDJSON — one object per line, every
//! stream ending in exactly one terminal line, `done` or `error`, as npcd's
//! streams do. One command runs per member at a time.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, MutexGuard};

use axum::body::Body;
use axum::http::{header, HeaderName};
use axum::response::{IntoResponse, Response};
use cluster::clock::now_ms;
use futures::stream::{self, Stream};
use serde_json::{json, Value};
use tokio::sync::watch;

/// The jobs kept for reading after they end.
pub const KEPT: usize = 32;

/// Proxies buffer a response body unless told not to, which would hold every
/// line until the job ended.
const ACCEL_BUFFERING: HeaderName = HeaderName::from_static("x-accel-buffering");

/// One command's run.
pub struct Job {
    pub id: u64,
    pub command: &'static str,
    pub started_ms: u64,
    lines: Mutex<Vec<Value>>,
    /// The lines written so far, and whether the last was terminal.
    written: watch::Sender<(usize, bool)>,
}

impl Job {
    fn lines(&self) -> MutexGuard<'_, Vec<Value>> {
        self.lines.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push(&self, mut line: Value, terminal: bool) {
        if let Value::Object(m) = &mut line {
            m.insert("at_ms".into(), json!(now_ms()));
        }
        let n = {
            let mut lines = self.lines();
            if lines.last().is_some_and(is_terminal) {
                return;
            }
            lines.push(line);
            lines.len()
        };
        self.written.send_replace((n, terminal));
    }

    /// A step of the job, as the page shows it under the member's row.
    pub fn step(&self, step: &str, detail: impl Into<String>) {
        self.push(
            json!({"event": "step", "step": step, "detail": detail.into()}),
            false,
        );
    }

    /// Whether it has ended.
    pub fn ended(&self) -> bool {
        self.written.borrow().1
    }

    /// Every line so far.
    pub fn snapshot(&self) -> Vec<Value> {
        self.lines().clone()
    }

    /// The job's lines from the first, as they are written, until its
    /// terminal line.
    pub fn stream(self: Arc<Self>) -> impl Stream<Item = Result<String, Infallible>> {
        let rx = self.written.subscribe();
        stream::unfold((self, rx, 0usize), |(job, mut rx, next)| async move {
            loop {
                let lines = job.snapshot();
                if next < lines.len() {
                    let line = format!("{}\n", lines[next]);
                    return Some((Ok(line), (job, rx, next + 1)));
                }
                if lines.last().is_some_and(is_terminal) || rx.changed().await.is_err() {
                    return None;
                }
            }
        })
    }

    /// The NDJSON response for [`Self::stream`].
    pub fn response(self: Arc<Self>) -> Response {
        (
            [
                (header::CONTENT_TYPE, "application/x-ndjson"),
                (header::CACHE_CONTROL, "no-cache, no-transform"),
                (ACCEL_BUFFERING, "no"),
            ],
            Body::from_stream(self.stream()),
        )
            .into_response()
    }
}

fn is_terminal(line: &Value) -> bool {
    matches!(line["event"].as_str(), Some("done" | "error"))
}

#[derive(Default)]
struct Inner {
    next: u64,
    running: Option<Arc<Job>>,
    kept: VecDeque<Arc<Job>>,
}

/// Every command run on this member since it started, the last [`KEPT`].
#[derive(Default)]
pub struct Jobs {
    inner: Mutex<Inner>,
}

impl Jobs {
    pub fn new() -> Self {
        Self::default()
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start `command`, with `args` on its first line; refused while another
    /// runs.
    pub fn start(&self, command: &'static str, args: Value) -> Result<Arc<Job>, String> {
        let mut s = self.inner();
        if let Some(running) = &s.running {
            return Err(format!(
                "{} (job {}) is running: one command runs at a time",
                running.command, running.id
            ));
        }
        s.next += 1;
        let job = Arc::new(Job {
            id: s.next,
            command,
            started_ms: now_ms(),
            lines: Mutex::new(Vec::new()),
            written: watch::channel((0, false)).0,
        });
        job.push(
            json!({"event": "started", "command": command, "args": args}),
            false,
        );
        s.running = Some(job.clone());
        if s.kept.len() == KEPT {
            s.kept.pop_front();
        }
        s.kept.push_back(job.clone());
        Ok(job)
    }

    /// End `job` with its result, or the reason it failed.
    pub fn finish(&self, job: &Job, outcome: Result<Value, String>) {
        let line = match outcome {
            Ok(result) => json!({"event": "done", "result": result}),
            Err(detail) => json!({"event": "error", "error": "command_failed", "detail": detail}),
        };
        job.push(line, true);
        let mut s = self.inner();
        if s.running.as_ref().is_some_and(|r| r.id == job.id) {
            s.running = None;
        }
    }

    pub fn get(&self, id: u64) -> Option<Arc<Job>> {
        self.inner().kept.iter().find(|j| j.id == id).cloned()
    }

    /// The command running now.
    pub fn running(&self) -> Option<Arc<Job>> {
        self.inner().running.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::StreamExt;

    fn without_time(mut v: Value) -> Value {
        v.as_object_mut().unwrap().remove("at_ms");
        v
    }

    /// One command at a time; the next starts once it ends.
    #[test]
    fn one_command_runs_at_a_time() {
        let jobs = Jobs::new();
        let a = jobs.start("compact", json!({})).unwrap();
        assert_eq!(a.id, 1);
        assert_eq!(
            jobs.start("drain", json!({})).err().unwrap(),
            "compact (job 1) is running: one command runs at a time"
        );
        jobs.finish(&a, Ok(json!({"ran": true})));
        assert!(a.ended());
        assert!(jobs.running().is_none());
        assert_eq!(jobs.start("drain", json!({})).unwrap().id, 2);
        assert_eq!(jobs.get(1).unwrap().command, "compact");
    }

    /// The stream carries every line in order and ends with the terminal one,
    /// however late it is read; nothing is written after it.
    #[tokio::test]
    async fn the_stream_ends_with_one_terminal_line() {
        let jobs = Jobs::new();
        let job = jobs.start("maintain", json!({"app": "zend"})).unwrap();
        let reader = tokio::spawn(job.clone().stream().collect::<Vec<_>>());
        job.step("maintain", "dropping a dead segment");
        jobs.finish(&job, Err("the disk is full".into()));
        job.step("late", "never written");
        let lines: Vec<Value> = reader
            .await
            .unwrap()
            .into_iter()
            .map(|l| without_time(serde_json::from_str(&l.unwrap()).unwrap()))
            .collect();
        assert_eq!(
            lines,
            [
                json!({"event": "started", "command": "maintain", "args": {"app": "zend"}}),
                json!({"event": "step", "step": "maintain", "detail": "dropping a dead segment"}),
                json!({"event": "error", "error": "command_failed", "detail": "the disk is full"}),
            ]
        );
        // Read again once it has ended: the same lines, then the end.
        assert_eq!(job.stream().count().await, 3);
    }

    #[test]
    fn only_the_last_jobs_are_kept() {
        let jobs = Jobs::new();
        for _ in 0..KEPT + 2 {
            let j = jobs.start("audit", json!({})).unwrap();
            jobs.finish(&j, Ok(Value::Null));
        }
        assert!(jobs.get(1).is_none());
        assert!(jobs.get(KEPT as u64 + 2).is_some());
    }
}
