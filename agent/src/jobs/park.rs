//! The queue waits on keep while the process restarts
//! (`docs/build_agent.md` §6). A drained hold parks every run not yet ended,
//! in its order, and the runs that ended, on keep's local port; the agent's
//! next process takes the park back once it is serving and runs them under
//! their ids. A run that arrives after the park joins it.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::watch;

use crate::keep::post;
use crate::wire::{JobLine, Park, ParkedEnded, ParkedRun};

use super::{keep_in, Entry, Gate, Jobs, Kind, Line, KEPT};

/// keep's local route that holds a park.
pub const PARK: &str = "/park";
/// keep's local route that answers a park and lets it go.
pub const UNPARK: &str = "/unpark";

impl Jobs {
    /// What a park of this agent's queue, as `node`, holds now: the run that
    /// holds the slot, then the queue in its order, then the runs still
    /// fetching or setting up; and the runs that ended, oldest first. A
    /// checkout's setup is not parked: the next process runs its own.
    pub(super) fn park_of(&self, node: u64) -> Park {
        let holder = self.queue.running().map(|h| h.id);
        let waiting = self.queue.waiting();
        let t = self.table();
        let rank = |id: &str| {
            if holder.as_deref() == Some(id) {
                return 0;
            }
            waiting
                .iter()
                .position(|w| w == id)
                .map_or(usize::MAX, |at| at + 1)
        };
        let mut runs: Vec<(usize, u64, ParkedRun)> = Vec::new();
        let mut ended: Vec<ParkedEnded> = Vec::new();
        for id in &t.1 {
            let Some(e) = t.0.get(id) else { continue };
            let done = *e.ended.borrow();
            match &e.kind {
                Kind::Run { request, gate, .. } if !done => {
                    if *gate == Gate::AfterSetup {
                        runs.push((
                            rank(id),
                            e.line.started_ms,
                            ParkedRun {
                                id: id.clone(),
                                run: request.clone(),
                            },
                        ));
                    }
                }
                Kind::Run { .. } | Kind::Recalled { .. } if done => {
                    if let (Ok(view), Ok(line)) = (self.view_in(&t, id), self.line_in(&t, id)) {
                        ended.push(ParkedEnded { line, view });
                    }
                }
                _ => {}
            }
        }
        runs.sort_by_key(|(rank, started, _)| (*rank, *started));
        Park {
            node: format!("{node:016x}"),
            runs: runs.into_iter().map(|(_, _, r)| r).collect(),
            ended,
        }
    }

    /// Park the queue on keep: once a hold has drained, and again for each
    /// run that arrives after.
    pub(super) async fn park(self: &Arc<Self>) {
        let Some(node) = self.node() else {
            return;
        };
        let park = self.park_of(node);
        let (runs, ended) = (park.runs.len(), park.ended.len());
        let body = match serde_json::to_value(&park) {
            Ok(body) => body,
            Err(e) => {
                tracing::error!("the queue's park does not encode: {e}");
                return;
            }
        };
        match post(PARK, body).await {
            Ok(answer) if answer.status == 200 => {
                self.parked.store(true, Ordering::Relaxed);
                tracing::info!(
                    runs,
                    ended,
                    "the queue is parked on keep for the next process"
                );
            }
            Ok(answer) => tracing::warn!(
                "keep did not take the queue's park: {} {}",
                answer.status,
                answer.body.trim()
            ),
            Err(e) => tracing::warn!("the queue was not parked: {e}"),
        }
    }

    /// Take this agent's park back from keep, if keep holds one: its ended
    /// runs are listed as ended, its runs queued again under their ids. keep
    /// lets the park go as it answers.
    pub(super) async fn unpark(self: &Arc<Self>) -> Option<Park> {
        let node = self.node()?;
        let answer = match post(UNPARK, json!({"node": format!("{node:016x}")})).await {
            Ok(answer) => answer,
            Err(e) => {
                tracing::warn!("the park was not asked for: {e}");
                return None;
            }
        };
        self.parked.store(false, Ordering::Relaxed);
        if answer.status != 200 {
            return None;
        }
        serde_json::from_str::<Option<Park>>(&answer.body)
            .map_err(|e| tracing::warn!("keep's park does not read: {e}"))
            .ok()
            .flatten()
    }

    /// Take the park back and queue what it holds.
    pub(super) async fn take_back(self: &Arc<Self>) {
        let Some(park) = self.unpark().await else {
            return;
        };
        let (runs, ended) = (park.runs.len(), park.ended.len());
        self.recall(park.ended);
        for parked in park.runs {
            if self.table().0.contains_key(&parked.id) {
                continue;
            }
            let id = parked.id.clone();
            if let Err(e) = self.run_as(parked.run, Gate::AfterSetup, parked.id).await {
                tracing::warn!(job = id, "a parked run was not queued again: {e}");
            }
        }
        tracing::info!(runs, ended, "the queue's park is taken back from keep");
    }

    /// List `ended`, runs of the previous process, as ended.
    fn recall(&self, ended: Vec<ParkedEnded>) {
        let mut t = self.table();
        for e in ended.into_iter().take(KEPT) {
            if t.0.contains_key(&e.line.id) {
                continue;
            }
            let JobLine {
                id,
                repo,
                branch,
                base,
                command,
                started_ms,
                slot_ms,
                ended_ms,
                ..
            } = e.line;
            let (ended, _) = watch::channel(true);
            keep_in(
                &mut t,
                id,
                Entry {
                    repo,
                    kind: Kind::Recalled { view: e.view },
                    line: Line {
                        branch,
                        base,
                        command,
                        started_ms,
                        slot_ms,
                        ended_ms,
                    },
                    ended,
                },
            );
        }
    }

    /// The queue opens: the process is serving. Its park, if keep holds one,
    /// is taken back first, so the parked runs go before any sent since.
    pub async fn open(self: &Arc<Self>) {
        self.take_back().await;
        self.queue.open();
    }

    /// A hold that ended without a restart: keep lets the park go, and the
    /// queue, which never left this process, goes on.
    pub(super) async fn discard_park(self: &Arc<Self>) {
        if self.parked.load(Ordering::Relaxed) {
            let _ = self.unpark().await;
        }
    }
}
