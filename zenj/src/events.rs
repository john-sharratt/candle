//! What happened, in order (§13 `zenj_events`): a track changed, a request
//! arrived, the mode changed, a fault. Each event has a number; a reader asks
//! for those after the last it saw. The newest few thousand are kept.

use std::collections::VecDeque;
use std::sync::Mutex;

use chrono::{DateTime, Local};
use serde::Serialize;

/// How many events are kept.
pub const KEPT: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Track,
    Next,
    Transition,
    Request,
    Mode,
    Moment,
    Fault,
    Said,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Event {
    pub id: u64,
    pub at: DateTime<Local>,
    pub kind: Kind,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct Events {
    inner: Mutex<(u64, VecDeque<Event>)>,
}

impl Events {
    pub fn new() -> Events {
        Events::default()
    }

    /// Record one, at `at`; its number.
    pub fn push_at(&self, at: DateTime<Local>, kind: Kind, text: impl Into<String>) -> u64 {
        let text = text.into();
        tracing::info!(kind = ?kind, "zenj: {text}");
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.0 += 1;
        let id = g.0;
        g.1.push_back(Event { id, at, kind, text });
        while g.1.len() > KEPT {
            g.1.pop_front();
        }
        id
    }

    pub fn push(&self, kind: Kind, text: impl Into<String>) -> u64 {
        self.push_at(Local::now(), kind, text)
    }

    /// Up to `limit` events after `cursor`, oldest first.
    pub fn since(&self, cursor: u64, limit: usize) -> Vec<Event> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.1.iter().filter(|e| e.id > cursor).take(limit).cloned().collect()
    }

    /// The newest event's number.
    pub fn last(&self) -> u64 {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reader_gets_what_came_after_its_cursor() {
        let e = Events::new();
        assert_eq!(e.push(Kind::Track, "a"), 1);
        e.push(Kind::Request, "b");
        e.push(Kind::Fault, "c");
        let after: Vec<String> = e.since(1, 10).into_iter().map(|x| x.text).collect();
        assert_eq!(after, ["b", "c"]);
        assert_eq!(e.since(1, 1).len(), 1);
        assert!(e.since(3, 10).is_empty());
        assert_eq!(e.last(), 3);
    }

    #[test]
    fn only_the_newest_are_kept() {
        let e = Events::new();
        for i in 0..KEPT + 10 {
            e.push(Kind::Said, i.to_string());
        }
        let all = e.since(0, usize::MAX);
        assert_eq!(all.len(), KEPT);
        assert_eq!(all[0].id, 11);
    }
}
