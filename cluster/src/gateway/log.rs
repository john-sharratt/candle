//! The coordination log (§12.3): every coordination decision this gateway made
//! or was told of, newest first, kept for 24 hours in memory. Each entry names
//! the gateway it came from, the app, and its subject — a member, a key — so
//! the page can filter by any of them.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;

/// How long an entry is kept.
pub const KEEP_MS: u64 = 24 * 60 * 60 * 1000;

/// What the log holds at most, whatever the age: a flapping member cannot
/// grow it without bound.
pub const CAP: usize = 50_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Entry {
    pub at_ms: u64,
    pub gateway: String,
    pub app: String,
    /// The member (`node` in hex) or key the entry is about.
    pub subject: String,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct CoordLog {
    entries: Mutex<VecDeque<Entry>>,
}

impl CoordLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, entry: Entry) {
        let mut e = self.entries.lock().unwrap();
        let cutoff = entry.at_ms.saturating_sub(KEEP_MS);
        while e.back().is_some_and(|old| old.at_ms < cutoff) || e.len() >= CAP {
            e.pop_back();
        }
        e.push_front(entry);
    }

    /// Newest first, at most `limit`, filtered by app and subject when given.
    pub fn read(&self, app: Option<&str>, subject: Option<&str>, limit: usize) -> Vec<Entry> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| app.is_none_or(|a| e.app == a))
            .filter(|e| subject.is_none_or(|s| e.subject == s))
            .take(limit)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(at_ms: u64, app: &str, subject: &str) -> Entry {
        Entry {
            at_ms,
            gateway: "g".into(),
            app: app.into(),
            subject: subject.into(),
            text: format!("{app} {subject} at {at_ms}"),
        }
    }

    /// Newest first; filtered by app and by subject; entries older than a
    /// day go as new ones arrive.
    #[test]
    fn newest_first_filtered_and_a_day_long() {
        let log = CoordLog::new();
        log.push(entry(1_000, "zend", "a"));
        log.push(entry(2_000, "npcd", "b"));
        log.push(entry(3_000, "zend", "b"));
        let all: Vec<u64> = log.read(None, None, 10).iter().map(|e| e.at_ms).collect();
        assert_eq!(all, [3_000, 2_000, 1_000]);
        assert_eq!(log.read(Some("zend"), None, 10).len(), 2);
        assert_eq!(log.read(Some("zend"), Some("b"), 10)[0].at_ms, 3_000);
        assert_eq!(log.read(None, None, 1).len(), 1);

        log.push(entry(KEEP_MS + 2_500, "zend", "c"));
        let kept: Vec<u64> = log.read(None, None, 10).iter().map(|e| e.at_ms).collect();
        assert_eq!(kept, [KEEP_MS + 2_500, 3_000]);
    }
}
