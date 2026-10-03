//! What the guardian has done, kept for an operator to read.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::npcs;

/// How many records are kept; the oldest go first.
pub const CAPACITY: usize = 500;

/// One thing the guardian did or found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Record {
    pub at_ms: u64,
    pub npc_id: String,
    /// `check`, `tick`, `nudge`, `restate`, `refresh`, `flag`, `held`, `failed`.
    pub kind: &'static str,
    pub detail: String,
}

#[derive(Default)]
pub struct GuardianLog {
    records: Mutex<VecDeque<Record>>,
}

impl GuardianLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, npc_id: u64, kind: &'static str, detail: impl Into<String>) {
        let at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        let mut records = self.records.lock().unwrap();
        if records.len() == CAPACITY {
            records.pop_front();
        }
        records.push_back(Record {
            at_ms,
            npc_id: npcs::npc_id_wire(npc_id),
            kind,
            detail: detail.into(),
        });
    }

    /// The records, oldest first.
    pub fn records(&self) -> Vec<Record> {
        self.records.lock().unwrap().iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_come_back_oldest_first_with_the_wire_id() {
        let log = GuardianLog::new();
        log.push(1, "check", "drift: something else");
        log.push(1, "nudge", "put the errand back");
        let r = log.records();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].kind, "check");
        assert_eq!(r[1].kind, "nudge");
        assert_eq!(r[0].npc_id, npcs::npc_id_wire(1));
    }

    #[test]
    fn the_oldest_record_goes_when_the_log_is_full() {
        let log = GuardianLog::new();
        for i in 0..CAPACITY + 3 {
            log.push(1, "check", format!("{i}"));
        }
        let r = log.records();
        assert_eq!(r.len(), CAPACITY);
        assert_eq!(r[0].detail, "3");
    }
}
