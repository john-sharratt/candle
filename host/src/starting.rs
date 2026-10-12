//! What a tenant says about itself while its `start` runs.
//!
//! A tenant's own load-state machine reports through its `Reports`, which the
//! host has only once `start` returns. Some of a start is long all the same —
//! npcd opens its substrate there, and a store written before identities is
//! rewritten whole before that open returns — so the host hands every tenant a
//! [`StartLine`] to report into until then, and the loading page reads it.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use crate::loading::LoadingView;

/// The step a tenant is on before `start` returns, if it names none.
const STARTING: (&str, &str) = ("starting", "Starting");

/// One tenant's start, as the loading page shows it.
pub struct StartLine {
    began: Instant,
    line: Mutex<Line>,
}

struct Line {
    id: &'static str,
    label: &'static str,
    progressed: u64,
    total: u64,
    unit: String,
    detail: String,
}

impl StartLine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            began: Instant::now(),
            line: Mutex::new(Line {
                id: STARTING.0,
                label: STARTING.1,
                progressed: 0,
                total: 0,
                unit: String::new(),
                detail: String::new(),
            }),
        })
    }

    fn line(&self) -> MutexGuard<'_, Line> {
        self.line.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Enter step `id`, printed as `label`. Clears the counter and the detail,
    /// which belonged to the step before.
    pub fn set_step(&self, id: &'static str, label: &'static str) {
        let mut l = self.line();
        *l = Line {
            id,
            label,
            progressed: 0,
            total: 0,
            unit: String::new(),
            detail: String::new(),
        };
    }

    /// `progressed` of `total` `unit`s; `total == 0` counts nothing.
    pub fn set_progress(&self, progressed: u64, total: u64, unit: &str) {
        let mut l = self.line();
        l.progressed = progressed;
        l.total = total;
        if l.unit != unit {
            l.unit = unit.to_string();
        }
    }

    /// The item in hand.
    pub fn set_detail(&self, detail: impl Into<String>) {
        self.line().detail = detail.into();
    }

    /// The line in the host's loading shape.
    pub fn view(&self) -> LoadingView {
        let l = self.line();
        LoadingView {
            current: l.id,
            label: l.label,
            progress: if l.total == 0 {
                0.0
            } else {
                (l.progressed as f32 / l.total as f32).clamp(0.0, 1.0)
            },
            completed: Vec::new(),
            completed_labels: Vec::new(),
            progressed: l.progressed,
            total: l.total,
            unit: l.unit.clone(),
            detail: l.detail.clone(),
            elapsed_ms: self.began.elapsed().as_millis() as u64,
            prefill_tokens: 0,
            prefill_tps: None,
            waiting_for: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_line_is_starting() {
        let v = StartLine::new().view();
        assert_eq!((v.current, v.label), ("starting", "Starting"));
        assert_eq!(v.progress, 0.0);
    }

    /// A step's counter and detail are its own: entering the next clears them.
    #[test]
    fn a_step_reports_its_count_and_the_next_starts_clean() {
        let s = StartLine::new();
        s.set_step("opening", "Opening the substrate");
        s.set_progress(250, 1_000, "records");
        s.set_detail("Stamping records");
        let v = s.view();
        assert_eq!((v.current, v.progressed, v.total), ("opening", 250, 1_000));
        assert_eq!(v.progress, 0.25);
        assert_eq!(
            (v.unit.as_str(), v.detail.as_str()),
            ("records", "Stamping records")
        );
        s.set_step("loading", "Loading");
        let v = s.view();
        assert_eq!((v.progressed, v.total), (0, 0));
        assert!(v.detail.is_empty() && v.unit.is_empty());
    }
}
