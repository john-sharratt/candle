//! A tenant's load, as `/v1/status` reports it.
//!
//! Each tenant keeps its own load-state machine — zend's and npcd's startups
//! are different sequences, and their steps stay theirs — and reports it in one
//! shape: [`LoadingView`]. Steps are named by **id** (`model`, `substrate`) so a
//! page can key on them, with the text beside them as `label`, the item in hand
//! as `detail`, and the prefill rate where the step prefills.

use serde::Serialize;

/// One tenant's load at a moment. Absent once the tenant is ready.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LoadingView {
    /// The step in progress, by id.
    pub current: &'static str,
    /// The step in progress, as the overlay prints it.
    pub label: &'static str,
    /// 0.0–1.0 within the step.
    pub progress: f32,
    /// The steps already done, by id, in execution order.
    pub completed: Vec<&'static str>,
    /// The same steps as the overlay prints them, one for one with
    /// `completed` — so a page that opened mid-load still lists them in the
    /// tenant's words.
    pub completed_labels: Vec<&'static str>,
    /// `progressed` of `total` `unit`s. `total == 0`: nothing countable has
    /// been reported yet. An empty `unit`: the counter is a scaled fraction, so
    /// only the bar shows.
    pub progressed: u64,
    pub total: u64,
    pub unit: String,
    /// The item being worked on right now — a file, a character. Empty when
    /// the step names none.
    pub detail: String,
    /// Milliseconds since the load began.
    pub elapsed_ms: u64,
    /// Tokens prefilled in this step, and the rate they arrived at. `None`
    /// before any are counted, so a page shows nothing rather than a zero that
    /// reads as a stalled engine.
    pub prefill_tokens: u64,
    pub prefill_tps: Option<f64>,
}

/// The rate `tokens` over `elapsed_ms` make, or `None` when there is nothing
/// to divide.
pub fn tokens_per_second(tokens: u64, elapsed_ms: u64) -> Option<f64> {
    (tokens > 0 && elapsed_ms > 0).then(|| tokens as f64 * 1000.0 / elapsed_ms as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_needs_tokens_and_time() {
        assert_eq!(tokens_per_second(0, 1000), None);
        assert_eq!(tokens_per_second(10, 0), None);
        assert_eq!(tokens_per_second(5000, 2000), Some(2500.0));
    }

    /// The wire shape both consoles read: ids, not labels, for the steps.
    #[test]
    fn the_view_serializes_steps_by_id() {
        let v = LoadingView {
            current: "substrate",
            label: "Loading substrate",
            progress: 0.5,
            completed: vec!["model"],
            completed_labels: vec!["Loading model"],
            progressed: 10,
            total: 20,
            unit: "turns".into(),
            detail: String::new(),
            elapsed_ms: 1500,
            prefill_tokens: 0,
            prefill_tps: None,
        };
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"current":"substrate","label":"Loading substrate","progress":0.5,"completed":["model"],"completed_labels":["Loading model"],"progressed":10,"total":20,"unit":"turns","detail":"","elapsed_ms":1500,"prefill_tokens":0,"prefill_tps":null}"#
        );
    }
}
