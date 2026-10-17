//! The syncing line a supervisor restarts its trial limit on.
//!
//! A member that joins the cluster catches up on what the group wrote while it
//! was away before it serves (`docs/replicated_substrate.md` §7.6), and that
//! catch-up is as long as the backlog. `keep` rolls back a trial that does not
//! prove itself within its limit, so a build that was merely behind would be
//! rolled back for it. Each time a tenant's catch-up receives more, the process
//! prints a [`SYNCING`] line, and `keep` restarts the limit there: a catch-up
//! that keeps receiving is waited out, and one that stalls prints nothing and
//! runs the limit down.

use cluster::upgrade::switch::SYNCING;

use crate::loading::LoadingView;

/// The step id both tenants give their catch-up.
const CATCH_UP: &str = "catchup";

/// The syncing line to print for one tenant's newest load, or `None`.
///
/// `last` is what the tenant had caught up on at the previous look, kept here
/// for the next one. A line is due only when the count has risen since: a
/// catch-up standing still says nothing, which is what lets a stalled one time
/// out.
pub fn next_syncing_line(
    tenant: usize,
    view: Option<&LoadingView>,
    last: &mut u64,
) -> Option<String> {
    let view = view.filter(|v| v.current == CATCH_UP)?;
    let risen = view.progressed > *last;
    *last = view.progressed;
    risen.then(|| {
        format!(
            "{SYNCING} tenant {tenant} {}/{} {}",
            view.progressed, view.total, view.unit
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use cluster::upgrade::switch::is_syncing;

    fn view(current: &'static str, progressed: u64) -> LoadingView {
        LoadingView {
            current,
            label: "",
            progress: 0.0,
            completed: Vec::new(),
            completed_labels: Vec::new(),
            progressed,
            total: 5_000,
            unit: "records".into(),
            detail: String::new(),
            elapsed_ms: 0,
            prefill_tokens: 0,
            prefill_tps: None,
            waiting_for: Vec::new(),
        }
    }

    /// **Received data prints a line; standing still prints none.** The line
    /// is one `keep` recognises, and it names what arrived.
    #[test]
    fn a_line_is_due_only_when_the_catch_up_has_received_more() {
        let mut last = 0;
        let line = next_syncing_line(0, Some(&view(CATCH_UP, 1_200)), &mut last).unwrap();
        assert_eq!(line, "INFO syncing tenant 0 1200/5000 records");
        assert!(is_syncing(&line));
        assert_eq!(
            next_syncing_line(0, Some(&view(CATCH_UP, 1_200)), &mut last),
            None
        );
        assert!(next_syncing_line(0, Some(&view(CATCH_UP, 1_201)), &mut last).is_some());
    }

    /// Another step's progress, a tenant already loaded, and a catch-up at
    /// nothing received yet say nothing: only sync data restarts the limit.
    #[test]
    fn nothing_but_received_sync_data_prints_a_line() {
        let mut last = 0;
        assert_eq!(
            next_syncing_line(0, Some(&view("substrate", 900)), &mut last),
            None
        );
        assert_eq!(next_syncing_line(0, None, &mut last), None);
        assert_eq!(
            next_syncing_line(0, Some(&view(CATCH_UP, 0)), &mut last),
            None
        );
    }

    /// A count that falls — a catch-up round that started over — is the new
    /// floor, and the next rise above it prints.
    #[test]
    fn a_catch_up_that_starts_over_is_counted_from_where_it_stands() {
        let mut last = 0;
        next_syncing_line(1, Some(&view(CATCH_UP, 4_000)), &mut last);
        assert_eq!(
            next_syncing_line(1, Some(&view(CATCH_UP, 10)), &mut last),
            None
        );
        assert!(next_syncing_line(1, Some(&view(CATCH_UP, 11)), &mut last).is_some());
    }
}
