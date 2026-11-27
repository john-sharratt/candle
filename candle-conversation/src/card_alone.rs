//! A GPU test's card to itself: the check that no live daemon holds it.
//!
//! `motherd`, `npcd` and `zend` each reserve most of the card at load — the
//! span, the expert zone, the arenas — so a test that reserves its own span
//! beside one of them is refused ("the card backed only … of the … asked
//! for") and reads as a failure of the code under test. It is not: the test
//! cannot run while the daemon does. [`cuda_alone`] opens the card and, when
//! one of [`LIVE_DAEMONS`] is running on this machine, aborts the test with a
//! warning that says so, instead of letting the reservation fail somewhere
//! inside it.
//!
//! What to do then is the caller's call, weighing the risk to the service
//! against the risk of the change: skip the test (`cargo test -- --skip …`), or
//! stop the daemon through keep (`POST http://127.0.0.1:8092/stop/<service>`,
//! `/start/<service>` after) and run it again.

use candle::Device;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

/// The daemons that hold the card while they run.
pub const LIVE_DAEMONS: [&str; 3] = ["motherd", "npcd", "zend"];

/// The daemon a process named `process` is, if it is one of
/// [`LIVE_DAEMONS`]: its name, case aside and with any `.exe` dropped, is the
/// daemon's exactly — a test binary or a tool merely named after one is not.
pub fn daemon_named(process: &str) -> Option<&'static str> {
    let lower = process.to_ascii_lowercase();
    let stem = lower.strip_suffix(".exe").unwrap_or(&lower);
    LIVE_DAEMONS.into_iter().find(|d| *d == stem)
}

/// The daemons of [`LIVE_DAEMONS`] running on this machine, each once, in
/// that list's order.
pub fn live_daemons() -> Vec<&'static str> {
    let mut sys = System::new();
    sys.refresh_processes_specifics(ProcessesToUpdate::All, ProcessRefreshKind::new());
    let running: Vec<&'static str> = sys
        .processes()
        .values()
        .filter_map(|p| daemon_named(&p.name().to_string_lossy()))
        .collect();
    LIVE_DAEMONS
        .into_iter()
        .filter(|d| running.contains(d))
        .collect()
}

/// The warning a test aborts with while `running` hold the card.
pub fn warning(running: &[&str]) -> String {
    format!(
        "this test needs the GPU to itself and cannot run while {} {} running on this \
         machine: the daemon holds the card, so the test's own reservation would be refused. \
         Either skip this test (`cargo test -- --skip <name>`) or stop {} through keep \
         (`curl -X POST http://127.0.0.1:8092/stop/<service>`, then `/start/<service>`) and run \
         it again — weigh the risk to the service against the risk of the change",
        running.join(", "),
        if running.len() == 1 { "is" } else { "are" },
        if running.len() == 1 { "it" } else { "them" },
    )
}

/// Abort the calling test when a live daemon holds the card ([`warning`]).
pub fn require_card_alone() {
    let running = live_daemons();
    if !running.is_empty() {
        panic!("{}", warning(&running));
    }
}

/// CUDA device `ordinal` for a test that needs the card to itself: `None` on
/// a machine or build without one, so the test skips as before; the device
/// once no live daemon holds the card; and an abort with [`warning`] while one
/// does.
pub fn cuda_alone(ordinal: usize) -> Option<Device> {
    match Device::cuda_if_available(ordinal) {
        Ok(d @ Device::Cuda(_)) => {
            require_card_alone();
            Some(d)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A process is a daemon only by its exact name**, case and `.exe`
    /// aside.
    #[test]
    fn a_process_is_a_daemon_only_by_its_exact_name() {
        assert_eq!(daemon_named("zend.exe"), Some("zend"));
        assert_eq!(daemon_named("MOTHERD.EXE"), Some("motherd"));
        assert_eq!(daemon_named("npcd"), Some("npcd"));
        assert_eq!(daemon_named("zend-1f2e3d4c5b6a.exe"), None);
        assert_eq!(daemon_named("zendesk.exe"), None);
        assert_eq!(daemon_named("keep.exe"), None);
    }

    /// **The warning names what runs and both ways out.**
    #[test]
    fn the_warning_names_what_runs_and_both_ways_out() {
        let one = warning(&["zend"]);
        assert!(one.contains("while zend is running"), "{one}");
        assert!(
            one.contains("--skip") && one.contains("/stop/<service>"),
            "{one}"
        );
        let two = warning(&["motherd", "npcd"]);
        assert!(two.contains("while motherd, npcd are running"), "{two}");
        assert!(two.contains("stop them"), "{two}");
    }
}
