//! A member that stopped reporting (§4.4): its socket open, its `CAPACITY`s
//! stopped. The transport keeps the socket heard while the process behind it
//! answers nothing — a member deadlocked on 2026-11-02 stayed `ready` here,
//! and every request routed to it waited out its deadline — so the gateway
//! judges the member by its reports, not by its socket.

use std::time::Duration;

use super::hub::{hex, Hub, CAPACITY_SILENCE};

/// Whether a member last heard reporting `since` ago is silent.
pub(super) fn is_silent(since: Duration) -> bool {
    since >= CAPACITY_SILENCE
}

impl Hub {
    /// `node`'s socket's timer: silent once it has reported nothing for
    /// [`CAPACITY_SILENCE`]. Its next `CAPACITY` ends it (`Hub::on_frame`).
    pub(super) fn silence_tick(&self, app: &str, node: u64) {
        let fell_silent = {
            let mut s = self.state();
            let Some(since) = s
                .sockets
                .get(&(app.to_string(), node))
                .map(|x| x.capacity_at.elapsed())
            else {
                return;
            };
            is_silent(since) && s.registry.set_silent(app, node, true)
        };
        if fell_silent {
            tracing::warn!(
                app,
                node = %hex(node),
                silent_s = CAPACITY_SILENCE.as_secs(),
                "a member sent no CAPACITY with its socket open: routed nothing another member can take"
            );
            self.note(
                app,
                hex(node),
                format!(
                    "silent: no CAPACITY for {}s, socket open: routed nothing another member can take",
                    CAPACITY_SILENCE.as_secs()
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A member is silent after fifteen missed reports**, not after a
    /// pause the records it applies can cause.
    #[test]
    fn a_member_is_silent_after_fifteen_seconds_without_a_report() {
        assert_eq!(CAPACITY_SILENCE, Duration::from_secs(15));
        assert!(!is_silent(Duration::from_secs(14)));
        assert!(is_silent(Duration::from_secs(15)));
        assert!(is_silent(Duration::from_secs(600)));
    }
}
