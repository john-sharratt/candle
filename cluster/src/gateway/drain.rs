//! Maintaining a gateway itself (§12.4): `drain-gateway` stops admitting
//! sockets and closes every member's with `1001`, so each carries on through
//! its other clusters, then waits for the requests in flight through it;
//! `undrain-gateway` admits again.

use std::time::Duration;

use tokio::time::{sleep, Instant};

use super::hub::Hub;

/// The longest a drain waits for the requests in flight.
pub const DRAIN_WAIT: Duration = Duration::from_secs(60);

impl Hub {
    pub fn draining(&self) -> bool {
        self.state().draining
    }

    /// Stop admitting, close every member socket with `1001`, and wait for
    /// in-flight requests — at most [`DRAIN_WAIT`]. Returns how many were
    /// still open when it stopped waiting.
    pub async fn drain_gateway(&self) -> u64 {
        self.state().draining = true;
        self.note(
            "",
            String::new(),
            "drain-gateway: admitting no sockets".into(),
        );
        self.close_all("this gateway is draining: carry on through the other clusters");
        let deadline = Instant::now() + DRAIN_WAIT;
        loop {
            let open = self.traffic.open_streams();
            if open == 0 || Instant::now() >= deadline {
                return open;
            }
            sleep(Duration::from_millis(50)).await;
        }
    }

    /// Admit sockets again.
    pub fn undrain_gateway(&self) {
        self.state().draining = false;
        self.note("", String::new(), "undrain-gateway: admitting".into());
    }
}
