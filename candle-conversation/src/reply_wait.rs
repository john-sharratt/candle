//! Waiting for the scheduler's reply, and saying so when it does not come.
//!
//! A sequence's calls hand the scheduler a request and block on its reply, for
//! as long as that takes: a reply that never comes holds the caller for ever,
//! and nothing on either side says so. WhiteWorld's load stopped in a chain link
//! for 5.6 hours on 2026-11-01 with the logs silent from the moment the link's
//! conversation was being built. [`replied`] waits the same, and names the
//! waiting call at [`FIRST_SAID`] and at each doubling.

use std::panic::Location;
use std::time::{Duration, Instant};

use flume::{Receiver, RecvError, RecvTimeoutError};

/// How long a reply may take before the wait is said.
const FIRST_SAID: Duration = Duration::from_secs(60);

/// The scheduler's reply on `rx`; an `Err` when it hung up without one.
#[track_caller]
pub(crate) fn replied<T>(rx: &Receiver<T>) -> Result<T, RecvError> {
    let site = Location::caller();
    let began = Instant::now();
    let mut patience = FIRST_SAID;
    loop {
        match rx.recv_timeout(patience) {
            Ok(reply) => return Ok(reply),
            Err(RecvTimeoutError::Disconnected) => return Err(RecvError::Disconnected),
            Err(RecvTimeoutError::Timeout) => {
                tracing::warn!(
                    target: "candle_conversation::reply_wait",
                    site = %site,
                    waited_s = began.elapsed().as_secs(),
                    "a call is still waiting for the scheduler's reply"
                );
                patience = patience.saturating_mul(2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply is handed back, and a scheduler gone is an error, not a wait.
    #[test]
    fn a_reply_or_a_hang_up_ends_the_wait() {
        let (tx, rx) = flume::bounded(1);
        tx.send(7).unwrap();
        assert_eq!(replied(&rx), Ok(7));
        drop(tx);
        assert_eq!(replied(&rx), Err(RecvError::Disconnected));
    }
}
