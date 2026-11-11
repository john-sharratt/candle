//! How much private mode and diaries are used, as counts alone
//! (`docs/private_mode.md` §8): never an id, a tool's arguments or a time of
//! a call — just how many since this process started, said once an hour.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// How often the counts are logged.
pub const SAID_EVERY: Duration = Duration::from_secs(60 * 60);

/// The counts.
#[derive(Debug, Default)]
pub struct Counters {
    private_begun: AtomicU64,
    private_calls: AtomicU64,
    diary_calls: AtomicU64,
}

impl Counters {
    /// A private conversation began.
    pub fn begun_private(&self) {
        self.private_begun.fetch_add(1, Ordering::Relaxed);
    }

    /// A call was made in a private conversation; `diary` when a diary's.
    pub fn private_call(&self, diary: bool) {
        self.private_calls.fetch_add(1, Ordering::Relaxed);
        if diary {
            self.diary_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Private conversations begun, calls made in them, and diary calls among
    /// those.
    pub fn read(&self) -> (u64, u64, u64) {
        (
            self.private_begun.load(Ordering::Relaxed),
            self.private_calls.load(Ordering::Relaxed),
            self.diary_calls.load(Ordering::Relaxed),
        )
    }
}

/// Log `counters` every [`SAID_EVERY`] for as long as the process runs.
pub async fn saying(counters: Arc<Counters>) {
    let mut every = tokio::time::interval(SAID_EVERY);
    every.tick().await;
    loop {
        every.tick().await;
        let (begun, calls, diaries) = counters.read();
        tracing::info!(
            private_begun = begun,
            private_calls = calls,
            diary_calls = diaries,
            "mcp: private mode since start"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_counts_are_counts_alone() {
        let c = Counters::default();
        c.begun_private();
        c.private_call(false);
        c.private_call(true);
        assert_eq!(c.read(), (1, 2, 1));
    }
}
