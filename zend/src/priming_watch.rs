//! What the priming chain is doing, said when it stops moving.
//!
//! A chain link can stop in many places, and on 2026-11-01 WhiteWorld's load sat
//! in its first link for 5.6 hours without a line to say where. So the chain
//! names the link it reads and the stage it is at ([`link`], [`stage`]), and a
//! thread beside it ([`Watch`]) reports a stage that has stopped moving, after
//! [`FIRST_REPORT`] and then at each doubling of the wait.
//!
//! A report carries what tells the two kinds of stall apart: how fast the
//! substrate publishes versions and mints identities. A loop that only tries
//! again publishes or mints without pause; a wait for a wave, a reply or a peer
//! does neither.

use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use candle_conversation::projection::Conversation;

/// How long one stage may run before it is reported.
const FIRST_REPORT: Duration = Duration::from_secs(60);

/// How often the watch looks.
const TICK: Duration = Duration::from_secs(5);

/// Versions published or identities minted a second past which a stall is a
/// spin: far above what a loading member writes.
const SPINNING_PER_S: u64 = 1_000;

/// The link being read, its stage, and when the stage began.
struct At {
    link: String,
    stage: &'static str,
    since: Instant,
}

static AT: Mutex<Option<At>> = Mutex::new(None);

/// The chain has moved on to `link`.
pub fn link(link: &str) {
    *AT.lock().unwrap_or_else(|e| e.into_inner()) = Some(At {
        link: link.to_string(),
        stage: "started",
        since: Instant::now(),
    });
}

/// The link being read has reached `stage`.
pub fn stage(stage: &'static str) {
    let mut at = AT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(at) = at.as_mut() {
        tracing::info!(
            target: "zend::priming_chain",
            link = %at.link,
            stage,
            previous_took_ms = at.since.elapsed().as_millis() as u64,
            "chain link stage",
        );
        at.stage = stage;
        at.since = Instant::now();
    }
}

/// Whether a stage that has run `waited`, reported `said` times, is due another
/// report: at [`FIRST_REPORT`], then at each doubling.
fn due(waited: Duration, said: u32) -> bool {
    waited >= FIRST_REPORT.saturating_mul(1u32.checked_shl(said).unwrap_or(u32::MAX))
}

/// What a stall's rates say it is.
fn reading(versions_per_s: u64, minted_per_s: u64) -> &'static str {
    if versions_per_s.max(minted_per_s) >= SPINNING_PER_S {
        "spinning: something is retrying without pause"
    } else {
        "waiting: nothing here is moving"
    }
}

/// The watch: reports stalls for as long as it is held.
pub struct Watch {
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Watch {
    /// Watch the chain being built over `conv`'s substrate.
    pub fn start(conv: Conversation) -> Watch {
        let (stop, stopped) = mpsc::channel::<()>();
        let thread = thread::Builder::new()
            .name("priming-watch".into())
            .spawn(move || {
                let mut last = (Instant::now(), conv.versions_published(), conv.next_oseq());
                let mut watching: Option<Instant> = None;
                let mut said = 0;
                while stopped.recv_timeout(TICK) == Err(RecvTimeoutError::Timeout) {
                    let now = (Instant::now(), conv.versions_published(), conv.next_oseq());
                    let at = AT.lock().unwrap_or_else(|e| e.into_inner());
                    let Some(at) = at.as_ref() else { continue };
                    if watching != Some(at.since) {
                        watching = Some(at.since);
                        said = 0;
                    }
                    let waited = at.since.elapsed();
                    if !due(waited, said) {
                        continue;
                    }
                    said += 1;
                    let secs = now.0.duration_since(last.0).as_secs().max(1);
                    let versions_per_s = now.1.saturating_sub(last.1) / secs;
                    let minted_per_s = now.2.saturating_sub(last.2) / secs;
                    tracing::warn!(
                        target: "zend::priming_chain",
                        link = %at.link,
                        stage = at.stage,
                        waited_s = waited.as_secs(),
                        versions_per_s,
                        minted_per_s,
                        generation = conv.generation(),
                        detached = conv.holds_detached(),
                        "a chain link has stopped moving: {}",
                        reading(versions_per_s, minted_per_s),
                    );
                    last = now;
                }
            })
            .ok();
        Watch {
            stop: Some(stop),
            thread,
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        *AT.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A stage is reported at a minute, then at each doubling.**
    #[test]
    fn a_stall_is_reported_at_each_doubling() {
        let secs = Duration::from_secs;
        assert!(!due(secs(59), 0));
        assert!(due(secs(60), 0));
        assert!(!due(secs(119), 1));
        assert!(due(secs(120), 1));
        assert!(!due(secs(239), 2));
        assert!(due(secs(240), 2));
        assert!(!due(secs(u64::from(u32::MAX) - 1), 40), "never overflows");
    }

    /// Either rate past the mark is a spin; a quiet substrate is a wait.
    #[test]
    fn a_rate_tells_a_spin_from_a_wait() {
        assert!(reading(32_000, 0).starts_with("spinning"));
        assert!(reading(0, 32_000).starts_with("spinning"));
        assert!(reading(12, 3).starts_with("waiting"));
    }
}
