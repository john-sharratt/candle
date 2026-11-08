//! Opening and closing rekordbox when asked (`docs/house_modes.md` §3): by
//! Zen's tool here (`zenj_rekordbox`), and by the house's mode switch over the
//! gateway's bus (`crate::bus`). Both come to [`Rekordbox::act`]: one open or
//! close at a time, a playing one never closed unless a person says so, and
//! each outcome on the night's record.

use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;
use tokio::sync::Mutex;
use tokio::task::spawn_blocking;

use super::process::{close, open, refuse_close};
use crate::events::{Events, Kind};
use crate::live::Live;

/// What is asked of rekordbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Open,
    /// Close it; `force` closes it with a deck playing.
    Close {
        force: bool,
    },
}

impl Act {
    /// `open` or `close`, as the tool and the bus name them.
    pub fn parse(act: &str, force: bool) -> Option<Act> {
        match act.trim() {
            "open" => Some(Act::Open),
            "close" => Some(Act::Close { force }),
            _ => None,
        }
    }
}

/// rekordbox, as the house asks for it.
pub struct Rekordbox {
    /// Its executable, started to open it.
    pub program: PathBuf,
    /// Its process's executable name, looked for to know it runs.
    pub exe: String,
    pub live: Arc<ArcSwap<Live>>,
    pub events: Arc<Events>,
    /// One open or close at a time.
    one: Mutex<()>,
}

impl Rekordbox {
    pub fn new(
        program: PathBuf,
        exe: impl Into<String>,
        live: Arc<ArcSwap<Live>>,
        events: Arc<Events>,
    ) -> Rekordbox {
        Rekordbox {
            program,
            exe: exe.into(),
            live,
            events,
            one: Mutex::new(()),
        }
    }

    /// Do `act`; the line saying what came of it, or why it did not happen.
    pub async fn act(&self, act: Act) -> Result<String, String> {
        let _one = self.one.lock().await;
        if let Act::Close { force } = act {
            if let Some(why) = refuse_close(self.live.load().live, force) {
                return Err(why);
            }
        }
        let (program, exe) = (self.program.clone(), self.exe.clone());
        let done = spawn_blocking(move || match act {
            Act::Open => open(&program, &exe),
            Act::Close { .. } => close(&exe),
        })
        .await
        .map_err(|e| format!("rekordbox could not be reached: {e}"))?;
        match &done {
            Ok(line) => self.events.push(Kind::Rekordbox, line.clone()),
            Err(why) => self.events.push(Kind::Fault, why.clone()),
        };
        done
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn rekordbox(playing: bool) -> Rekordbox {
        let live = Live {
            live: playing,
            ..Live::default()
        };
        Rekordbox::new(
            Path::new("Z:/no/such/rekordbox.exe").into(),
            "zenj-no-such-process.exe",
            Arc::new(ArcSwap::from_pointee(live)),
            Arc::new(Events::new()),
        )
    }

    #[test]
    fn acts_are_named_open_and_close() {
        assert_eq!(Act::parse("open", true), Some(Act::Open));
        assert_eq!(Act::parse(" close", true), Some(Act::Close { force: true }));
        assert_eq!(Act::parse("restart", false), None);
    }

    /// **A playing rekordbox is not closed unless forced**, and what is
    /// refused touches no process and records nothing.
    #[tokio::test]
    async fn a_playing_rekordbox_is_not_closed_unless_forced() {
        let r = rekordbox(true);
        let refused = r.act(Act::Close { force: false }).await.unwrap_err();
        assert!(refused.starts_with("a deck is playing"), "{refused}");
        assert!(r.events.since(0, 10).is_empty());
        assert_eq!(
            r.act(Act::Close { force: true }).await,
            Ok("rekordbox is already closed".into())
        );
        assert_eq!(r.events.since(0, 10)[0].kind, Kind::Rekordbox);
    }

    /// **An open that cannot start rekordbox is a fault on the record.**
    #[tokio::test]
    async fn an_open_that_fails_is_a_fault() {
        let r = rekordbox(false);
        assert!(r.act(Act::Open).await.is_err());
        assert_eq!(r.events.since(0, 10)[0].kind, Kind::Fault);
    }
}
