//! What a screen does ahead of a phrase (§8), so a scene change lands on it
//! rather than after it: the next preset's file is read off the render thread
//! a phrase early ([`Prefetch`]), and how long a preset takes to load into
//! the engine is kept ([`LoadTime`]), so the change begins that much before
//! the phrase, on top of the screen's own lag.

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Duration;

/// A preset's file, being read on a thread of its own.
pub struct Prefetch {
    index: usize,
    bytes: Receiver<Option<Vec<u8>>>,
    ready: Option<Vec<u8>>,
}

impl Prefetch {
    /// Begin reading preset `index` from `path`.
    pub fn start(index: usize, path: PathBuf) -> Prefetch {
        let (tx, rx) = mpsc::channel();
        let spawned = thread::Builder::new()
            .name("zenj-screen-prefetch".into())
            .spawn(move || {
                let _ = tx.send(fs::read(path).ok());
            });
        if spawned.is_err() {
            tracing::debug!("screens: the next scene is read when it is due");
        }
        Prefetch {
            index,
            bytes: rx,
            ready: None,
        }
    }

    /// The bytes of preset `index`, if they are read by now — never waited
    /// for: not read yet, the file is read when it is due, as before.
    pub fn take(&mut self, index: usize) -> Option<Vec<u8>> {
        if index != self.index {
            return None;
        }
        if self.ready.is_none() {
            match self.bytes.try_recv() {
                Ok(read) => self.ready = read,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => {}
            }
        }
        self.ready.take()
    }
}

/// How long a preset takes to load: a running average, and the slowest of
/// late, so a change begins early enough for the slow ones too.
#[derive(Debug, Clone, Default)]
pub struct LoadTime {
    avg_ms: Option<f64>,
    worst_ms: f64,
}

/// How much of a new load the average takes.
const ALPHA: f64 = 0.2;
/// How much of the slowest a new load keeps, each time.
const FORGET: f64 = 0.9;
/// The most a change is begun early for a load: past this the preset is a
/// slow one, and the change is late rather than the scene early by seconds.
const CAP_MS: f64 = 1500.0;

impl LoadTime {
    pub fn took(&mut self, d: Duration) {
        let ms = d.as_secs_f64() * 1000.0;
        self.avg_ms = Some(match self.avg_ms {
            Some(a) => a + ALPHA * (ms - a),
            None => ms,
        });
        self.worst_ms = (self.worst_ms * FORGET).max(ms);
    }

    /// How early a change begins for its load, in milliseconds.
    pub fn lead_ms(&self) -> f64 {
        let avg = self.avg_ms.unwrap_or(0.0);
        avg.max(self.worst_ms).min(CAP_MS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The next preset's file is read ahead**, and taken once, for its own
    /// index only.
    #[test]
    fn the_next_file_is_read_ahead() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("next.milk");
        fs::write(&path, b"[preset00]").unwrap();
        let mut p = Prefetch::start(7, path);
        assert_eq!(p.take(6), None, "another preset's bytes are not these");
        let mut got = None;
        for _ in 0..200 {
            got = p.take(7);
            if got.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(got.as_deref(), Some(&b"[preset00]"[..]));
        assert_eq!(p.take(7), None, "taken once");
    }

    /// **A change begins as early as the loads have been slow**, the slowest
    /// of late counted, never past the cap.
    #[test]
    fn a_change_begins_as_early_as_loads_are_slow() {
        let mut t = LoadTime::default();
        assert_eq!(t.lead_ms(), 0.0);
        t.took(Duration::from_millis(100));
        assert_eq!(t.lead_ms(), 100.0);
        t.took(Duration::from_millis(400));
        assert_eq!(t.lead_ms(), 400.0, "the slowest of late");
        t.took(Duration::from_millis(100));
        assert_eq!(t.lead_ms(), 360.0, "the slowest fading");
        t.took(Duration::from_secs(10));
        assert_eq!(t.lead_ms(), CAP_MS);
    }
}
