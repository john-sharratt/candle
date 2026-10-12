//! Whether an agent takes jobs (`docs/replicated_substrate.md` §11.7): an
//! admin's switch, kept as a [`DISABLED_FILE`] beside the job logs so an agent
//! switched off stays off across a restart.
//!
//! Off, its `CAPACITY` reports it `disabled`, so no gateway routes it a job;
//! a job sent to it anyway — by a member cut off from every cluster, reaching
//! the agent in its own process — is refused. The jobs it is running carry on
//! to their end, and every job it kept can still be read.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// The file whose presence switches an agent off.
pub const DISABLED_FILE: &str = "DISABLED";

pub struct Switch {
    path: PathBuf,
    on: AtomicBool,
}

impl Switch {
    /// The switch kept in `dir`: on unless its [`DISABLED_FILE`] is there.
    pub fn open(dir: &Path) -> Switch {
        let path = dir.join(DISABLED_FILE);
        let on = !path.exists();
        Switch {
            path,
            on: AtomicBool::new(on),
        }
    }

    /// Whether the agent takes jobs.
    pub fn on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// Switch the agent on or off, and keep it so.
    pub fn set(&self, on: bool) -> io::Result<()> {
        if on {
            match fs::remove_file(&self.path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        } else {
            if let Some(dir) = self.path.parent() {
                fs::create_dir_all(dir)?;
            }
            fs::write(&self.path, b"")?;
        }
        self.on.store(on, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A new agent is on; switched off it stays off across a restart**, and
    /// switched back on it leaves no file behind.
    #[test]
    fn the_switch_is_kept_beside_the_logs() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = dir.path().join("jobs");
        assert!(Switch::open(&jobs).on());

        Switch::open(&jobs).set(false).unwrap();
        assert!(jobs.join(DISABLED_FILE).is_file());
        let reopened = Switch::open(&jobs);
        assert!(!reopened.on());

        reopened.set(true).unwrap();
        assert!(reopened.on());
        assert!(!jobs.join(DISABLED_FILE).exists());
        assert!(Switch::open(&jobs).on());
        // On twice is still on: a missing file is not an error.
        reopened.set(true).unwrap();
    }
}
