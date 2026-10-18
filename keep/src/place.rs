//! The stable file, and the previous one set aside while a trial runs.
//!
//! **A trial runs from the stable path itself.** Windows Firewall allows
//! `web` and `motherd` by the path a process was started from, and a daemon
//! that relaunches itself starts that same path again — so the process left
//! running after a promotion must have been started from the stable file.
//! Starting the trial there is what lets a promotion leave it running: the
//! stable file it was started from is set aside first ([`set_aside`]), the
//! trial's bytes are installed in its place ([`install`]), and the trial is
//! started. Proven, the set-aside file is let go ([`discard_previous`]);
//! not, it is put back ([`restore`]).
//!
//! The set-aside file exists only while a trial is unproven, so a keep that
//! starts and finds one ([`previous_of`]) was stopped mid-trial and rolls it
//! back.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

/// How long a file a process just exited from may stay locked.
const UNLOCK: Duration = Duration::from_secs(5);

/// `stable` with `suffix` appended to its name.
fn beside(stable: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = stable.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Where the previous stable file waits while a trial runs.
pub fn previous_of(stable: &Path) -> PathBuf {
    beside(stable, ".prev")
}

/// Where an accepted build leaves the stable file it replaced, for a rollback
/// by hand. Not [`previous_of`]: that file's presence means a trial was cut
/// short, and a keep starting beside one would roll the accepted build back.
pub fn last_of(stable: &Path) -> PathBuf {
    beside(stable, ".last")
}

/// Accept `new` as the stable file with no trial: the stable file it replaces
/// is copied to [`last_of`], and `new`'s bytes installed. The process the
/// stable file was started from has been stopped, though its image may stay
/// locked a moment.
pub fn accept(new: &Path, stable: &Path) -> io::Result<()> {
    if stable.exists() {
        retrying(|| fs::copy(stable, last_of(stable)).map(|_| ()))?;
    }
    retrying(|| install(new, stable))
}

/// Move the stable file aside, for a trial to run from its path. Nothing to
/// set aside — no stable file yet — is not an error. The process it was
/// started from has been stopped, so it is not locked.
pub fn set_aside(stable: &Path) -> io::Result<()> {
    if !stable.exists() {
        return Ok(());
    }
    retrying(|| fs::rename(stable, previous_of(stable)))
}

/// Copy `trial` onto the stable path — written beside it first and then moved
/// over it, so a failure part way never leaves a torn stable file.
pub fn install(trial: &Path, stable: &Path) -> io::Result<()> {
    if same_file(trial, stable) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the trial file is the stable file",
        ));
    }
    if let Some(parent) = stable.parent() {
        fs::create_dir_all(parent)?;
    }
    let next = beside(stable, ".next");
    fs::copy(trial, &next)?;
    fs::rename(&next, stable)
}

/// Take the unproven trial's bytes off the stable path and put the set-aside
/// file back — none, for a first trial, leaves no stable file. The trial has
/// exited, though its image may stay locked a moment.
pub fn restore(stable: &Path) -> io::Result<()> {
    retrying(|| match fs::remove_file(stable) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    })?;
    let previous = previous_of(stable);
    if previous.exists() {
        fs::rename(previous, stable)?;
    }
    Ok(())
}

/// The trial is proven: the set-aside file is not needed — and must go, or
/// the next keep to start would read it as an unproven trial.
pub fn discard_previous(stable: &Path) -> io::Result<()> {
    let previous = previous_of(stable);
    retrying(|| match fs::remove_file(&previous) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    })
}

/// `op`, again until it succeeds or [`UNLOCK`] passes.
fn retrying(mut op: impl FnMut() -> io::Result<()>) -> io::Result<()> {
    let deadline = Instant::now() + UNLOCK;
    loop {
        match op() {
            Ok(()) => return Ok(()),
            Err(e) if Instant::now() >= deadline => return Err(e),
            Err(_) => thread::sleep(Duration::from_millis(50)),
        }
    }
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    if let (Ok(a), Ok(b)) = (fs::canonicalize(a), fs::canonicalize(b)) {
        return a == b;
    }
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "keep-place-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// **A trial installed and proven leaves its bytes on the stable path and
    /// nothing beside it**; the staged file is untouched.
    #[test]
    fn a_proven_trial_is_the_stable_file_and_nothing_is_set_aside() {
        let dir = scratch();
        let staged = dir.join("staged.exe");
        let stable = dir.join("stable").join("svc.exe");
        fs::write(&staged, b"new").unwrap();
        fs::create_dir_all(stable.parent().unwrap()).unwrap();
        fs::write(&stable, b"old").unwrap();

        set_aside(&stable).unwrap();
        install(&staged, &stable).unwrap();
        assert_eq!(fs::read(previous_of(&stable)).unwrap(), b"old");
        discard_previous(&stable).unwrap();

        assert_eq!(fs::read(&stable).unwrap(), b"new");
        assert_eq!(fs::read(&staged).unwrap(), b"new");
        assert!(!previous_of(&stable).exists());
        assert!(!beside(&stable, ".next").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// **A trial rolled back leaves the previous stable file where it was.**
    #[test]
    fn a_rolled_back_trial_puts_the_previous_file_back() {
        let dir = scratch();
        let staged = dir.join("staged.exe");
        let stable = dir.join("svc.exe");
        fs::write(&staged, b"new").unwrap();
        fs::write(&stable, b"old").unwrap();

        set_aside(&stable).unwrap();
        install(&staged, &stable).unwrap();
        restore(&stable).unwrap();

        assert_eq!(fs::read(&stable).unwrap(), b"old");
        assert!(!previous_of(&stable).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// **An accepted build is the stable file at once**, the one it replaced
    /// kept as `.last` — never `.prev`, which reads as a trial cut short.
    #[test]
    fn an_accepted_build_replaces_the_stable_file_and_keeps_the_last() {
        let dir = scratch();
        let staged = dir.join("staged.exe");
        let stable = dir.join("svc.exe");
        fs::write(&staged, b"new").unwrap();
        fs::write(&stable, b"old").unwrap();

        accept(&staged, &stable).unwrap();

        assert_eq!(fs::read(&stable).unwrap(), b"new");
        assert_eq!(fs::read(last_of(&stable)).unwrap(), b"old");
        assert!(!previous_of(&stable).exists());
        assert!(!beside(&stable, ".next").exists());

        // A first accept, with no stable file yet, keeps nothing as the last.
        let fresh = dir.join("fresh.exe");
        accept(&staged, &fresh).unwrap();
        assert_eq!(fs::read(&fresh).unwrap(), b"new");
        assert!(!last_of(&fresh).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// **A first trial rolled back leaves no stable file** — never the
    /// unproven trial's bytes standing in for one.
    #[test]
    fn a_first_trial_rolled_back_leaves_no_stable_file() {
        let dir = scratch();
        let staged = dir.join("staged.exe");
        let stable = dir.join("svc.exe");
        fs::write(&staged, b"new").unwrap();
        set_aside(&stable).unwrap();
        assert!(!previous_of(&stable).exists());
        install(&staged, &stable).unwrap();
        restore(&stable).unwrap();
        assert!(!stable.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
