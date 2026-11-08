//! **Git's lock files a killed git left in a checkout of the agent's own.**
//!
//! Git takes `index.lock`, `HEAD.lock`, a ref's `.lock` and the like by
//! creating the file, and removes it when the operation ends. A git killed part
//! way — an agent restarted under a running job — leaves the file behind, and
//! every later git that needs it refuses: `Unable to create '…/index.lock':
//! File exists`. Nothing in git removes one; it says to remove it by hand.
//!
//! **A lock is stale when no git that could have made it is still running.**
//! Git creates a lock only while it runs and refuses to create one that exists,
//! so the git holding a lock started before the lock was written. A lock with
//! no live git started at or before its write time has no owner. The lock's
//! time is read before the processes are listed, so its owner, if alive, is
//! among them. A git started since — the host's journal sync, a fetch — cannot
//! hold it, and is not mistaken for its owner.
//!
//! Run when the agent opens its checkouts: they are its own, cloned for it and
//! worked in by its jobs alone, and none of its jobs is running yet.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

/// The lock files git keeps at the top of its folder.
const TOP_LOCKS: [&str; 7] = [
    "index.lock",
    "HEAD.lock",
    "ORIG_HEAD.lock",
    "FETCH_HEAD.lock",
    "packed-refs.lock",
    "config.lock",
    "shallow.lock",
];

/// Process start times are whole seconds: a git started within this of a
/// lock's write time may have written it.
const START_GRAIN: Duration = Duration::from_secs(1);

/// Every lock file in the git folder `git_dir`: those at its top, and any
/// ref's under `refs/`.
fn locks_in(git_dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = TOP_LOCKS
        .iter()
        .map(|name| git_dir.join(name))
        .filter(|p| p.is_file())
        .collect();
    let mut dirs = vec![git_dir.join("refs")];
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => dirs.push(path),
                Ok(t) if t.is_file() && path.extension().is_some_and(|e| e == "lock") => {
                    out.push(path)
                }
                _ => {}
            }
        }
    }
    out
}

/// When each git process running now started: every process whose name
/// begins `git` — `git`, and the helpers it runs, `git-remote-https` and the
/// rest.
fn git_starts() -> Vec<SystemTime> {
    let mut sys = System::new();
    sys.refresh_processes_specifics(ProcessesToUpdate::All, ProcessRefreshKind::new());
    sys.processes()
        .values()
        .filter(|p| p.name().to_ascii_lowercase().starts_with("git"))
        .map(|p| UNIX_EPOCH + Duration::from_secs(p.start_time()))
        .collect()
}

/// Whether a lock written at `written` may belong to a git that started at one
/// of `starts`: one started no later than the write, to the grain of a start
/// time.
fn may_own(written: SystemTime, starts: &[SystemTime]) -> bool {
    starts.iter().any(|&s| s <= written + START_GRAIN)
}

/// What one sweep found: the locks removed, and those a running git may own.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Swept {
    pub removed: Vec<PathBuf>,
    pub held: Vec<PathBuf>,
}

/// Remove every lock of `written_of` — each with its write time — that no git
/// started at `starts` could own, and name those one could. A lock gone
/// meanwhile is neither.
fn sweep_with(written_of: &[(PathBuf, SystemTime)], starts: &[SystemTime]) -> Swept {
    let mut swept = Swept::default();
    for (lock, written) in written_of {
        if may_own(*written, starts) {
            swept.held.push(lock.clone());
            continue;
        }
        match std::fs::remove_file(lock) {
            Ok(()) => swept.removed.push(lock.clone()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                "{}: a stale git lock could not be removed: {e}",
                lock.display()
            ),
        }
    }
    swept
}

/// Clear the stale locks in the checkout at `dir`. A checkout whose `.git` is
/// not a folder holds no locks of its own here.
pub fn clear(dir: &Path) -> Swept {
    let git_dir = dir.join(".git");
    if !git_dir.is_dir() {
        return Swept::default();
    }
    // Each lock's time first, then the processes: the owner of a lock read
    // here started before it was written, so it is listed if it is alive.
    let written_of: Vec<(PathBuf, SystemTime)> = locks_in(&git_dir)
        .into_iter()
        .filter_map(|lock| {
            let written = std::fs::metadata(&lock).and_then(|m| m.modified()).ok()?;
            Some((lock, written))
        })
        .collect();
    if written_of.is_empty() {
        return Swept::default();
    }
    sweep_with(&written_of, &git_starts())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock_at(path: &Path) -> PathBuf {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(path).unwrap();
        path.to_path_buf()
    }

    /// **A git started before a lock was written may own it; one started
    /// after cannot**, to the grain of a start time.
    #[test]
    fn only_a_git_started_by_the_write_may_own_a_lock() {
        let written = UNIX_EPOCH + Duration::from_secs(1_000);
        let at = |s| UNIX_EPOCH + Duration::from_secs(s);
        assert!(may_own(written, &[at(900)]), "started before");
        assert!(may_own(written, &[at(1_001)]), "inside the grain");
        assert!(!may_own(written, &[at(1_120)]), "started after: AW's case");
        assert!(!may_own(written, &[]), "no git running");
    }

    /// **A lock nothing running could own goes; one a live git may own
    /// stays**, at the top of the git folder and under `refs/` alike, and a
    /// ref is never taken for a lock.
    #[test]
    fn unowned_locks_go_and_owned_ones_stay() {
        let dir = tempfile::tempdir().unwrap();
        let git = dir.path();
        let index = lock_at(&git.join("index.lock"));
        let remote = lock_at(&git.join("refs/remotes/origin/main.lock"));
        let head = lock_at(&git.join("HEAD.lock"));
        let not_a_lock = lock_at(&git.join("refs/heads/main"));
        let mut found = locks_in(git);
        found.sort();
        let mut want = vec![head.clone(), index.clone(), remote.clone()];
        want.sort();
        assert_eq!(found, want, "the ref itself is not listed");

        let t = |s| UNIX_EPOCH + Duration::from_secs(s);
        let written = vec![
            (index.clone(), t(1_000)),
            (remote.clone(), t(1_000)),
            (head.clone(), t(2_000)),
        ];
        // One git running, started at 1,500: after the first two were
        // written, before the third.
        let swept = sweep_with(&written, &[t(1_500)]);
        assert_eq!(swept.removed, vec![index.clone(), remote.clone()]);
        assert_eq!(swept.held, vec![head.clone()]);
        assert!(!index.exists() && !remote.exists());
        assert!(head.exists(), "a lock a live git may hold is left");
        assert!(not_a_lock.exists());
    }

    /// **A checkout with no locks is left as it is**, and one with no `.git`
    /// folder of its own is not looked into.
    #[test]
    fn nothing_to_clear_is_nothing_cleared() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(clear(dir.path()), Swept::default(), "no .git folder");
        std::fs::create_dir_all(dir.path().join(".git/refs/heads")).unwrap();
        assert_eq!(clear(dir.path()), Swept::default(), "no locks");
    }
}
