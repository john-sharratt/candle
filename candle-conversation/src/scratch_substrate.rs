//! A throwaway substrate directory that cleans itself up — including after a
//! crash.
//!
//! # Why this exists
//!
//! A substrate is opened by path, and the default path is the repository's own
//! `.substrate`. Anything that opens one without saying otherwise therefore
//! attaches to the *live* store: a measurement harness appends its conversations
//! to the user's real history, a test that wipes on exit wipes the user's real
//! history, and a 3.2 GB `.substrate` once ended up committed to the tree because
//! a test's workspace defaulted to the crate directory. So a harness must name a
//! scratch path explicitly, and this is the type that owns one.
//!
//! # Deleted while still open, so a crash cannot leave the data behind
//!
//! Three layers, strongest first.
//!
//! **1. The kernel deletes the files, even on a crash.** Every file the engine
//! creates under the scratch directory gets a second handle opened with
//! `FILE_FLAG_DELETE_ON_CLOSE` and full `FILE_SHARE_*`, held by [`Janitor`]. The
//! file is then deleted by Windows when its *last* handle closes — and a crash
//! closes every handle, so a `SIGKILL`, a CUDA fault or `panic = "abort"` all
//! reclaim the bytes. Nothing has to run for this to work.
//!
//! This is the property POSIX `unlink`-while-open does **not** have, and the
//! reason the platforms are treated differently. Without
//! `FILE_DISPOSITION_POSIX_SEMANTICS`, Windows keeps the *name* in the directory
//! for as long as a handle is open and removes it at last close. So the engine can
//! still close and reopen its redo log **by name** while the deletion is pending —
//! whereas a POSIX `unlink` removes the name immediately and the engine's next
//! reopen would create a second, empty log. On Unix this layer is therefore off,
//! and the two below carry it.
//!
//! **2. Drop removes the directory**, on the ordinary path and on unwind.
//!
//! **3. Construction sweeps the corpses of previous runs**, matched on one parent
//! and one recognisable prefix.
//!
//! So on Windows a crash leaves at most an empty directory, and on Unix at most
//! one run's corpse, which the next run removes. A [`tempfile::TempDir`] alone
//! gives layer 2 only.
//!
//! # It can only ever delete its own
//!
//! Layer 1 is the dangerous one — it asks the kernel to destroy files — so it is
//! fenced twice. [`Janitor`] only ever walks the directory this type created, and
//! [`claim_delete_on_close`] refuses any path whose parent directory name does not
//! start with [`PREFIX`]. A real `.substrate` cannot satisfy that, so pointing
//! this at one is a refusal rather than a data loss.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Directory-name prefix that marks a scratch substrate, and the key to the
/// crash sweep: anything under the parent starting with this belongs to a run of
/// this harness and nothing else, so deleting a stale one cannot destroy data
/// somebody wanted.
const PREFIX: &str = "candle-scratch-substrate-";

/// File written at creation whose mtime dates the scratch directory.
///
/// The directory's own mtime would be the obvious signal and is not portably
/// writable — Windows needs backup semantics to open a directory handle at all,
/// so a test could not age one. A regular file's mtime is writable everywhere,
/// which makes the staleness rule testable rather than merely plausible.
const MARKER: &str = "scratch-created";

/// A scratch substrate directory, removed on drop and swept on creation.
///
/// Hold it for as long as the engine is open — dropping it deletes the store out
/// from under a live engine, which is the one misuse the type cannot prevent.
pub struct ScratchSubstrate {
    path: PathBuf,
    /// Corpses of earlier runs removed when this one was created, for the
    /// harness's own log line: a non-zero count means something died last time,
    /// which is worth saying out loud rather than silently tidying.
    swept: usize,
    /// Holds a delete-on-close handle to every file under `path`, so the kernel
    /// reclaims them even if this process never runs another instruction.
    janitor: Option<Janitor>,
}

/// Holds a `FILE_FLAG_DELETE_ON_CLOSE` handle to each file under one scratch
/// directory, claiming new ones as the engine creates them.
///
/// A poll rather than `ReadDirectoryChangesW`: the set is a handful of files that
/// appear within the first seconds, the cost is one `read_dir` a quarter second,
/// and the notification API would add a platform surface for no gain a harness can
/// measure.
struct Janitor {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Every handle claimed. Dropping these is what triggers the deletions on the
    /// ordinary path; a crash gets the same effect from the kernel.
    held: Arc<Mutex<Vec<fs::File>>>,
}

impl Janitor {
    /// Start watching `dir`. Only ever called with a directory this module
    /// created — see [`claim_delete_on_close`] for the check that enforces it.
    fn start(dir: PathBuf) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let held: Arc<Mutex<Vec<fs::File>>> = Arc::new(Mutex::new(Vec::new()));
        let thread = {
            let stop = Arc::clone(&stop);
            let held = Arc::clone(&held);
            std::thread::Builder::new()
                .name("scratch-janitor".into())
                .spawn(move || {
                    let mut claimed: Vec<PathBuf> = Vec::new();
                    while !stop.load(Ordering::Relaxed) {
                        claim_new_files(&dir, &mut claimed, &held);
                        std::thread::sleep(std::time::Duration::from_millis(250));
                    }
                    // One last pass: a file created between the final poll and
                    // the stop still wants a handle, because the process may be
                    // about to die rather than to drop cleanly.
                    claim_new_files(&dir, &mut claimed, &held);
                })
                .ok()
        };
        Self { stop, thread, held }
    }
}

impl Drop for Janitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        // Release the handles so the pending deletions complete before
        // `remove_dir_all` runs: on Windows a directory holding a
        // delete-pending file cannot be removed, so dropping these first is what
        // lets the ordinary cleanup path succeed.
        if let Ok(mut held) = self.held.lock() {
            held.clear();
        }
    }
}

/// Claim a delete-on-close handle for every file under `dir` not already held.
fn claim_new_files(dir: &Path, claimed: &mut Vec<PathBuf>, held: &Arc<Mutex<Vec<fs::File>>>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || claimed.contains(&path) {
            continue;
        }
        // The marker is ours and is read by the next run's sweep; a handle on it
        // would have it vanish on a crash and hide the corpse from that sweep.
        if path.file_name().is_some_and(|n| n == MARKER) {
            claimed.push(path);
            continue;
        }
        match claim_delete_on_close(&path) {
            Ok(Some(f)) => {
                if let Ok(mut h) = held.lock() {
                    h.push(f);
                }
                claimed.push(path);
            }
            // A sharing violation means whoever holds it did not permit delete.
            // Best effort: layers 2 and 3 still cover the file.
            Ok(None) | Err(_) => claimed.push(path),
        }
    }
}

/// Open `path` with delete-on-close, so the kernel removes it when the last
/// handle closes.
///
/// `Ok(None)` on a platform where this would break the engine rather than help it
/// — see the module note on why Unix is excluded.
///
/// **Refuses any path not inside a scratch directory.** The check is on the parent
/// directory's name carrying [`PREFIX`], which only a directory this module
/// created can satisfy, so the one operation here that destroys data cannot be
/// aimed at a real `.substrate` by a mistaken caller.
fn claim_delete_on_close(path: &Path) -> std::io::Result<Option<fs::File>> {
    let in_scratch = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with(PREFIX));
    if !in_scratch {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "refusing a delete-on-close handle outside a scratch substrate: {}",
                path.display()
            ),
        ));
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Share every mode, or the engine's own open fails while we hold this.
        const FILE_SHARE_READ: u32 = 0x0000_0001;
        const FILE_SHARE_WRITE: u32 = 0x0000_0002;
        const FILE_SHARE_DELETE: u32 = 0x0000_0004;
        const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
        let f = fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_DELETE_ON_CLOSE)
            .open(path)?;
        Ok(Some(f))
    }
    #[cfg(not(windows))]
    {
        // Deliberately nothing. The Unix equivalent — `unlink` now, keep the
        // handle — removes the name immediately, and the engine closes and
        // reopens its redo log by name, so its next reopen would silently create a
        // second empty log. Layers 2 and 3 cover this platform.
        let _ = path;
        Ok(None)
    }
}

impl ScratchSubstrate {
    /// Create a fresh scratch substrate under the system temp directory, first
    /// removing any left by a previous run.
    pub fn new() -> std::io::Result<Self> {
        Self::under(&std::env::temp_dir())
    }

    /// As [`Self::new`], with an explicit parent — for a test that wants the
    /// sweep observable, or a machine whose temp directory is on the wrong disk.
    pub fn under(parent: &Path) -> std::io::Result<Self> {
        fs::create_dir_all(parent)?;
        let swept = sweep(parent);

        // Process id and a monotonic clock reading: unique across concurrent
        // runs on one machine, and unique across runs of one pid after a reboot
        // recycles it.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let path = parent.join(format!("{PREFIX}{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path)?;
        // The age marker the sweep reads. A file rather than the directory's own
        // mtime because a directory's timestamp is not portably writable — on
        // Windows opening one needs backup semantics — and because a file's mtime
        // is something a future heartbeat could refresh.
        fs::write(path.join(MARKER), std::process::id().to_string())?;
        let janitor = Some(Janitor::start(path.clone()));
        Ok(Self {
            path,
            swept,
            janitor,
        })
    }

    /// The directory to point a substrate at.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How many corpses of earlier runs this one removed on creation.
    pub fn swept(&self) -> usize {
        self.swept
    }
}

impl Drop for ScratchSubstrate {
    fn drop(&mut self) {
        // The janitor FIRST: its handles hold pending deletions, and on Windows a
        // directory containing a delete-pending file cannot be removed. Dropping
        // it releases those handles, the files go, and the directory is then
        // removable.
        self.janitor = None;
        // Best effort, and deliberately silent on failure: a harness that has
        // finished measuring should not fail because Windows still held a handle
        // on the log for a moment. The next run's sweep collects it.
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A scratch directory younger than this is assumed to belong to a run that is
/// still going, and is left alone.
///
/// **Why an age and not a liveness check.** The sweep has to distinguish a corpse
/// from a sibling, and asking the OS whether a pid is alive is neither portable
/// nor reliable (pids are recycled). An age is both, and the window only has to
/// be longer than the gap between a process dying and the next one starting,
/// which for anything that loads a model is dominated by the load itself.
///
/// The cost of the window is stated rather than hidden: a run that crashes and is
/// restarted within a minute leaves its corpse for the run after that.
const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(60);

/// Remove scratch substrates under `parent` that are old enough to be corpses,
/// returning how many went.
///
/// Two conditions, and both are load-bearing. Matching on [`PREFIX`] means the
/// sweep cannot reach a directory it did not create, so pointing a harness at a
/// populated parent deletes nothing of that parent's own. Requiring an age of at
/// least [`STALE_AFTER`] means it cannot reach a *live* sibling either — without
/// that, creating a second scratch would delete the first one's store out from
/// under a running engine, which is a far worse failure than a leaked directory.
fn sweep(parent: &Path) -> usize {
    let Ok(entries) = fs::read_dir(parent) else {
        return 0;
    };
    let now = std::time::SystemTime::now();
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(PREFIX) || !entry.path().is_dir() {
            continue;
        }
        // A missing marker, unreadable metadata, or a clock that moved backwards
        // all read as "not provably stale" and are left alone. Leaking a
        // directory is recoverable; deleting a live store is not, so every
        // uncertainty resolves toward leaving it.
        let stale = fs::metadata(entry.path().join(MARKER))
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .is_some_and(|age| age >= STALE_AFTER);
        if stale && fs::remove_dir_all(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_creates_a_directory_and_removes_it_on_drop() {
        let parent = tempfile::tempdir().unwrap();
        let path = {
            let s = ScratchSubstrate::under(parent.path()).unwrap();
            assert!(s.path().is_dir(), "the scratch directory must exist");
            s.path().to_path_buf()
        };
        assert!(!path.exists(), "drop must remove the scratch directory");
    }

    /// The crash half: a directory left behind by a process that never ran its
    /// destructors is collected by a later run.
    ///
    /// Ages the corpse past [`STALE_AFTER`] by setting its mtime, rather than by
    /// sleeping a minute — the rule under test is "old enough", and the clock is
    /// an input to it.
    #[test]
    fn it_sweeps_a_corpse_left_by_a_previous_run() {
        let parent = tempfile::tempdir().unwrap();
        // Forge a corpse: the same prefix, a pid that is not ours.
        let corpse = parent.path().join(format!("{PREFIX}999999-123"));
        fs::create_dir_all(corpse.join("nested")).unwrap();
        fs::write(corpse.join("substrate.log"), b"stale").unwrap();
        fs::write(corpse.join(MARKER), b"999999").unwrap();
        age_past_grace(&corpse.join(MARKER));

        let s = ScratchSubstrate::under(parent.path()).unwrap();
        assert_eq!(s.swept(), 1, "the corpse must be counted");
        assert!(!corpse.exists(), "the corpse must be gone");
        assert!(s.path().is_dir(), "and ours must exist");
    }

    /// A corpse younger than the grace window is NOT swept — it might be a live
    /// sibling, and deleting a live store is worse than leaking a directory.
    #[test]
    fn a_fresh_scratch_is_never_swept() {
        let parent = tempfile::tempdir().unwrap();
        let fresh = parent.path().join(format!("{PREFIX}999999-456"));
        fs::create_dir_all(&fresh).unwrap();
        fs::write(fresh.join(MARKER), b"999999").unwrap();

        let s = ScratchSubstrate::under(parent.path()).unwrap();
        assert_eq!(s.swept(), 0, "a fresh sibling must be left alone");
        assert!(fresh.is_dir());
    }

    /// A directory with no marker is left alone however old it looks — the
    /// uncertain case resolves toward not deleting.
    #[test]
    fn a_directory_with_no_marker_is_left_alone() {
        let parent = tempfile::tempdir().unwrap();
        let odd = parent.path().join(format!("{PREFIX}1-2"));
        fs::create_dir_all(&odd).unwrap();
        let s = ScratchSubstrate::under(parent.path()).unwrap();
        assert_eq!(s.swept(), 0);
        assert!(odd.is_dir());
    }

    /// **The kernel deletes the file when the last handle closes.**
    ///
    /// This is layer 1, and it is the property a crash relies on: nothing runs,
    /// the process's handles close, the bytes go. Simulated by dropping the handle
    /// rather than by killing a process — the mechanism is the same, and it is the
    /// handle close that does the work.
    #[cfg(windows)]
    #[test]
    fn a_claimed_file_is_deleted_when_its_handle_closes() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join(format!("{PREFIX}1-1"));
        fs::create_dir_all(&dir).unwrap();
        let victim = dir.join("substrate.log");
        fs::write(&victim, b"some records").unwrap();

        let handle = claim_delete_on_close(&victim)
            .expect("claim must succeed inside a scratch dir")
            .expect("windows must return a handle");
        assert!(
            victim.exists(),
            "the NAME must survive while a handle is open — the engine reopens \
             its log by name while the deletion is pending",
        );
        // Still writable and readable by others: full FILE_SHARE_*.
        assert!(fs::read(&victim).is_ok(), "shared for reading");

        drop(handle);
        assert!(
            !victim.exists(),
            "the last handle closing must delete the file",
        );
    }

    /// **It cannot be aimed at a real substrate.** The one operation here that
    /// destroys data refuses any path whose parent is not a scratch directory, so
    /// a mistaken caller gets an error rather than a data loss.
    #[test]
    fn a_delete_on_close_claim_refuses_a_path_outside_a_scratch_dir() {
        let parent = tempfile::tempdir().unwrap();
        let real = parent.path().join(".substrate");
        fs::create_dir_all(&real).unwrap();
        let precious = real.join("substrate.log");
        fs::write(&precious, b"the user's real history").unwrap();

        let err = claim_delete_on_close(&precious).expect_err("must refuse");
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(precious.exists(), "and must not have touched it");

        // A bare file with no parent-directory prefix is refused too.
        let loose = parent.path().join("loose.log");
        fs::write(&loose, b"x").unwrap();
        assert!(claim_delete_on_close(&loose).is_err());
        assert!(loose.exists());
    }

    /// The janitor claims a file the "engine" creates after construction, and the
    /// whole directory is still removed on drop — the layers compose rather than
    /// fighting (a delete-pending file would otherwise block `remove_dir_all`).
    #[test]
    fn the_janitor_claims_later_files_and_drop_still_cleans_up() {
        let parent = tempfile::tempdir().unwrap();
        let path = {
            let s = ScratchSubstrate::under(parent.path()).unwrap();
            // Stand in for the engine opening its log after the engine is built.
            fs::write(s.path().join("substrate.log"), b"records").unwrap();
            // Give the poll a turn to notice it.
            std::thread::sleep(std::time::Duration::from_millis(600));
            s.path().to_path_buf()
        };
        assert!(
            !path.exists(),
            "drop must still remove the directory with a claimed file inside",
        );
    }

    /// Backdate a FILE's mtime past the grace window.
    fn age_past_grace(file: &Path) {
        let when = std::time::SystemTime::now() - STALE_AFTER - std::time::Duration::from_secs(30);
        let f = fs::File::options()
            .write(true)
            .open(file)
            .expect("open marker for mtime");
        f.set_times(fs::FileTimes::new().set_modified(when))
            .expect("backdate mtime");
    }

    /// **The sweep may only touch what it made.** A harness pointed at a
    /// populated directory must not delete that directory's own contents, which
    /// is the difference between a cleanup and a data loss.
    #[test]
    fn the_sweep_never_touches_a_directory_it_did_not_create() {
        let parent = tempfile::tempdir().unwrap();
        let precious = parent.path().join("someones-real-data");
        fs::create_dir_all(&precious).unwrap();
        fs::write(precious.join("keep.txt"), b"important").unwrap();
        let also = parent.path().join(".substrate");
        fs::create_dir_all(&also).unwrap();

        let s = ScratchSubstrate::under(parent.path()).unwrap();
        assert_eq!(s.swept(), 0);
        assert!(
            precious.join("keep.txt").exists(),
            "unrelated data survived"
        );
        assert!(also.is_dir(), "a real .substrate beside it survived");
    }

    /// Two live scratch substrates never collide, so concurrent harnesses do not
    /// share a store — and creating the second does not sweep the first.
    ///
    /// This is the case the age grace exists for: without it the second
    /// construction would delete the first's store while its engine was open.
    #[test]
    fn two_live_scratches_are_distinct_and_do_not_sweep_each_other() {
        let parent = tempfile::tempdir().unwrap();
        let a = ScratchSubstrate::under(parent.path()).unwrap();
        let b = ScratchSubstrate::under(parent.path()).unwrap();
        assert_ne!(a.path(), b.path());
        assert_eq!(b.swept(), 0);
        assert!(
            a.path().is_dir(),
            "creating the second must not sweep the first — it is still live",
        );
    }
}
