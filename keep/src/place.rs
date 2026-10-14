//! The trial file becomes the stable file. A rename is preferred: the file
//! moves onto the stable path and nothing is left behind. A running process
//! keeps the path it was started from; the caller starts the stable path
//! after that process exits. When the rename fails, the bytes are copied onto
//! the stable path and the trial file is deleted once it is no longer locked.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Which way the trial file became the stable file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum How {
    /// The running file was renamed onto the stable path.
    Rename,
    /// The running file could not be renamed. Its bytes were copied, and the
    /// trial file is removed once it is no longer locked.
    Copy,
}

impl std::fmt::Display for How {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            How::Rename => "rename",
            How::Copy => "copy",
        })
    }
}

/// Move `trial` onto `stable`. `trial` may be a running executable.
pub fn promote(trial: &Path, stable: &Path) -> io::Result<How> {
    if same_file(trial, stable) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the trial file is the stable file",
        ));
    }
    if let Some(parent) = stable.parent() {
        fs::create_dir_all(parent)?;
    }
    // A rename onto an existing path fails, which would hide the question of
    // whether the running image itself can be renamed. The previous stable
    // file is not the running image.
    if stable.exists() && fs::remove_file(stable).is_err() {
        return copy_onto(trial, stable);
    }
    match fs::rename(trial, stable) {
        Ok(()) => Ok(How::Rename),
        Err(_) => copy_onto(trial, stable),
    }
}

fn copy_onto(trial: &Path, stable: &Path) -> io::Result<How> {
    fs::copy(trial, stable)?;
    delete_when_unlocked(trial.to_path_buf());
    Ok(How::Copy)
}

/// Remove `path` as soon as it is not locked. A running image stays locked
/// until its process exits, so the retries outlive the call.
fn delete_when_unlocked(path: PathBuf) {
    if remove_if_free(&path) {
        return;
    }
    let _ = thread::Builder::new()
        .name("keep-unlink".into())
        .spawn(move || {
            for _ in 0..(24 * 60 * 60) {
                thread::sleep(Duration::from_secs(1));
                if remove_if_free(&path) {
                    return;
                }
            }
        });
}

fn remove_if_free(path: &Path) -> bool {
    match fs::remove_file(path) {
        Ok(()) => true,
        Err(e) if e.kind() == io::ErrorKind::NotFound => true,
        Err(_) => false,
    }
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    if let (Ok(a), Ok(b)) = (fs::canonicalize(a), fs::canonicalize(b)) {
        return a == b;
    }
    a == b
}

/// `rename` or `copy`, and for a copy the trial path still to be deleted.
pub fn record(path: &Path, how: How, trial: &Path) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let body = match how {
        How::Rename => "rename\n".to_string(),
        How::Copy => format!("copy\n{}\n", trial.display()),
    };
    fs::write(path, body)
}

/// The trial path a previous copy still has to delete, when the place file
/// records one.
pub fn pending_copy(path: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    if lines.next()? != "copy" {
        return None;
    }
    let trial = lines.next()?;
    if trial.is_empty() {
        None
    } else {
        Some(PathBuf::from(trial))
    }
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

    #[test]
    fn a_free_file_is_renamed_and_nothing_is_left_behind() {
        let dir = scratch();
        let trial = dir.join("trial.bin");
        let stable = dir.join("stable.bin");
        fs::write(&trial, b"new-bytes").unwrap();
        fs::write(&stable, b"old").unwrap();
        let how = promote(&trial, &stable).unwrap();
        assert_eq!(how, How::Rename);
        assert!(!trial.exists());
        assert_eq!(fs::read(&stable).unwrap(), b"new-bytes");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The question this records: can a running executable be renamed on this
    /// machine? Either answer is a working promotion. The place is whichever
    /// one happened, and the stable path holds the trial's bytes.
    #[cfg(windows)]
    #[test]
    fn a_running_executable_is_promoted_onto_the_stable_path() {
        let ping = Path::new(r"C:\Windows\System32\ping.exe");
        if !ping.is_file() {
            return;
        }
        let dir = scratch();
        let trial = dir.join("trial.exe");
        let stable = dir.join("stable.exe");
        fs::copy(ping, &trial).unwrap();
        fs::write(&stable, b"previous").unwrap();
        let bytes = fs::read(&trial).unwrap();
        let mut child = std::process::Command::new(&trial)
            .args(["-n", "30", "127.0.0.1"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_millis(200));
        assert!(child.try_wait().unwrap().is_none());
        let how = promote(&trial, &stable).unwrap();
        assert_eq!(fs::read(&stable).unwrap(), bytes);
        assert!(child.try_wait().unwrap().is_none());
        match how {
            How::Rename => assert!(!trial.exists()),
            How::Copy => assert!(trial.exists()),
        }
        eprintln!("running executable promotion: {how}");
        let _ = child.kill();
        let _ = child.wait();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while trial.exists() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(100));
        }
        assert!(!trial.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
