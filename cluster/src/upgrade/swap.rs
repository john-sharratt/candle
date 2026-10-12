//! The in-place swap (§3.9): the running executable moves into its kept-build
//! directory, and the new build takes its path — so the relaunch, with the same
//! arguments in the same working directory, starts the new build without
//! anything about the invocation changing. The UI a build serves is compiled
//! into its executable, so the executable is the whole build.
//!
//! Every move is a rename on the executable's own volume, which Windows allows
//! for a running image. The new executable is copied beside its path first and
//! renamed into place, so an artifact on another volume swaps the same way.
//!
//! [`restore`] undoes a swap: the watchdog runs it when the new build fails, and
//! the old build is back at its path, its kept directory gone, as before. A
//! swap from a kept build puts that build back in `builds/` when restored.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::kept::{builds_dir, name, read_json, write_json, BuildJson, FileSum};
use super::sum::{file_sha256, hex};
use crate::version::info::BuildInfo;

/// The build running now, and where it is.
#[derive(Debug, Clone)]
pub struct Running<'a> {
    pub exe: &'a Path,
    pub info: &'a BuildInfo,
    pub installed_ms: u64,
}

/// Where the new build comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A verified download of the executable.
    Fetched(PathBuf),
    /// A build kept in `builds/`, by its directory name.
    Kept(String),
}

/// A finished swap, as [`restore`] needs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Swapped {
    pub exe: PathBuf,
    /// The kept directory the running build moved into.
    pub kept: PathBuf,
    /// The kept build the new one came out of, and its `build.json`: a restore
    /// puts it back.
    pub from_kept: Option<(PathBuf, BuildJson)>,
}

fn exe_name(exe: &Path) -> io::Result<String> {
    exe.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the executable has no name"))
}

/// The `build.json` of the executable at `exe`.
fn describe(
    exe: &Path,
    info: &BuildInfo,
    installed_ms: u64,
    retired_ms: u64,
) -> io::Result<BuildJson> {
    Ok(BuildJson {
        info: info.clone(),
        files: vec![FileSum {
            path: exe_name(exe)?,
            sha256: hex(&file_sha256(exe)?),
        }],
        installed_ms,
        retired_ms,
    })
}

/// Move the executable at `exe` into `dir`, with `json`.
fn keep_into(dir: &Path, exe: &Path, json: &BuildJson) -> io::Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir)?;
    fs::rename(exe, dir.join(exe_name(exe)?))?;
    write_json(dir, json)
}

/// Move the build kept in `dir` back to `exe`, and remove the directory.
fn take_out(dir: &Path, exe: &Path) -> io::Result<()> {
    fs::rename(dir.join(exe_name(exe)?), exe)?;
    fs::remove_dir_all(dir)
}

/// Copy `from` beside `exe` and rename it into place.
fn place_exe(from: &Path, exe: &Path) -> io::Result<()> {
    let staged = exe.with_file_name(format!("{}.new", exe_name(exe)?));
    fs::copy(from, &staged)?;
    fs::rename(staged, exe)
}

/// Swap `running` for `source` at `now_ms`.
pub fn swap(running: &Running<'_>, source: &Source, now_ms: u64) -> io::Result<Swapped> {
    let builds = builds_dir(running.exe);
    let kept = builds.join(name(&running.info.build()));
    let json = describe(running.exe, running.info, running.installed_ms, now_ms)?;
    // Checked before anything moves: a source that cannot be placed leaves
    // the running build where it is.
    let from_kept = match source {
        Source::Fetched(exe) => {
            if !exe.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("no artifact at {}", exe.display()),
                ));
            }
            keep_into(&kept, running.exe, &json)?;
            place_exe(exe, running.exe)?;
            None
        }
        Source::Kept(n) => {
            let dir = builds.join(n);
            let theirs = read_json(&dir)?;
            keep_into(&kept, running.exe, &json)?;
            take_out(&dir, running.exe)?;
            Some((dir, theirs))
        }
    };
    Ok(Swapped {
        exe: running.exe.to_path_buf(),
        kept,
        from_kept,
    })
}

/// Undo `s`: the build swapped in leaves — back into `builds/` when it came
/// from there — and the build swapped out returns to its path.
pub fn restore(s: &Swapped) -> io::Result<()> {
    match &s.from_kept {
        Some((dir, json)) => keep_into(dir, &s.exe, json)?,
        None => {
            if s.exe.exists() {
                fs::remove_file(&s.exe)?;
            }
        }
    }
    take_out(&s.kept, &s.exe)
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    use crate::upgrade::kept::list;
    use crate::upgrade::sum::sha256;
    use crate::version::info::{Opens, ZEND};

    fn info(count: u32) -> BuildInfo {
        BuildInfo {
            kind: ZEND.into(),
            count,
            hash: format!("h{count}"),
            local: String::new(),
            platform: "windows-x86_64".into(),
            opens: Opens::default(),
        }
    }

    /// A machine: `bin/zend`.
    fn machine(exe_text: &str) -> (TempDir, PathBuf) {
        let t = TempDir::new().unwrap();
        fs::create_dir_all(t.path().join("bin")).unwrap();
        let exe = t.path().join("bin/zend");
        fs::write(&exe, exe_text).unwrap();
        (t, exe)
    }

    fn read_text(p: &Path) -> String {
        fs::read_to_string(p).unwrap()
    }

    /// A fetched build takes the executable's path; the old build is kept
    /// with its `build.json`.
    #[test]
    fn a_fetched_build_takes_the_path_and_the_old_one_is_kept() {
        let (t, exe) = machine("old exe");
        let new_exe = t.path().join("new-zend");
        fs::write(&new_exe, "new exe").unwrap();
        let running = Running {
            exe: &exe,
            info: &info(10),
            installed_ms: 5,
        };
        let s = swap(&running, &Source::Fetched(new_exe), 100).unwrap();
        assert_eq!(read_text(&exe), "new exe");
        let kept = t.path().join("bin/builds/10-h10");
        assert_eq!(s.kept, kept);
        assert_eq!(read_text(&kept.join("zend")), "old exe");
        let json = read_json(&kept).unwrap();
        assert_eq!(
            (json.info, json.installed_ms, json.retired_ms),
            (info(10), 5, 100)
        );
        assert_eq!(
            json.files,
            vec![FileSum {
                path: "zend".into(),
                sha256: hex(&sha256(b"old exe")),
            }]
        );
        assert_eq!(list(&t.path().join("bin/builds")).len(), 1);
    }

    /// A restore puts the old build back where it was and leaves no kept
    /// directory.
    #[test]
    fn a_restore_puts_the_old_build_back() {
        let (t, exe) = machine("old exe");
        let new_exe = t.path().join("new-zend");
        fs::write(&new_exe, "new exe").unwrap();
        let running = Running {
            exe: &exe,
            info: &info(10),
            installed_ms: 5,
        };
        let s = swap(&running, &Source::Fetched(new_exe), 100).unwrap();
        restore(&s).unwrap();
        assert_eq!(read_text(&exe), "old exe");
        assert!(list(&t.path().join("bin/builds")).is_empty());
    }

    /// A rollback swaps from a kept build: it takes the path and leaves
    /// `builds/`, and the running build goes there in turn; restored, both
    /// are back where they were.
    #[test]
    fn a_rollback_swaps_with_a_kept_build_and_restores() {
        let (t, exe) = machine("build 12");
        // Build 10 kept, as a swap would have kept it.
        let kept10 = t.path().join("bin/builds/10-h10");
        fs::create_dir_all(&kept10).unwrap();
        fs::write(kept10.join("zend"), "build 10").unwrap();
        let json10 = BuildJson {
            info: info(10),
            files: vec![],
            installed_ms: 1,
            retired_ms: 2,
        };
        write_json(&kept10, &json10).unwrap();

        let running = Running {
            exe: &exe,
            info: &info(12),
            installed_ms: 3,
        };
        let s = swap(&running, &Source::Kept("10-h10".into()), 50).unwrap();
        assert_eq!(read_text(&exe), "build 10");
        let names: Vec<String> = list(&t.path().join("bin/builds"))
            .into_iter()
            .map(|k| k.name)
            .collect();
        assert_eq!(names, ["12-h12"]);

        restore(&s).unwrap();
        assert_eq!(read_text(&exe), "build 12");
        let back = list(&t.path().join("bin/builds"));
        assert_eq!(back.len(), 1);
        assert_eq!((back[0].name.as_str(), &back[0].json), ("10-h10", &json10));
        assert_eq!(read_text(&kept10.join("zend")), "build 10");
    }

    /// A source that cannot be placed leaves everything where it was.
    #[test]
    fn a_missing_artifact_moves_nothing() {
        let (t, exe) = machine("old exe");
        let running = Running {
            exe: &exe,
            info: &info(10),
            installed_ms: 5,
        };
        let gone = t.path().join("no-such-artifact");
        let err = swap(&running, &Source::Fetched(gone.clone()), 100).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("no artifact at {}", gone.display())
        );
        assert_eq!(read_text(&exe), "old exe");
        assert!(!t.path().join("bin/builds").exists());
    }
}
