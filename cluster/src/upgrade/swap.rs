//! The in-place swap (§3.9): the running executable and the UI it serves move
//! into its kept-build directory, and the new build takes their paths — so the
//! relaunch, with the same arguments in the same working directory, starts the
//! new build without anything about the invocation changing.
//!
//! Every move is a rename on the executable's own volume, which Windows allows
//! for a running image. The new executable is copied beside its path first and
//! renamed into place, so an artifact on another volume swaps the same way.
//!
//! [`restore`] undoes a swap: the watchdog runs it when the new build fails, and
//! the old build is back at its paths, its kept directory gone, as before. A
//! swap from a kept build puts that build back in `builds/` when restored.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::kept::{builds_dir, name, read_json, write_json, BuildJson, FileSum, UI};
use super::sum::{file_sha256, hex, sha256};
use super::ui_archive::{entries, pack, read, Entry};
use crate::version::info::BuildInfo;

/// The build running now, and where it is.
#[derive(Debug, Clone)]
pub struct Running<'a> {
    pub exe: &'a Path,
    /// The directory the UI names are relative to: the working directory.
    pub root: &'a Path,
    /// The UI directories it serves, by name: `zend/web`.
    pub ui: &'a [String],
    pub info: &'a BuildInfo,
    pub installed_ms: u64,
}

/// Where the new build comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A verified download: the executable and the UI archive.
    Fetched { exe: PathBuf, ui: PathBuf },
    /// A build kept in `builds/`, by its directory name.
    Kept(String),
}

/// A finished swap, as [`restore`] needs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Swapped {
    pub exe: PathBuf,
    pub root: PathBuf,
    pub ui: Vec<String>,
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

/// The UI the running build serves, read from `root`.
fn served(root: &Path, ui: &[String]) -> io::Result<Vec<Entry>> {
    let dirs: Vec<(String, PathBuf)> = ui.iter().map(|n| (n.clone(), root.join(n))).collect();
    read(&dirs)
}

/// The `build.json` of what is at `exe` and `ui` under `root`.
fn describe(
    exe: &Path,
    root: &Path,
    ui: &[String],
    info: &BuildInfo,
    installed_ms: u64,
    retired_ms: u64,
) -> io::Result<BuildJson> {
    let entries = served(root, ui)?;
    let mut files = vec![FileSum {
        path: exe_name(exe)?,
        sha256: hex(&file_sha256(exe)?),
    }];
    files.extend(entries.iter().map(|e| FileSum {
        path: format!("{UI}/{}/{}", e.dir, e.file),
        sha256: hex(&sha256(&e.bytes)),
    }));
    Ok(BuildJson {
        info: info.clone(),
        files,
        ui: ui.to_vec(),
        ui_archive: hex(&sha256(&pack(entries))),
        installed_ms,
        retired_ms,
    })
}

/// Move what is at `exe` and `ui` under `root` into `dir`, with `json`.
fn keep_into(
    dir: &Path,
    exe: &Path,
    root: &Path,
    ui: &[String],
    json: &BuildJson,
) -> io::Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir.join(UI))?;
    fs::rename(exe, dir.join(exe_name(exe)?))?;
    for n in ui {
        let to = dir.join(UI).join(n);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(root.join(n), to)?;
    }
    write_json(dir, json)
}

/// Move the build kept in `dir` back to `exe` and `ui` under `root`, and
/// remove the directory.
fn take_out(dir: &Path, exe: &Path, root: &Path, ui: &[String]) -> io::Result<()> {
    fs::rename(dir.join(exe_name(exe)?), exe)?;
    for n in ui {
        let at = root.join(n);
        if let Some(parent) = at.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(dir.join(UI).join(n), at)?;
    }
    fs::remove_dir_all(dir)
}

/// The served directories of an archive, written under `root`; refused when
/// the archive lacks one.
fn unpack_served(archive: &[u8], root: &Path, ui: &[String]) -> io::Result<()> {
    let all = entries(archive)?;
    for n in ui {
        if !all.iter().any(|e| &e.dir == n) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("the build's UI has no {n}"),
            ));
        }
    }
    for e in all.into_iter().filter(|e| ui.contains(&e.dir)) {
        let to = root.join(&e.dir).join(&e.file);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(to, e.bytes)?;
    }
    Ok(())
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
    let json = describe(
        running.exe,
        running.root,
        running.ui,
        running.info,
        running.installed_ms,
        now_ms,
    )?;
    // Checked before anything moves: a source that cannot be placed leaves
    // the running build where it is.
    let from_kept = match source {
        Source::Fetched { exe, ui } => {
            let archive = fs::read(ui)?;
            let names: Vec<String> = entries(&archive)?.into_iter().map(|e| e.dir).collect();
            if let Some(missing) = running.ui.iter().find(|n| !names.contains(n)) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("the build's UI has no {missing}"),
                ));
            }
            if !exe.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("no artifact at {}", exe.display()),
                ));
            }
            keep_into(&kept, running.exe, running.root, running.ui, &json)?;
            place_exe(exe, running.exe)?;
            unpack_served(&archive, running.root, running.ui)?;
            None
        }
        Source::Kept(n) => {
            let dir = builds.join(n);
            let theirs = read_json(&dir)?;
            if let Some(missing) = running.ui.iter().find(|u| !theirs.ui.contains(u)) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("the kept build {n} has no {missing}"),
                ));
            }
            keep_into(&kept, running.exe, running.root, running.ui, &json)?;
            take_out(&dir, running.exe, running.root, running.ui)?;
            Some((dir, theirs))
        }
    };
    Ok(Swapped {
        exe: running.exe.to_path_buf(),
        root: running.root.to_path_buf(),
        ui: running.ui.to_vec(),
        kept,
        from_kept,
    })
}

/// Undo `s`: the build swapped in leaves — back into `builds/` when it came
/// from there — and the build swapped out returns to its paths.
pub fn restore(s: &Swapped) -> io::Result<()> {
    match &s.from_kept {
        Some((dir, json)) => keep_into(dir, &s.exe, &s.root, &s.ui, json)?,
        None => {
            if s.exe.exists() {
                fs::remove_file(&s.exe)?;
            }
            for n in &s.ui {
                let at = s.root.join(n);
                if at.exists() {
                    fs::remove_dir_all(at)?;
                }
            }
        }
    }
    take_out(&s.kept, &s.exe, &s.root, &s.ui)
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    use crate::upgrade::kept::list;
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

    /// A machine: `bin/zend` and the UI `zend/web` under `root`.
    fn machine(exe_text: &str, page: &str) -> (TempDir, PathBuf) {
        let t = TempDir::new().unwrap();
        fs::create_dir_all(t.path().join("bin")).unwrap();
        fs::create_dir_all(t.path().join("zend/web")).unwrap();
        let exe = t.path().join("bin/zend");
        fs::write(&exe, exe_text).unwrap();
        fs::write(t.path().join("zend/web/index.html"), page).unwrap();
        (t, exe)
    }

    fn ui_archive(at: &Path, page: &str) -> PathBuf {
        let archive = pack(vec![
            Entry {
                dir: "zend/web".into(),
                file: "index.html".into(),
                bytes: page.as_bytes().to_vec(),
            },
            Entry {
                dir: "web/content/npcd".into(),
                file: "index.html".into(),
                bytes: b"npcd".to_vec(),
            },
        ]);
        fs::write(at, archive).unwrap();
        at.to_path_buf()
    }

    fn read_text(p: &Path) -> String {
        fs::read_to_string(p).unwrap()
    }

    /// A fetched build takes the executable's and the UI's paths; the old
    /// build is kept with its UI and `build.json`; a UI directory this member
    /// does not serve is not written.
    #[test]
    fn a_fetched_build_takes_the_paths_and_the_old_one_is_kept() {
        let (t, exe) = machine("old exe", "old page");
        let ui = vec!["zend/web".to_string()];
        let new_exe = t.path().join("new-zend");
        fs::write(&new_exe, "new exe").unwrap();
        let archive = ui_archive(&t.path().join("ui.archive"), "new page");
        let running = Running {
            exe: &exe,
            root: t.path(),
            ui: &ui,
            info: &info(10),
            installed_ms: 5,
        };
        let s = swap(
            &running,
            &Source::Fetched {
                exe: new_exe,
                ui: archive,
            },
            100,
        )
        .unwrap();
        assert_eq!(read_text(&exe), "new exe");
        assert_eq!(read_text(&t.path().join("zend/web/index.html")), "new page");
        assert!(!t.path().join("web/content/npcd").exists());
        let kept = t.path().join("bin/builds/10-h10");
        assert_eq!(s.kept, kept);
        assert_eq!(read_text(&kept.join("zend")), "old exe");
        assert_eq!(read_text(&kept.join("ui/zend/web/index.html")), "old page");
        let json = read_json(&kept).unwrap();
        assert_eq!(
            (json.info, json.installed_ms, json.retired_ms),
            (info(10), 5, 100)
        );
        assert_eq!(json.files[0].path, "zend");
        assert_eq!(json.files[0].sha256, hex(&sha256(b"old exe")));
        assert_eq!(json.files[1].path, "ui/zend/web/index.html");
        // The kept UI packs to the archive it is announced under.
        let repacked = pack(read(&[("zend/web".into(), kept.join("ui/zend/web"))]).unwrap());
        assert_eq!(json.ui_archive, hex(&sha256(&repacked)));
        assert_eq!(list(&t.path().join("bin/builds")).len(), 1);
    }

    /// A restore puts the old build back where it was and leaves no kept
    /// directory.
    #[test]
    fn a_restore_puts_the_old_build_back() {
        let (t, exe) = machine("old exe", "old page");
        let ui = vec!["zend/web".to_string()];
        let new_exe = t.path().join("new-zend");
        fs::write(&new_exe, "new exe").unwrap();
        let archive = ui_archive(&t.path().join("ui.archive"), "new page");
        let running = Running {
            exe: &exe,
            root: t.path(),
            ui: &ui,
            info: &info(10),
            installed_ms: 5,
        };
        let s = swap(
            &running,
            &Source::Fetched {
                exe: new_exe,
                ui: archive,
            },
            100,
        )
        .unwrap();
        restore(&s).unwrap();
        assert_eq!(read_text(&exe), "old exe");
        assert_eq!(read_text(&t.path().join("zend/web/index.html")), "old page");
        assert!(list(&t.path().join("bin/builds")).is_empty());
    }

    /// A rollback swaps from a kept build: it takes the paths and leaves
    /// `builds/`, and the running build goes there in turn; restored, both
    /// are back where they were.
    #[test]
    fn a_rollback_swaps_with_a_kept_build_and_restores() {
        let (t, exe) = machine("build 12", "page 12");
        let ui = vec!["zend/web".to_string()];
        // Build 10 kept, as a swap would have kept it.
        let old_exe = t.path().join("old");
        fs::write(&old_exe, "build 10").unwrap();
        fs::create_dir_all(t.path().join("old-ui/zend/web")).unwrap();
        fs::write(t.path().join("old-ui/zend/web/index.html"), "page 10").unwrap();
        let kept10 = t.path().join("bin/builds/10-h10");
        fs::create_dir_all(kept10.join("ui/zend")).unwrap();
        fs::rename(&old_exe, kept10.join("zend")).unwrap();
        fs::rename(t.path().join("old-ui/zend/web"), kept10.join("ui/zend/web")).unwrap();
        let json10 = BuildJson {
            info: info(10),
            files: vec![],
            ui: ui.clone(),
            ui_archive: String::new(),
            installed_ms: 1,
            retired_ms: 2,
        };
        write_json(&kept10, &json10).unwrap();

        let running = Running {
            exe: &exe,
            root: t.path(),
            ui: &ui,
            info: &info(12),
            installed_ms: 3,
        };
        let s = swap(&running, &Source::Kept("10-h10".into()), 50).unwrap();
        assert_eq!(read_text(&exe), "build 10");
        assert_eq!(read_text(&t.path().join("zend/web/index.html")), "page 10");
        let names: Vec<String> = list(&t.path().join("bin/builds"))
            .into_iter()
            .map(|k| k.name)
            .collect();
        assert_eq!(names, ["12-h12"]);

        restore(&s).unwrap();
        assert_eq!(read_text(&exe), "build 12");
        assert_eq!(read_text(&t.path().join("zend/web/index.html")), "page 12");
        let back = list(&t.path().join("bin/builds"));
        assert_eq!(back.len(), 1);
        assert_eq!((back[0].name.as_str(), &back[0].json), ("10-h10", &json10));
        assert_eq!(read_text(&kept10.join("zend")), "build 10");
    }

    /// A source that cannot be placed leaves everything where it was.
    #[test]
    fn a_source_missing_the_served_ui_moves_nothing() {
        let (t, exe) = machine("old exe", "old page");
        let ui = vec!["zend/web".to_string(), "web/content/common".to_string()];
        fs::create_dir_all(t.path().join("web/content/common")).unwrap();
        let new_exe = t.path().join("new-zend");
        fs::write(&new_exe, "new exe").unwrap();
        let archive = ui_archive(&t.path().join("ui.archive"), "new page");
        let running = Running {
            exe: &exe,
            root: t.path(),
            ui: &ui,
            info: &info(10),
            installed_ms: 5,
        };
        let err = swap(
            &running,
            &Source::Fetched {
                exe: new_exe,
                ui: archive,
            },
            100,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "the build's UI has no web/content/common");
        assert_eq!(read_text(&exe), "old exe");
        assert!(!t.path().join("bin/builds").exists());
    }
}
