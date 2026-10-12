//! The builds a member keeps for rollback (§3.9): `<exe dir>/builds/`, one
//! directory per build, named `<count>-<hash>`, holding the executable, the UI
//! it served under `ui/<dir>`, and `build.json` — its `--build-info` answer,
//! each file's SHA-256, the SHA-256 of its UI as an archive, and when it was
//! installed and retired.
//!
//! The running build is never in `builds/`, so it is never pruned. Pruning
//! keeps the newest *N* by retirement and never deletes a build a rollout in
//! progress targets.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::frame::BuildId;
use crate::version::info::BuildInfo;

/// The kept builds' directory, beside the executable.
pub const BUILDS: &str = "builds";
pub const BUILD_JSON: &str = "build.json";
/// A kept build's UI, under its directory.
pub const UI: &str = "ui";

/// The default of `--keep-builds`.
pub const KEEP_DEFAULT: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSum {
    /// Relative to the kept build's directory, `/` between parts.
    pub path: String,
    pub sha256: String,
}

/// `build.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildJson {
    pub info: BuildInfo,
    /// The executable, then every UI file, in path order.
    pub files: Vec<FileSum>,
    /// The UI directories it served, as its tenants name them.
    pub ui: Vec<String>,
    /// The SHA-256 of its UI packed as an archive: what it is offered under.
    pub ui_archive: String,
    pub installed_ms: u64,
    pub retired_ms: u64,
}

/// One kept build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kept {
    /// `<count>-<hash>`.
    pub name: String,
    pub dir: PathBuf,
    pub json: BuildJson,
    /// Its bytes on disk.
    pub bytes: u64,
}

/// A build's directory name.
pub fn name(build: &BuildId) -> String {
    format!("{}-{}", build.count, build.hash)
}

/// The kept builds' directory for the executable at `exe`.
pub fn builds_dir(exe: &Path) -> PathBuf {
    exe.parent()
        .map(|p| p.join(BUILDS))
        .unwrap_or_else(|| PathBuf::from(BUILDS))
}

fn dir_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

pub fn read_json(dir: &Path) -> io::Result<BuildJson> {
    let text = fs::read_to_string(dir.join(BUILD_JSON))?;
    serde_json::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write_json(dir: &Path, json: &BuildJson) -> io::Result<()> {
    let text = serde_json::to_string_pretty(json).expect("build.json always serializes");
    let tmp = dir.join(format!("{BUILD_JSON}.tmp"));
    fs::write(&tmp, text)?;
    fs::rename(tmp, dir.join(BUILD_JSON))
}

/// Every build kept in `builds`, the most recently retired first. A directory
/// without a readable `build.json` is no kept build and is left alone.
pub fn list(builds: &Path) -> Vec<Kept> {
    let Ok(entries) = fs::read_dir(builds) else {
        return Vec::new();
    };
    let mut kept: Vec<Kept> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter_map(|e| {
            let dir = e.path();
            let json = read_json(&dir).ok()?;
            Some(Kept {
                name: e.file_name().to_string_lossy().into_owned(),
                bytes: dir_bytes(&dir),
                dir,
                json,
            })
        })
        .collect();
    kept.sort_by(|a, b| (b.json.retired_ms, &b.name).cmp(&(a.json.retired_ms, &a.name)));
    kept
}

/// The kept build of `build`, if `builds` holds it.
pub fn find(builds: &Path, build: &BuildId) -> Option<Kept> {
    list(builds).into_iter().find(|k| k.name == name(build))
}

/// Delete the kept builds beyond the newest `keep`, never one in `protect`.
/// The names deleted.
pub fn prune(builds: &Path, keep: usize, protect: &[BuildId]) -> io::Result<Vec<String>> {
    let protected: Vec<String> = protect.iter().map(name).collect();
    let mut gone = Vec::new();
    for k in list(builds).into_iter().skip(keep) {
        if protected.contains(&k.name) {
            continue;
        }
        fs::remove_dir_all(&k.dir)?;
        gone.push(k.name);
    }
    Ok(gone)
}

/// Delete the kept builds named, never one in `protect`; refused, deleting
/// nothing, when a name is not kept or is protected.
pub fn remove(builds: &Path, names: &[String], protect: &[BuildId]) -> Result<Vec<String>, String> {
    let kept = list(builds);
    let protected: Vec<String> = protect.iter().map(name).collect();
    for n in names {
        if !kept.iter().any(|k| &k.name == n) {
            return Err(format!("no kept build {n}"));
        }
        if protected.contains(n) {
            return Err(format!("{n} is the target of a rollout in progress"));
        }
    }
    for k in kept.iter().filter(|k| names.contains(&k.name)) {
        fs::remove_dir_all(&k.dir).map_err(|e| format!("{}: {e}", k.name))?;
    }
    Ok(names.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    use crate::version::info::{Opens, ZEND};

    fn json(count: u32, retired_ms: u64) -> BuildJson {
        BuildJson {
            info: BuildInfo {
                kind: ZEND.into(),
                count,
                hash: format!("h{count}"),
                local: String::new(),
                platform: "windows-x86_64".into(),
                opens: Opens::default(),
            },
            files: vec![FileSum {
                path: "zend.exe".into(),
                sha256: "00".repeat(32),
            }],
            ui: vec!["zend/web".into()],
            ui_archive: "11".repeat(32),
            installed_ms: retired_ms - 10,
            retired_ms,
        }
    }

    fn keep(builds: &Path, count: u32, retired_ms: u64) {
        let dir = builds.join(format!("{count}-h{count}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("zend.exe"), b"exe").unwrap();
        write_json(&dir, &json(count, retired_ms)).unwrap();
    }

    fn b(count: u32) -> BuildId {
        BuildId {
            count,
            hash: format!("h{count}"),
            local: String::new(),
        }
    }

    /// Listed newest-retired first, with their bytes; a directory with no
    /// `build.json` is not a kept build.
    #[test]
    fn kept_builds_list_newest_first() {
        let t = TempDir::new().unwrap();
        keep(t.path(), 10, 100);
        keep(t.path(), 12, 300);
        keep(t.path(), 11, 200);
        fs::create_dir_all(t.path().join("stray")).unwrap();
        let names: Vec<String> = list(t.path()).into_iter().map(|k| k.name).collect();
        assert_eq!(names, ["12-h12", "11-h11", "10-h10"]);
        let top = &list(t.path())[0];
        assert_eq!(top.json, json(12, 300));
        assert!(top.bytes > 3);
        assert_eq!(find(t.path(), &b(11)).unwrap().name, "11-h11");
        assert!(find(t.path(), &b(9)).is_none());
    }

    /// Pruning keeps the newest N and never deletes a protected build.
    #[test]
    fn pruning_keeps_the_newest_and_the_protected() {
        let t = TempDir::new().unwrap();
        for (c, at) in [(10, 100), (11, 200), (12, 300), (13, 400)] {
            keep(t.path(), c, at);
        }
        assert_eq!(prune(t.path(), 2, &[b(10)]).unwrap(), ["11-h11"]);
        let left: Vec<String> = list(t.path()).into_iter().map(|k| k.name).collect();
        assert_eq!(left, ["13-h13", "12-h12", "10-h10"]);
        assert!(prune(t.path(), 5, &[]).unwrap().is_empty());
    }

    /// Removing by name refuses, deleting nothing, a name not kept or one a
    /// rollout targets.
    #[test]
    fn removing_named_builds_refuses_what_it_must_not_touch() {
        let t = TempDir::new().unwrap();
        keep(t.path(), 10, 100);
        keep(t.path(), 11, 200);
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            remove(t.path(), &names(&["10-h10", "9-h9"]), &[]).unwrap_err(),
            "no kept build 9-h9"
        );
        assert_eq!(
            remove(t.path(), &names(&["11-h11"]), &[b(11)]).unwrap_err(),
            "11-h11 is the target of a rollout in progress"
        );
        assert_eq!(list(t.path()).len(), 2);
        assert_eq!(
            remove(t.path(), &names(&["10-h10"]), &[]).unwrap(),
            ["10-h10"]
        );
        assert_eq!(list(t.path()).len(), 1);
    }

    #[test]
    fn the_builds_directory_is_beside_the_executable() {
        assert_eq!(
            builds_dir(Path::new("/opt/tokera/zend")),
            PathBuf::from("/opt/tokera/builds")
        );
        assert_eq!(name(&b(4815)), "4815-h4815");
    }
}
