//! What a build offers (§3.9): every binary built beside it that reports the
//! same build — found by running each with `--build-info`. Each binary carries
//! its tenants' UI compiled in, so the binaries are the whole build. Each is an
//! [`Artifact`]: kind, platform, size and SHA-256, and the path it is served
//! from.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::sum::file_sha256;
use crate::frame::{Artifact, BuildId};
use crate::version::info::{BuildInfo, AGENT, BUILD_INFO_FLAG, MOTHERD, NPCD, ZEND};

/// Every binary a build produces, by kind.
pub const KINDS: [&str; 4] = [ZEND, NPCD, MOTHERD, AGENT];

/// Asks a binary for its build.
pub trait Probe {
    fn build_info(&self, exe: &Path) -> Option<BuildInfo>;
}

/// Runs `<exe> --build-info`.
pub struct RunProbe;

impl Probe for RunProbe {
    fn build_info(&self, exe: &Path) -> Option<BuildInfo> {
        let out = Command::new(exe)
            .arg(BUILD_INFO_FLAG)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        BuildInfo::decode(&String::from_utf8_lossy(&out.stdout))
    }
}

/// One artifact, where it is, and — for a binary — its answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Offered {
    pub artifact: Artifact,
    pub path: PathBuf,
    pub info: Option<BuildInfo>,
}

/// A binary of `kind` in `dir`: `zend.exe` on Windows, `zend` elsewhere.
pub fn binary(dir: &Path, kind: &str) -> PathBuf {
    dir.join(format!("{kind}{}", std::env::consts::EXE_SUFFIX))
}

/// The artifact of the file at `path`.
pub fn artifact_of(path: &Path, kind: &str, platform: &str) -> io::Result<Artifact> {
    Ok(Artifact {
        kind: kind.to_string(),
        platform: platform.to_string(),
        size: fs::metadata(path)?.len(),
        sha256: file_sha256(path)?,
    })
}

/// The binaries in `dir` that report `build`.
pub fn siblings(dir: &Path, build: &BuildId, probe: &dyn Probe) -> io::Result<Vec<Offered>> {
    let mut out = Vec::new();
    for kind in KINDS {
        let path = binary(dir, kind);
        if !path.is_file() {
            continue;
        }
        let Some(info) = probe.build_info(&path) else {
            continue;
        };
        if info.kind != kind || info.build() != *build {
            continue;
        }
        out.push(Offered {
            artifact: artifact_of(&path, kind, &info.platform)?,
            path,
            info: Some(info),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use tempfile::TempDir;

    use crate::upgrade::sum::sha256;
    use crate::version::info::Opens;

    /// Answers by file name, as a built directory's binaries would.
    struct Answers(HashMap<String, BuildInfo>);

    impl Probe for Answers {
        fn build_info(&self, exe: &Path) -> Option<BuildInfo> {
            let name = exe.file_stem()?.to_string_lossy().into_owned();
            self.0.get(&name).cloned()
        }
    }

    fn info(kind: &str, count: u32) -> BuildInfo {
        BuildInfo {
            kind: kind.into(),
            count,
            hash: format!("h{count}"),
            local: String::new(),
            platform: "linux-x86_64".into(),
            opens: Opens::default(),
        }
    }

    /// Only the binaries reporting the same build are offered; one from
    /// another build, or that does not answer, is not.
    #[test]
    fn only_siblings_of_the_same_build_are_offered() {
        let t = TempDir::new().unwrap();
        for kind in KINDS {
            fs::write(binary(t.path(), kind), kind).unwrap();
        }
        let answers = Answers(HashMap::from([
            (ZEND.to_string(), info(ZEND, 12)),
            (MOTHERD.to_string(), info(MOTHERD, 12)),
            (NPCD.to_string(), info(NPCD, 11)),
        ]));
        let build = info(ZEND, 12).build();
        let offered = siblings(t.path(), &build, &answers).unwrap();
        let kinds: Vec<&str> = offered.iter().map(|o| o.artifact.kind.as_str()).collect();
        assert_eq!(kinds, [ZEND, MOTHERD]);
        assert_eq!(
            offered[0].artifact,
            Artifact {
                kind: ZEND.into(),
                platform: "linux-x86_64".into(),
                size: 4,
                sha256: sha256(b"zend"),
            }
        );
    }
}
