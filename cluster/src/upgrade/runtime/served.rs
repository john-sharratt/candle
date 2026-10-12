//! The artifacts a member can serve (§3.9): its running build's executable and
//! UI, each build it keeps, each it verified, and — as a candidate — every
//! sibling binary of its build. Each is announced with `BUILD_HAVE` (or
//! `BUILD_OFFER`), and a `BUILD_FETCH` for its SHA-256 is answered from the
//! file it names.
//!
//! A UI is served as an archive: the running build's and each kept build's
//! are packed into `<data>/upgrade/` when the member starts, so every one is a
//! file of known size before it is announced.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::setup::Setup;
use crate::frame::{Artifact, BuildArtifacts, BuildId};
use crate::upgrade::artifacts::{artifact_of, Offered, ANY_PLATFORM, UI_KIND};
use crate::upgrade::kept::{builds_dir, list, UI};
use crate::upgrade::parts::Parts;
use crate::upgrade::sum::{hex, sha256, unhex};
use crate::upgrade::ui_archive::{pack, read};
use crate::version::info::BuildInfo;

/// Where a served artifact comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeldAs {
    Running,
    Kept,
    Verified,
    Offered,
}

impl HeldAs {
    pub fn name(self) -> &'static str {
        match self {
            HeldAs::Running => "running",
            HeldAs::Kept => "kept",
            HeldAs::Verified => "verified",
            HeldAs::Offered => "offered",
        }
    }
}

/// One artifact this member serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub build: BuildId,
    pub artifact: Artifact,
    pub from: HeldAs,
    pub path: PathBuf,
    /// A binary's `--build-info` answer, when known.
    pub info: Option<BuildInfo>,
}

/// An offer as `<data>/upgrade/<hash>/offer.json` keeps it, so the verified
/// downloads it names are known again after a restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfferFile {
    pub build: BuildId,
    pub artifacts: Vec<ArtifactFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFile {
    pub kind: String,
    pub platform: String,
    pub size: u64,
    pub sha256: String,
}

pub const OFFER: &str = "offer.json";

impl OfferFile {
    pub fn of(b: &BuildArtifacts) -> OfferFile {
        OfferFile {
            build: b.build.clone(),
            artifacts: b
                .artifacts
                .iter()
                .map(|a| ArtifactFile {
                    kind: a.kind.clone(),
                    platform: a.platform.clone(),
                    size: a.size,
                    sha256: hex(&a.sha256),
                })
                .collect(),
        }
    }

    pub fn artifacts(&self) -> Vec<Artifact> {
        self.artifacts
            .iter()
            .filter_map(|a| {
                Some(Artifact {
                    kind: a.kind.clone(),
                    platform: a.platform.clone(),
                    size: a.size,
                    sha256: unhex(&a.sha256)?,
                })
            })
            .collect()
    }

    /// Keep `b` beside its downloads.
    pub fn write(parts: &Parts, b: &BuildArtifacts) -> io::Result<()> {
        let dir = parts.root().join(&b.build.hash);
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join(OFFER),
            serde_json::to_vec(&OfferFile::of(b)).expect("an offer serializes"),
        )
    }
}

/// Pack the UI directories `ui` under `root` into `to`, as an artifact.
fn ui_item(
    build: &BuildId,
    from: HeldAs,
    root: &Path,
    ui: &[String],
    to: &Path,
) -> io::Result<Item> {
    let dirs: Vec<(String, PathBuf)> = ui.iter().map(|n| (n.clone(), root.join(n))).collect();
    let archive = pack(read(&dirs)?);
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(to, &archive)?;
    Ok(Item {
        build: build.clone(),
        artifact: Artifact {
            kind: UI_KIND.into(),
            platform: ANY_PLATFORM.into(),
            size: archive.len() as u64,
            sha256: sha256(&archive),
        },
        from,
        path: to.to_path_buf(),
        info: None,
    })
}

/// The running build's executable and UI.
pub fn running(setup: &Setup) -> io::Result<Vec<Item>> {
    let build = setup.info.build();
    let mut items = vec![Item {
        build: build.clone(),
        artifact: artifact_of(&setup.exe, &setup.info.kind, &setup.info.platform)?,
        from: HeldAs::Running,
        path: setup.exe.clone(),
        info: Some(setup.info.clone()),
    }];
    if !setup.ui.is_empty() {
        items.push(ui_item(
            &build,
            HeldAs::Running,
            &setup.root,
            &setup.ui,
            &setup.upgrade_dir.join("running.ui"),
        )?);
    }
    Ok(items)
}

/// Every kept build's executable and UI.
pub fn kept(setup: &Setup) -> Vec<Item> {
    let mut items = Vec::new();
    for k in list(&builds_dir(&setup.exe)) {
        let build = k.json.info.build();
        let Some(exe) = k.json.files.first() else {
            continue;
        };
        let path = k.dir.join(&exe.path);
        if let Ok(artifact) = artifact_of(&path, &k.json.info.kind, &k.json.info.platform) {
            items.push(Item {
                build: build.clone(),
                artifact,
                from: HeldAs::Kept,
                path,
                info: Some(k.json.info.clone()),
            });
        }
        if !k.json.ui.is_empty() {
            let packed = setup
                .upgrade_dir
                .join("kept")
                .join(format!("{}.ui", k.name));
            match ui_item(&build, HeldAs::Kept, &k.dir.join(UI), &k.json.ui, &packed) {
                Ok(item) => items.push(item),
                Err(e) => tracing::warn!(build = %k.name, "a kept build's UI is unreadable: {e}"),
            }
        }
    }
    items
}

/// Every verified download under `parts`.
pub fn verified(parts: &Parts) -> Vec<Item> {
    let Ok(dirs) = fs::read_dir(parts.root()) else {
        return Vec::new();
    };
    let mut items = Vec::new();
    for dir in dirs.flatten() {
        let Ok(bytes) = fs::read(dir.path().join(OFFER)) else {
            continue;
        };
        let Ok(offer) = serde_json::from_slice::<OfferFile>(&bytes) else {
            continue;
        };
        for a in offer.artifacts() {
            let path = parts.verified(&offer.build, &a);
            if path.is_file() {
                items.push(Item {
                    build: offer.build.clone(),
                    artifact: a,
                    from: HeldAs::Verified,
                    path,
                    info: None,
                });
            }
        }
    }
    items
}

/// A candidate's offer, as served items.
pub fn offered(build: &BuildId, offered: Vec<Offered>) -> Vec<Item> {
    offered
        .into_iter()
        .map(|o| Item {
            build: build.clone(),
            artifact: o.artifact,
            from: HeldAs::Offered,
            path: o.path,
            info: o.info,
        })
        .collect()
}

/// Every build in `items`, with its artifacts, once each.
pub fn grouped(items: &[Item]) -> Vec<BuildArtifacts> {
    let mut out: Vec<BuildArtifacts> = Vec::new();
    for i in items {
        let at = match out.iter().position(|b| b.build == i.build) {
            Some(at) => at,
            None => {
                out.push(BuildArtifacts {
                    build: i.build.clone(),
                    artifacts: Vec::new(),
                });
                out.len() - 1
            }
        };
        if !out[at]
            .artifacts
            .iter()
            .any(|a| a.sha256 == i.artifact.sha256)
        {
            out[at].artifacts.push(i.artifact.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    use crate::upgrade::kept::{write_json, BuildJson, FileSum};
    use crate::version::info::{Opens, ZEND};

    fn info(count: u32) -> BuildInfo {
        BuildInfo {
            kind: ZEND.into(),
            count,
            hash: format!("h{count}"),
            local: String::new(),
            platform: "linux-x86_64".into(),
            opens: Opens::default(),
        }
    }

    fn setup(t: &TempDir) -> Setup {
        fs::create_dir_all(t.path().join("bin")).unwrap();
        fs::create_dir_all(t.path().join("zend/web")).unwrap();
        fs::write(t.path().join("bin/zend"), b"exe 12").unwrap();
        fs::write(t.path().join("zend/web/i.html"), b"page 12").unwrap();
        Setup {
            info: info(12),
            exe: t.path().join("bin/zend"),
            root: t.path().to_path_buf(),
            ui: vec!["zend/web".into()],
            upgrade_dir: t.path().join("data/upgrade"),
            keep_builds: 5,
            failed_from: None,
        }
    }

    /// The running build serves its executable and its UI as an archive.
    #[test]
    fn the_running_build_is_served() {
        let t = TempDir::new().unwrap();
        let s = setup(&t);
        let items = running(&s).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].artifact.sha256, sha256(b"exe 12"));
        assert_eq!(items[0].from, HeldAs::Running);
        assert_eq!(items[1].artifact.kind, UI_KIND);
        assert_eq!(
            items[1].artifact.sha256,
            sha256(&fs::read(&items[1].path).unwrap())
        );
        let groups = grouped(&items);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].artifacts.len(), 2);
    }

    /// A kept build serves its executable and its UI packed from `ui/`.
    #[test]
    fn a_kept_build_is_served() {
        let t = TempDir::new().unwrap();
        let s = setup(&t);
        let dir = t.path().join("bin/builds/10-h10");
        fs::create_dir_all(dir.join("ui/zend/web")).unwrap();
        fs::write(dir.join("zend"), b"exe 10").unwrap();
        fs::write(dir.join("ui/zend/web/i.html"), b"page 10").unwrap();
        write_json(
            &dir,
            &BuildJson {
                info: info(10),
                files: vec![FileSum {
                    path: "zend".into(),
                    sha256: hex(&sha256(b"exe 10")),
                }],
                ui: vec!["zend/web".into()],
                ui_archive: String::new(),
                installed_ms: 1,
                retired_ms: 2,
            },
        )
        .unwrap();
        let items = kept(&s);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].build, info(10).build());
        assert_eq!(items[0].artifact.sha256, sha256(b"exe 10"));
        assert_eq!(items[1].from, HeldAs::Kept);
    }

    /// A verified download is known again after a restart, by its offer.
    #[test]
    fn a_verified_download_is_found_by_its_offer() {
        let t = TempDir::new().unwrap();
        let parts = Parts::new(t.path());
        let a = Artifact {
            kind: ZEND.into(),
            platform: "linux-x86_64".into(),
            size: 2,
            sha256: sha256(b"ok"),
        };
        let b = BuildArtifacts {
            build: info(13).build(),
            artifacts: vec![a.clone()],
        };
        OfferFile::write(&parts, &b).unwrap();
        assert!(verified(&parts).is_empty(), "nothing verified yet");
        fs::write(parts.verified(&b.build, &a), b"ok").unwrap();
        let items = verified(&parts);
        assert_eq!(items.len(), 1);
        assert_eq!((items[0].from, &items[0].artifact), (HeldAs::Verified, &a));
    }
}
