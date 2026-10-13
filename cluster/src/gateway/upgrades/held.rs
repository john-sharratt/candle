//! The builds a gateway knows are held (§3.9), from the `BUILD_OFFER`s and
//! `BUILD_HAVE`s it relays: every member announces the build it runs, each one it
//! keeps and each it verified, and a candidate offers its build. From this the
//! gateway knows whether a member below the floor can be admitted to upgrade —
//! some member holds the floor's build — and the status page lists every build
//! held, with its holders.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::frame::{Artifact, BuildArtifacts, BuildId};
use crate::upgrade::sum::hex;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Held {
    artifacts: BTreeMap<[u8; 32], Artifact>,
    /// Who holds each artifact.
    holders: BTreeMap<[u8; 32], BTreeSet<u64>>,
    /// The candidate that offered it, if one did.
    offered_by: Option<u64>,
}

/// One build held, for the status page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HeldBuild {
    pub count: u32,
    pub hash: String,
    pub offered_by: Option<String>,
    pub artifacts: Vec<HeldArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HeldArtifact {
    pub kind: String,
    pub platform: String,
    pub size: u64,
    pub sha256: String,
    pub holders: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct HeldBuilds {
    builds: BTreeMap<(u32, String), Held>,
}

impl HeldBuilds {
    /// `node` holds `b`'s artifacts; `offered` when it is the candidate's
    /// offer.
    pub fn announce(&mut self, node: u64, b: &BuildArtifacts, offered: bool) {
        let held = self
            .builds
            .entry((b.build.count, b.build.hash.clone()))
            .or_default();
        for a in &b.artifacts {
            held.artifacts.insert(a.sha256, a.clone());
            held.holders.entry(a.sha256).or_default().insert(node);
        }
        if offered {
            held.offered_by = Some(node);
        }
    }

    /// `node` is gone from this gateway: it holds nothing here.
    pub fn forget(&mut self, node: u64) {
        for held in self.builds.values_mut() {
            for holders in held.holders.values_mut() {
                holders.remove(&node);
            }
            held.holders.retain(|_, h| !h.is_empty());
            if held.offered_by == Some(node) {
                held.offered_by = None;
            }
        }
        self.builds.retain(|_, h| !h.holders.is_empty());
    }

    /// A rejected build's artifacts are dropped everywhere.
    pub fn drop_build(&mut self, build: &BuildId) {
        self.builds.remove(&(build.count, build.hash.clone()));
    }

    /// Whether some member holds any artifact of `build`.
    pub fn holds(&self, build: &BuildId) -> bool {
        self.builds
            .get(&(build.count, build.hash.clone()))
            .is_some_and(|h| !h.holders.is_empty())
    }

    /// `build`'s artifacts, as an offer to a member that is to fetch it.
    pub fn offer(&self, build: &BuildId) -> Option<BuildArtifacts> {
        let held = self.builds.get(&(build.count, build.hash.clone()))?;
        Some(BuildArtifacts {
            build: build.clone(),
            artifacts: held.artifacts.values().cloned().collect(),
        })
    }

    /// The members holding the artifact `sha256` of `build`.
    pub fn holders(&self, build: &BuildId, sha256: &[u8; 32]) -> Vec<u64> {
        self.builds
            .get(&(build.count, build.hash.clone()))
            .and_then(|h| h.holders.get(sha256))
            .map(|h| h.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The members holding the artifact `sha256`, whatever its build: where a
    /// `BUILD_FETCH` naming no member is sent.
    pub fn holders_any(&self, sha256: &[u8; 32]) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .builds
            .values()
            .filter_map(|h| h.holders.get(sha256))
            .flat_map(|h| h.iter().copied())
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether `build` includes a binary of `kind`.
    pub fn has_kind(&self, build: &BuildId, kind: &str) -> bool {
        self.builds
            .get(&(build.count, build.hash.clone()))
            .is_some_and(|h| h.artifacts.values().any(|a| a.kind == kind))
    }

    /// Every build held, newest first.
    pub fn view(&self) -> Vec<HeldBuild> {
        self.builds
            .iter()
            .rev()
            .map(|((count, hash), h)| HeldBuild {
                count: *count,
                hash: hash.clone(),
                offered_by: h.offered_by.map(|n| format!("{n:016x}")),
                artifacts: h
                    .artifacts
                    .values()
                    .map(|a| HeldArtifact {
                        kind: a.kind.clone(),
                        platform: a.platform.clone(),
                        size: a.size,
                        sha256: hex(&a.sha256),
                        holders: h
                            .holders
                            .get(&a.sha256)
                            .map(|s| s.iter().map(|n| format!("{n:016x}")).collect())
                            .unwrap_or_default(),
                    })
                    .collect(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(count: u32) -> BuildId {
        BuildId {
            count,
            hash: format!("h{count}"),
            local: String::new(),
        }
    }

    fn artifact(kind: &str, sha: u8) -> Artifact {
        Artifact {
            kind: kind.into(),
            platform: "linux-x86_64".into(),
            size: 9,
            sha256: [sha; 32],
        }
    }

    #[test]
    fn holders_are_tracked_per_artifact_and_forgotten_when_they_leave() {
        let mut h = HeldBuilds::default();
        let b = BuildArtifacts {
            build: build(12),
            artifacts: vec![artifact("zend", 1), artifact("motherd", 2)],
        };
        h.announce(7, &b, true);
        h.announce(
            9,
            &BuildArtifacts {
                build: build(12),
                artifacts: vec![artifact("zend", 1)],
            },
            false,
        );
        assert!(h.holds(&build(12)));
        assert!(!h.holds(&build(11)));
        assert_eq!(h.holders(&build(12), &[1; 32]), [7, 9]);
        assert_eq!(h.holders_any(&[1; 32]), [7, 9]);
        assert_eq!(h.holders_any(&[2; 32]), [7]);
        assert!(h.has_kind(&build(12), "zend") && !h.has_kind(&build(12), "npcd"));
        assert_eq!(h.offer(&build(12)).unwrap().artifacts.len(), 2);
        let v = h.view();
        assert_eq!(v[0].offered_by.as_deref(), Some("0000000000000007"));
        h.forget(7);
        assert_eq!(h.holders(&build(12), &[1; 32]), [9]);
        assert_eq!(h.view()[0].offered_by, None);
        h.forget(9);
        assert!(!h.holds(&build(12)));
        h.announce(7, &b, false);
        h.drop_build(&build(12));
        assert!(h.view().is_empty());
    }
}
