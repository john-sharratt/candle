//! Fetching a build, and serving one (§3.9).
//!
//! **Fetching.** A member fetches the one artifact of the build it would
//! install — or of the newest build offered above its own — that it needs: its
//! own binary, by kind and platform, which carries its UI compiled in. One
//! chunk at a time: `BUILD_FETCH` asks for the next up to 1 MiB at the part
//! file's length, and the gateway picks a holder for it (`from_node` 0), so
//! later fetches spread across every holder. A chunk that does not come within
//! [`CHUNK_WAIT`] is asked for again — a resume from the part file, from
//! whichever holder the gateway picks now. A finished artifact verifies or
//! starts over.
//!
//! **Stranded.** A build with no artifact for this member's kind and platform
//! cannot be installed here: the member says so.
//!
//! **Serving.** A `BUILD_FETCH` for an artifact this member serves is answered
//! with a `BUILD_CHUNK` on the catch-up lane, to the requester alone.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::decide::prefetch;
use super::served::{HeldAs, Item, OfferFile};
use super::Upgrader;
use crate::frame::{
    Artifact, BuildArtifacts, BuildChunk, BuildFetch, BuildId, Frame, MAX_BUILD_CHUNK,
};
use crate::upgrade::decisions::build_name;
use crate::upgrade::parts::{Finished, Standing, Wrote};

/// How long a chunk may take before it is asked for again.
pub const CHUNK_WAIT: Duration = Duration::from_secs(30);

/// An artifact being fetched.
#[derive(Debug, Clone)]
pub struct Fetching {
    pub build: BuildId,
    pub artifact: Artifact,
    pub offset: u64,
    pub asked: Instant,
    pub resumes: u64,
    /// The node the chunks are addressed to: this member's, in the app the
    /// fetch went out on.
    pub reply_to: u64,
    pub app: String,
}

/// The artifact of `offer` a member of `kind` on `platform` needs, or why it
/// cannot install the build.
pub fn needed(offer: &BuildArtifacts, kind: &str, platform: &str) -> Result<Artifact, String> {
    offer
        .artifacts
        .iter()
        .find(|a| a.kind == kind && a.platform == platform)
        .cloned()
        .ok_or_else(|| {
            format!(
                "build {} has no {kind} for {platform}",
                build_name(&offer.build)
            )
        })
}

/// Up to [`MAX_BUILD_CHUNK`] bytes of the file at `path` from `offset`.
fn read_chunk(path: &Path, offset: u64) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::with_capacity(MAX_BUILD_CHUNK);
    f.take(MAX_BUILD_CHUNK as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

impl Upgrader {
    /// Whether `build` is kept here, so it needs no fetch.
    fn keeps(&self, build: &BuildId) -> bool {
        self.state().served.iter().any(|i| {
            i.build == *build && i.from == HeldAs::Kept && i.artifact.kind == self.setup.info.kind
        })
    }

    pub(super) async fn fetch_tick(self: &Arc<Self>) {
        let (want, offer) = {
            let s = self.state();
            let Some(want) = prefetch(&s.facts) else {
                drop(s);
                self.state().fetch = None;
                return;
            };
            let offer = s.known.get(&(want.count, want.hash.clone())).cloned();
            (want, offer)
        };
        if self.keeps(&want) {
            return;
        }
        let Some(offer) = offer else {
            return;
        };
        let artifact = match needed(&offer, &self.setup.info.kind, &self.setup.info.platform) {
            Ok(a) => a,
            Err(reason) => {
                self.state().stranded = Some(reason);
                return;
            }
        };
        self.state().stranded = None;
        let offset = match self.parts.standing(&offer.build, &artifact) {
            Standing::Verified(path) => {
                self.verified(&offer.build, &artifact, &path).await;
                self.state().fetch = None;
                return;
            }
            Standing::Resume(offset) => offset,
        };
        let ask = {
            let mut s = self.state();
            match &mut s.fetch {
                Some(f) if f.artifact == artifact => {
                    if f.asked.elapsed() < CHUNK_WAIT {
                        None
                    } else {
                        f.resumes += 1;
                        f.offset = offset;
                        f.asked = Instant::now();
                        Some(f.clone())
                    }
                }
                _ => {
                    let Some(m) = self
                        .apps()
                        .into_iter()
                        .find(|m| !m.admitted_sockets().is_empty())
                    else {
                        return;
                    };
                    let f = Fetching {
                        build: offer.build.clone(),
                        artifact,
                        offset,
                        asked: Instant::now(),
                        resumes: (offset > 0) as u64,
                        reply_to: m.node(),
                        app: m.app().to_string(),
                    };
                    s.fetch = Some(f.clone());
                    Some(f)
                }
            }
        };
        if let Some(f) = ask {
            let parts = self.parts.clone();
            let o = offer.clone();
            let _ = tokio::task::spawn_blocking(move || OfferFile::write(&parts, &o)).await;
            self.ask(&f);
        }
    }

    fn ask(&self, f: &Fetching) {
        let Some(m) = self.member(&f.app) else {
            return;
        };
        if let Some(c) = m.admitted_sockets().first() {
            m.send_on(
                *c,
                Frame::BuildFetch(BuildFetch {
                    artifact: f.artifact.sha256,
                    offset: f.offset,
                    from_node: 0,
                    reply_to: f.reply_to,
                }),
            );
        }
    }

    /// An artifact verified at `path`: served from here on, and announced.
    async fn verified(&self, build: &BuildId, a: &Artifact, path: &Path) {
        if self
            .state()
            .served
            .iter()
            .any(|i| i.artifact.sha256 == a.sha256 && i.from == HeldAs::Verified)
        {
            return;
        }
        let probe = self.probe.clone();
        let p = path.to_path_buf();
        let info = tokio::task::spawn_blocking(move || probe.build_info(&p))
            .await
            .ok()
            .flatten();
        self.state().served.push(Item {
            build: build.clone(),
            artifact: a.clone(),
            from: HeldAs::Verified,
            path: path.to_path_buf(),
            info,
        });
        self.announce_now(&BuildArtifacts {
            build: build.clone(),
            artifacts: vec![a.clone()],
        });
    }

    /// A chunk of the artifact being fetched.
    pub(super) fn on_chunk(self: &Arc<Self>, chunk: BuildChunk) {
        let f = {
            let s = self.state();
            match &s.fetch {
                Some(f) if f.artifact.sha256 == chunk.artifact && f.offset == chunk.offset => {
                    f.clone()
                }
                _ => return,
            }
        };
        if chunk.bytes.is_empty() {
            // The holder the gateway picked lacks it: the wait asks again.
            return;
        }
        let me = self.clone();
        tokio::spawn(async move {
            let parts = me.parts.clone();
            let (build, artifact) = (f.build.clone(), f.artifact.clone());
            let wrote = tokio::task::spawn_blocking(move || {
                let wrote = parts.write(&build, &artifact, chunk.offset, &chunk.bytes)?;
                let finished = match wrote {
                    Wrote::Appended(len) if len == artifact.size => {
                        Some(parts.finish(&build, &artifact)?)
                    }
                    _ => None,
                };
                Ok::<_, std::io::Error>((wrote, finished))
            })
            .await;
            let next = match wrote {
                Ok(Ok((Wrote::Appended(_), Some(Finished::Verified(path))))) => {
                    me.verified(&f.build, &f.artifact, &path).await;
                    me.state().fetch = None;
                    None
                }
                Ok(Ok((Wrote::Appended(_), Some(Finished::Mismatch)))) => {
                    tracing::warn!(
                        build = %build_name(&f.build),
                        kind = %f.artifact.kind,
                        "a fetched artifact failed its SHA-256: fetching it again"
                    );
                    Some(0)
                }
                Ok(Ok((Wrote::Appended(len), None))) => Some(len),
                Ok(Ok((Wrote::Stale, _))) => None,
                Ok(Err(e)) => {
                    tracing::error!("a fetched chunk was not written: {e}");
                    None
                }
                Err(_) => None,
            };
            if let Some(offset) = next {
                let ask = {
                    let mut s = me.state();
                    match &mut s.fetch {
                        Some(cur) if cur.artifact == f.artifact => {
                            if offset == 0 {
                                cur.resumes += 1;
                            }
                            cur.offset = offset;
                            cur.asked = Instant::now();
                            Some(cur.clone())
                        }
                        _ => None,
                    }
                };
                if let Some(cur) = ask {
                    me.ask(&cur);
                }
            }
        });
    }

    /// Answer a `BUILD_FETCH` for an artifact this member serves.
    pub(super) fn serve_fetch(self: &Arc<Self>, app: &str, c: usize, f: BuildFetch) {
        let path = self
            .state()
            .served
            .iter()
            .find(|i| i.artifact.sha256 == f.artifact)
            .map(|i| i.path.clone());
        let (Some(path), Some(m)) = (path, self.member(app)) else {
            return;
        };
        tokio::spawn(async move {
            let offset = f.offset;
            let bytes = tokio::task::spawn_blocking(move || read_chunk(&path, offset))
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default();
            m.send_catchup_on(
                c,
                Frame::BuildChunk(BuildChunk {
                    artifact: f.artifact,
                    offset: f.offset,
                    reply_to: f.reply_to,
                    bytes,
                }),
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(kind: &str, platform: &str, sha: u8) -> Artifact {
        Artifact {
            kind: kind.into(),
            platform: platform.into(),
            size: 1,
            sha256: [sha; 32],
        }
    }

    fn offer(artifacts: Vec<Artifact>) -> BuildArtifacts {
        BuildArtifacts {
            build: BuildId {
                count: 12,
                hash: "h12".into(),
                local: String::new(),
            },
            artifacts,
        }
    }

    /// A member takes its own kind and platform; with no match it is
    /// stranded, with the reason.
    #[test]
    fn a_member_fetches_only_its_own_kind_and_platform() {
        let o = offer(vec![
            a("zend", "windows-x86_64", 1),
            a("zend", "linux-x86_64", 2),
            a("motherd", "linux-x86_64", 3),
        ]);
        assert_eq!(
            needed(&o, "zend", "linux-x86_64").unwrap(),
            a("zend", "linux-x86_64", 2)
        );
        assert_eq!(
            needed(&o, "tokera-agent", "linux-x86_64").unwrap_err(),
            "build 12-h12 has no tokera-agent for linux-x86_64"
        );
    }

    #[test]
    fn a_chunk_is_at_most_a_mebibyte_from_its_offset() {
        let t = tempfile::TempDir::new().unwrap();
        let p = t.path().join("f");
        let bytes: Vec<u8> = (0..(MAX_BUILD_CHUNK + 10)).map(|i| i as u8).collect();
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(read_chunk(&p, 0).unwrap().len(), MAX_BUILD_CHUNK);
        assert_eq!(
            read_chunk(&p, MAX_BUILD_CHUNK as u64).unwrap(),
            &bytes[MAX_BUILD_CHUNK..]
        );
        assert!(read_chunk(&p, bytes.len() as u64).unwrap().is_empty());
    }
}
