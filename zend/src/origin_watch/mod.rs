//! Watching each repository's origin, so a branch that moves there is
//! fetched and ingested within seconds (`docs/zend_branch_ingest.md` §4).
//!
//! One task per git repository with an `origin`. Every couple of seconds it
//! asks origin for its branch tips — one small request on git's own protocol
//! ([`probe`]) — and compares them with the tracking refs; only a difference
//! costs a fetch, and a fetch that moved a ref wakes the ingest worker. What
//! zend itself publishes needs no watcher: a push moves the tracking refs as
//! it lands, and the session wakes the ingest worker after the round.

pub mod cadence;
pub mod probe;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use zend_vfs::probe::BranchTip;
use zend_vfs::{FetchSpec, GitError, RemoteName, Repo, Workspace};

use self::cadence::{Cadence, Jitter, FAST, SLOW};
use self::probe::{ProbeFailure, Transport, PROBE_TIMEOUT};
use crate::code_read::is_upload_path;

/// What one check found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Checked {
    /// Origin holds what the tracking refs hold: nothing was fetched.
    Unchanged,
    /// Origin differed and was fetched; this many tracking refs moved.
    Fetched(usize),
}

/// Compare origin's `tips` with the tracking refs and fetch when they differ —
/// blocking; call it off the async runtime.
pub fn check(repo: &Repo, origin: &RemoteName, tips: &[BranchTip]) -> Result<Checked, GitError> {
    let record = repo.record_branches()?;
    let same = record.len() == tips.len()
        && record
            .iter()
            .zip(tips)
            .all(|(r, t)| r.name == t.name && r.tip == t.tip);
    if same {
        return Ok(Checked::Unchanged);
    }
    let moved = repo.fetch(origin, &FetchSpec::AllBranches)?;
    Ok(Checked::Fetched(moved.len()))
}

/// Consecutive HTTPS probes that fail, not refused but unanswered, before a
/// watcher gives HTTPS up for git: an SSH host alias (`git@github-work:…`)
/// names no host HTTPS can reach, and a connection that never comes is never
/// a refusal.
const HTTPS_GIVE_UP: u32 = 3;

/// The running watchers.
pub struct OriginWatch {
    /// Holds `true` once stopping: a value, not a wake-up, so a watcher busy
    /// probing when it is sent still sees it the moment it next waits.
    stop: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
}

impl OriginWatch {
    /// Start a watcher for every git repository of `workspace` — each one
    /// with no `origin` ends at once, having nothing to watch. `on_change`
    /// runs after every fetch that moved a ref.
    pub fn start(
        workspace: &Workspace,
        github_token: Option<&str>,
        on_change: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let (stop, _) = watch::channel(false);
        let client = Client::builder()
            .timeout(PROBE_TIMEOUT)
            .user_agent(concat!("zend/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("the HTTP client builds from constants");
        let mut tasks = Vec::new();
        for spec in workspace.repos() {
            if is_upload_path(&spec.name) {
                continue;
            }
            let watcher = Watcher {
                name: spec.name.clone(),
                dir: spec.dir.clone(),
                token: github_token.map(str::to_string),
                client: client.clone(),
                stop: stop.subscribe(),
                on_change: Arc::clone(&on_change),
            };
            tasks.push(tokio::spawn(watcher.run()));
        }
        Self { stop, tasks }
    }

    /// Stop every watcher and wait for them. A probe in flight is bounded by
    /// [`PROBE_TIMEOUT`], a fetch by the git layer's network deadline.
    pub async fn stop(self) {
        self.stop.send_replace(true);
        for task in self.tasks {
            if task.await.is_err() {
                tracing::warn!(target: "zend::origin_watch", "a watcher panicked");
            }
        }
    }
}

/// One repository's watcher.
struct Watcher {
    name: String,
    dir: PathBuf,
    token: Option<String>,
    client: Client,
    stop: watch::Receiver<bool>,
    on_change: Arc<dyn Fn() + Send + Sync>,
}

impl Watcher {
    async fn run(mut self) {
        let dir = self.dir.clone();
        let opened = tokio::task::spawn_blocking(move || -> Result<_, GitError> {
            let repo = Repo::open(&dir)?;
            let Some(origin) = repo.origin()? else {
                return Ok(None);
            };
            let url = repo
                .remotes()?
                .into_iter()
                .find(|r| r.name == origin)
                .map(|r| r.fetch_url)
                .unwrap_or_default();
            Ok(Some((Arc::new(repo), origin, url)))
        })
        .await;
        let (repo, origin, url) = match opened {
            Ok(Ok(Some(found))) => found,
            Ok(Ok(None)) => return,
            Ok(Err(GitError::NotARepository { .. })) => return,
            Ok(Err(e)) => {
                tracing::warn!(target: "zend::origin_watch", repo = %self.name, "not watched: {e}");
                return;
            }
            Err(e) => {
                tracing::warn!(target: "zend::origin_watch", repo = %self.name, "not watched: {e}");
                return;
            }
        };
        let mut transport = Transport::for_origin(&url, self.token.as_deref());
        if let Transport::Https { url, token } = &transport {
            if let Err(ProbeFailure::Refused(why)) =
                probe::check_https(&self.client, url, token.as_deref()).await
            {
                tracing::info!(
                    target: "zend::origin_watch",
                    repo = %self.name,
                    "origin will not answer over HTTPS ({why}); probing with git instead",
                );
                transport = Transport::Git { local: false };
            }
        }
        tracing::info!(target: "zend::origin_watch", repo = %self.name, ?transport, "watching origin");
        let mut cadence = Cadence::new(interval(&transport));
        let mut jitter = Jitter::new(&self.name);
        loop {
            let wait = cadence.next_delay(jitter.draw());
            if !*self.stop.borrow() {
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {}
                    // The sender gone is the watch gone: stop too.
                    changed = self.stop.changed() => if changed.is_err() { return; },
                }
            }
            if *self.stop.borrow() {
                return;
            }
            let tips = match &transport {
                Transport::Https { url, token } => {
                    probe::https_tips(&self.client, url, token.as_deref(), repo.format()).await
                }
                Transport::Git { .. } => {
                    let (r, o) = (Arc::clone(&repo), origin.clone());
                    tokio::task::spawn_blocking(move || probe::git_tips(&r, &o))
                        .await
                        .unwrap_or_else(|e| Err(ProbeFailure::Transient(e.to_string())))
                }
            };
            let tips = match tips {
                Ok(tips) => tips,
                Err(ProbeFailure::Refused(why)) if matches!(transport, Transport::Https { .. }) => {
                    tracing::info!(
                        target: "zend::origin_watch",
                        repo = %self.name,
                        "origin refused the HTTPS probe ({why}); probing with git instead",
                    );
                    transport = Transport::Git { local: false };
                    cadence = Cadence::new(interval(&transport));
                    continue;
                }
                Err(ProbeFailure::Refused(why) | ProbeFailure::Transient(why)) => {
                    self.failed(&mut cadence, &why);
                    if matches!(transport, Transport::Https { .. })
                        && cadence.failures() >= HTTPS_GIVE_UP
                    {
                        tracing::info!(
                            target: "zend::origin_watch",
                            repo = %self.name,
                            "origin has not answered over HTTPS {HTTPS_GIVE_UP} times running; \
                             probing with git instead",
                        );
                        transport = Transport::Git { local: false };
                        cadence = Cadence::new(interval(&transport));
                    }
                    continue;
                }
            };
            let (r, o) = (Arc::clone(&repo), origin.clone());
            match tokio::task::spawn_blocking(move || check(&r, &o, &tips)).await {
                Ok(Ok(Checked::Unchanged)) => cadence.succeeded(),
                Ok(Ok(Checked::Fetched(moved))) => {
                    cadence.succeeded();
                    tracing::info!(
                        target: "zend::origin_watch",
                        repo = %self.name,
                        moved,
                        "origin moved; fetched",
                    );
                    if moved > 0 {
                        (self.on_change)();
                    }
                }
                Ok(Err(e)) => self.failed(&mut cadence, &format!("fetch: {e}")),
                Err(e) => self.failed(&mut cadence, &format!("fetch: {e}")),
            }
        }
    }

    /// Back off; say so once per failing streak, not once per attempt.
    fn failed(&self, cadence: &mut Cadence, why: &str) {
        if !cadence.failing() {
            tracing::warn!(
                target: "zend::origin_watch",
                repo = %self.name,
                "origin could not be checked, backing off: {why}",
            );
        } else {
            tracing::debug!(target: "zend::origin_watch", repo = %self.name, "still failing: {why}");
        }
        cadence.failed();
    }
}

/// How often `transport` probes when healthy.
fn interval(transport: &Transport) -> Duration {
    match transport {
        Transport::Https { .. } | Transport::Git { local: true } => FAST,
        Transport::Git { local: false } => SLOW,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use zend_vfs::RepoSpec;

    use super::*;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "core.hooksPath=",
                "-c",
                "user.name=T",
                "-c",
                "user.email=t@x",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    /// A bare origin, a clone of it that pushes as someone else would, and
    /// the repository under watch, fetched up to date.
    fn watched() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let (origin, other, repo) = (
            root.path().join("origin"),
            root.path().join("other"),
            root.path().join("repo"),
        );
        for d in [&origin, &other, &repo] {
            std::fs::create_dir_all(d).unwrap();
        }
        git(&origin, &["init", "-q", "--bare"]);
        git(&other, &["init", "-q"]);
        git(&other, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        std::fs::write(other.join("a.rs"), "fn a() {}\n").unwrap();
        git(&other, &["add", "-A"]);
        git(&other, &["commit", "-q", "-m", "first"]);
        git(
            &other,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git(&other, &["push", "-q", "origin", "main"]);
        git(&repo, &["init", "-q"]);
        git(
            &repo,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git(&repo, &["fetch", "-q", "origin"]);
        (root, other, repo)
    }

    fn origin() -> RemoteName {
        RemoteName::parse("origin").unwrap()
    }

    /// **An origin holding what the tracking refs hold costs no fetch; a push
    /// is seen and fetched**, and afterwards the record holds it.
    #[test]
    fn a_push_is_fetched_and_an_unchanged_origin_is_not() {
        let (_root, other, dir) = watched();
        let repo = Repo::open(&dir).unwrap();
        let tips = probe::git_tips(&repo, &origin()).unwrap();
        assert_eq!(check(&repo, &origin(), &tips).unwrap(), Checked::Unchanged);

        std::fs::write(other.join("a.rs"), "fn b() {}\n").unwrap();
        git(&other, &["commit", "-q", "-am", "second"]);
        git(&other, &["push", "-q", "origin", "main", "main:topic"]);
        let tips = probe::git_tips(&repo, &origin()).unwrap();
        assert_eq!(tips.len(), 2);
        assert_eq!(check(&repo, &origin(), &tips).unwrap(), Checked::Fetched(2));
        let record = repo.record_branches().unwrap();
        assert_eq!(
            record
                .iter()
                .map(|b| (b.name.as_str(), b.tip.clone()))
                .collect::<Vec<_>>(),
            tips.iter()
                .map(|t| (t.name.as_str(), t.tip.clone()))
                .collect::<Vec<_>>(),
        );
        assert_eq!(check(&repo, &origin(), &tips).unwrap(), Checked::Unchanged);
    }

    /// **The watcher, end to end**: a push someone else makes to origin is
    /// probed, fetched and reported within a few ticks, and stopping returns
    /// at once rather than after the watcher's next tick.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_push_wakes_the_watcher_and_stop_returns_promptly() {
        let (root, other, _repo) = watched();
        let workspace = Workspace::new(root.path(), vec![RepoSpec::named("repo")]).unwrap();
        let changed = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&changed);
        let watch = OriginWatch::start(
            &workspace,
            None,
            Arc::new(move || {
                seen.fetch_add(1, Ordering::SeqCst);
            }),
        );

        std::fs::write(other.join("a.rs"), "fn pushed() {}\n").unwrap();
        git(&other, &["commit", "-q", "-am", "pushed"]);
        git(&other, &["push", "-q", "origin", "main"]);
        let deadline = Instant::now() + Duration::from_secs(20);
        while changed.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(
            changed.load(Ordering::SeqCst),
            1,
            "the push was fetched once"
        );

        let stopping = Instant::now();
        tokio::time::timeout(Duration::from_secs(5), watch.stop())
            .await
            .expect("stop returns without waiting out a tick");
        assert!(stopping.elapsed() < Duration::from_secs(5));
    }

    /// **A branch deleted on origin is pruned** from the record.
    #[test]
    fn a_branch_deleted_on_origin_leaves_the_record() {
        let (_root, other, dir) = watched();
        git(&other, &["push", "-q", "origin", "main:gone"]);
        let repo = Repo::open(&dir).unwrap();
        check(
            &repo,
            &origin(),
            &probe::git_tips(&repo, &origin()).unwrap(),
        )
        .unwrap();
        assert_eq!(repo.record_branches().unwrap().len(), 2);

        git(&other, &["push", "-q", "origin", "--delete", "gone"]);
        let tips = probe::git_tips(&repo, &origin()).unwrap();
        assert_eq!(check(&repo, &origin(), &tips).unwrap(), Checked::Fetched(1));
        let names: Vec<String> = repo
            .record_branches()
            .unwrap()
            .into_iter()
            .map(|b| b.name.as_str().to_string())
            .collect();
        assert_eq!(names, ["main"]);
    }
}
