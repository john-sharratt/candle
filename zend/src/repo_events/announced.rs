//! The pusher every workspace repository's pushes go through
//! (`docs/mcp_file_tools.md` §5.3, §5.4).
//!
//! **Gated.** For a repository with a `gate:`, each branch a push would move is
//! first pushed as a candidate branch, `zen/candidate/<id>`, and asked of the
//! gate on `gate/<repo>`:
//!
//! ```json
//! {"id": "3f2a…", "ref": "refs/heads/zen/candidate/3f2a…", "commit": "4f1c…",
//!  "base": "a3f9…", "branch": "main", "paths": ["apps/recipes/pages/home.js"]}
//! ```
//!
//! The gate fetches the candidate, checks it, and answers on
//! `gate/<repo>/result/<id>` with `{"ok": true}` or `{"ok": false, "reason"}`.
//! The candidate is deleted whatever the answer. Only on `ok` is the real push
//! made; a refusal, a gate that is not connected, or one that does not answer
//! within [`GATE_WAIT`] rejects every spec of the push as a remote's hook
//! would (`Rejection::Remote`), so each caller reports it as it reports any
//! refusal and nothing moved.
//!
//! **Announced.** Each branch the push moved is published on `repo/<repo>`:
//!
//! ```json
//! {"repo": "yenbot", "branch": "main", "commit": "4f1c…", "paths": [...]}
//! ```

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use uuid::Uuid;
use zend_vfs::{
    BranchName, GitError, Lease, Oid, Push, PushAction, PushOutcome, PushResult, PushSpec,
    PushTarget, Rejection, RemoteName, Repo, Rev,
};

use super::{Bus, Unasked};

/// The longest a push waits on its gate.
pub const GATE_WAIT: Duration = Duration::from_secs(10);

/// The branch a gated push's candidate is pushed to.
fn candidate_branch(id: &str) -> Result<BranchName, GitError> {
    BranchName::parse(&format!("zen/candidate/{id}"))
}

pub struct Announced {
    inner: Arc<dyn Push>,
    repo: String,
    gate: Option<String>,
    bus: Arc<dyn Bus>,
}

impl Announced {
    /// `inner`'s pushes of the workspace repository `repo`, gated by `gate`
    /// when it has one, announced on `bus`.
    pub fn shared(
        inner: Arc<dyn Push>,
        repo: impl Into<String>,
        gate: Option<String>,
        bus: Arc<dyn Bus>,
    ) -> Arc<dyn Push> {
        Arc::new(Announced {
            inner,
            repo: repo.into(),
            gate,
            bus,
        })
    }

    /// The paths that differ from `base` to `commit`; empty when they cannot
    /// be read, which only makes the message less specific.
    fn paths(&self, repo: &Repo, base: Option<&Oid>, commit: &Oid) -> Vec<String> {
        let Some(base) = base else {
            return Vec::new();
        };
        match repo.diff(&Rev::Oid(base.clone()), &Rev::Oid(commit.clone()), &[]) {
            Ok(entries) => entries
                .iter()
                .filter_map(|e| e.new.as_ref().or(e.old.as_ref()))
                .map(|side| side.path.as_str().to_string())
                .collect(),
            Err(e) => {
                tracing::warn!(repo = %self.repo, "the paths a push changes could not be read: {e}");
                Vec::new()
            }
        }
    }

    /// Ask the gate about moving `branch` to `commit`: `None` when it may,
    /// else why not.
    fn ask_gate(
        &self,
        gate: &str,
        repo: &Repo,
        remote: &RemoteName,
        commit: &Oid,
        branch: &BranchName,
        lease: &Lease,
    ) -> Result<Option<String>, GitError> {
        let id = Uuid::new_v4().simple().to_string();
        let candidate = candidate_branch(&id)?;
        let placed = self.inner.push(
            repo,
            remote,
            &[PushSpec::branch(
                commit.clone(),
                candidate.clone(),
                Lease::Absent,
            )],
        )?;
        if let Some(r) = placed.iter().find(|r| !r.accepted()) {
            return Ok(Some(format!(
                "the candidate for the gate could not be pushed: {:?}",
                r.outcome
            )));
        }
        let base = match lease {
            Lease::Expect(oid) => Some(oid),
            Lease::Absent => None,
        };
        let ask = json!({
            "id": id,
            "ref": candidate.to_ref().as_str(),
            "commit": commit.as_str(),
            "base": base.map(Oid::as_str),
            "branch": branch.as_str(),
            "paths": self.paths(repo, base, commit),
        });
        let topic = format!("gate/{}", self.repo);
        let answer = self
            .bus
            .ask(&topic, ask, &format!("{topic}/result/{id}"), GATE_WAIT);
        let removed = self.inner.push(
            repo,
            remote,
            &[PushSpec::delete_branch(candidate.clone(), commit.clone())],
        );
        if !removed.is_ok_and(|r| r.iter().all(PushResult::accepted)) {
            tracing::warn!(repo = %self.repo, candidate = %candidate.as_str(), "a gate's candidate branch could not be deleted");
        }
        Ok(verdict(gate, answer))
    }

    fn announce(&self, repo: &Repo, spec: &PushSpec) {
        let (PushAction::Update(commit), PushTarget::Branch(branch)) = (&spec.action, &spec.target)
        else {
            return;
        };
        let base = match &spec.lease {
            Lease::Expect(oid) => Some(oid),
            Lease::Absent => None,
        };
        self.bus.publish(
            &format!("repo/{}", self.repo),
            json!({
                "repo": self.repo,
                "branch": branch.as_str(),
                "commit": commit.as_str(),
                "paths": self.paths(repo, base, commit),
            }),
        );
    }
}

/// What a gate's answer means: `None` to go ahead, else the refusal.
fn verdict(gate: &str, answer: Result<Value, Unasked>) -> Option<String> {
    match answer {
        Ok(v) if v["ok"] == json!(true) => None,
        Ok(v) => Some(format!(
            "the gate {gate} refused it: {}",
            v["reason"].as_str().unwrap_or("no reason given")
        )),
        Err(Unasked::NoSubscriber) => Some(format!(
            "the gate {gate} is not connected, so the change was not checked and nothing moved"
        )),
        Err(Unasked::Timeout) => Some(format!(
            "the gate {gate} did not answer within {} s, so nothing moved",
            GATE_WAIT.as_secs()
        )),
        Err(Unasked::Failed(why)) => Some(format!(
            "the gate {gate} could not be asked ({why}), so nothing moved"
        )),
    }
}

impl Push for Announced {
    fn push(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        if let Some(gate) = &self.gate {
            for spec in specs {
                let (PushAction::Update(commit), PushTarget::Branch(branch)) =
                    (&spec.action, &spec.target)
                else {
                    continue;
                };
                if let Some(refusal) =
                    self.ask_gate(gate, repo, remote, commit, branch, &spec.lease)?
                {
                    return Ok(specs
                        .iter()
                        .map(|s| PushResult {
                            target: s.target.clone(),
                            outcome: PushOutcome::Rejected(Rejection::Remote(refusal.clone())),
                        })
                        .collect());
                }
            }
        }
        let results = self.inner.push(repo, remote, specs)?;
        for spec in specs {
            let moved = results.iter().any(|r| {
                r.target == spec.target && r.accepted() && r.outcome != PushOutcome::UpToDate
            });
            if moved {
                self.announce(repo, spec);
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;
    use std::sync::Mutex;

    use zend_vfs::TagName;

    use super::*;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    /// A repository with two commits; their ids, oldest first.
    fn two_commits(dir: &Path) -> (Oid, Oid) {
        git(dir, &["init", "-q", "-b", "main"]);
        std::fs::write(dir.join("a.txt"), "a\n").unwrap();
        git(dir, &["add", "a.txt"]);
        git(dir, &["commit", "-q", "-m", "one"]);
        let one = git(dir, &["rev-parse", "HEAD"]);
        std::fs::create_dir_all(dir.join("apps/x")).unwrap();
        std::fs::write(dir.join("apps/x/app.yaml"), "id: x\n").unwrap();
        git(dir, &["add", "apps/x/app.yaml"]);
        git(dir, &["commit", "-q", "-m", "two"]);
        let two = git(dir, &["rev-parse", "HEAD"]);
        (Oid::parse(&one).unwrap(), Oid::parse(&two).unwrap())
    }

    /// A pusher that takes everything and records what it was asked.
    #[derive(Default)]
    struct Taker(Mutex<Vec<PushSpec>>);

    impl Push for Taker {
        fn push(
            &self,
            _repo: &Repo,
            _remote: &RemoteName,
            specs: &[PushSpec],
        ) -> Result<Vec<PushResult>, GitError> {
            self.0.lock().unwrap().extend(specs.iter().cloned());
            Ok(specs
                .iter()
                .map(|s| PushResult {
                    target: s.target.clone(),
                    outcome: match s.action {
                        PushAction::Delete => PushOutcome::Deleted,
                        PushAction::Update(_) => PushOutcome::FastForward,
                    },
                })
                .collect())
        }
    }

    /// A bus that records what it is sent and answers asks with `answer`.
    struct Recorder {
        answer: Result<Value, Unasked>,
        published: Mutex<Vec<(String, Value)>>,
        asked: Mutex<Vec<(String, Value, String)>>,
    }

    impl Recorder {
        fn answering(answer: Result<Value, Unasked>) -> Arc<Recorder> {
            Arc::new(Recorder {
                answer,
                published: Mutex::default(),
                asked: Mutex::default(),
            })
        }
    }

    impl Bus for Recorder {
        fn publish(&self, topic: &str, payload: Value) {
            self.published.lock().unwrap().push((topic.into(), payload));
        }

        fn ask(
            &self,
            topic: &str,
            payload: Value,
            reply: &str,
            _wait: Duration,
        ) -> Result<Value, Unasked> {
            self.asked
                .lock()
                .unwrap()
                .push((topic.into(), payload, reply.into()));
            self.answer.clone()
        }
    }

    /// A repository with two commits, pushed through [`Announced`] over a
    /// recording pusher and bus.
    struct Setup {
        _dir: tempfile::TempDir,
        repo: Repo,
        taker: Arc<Taker>,
        bus: Arc<Recorder>,
        push: Arc<dyn Push>,
        one: Oid,
        two: Oid,
    }

    fn setup(gate: Option<&str>, answer: Result<Value, Unasked>) -> Setup {
        let dir = tempfile::tempdir().unwrap();
        let (one, two) = two_commits(dir.path());
        let repo = Repo::open(dir.path()).unwrap();
        let taker = Arc::new(Taker::default());
        let bus = Recorder::answering(answer);
        let push = Announced::shared(
            taker.clone(),
            "yenbot",
            gate.map(str::to_string),
            bus.clone(),
        );
        Setup {
            _dir: dir,
            repo,
            taker,
            bus,
            push,
            one,
            two,
        }
    }

    fn main_to(two: &Oid, one: &Oid) -> PushSpec {
        PushSpec::branch(
            two.clone(),
            BranchName::parse("main").unwrap(),
            Lease::Expect(one.clone()),
        )
    }

    /// **An ungated push is made and announced, with the paths it changed.**
    #[test]
    fn an_ungated_push_is_announced() {
        let Setup {
            _dir,
            repo,
            taker,
            bus,
            push,
            one,
            two,
        } = setup(None, Ok(json!({})));
        let origin = RemoteName::parse("origin").unwrap();
        let results = push.push(&repo, &origin, &[main_to(&two, &one)]).unwrap();
        assert!(results.iter().all(PushResult::accepted));
        assert_eq!(taker.0.lock().unwrap().len(), 1);
        assert!(bus.asked.lock().unwrap().is_empty());
        assert_eq!(
            *bus.published.lock().unwrap(),
            vec![(
                "repo/yenbot".to_string(),
                json!({"repo": "yenbot", "branch": "main", "commit": two.as_str(),
                       "paths": ["apps/x/app.yaml"]})
            )]
        );
    }

    /// **A gate's ok lets the push through**: the candidate is pushed, asked
    /// about, deleted, and then the branch moves and is announced.
    #[test]
    fn a_gates_ok_moves_the_branch() {
        let Setup {
            _dir,
            repo,
            taker,
            bus,
            push,
            one,
            two,
        } = setup(Some("yenbot"), Ok(json!({"ok": true})));
        let origin = RemoteName::parse("origin").unwrap();
        let results = push.push(&repo, &origin, &[main_to(&two, &one)]).unwrap();
        assert!(results.iter().all(PushResult::accepted));

        let asked = bus.asked.lock().unwrap();
        let (topic, ask, reply) = &asked[0];
        let id = ask["id"].as_str().unwrap();
        assert_eq!(topic, "gate/yenbot");
        assert_eq!(reply, &format!("gate/yenbot/result/{id}"));
        assert_eq!(
            *ask,
            json!({"id": id, "ref": format!("refs/heads/zen/candidate/{id}"),
                   "commit": two.as_str(), "base": one.as_str(), "branch": "main",
                   "paths": ["apps/x/app.yaml"]})
        );
        let candidate = candidate_branch(id).unwrap();
        assert_eq!(
            *taker.0.lock().unwrap(),
            vec![
                PushSpec::branch(two.clone(), candidate.clone(), Lease::Absent),
                PushSpec::delete_branch(candidate, two.clone()),
                main_to(&two, &one),
            ]
        );
        assert_eq!(bus.published.lock().unwrap().len(), 1);
    }

    /// **A refusal, a missing gate, or a silent one moves nothing**: every
    /// spec is rejected with the reason, the candidate is still deleted, and
    /// nothing is announced.
    #[test]
    fn a_gate_that_does_not_say_ok_moves_nothing() {
        for (answer, says) in [
            (
                Ok(json!({"ok": false, "reason": "home.js:14 uses a literal colour"})),
                "the gate yenbot refused it: home.js:14 uses a literal colour",
            ),
            (
                Err(Unasked::NoSubscriber),
                "the gate yenbot is not connected, so the change was not checked and nothing moved",
            ),
            (
                Err(Unasked::Timeout),
                "the gate yenbot did not answer within 10 s, so nothing moved",
            ),
        ] {
            let Setup {
                _dir,
                repo,
                taker,
                bus,
                push,
                one,
                two,
            } = setup(Some("yenbot"), answer);
            let origin = RemoteName::parse("origin").unwrap();
            let results = push.push(&repo, &origin, &[main_to(&two, &one)]).unwrap();
            assert_eq!(
                results,
                vec![PushResult {
                    target: PushTarget::Branch(BranchName::parse("main").unwrap()),
                    outcome: PushOutcome::Rejected(Rejection::Remote(says.into())),
                }]
            );
            let pushed = taker.0.lock().unwrap();
            assert_eq!(pushed.len(), 2, "the candidate, and its deletion");
            assert_eq!(pushed[1].action, PushAction::Delete);
            assert!(bus.published.lock().unwrap().is_empty());
        }
    }

    /// **A tag is not a branch move**: it is neither gated nor announced.
    #[test]
    fn a_tag_is_neither_gated_nor_announced() {
        let Setup {
            _dir,
            repo,
            taker,
            bus,
            push,
            two,
            ..
        } = setup(Some("yenbot"), Err(Unasked::NoSubscriber));
        let origin = RemoteName::parse("origin").unwrap();
        let tag = TagName::parse("v1").unwrap();
        let results = push
            .push(&repo, &origin, &[PushSpec::tag(two, tag)])
            .unwrap();
        assert!(results.iter().all(PushResult::accepted));
        assert_eq!(taker.0.lock().unwrap().len(), 1);
        assert!(bus.asked.lock().unwrap().is_empty());
        assert!(bus.published.lock().unwrap().is_empty());
    }
}
