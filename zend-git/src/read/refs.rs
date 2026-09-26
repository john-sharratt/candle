//! Resolving revisions, and listing branches and remotes.

use std::collections::BTreeMap;

use crate::error::GitError;
use crate::redact::redact_urls;
use crate::runner::utf8;
use crate::types::{BranchName, Oid, RefName, RemoteName, Rev};
use crate::Repo;

/// A local branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    pub name: BranchName,
    pub oid: Oid,
    pub upstream: Option<Upstream>,
}

/// The branch a local branch tracks, and how far apart they are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upstream {
    /// `None` when the upstream is another local branch — `git branch
    /// --track x main`, which git records as the remote `.`.
    pub remote: Option<RemoteName>,
    pub branch: BranchName,
    pub ahead: u32,
    pub behind: u32,
    /// The tracked branch no longer exists.
    pub gone: bool,
}

impl Upstream {
    /// The ref this branch is compared against: the remote-tracking branch,
    /// or the local branch itself.
    pub fn tracking_ref(&self) -> RefName {
        match &self.remote {
            Some(remote) => remote.tracking(&self.branch),
            None => self.branch.to_ref(),
        }
    }
}

/// Where a first-parent walk back from a revision lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ancestor {
    Found(Oid),
    /// The history ends first: the root is `depth` commits back.
    PastRoot {
        depth: u32,
    },
}

/// A configured remote. Its URLs are for display: any credential embedded
/// in one (`https://user:token@host`) is redacted to `***`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remote {
    pub name: RemoteName,
    pub fetch_url: String,
    pub push_url: Option<String>,
}

/// `ahead 2, behind 1` / `ahead 2` / `behind 1` / `gone` / empty.
fn parse_track(track: &str) -> Result<(u32, u32, bool), GitError> {
    if track == "gone" {
        return Ok((0, 0, true));
    }
    let (mut ahead, mut behind) = (0, 0);
    for part in track.split(", ").filter(|p| !p.is_empty()) {
        let bad = || GitError::malformed("for-each-ref", format!("tracking {track:?}"));
        let (word, n) = part.split_once(' ').ok_or_else(bad)?;
        let n: u32 = n.parse().map_err(|_| bad())?;
        match word {
            "ahead" => ahead = n,
            "behind" => behind = n,
            _ => return Err(bad()),
        }
    }
    Ok((ahead, behind, false))
}

const BRANCH_FORMAT: &str =
    "%(refname)%00%(objectname)%00%(upstream:remotename)%00%(upstream:remoteref)%00%(upstream:track,nobracket)";

fn parse_branches(out: &str) -> Result<Vec<Branch>, GitError> {
    let mut branches = Vec::new();
    for line in out.lines().filter(|l| !l.is_empty()) {
        let fields: Vec<&str> = line.split('\0').collect();
        let [refname, oid, remote, remote_ref, track] = fields[..] else {
            return Err(GitError::malformed("for-each-ref", line.to_string()));
        };
        let name = RefName::parse(refname)?
            .branch()
            .ok_or_else(|| GitError::malformed("for-each-ref", refname.to_string()))?;
        let upstream = if remote.is_empty() || remote_ref.is_empty() {
            None
        } else {
            let branch = RefName::parse(remote_ref)?
                .branch()
                .ok_or_else(|| GitError::malformed("for-each-ref", remote_ref.to_string()))?;
            let (ahead, behind, gone) = parse_track(track)?;
            Some(Upstream {
                remote: match remote {
                    "." => None,
                    name => Some(RemoteName::parse(name)?),
                },
                branch,
                ahead,
                behind,
                gone,
            })
        };
        branches.push(Branch {
            name,
            oid: Oid::parse(oid)?,
            upstream,
        });
    }
    Ok(branches)
}

/// `config -z` output: `key\nvalue\0` per entry.
fn parse_remotes(out: &[u8]) -> Result<Vec<Remote>, GitError> {
    let text =
        std::str::from_utf8(out).map_err(|e| GitError::malformed("config", e.to_string()))?;
    let mut fetch: BTreeMap<String, String> = BTreeMap::new();
    let mut push: BTreeMap<String, String> = BTreeMap::new();
    for entry in text.split('\0').filter(|e| !e.is_empty()) {
        let (key, value) = entry
            .split_once('\n')
            .ok_or_else(|| GitError::malformed("config", entry.to_string()))?;
        let rest = key
            .strip_prefix("remote.")
            .ok_or_else(|| GitError::malformed("config", key.to_string()))?;
        if let Some(name) = rest.strip_suffix(".pushurl") {
            push.insert(name.to_string(), value.to_string());
        } else if let Some(name) = rest.strip_suffix(".url") {
            fetch.insert(name.to_string(), value.to_string());
        }
    }
    fetch
        .into_iter()
        .map(|(name, url)| {
            Ok(Remote {
                push_url: push.get(&name).map(|u| redact_urls(u)),
                name: RemoteName::parse(&name)?,
                fetch_url: redact_urls(&url),
            })
        })
        .collect()
}

impl Repo {
    /// The commit `rev` names.
    ///
    /// `rev-parse` takes no `--end-of-options` before 2.30; the value is
    /// safe without it, because no [`Rev`] spelling begins with `-`.
    pub fn resolve(&self, rev: &Rev) -> Result<Oid, GitError> {
        let spec = rev.spec();
        let out = self
            .git("rev-parse")
            .arg("--verify")
            .arg(format!("{spec}^{{commit}}"))
            .read_only()
            .about_rev(spec)
            .run_ok()?;
        Oid::parse(utf8("rev-parse", out)?.trim_end())
    }

    /// The object `rev` names, unpeeled: an annotated tag's own id rather
    /// than its commit's. Publishing a tag sends this, or the remote gets a
    /// lightweight tag in its place.
    pub fn resolve_object(&self, rev: &Rev) -> Result<Oid, GitError> {
        let spec = rev.spec();
        let out = self
            .git("rev-parse")
            .arg("--verify")
            .arg(format!("{spec}^{{object}}"))
            .read_only()
            .about_rev(spec)
            .run_ok()?;
        Oid::parse(utf8("rev-parse", out)?.trim_end())
    }

    /// The commit `back` steps behind `rev` along first parents — git's
    /// `rev~back`, the mainline a merge was made on — or, when the history
    /// is shorter than that, how far back it does go.
    pub fn first_parent_ancestor(&self, rev: &Rev, back: u32) -> Result<Ancestor, GitError> {
        let base = self.resolve(rev)?;
        let out = self
            .git("rev-parse")
            .args(["-q", "--verify"])
            .arg(format!("{base}~{back}^{{commit}}"))
            .read_only()
            .run_accepting(&[0, 1])?;
        if out.status == Some(0) {
            return Ok(Ancestor::Found(Oid::parse(
                utf8("rev-parse", out.stdout)?.trim_end(),
            )?));
        }
        let count = self
            .git("rev-list")
            .args(["--first-parent", "--count", "--end-of-options"])
            .arg(base.as_str())
            .read_only()
            .run_ok()?;
        let count: u32 = utf8("rev-list", count)?
            .trim()
            .parse()
            .map_err(|_| GitError::malformed("rev-list", "a commit count"))?;
        Ok(Ancestor::PastRoot {
            depth: count.saturating_sub(1),
        })
    }

    /// What `name` points at, or `None` when it does not exist. A
    /// [`RefName`] begins with `refs/`, so it is never read as a flag.
    pub fn ref_target(&self, name: &RefName) -> Result<Option<Oid>, GitError> {
        let out = self
            .git("rev-parse")
            .args(["-q", "--verify"])
            .arg(name.as_str())
            .read_only()
            .run_accepting(&[0, 1])?;
        match out.status {
            Some(0) => Ok(Some(Oid::parse(utf8("rev-parse", out.stdout)?.trim_end())?)),
            _ => Ok(None),
        }
    }

    /// Every local branch, with its upstream when one is configured.
    pub fn branches(&self) -> Result<Vec<Branch>, GitError> {
        let out = self
            .git("for-each-ref")
            .arg(format!("--format={BRANCH_FORMAT}"))
            .arg("refs/heads/")
            .read_only()
            .run_ok()?;
        parse_branches(&utf8("for-each-ref", out)?)
    }

    /// Every configured remote.
    pub fn remotes(&self) -> Result<Vec<Remote>, GitError> {
        let out = self
            .git("config")
            .args(["-z", "--get-regexp", r"^remote\..*\.(url|pushurl)$"])
            .read_only()
            .run_accepting(&[0, 1])?;
        parse_remotes(&out.stdout)
    }

    /// Refuse `name` unless it is a configured remote.
    ///
    /// Git takes a remote argument as a URL or a path when no remote has
    /// that name, so `push("foo")` with no `foo` configured would push to a
    /// repository at `<repo>/foo`. Every network operation checks first.
    pub(crate) fn require_remote(&self, name: &RemoteName) -> Result<(), GitError> {
        if self.remotes()?.iter().any(|r| &r.name == name) {
            Ok(())
        } else {
            Err(GitError::UnknownRemote {
                remote: name.clone(),
            })
        }
    }

    /// The best common ancestor of `a` and `b`, if they share history.
    pub fn merge_base(&self, a: &Rev, b: &Rev) -> Result<Option<Oid>, GitError> {
        let out = self
            .git("merge-base")
            .arg("--end-of-options")
            .args([a.spec(), b.spec()])
            .read_only()
            .run_accepting(&[0, 1])?;
        match out.status {
            Some(0) => Ok(Some(Oid::parse(
                utf8("merge-base", out.stdout)?.trim_end(),
            )?)),
            _ => Ok(None),
        }
    }

    /// Whether `ancestor` is reachable from `descendant`.
    pub fn is_ancestor(&self, ancestor: &Rev, descendant: &Rev) -> Result<bool, GitError> {
        let out = self
            .git("merge-base")
            .args(["--is-ancestor", "--end-of-options"])
            .args([ancestor.spec(), descendant.spec()])
            .read_only()
            .run_accepting(&[0, 1])?;
        Ok(out.status == Some(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::types::TagName;

    #[test]
    fn tracking_counts_parse() {
        assert_eq!(parse_track("").unwrap(), (0, 0, false));
        assert_eq!(parse_track("ahead 2").unwrap(), (2, 0, false));
        assert_eq!(parse_track("behind 3").unwrap(), (0, 3, false));
        assert_eq!(parse_track("ahead 2, behind 3").unwrap(), (2, 3, false));
        assert_eq!(parse_track("gone").unwrap(), (0, 0, true));
        assert!(parse_track("sideways 1").is_err());
    }

    #[test]
    fn remotes_parse_from_config_bytes() {
        let bytes = b"remote.origin.url\ngit@github.com:x/y.git\0remote.origin.pushurl\nssh://push/y\0remote.up.stream.url\nfile:///u\0";
        let remotes = parse_remotes(bytes).unwrap();
        assert_eq!(remotes.len(), 2);
        assert_eq!(remotes[0].name.as_str(), "origin");
        assert_eq!(remotes[0].fetch_url, "git@github.com:x/y.git");
        assert_eq!(remotes[0].push_url.as_deref(), Some("ssh://push/y"));
        assert_eq!(remotes[1].name.as_str(), "up.stream");
        assert_eq!(remotes[1].push_url, None);
    }

    #[test]
    fn credentials_in_remote_urls_are_redacted() {
        let bytes = b"remote.origin.url\nhttps://me:ghp_secret@github.com/x/y.git\0remote.origin.pushurl\nhttps://ghp_push@github.com/x/y.git\0";
        let remotes = parse_remotes(bytes).unwrap();
        assert_eq!(remotes[0].fetch_url, "https://***@github.com/x/y.git");
        assert_eq!(
            remotes[0].push_url.as_deref(),
            Some("https://***@github.com/x/y.git")
        );
    }

    #[test]
    fn resolve_names_commits_and_refuses_unknown_revisions() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let c = t.commit_all("first");
        let repo = t.repo();
        assert_eq!(repo.resolve(&Rev::Head).unwrap(), c);
        assert_eq!(
            repo.resolve(&Rev::Branch(BranchName::parse("main").unwrap()))
                .unwrap(),
            c
        );
        let missing = Rev::Branch(BranchName::parse("nope").unwrap());
        match repo.resolve(&missing) {
            Err(GitError::UnknownRevision { rev }) => assert_eq!(rev, "refs/heads/nope"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_branch_named_like_a_file_resolves_to_the_branch() {
        let t = TestRepo::init();
        t.write("feature", b"a file, not the branch\n");
        let first = t.commit_all("first");
        t.git(&["branch", "feature"]);
        t.write("b.txt", b"b\n");
        t.commit_all("second");
        let repo = t.repo();
        assert_eq!(
            repo.resolve(&Rev::Branch(BranchName::parse("feature").unwrap()))
                .unwrap(),
            first
        );
    }

    /// **An annotated tag resolves to itself unpeeled** and to its commit
    /// peeled; a lightweight tag is its commit either way.
    #[test]
    fn resolve_object_keeps_an_annotated_tag() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let c = t.commit_all("first");
        t.git(&["tag", "-a", "v1", "-m", "release"]);
        t.git(&["tag", "light"]);
        let repo = t.repo();
        let v1 = Rev::Tag(TagName::parse("v1").unwrap());
        let light = Rev::Tag(TagName::parse("light").unwrap());
        assert_eq!(repo.resolve_object(&v1).unwrap(), t.oid("refs/tags/v1"));
        assert_ne!(repo.resolve_object(&v1).unwrap(), c);
        assert_eq!(repo.resolve(&v1).unwrap(), c);
        assert_eq!(repo.resolve_object(&light).unwrap(), c);
    }

    /// **Counting back follows first parents**: at a merge, one back is the
    /// mainline the merge was made on, never the merged branch's tip.
    #[test]
    fn first_parent_ancestry_stays_on_the_mainline() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        t.git(&["checkout", "-q", "-b", "side"]);
        t.write("s.txt", b"s\n");
        t.commit_all("side one");
        t.write("s.txt", b"s2\n");
        t.commit_all("side two");
        t.git(&["checkout", "-q", "main"]);
        t.write("m.txt", b"m\n");
        let main = t.commit_all("main");
        t.git(&["merge", "-q", "--no-ff", "-m", "merge", "side"]);
        let merge = t.oid("HEAD");
        let repo = t.repo();

        let head = Rev::Oid(merge.clone());
        let back = |n| repo.first_parent_ancestor(&head, n).unwrap();
        assert_eq!(back(0), Ancestor::Found(merge));
        assert_eq!(back(1), Ancestor::Found(main));
        assert_eq!(back(2), Ancestor::Found(base));
        // The merged branch's two commits do not count towards the depth.
        assert_eq!(back(3), Ancestor::PastRoot { depth: 2 });
        assert!(repo
            .first_parent_ancestor(&Rev::Branch(BranchName::parse("nope").unwrap()), 1)
            .is_err());
    }

    #[test]
    fn ref_target_reports_absence_as_none() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let c = t.commit_all("first");
        let repo = t.repo();
        let main = BranchName::parse("main").unwrap().to_ref();
        assert_eq!(repo.ref_target(&main).unwrap(), Some(c));
        let absent = RefName::parse("refs/heads/absent").unwrap();
        assert_eq!(repo.ref_target(&absent).unwrap(), None);
    }

    #[test]
    fn branches_report_their_upstream_and_distance() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "-u", "origin", "main"]);
        t.write("b.txt", b"b\n");
        let second = t.commit_all("second");
        t.git(&["branch", "loose"]);

        let branches = t.repo().branches().unwrap();
        assert_eq!(branches.len(), 2);
        let loose = &branches[0];
        assert_eq!(loose.name.as_str(), "loose");
        assert_eq!(loose.upstream, None);
        let main = &branches[1];
        assert_eq!(main.name.as_str(), "main");
        assert_eq!(main.oid, second);
        assert_eq!(
            main.upstream,
            Some(Upstream {
                remote: Some(RemoteName::parse("origin").unwrap()),
                branch: BranchName::parse("main").unwrap(),
                ahead: 1,
                behind: 0,
                gone: false,
            })
        );
        assert_eq!(
            main.upstream.as_ref().unwrap().tracking_ref().as_str(),
            "refs/remotes/origin/main"
        );
    }

    /// **A branch tracking another local branch lists like any other.** Git
    /// records its remote as `.`, which is no remote name.
    #[test]
    fn a_branch_tracking_a_local_branch_is_listed() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("first");
        t.git(&["branch", "--track", "topic", "main"]);
        t.git(&["checkout", "-q", "topic"]);
        t.write("b.txt", b"b\n");
        t.commit_all("second");

        let branches = t.repo().branches().unwrap();
        let topic = branches
            .iter()
            .find(|b| b.name.as_str() == "topic")
            .unwrap();
        let up = topic.upstream.as_ref().unwrap();
        assert_eq!(
            up,
            &Upstream {
                remote: None,
                branch: BranchName::parse("main").unwrap(),
                ahead: 1,
                behind: 0,
                gone: false,
            }
        );
        assert_eq!(up.tracking_ref().as_str(), "refs/heads/main");
    }

    #[test]
    fn remotes_lists_configured_remotes() {
        let t = TestRepo::init();
        assert!(t.repo().remotes().unwrap().is_empty());
        t.git(&["remote", "add", "origin", "git@example.com:x/y.git"]);
        let remotes = t.repo().remotes().unwrap();
        assert_eq!(
            remotes,
            vec![Remote {
                name: RemoteName::parse("origin").unwrap(),
                fetch_url: "git@example.com:x/y.git".into(),
                push_url: None,
            }]
        );
    }

    #[test]
    fn merge_base_and_ancestry() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        t.git(&["checkout", "-q", "-b", "side"]);
        t.write("s.txt", b"s\n");
        let side = t.commit_all("side");
        t.git(&["checkout", "-q", "main"]);
        t.write("m.txt", b"m\n");
        let main = t.commit_all("main");
        let repo = t.repo();
        let (side, main, base) = (Rev::Oid(side), Rev::Oid(main), base);
        assert_eq!(repo.merge_base(&side, &main).unwrap(), Some(base.clone()));
        assert!(repo.is_ancestor(&Rev::Oid(base.clone()), &side).unwrap());
        assert!(!repo.is_ancestor(&side, &main).unwrap());

        t.git(&["checkout", "-q", "--orphan", "unrelated"]);
        t.git(&["rm", "-q", "-rf", "."]);
        t.write("u.txt", b"u\n");
        let unrelated = Rev::Oid(t.commit_all("unrelated"));
        assert_eq!(repo.merge_base(&unrelated, &main).unwrap(), None);
    }
}
