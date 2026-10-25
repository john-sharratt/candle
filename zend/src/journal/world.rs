//! What the workspace can say about a thing a journal claim names.
//!
//! A typed claim about the workspace is checked against the conversation's own
//! files — its branch in each repository and the changes its tool rounds made
//! there — when the entry is checked. Two subjects are answered:
//!
//! | Subject | Attribute | Read as |
//! |---|---|---|
//! | `file:<repo>/<path>` | `exists` | `true` or `false` |
//! | `file:<repo>/<path>` | `blob` | the git blob id of the file the conversation reads, while it is the branch's |
//! | `branch:<repo>` | `head` | the commit the conversation's branch is at |
//! | `branch:<repo>` | `name` | the branch the conversation reads the repository through |
//! | `entry:<journal>#<n>` | `exists` | `true` when `<journal>` — a path, or a bare name resolved upward — is a journal above the conversation holding entry `n`; `false` otherwise |
//!
//! The last is the cross-journal citation check of a conversation owned
//! through `/mcp` (`docs/claude_zend_gateway.md` §5.2): a worker's claim that
//! rests on its thread's decision cites it, and an entry that is not there, or
//! not an ancestor's, is corrected to say so.
//!
//! Anything else reads as no answer, which the check treats as "the workspace
//! cannot say" rather than as a disagreement: the claim stays, as prose. So
//! does a file the session has changed, whose bytes are no blob yet.

use std::sync::Arc;

use journal_core::verify::World;
use zend_tools::tools::journal::address as journal_address;
use zend_vfs::{RepoFiles, Rev, VfsStore};

/// The subject prefix of a file.
pub const FILE: &str = "file:";
/// The subject prefix of a repository's branch.
pub const BRANCH: &str = "branch:";
/// The subject prefix of an entry of a journal above the conversation.
pub const ENTRY: &str = "entry:";

/// A journal above the conversation, as a citation names it: its path, the
/// bare name it answers to, and the entries a citation may name — those it
/// holds, and those a fold took, whose addresses still read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citable {
    pub path: String,
    pub name: String,
    pub entries: Vec<u64>,
}

/// One conversation's view of the workspace, and of the journals above it.
pub struct ConversationFiles {
    files: Arc<RepoFiles>,
    /// Nearest first.
    above: Vec<Citable>,
}

impl ConversationFiles {
    pub fn of(files: Arc<RepoFiles>) -> Self {
        Self {
            files,
            above: Vec::new(),
        }
    }

    /// This view, citing `above` — the journals above the conversation,
    /// nearest first.
    pub fn citing(self, above: Vec<Citable>) -> Self {
        Self { above, ..self }
    }

    /// Whether `address` — `<journal>#<n>`, or `#<n>.<m>` for one of its
    /// claims — names an entry of a journal above; `None` when it is no
    /// address.
    fn entry(&self, address: &str, attribute: &str) -> Option<String> {
        if attribute != "exists" {
            return None;
        }
        let (journal, Some(entry)) = journal_address(address.trim()) else {
            return None;
        };
        let held = self
            .above
            .iter()
            .filter(|a| a.path == journal || a.name == journal)
            .any(|a| a.entries.contains(&entry));
        Some(held.to_string())
    }

    fn file(&self, repo: &str, path: &str, attribute: &str) -> Option<String> {
        let store = self.files.repo(repo).ok()?;
        match attribute {
            "exists" => Some(store.read(path).ok()?.is_some().to_string()),
            "blob" => store
                .content_id(path)
                .ok()?
                .map(|(blob, _)| blob.to_string()),
            _ => None,
        }
    }

    fn branch(&self, repo: &str, attribute: &str) -> Option<String> {
        let store = self.files.repo(repo).ok()?;
        match attribute {
            "head" => head(&store),
            "name" => match store.rev()? {
                Rev::Branch(branch) => Some(branch.as_str().to_string()),
                _ => None,
            },
            _ => None,
        }
    }
}

fn head(store: &VfsStore) -> Option<String> {
    let base = store.peek_base().ok()??;
    base.commit().map(|commit| commit.to_string())
}

impl World for ConversationFiles {
    fn read(&self, subject: &str, attribute: &str) -> Option<String> {
        let subject = subject.trim();
        let attribute = attribute.trim().to_lowercase();
        if let Some(rest) = subject.strip_prefix(FILE) {
            let (repo, path) = rest.split_once('/')?;
            return self.file(repo.trim(), path.trim(), &attribute);
        }
        if let Some(repo) = subject.strip_prefix(BRANCH) {
            return self.branch(repo.trim(), &attribute);
        }
        if let Some(address) = subject.strip_prefix(ENTRY) {
            return self.entry(address, &attribute);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use zend_vfs::{RepoSpec, Workspace};

    use super::*;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "core.hooksPath=", "-c", "core.autocrlf=false"])
            .args(["-c", "user.name=T", "-c", "user.email=t@x"])
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

    /// A workspace with one repository `r` on `main`, holding `a.txt` as
    /// `hello\n`; and the commit `main` is at.
    fn repository() -> (tempfile::TempDir, Arc<RepoFiles>, String) {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("r");
        std::fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init", "-q"]);
        git(&dir, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        std::fs::write(dir.join("a.txt"), "hello\n").unwrap();
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "first"]);
        let head = git(&dir, &["rev-parse", "HEAD"]).trim().to_string();
        let workspace = Workspace::new(root.path(), vec![RepoSpec::named("r")]).unwrap();
        let files = Arc::new(RepoFiles::overlay(workspace).unwrap());
        (root, files, head)
    }

    /// The blob id git gives `hello\n`.
    const HELLO: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

    #[test]
    fn a_file_on_the_branch_reads_as_existing_at_its_blob() {
        let (_root, files, _) = repository();
        let world = ConversationFiles::of(files);
        assert_eq!(
            world.read("file:r/a.txt", "exists").as_deref(),
            Some("true")
        );
        assert_eq!(world.read("file:r/a.txt", "blob").as_deref(), Some(HELLO));
        assert_eq!(
            world.read(" file:r/a.txt ", " Blob ").as_deref(),
            Some(HELLO),
            "padding and case do not matter"
        );
    }

    #[test]
    fn a_missing_file_reads_as_not_existing_and_has_no_blob() {
        let (_root, files, _) = repository();
        let world = ConversationFiles::of(files);
        assert_eq!(
            world.read("file:r/nope.txt", "exists").as_deref(),
            Some("false")
        );
        assert_eq!(world.read("file:r/nope.txt", "blob"), None);
    }

    #[test]
    fn a_file_the_session_changed_exists_and_has_no_blob_yet() {
        let (_root, files, _) = repository();
        files
            .repo("r")
            .unwrap()
            .write("a.txt", "changed\n".into())
            .unwrap();
        let world = ConversationFiles::of(files);
        assert_eq!(
            world.read("file:r/a.txt", "exists").as_deref(),
            Some("true")
        );
        assert_eq!(world.read("file:r/a.txt", "blob"), None);
    }

    #[test]
    fn a_branch_reads_its_head_and_its_name() {
        let (_root, files, head) = repository();
        let world = ConversationFiles::of(files);
        assert_eq!(world.read("branch:r", "head"), Some(head));
        assert_eq!(world.read("branch:r", "name").as_deref(), Some("main"));
    }

    #[test]
    fn what_the_workspace_cannot_say_is_no_answer() {
        let (_root, files, _) = repository();
        let world = ConversationFiles::of(files);
        assert_eq!(
            world.read("file:nope/a.txt", "exists"),
            None,
            "no such repo"
        );
        assert_eq!(world.read("file:r", "exists"), None, "no path");
        assert_eq!(world.read("file:r/a.txt", "size"), None);
        assert_eq!(world.read("branch:r", "upstream"), None);
        assert_eq!(world.read("test:scan", "passed"), None);
    }

    /// **A citation of an entry above is checked**: there and an ancestor's
    /// reads true, by path or bare name, a claim's address by its entry;
    /// anything else reads false, so the claim is corrected.
    #[test]
    fn an_entry_above_is_cited_by_path_or_name_and_nothing_else_is() {
        let world = ConversationFiles::of(Arc::new(RepoFiles::detached())).citing(vec![
            Citable {
                path: "T:review/verify".into(),
                name: "verify".into(),
                entries: vec![2, 3],
            },
            Citable {
                path: "T:review".into(),
                name: "review".into(),
                entries: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            },
        ]);
        let reads = |s: &str| world.read(s, "exists");
        assert_eq!(reads("entry:T:review#12").as_deref(), Some("true"));
        assert_eq!(reads("entry:review#12").as_deref(), Some("true"));
        assert_eq!(reads("entry:verify#3.2").as_deref(), Some("true"));
        assert_eq!(reads("entry:T:review/verify#9").as_deref(), Some("false"));
        assert_eq!(
            reads("entry:T:other#1").as_deref(),
            Some("false"),
            "not an ancestor's"
        );
        assert_eq!(reads("entry:T:review"), None, "no entry named");
        assert_eq!(world.read("entry:T:review#1", "size"), None);
    }

    #[test]
    fn a_folder_not_under_git_has_no_branch() {
        let files = Arc::new(RepoFiles::detached());
        files
            .repo("scratch")
            .unwrap()
            .write("x.txt", "x".into())
            .unwrap();
        let world = ConversationFiles::of(files);
        assert_eq!(
            world.read("file:scratch/x.txt", "exists").as_deref(),
            Some("true")
        );
        assert_eq!(world.read("branch:scratch", "head"), None);
        assert_eq!(world.read("branch:scratch", "name"), None);
    }
}
