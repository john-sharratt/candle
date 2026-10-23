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
//!
//! Anything else reads as no answer, which the check treats as "the workspace
//! cannot say" rather than as a disagreement: the claim stays, as prose. So
//! does a file the session has changed, whose bytes are no blob yet.

use std::sync::Arc;

use journal_core::verify::World;
use zend_vfs::{RepoFiles, Rev, VfsStore};

/// The subject prefix of a file.
pub const FILE: &str = "file:";
/// The subject prefix of a repository's branch.
pub const BRANCH: &str = "branch:";

/// One conversation's view of the workspace.
pub struct ConversationFiles {
    files: Arc<RepoFiles>,
}

impl ConversationFiles {
    pub fn of(files: Arc<RepoFiles>) -> Self {
        Self { files }
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
