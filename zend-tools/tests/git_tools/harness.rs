//! A real git repository on disk for the `git_*` tool tests.
//!
//! These live outside `src/` deliberately. The guards in `crate::exec` and
//! `crate::disk` hold every line under `src/tools` to the capability-checked
//! primitives, and a fixture that builds a repository has to start `git` and
//! write files — so it belongs here, where it cannot be mistaken for tool
//! code, rather than weakening a guard that protects the production path.
//!
//! Setup drives the `git` program directly rather than going through the
//! layer under test, the same convention `zend_vfs`'s own tests follow, so a
//! fixture can never be shaped by the bug it is hunting. The identity and
//! dates are fixed, so commit ids are reproducible.

use std::io::Write;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use serde_json::{json, Value};
use tempfile::TempDir;
use zend_tools::registry::find;
use zend_tools::sandboxes::Sandboxes;
use zend_tools::{Grants, ToolContext};
use zend_vfs::{CommandPolicy, RepoSpec, Workspace};

/// A workspace holding one git repository called `app`.
pub struct GitWorkspace {
    pub dir: TempDir,
}

/// The fixed date every setup commit carries.
const SETUP_DATE: &str = "1700000000 +0000";

/// git in `dir` with the setup identity, dates and configuration.
fn setup_git(dir: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(dir)
        .args(["-c", "core.hooksPath=", "-c", "commit.gpgSign=false"])
        .args(args)
        .env("LC_ALL", "C")
        .env("GIT_AUTHOR_NAME", "Setup")
        .env("GIT_AUTHOR_EMAIL", "setup@example.com")
        .env("GIT_AUTHOR_DATE", SETUP_DATE)
        .env("GIT_COMMITTER_NAME", "Setup")
        .env("GIT_COMMITTER_EMAIL", "setup@example.com")
        .env("GIT_COMMITTER_DATE", SETUP_DATE)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE");
    cmd
}

/// stdout of a finished setup command; panics on failure.
fn setup_output(args: &[&str], out: std::process::Output) -> String {
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8")
}

/// Run git in `dir` for setup, returning stdout; panics on failure.
pub fn git_in(dir: &Path, args: &[&str]) -> String {
    setup_output(args, setup_git(dir, args).output().expect("git runs"))
}

/// [`git_in`], with `input` on git's stdin.
fn git_in_fed(dir: &Path, args: &[&str], input: &[u8]) -> String {
    let mut child = setup_git(dir, args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("git runs");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(input)
        .expect("git reads its input");
    setup_output(args, child.wait_with_output().expect("git finishes"))
}

impl GitWorkspace {
    /// A workspace whose `app` repository holds two commits on `main`: an
    /// initial one with `README.md` and `src/lib.rs`, and a second that
    /// rewrites one line of `src/lib.rs`.
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let app = dir.path().join("app");
        std::fs::create_dir_all(&app).unwrap();
        // `init -b` needs git 2.28; the layer supports 2.24.
        git_in(&app, &["init", "-q"]);
        git_in(&app, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        for (k, v) in [
            ("core.autocrlf", "false"),
            ("user.name", "Setup"),
            ("user.email", "setup@example.com"),
        ] {
            git_in(&app, &["config", k, v]);
        }
        let ws = Self { dir };
        ws.write_worktree("README.md", "# app\n\nthe app.\n");
        ws.write_worktree(
            "src/lib.rs",
            "pub fn hello() -> &'static str {\n    \"hi\"\n}\n",
        );
        ws.commit_all("initial commit");
        ws.write_worktree(
            "src/lib.rs",
            "pub fn hello() -> &'static str {\n    \"hello\"\n}\n",
        );
        ws.commit_all("say hello properly");
        ws
    }

    /// The repository's folder on disk.
    pub fn repo_dir(&self) -> PathBuf {
        self.dir.path().join("app")
    }

    pub fn git(&self, args: &[&str]) -> String {
        git_in(&self.repo_dir(), args)
    }

    /// Stage everything and commit it, returning the new commit's id.
    pub fn commit_all(&self, message: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "--allow-empty", "-m", message]);
        self.oid("HEAD")
    }

    /// One commit per step on `main`, each writing the step number to `path`
    /// with the message `step N` — what [`Self::write_worktree`] plus
    /// [`Self::commit_all`] per step would make, from one `git fast-import`
    /// rather than three processes a step. The working tree and index follow
    /// `main` to the last step. Returns the last commit's id.
    pub fn commit_steps(&self, path: &str, steps: RangeInclusive<usize>) -> String {
        let mut stream = String::new();
        let mut first = true;
        for n in steps {
            let message = format!("step {n}");
            let content = format!("{n}\n");
            stream.push_str("commit refs/heads/main\n");
            stream.push_str(&format!("author Setup <setup@example.com> {SETUP_DATE}\n"));
            stream.push_str(&format!(
                "committer Setup <setup@example.com> {SETUP_DATE}\n"
            ));
            stream.push_str(&format!("data {}\n{message}\n", message.len()));
            if first {
                stream.push_str(&format!("from {}\n", self.oid("main")));
                first = false;
            }
            stream.push_str(&format!("M 100644 inline {path}\n"));
            stream.push_str(&format!("data {}\n{content}\n", content.len()));
        }
        git_in_fed(
            &self.repo_dir(),
            &["fast-import", "--quiet"],
            stream.as_bytes(),
        );
        self.git(&["reset", "-q", "--hard", "main"]);
        self.oid("main")
    }

    /// The full object id `rev` resolves to.
    pub fn oid(&self, rev: &str) -> String {
        self.git(&["rev-parse", rev]).trim().to_string()
    }

    /// Write into the working tree without committing, so `git_status` and a
    /// worktree diff have something to report.
    pub fn write_worktree(&self, path: &str, content: &str) {
        let full = self.repo_dir().join(path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(full, content).unwrap();
    }

    /// A context over this workspace, under `grants`.
    pub fn ctx(&self, grants: Grants) -> ToolContext {
        let ws = Workspace::new(self.dir.path(), vec![RepoSpec::named("app")]).unwrap();
        ToolContext::with_workspace(ws).granting(grants)
    }

    /// A context that may run the writers — Comprehensive's grants.
    pub fn comprehensive_ctx(&self) -> ToolContext {
        self.ctx(Grants::ALL)
    }

    /// A context that may run only the readers — Restricted's grants, which
    /// hold nothing, `DiskWrite` included.
    pub fn restricted_ctx(&self) -> ToolContext {
        self.ctx(Grants::NONE)
    }

    /// One conversation over this workspace: a writable context kept across
    /// calls, so what it writes stays its own uncommitted work and the
    /// branch it switches to stays its branch.
    pub fn conversation(&self) -> Conversation {
        Conversation {
            ctx: self.comprehensive_ctx(),
        }
    }

    /// One conversation that can run `programs` in the workspace's sandboxes.
    pub fn conversation_running(&self, programs: &[&str]) -> Conversation {
        let ws = Workspace::new(self.dir.path(), vec![RepoSpec::named("app")]).unwrap();
        let sandboxes =
            Sandboxes::for_workspace(&ws, &CommandPolicy::allowing(programs.iter().copied()))
                .unwrap();
        Conversation {
            ctx: ToolContext::with_workspace(ws)
                .granting(Grants::ALL)
                .with_sandboxes(Arc::new(sandboxes)),
        }
    }

    /// Call a tool against this workspace's read-only context.
    pub fn read(&self, tool: &str, args: Value) -> Value {
        find(tool)
            .unwrap_or_else(|| panic!("{tool} is registered"))
            .call(&self.restricted_ctx(), &args)
    }

    /// Call a tool against this workspace's writable context.
    pub fn write(&self, tool: &str, args: Value) -> Value {
        find(tool)
            .unwrap_or_else(|| panic!("{tool} is registered"))
            .call(&self.comprehensive_ctx(), &args)
    }

    /// A bare repository beside the workspace acting as `origin`, reachable
    /// over `file://` so the real transport is exercised rather than git's
    /// local-copy shortcut.
    pub fn with_origin(&self) -> PathBuf {
        let origin = self.dir.path().join("origin.git");
        std::fs::create_dir_all(&origin).unwrap();
        git_in(&origin, &["init", "-q", "--bare"]);
        git_in(&origin, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        self.git(&["remote", "add", "origin", &file_url(&origin)]);
        origin
    }

    /// A second clone of `origin`, standing in for another developer.
    pub fn other_clone(&self, origin: &Path) -> PathBuf {
        let other = self.dir.path().join("other");
        git_in(
            self.dir.path(),
            &["clone", "-q", &file_url(origin), "other"],
        );
        git_in(&other, &["config", "user.name", "Other"]);
        git_in(&other, &["config", "user.email", "other@example.com"]);
        other
    }
}

/// One conversation: tool calls against one context, whose files are its own.
pub struct Conversation {
    pub ctx: ToolContext,
}

impl Conversation {
    pub fn call(&self, tool: &str, args: Value) -> Value {
        find(tool)
            .unwrap_or_else(|| panic!("{tool} is registered"))
            .call(&self.ctx, &args)
    }

    /// Write `path` in the conversation's files, uncommitted.
    pub fn write(&self, path: &str, content: &str) {
        self.ctx
            .files
            .repo("app")
            .unwrap()
            .write(path, content.to_string())
            .unwrap();
    }

    /// Delete `path` from the conversation's files, uncommitted.
    pub fn delete(&self, path: &str) {
        assert!(self.ctx.files.repo("app").unwrap().delete(path), "{path}");
    }

    /// `path` as the conversation reads it.
    pub fn read(&self, path: &str) -> Option<String> {
        self.ctx.files.repo("app").unwrap().read(path).unwrap()
    }

    pub fn status(&self) -> Value {
        self.call("git_status", json!({"repo": "app", "page": 0}))
    }
}

/// A `file://` URL for a local path, in the form git accepts on Windows too.
pub fn file_url(path: &Path) -> String {
    let p = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let s = p.to_string_lossy().replace('\\', "/");
    let s = s.trim_start_matches("//?/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

/// What a remote ref holds.
pub fn remote_oid(origin: &Path, r: &str) -> String {
    git_in(origin, &["rev-parse", r]).trim().to_string()
}

// ── Revision arguments ───────────────────────────────────────────────────────

pub fn branch_rev(branch: &str) -> Value {
    json!({"kind": "branch", "name": branch})
}

pub fn head_rev() -> Value {
    json!({"kind": "head"})
}

pub fn commit_rev(oid: &str) -> Value {
    json!({"kind": "commit", "name": oid})
}

pub fn tag_rev(tag: &str) -> Value {
    json!({"kind": "tag", "name": tag})
}

/// `back` commits before `from` — the arm that exists so "the previous
/// commit" never needs an object id the model does not hold.
pub fn parent_rev(from: &str, back: Option<u32>) -> Value {
    match back {
        Some(n) => json!({"kind": "parent", "name": from, "back": n}),
        None => json!({"kind": "parent", "name": from}),
    }
}
