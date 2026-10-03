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
use std::sync::{Arc, OnceLock};

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
        // A commit otherwise ends by starting `git maintenance` in another
        // process, which a fixture that lives for one test has no use for.
        .args(["-c", "maintenance.auto=false", "-c", "gc.auto=0"])
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

/// Write `content` at `rel` under `root`, creating its folders.
fn write_file(root: &Path, rel: &str, content: &str) {
    let full = root.join(rel);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(full, content).unwrap();
}

/// Stage everything in `repo` and commit it.
///
/// In this process — `git add -A` and `git commit` are two `git` processes, and
/// a run makes hundreds of commits — unless a merge, cherry-pick or revert is
/// under way, which `git commit` finishes and libgit2's `commit` does not. The
/// commit is the one the `git` programs would make: the same tree, the setup
/// identity at the setup date, and the message with the newline `git commit`
/// ends it with.
fn commit_everything(repo: &Path, message: &str) {
    if commit_in_process(repo, message).is_none() {
        git_in(repo, &["add", "-A"]);
        git_in(repo, &["commit", "-q", "--allow-empty", "-m", message]);
    }
}

fn commit_in_process(repo_dir: &Path, message: &str) -> Option<()> {
    let git_dir = repo_dir.join(".git");
    let mid_operation = ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD"]
        .iter()
        .any(|marker| git_dir.join(marker).exists());
    if mid_operation {
        return None;
    }
    let repo = git2::Repository::open(repo_dir).ok()?;
    let mut index = repo.index().ok()?;
    index
        .add_all(["*"], git2::IndexAddOption::DEFAULT, None)
        .ok()?;
    index.update_all(["*"], None).ok()?;
    index.write().ok()?;
    let tree = repo.find_tree(index.write_tree().ok()?).ok()?;
    let setup = git2::Signature::new(
        "Setup",
        "setup@example.com",
        &git2::Time::new(1_700_000_000, 0),
    )
    .ok()?;
    let parent = repo.head().ok().and_then(|head| head.peel_to_commit().ok());
    let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
    repo.commit(
        Some("HEAD"),
        &setup,
        &setup,
        &format!("{message}\n"),
        &tree,
        &parents,
    )
    .ok()
    .map(|_| ())
}

/// Build the repository every [`GitWorkspace::new`] starts from in `app`: two
/// commits on `main`, an initial one with `README.md` and `src/lib.rs`, and a
/// second that rewrites one line of `src/lib.rs`.
fn build_app(app: &Path) {
    std::fs::create_dir_all(app).unwrap();
    // `init -b` needs git 2.28; the layer supports 2.24.
    git_in(app, &["init", "-q"]);
    git_in(app, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    for (k, v) in [
        ("core.autocrlf", "false"),
        ("user.name", "Setup"),
        ("user.email", "setup@example.com"),
    ] {
        git_in(app, &["config", k, v]);
    }
    write_file(app, "README.md", "# app\n\nthe app.\n");
    write_file(
        app,
        "src/lib.rs",
        "pub fn hello() -> &'static str {\n    \"hi\"\n}\n",
    );
    commit_everything(app, "initial commit");
    write_file(
        app,
        "src/lib.rs",
        "pub fn hello() -> &'static str {\n    \"hello\"\n}\n",
    );
    commit_everything(app, "say hello properly");
}

/// The repository [`build_app`] makes, built once per test process — under
/// Cargo's temporary folder for this binary, replaced at each run — and copied
/// for each workspace. Building it takes eleven git processes; a test's own
/// copy takes none, and is still its alone. The identity and dates are fixed,
/// so a copy's commit ids are the ones a fresh build gives.
fn app_template() -> &'static Path {
    static TEMPLATE: OnceLock<PathBuf> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fixture-git_tools-app");
        let _ = std::fs::remove_dir_all(&root);
        let app = root.join("app");
        build_app(&app);
        app
    })
}

/// An empty bare repository on `main`, the `origin` of a test that has one:
/// built once per process like [`app_template`] and copied. Its config says what
/// a fixture's origin never wants — maintenance started after every push it
/// receives, in a process of its own.
fn origin_template() -> &'static Path {
    static TEMPLATE: OnceLock<PathBuf> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fixture-git_tools-origin");
        let _ = std::fs::remove_dir_all(&root);
        let origin = root.join("origin.git");
        std::fs::create_dir_all(&origin).unwrap();
        git_in(&origin, &["init", "-q", "--bare"]);
        git_in(&origin, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        append_config(&origin, "[maintenance]\n\tauto = false\n[gc]\n\tauto = 0\n");
        origin
    })
}

/// Add `text` — whole `[section]` blocks — to the end of the repository's
/// config, which is what `git config` or `git remote add` would have written,
/// without the process. A bare repository keeps its config at its top level,
/// a working tree's inside `.git`.
fn append_config(repo: &Path, text: &str) {
    let path = if repo.join(".git").is_dir() {
        repo.join(".git").join("config")
    } else {
        repo.join("config")
    };
    let mut file = std::fs::OpenOptions::new().append(true).open(path).unwrap();
    file.write_all(text.as_bytes()).unwrap();
}

/// Copy the folder `from` to `to`, everything in it included.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

impl GitWorkspace {
    /// A workspace whose `app` repository holds two commits on `main`: an
    /// initial one with `README.md` and `src/lib.rs`, and a second that
    /// rewrites one line of `src/lib.rs`.
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        copy_tree(app_template(), &dir.path().join("app"));
        Self { dir }
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
        commit_everything(&self.repo_dir(), message);
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
        if rev == "HEAD" {
            if let Some(oid) = self.head_from_files() {
                return oid;
            }
        }
        self.git(&["rev-parse", rev]).trim().to_string()
    }

    /// The commit `HEAD` names, read from the files: the branch ref a commit
    /// has just written, or a detached id. A fixture asks this after every
    /// commit, and a process per answer is most of what a commit costs. `None`
    /// when the answer is not in a file — a packed ref, an unborn branch — and
    /// git is asked instead.
    fn head_from_files(&self) -> Option<String> {
        let git_dir = self.repo_dir().join(".git");
        let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
        let head = head.trim();
        let target = match head.strip_prefix("ref: ") {
            Some(name) => std::fs::read_to_string(git_dir.join(name)).ok()?,
            None => head.to_string(),
        };
        let target = target.trim();
        (target.len() == 40 && target.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| target.to_string())
    }

    /// Write into the working tree without committing, so `git_status` and a
    /// worktree diff have something to report.
    pub fn write_worktree(&self, path: &str, content: &str) {
        write_file(&self.repo_dir(), path, content);
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
        copy_tree(origin_template(), &origin);
        // What `git remote add origin <url>` writes.
        append_config(
            &self.repo_dir(),
            &format!(
                "[remote \"origin\"]\n\turl = {}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n",
                file_url(&origin)
            ),
        );
        origin
    }

    /// A second clone of `origin`, standing in for another developer.
    pub fn other_clone(&self, origin: &Path) -> PathBuf {
        let other = self.dir.path().join("other");
        git_in(
            self.dir.path(),
            &["clone", "-q", &file_url(origin), "other"],
        );
        append_config(
            &other,
            "[user]\n\tname = Other\n\temail = other@example.com\n",
        );
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
