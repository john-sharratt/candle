//! Building and running one git invocation.
//!
//! Every git process the layer starts is built here. The rules it applies to
//! all of them:
//!
//! - `-C <repository>` first, so nothing depends on the daemon's working
//!   directory.
//! - The user's own configuration is inherited — it carries `core.autocrlf`,
//!   `.gitattributes` handling and the SSH settings their pushes use — with
//!   the settings that are unsafe under a daemon overridden by `-c`: hooks,
//!   fsmonitor, automatic gc and maintenance, signing, pager and colour.
//! - The environment is the daemon's minus anything that would redirect git
//!   at another repository or inject configuration or identity, plus the `C`
//!   locale (so the classifier reads stable messages) and no terminal prompts.
//! - Arguments are separate values, never a shell string; bulk input goes
//!   through stdin.
//! - A timeout kills the whole process tree.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::classify::classify;
use crate::error::GitError;
use crate::kill_tree::{self, ProcessTree};
use crate::redact::redact_urls;
use crate::types::RemoteName;

/// Timeout for an operation that stays on this machine.
pub(crate) const LOCAL_TIMEOUT: Duration = Duration::from_secs(30);
/// Timeout for an operation that talks to a remote.
pub(crate) const NETWORK_TIMEOUT: Duration = Duration::from_secs(120);
/// Timeout for moving a repository's objects — a clone, a fetch, a push —
/// which takes as long as the repository is large. A transfer that stalls
/// over HTTP ends well before this, on git's own low-speed limit.
pub(crate) const TRANSFER_TIMEOUT: Duration = Duration::from_secs(60 * 60);
/// Bytes per second, sustained for [`STALL_SECONDS`], below which git
/// abandons an HTTP transfer as stalled.
const STALL_BYTES_PER_SECOND: &str = "1000";
const STALL_SECONDS: &str = "60";

/// Environment variables removed from every child: each would point git at
/// a different repository, index or object store, inject configuration,
/// re-enable a transport the config disables (`GIT_ALLOW_PROTOCOL` overrides
/// `protocol.ext.allow=never`), or supply an identity the caller did not
/// choose.
const SCRUBBED_ENV: &[&str] = &[
    "GIT_ALLOW_PROTOCOL",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_DATE",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "GIT_COMMITTER_DATE",
    "LANGUAGE",
];

/// `core.hooksPath` for every invocation: the null device, under which no
/// hook file can ever exist, so no hook in the user's repository runs under
/// the daemon. A real empty folder would have to live somewhere — in a
/// shared temp directory another user could create it first and put a
/// `pre-push` in it.
#[cfg(windows)]
const NO_HOOKS: &str = "NUL";
#[cfg(not(windows))]
const NO_HOOKS: &str = "/dev/null";

/// What an invocation was about, for turning its failure into a typed error.
#[derive(Debug, Clone, Default)]
pub(crate) struct Context {
    pub remote: Option<RemoteName>,
    pub rev: Option<String>,
}

/// A finished git process.
#[derive(Debug)]
pub(crate) struct Output {
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: String,
}

/// One git command, built up and then run.
pub(crate) struct Invocation {
    dir: PathBuf,
    config: Vec<(String, String)>,
    subcommand: String,
    args: Vec<OsString>,
    env: Vec<(String, OsString)>,
    stdin: Option<Vec<u8>>,
    timeout: Duration,
    context: Context,
}

impl Invocation {
    pub(crate) fn new(dir: &Path, subcommand: &str) -> Self {
        Self {
            dir: dir.to_path_buf(),
            config: Vec::new(),
            subcommand: subcommand.to_string(),
            args: Vec::new(),
            env: Vec::new(),
            stdin: None,
            timeout: LOCAL_TIMEOUT,
            context: Context::default(),
        }
    }

    pub(crate) fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub(crate) fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// A read that must not take optional locks — `status` otherwise takes
    /// `index.lock` and rewrites the user's index while they work.
    pub(crate) fn read_only(self) -> Self {
        self.env("GIT_OPTIONAL_LOCKS", "0")
    }

    /// An operation that talks to a remote.
    pub(crate) fn network(mut self, remote: &RemoteName) -> Self {
        self.timeout = NETWORK_TIMEOUT;
        self.context.remote = Some(remote.clone());
        self
    }

    /// An operation that moves objects to or from a remote: timed by
    /// [`TRANSFER_TIMEOUT`], with a stalled HTTP transfer abandoned by git.
    pub(crate) fn transfer(self, remote: &RemoteName) -> Self {
        let mut inv = self.network(remote);
        inv.timeout = TRANSFER_TIMEOUT;
        inv.config.extend([
            ("http.lowSpeedLimit".into(), STALL_BYTES_PER_SECOND.into()),
            ("http.lowSpeedTime".into(), STALL_SECONDS.into()),
        ]);
        inv
    }

    #[cfg(test)]
    pub(crate) fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The revision this invocation resolves, named in an unknown-revision
    /// error.
    pub(crate) fn about_rev(mut self, rev: impl Into<String>) -> Self {
        self.context.rev = Some(rev.into());
        self
    }

    /// A `-c` setting for this invocation only.
    #[cfg(test)]
    pub(crate) fn config(mut self, key: &str, value: &str) -> Self {
        self.config.push((key.to_string(), value.to_string()));
        self
    }

    pub(crate) fn env(mut self, key: &str, value: impl Into<OsString>) -> Self {
        self.env.push((key.to_string(), value.into()));
        self
    }

    pub(crate) fn stdin(mut self, bytes: Vec<u8>) -> Self {
        self.stdin = Some(bytes);
        self
    }

    /// The argument list, for error messages, with any credential in a URL
    /// argument (a clone's source) redacted.
    fn arg_strings(&self) -> Vec<String> {
        std::iter::once(self.subcommand.clone())
            .chain(self.args.iter().map(|a| redact_urls(&a.to_string_lossy())))
            .collect()
    }

    /// The configured command, not yet started.
    pub(crate) fn command(&self) -> Result<Command, GitError> {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(&self.dir);
        let mut overrides: Vec<(String, String)> = vec![
            ("core.hooksPath".into(), NO_HOOKS.into()),
            // Empty, not `false`: before 2.36 this setting named a hook to
            // run, and `false` would run a program by that name. Empty means
            // off in every version.
            ("core.fsmonitor".into(), String::new()),
            ("core.quotePath".into(), "false".into()),
            ("core.pager".into(), "cat".into()),
            ("color.ui".into(), "false".into()),
            ("gc.auto".into(), "0".into()),
            ("maintenance.auto".into(), "false".into()),
            ("commit.gpgSign".into(), "false".into()),
            ("tag.gpgSign".into(), "false".into()),
            // Transports that run commands instead of fetching.
            ("protocol.ext.allow".into(), "never".into()),
            ("protocol.fd.allow".into(), "never".into()),
            // Submodules are never recursed into: every clone-time code
            // execution fix of recent years (CVE-2024-32002, CVE-2025-48384)
            // needs a recursive clone to reach.
            ("submodule.recurse".into(), "false".into()),
            // Bundle URIs let a server name further downloads; one was the
            // path for CVE-2025-48385.
            ("transfer.bundleURI".into(), "false".into()),
            ("advice.detachedHead".into(), "false".into()),
            ("advice.pushUpdateRejected".into(), "false".into()),
            ("advice.pushNonFFCurrent".into(), "false".into()),
            ("advice.pushFetchFirst".into(), "false".into()),
        ];
        overrides.extend(self.config.iter().cloned());
        for (key, value) in overrides {
            cmd.arg("-c").arg(format!("{key}={value}"));
        }
        cmd.arg(&self.subcommand);
        cmd.args(&self.args);
        for key in SCRUBBED_ENV {
            cmd.env_remove(key);
        }
        // Literal pathspecs: a path such as `:(glob)*` names that file, never
        // pathspec magic. No replace refs: a read must show the objects a push
        // would send, not a local substitute for them.
        cmd.env("LC_ALL", "C")
            .env("LANG", "C")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_LITERAL_PATHSPECS", "1")
            .env("GIT_NO_REPLACE_OBJECTS", "1");
        for (key, value) in &self.env {
            cmd.env(key, value);
        }
        kill_tree::prepare(&mut cmd);
        Ok(cmd)
    }

    /// Run to completion. An error here means git could not be started or
    /// timed out; a non-zero exit is returned as an [`Output`].
    pub(crate) fn run(mut self) -> Result<Output, GitError> {
        let args = self.arg_strings();
        let mut cmd = self.command()?;
        cmd.stdin(if self.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GitError::GitMissing(e)
            } else {
                GitError::Io(e)
            }
        })?;
        let tree = match ProcessTree::adopt(&child) {
            Ok(tree) => tree,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(GitError::Io(e));
            }
        };

        let writer = match (self.stdin.take(), child.stdin.take()) {
            (Some(bytes), Some(mut pipe)) => Some(thread::spawn(move || {
                // A child that exits without reading all its input closes the
                // pipe; its exit status reports why.
                let _ = pipe.write_all(&bytes);
            })),
            _ => None,
        };
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let mut stderr = child.stderr.take().expect("stderr is piped");
        let out_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        let err_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        // Blocked on the child's exit rather than polled for it: most git
        // invocations take tens of milliseconds, and a poll that backs off
        // notices each one late — across the hundreds of calls one operation
        // makes, that wait was most of its time.
        let (exited, exit) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let status = child.wait();
            let _ = exited.send(());
            status
        });
        let timed_out = exit.recv_timeout(self.timeout).is_err();
        // Whatever the child left behind — a finished hook's shell, an `ssh`
        // still holding the pipes — goes with it, so the readers see EOF. On a
        // timeout this kills the child itself, which ends the waiter's `wait`.
        tree.kill();
        let status = waiter
            .join()
            .map_err(|_| GitError::Io(std::io::Error::other("the git waiter thread panicked")))??;
        if timed_out {
            let _ = out_reader.join();
            let _ = err_reader.join();
            return Err(GitError::Timeout {
                args,
                after: self.timeout,
            });
        }
        if let Some(w) = writer {
            let _ = w.join();
        }
        let stdout = out_reader.join().unwrap_or_default();
        let stderr = String::from_utf8_lossy(&err_reader.join().unwrap_or_default()).into_owned();
        Ok(Output {
            status: status.code(),
            stdout,
            stderr,
        })
    }

    /// Run, and turn any non-zero exit into a typed error.
    pub(crate) fn run_ok(self) -> Result<Vec<u8>, GitError> {
        Ok(self.run_accepting(&[0])?.stdout)
    }

    /// Run, accepting the listed exit codes and classifying any other.
    pub(crate) fn run_accepting(self, ok: &[i32]) -> Result<Output, GitError> {
        let dir = self.dir.clone();
        let context = self.context.clone();
        let args = self.arg_strings();
        let output = self.run()?;
        match output.status {
            Some(code) if ok.contains(&code) => Ok(output),
            status => Err(classify(&dir, &context, args, status, output.stderr)),
        }
    }
}

/// Bytes git printed, as UTF-8, or a malformed-output error naming `command`.
pub(crate) fn utf8(command: &'static str, bytes: Vec<u8>) -> Result<String, GitError> {
    String::from_utf8(bytes).map_err(|e| GitError::malformed(command, e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::time::Instant;

    use super::*;
    use crate::testing::TestRepo;
    use crate::types::{RepoPath, Rev};

    /// **A timeout kills the whole tree.** The alias runs a shell that
    /// sleeps and then writes a marker: git → sh → sleep. Killing only `git`
    /// would leave the shell to write the marker after the timeout.
    #[test]
    fn a_timeout_kills_the_process_tree() {
        let t = TestRepo::init();
        let started = Instant::now();
        let result = Invocation::new(&t.path, "slow")
            .config("alias.slow", "!sleep 1 && echo late > marker.txt")
            .timeout(Duration::from_millis(200))
            .run();
        assert!(
            matches!(result, Err(GitError::Timeout { .. })),
            "{result:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "returned at the timeout"
        );
        // Past the moment the grandchild would have written.
        thread::sleep(Duration::from_millis(1300));
        assert!(
            !t.path.join("marker.txt").exists(),
            "a grandchild outlived the timeout"
        );
    }

    #[test]
    fn stdout_stderr_and_exit_status_are_captured() {
        let t = TestRepo::init();
        let out = Invocation::new(&t.path, "hash-object")
            .arg("--stdin")
            .stdin(b"hello\n".to_vec())
            .run()
            .unwrap();
        assert_eq!(out.status, Some(0));
        assert_eq!(out.stdout, b"ce013625030ba8dba906f756967f9e9ca394464a\n");

        let out = Invocation::new(&t.path, "rev-parse")
            .args(["--verify", "no-such-thing"])
            .run()
            .unwrap();
        assert_eq!(out.status, Some(128));
        assert!(out.stderr.starts_with("fatal: "), "{:?}", out.stderr);
    }

    /// Variables that would aim git at another repository, inject
    /// configuration or pick an identity are removed from every child; the
    /// locale and prompt settings are pinned.
    #[test]
    fn the_child_environment_is_scrubbed_and_pinned() {
        let t = TestRepo::init();
        let cmd = Invocation::new(&t.path, "status").command().unwrap();
        let envs: Vec<(&OsStr, Option<&OsStr>)> = cmd.get_envs().collect();
        let value = |key: &str| {
            envs.iter()
                .find(|(k, _)| *k == OsStr::new(key))
                .map(|(_, v)| *v)
        };
        for key in SCRUBBED_ENV {
            assert_eq!(value(key), Some(None), "{key} must be removed");
        }
        for (key, expected) in [
            ("LC_ALL", "C"),
            ("LANG", "C"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GIT_LITERAL_PATHSPECS", "1"),
            ("GIT_NO_REPLACE_OBJECTS", "1"),
        ] {
            assert_eq!(value(key), Some(Some(OsStr::new(expected))), "{key}");
        }
        assert!(SCRUBBED_ENV.contains(&"GIT_ALLOW_PROTOCOL"));
    }

    /// **A replace ref never changes what a read shows.** `git replace`
    /// makes git show one object in place of another; a read must show the
    /// object a push would actually send.
    #[test]
    fn replace_refs_are_ignored() {
        let t = TestRepo::init();
        t.write("f.txt", b"real\n");
        t.commit_all("base");
        let real = t.git(&["rev-parse", "HEAD:f.txt"]);
        std::fs::write(t.path.join("fake.txt"), b"substitute\n").unwrap();
        let fake = t.git(&["hash-object", "-w", "fake.txt"]);
        t.git(&["replace", real.trim(), fake.trim()]);
        assert_eq!(
            t.git(&["cat-file", "blob", real.trim()]),
            "substitute\n",
            "plain git honours the replacement"
        );
        let got = t
            .repo()
            .blobs()
            .read_at(&Rev::Head, &RepoPath::parse("f.txt").unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(got, b"real\n");
    }

    /// **A transfer is timed for a large repository**, with git told to
    /// abandon a stalled HTTP transfer rather than wait out the hour.
    #[test]
    fn a_transfer_gets_the_long_timeout_and_a_stall_limit() {
        let t = TestRepo::init();
        let origin = RemoteName::parse("origin").unwrap();
        let inv = Invocation::new(&t.path, "fetch").transfer(&origin);
        assert_eq!(inv.timeout, TRANSFER_TIMEOUT);
        assert_eq!(inv.context.remote.as_ref(), Some(&origin));
        let args: Vec<String> = inv
            .command()
            .unwrap()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        for setting in ["http.lowSpeedLimit=1000", "http.lowSpeedTime=60"] {
            assert!(args.iter().any(|a| a == setting), "{setting} in {args:?}");
        }
        assert_eq!(
            Invocation::new(&t.path, "ls-remote")
                .network(&origin)
                .timeout,
            NETWORK_TIMEOUT
        );
    }

    /// Hooks point at the null device, where no hook file can exist.
    #[test]
    fn hooks_point_at_the_null_device() {
        let t = TestRepo::init();
        let cmd = Invocation::new(&t.path, "status").command().unwrap();
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(
            args.contains(&format!("core.hooksPath={NO_HOOKS}")),
            "{args:?}"
        );
    }

    /// Every unsafe setting is overridden on the command line, before the
    /// subcommand.
    #[test]
    fn unsafe_settings_are_overridden_before_the_subcommand() {
        let t = TestRepo::init();
        let cmd = Invocation::new(&t.path, "status").command().unwrap();
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let sub = args.iter().position(|a| a == "status").unwrap();
        for setting in [
            "core.hooksPath=",
            "core.fsmonitor=",
            "submodule.recurse=false",
            "transfer.bundleURI=false",
            "protocol.ext.allow=never",
            "gc.auto=0",
            "maintenance.auto=false",
            "commit.gpgSign=false",
        ] {
            let at = args
                .iter()
                .position(|a| a.starts_with(setting))
                .unwrap_or_else(|| panic!("{setting} missing from {args:?}"));
            assert!(at < sub, "{setting} must precede the subcommand");
            assert_eq!(args[at - 1], "-c");
        }
        assert_eq!(
            &args[..2],
            &["-C".to_string(), t.path.to_string_lossy().into_owned()]
        );
    }
}
