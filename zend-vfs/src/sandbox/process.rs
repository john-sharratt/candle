//! Starting a sandboxed command and reading what it prints.
//!
//! The command runs in the repository's folder with nothing on its standard
//! input, and its output is read while it runs — both streams at once, so a
//! program that fills one pipe while the other is being waited on cannot
//! stall. Each stream keeps its first [`MAX_STREAM_BYTES`] and counts the
//! rest, so a runaway log cannot exhaust the daemon's memory.
//!
//! The command and everything it starts are one process tree
//! ([`ProcessTree`]). The tree is killed when the command exits — a build
//! server or a watcher it left behind must not go on changing the checkout
//! after the run has captured it and handed it to the next conversation — and
//! when it outlives its timeout.

use std::io;
use std::path::Path;
use std::process::{Command, Stdio};

use tokio::io::{AsyncRead, AsyncReadExt};

use super::command::SandboxCommand;
use super::outcome::Stream;
use crate::kill_tree::{self, ProcessTree};

/// The most of each output stream a run keeps.
pub const MAX_STREAM_BYTES: usize = 1024 * 1024;

/// What running a command produced.
#[derive(Debug)]
pub(crate) struct Executed {
    /// The exit code; `None` when the command was killed or ended by a signal.
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout: Stream,
    pub stderr: Stream,
}

/// Run `command` in `dir` to completion, or until its timeout.
pub(crate) async fn execute(dir: &Path, command: &SandboxCommand) -> io::Result<Executed> {
    let program = if command.is_repository_program() {
        let rel = command
            .program
            .strip_prefix("./")
            .unwrap_or(&command.program);
        dir.join(rel).into_os_string()
    } else {
        command.program.clone().into()
    };
    let mut std_command = Command::new(program);
    std_command
        .args(&command.args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // A git the program runs works on the repository it is in, not on
        // one the daemon's environment happens to name.
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE");
    kill_tree::prepare(&mut std_command);
    let mut child = tokio::process::Command::from(std_command)
        .kill_on_drop(true)
        .spawn()?;
    let tree = match ProcessTree::adopt_async(&child) {
        Ok(tree) => tree,
        Err(e) => {
            let _ = child.kill().await;
            return Err(e);
        }
    };
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");

    let waited = async {
        let finished = tokio::time::timeout(command.timeout, child.wait()).await;
        // Whatever the command left running goes with it — or, on a timeout,
        // the command itself.
        tree.kill();
        match finished {
            Ok(status) => status.map(|s| (s.code(), false)),
            Err(_) => child.wait().await.map(|_| (None, true)),
        }
    };
    let (status, stdout, stderr) = tokio::join!(waited, read_capped(stdout), read_capped(stderr));
    let (exit_code, timed_out) = status?;
    Ok(Executed {
        exit_code,
        timed_out,
        stdout: stdout?,
        stderr: stderr?,
    })
}

/// Read `reader` to its end, keeping the first [`MAX_STREAM_BYTES`].
async fn read_capped(mut reader: impl AsyncRead + Unpin) -> io::Result<Stream> {
    let mut kept = Vec::new();
    let mut total: u64 = 0;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        total += n as u64;
        let room = MAX_STREAM_BYTES.saturating_sub(kept.len());
        kept.extend_from_slice(&buf[..n.min(room)]);
    }
    Ok(Stream {
        text: String::from_utf8_lossy(&kept).into_owned(),
        bytes: total,
        truncated: total > kept.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// A command that runs `script` in the platform's shell.
    fn shell(script: &str) -> SandboxCommand {
        if cfg!(windows) {
            SandboxCommand::new("cmd").args(["/D", "/C", script])
        } else {
            SandboxCommand::new("sh").args(["-c", script])
        }
    }

    /// **Both streams and the exit code come back**, and the command runs in
    /// the folder it was given.
    #[tokio::test]
    async fn output_and_exit_code_are_captured() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("here.txt"), b"in the folder").unwrap();
        let script = if cfg!(windows) {
            "type here.txt & echo to-stderr 1>&2 & exit 3"
        } else {
            "cat here.txt; echo to-stderr 1>&2; exit 3"
        };
        let done = execute(dir.path(), &shell(script)).await.unwrap();
        assert_eq!(done.exit_code, Some(3));
        assert!(!done.timed_out);
        assert!(
            done.stdout.text.starts_with("in the folder"),
            "{:?}",
            done.stdout
        );
        assert_eq!(done.stderr.text.trim_end(), "to-stderr");
        assert!(!done.stdout.truncated);
    }

    /// **A command that outlives its timeout is killed** — promptly, not when
    /// it would have finished — and reported as timed out.
    #[tokio::test]
    async fn a_command_past_its_timeout_is_killed() {
        let dir = tempfile::tempdir().unwrap();
        let script = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };
        let started = Instant::now();
        let done = execute(
            dir.path(),
            &shell(script).timeout(Duration::from_millis(500)),
        )
        .await
        .unwrap();
        assert!(done.timed_out);
        assert_eq!(done.exit_code, None);
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "{:?}",
            started.elapsed()
        );
    }

    /// **Output past the cap is counted, not kept.**
    #[tokio::test]
    async fn output_past_the_cap_is_counted_not_kept() {
        let big = vec![b'x'; MAX_STREAM_BYTES + 10];
        let stream = read_capped(&big[..]).await.unwrap();
        assert_eq!(stream.text.len(), MAX_STREAM_BYTES);
        assert_eq!(stream.bytes, (MAX_STREAM_BYTES + 10) as u64);
        assert!(stream.truncated);
        let small = read_capped(&b"abc"[..]).await.unwrap();
        assert_eq!(
            (small.text.as_str(), small.bytes, small.truncated),
            ("abc", 3, false)
        );
    }

    /// A program that does not exist is an error, not an outcome.
    #[tokio::test]
    async fn a_missing_program_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let e = execute(dir.path(), &SandboxCommand::new("no-such-program-zend-vfs"))
            .await
            .unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
    }
}
