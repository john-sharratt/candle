//! Starting a sandboxed command and passing on what it prints.
//!
//! The command runs in the repository's folder with nothing on its standard
//! input. Both of its output streams are read while it runs — at once, so a
//! program that fills one pipe while the other is being waited on cannot stall
//! — and written to one sink the caller gives, each chunk as it arrives: the
//! sink holds what the command printed in the order the two streams delivered
//! it. Past [`MAX_OUTPUT_BYTES`] the rest is read and counted but not written,
//! so a runaway log cannot fill the disk; a sink that fails to take a write is
//! treated the same way, and the command runs on.
//!
//! The command and everything it starts are one process tree
//! ([`ProcessTree`]). The tree is killed when the command exits — a build
//! server or a watcher it left behind must not go on changing the checkout
//! after the run has captured it and handed it to the next conversation —
//! when it outlives its timeout, and when the run is abandoned part way.

use std::io;
use std::path::Path;
use std::process::{Command, Stdio};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::command::SandboxCommand;
use super::outcome::Output;
use crate::kill_tree::{self, ProcessTree};

/// The most of a command's output a run writes to its sink.
pub const MAX_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;

/// What running a command produced.
#[derive(Debug)]
pub(crate) struct Executed {
    /// The exit code; `None` when the command was killed or ended by a signal.
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub output: Output,
}

/// Run `command` in `dir` to completion, or until its timeout, writing what
/// it prints to `sink`.
pub(crate) async fn execute(
    dir: &Path,
    command: &SandboxCommand,
    sink: &mut (dyn AsyncWrite + Unpin + Send),
) -> io::Result<Executed> {
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
    let (status, output) = tokio::join!(waited, pump(stdout, stderr, sink, MAX_OUTPUT_BYTES));
    let (exit_code, timed_out) = status?;
    Ok(Executed {
        exit_code,
        timed_out,
        output: output?,
    })
}

/// Read both streams to their ends, writing each chunk to `sink` as it comes,
/// up to `cap` bytes in all.
async fn pump(
    mut stdout: impl AsyncRead + Unpin,
    mut stderr: impl AsyncRead + Unpin,
    sink: &mut (dyn AsyncWrite + Unpin + Send),
    cap: u64,
) -> io::Result<Output> {
    let mut out_buf = vec![0u8; 64 * 1024];
    let mut err_buf = vec![0u8; 64 * 1024];
    let (mut out_open, mut err_open) = (true, true);
    let mut bytes: u64 = 0;
    let mut written: u64 = 0;
    let mut sink_open = true;
    while out_open || err_open {
        let (n, from_out) = tokio::select! {
            read = stdout.read(&mut out_buf), if out_open => (read?, true),
            read = stderr.read(&mut err_buf), if err_open => (read?, false),
        };
        if n == 0 {
            if from_out {
                out_open = false;
            } else {
                err_open = false;
            }
            continue;
        }
        bytes += n as u64;
        let room = cap.saturating_sub(written).min(n as u64) as usize;
        if room == 0 || !sink_open {
            continue;
        }
        let chunk = if from_out {
            &out_buf[..room]
        } else {
            &err_buf[..room]
        };
        match sink.write_all(chunk).await {
            Ok(()) => written += room as u64,
            Err(_) => sink_open = false,
        }
    }
    if sink_open && sink.flush().await.is_err() {
        sink_open = false;
    }
    Ok(Output {
        bytes,
        truncated: !sink_open || written < bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context, Poll};
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

    /// **Both streams reach the sink and the exit code comes back**, and the
    /// command runs in the folder it was given.
    #[tokio::test]
    async fn output_and_exit_code_come_back() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("here.txt"), b"in the folder").unwrap();
        let script = if cfg!(windows) {
            "type here.txt & echo to-stderr 1>&2 & exit 3"
        } else {
            "cat here.txt; echo to-stderr 1>&2; exit 3"
        };
        let mut sink = Vec::new();
        let done = execute(dir.path(), &shell(script), &mut sink)
            .await
            .unwrap();
        assert_eq!(done.exit_code, Some(3));
        assert!(!done.timed_out);
        let text = String::from_utf8(sink).unwrap();
        assert!(text.contains("in the folder"), "{text:?}");
        assert!(text.contains("to-stderr"), "{text:?}");
        assert_eq!(done.output.bytes, text.len() as u64);
        assert!(!done.output.truncated);
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
            &mut Vec::new(),
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

    /// **Output past the cap is counted, not written**, from either stream.
    #[tokio::test]
    async fn output_past_the_cap_is_counted_not_written() {
        let mut sink = Vec::new();
        let out = pump(&b"abcdef"[..], &b"ghij"[..], &mut sink, 8)
            .await
            .unwrap();
        assert_eq!(out.bytes, 10);
        assert!(out.truncated);
        assert_eq!(sink.len(), 8);

        let mut sink = Vec::new();
        let out = pump(&b"abc"[..], &b""[..], &mut sink, 8).await.unwrap();
        assert_eq!(
            (sink.as_slice(), out.bytes, out.truncated),
            (&b"abc"[..], 3, false)
        );
    }

    /// A sink that refuses writes.
    struct Broken;

    impl AsyncWrite for Broken {
        fn poll_write(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            _: &[u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(io::Error::other("disk full")))
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// **A sink that fails is given up on, and the command still runs to its
    /// end** — its output read and counted, so it never blocks on a full pipe.
    #[tokio::test]
    async fn a_failing_sink_does_not_stop_the_command() {
        let out = pump(&b"abcdef"[..], &b"gh"[..], &mut Broken, 64)
            .await
            .unwrap();
        assert_eq!(out.bytes, 8);
        assert!(out.truncated);
    }

    /// A program that does not exist is an error, not an outcome.
    #[tokio::test]
    async fn a_missing_program_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let e = execute(
            dir.path(),
            &SandboxCommand::new("no-such-program-zend-vfs"),
            &mut Vec::new(),
        )
        .await
        .unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
    }
}
