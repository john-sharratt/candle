//! File contents at any revision, from one long-running `cat-file --batch`.
//!
//! Starting a process per read costs milliseconds on Windows, so a reader
//! keeps one `git cat-file --batch` per repository and feeds it requests
//! over stdin, one per line. A request is a revision spec and a
//! [`RepoPath`], and neither can hold a newline, so the line cannot be split. Reads are serialised by the reader's own lock; the child is
//! restarted if it exits, and killed with its process tree when the reader
//! is dropped.
//!
//! Two bounds keep one read from taking the daemon down with it:
//!
//! - **Size.** An object larger than [`MAX_BLOB_BYTES`] is refused from its
//!   header, before any buffer is allocated for it; a multi-gigabyte blob
//!   would otherwise be a multi-gigabyte allocation.
//! - **Time.** Each read has a deadline. In a partial clone a read may fetch
//!   from the network, and a fetch that stalls would otherwise hold the
//!   reader's lock forever; a watchdog kills the child's process tree when
//!   the deadline passes, and the read returns [`GitError::Timeout`].

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::error::GitError;
use crate::kill_tree::ProcessTree;
use crate::runner::{Invocation, NETWORK_TIMEOUT};
use crate::types::{Oid, RepoPath, Rev};
use crate::Repo;

/// The largest object a reader returns.
pub const MAX_BLOB_BYTES: u64 = 64 * 1024 * 1024;

struct Batch {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    tree: Arc<ProcessTree>,
}

impl Drop for Batch {
    fn drop(&mut self) {
        self.tree.kill();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct BlobReader {
    dir: PathBuf,
    batch: Mutex<Option<Batch>>,
    args: &'static [&'static str],
    deadline: Duration,
    max_bytes: u64,
}

/// What one request's header line says.
#[derive(Debug, PartialEq, Eq)]
enum Header {
    Missing,
    Found { kind: String, size: u64 },
}

fn parse_header(line: &[u8], request: &str) -> Result<Header, GitError> {
    let line = std::str::from_utf8(line)
        .map_err(|e| GitError::malformed("cat-file", e.to_string()))?
        .trim_end_matches('\n');
    if line == format!("{request} missing") || line == format!("{request} ambiguous") {
        return Ok(Header::Missing);
    }
    let parts: Vec<&str> = line.split(' ').collect();
    let [oid, kind, size] = parts[..] else {
        return Err(GitError::malformed("cat-file", line.to_string()));
    };
    Oid::parse(oid)?;
    let size = size
        .parse()
        .map_err(|_| GitError::malformed("cat-file", line.to_string()))?;
    Ok(Header::Found {
        kind: kind.to_string(),
        size,
    })
}

/// A request's answer.
enum Reply {
    Missing,
    Found {
        kind: String,
        body: Vec<u8>,
    },
    /// Refused from its header; the child is now out of step.
    TooLarge {
        size: u64,
    },
}

impl BlobReader {
    fn start(&self) -> Result<Batch, GitError> {
        let mut cmd = Invocation::new(&self.dir, "cat-file")
            .args(self.args.iter().copied())
            .command()?;
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = cmd.spawn()?;
        let tree = Arc::new(ProcessTree::adopt(&child)?);
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout is piped"));
        Ok(Batch {
            child,
            stdin,
            stdout,
            tree,
        })
    }

    fn request(batch: &mut Batch, request: &str, max_bytes: u64) -> Result<Reply, std::io::Error> {
        batch.stdin.write_all(request.as_bytes())?;
        batch.stdin.write_all(b"\n")?;
        batch.stdin.flush()?;
        let mut line = Vec::new();
        if batch.stdout.read_until(b'\n', &mut line)? == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        match parse_header(&line, request).map_err(std::io::Error::other)? {
            Header::Missing => Ok(Reply::Missing),
            Header::Found { size, .. } if size > max_bytes => Ok(Reply::TooLarge { size }),
            Header::Found { kind, size } => {
                let mut body = vec![0; size as usize + 1];
                batch.stdout.read_exact(&mut body)?;
                body.pop(); // the newline after the contents
                Ok(Reply::Found { kind, body })
            }
        }
    }

    /// [`Self::request`] under the deadline: a watchdog kills the child's
    /// tree if the answer has not come by then, which unblocks the read.
    fn request_with_deadline(&self, batch: &mut Batch, request: &str) -> Result<Reply, GitError> {
        let (done, wait) = mpsc::channel::<()>();
        let expired = Arc::new(AtomicBool::new(false));
        let watchdog = {
            let tree = Arc::clone(&batch.tree);
            let expired = Arc::clone(&expired);
            let deadline = self.deadline;
            thread::spawn(move || {
                if let Err(RecvTimeoutError::Timeout) = wait.recv_timeout(deadline) {
                    expired.store(true, Ordering::SeqCst);
                    tree.kill();
                }
            })
        };
        let reply = Self::request(batch, request, self.max_bytes);
        let _ = done.send(());
        let _ = watchdog.join();
        if expired.load(Ordering::SeqCst) {
            return Err(GitError::Timeout {
                args: vec!["cat-file".into(), "--batch".into(), request.to_string()],
                after: self.deadline,
            });
        }
        reply.map_err(GitError::Io)
    }

    fn read_spec(&self, request: String) -> Result<Option<Vec<u8>>, GitError> {
        let mut guard = self.batch.lock().unwrap_or_else(|e| e.into_inner());
        // One restart: a child that exited since the last read is replaced.
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(self.start()?);
            }
            let batch = guard.as_mut().expect("started above");
            match self.request_with_deadline(batch, &request) {
                Ok(Reply::Missing) => return Ok(None),
                Ok(Reply::Found { kind, body }) if kind == "blob" => return Ok(Some(body)),
                Ok(Reply::Found { kind, .. }) => {
                    return Err(GitError::NotABlob {
                        object: request,
                        kind,
                    })
                }
                Ok(Reply::TooLarge { size }) => {
                    // Its bytes are still in the pipe: start afresh next read.
                    *guard = None;
                    return Err(GitError::BlobTooLarge {
                        object: request,
                        size,
                        limit: self.max_bytes,
                    });
                }
                Err(e @ GitError::Timeout { .. }) => {
                    *guard = None;
                    return Err(e);
                }
                Err(e) => {
                    *guard = None;
                    if attempt == 1 {
                        return Err(e);
                    }
                }
            }
        }
        unreachable!("the loop returns on its second attempt")
    }

    /// The contents of `path` at `rev`, or `None` when `rev` holds no such
    /// path.
    pub fn read_at(&self, rev: &Rev, path: &RepoPath) -> Result<Option<Vec<u8>>, GitError> {
        self.read_spec(format!("{}:{}", rev.spec(), path))
    }

    /// The blob `oid`, or `None` when the repository does not hold it.
    pub fn read_blob(&self, oid: &Oid) -> Result<Option<Vec<u8>>, GitError> {
        self.read_spec(oid.to_string())
    }
}

impl Repo {
    /// A reader for file contents in this repository. The child starts on
    /// the first read. Each read may fetch from the network in a partial
    /// clone, so the deadline is the network one.
    pub fn blobs(&self) -> BlobReader {
        BlobReader {
            dir: self.dir.clone(),
            batch: Mutex::new(None),
            args: &["--batch"],
            deadline: NETWORK_TIMEOUT,
            max_bytes: MAX_BLOB_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::types::BranchName;

    #[test]
    fn headers_parse() {
        assert_eq!(
            parse_header(b"ce013625030ba8dba906f756967f9e9ca394464a blob 6\n", "x").unwrap(),
            Header::Found {
                kind: "blob".into(),
                size: 6
            }
        );
        assert_eq!(
            parse_header(b"HEAD:x missing\n", "HEAD:x").unwrap(),
            Header::Missing
        );
        assert!(parse_header(b"garbage\n", "x").is_err());
    }

    #[test]
    fn files_read_at_any_revision_byte_for_byte() {
        let t = TestRepo::init();
        let binary: Vec<u8> = (0..=255u8).chain(b"\r\n\0\n".iter().copied()).collect();
        t.write("bin.dat", &binary);
        t.write("dir with space/ünïcode.txt", b"first\n");
        let first = t.commit_all("first");
        t.git(&["branch", "old"]);
        t.write("dir with space/ünïcode.txt", b"second\n");
        t.commit_all("second");

        let repo = t.repo();
        let blobs = repo.blobs();
        let path = RepoPath::parse("dir with space/ünïcode.txt").unwrap();
        assert_eq!(
            blobs.read_at(&Rev::Head, &path).unwrap().unwrap(),
            b"second\n"
        );
        let old = Rev::Branch(BranchName::parse("old").unwrap());
        assert_eq!(blobs.read_at(&old, &path).unwrap().unwrap(), b"first\n");
        assert_eq!(
            blobs.read_at(&Rev::Oid(first), &path).unwrap().unwrap(),
            b"first\n"
        );
        assert_eq!(
            blobs
                .read_at(&Rev::Head, &RepoPath::parse("bin.dat").unwrap())
                .unwrap()
                .unwrap(),
            binary
        );
        let hello = Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
        assert_eq!(blobs.read_blob(&hello).unwrap(), None);
    }

    #[test]
    fn a_missing_path_is_none_and_a_folder_is_not_a_blob() {
        let t = TestRepo::init();
        t.write("src/lib.rs", b"x\n");
        t.commit_all("base");
        let repo = t.repo();
        let blobs = repo.blobs();
        let missing = RepoPath::parse("nope.rs").unwrap();
        assert_eq!(blobs.read_at(&Rev::Head, &missing).unwrap(), None);
        let folder = RepoPath::parse("src").unwrap();
        assert!(matches!(
            blobs.read_at(&Rev::Head, &folder),
            Err(GitError::NotABlob { .. })
        ));
        // The reader is still in step after an error.
        let lib = RepoPath::parse("src/lib.rs").unwrap();
        assert_eq!(blobs.read_at(&Rev::Head, &lib).unwrap().unwrap(), b"x\n");
    }

    #[test]
    fn a_dead_child_is_restarted() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        let repo = t.repo();
        let blobs = repo.blobs();
        let a = RepoPath::parse("a.txt").unwrap();
        assert!(blobs.read_at(&Rev::Head, &a).unwrap().is_some());
        {
            let mut guard = blobs.batch.lock().unwrap();
            let batch = guard.as_mut().unwrap();
            batch.tree.kill();
            let _ = batch.child.wait();
        }
        assert_eq!(blobs.read_at(&Rev::Head, &a).unwrap().unwrap(), b"a\n");
    }

    /// **An oversized object is refused from its header**, and the next read
    /// — on a fresh child — still works.
    #[test]
    fn an_object_over_the_limit_is_refused_and_the_reader_recovers() {
        let t = TestRepo::init();
        t.write("big.txt", b"0123456789\n");
        t.write("small.txt", b"ok\n");
        t.commit_all("base");
        let repo = t.repo();
        let blobs = BlobReader {
            max_bytes: 4,
            ..repo.blobs()
        };
        match blobs.read_at(&Rev::Head, &RepoPath::parse("big.txt").unwrap()) {
            Err(GitError::BlobTooLarge { size, limit, .. }) => assert_eq!((size, limit), (11, 4)),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            blobs
                .read_at(&Rev::Head, &RepoPath::parse("small.txt").unwrap())
                .unwrap()
                .unwrap(),
            b"ok\n"
        );
    }

    /// **A read that never answers times out**, killing the child. With
    /// `--buffer`, `cat-file` holds its answers until its input ends, which
    /// a reader never does — a stand-in for a lazy fetch that stalls.
    #[test]
    fn a_read_that_never_answers_times_out() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        let repo = t.repo();
        let blobs = BlobReader {
            args: &["--batch", "--buffer"],
            deadline: Duration::from_millis(500),
            ..repo.blobs()
        };
        let started = std::time::Instant::now();
        let got = blobs.read_at(&Rev::Head, &RepoPath::parse("a.txt").unwrap());
        assert!(matches!(got, Err(GitError::Timeout { .. })), "{got:?}");
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(
            blobs.batch.lock().unwrap().is_none(),
            "the stuck child was dropped"
        );
    }
}
