//! File contents at any revision.
//!
//! A checkout's are read from one long-running `cat-file --batch`; a
//! repository read from the substrate's, through libgit2 over its database
//! (`docs/substrate_git.md` §4.1).
//!
//! Starting a process per read costs milliseconds on Windows, so a checkout's
//! reader keeps one `git cat-file --batch` per repository and feeds it
//! requests over stdin, one per line. A request is a revision spec and a
//! [`RepoPath`], and neither can hold a newline, so the line cannot be split.
//! Reads are serialised by the reader's own lock; the child is restarted if it
//! exits, and killed with its process tree when the reader is dropped.
//!
//! Two bounds keep one read from taking the daemon down with it:
//!
//! - **Size.** An object larger than [`MAX_BLOB_BYTES`] is refused from its
//!   header, before any buffer is allocated for it; a multi-gigabyte blob
//!   would otherwise be a multi-gigabyte allocation.
//! - **Time.** Each read of a checkout has a deadline. In a partial clone a
//!   read may fetch from the network, and a fetch that stalls would otherwise
//!   hold the reader's lock forever; a watchdog kills the child's process tree
//!   when the deadline passes, and the read returns [`GitError::Timeout`].

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use std::str;

use git2::{ErrorCode, ObjectType};

use crate::error::GitError;
use crate::kill_tree::ProcessTree;
use crate::library::{failed, is_absent, oid_of};
use crate::odb::repository::repository;
use crate::odb::ObjectDb;
use crate::runner::{Invocation, NETWORK_TIMEOUT};
use crate::sandbox::Priority;
use crate::types::{Oid, RepoPath, Rev};
use crate::{Place, Repo};

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

/// What a reader reads through.
enum Source {
    /// A checkout: one `cat-file --batch` child, started on the first read.
    Batch {
        dir: PathBuf,
        batch: Mutex<Option<Batch>>,
        args: &'static [&'static str],
        deadline: Duration,
    },
    /// A repository read from the substrate: libgit2 over its database, made
    /// on the first read, which reports any failure to make it.
    Database {
        db: ObjectDb,
        lib: Mutex<Option<git2::Repository>>,
    },
}

pub struct BlobReader {
    source: Source,
    max_bytes: u64,
}

/// What one request's header line says.
#[derive(Debug, PartialEq, Eq)]
enum Header {
    Missing,
    Found { oid: Oid, kind: String, size: u64 },
}

fn parse_header(line: &[u8], request: &str) -> Result<Header, GitError> {
    let line = str::from_utf8(line)
        .map_err(|e| GitError::malformed("cat-file", e.to_string()))?
        .trim_end_matches('\n');
    if line == format!("{request} missing") || line == format!("{request} ambiguous") {
        return Ok(Header::Missing);
    }
    let parts: Vec<&str> = line.split(' ').collect();
    let [oid, kind, size] = parts[..] else {
        return Err(GitError::malformed("cat-file", line.to_string()));
    };
    let oid = Oid::parse(oid)?;
    let size = size
        .parse()
        .map_err(|_| GitError::malformed("cat-file", line.to_string()))?;
    Ok(Header::Found {
        oid,
        kind: kind.to_string(),
        size,
    })
}

/// A request's answer.
enum Reply {
    Missing,
    Found {
        oid: Oid,
        kind: String,
        body: Vec<u8>,
    },
    /// Refused from its header; the child is now out of step.
    TooLarge {
        size: u64,
    },
}

fn start(dir: &Path, args: &[&str]) -> Result<Batch, GitError> {
    let mut cmd = Invocation::new(dir, "cat-file")
        .args(args.iter().copied())
        .command()?;
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn()?;
    let tree = Arc::new(ProcessTree::adopt(&child, Priority::Normal)?);
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
        Header::Found { oid, kind, size } => {
            let mut body = vec![0; size as usize + 1];
            batch.stdout.read_exact(&mut body)?;
            body.pop(); // the newline after the contents
            Ok(Reply::Found { oid, kind, body })
        }
    }
}

/// [`request`] under `deadline`: a watchdog kills the child's tree if the
/// answer has not come by then, which unblocks the read.
fn request_with_deadline(
    batch: &mut Batch,
    req: &str,
    deadline: Duration,
    max_bytes: u64,
) -> Result<Reply, GitError> {
    let (done, wait) = mpsc::channel::<()>();
    let expired = Arc::new(AtomicBool::new(false));
    let watchdog = {
        let tree = Arc::clone(&batch.tree);
        let expired = Arc::clone(&expired);
        thread::spawn(move || {
            if let Err(RecvTimeoutError::Timeout) = wait.recv_timeout(deadline) {
                expired.store(true, Ordering::SeqCst);
                tree.kill();
            }
        })
    };
    let reply = request(batch, req, max_bytes);
    let _ = done.send(());
    let _ = watchdog.join();
    if expired.load(Ordering::SeqCst) {
        return Err(GitError::Timeout {
            args: vec!["cat-file".into(), "--batch".into(), req.to_string()],
            after: deadline,
        });
    }
    reply.map_err(GitError::Io)
}

/// The kind libgit2 names an object of, as `cat-file` prints it.
fn kind_name(kind: Option<ObjectType>) -> String {
    match kind {
        Some(ObjectType::Blob) => "blob",
        Some(ObjectType::Tree) => "tree",
        Some(ObjectType::Commit) => "commit",
        Some(ObjectType::Tag) => "tag",
        _ => "unknown",
    }
    .to_string()
}

impl BlobReader {
    /// The object `request` names — a revision spec as `cat-file` takes one —
    /// which must be a `kind`: its id and contents, or `None` when the
    /// repository holds no such object.
    fn read_kind(&self, request: String, kind: &str) -> Result<Option<(Oid, Vec<u8>)>, GitError> {
        match &self.source {
            Source::Batch {
                dir,
                batch,
                args,
                deadline,
            } => {
                let mut guard = batch.lock().unwrap_or_else(|e| e.into_inner());
                // One restart: a child that exited since the last read is
                // replaced.
                for attempt in 0..2 {
                    if guard.is_none() {
                        *guard = Some(start(dir, args)?);
                    }
                    let child = guard.as_mut().expect("started above");
                    match request_with_deadline(child, &request, *deadline, self.max_bytes) {
                        Ok(Reply::Missing) => return Ok(None),
                        Ok(Reply::Found {
                            oid,
                            kind: found,
                            body,
                        }) if found == kind => return Ok(Some((oid, body))),
                        Ok(Reply::Found { kind, .. }) => {
                            return Err(GitError::NotABlob {
                                object: request,
                                kind,
                            })
                        }
                        Ok(Reply::TooLarge { size }) => {
                            // Its bytes are still in the pipe: start afresh
                            // next read.
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
            Source::Database { db, lib } => {
                let mut held = lib.lock().unwrap_or_else(|e| e.into_inner());
                if held.is_none() {
                    *held = Some(repository(db)?);
                }
                let lib = held.as_ref().expect("made above");
                let object = match lib.revparse_single(&request) {
                    Ok(object) => object,
                    Err(e) if is_absent(&e) || e.code() == ErrorCode::Peel => return Ok(None),
                    Err(e) => return Err(failed("revparse", e)),
                };
                let found = kind_name(object.kind());
                if found != kind {
                    return Err(GitError::NotABlob {
                        object: request,
                        kind: found,
                    });
                }
                let odb = lib.odb().map_err(|e| failed("odb", e))?;
                let (size, _) = odb
                    .read_header(object.id())
                    .map_err(|e| failed("read_header", e))?;
                if size as u64 > self.max_bytes {
                    return Err(GitError::BlobTooLarge {
                        object: request,
                        size: size as u64,
                        limit: self.max_bytes,
                    });
                }
                let read = odb.read(object.id()).map_err(|e| failed("read", e))?;
                Ok(Some((oid_of(object.id())?, read.data().to_vec())))
            }
        }
    }

    /// The contents of `path` at `rev`, or `None` when `rev` holds no such
    /// path.
    pub fn read_at(&self, rev: &Rev, path: &RepoPath) -> Result<Option<Vec<u8>>, GitError> {
        let found = self.read_kind(format!("{}:{}", rev.spec(), path), "blob")?;
        Ok(found.map(|(_, body)| body))
    }

    /// The blob `oid`, or `None` when the repository does not hold it.
    pub fn read_blob(&self, oid: &Oid) -> Result<Option<Vec<u8>>, GitError> {
        Ok(self
            .read_kind(oid.to_string(), "blob")?
            .map(|(_, body)| body))
    }

    /// The commit `rev` names and that commit's root tree, or `None` when
    /// `rev` names nothing — a branch that does not exist, or a `HEAD` with
    /// no commit yet. One request to the running child: the tree is the first
    /// line of the commit object.
    pub fn commit_of(&self, rev: &Rev) -> Result<Option<(Oid, Oid)>, GitError> {
        let request = format!("{}^{{commit}}", rev.spec());
        let Some((commit, body)) = self.read_kind(request, "commit")? else {
            return Ok(None);
        };
        let tree = body
            .strip_prefix(b"tree ")
            .and_then(|rest| rest.split(|&b| b == b'\n').next())
            .and_then(|id| str::from_utf8(id).ok())
            .ok_or_else(|| {
                GitError::malformed("cat-file", format!("commit {commit} names no tree"))
            })?;
        Ok(Some((commit, Oid::parse(tree)?)))
    }
}

impl Repo {
    /// A reader for file contents in this repository. A checkout's child
    /// starts on the first read; each read may fetch from the network in a
    /// partial clone, so the deadline is the network one.
    pub fn blobs(&self) -> BlobReader {
        let source = match &self.place {
            Place::Substrate(db) => Source::Database {
                db: db.clone(),
                lib: Mutex::new(None),
            },
            Place::Folder { dir, .. } => Source::Batch {
                dir: dir.clone(),
                batch: Mutex::new(None),
                args: &["--batch"],
                deadline: NETWORK_TIMEOUT,
            },
        };
        BlobReader {
            source,
            max_bytes: MAX_BLOB_BYTES,
        }
    }
}

#[cfg(test)]
mod tests;
