//! The sandbox server end to end: jobs started in the background, their
//! output followed as a stream and kept in a log file, their status asked
//! after, and cancelled by dropping their handle — the folder put back as it
//! was however they end.

mod support;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use support::*;
use zend_vfs::{
    CommandPolicy, JobId, JobNotFound, JobRequest, JobStatus, OutputStream, Repo, Sandbox,
    SandboxCommand, SandboxServer, StartedJob, VfsStore,
};

/// A server over the fixture's repository, its logs in a `jobs` folder that
/// does not exist yet.
fn server(f: &Fixture) -> SandboxServer {
    let sandbox = Sandbox::new(
        Repo::open(&f.root).unwrap(),
        CommandPolicy::allowing([SHELL]),
    );
    SandboxServer::new(sandbox, f.dir.path().join("jobs")).unwrap()
}

fn start(server: &SandboxServer, files: &Arc<VfsStore>, command: SandboxCommand) -> StartedJob {
    server
        .start_job(
            grant(),
            JobRequest {
                branch: branch("main"),
                files: Arc::clone(files),
                command,
            },
        )
        .unwrap()
}

async fn collect(mut output: OutputStream) -> Vec<u8> {
    let mut all = Vec::new();
    while let Some(chunk) = output.next().await {
        all.extend(chunk.unwrap());
    }
    all
}

/// **A job runs in the background with its output in its log and in its
/// stream** — the same bytes — and its handle gives the outcome: the files
/// it changed, now the conversation's. Asked after, it has exited, with the
/// log's line count.
#[tokio::test]
async fn a_job_writes_its_log_streams_it_and_ends() {
    let f = fixture();
    let server = server(&f);
    assert!(server.jobs_dir().is_dir(), "the jobs folder is made");
    let files = Arc::new(f.store());
    let job = start(
        &server,
        &files,
        shell(
            "echo one & echo two 1>&2 & echo made> made.txt",
            "echo one; echo two 1>&2; echo made > made.txt",
        ),
    );
    assert_eq!(job.id.as_str().len(), 11);
    assert_eq!(job.log, server.jobs_dir().join(format!("{}.log", job.id)));

    let streamed = collect(job.output).await;
    let outcome = job.handle.finish().await.unwrap();
    assert_eq!(outcome.exit_code, Some(0));
    assert_eq!(
        outcome
            .changed
            .iter()
            .map(|c| c.path.as_str())
            .collect::<Vec<_>>(),
        ["made.txt"]
    );
    assert_eq!(files.read("made.txt").unwrap().unwrap().trim_end(), "made");

    let logged = std::fs::read(&job.log).unwrap();
    assert_eq!(streamed, logged);
    let text = String::from_utf8(logged).unwrap();
    assert!(text.contains("one") && text.contains("two"), "{text:?}");
    let info = server.query_job(&job.id).unwrap();
    assert_eq!(info.status, JobStatus::Exited { code: Some(0) });
    assert_eq!(info.lines, 2);
    assert_eq!(info.bytes, text.len() as u64);
    assert_eq!(info.log, job.log);
}

/// **Dropping the handle cancels the job**: its command is killed, its
/// stream ends, and it reports cancelled — never what it would have done.
#[tokio::test]
async fn dropping_the_handle_cancels_the_job() {
    let f = fixture();
    let server = server(&f);
    let files = Arc::new(f.store());
    let job = start(
        &server,
        &files,
        shell(
            "echo started & ping -n 30 127.0.0.1 >NUL & echo late> late.txt",
            "echo started; sleep 30; echo late > late.txt",
        ),
    );
    // Wait for the command to be running.
    let mut output = job.output;
    let first = output.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).contains("started"));
    drop(job.handle);

    let rest = tokio::time::timeout(Duration::from_secs(10), collect(output))
        .await
        .expect("the stream ended with the job");
    assert!(!String::from_utf8_lossy(&rest).contains("late"));
    assert_eq!(
        server.query_job(&job.id).unwrap().status,
        JobStatus::Cancelled
    );
    assert!(!files.is_modified("late.txt"));
}

/// **A cancelled job puts the folder back as it was** — someone's
/// uncommitted edit and new file included — before the next job runs.
#[tokio::test]
async fn a_cancelled_job_puts_the_folder_back() {
    let f = fixture();
    put(&f.root, "README.md", b"# app\n\nmy edit.\n");
    put(&f.root, "mine.txt", b"my new file\n");
    let before = git(
        &f.root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    );
    let server = server(&f);
    let files = Arc::new(f.store());
    files
        .write("README.md", "# the conversation's\n".into())
        .unwrap();
    let job = start(
        &server,
        &files,
        shell(
            "echo started & ping -n 30 127.0.0.1 >NUL",
            "echo started; sleep 30",
        ),
    );
    let mut output = job.output;
    output.next().await.unwrap().unwrap();
    assert_eq!(
        disk(&f.root, "README.md").unwrap(),
        b"# the conversation's\n",
        "the conversation's files are down while it runs"
    );
    drop(job.handle);

    let next = start(&server, &Arc::new(f.store()), shell("echo x", "echo x"));
    next.handle.finish().await.unwrap();
    assert_eq!(
        git(
            &f.root,
            &["status", "--porcelain=v1", "--untracked-files=all"]
        ),
        before
    );
    assert_eq!(disk(&f.root, "README.md").unwrap(), b"# app\n\nmy edit.\n");
    assert_eq!(disk(&f.root, "mine.txt").unwrap(), b"my new file\n");
    assert_nothing_set_aside(&f);
}

/// **A job waiting for another holds its place as queued**, then runs.
#[tokio::test]
async fn a_job_behind_another_is_queued() {
    let f = fixture();
    let server = server(&f);
    let files = Arc::new(f.store());
    let slow = start(
        &server,
        &files,
        shell("ping -n 3 127.0.0.1 >NUL", "sleep 2"),
    );
    // The slow job holds the checkout once it reports running.
    for _ in 0..200 {
        if server.query_job(&slow.id).unwrap().status == JobStatus::Running {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let waiting = start(&server, &files, shell("echo x", "echo x"));
    assert_eq!(
        server.query_job(&waiting.id).unwrap().status,
        JobStatus::Queued
    );
    slow.handle.finish().await.unwrap();
    waiting.handle.finish().await.unwrap();
    assert!(matches!(
        server.query_job(&waiting.id).unwrap().status,
        JobStatus::Exited { .. }
    ));
}

/// **A refused job reports why, and an unknown id is not found.**
#[tokio::test]
async fn a_refused_job_and_an_unknown_one() {
    let f = fixture();
    let server = server(&f);
    let files = Arc::new(f.store());
    let job = start(&server, &files, SandboxCommand::new("git").arg("status"));
    assert!(job.handle.finish().await.is_err());
    match server.query_job(&job.id).unwrap().status {
        JobStatus::Refused { why } => assert!(why.contains("git tools"), "{why}"),
        other => panic!("{other:?}"),
    }
    let unknown = JobId::parse("AAAAAAAAAAA").unwrap();
    assert_eq!(
        server.query_job(&unknown).unwrap_err(),
        JobNotFound(unknown)
    );
}
