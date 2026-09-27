//! Sandbox runs at the edges: every shape of change a command can make to the
//! checkout, what must not reach the conversation, a command that changes the
//! checkout's git state behind the sandbox's back, processes that misbehave,
//! runs that overlap or are abandoned, and links out of the repository.
//!
//! Every test ends the same way: the next job puts the checkout back to the
//! branch, and nothing outside the repository has been touched.

mod support;

use std::sync::Arc;
use std::time::Duration;

use support::*;
use zend_vfs::file_delta::FileDelta;
use zend_vfs::{Oid, RunOutcome, SandboxCommand, SandboxError, VfsStore};

fn main_at(f: &Fixture) -> Oid {
    Oid::parse(git(&f.root, &["rev-parse", "refs/heads/main"]).trim()).unwrap()
}

fn delta<'a>(done: &'a RunOutcome, path: &str) -> &'a FileDelta {
    &done
        .changed
        .iter()
        .find(|c| c.path == path)
        .unwrap_or_else(|| panic!("{path} was not recorded: {done:?}"))
        .delta
        .delta
}

/// A file in a folder beside the repository's, `../outside/outside.txt`,
/// that no run may touch. A sibling rather than the repository's parent: a
/// link to an ancestor is a cycle, which git walks until the path is too long.
fn outside(f: &Fixture) -> std::path::PathBuf {
    let dir = f.dir.path().join("outside");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("outside.txt");
    std::fs::write(&p, b"outside the repository\n").unwrap();
    p
}

// ── Every shape of change ────────────────────────────────────────────────────

/// **A command editing a file the conversation already edited records an
/// edit on top of the conversation's** — made against the conversation's
/// content, not the branch's.
#[tokio::test]
async fn an_edit_on_top_of_the_conversations_edit() {
    let f = fixture();
    let files = f.store();
    files
        .edit("src/lib.rs", "pub fn one() -> u8 {\n    11\n}\n".into())
        .unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell("echo // tool>> src\\lib.rs", "echo '// tool' >> src/lib.rs"),
    )
    .await;
    assert_eq!(changed(&done), ["src/lib.rs"]);
    assert!(matches!(delta(&done, "src/lib.rs"), FileDelta::Edit { .. }));
    let text = read(&files, "src/lib.rs").unwrap();
    assert!(
        text.starts_with("pub fn one() -> u8 {\n    11\n}\n// tool"),
        "{text:?}"
    );
    assert_eq!(files.deltas("src/lib.rs").unwrap().len(), 2);
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "src/lib.rs").unwrap(), LIB);
}

/// **Deleting a tracked file records a delete**; the checkout gets it back.
#[tokio::test]
async fn deleting_a_tracked_file() {
    let f = fixture();
    let files = f.store();
    let done = run(&f, "main", &files, &shell("del README.md", "rm README.md")).await;
    assert_eq!(changed(&done), ["README.md"]);
    assert_eq!(delta(&done, "README.md"), &FileDelta::Delete);
    assert_eq!(read(&files, "README.md"), None);
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "README.md").unwrap(), README);
}

/// **A file the conversation deleted is absent for the command**, and one the
/// command then writes is recorded over the deletion.
#[tokio::test]
async fn recreating_a_file_the_conversation_deleted() {
    let f = fixture();
    let files = f.store();
    assert!(files.delete("README.md"));
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "if exist README.md (echo present) else (echo absent) & echo again> README.md",
            "if [ -e README.md ]; then echo present; else echo absent; fi; echo again > README.md",
        ),
    )
    .await;
    assert_eq!(done.printed.trim_end(), "absent");
    assert_eq!(changed(&done), ["README.md"]);
    assert_eq!(read(&files, "README.md").unwrap().trim_end(), "again");
    assert_put_back(&f, "main").await;
}

/// **A command putting a file back to the branch's content records that**:
/// the conversation's change is undone, byte for byte.
#[tokio::test]
async fn putting_a_changed_file_back_to_the_branch() {
    let f = fixture();
    let files = f.store();
    files.write("src/lib.rs", "rewritten\n".into()).unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "copy /Y templates\\lib.rs.orig src\\lib.rs >NUL",
            "cp templates/lib.rs.orig src/lib.rs",
        ),
    )
    .await;
    assert_eq!(changed(&done), ["src/lib.rs"]);
    assert_eq!(read(&files, "src/lib.rs").unwrap().as_bytes(), LIB);
    assert_put_back(&f, "main").await;
}

/// **Rewriting a file with the bytes it already holds records nothing**, and
/// neither does a file made and removed within the run.
#[tokio::test]
async fn rewriting_the_same_bytes_records_nothing() {
    let f = fixture();
    let files = f.store();
    files.write("mine.txt", "mine\n".into()).unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "copy /Y README.md tmp.txt >NUL & copy /Y tmp.txt README.md >NUL & \
             copy /Y mine.txt tmp.txt >NUL & copy /Y tmp.txt mine.txt >NUL & del tmp.txt",
            "cp README.md tmp.txt; cp tmp.txt README.md; cp mine.txt tmp.txt; \
             cp tmp.txt mine.txt; rm tmp.txt",
        ),
    )
    .await;
    assert!(done.changed.is_empty(), "{done:?}");
    assert!(done.unrecorded.is_empty(), "{done:?}");
    assert_put_back(&f, "main").await;
}

/// **New files deep in new folders, with spaces and non-ASCII names, and an
/// empty file, are all recorded** — the empty one as an empty file, not as
/// nothing.
#[tokio::test]
async fn new_nested_files_with_awkward_names_and_an_empty_file() {
    let f = fixture();
    let files = f.store();
    // Quoted names go through a batch file: `cmd` does not read the escaped
    // quotes a quoted argument arrives with.
    files
        .write(
            "mk.cmd",
            "@chcp 65001 >NUL\r\n@mkdir \"a b\\c\"\r\n@echo x> \"a b\\c\\café ü.txt\"\r\n\
             @type NUL > empty.txt\r\n"
                .into(),
        )
        .unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            ".\\mk.cmd",
            "mkdir -p 'a b/c'; echo x > 'a b/c/café ü.txt'; : > empty.txt",
        ),
    )
    .await;
    assert_eq!(
        changed(&done),
        ["a b/c/café ü.txt", "empty.txt"],
        "{done:?}"
    );
    assert_eq!(read(&files, "a b/c/café ü.txt").unwrap().trim_end(), "x");
    assert_eq!(read(&files, "empty.txt").unwrap(), "");
    assert_put_back(&f, "main").await;
    assert!(
        !f.root.join("a b").exists(),
        "the new folder was left behind"
    );
}

/// **Deleting a whole tracked folder records every file in it.**
#[tokio::test]
async fn deleting_a_whole_tracked_folder() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell("rmdir /S /Q templates", "rm -r templates"),
    )
    .await;
    assert_eq!(changed(&done), ["templates/lib.rs.orig"]);
    assert_eq!(read(&files, "templates/lib.rs.orig"), None);
    assert_put_back(&f, "main").await;
}

/// **A file replaced by a folder of the same name is a delete of the file
/// and new files inside the folder** — and the conversation, run again,
/// puts that folder back.
#[tokio::test]
async fn a_file_replaced_by_a_folder() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "del src\\lib.rs & mkdir src\\lib.rs & echo inner> src\\lib.rs\\inner.txt",
            "rm src/lib.rs; mkdir src/lib.rs; echo inner > src/lib.rs/inner.txt",
        ),
    )
    .await;
    assert_eq!(changed(&done), ["src/lib.rs", "src/lib.rs/inner.txt"]);
    assert_eq!(delta(&done, "src/lib.rs"), &FileDelta::Delete);
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "src/lib.rs").unwrap(), LIB);

    let again = run(
        &f,
        "main",
        &files,
        &shell("type src\\lib.rs\\inner.txt", "cat src/lib.rs/inner.txt"),
    )
    .await;
    assert_eq!(again.printed.trim_end(), "inner");
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "src/lib.rs").unwrap(), LIB);
}

// ── What must not reach the conversation ─────────────────────────────────────

/// **A by-product under an ignore rule is not recorded, and stays as a build
/// cache would** — but an ignored file the conversation itself wrote is its
/// own, recorded when changed and cleared from the checkout afterwards.
#[tokio::test]
async fn ignored_output_is_not_recorded_but_the_conversations_ignored_file_is() {
    let f = fixture();
    let files = f.store();
    files
        .write("notes.log", "the conversation's\n".into())
        .unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "echo more>> notes.log & mkdir target & echo built> target\\out.bin",
            "echo more >> notes.log; mkdir target; echo built > target/out.bin",
        ),
    )
    .await;
    assert_eq!(changed(&done), ["notes.log"]);
    assert!(read(&files, "notes.log").unwrap().contains("more"));
    assert!(!files.is_modified("target/out.bin"));
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "notes.log"), None);
    assert!(disk(&f.root, "target/out.bin").is_some(), "the build cache");
}

/// **A file under a protected folder is reported, never recorded, and never
/// left in the checkout**; a change inside `.git` is not a file change at
/// all.
#[tokio::test]
async fn a_protected_file_the_command_writes_is_unrecorded_and_removed() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "mkdir secrets & echo key> secrets\\key.txt & echo x> .git\\zend-marker",
            "mkdir secrets; echo key > secrets/key.txt; echo x > .git/zend-marker",
        ),
    )
    .await;
    assert!(done.changed.is_empty(), "{done:?}");
    assert_eq!(unrecorded(&done), ["secrets/key.txt"]);
    assert!(done.unrecorded[0].why.contains("protected"), "{done:?}");
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "secrets/key.txt"), None);
}

/// **What would overflow the store's cap is reported, not lost silently.**
#[tokio::test]
async fn changes_past_the_stores_cap_are_reported() {
    let f = fixture();
    let files = f.store();
    let big = "0123456789abcdef\n".repeat(6 * 1024 * 1024 / 17);
    files.write("big.txt", big).unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "copy /Y big.txt a.txt >NUL & copy /Y big.txt b.txt >NUL",
            "cp big.txt a.txt; cp big.txt b.txt",
        ),
    )
    .await;
    assert!(done.changed.is_empty(), "{:?}", changed(&done));
    assert_eq!(unrecorded(&done), ["a.txt", "b.txt"]);
    assert!(done.unrecorded.iter().all(|u| u.why.contains("limit")));
    assert_eq!(read(&files, "a.txt"), None);
    assert_put_back(&f, "main").await;
}

// ── Git state changed behind the sandbox's back ──────────────────────────────
//
// git named in a command is refused (see `sandbox_git.rs`); a program that
// runs git itself is not. These run git from a script file the conversation
// wrote — the command names only the script — and check the sandbox takes
// back whatever it did to the checkout's git state.

/// A command running a script the conversation writes: `lines` as a batch
/// file on Windows, as an sh script elsewhere.
fn script(files: &VfsStore, windows: &[&str], unix: &[&str]) -> SandboxCommand {
    let batch: String = windows.iter().map(|l| format!("@{l}\r\n")).collect();
    files.write("tool.cmd", batch).unwrap();
    let sh: String = unix.iter().map(|l| format!("{l}\n")).collect();
    files.write("tool.sh", sh).unwrap();
    shell(".\\tool.cmd", "sh tool.sh")
}

/// **Staging is undone**, and what was staged is recorded as the file change
/// it is.
#[tokio::test]
async fn a_program_that_stages() {
    let f = fixture();
    let files = f.store();
    let command = script(
        &files,
        &[
            "echo new> staged.txt",
            "git add staged.txt",
            "echo more>> README.md",
            "git add README.md",
        ],
        &[
            "echo new > staged.txt",
            "git add staged.txt",
            "echo more >> README.md",
            "git add README.md",
        ],
    );
    let done = run(&f, "main", &files, &command).await;
    assert_eq!(changed(&done), ["README.md", "staged.txt"]);
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "staged.txt"), None);
}

/// **A commit a program makes does not move the conversation's branch**:
/// the branch is put back, and what the commit held is recorded as the
/// program's changes.
#[tokio::test]
async fn a_program_that_commits() {
    let f = fixture();
    let files = f.store();
    let commit = "git -c user.name=t -c user.email=t@example.com -c commit.gpgSign=false \
                  commit -q -m tool";
    let command = script(
        &files,
        &[
            "echo c> committed.txt",
            "del README.md",
            "git add -A",
            commit,
        ],
        &[
            "echo c > committed.txt",
            "rm README.md",
            "git add -A",
            commit,
        ],
    );
    let done = run(&f, "main", &files, &command).await;
    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert_eq!(main_at(&f), f.base, "the branch moved");
    assert_eq!(changed(&done), ["README.md", "committed.txt"]);
    assert_eq!(read(&files, "README.md"), None);
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "README.md").unwrap(), README);
}

/// **Switching branch is undone**; what the switch changed on disk is the
/// program's change.
#[tokio::test]
async fn a_program_that_switches_branch() {
    let f = fixture();
    let files = f.store();
    let feature = git(&f.root, &["rev-parse", "refs/heads/feature"]);
    let command = script(
        &files,
        &["git checkout -q feature"],
        &["git checkout -q feature"],
    );
    let done = run(&f, "main", &files, &command).await;
    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert_eq!(changed(&done), ["FEATURE.md"]);
    assert_eq!(
        read(&files, "FEATURE.md").unwrap(),
        "only on the feature branch\n"
    );
    assert_put_back(&f, "main").await;
    assert_eq!(git(&f.root, &["rev-parse", "refs/heads/feature"]), feature);
}

/// **A detached `HEAD` is put back on the branch.**
#[tokio::test]
async fn a_program_that_detaches_head() {
    let f = fixture();
    let files = f.store();
    let command = script(
        &files,
        &["git checkout -q --detach"],
        &["git checkout -q --detach"],
    );
    let done = run(&f, "main", &files, &command).await;
    assert!(done.changed.is_empty(), "{done:?}");
    assert_put_back(&f, "main").await;
}

/// **A deleted branch is restored** where it was.
#[tokio::test]
async fn a_program_that_deletes_the_branch() {
    let f = fixture();
    let files = f.store();
    let command = script(
        &files,
        &["git update-ref -d refs/heads/main"],
        &["git update-ref -d refs/heads/main"],
    );
    let done = run(&f, "main", &files, &command).await;
    assert!(done.changed.is_empty(), "{done:?}");
    assert_eq!(main_at(&f), f.base);
    assert_put_back(&f, "main").await;
}

// ── Processes that misbehave ─────────────────────────────────────────────────

/// **Nothing a command starts outlives its run**: a background writer it
/// leaves — which demonstrably ran, its first file recorded — is killed with
/// it, so it cannot change the checkout after the run has handed it to the
/// next conversation, and the run does not wait for it.
#[tokio::test]
async fn a_background_process_does_not_outlive_the_run() {
    let f = fixture();
    let files = f.store();
    // The writer writes `early.txt`, then `late.txt` a second later. The
    // command starts it, waits only until `early.txt` exists — the writer is
    // demonstrably running — and exits. Alive at the run's end, the writer
    // would be waited for and its file captured, or would write it after the
    // run.
    files
        .write(
            "late.cmd",
            "@echo early> early.txt\r\n@ping -n 2 127.0.0.1 >NUL\r\n@echo late> late.txt\r\n"
                .into(),
        )
        .unwrap();
    files
        .write(
            "main.cmd",
            "@start /B .\\late.cmd\r\n:wait\r\n@if not exist early.txt goto wait\r\n@echo started\r\n"
                .into(),
        )
        .unwrap();
    files
        .write(
            "late.sh",
            "echo early > early.txt\nsleep 1\necho late > late.txt\n".into(),
        )
        .unwrap();
    files
        .write(
            "main.sh",
            "sh late.sh &\nwhile [ ! -e early.txt ]; do :; done\necho started\n".into(),
        )
        .unwrap();
    let done = run(&f, "main", &files, &shell(".\\main.cmd", "sh main.sh")).await;
    assert!(done.printed.contains("started"), "{done:?}");
    assert_eq!(
        changed(&done),
        ["early.txt"],
        "the writer never started, or the run waited for it"
    );
    // Past the moment the writer would have written.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert_eq!(
        disk(&f.root, "late.txt"),
        None,
        "the background writer lived"
    );
    assert!(!files.is_modified("late.txt"));
    assert_put_back(&f, "main").await;
}

/// **A command that reads its input gets its end at once**, never the
/// daemon's own.
#[tokio::test]
async fn a_command_reading_its_input_sees_the_end() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell("sort", "cat").timeout(Duration::from_secs(20)),
    )
    .await;
    assert!(!done.timed_out);
    assert_eq!(done.exit_code, Some(0));
    assert_put_back(&f, "main").await;
}

/// **Megabytes of output reach the sink whole**, counted, without stalling
/// the command. (The cap on what a sink is given is pinned in `process`.)
#[tokio::test]
async fn megabytes_of_output_reach_the_sink_whole() {
    let f = fixture();
    let files = f.store();
    let big = "0123456789abcde\n".repeat(3 * 1024 * 1024 / 16);
    files.write("big.txt", big.clone()).unwrap();
    let done = run(&f, "main", &files, &shell("type big.txt", "cat big.txt")).await;
    assert!(!done.output.truncated);
    assert_eq!(done.output.bytes, big.len() as u64);
    assert_eq!(done.printed, big);
    assert_put_back(&f, "main").await;
}

/// **Bytes that are not UTF-8 reach the sink as they are**, counted.
#[tokio::test]
async fn output_that_is_not_text() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell("type assets\\logo.bin", "cat assets/logo.bin"),
    )
    .await;
    assert_eq!(done.output.bytes, BINARY.len() as u64);
    assert_eq!(done.printed, String::from_utf8_lossy(BINARY));
    assert_put_back(&f, "main").await;
}

/// **A listed program that is not installed fails to start**, records
/// nothing, and leaves the lock free for the next run.
#[tokio::test]
async fn a_program_that_is_not_there() {
    let f = fixture_allowing(&["no-such-program-zend-vfs"]);
    let files = f.store();
    files.write("mine.txt", "mine\n".into()).unwrap();
    let before = files.changes().unwrap();
    for _ in 0..2 {
        let result = try_run(
            &f,
            "main",
            &files,
            &SandboxCommand::new("no-such-program-zend-vfs"),
        )
        .await;
        assert!(
            matches!(result, Err(SandboxError::Start { .. })),
            "{result:?}"
        );
        assert_eq!(files.changes().unwrap(), before);
    }
}

/// **A zero timeout kills the command at once** and still resets.
#[tokio::test]
async fn a_zero_timeout() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell("ping -n 30 127.0.0.1 >NUL", "sleep 30").timeout(Duration::ZERO),
    )
    .await;
    assert!(done.timed_out);
    assert_put_back(&f, "main").await;
}

/// **A command that fails still has its changes recorded.**
#[tokio::test]
async fn a_failing_command_still_records() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell("echo x> x.txt & exit 1", "echo x > x.txt; exit 1"),
    )
    .await;
    assert_eq!(done.exit_code, Some(1));
    assert_eq!(changed(&done), ["x.txt"]);
    assert_put_back(&f, "main").await;
}

/// **A file the command makes read-only does not wedge the checkout** for the
/// conversation that next writes it.
#[tokio::test]
async fn a_read_only_file_left_by_a_command() {
    let f = fixture();
    let first = f.store();
    run(
        &f,
        "main",
        &first,
        &shell("attrib +R README.md", "chmod a-w README.md"),
    )
    .await;
    let second = f.store();
    second.write("README.md", "# second\n".into()).unwrap();
    let done = run(
        &f,
        "main",
        &second,
        &shell("type README.md", "cat README.md"),
    )
    .await;
    assert_eq!(done.printed.trim_end(), "# second");
    assert_put_back(&f, "main").await;
}

// ── Runs over time, together, and abandoned ──────────────────────────────────

/// **One conversation's runs build on each other**: each sees what the last
/// recorded, laid down afresh — another conversation's run between them
/// included — and none leaves its files in the folder.
#[tokio::test]
async fn successive_runs_of_one_conversation() {
    let f = fixture();
    let files = f.store();
    let one = shell("echo one> log.txt", "echo one > log.txt");
    run_job(&f.sandbox, "one", "main", &files, &one)
        .await
        .unwrap();
    assert_eq!(disk(&f.root, "log.txt"), None, "the job's file is not left");
    let two = shell("echo two>> log.txt", "echo two >> log.txt");
    run_job(&f.sandbox, "two", "main", &files, &two)
        .await
        .unwrap();
    run(&f, "main", &f.store(), &shell("echo x", "echo x")).await;
    let list = shell("type log.txt", "cat log.txt");
    let done = run_job(&f.sandbox, "three", "main", &files, &list)
        .await
        .unwrap();
    let lines: Vec<&str> = done.printed.lines().map(str::trim_end).collect();
    assert_eq!(lines, ["one", "two"]);
    assert_eq!(files.deltas("log.txt").unwrap().len(), 2);
    assert_put_back(&f, "main").await;
}

/// **Many conversations at once, on real threads, each see only their own
/// files** and record only their own changes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_conversations_at_once() {
    const CONVERSATIONS: usize = 4;
    let f = Arc::new(fixture());
    let mut runs = tokio::task::JoinSet::new();
    for n in 0..CONVERSATIONS {
        let f = Arc::clone(&f);
        runs.spawn(async move {
            let files = f.store();
            files
                .write("id.txt", format!("conversation {n}\n"))
                .unwrap();
            files
                .write(&format!("only-{n}.txt"), "mine\n".into())
                .unwrap();
            let command = shell(
                "type id.txt & echo seen>> id.txt & dir /B",
                "cat id.txt; echo seen >> id.txt; ls",
            );
            let done = run(&f, "main", &files, &command).await;
            (n, done, read(&files, "id.txt").unwrap())
        });
    }
    let mut seen = 0;
    while let Some(joined) = runs.join_next().await {
        let (n, done, id) = joined.unwrap();
        assert!(
            done.printed.starts_with(&format!("conversation {n}")),
            "{n}: {:?}",
            done.printed
        );
        for m in 0..CONVERSATIONS {
            assert_eq!(
                done.printed.contains(&format!("only-{m}.txt")),
                m == n,
                "conversation {n} listing only-{m}.txt: {:?}",
                done.printed
            );
        }
        assert_eq!(changed(&done), ["id.txt"]);
        assert!(id.starts_with(&format!("conversation {n}\n")) && id.contains("seen"));
        seen += 1;
    }
    assert_eq!(seen, CONVERSATIONS);
    assert_put_back(&f, "main").await;
}

/// **An abandoned run records nothing and leaves nothing for the next**: the
/// next conversation sees neither its files nor what its command wrote.
#[tokio::test]
async fn an_abandoned_run() {
    let f = fixture();
    let a = f.store();
    a.write("a.txt", "a's\n".into()).unwrap();
    let slow = shell(
        "echo x> partial.txt & ping -n 30 127.0.0.1 >NUL",
        "echo x > partial.txt; sleep 30",
    );
    let abandoned = tokio::time::timeout(
        Duration::from_millis(600),
        run_job(&f.sandbox, "abandoned", "main", &a, &slow),
    )
    .await;
    assert!(abandoned.is_err(), "the run finished: {abandoned:?}");
    assert!(!a.is_modified("partial.txt"));

    let b = f.store();
    let list = shell("dir /B", "ls");
    let done = run_job(&f.sandbox, "b", "main", &b, &list).await.unwrap();
    assert!(!done.printed.contains("a.txt"), "{:?}", done.printed);
    assert!(!done.printed.contains("partial.txt"), "{:?}", done.printed);
    assert_put_back(&f, "main").await;
}

/// **A branch that does not exist fails the run before the checkout is
/// touched.**
#[tokio::test]
async fn a_branch_that_does_not_exist() {
    let f = fixture();
    put(&f.root, "untracked-dev-file.txt", b"left alone\n");
    let files = f.store();
    files.write("mine.txt", "mine\n".into()).unwrap();
    let result = try_run(&f, "absent", &files, &shell("echo x", "echo x")).await;
    assert!(
        matches!(
            &result,
            Err(SandboxError::BaseNotBranch { branch, tip, .. })
                if branch == "absent" && tip == "no commit"
        ),
        "{result:?}"
    );
    assert_eq!(
        disk(&f.root, "untracked-dev-file.txt").unwrap(),
        b"left alone\n"
    );
    assert_eq!(disk(&f.root, "mine.txt"), None);
}

/// **A conversation whose base the branch has moved past is refused before
/// the checkout is touched**: its changes were made on another commit than
/// a checkout of the branch would hold.
#[tokio::test]
async fn a_base_the_branch_moved_past_is_refused() {
    let f = fixture();
    let files = f.store();
    files.write("mine.txt", "mine\n".into()).unwrap();
    let pinned = files.base().unwrap().unwrap();
    // Another conversation's commit: objects and a ref, the checkout alone.
    let tree = git(&f.root, &["rev-parse", "main^{tree}"]);
    let moved = git(
        &f.root,
        &["commit-tree", tree.trim(), "-p", "main", "-m", "elsewhere"],
    );
    git(&f.root, &["update-ref", "refs/heads/main", moved.trim()]);
    let result = try_run(&f, "main", &files, &shell("echo x", "echo x")).await;
    assert!(
        matches!(
            &result,
            Err(SandboxError::BaseNotBranch { base, tip, .. })
                if *base == pinned.commit().unwrap().to_string() && tip == moved.trim()
        ),
        "{result:?}"
    );
    assert_eq!(disk(&f.root, "mine.txt"), None);
}

// ── Links out of the repository ──────────────────────────────────────────────

/// **A link the command makes to outside the repository is neither followed
/// nor recorded, and is removed** — the link, never what it points at.
#[tokio::test]
async fn a_link_to_outside_the_repository() {
    let f = fixture();
    let outside = outside(&f);
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "mklink /J linked ..\\outside >NUL",
            "ln -s ../outside linked",
        ),
    )
    .await;
    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert!(done.changed.is_empty(), "{done:?}");
    assert!(done.unrecorded.is_empty(), "{done:?}");
    assert_put_back(&f, "main").await;
    assert!(std::fs::symlink_metadata(f.root.join("linked")).is_err());
    assert_eq!(
        std::fs::read(&outside).unwrap(),
        b"outside the repository\n"
    );
    assert!(f.root.join("README.md").is_file());
}

/// **A tracked folder replaced by a link to outside** is the folder's files
/// deleted; the checkout gets the folder back and the outside is untouched.
#[tokio::test]
async fn a_folder_replaced_by_a_link_to_outside() {
    let f = fixture();
    let outside = outside(&f);
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "rmdir /S /Q templates & mklink /J templates ..\\outside >NUL",
            "rm -r templates; ln -s ../outside templates",
        ),
    )
    .await;
    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert_eq!(changed(&done), ["templates/lib.rs.orig"]);
    assert_eq!(read(&files, "templates/lib.rs.orig"), None);
    assert_put_back(&f, "main").await;
    assert_eq!(disk(&f.root, "templates/lib.rs.orig").unwrap(), LIB);
    assert_eq!(
        std::fs::read(&outside).unwrap(),
        b"outside the repository\n"
    );
}
