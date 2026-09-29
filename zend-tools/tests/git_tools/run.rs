//! `run_command` and `run_output`: a program run in the repository's sandbox,
//! on the conversation's branch with its uncommitted changes, what it changes
//! coming back into them.

use serde_json::{json, Value};

use crate::harness::{branch_rev, GitWorkspace};

/// The platform's shell, as the program a test runs.
fn shell() -> &'static str {
    if cfg!(windows) {
        "cmd"
    } else {
        "sh"
    }
}

/// `run_command`'s arguments for `script` in the platform's shell.
fn script(script: &str) -> Value {
    let args: Vec<&str> = if cfg!(windows) {
        vec!["/D", "/C", script]
    } else {
        vec!["-c", script]
    };
    json!({"repo": "app", "program": shell(), "args": args})
}

/// **A program runs over the conversation's own files, and what it writes
/// comes back into them** — while the repository's folder is left as it was.
#[test]
fn a_program_sees_the_conversations_files_and_keeps_what_it_writes() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&[shell()]);
    conv.write("input.txt", "from the conversation\n");
    let line = if cfg!(windows) {
        "type input.txt& echo made> out.txt"
    } else {
        "cat input.txt; echo made > out.txt"
    };
    let out = conv.call("run_command", script(line));
    assert_eq!(out["exit_code"], 0, "{out}");
    assert_eq!(out["timed_out"], false);
    assert_eq!(
        out["changed"],
        json!([{"path": "out.txt", "change": "written"}]),
        "{out}"
    );
    assert_eq!(
        out["output"]["text"].as_str().unwrap().trim_end(),
        "from the conversation"
    );
    assert_eq!(conv.read("out.txt").unwrap().trim_end(), "made");
    assert!(
        !ws.repo_dir().join("out.txt").exists(),
        "the folder is back"
    );
    assert!(
        !ws.repo_dir().join("input.txt").exists(),
        "the folder is back"
    );
    // The written file is uncommitted work like any other.
    let status = conv.status();
    let changed: Vec<&str> = status["changes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["path"].as_str().unwrap())
        .collect();
    assert_eq!(changed, ["input.txt", "out.txt"], "{status}");
}

/// **Long output comes back a page at a time**: the run returns page 0,
/// and `run_output` the rest.
#[test]
fn long_output_pages() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&[shell()]);
    let lines = if cfg!(windows) {
        "for /L %i in (1,1,250) do @echo line %i"
    } else {
        "for i in $(seq 1 250); do echo line $i; done"
    };
    let out = conv.call("run_command", script(lines));
    assert_eq!(out["exit_code"], 0, "{out}");
    let first = &out["output"];
    assert_eq!(
        (&first["page"], &first["pages"], &first["total_lines"]),
        (&json!(0), &json!(2), &json!(250))
    );
    let job = out["job"].as_str().unwrap();
    let rest = conv.call("run_output", json!({"repo": "app", "job": job, "page": 1}));
    assert_eq!(rest["status"], "exited", "{rest}");
    assert_eq!(rest["exit_code"], 0);
    assert_eq!(
        (&rest["output"]["first_line"], &rest["output"]["last_line"]),
        (&json!(201), &json!(250))
    );
    assert!(rest["output"]["text"]
        .as_str()
        .unwrap()
        .ends_with("line 250"));
}

/// **git is refused towards the git tools**, and a program the policy does
/// not list never runs.
#[test]
fn git_and_unlisted_programs_are_refused() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&[shell()]);
    let git = conv.call(
        "run_command",
        json!({"repo": "app", "program": "git", "args": ["status"]}),
    );
    assert_eq!(git["error"], "refused", "{git}");
    assert!(git["detail"].as_str().unwrap().contains("git_status"));
    // A git command line goes straight to the same refusal, not to one that
    // asks for it taken apart first.
    let line = conv.call(
        "run_command",
        json!({"repo": "app", "program": "git log -1", "args": []}),
    );
    assert_eq!(line["error"], "refused", "{line}");
    assert!(
        line["detail"].as_str().unwrap().contains("git_log"),
        "{line}"
    );
    let python = conv.call(
        "run_command",
        json!({"repo": "app", "program": "python", "args": []}),
    );
    assert_eq!(python["error"], "refused", "{python}");
}

/// **A conversation behind its branch is sent to merge first**: a checkout
/// of the branch is not what its changes were made on.
#[test]
fn a_conversation_behind_its_branch_is_sent_to_merge() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&[shell()]);
    // The first read pins the conversation's base; another writer then moves
    // the branch on.
    assert!(conv.read("README.md").is_some());
    ws.write_worktree("other.txt", "someone else\n");
    ws.commit_all("someone else's commit");
    let out = conv.call("run_command", script("echo hi"));
    assert_eq!(out["error"], "behind", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("merge"), "{out}");
}

/// **Merge a branch, publish it, then test it**: a fast-forward merge of
/// another branch puts the conversation's files past its branch, a command is
/// refused towards git_commit, and `from: all_changes` with no change of the
/// conversation's own moves the branch onto the merged commit — after which
/// the command runs.
#[test]
fn a_merged_branch_is_published_before_a_command_runs() {
    let ws = GitWorkspace::new();
    ws.git(&["switch", "-q", "-c", "feature"]);
    ws.write_worktree("feature.txt", "restock\n");
    let feature = ws.commit_all("add the feature");
    ws.git(&["switch", "-q", "main"]);
    let conv = ws.conversation_running(&[shell()]);
    let merged = conv.call(
        "git_merge",
        json!({"repo": "app", "from": branch_rev("feature")}),
    );
    assert_eq!(merged["merged"], "fast_forward", "{merged}");
    assert!(merged["next"].as_str().unwrap().contains("git_commit"));

    let refused = conv.call("run_command", script("echo hi"));
    assert_eq!(refused["error"], "unpublished", "{refused}");
    assert!(
        refused["detail"].as_str().unwrap().contains("git_commit"),
        "{refused}"
    );

    let main_before = ws.oid("main");
    let published = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "all_changes", "message": "merge the feature"}),
    );
    assert_eq!(published["applied"], true, "{published}");
    assert_eq!(published["commit"], feature.as_str(), "{published}");
    assert_eq!(
        published["note"],
        format!(
            "no new commit was needed: main moved from {main_before} to {feature}, the commit \
             a merge brought your files onto"
        )
    );
    assert_eq!(ws.oid("main"), feature, "no commit of its own was made");

    let ran = conv.call("run_command", script("echo hi"));
    assert_eq!(ran["exit_code"], 0, "{ran}");
    assert_eq!(ran["output"]["text"].as_str().unwrap().trim_end(), "hi");
}

/// **A command line in `program` is refused with the call it should have
/// been**, and nothing runs.
#[test]
fn a_command_line_as_the_program_is_taken_apart_in_the_refusal() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&["npm"]);
    let out = conv.call(
        "run_command",
        json!({"repo": "app", "program": "npm test", "args": ["--", "-x"]}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert_eq!(
        out["detail"],
        "`npm test` is a program and its arguments together; run_command takes them apart — \
         call it again with program \"npm\" and args [\"test\", \"--\", \"-x\"]"
    );
    let missing = conv.call("run_command", json!({"repo": "app", "program": "npm"}));
    assert_eq!(
        missing["error"], "invalid_arguments",
        "args is required: {missing}"
    );
}

/// **Shell syntax in `args` is refused with the call it should have been**
/// — an operator, or a value quoted as a shell would quote it — while an
/// escaped quote inside code is the code's own and runs.
#[test]
fn shell_syntax_in_the_arguments_is_refused() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&["npm", shell()]);
    let chained = conv.call(
        "run_command",
        json!({"repo": "app", "program": "npm", "args": ["test", "&&", "npm", "run", "lint"]}),
    );
    assert_eq!(chained["error"], "invalid_arguments", "{chained}");
    assert!(
        chained["detail"]
            .as_str()
            .unwrap()
            .starts_with("`&&` is shell syntax"),
        "{chained}"
    );
    let quoted = conv.call(
        "run_command",
        json!({"repo": "app", "program": "npm",
               "args": ["pkg", "set", "scripts.lint=\\\"node --check src/index.js\\\""]}),
    );
    assert_eq!(quoted["error"], "invalid_arguments", "{quoted}");
    assert!(
        quoted["detail"]
            .as_str()
            .unwrap()
            .contains(r#"here "scripts.lint=node --check src/index.js""#),
        "{quoted}"
    );
    let code = if cfg!(windows) {
        "echo a \\\"b\\\""
    } else {
        "echo 'a \\\"b\\\"'"
    };
    let ran = conv.call("run_command", script(code));
    assert_eq!(ran["exit_code"], 0, "{ran}");
}

/// A job id that is not one is bad arguments; one the sandbox never ran is
/// not found.
#[test]
fn run_output_needs_a_real_job() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation_running(&[shell()]);
    let bad = conv.call(
        "run_output",
        json!({"repo": "app", "job": "../../etc", "page": 0}),
    );
    assert_eq!(bad["error"], "invalid_arguments", "{bad}");
    let unknown = conv.call(
        "run_output",
        json!({"repo": "app", "job": "AAAAAAAAAAA", "page": 0}),
    );
    assert_eq!(unknown["error"], "not_found", "{unknown}");
}
