mod harness;

use serde_json::json;
use std::sync::{Arc, Mutex};
use zend_tools::{SubagentRequest, SubagentResponse, SubagentRunner, ToolContext};
use zend_vfs::RepoFiles;

// ── Reusable mock runners ─────────────────────────────────────────────────────

/// Answers a part under `parent/name` that ended `done` with its prompt.
struct EchoRunner;
impl SubagentRunner for EchoRunner {
    fn run(&self, parent: &str, req: SubagentRequest) -> Result<SubagentResponse, String> {
        Ok(SubagentResponse {
            path: format!("{parent}/{}", req.name),
            state: "done".into(),
            result: format!("echo: {}", req.prompt),
        })
    }
}

struct FailRunner;
impl SubagentRunner for FailRunner {
    fn run(&self, _parent: &str, _req: SubagentRequest) -> Result<SubagentResponse, String> {
        Err("model crashed".into())
    }
}

/// Keeps the parent and request it was called with.
#[derive(Default)]
struct CapturingRunner {
    captured: Mutex<Option<(String, String, String)>>,
}

impl SubagentRunner for CapturingRunner {
    fn run(&self, parent: &str, req: SubagentRequest) -> Result<SubagentResponse, String> {
        *self.captured.lock().unwrap() = Some((parent.to_string(), req.name, req.prompt));
        Ok(SubagentResponse {
            path: "T:t/w/p".into(),
            state: "stuck".into(),
            result: "captured".into(),
        })
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

#[test]
fn a_context_with_no_runner_cannot_start_parts() {
    let resp = harness::invoke("sub_run", json!({"name": "a", "prompt": "do something"}));
    harness::expect_error(&resp, "not_configured");
}

#[test]
fn the_parts_path_state_and_result_are_the_response() {
    let ctx = harness::granted().with_subagent_runner(Arc::new(EchoRunner));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"name": "read-parser", "prompt": "test task"}),
        &ctx,
    ));
    assert_eq!(
        resp,
        json!({"path": "/read-parser", "state": "done", "result": "echo: test task"})
    );
}

#[test]
fn a_runner_failure_is_the_tools_error_with_its_message() {
    let ctx = harness::granted().with_subagent_runner(Arc::new(FailRunner));
    let resp = harness::invoke_with_ctx("sub_run", json!({"name": "a", "prompt": "fail"}), &ctx);
    let detail = harness::expect_error(&resp, "subagent_failed");
    assert!(detail.contains("model crashed"), "detail was: {detail}");
}

/// **The part hangs under the calling conversation**: the runner is told the
/// conversation the context's calls belong to, with the call's name and prompt
/// as given.
#[test]
fn the_runner_is_given_the_calling_conversation_and_the_request() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = harness::granted()
        .with_files(Arc::new(RepoFiles::detached()), "mcp:o:review/verify")
        .with_subagent_runner(runner.clone());
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"name": "check-plan", "prompt": "check the plan"}),
        &ctx,
    ));
    assert_eq!(resp["state"], "stuck");
    assert_eq!(
        runner.captured.lock().unwrap().clone().unwrap(),
        (
            "mcp:o:review/verify".to_string(),
            "check-plan".to_string(),
            "check the plan".to_string()
        )
    );
}

// ── Validation ────────────────────────────────────────────────────────────────

#[test]
fn an_empty_name_or_prompt_is_refused_before_the_runner() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = harness::granted().with_subagent_runner(runner.clone());
    for args in [
        json!({"name": "", "prompt": "x"}),
        json!({"name": "a", "prompt": ""}),
        json!({"prompt": "x"}),
        json!({"name": "a"}),
        json!({"name": "a".repeat(41), "prompt": "x"}),
    ] {
        let resp = harness::invoke_with_ctx("sub_run", args.clone(), &ctx);
        harness::expect_error(&resp, "invalid_arguments");
    }
    assert!(runner.captured.lock().unwrap().is_none());
}

/// A resumed turn issues `sub_run` again: the runner finds the part it
/// started by name, so running the call twice starts it once.
#[test]
fn a_part_call_is_safe_to_issue_again_after_a_restart() {
    assert_eq!(
        zend_tools::replay("sub_run", &json!({"name": "a", "prompt": "x"})),
        zend_tools::Replay::Safe
    );
}

#[test]
fn contexts_keep_their_own_runners() {
    struct IdRunner {
        id: &'static str,
    }
    impl SubagentRunner for IdRunner {
        fn run(&self, _parent: &str, _req: SubagentRequest) -> Result<SubagentResponse, String> {
            Ok(SubagentResponse {
                path: String::new(),
                state: "done".into(),
                result: self.id.into(),
            })
        }
    }
    let ctx_a: ToolContext =
        harness::granted().with_subagent_runner(Arc::new(IdRunner { id: "runner_A" }));
    let ctx_b = harness::granted().with_subagent_runner(Arc::new(IdRunner { id: "runner_B" }));
    let args = json!({"name": "a", "prompt": "x"});
    let a = harness::expect_success(harness::invoke_with_ctx("sub_run", args.clone(), &ctx_a));
    let b = harness::expect_success(harness::invoke_with_ctx("sub_run", args, &ctx_b));
    assert_eq!(a["result"], "runner_A");
    assert_eq!(b["result"], "runner_B");
}
