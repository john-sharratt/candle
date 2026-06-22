mod harness;

use serde_json::json;
use std::sync::{Arc, Mutex};
use zend_tools::{SubagentRequest, SubagentResponse, SubagentRunner, ToolContext};

// ── Reusable mock runners ─────────────────────────────────────────────────────

struct EchoRunner;
impl SubagentRunner for EchoRunner {
    fn run(&self, req: SubagentRequest) -> Result<SubagentResponse, String> {
        Ok(SubagentResponse {
            result: format!("echo: {}", req.instruction),
            turns: 1,
            tool_calls_made: 0,
        })
    }
}

struct FailRunner;
impl SubagentRunner for FailRunner {
    fn run(&self, _req: SubagentRequest) -> Result<SubagentResponse, String> {
        Err("model crashed".into())
    }
}

#[derive(Default)]
struct CapturingRunner {
    captured: Mutex<Option<SubagentRequest>>,
}

impl SubagentRunner for CapturingRunner {
    fn run(&self, req: SubagentRequest) -> Result<SubagentResponse, String> {
        *self.captured.lock().unwrap() = Some(req);
        Ok(SubagentResponse {
            result: "captured".into(),
            turns: 2,
            tool_calls_made: 5,
        })
    }
}

// ── Basic lifecycle ───────────────────────────────────────────────────────────

#[test]
fn subagent_not_configured() {
    let resp = harness::invoke("sub_run", json!({"instruction": "do something"}));
    harness::expect_error(&resp, "not_configured");
}

#[test]
fn subagent_with_mock_runner() {
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(EchoRunner));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "test task"}),
        &ctx,
    ));
    assert!(resp["result"].as_str().unwrap().contains("test task"));
    assert_eq!(resp["turns"], 1);
    assert_eq!(resp["tool_calls_made"], 0);
}

#[test]
fn subagent_runner_failed() {
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(FailRunner));
    let resp = harness::invoke_with_ctx("sub_run", json!({"instruction": "fail please"}), &ctx);
    harness::expect_error(&resp, "subagent_failed");
}

#[test]
fn subagent_runner_error_detail_contains_message() {
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(FailRunner));
    let resp = harness::invoke_with_ctx("sub_run", json!({"instruction": "trigger failure"}), &ctx);
    let detail = harness::expect_error(&resp, "subagent_failed");
    assert!(detail.contains("model crashed"), "detail was: {detail}");
}

// ── Default and custom max_turns ──────────────────────────────────────────────

#[test]
fn subagent_max_turns_default() {
    static CAPTURED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    struct TurnsCapture;
    impl SubagentRunner for TurnsCapture {
        fn run(&self, req: SubagentRequest) -> Result<SubagentResponse, String> {
            CAPTURED.store(req.max_turns, std::sync::atomic::Ordering::Relaxed);
            Ok(SubagentResponse {
                result: "ok".into(),
                turns: 1,
                tool_calls_made: 0,
            })
        }
    }

    let ctx = ToolContext::new().with_subagent_runner(Arc::new(TurnsCapture));
    harness::invoke_with_ctx("sub_run", json!({"instruction": "x"}), &ctx);
    assert_eq!(CAPTURED.load(std::sync::atomic::Ordering::Relaxed), 10);
}

#[test]
fn subagent_custom_max_turns() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "short task", "max_turns": 3}),
        &ctx,
    );
    let captured = runner.captured.lock().unwrap();
    assert_eq!(captured.as_ref().unwrap().max_turns, 3);
}

// ── Instruction validation ────────────────────────────────────────────────────

#[test]
fn subagent_empty_instruction_rejected() {
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(EchoRunner));
    let resp = harness::invoke_with_ctx("sub_run", json!({"instruction": ""}), &ctx);
    harness::expect_error(&resp, "invalid_arguments");
}

#[test]
fn subagent_instruction_too_long_accepted_by_validator() {
    // validator only checks min=1; long instructions should succeed
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(EchoRunner));
    let long = "x".repeat(4096);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": long}),
        &ctx,
    ));
    assert!(resp["result"].as_str().is_some());
}

// ── Optional parameter passthrough ───────────────────────────────────────────

#[test]
fn subagent_tool_filter_passed_to_runner() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "use files", "tools": ["write", "file_read"]}),
        &ctx,
    );
    let captured = runner.captured.lock().unwrap();
    let tools = captured.as_ref().unwrap().tools.as_ref().unwrap();
    assert_eq!(tools, &["write", "file_read"]);
}

#[test]
fn subagent_default_tools_is_none() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "no tools specified"}),
        &ctx,
    );
    let captured = runner.captured.lock().unwrap();
    assert!(captured.as_ref().unwrap().tools.is_none());
}

#[test]
fn subagent_model_override_passed_to_runner() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "use big model", "model": "claude-opus-4-7"}),
        &ctx,
    );
    let captured = runner.captured.lock().unwrap();
    assert_eq!(
        captured.as_ref().unwrap().model.as_deref(),
        Some("claude-opus-4-7")
    );
}

#[test]
fn subagent_default_model_is_none() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    harness::invoke_with_ctx("sub_run", json!({"instruction": "default model"}), &ctx);
    let captured = runner.captured.lock().unwrap();
    assert!(captured.as_ref().unwrap().model.is_none());
}

#[test]
fn subagent_endpoint_override_passed_to_runner() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "custom endpoint", "endpoint": "https://custom.api/v1"}),
        &ctx,
    );
    let captured = runner.captured.lock().unwrap();
    assert_eq!(
        captured.as_ref().unwrap().endpoint.as_deref(),
        Some("https://custom.api/v1"),
    );
}

// ── Response field verification ───────────────────────────────────────────────

#[test]
fn subagent_result_is_string() {
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(EchoRunner));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "return string"}),
        &ctx,
    ));
    assert!(resp["result"].is_string());
}

#[test]
fn subagent_turns_and_tool_calls_present() {
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(EchoRunner));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "check fields"}),
        &ctx,
    ));
    assert!(resp["turns"].is_number());
    assert!(resp["tool_calls_made"].is_number());
}

#[test]
fn subagent_many_tool_calls_propagated() {
    struct HighUsageRunner;
    impl SubagentRunner for HighUsageRunner {
        fn run(&self, _req: SubagentRequest) -> Result<SubagentResponse, String> {
            Ok(SubagentResponse {
                result: "done".into(),
                turns: 7,
                tool_calls_made: 50,
            })
        }
    }
    let ctx = ToolContext::new().with_subagent_runner(Arc::new(HighUsageRunner));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "complex task"}),
        &ctx,
    ));
    assert_eq!(resp["turns"], 7);
    assert_eq!(resp["tool_calls_made"], 50);
}

// ── Context isolation ─────────────────────────────────────────────────────────

#[test]
fn subagent_contexts_are_independent() {
    struct IdRunner {
        id: &'static str,
    }
    impl SubagentRunner for IdRunner {
        fn run(&self, _req: SubagentRequest) -> Result<SubagentResponse, String> {
            Ok(SubagentResponse {
                result: self.id.into(),
                turns: 1,
                tool_calls_made: 0,
            })
        }
    }

    let ctx_a = ToolContext::new().with_subagent_runner(Arc::new(IdRunner { id: "runner_A" }));
    let ctx_b = ToolContext::new().with_subagent_runner(Arc::new(IdRunner { id: "runner_B" }));

    let a = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "x"}),
        &ctx_a,
    ));
    let b = harness::expect_success(harness::invoke_with_ctx(
        "sub_run",
        json!({"instruction": "x"}),
        &ctx_b,
    ));
    assert_eq!(a["result"], "runner_A");
    assert_eq!(b["result"], "runner_B");
}

#[test]
fn subagent_no_runner_in_default_context() {
    // A fresh ToolContext without with_subagent_runner must return not_configured
    let ctx = ToolContext::new();
    let resp = harness::invoke_with_ctx("sub_run", json!({"instruction": "anything"}), &ctx);
    harness::expect_error(&resp, "not_configured");
}

// ── Instruction echo roundtrip ────────────────────────────────────────────────

#[test]
fn subagent_instruction_roundtrip() {
    let runner = Arc::new(CapturingRunner::default());
    let ctx = ToolContext::new().with_subagent_runner(runner.clone());
    let instruction = "process the dataset and produce a summary";
    harness::invoke_with_ctx("sub_run", json!({"instruction": instruction}), &ctx);
    let captured = runner.captured.lock().unwrap();
    assert_eq!(captured.as_ref().unwrap().instruction, instruction);
}
