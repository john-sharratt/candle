//! `web_search` against the real Tavily API, with the deployment's own key.
//!
//! The live case is `#[ignore]`d because it spends real API quota and needs
//! network egress — the same reason the model forward gates are ignored. Run it
//! deliberately:
//!
//! ```text
//! cargo test -p zend-tools --test web_search_live -- --ignored --nocapture
//! ```
//!
//! It reads the key the daemon reads by default, from `~/.zend/secrets.yaml`,
//! rather than taking one from the test. That is the point: a green run here
//! says *this machine's configured key works against the live API*, which is
//! the thing no amount of parsing coverage can establish.
//!
//! The unconfigured case below is NOT ignored — it reaches no network and holds
//! the contract that matters when a deployment has no key.

mod harness;

use std::sync::Arc;

use serde_json::json;
use zend_tools::state::Secrets;
use zend_tools::{Grants, ToolContext};

/// A context carrying this machine's real key, or a panic naming the file to
/// edit. A live test that silently passed with no key would be worthless.
fn ctx_with_real_key() -> ToolContext {
    let path = Secrets::default_path().expect("this machine reports a home folder");
    let secrets = Secrets::load(&path)
        .unwrap_or_else(|e| panic!("{} could not be read: {e}", path.display()));
    assert!(
        secrets.tavily_api_key().is_some(),
        "no tavily_api_key in {} — this live test needs the deployment's key",
        path.display(),
    );
    // `web_search` never touches `ctx.files`, so no workspace is needed.
    ToolContext::new()
        .with_secrets(Arc::new(secrets))
        .granting(Grants::ALL)
}

/// **The configured key works against the live API.**
///
/// Asserts the shape the model is handed, not just an HTTP 200: every result
/// needs a usable URL and a non-empty title, because a result missing either is
/// one the model cannot act on, and `max_results` has to be honoured or a search
/// can flood a turn.
#[test]
#[ignore = "spends real Tavily API quota and needs network egress"]
fn a_live_search_returns_usable_results() {
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "web_search",
        json!({"query": "Rust programming language memory safety", "max_results": 3}),
        &ctx_with_real_key(),
    ));

    let results = resp["results"]
        .as_array()
        .unwrap_or_else(|| panic!("a results array was expected: {resp}"));
    assert!(
        !results.is_empty(),
        "a live search for a common topic returned nothing: {resp}"
    );
    assert!(
        results.len() <= 3,
        "max_results was not honoured: {} results",
        results.len()
    );
    for r in results {
        let url = r["url"].as_str().unwrap_or_default();
        assert!(
            url.starts_with("http"),
            "every result needs a usable URL: {r}"
        );
        assert!(
            !r["title"].as_str().unwrap_or_default().is_empty(),
            "every result needs a title: {r}"
        );
        assert!(
            r["score"].is_number(),
            "every result carries a relevance score: {r}"
        );
    }
}

/// A second live call with a different shape of query, so a pass is not one
/// lucky request: a natural-language question rather than keywords, at the
/// default `max_results`.
#[test]
#[ignore = "spends real Tavily API quota and needs network egress"]
fn a_live_question_style_query_also_returns_results() {
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "web_search",
        json!({"query": "what is the latest stable Rust release"}),
        &ctx_with_real_key(),
    ));
    let results = resp["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "no results: {resp}");
    assert!(
        results.len() <= 10,
        "the tool caps at ten results, got {}",
        results.len()
    );
}

/// **Unconfigured is a clean refusal that leaks nothing** — no key, and no
/// hint of where the secrets file lives.
///
/// Not ignored: it reaches no network. A context built without secrets is what
/// every test and every non-daemon caller gets, so this is also the assertion
/// that such a caller cannot accidentally spend quota — even granted the
/// network, as this one is.
#[test]
fn without_a_key_the_tool_reports_itself_unconfigured() {
    let resp = harness::invoke("web_search", json!({"query": "anything"}));
    let detail = harness::expect_error(&resp, "search_unavailable");
    assert!(
        detail.contains("no tavily_api_key configured")
            && !detail.contains(Secrets::DEFAULT_RELATIVE_PATH),
        "the refusal says what is missing and never where secrets live: {detail}"
    );
    assert!(
        !detail.contains("tvly-"),
        "a refusal must never echo key material: {detail}"
    );
}
