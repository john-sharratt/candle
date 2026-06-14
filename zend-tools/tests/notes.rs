mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

#[test]
fn notes_write_read_roundtrip() {
    let ctx = ctx();
    let wr = harness::expect_success(harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "rt_key", "content": "roundtrip content", "tags": ["foo", "bar"]}),
        &ctx,
    ));
    assert_eq!(wr["created"], true);
    assert_eq!(wr["key"], "rt_key");

    let rd = harness::expect_success(harness::invoke_with_ctx(
        "notes_read",
        json!({"key": "rt_key"}),
        &ctx,
    ));
    assert_eq!(rd["content"], "roundtrip content");
    let tags = rd["tags"].as_array().unwrap();
    assert!(tags.iter().any(|t| t == "foo"));
    assert!(tags.iter().any(|t| t == "bar"));
}

#[test]
fn notes_write_empty_content() {
    // Writing empty content stores an empty note (not deleted)
    let ctx = ctx();
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "empty_key", "content": "data"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "empty_key", "content": ""}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "notes_read",
        json!({"key": "empty_key"}),
        &ctx,
    ));
    // The note is still there but with empty content
    assert_eq!(resp["content"].as_str().unwrap(), "");
    assert_eq!(resp["bytes"], 0);
}

#[test]
fn notes_search_by_query() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "n1", "content": "the quick brown fox"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "n2", "content": "lazy dog"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "notes_search",
        json!({"query": "quick"}),
        &ctx,
    ));
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["key"], "n1");
}

#[test]
fn notes_list_with_prefix() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "proj/a", "content": "1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "proj/b", "content": "2"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "other/c", "content": "3"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "notes_list",
        json!({"prefix": "proj/"}),
        &ctx,
    ));
    let notes = resp["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 2);
}

#[test]
fn notes_overwrite_created_false() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "dup", "content": "v1"}),
        &ctx,
    ));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "dup", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(r2["created"], false);
}

#[test]
fn notes_write_read() {
    let ctx = ctx();
    let wr = harness::expect_success(harness::invoke_with_ctx(
        "notes_write",
        json!({
            "key": "mykey",
            "content": "hello notes",
            "tags": ["test"]
        }),
        &ctx,
    ));
    assert_eq!(wr["created"], true);
    assert_eq!(wr["key"], "mykey");

    let rd = harness::expect_success(harness::invoke_with_ctx(
        "notes_read",
        json!({"key": "mykey"}),
        &ctx,
    ));
    assert_eq!(rd["content"], "hello notes");
    assert_eq!(rd["tags"][0], "test");
}

#[test]
fn notes_read_not_found() {
    let resp = harness::invoke("notes_read", json!({"key": "nonexistent"}));
    harness::expect_error(&resp, "not_found");
}

#[test]
fn notes_search_by_content() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "k1", "content": "apples and oranges"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "k2", "content": "bananas"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "notes_search",
        json!({
            "query": "apples"
        }),
        &ctx,
    ));
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["key"], "k1");
}

#[test]
fn notes_search_no_criteria() {
    let resp = harness::invoke("notes_search", json!({}));
    harness::expect_error(&resp, "no_search_criteria");
}

#[test]
fn notes_list() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "prefix/a", "content": "1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "prefix/b", "content": "2"}),
        &ctx,
    );
    harness::invoke_with_ctx("notes_write", json!({"key": "other", "content": "3"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "notes_list",
        json!({"prefix": "prefix/"}),
        &ctx,
    ));
    let notes = resp["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 2);
}

#[test]
fn notes_update() {
    let ctx = ctx();
    harness::invoke_with_ctx("notes_write", json!({"key": "upd", "content": "v1"}), &ctx);
    let wr2 = harness::expect_success(harness::invoke_with_ctx(
        "notes_write",
        json!({"key": "upd", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(wr2["created"], false);
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "notes_read",
        json!({"key": "upd"}),
        &ctx,
    ));
    assert_eq!(rd["content"], "v2");
}
