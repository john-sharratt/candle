mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

#[test]
fn file_write_unicode() {
    let ctx = ctx();
    let content = "こんにちは 🌍 — Unicode test";
    harness::invoke_with_ctx("file_write", json!({"path": "uni.txt", "content": content}), &ctx);
    let resp =
        harness::expect_success(harness::invoke_with_ctx("file_read", json!({"path": "uni.txt"}), &ctx));
    assert_eq!(resp["content"].as_str().unwrap(), content);
}

#[test]
fn file_edit_not_found() {
    let resp = harness::invoke(
        "file_edit",
        json!({"path": "nonexistent.txt", "old_str": "x", "new_str": "y"}),
    );
    harness::expect_error(&resp, "not_found");
}

#[test]
fn file_list_prefix() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "alpha/a.txt", "content": "1"}), &ctx);
    harness::invoke_with_ctx("file_write", json!({"path": "alpha/b.txt", "content": "2"}), &ctx);
    harness::invoke_with_ctx("file_write", json!({"path": "beta/c.txt", "content": "3"}), &ctx);
    let resp =
        harness::expect_success(harness::invoke_with_ctx("file_list", json!({"prefix": "alpha/"}), &ctx));
    let files = resp["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    for f in files {
        assert!(f["path"].as_str().unwrap().starts_with("alpha/"));
    }
}

#[test]
fn file_delete_idempotent() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "idem.txt", "content": "x"}), &ctx);
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "idem.txt"}),
        &ctx,
    ));
    let r2 = harness::invoke_with_ctx("file_delete", json!({"path": "idem.txt"}), &ctx);
    harness::expect_error(&r2, "not_found");
}

#[test]
fn file_write_overwrite_created_false() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx(
        "file_write",
        json!({"path": "ow.txt", "content": "v1"}),
        &ctx,
    ));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx(
        "file_write",
        json!({"path": "ow.txt", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(r2["created"], false);
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "ow.txt"}),
        &ctx,
    ));
    assert_eq!(rd["content"], "v2");
}

#[test]
fn file_edit_round_trip() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "file_write",
        json!({"path": "rt.txt", "content": "hello world"}),
        &ctx,
    );
    harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"path": "rt.txt", "old_str": "world", "new_str": "Rust"}),
        &ctx,
    ));
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "rt.txt"}),
        &ctx,
    ));
    assert_eq!(rd["content"], "hello Rust");
}

#[test]
fn file_write_read_roundtrip() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "hello.txt", "content": "hello world"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx("file_read", json!({"path": "hello.txt"}), &ctx));
    assert_eq!(resp["content"], "hello world");
    assert_eq!(resp["lines"].as_u64().unwrap(), 1);
}

#[test]
fn file_write_creates_vs_overwrites() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx("file_write", json!({"path": "a.txt", "content": "v1"}), &ctx));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx("file_write", json!({"path": "a.txt", "content": "v2"}), &ctx));
    assert_eq!(r2["created"], false);
}

#[test]
fn file_read_not_found() {
    let resp = harness::invoke("file_read", json!({"path": "nosuchfile.txt"}));
    harness::expect_error(&resp, "not_found");
}

#[test]
fn file_edit_success() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "edit.txt", "content": "foo bar baz"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx("file_edit", json!({
        "path": "edit.txt",
        "old_str": "bar",
        "new_str": "qux"
    }), &ctx));
    assert!(resp["bytes"].as_u64().unwrap() > 0);
    let read = harness::expect_success(harness::invoke_with_ctx("file_read", json!({"path": "edit.txt"}), &ctx));
    assert_eq!(read["content"], "foo qux baz");
}

#[test]
fn file_edit_ambiguous() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "dup.txt", "content": "aa bb aa"}), &ctx);
    let resp = harness::invoke_with_ctx("file_edit", json!({
        "path": "dup.txt",
        "old_str": "aa",
        "new_str": "xx"
    }), &ctx);
    harness::expect_error(&resp, "ambiguous");
}

#[test]
fn file_list() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "a/b.txt", "content": "1"}), &ctx);
    harness::invoke_with_ctx("file_write", json!({"path": "a/c.txt", "content": "2"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx("file_list", json!({"prefix": "a/"}), &ctx));
    let files = resp["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
}

#[test]
fn file_delete() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "del.txt", "content": "bye"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx("file_delete", json!({"path": "del.txt"}), &ctx));
    assert_eq!(resp["deleted"], true);
    let r2 = harness::invoke_with_ctx("file_delete", json!({"path": "del.txt"}), &ctx);
    harness::expect_error(&r2, "not_found");
}

#[test]
fn file_present_found_and_missing() {
    let ctx = ctx();
    harness::invoke_with_ctx("file_write", json!({"path": "p.txt", "content": "hi"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx("file_present", json!({
        "paths": ["p.txt", "missing.txt"]
    }), &ctx));
    let presented = resp["presented"].as_array().unwrap();
    assert_eq!(presented.len(), 1);
    let missing = resp["missing"].as_array().unwrap();
    assert_eq!(missing.len(), 1);
}

#[test]
fn file_present_all_missing() {
    let resp = harness::invoke("file_present", json!({"paths": ["nope.txt"]}));
    harness::expect_error(&resp, "no_files_found");
}
