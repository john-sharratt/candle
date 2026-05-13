mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

#[test]
fn hash_compute_sha512() {
    let resp = harness::expect_success(harness::invoke(
        "hash_compute",
        json!({"algorithm": "sha512", "data": "hello", "data_encoding": "text"}),
    ));
    let digest = resp["digest"].as_str().unwrap();
    assert_eq!(digest.len(), 128, "sha512 should be 128 hex chars, got {}", digest.len());
}

#[test]
fn hash_compute_sha3_256() {
    let resp = harness::expect_success(harness::invoke(
        "hash_compute",
        json!({"algorithm": "sha3_256", "data": "hello", "data_encoding": "text"}),
    ));
    let digest = resp["digest"].as_str().unwrap();
    assert_eq!(digest.len(), 64, "sha3_256 should be 64 hex chars");
}

#[test]
fn hash_compute_blake3() {
    let resp = harness::expect_success(harness::invoke(
        "hash_compute",
        json!({"algorithm": "blake3", "data": "hello", "data_encoding": "text"}),
    ));
    let digest = resp["digest"].as_str().unwrap();
    assert_eq!(digest.len(), 64, "blake3 default output should be 64 hex chars");
}

#[test]
fn hash_state_multi_update() {
    let ctx = ctx();
    // Init sha256 state — response field is "id"
    let init = harness::expect_success(harness::invoke_with_ctx(
        "hash_state_init",
        json!({"algorithm": "sha256"}),
        &ctx,
    ));
    let state_id = init["id"].as_str().unwrap().to_string();

    // Update with "ab"
    harness::expect_success(harness::invoke_with_ctx(
        "hash_state_update",
        json!({"id": state_id, "data": "ab", "data_encoding": "text"}),
        &ctx,
    ));

    // Update with "cd"
    harness::expect_success(harness::invoke_with_ctx(
        "hash_state_update",
        json!({"id": state_id, "data": "cd", "data_encoding": "text"}),
        &ctx,
    ));

    // Finalize
    let fin = harness::expect_success(harness::invoke_with_ctx(
        "hash_state_finalize",
        json!({"id": state_id}),
        &ctx,
    ));
    let digest = fin["digest"].as_str().unwrap();

    // SHA256("abcd") = 88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589
    assert_eq!(
        digest,
        "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"
    );
}

#[test]
fn hash_state_keep_true() {
    let ctx = ctx();
    let init = harness::expect_success(harness::invoke_with_ctx(
        "hash_state_init",
        json!({"algorithm": "sha256"}),
        &ctx,
    ));
    let state_id = init["id"].as_str().unwrap().to_string();
    harness::invoke_with_ctx(
        "hash_state_update",
        json!({"id": state_id, "data": "hello", "data_encoding": "text"}),
        &ctx,
    );
    let fin1 = harness::expect_success(harness::invoke_with_ctx(
        "hash_state_finalize",
        json!({"id": state_id, "keep": true}),
        &ctx,
    ));
    let fin2 = harness::expect_success(harness::invoke_with_ctx(
        "hash_state_finalize",
        json!({"id": state_id, "keep": true}),
        &ctx,
    ));
    assert_eq!(fin1["digest"], fin2["digest"]);
}

#[test]
fn hash_compute_sha256() {
    let resp = harness::expect_success(harness::invoke("hash_compute", json!({
        "algorithm": "sha256",
        "data": "hello",
        "data_encoding": "text"
    })));
    // SHA256 of "hello"
    assert_eq!(
        resp["digest"].as_str().unwrap(),
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn hash_compute_md5() {
    let resp = harness::expect_success(harness::invoke("hash_compute", json!({
        "algorithm": "md5",
        "data": "hello",
        "data_encoding": "text"
    })));
    assert_eq!(resp["digest"].as_str().unwrap(), "5d41402abc4b2a76b9719d911017c592");
}

#[test]
fn hash_compute_base64_output() {
    let resp = harness::expect_success(harness::invoke("hash_compute", json!({
        "algorithm": "sha256",
        "data": "hello",
        "data_encoding": "text",
        "output_encoding": "base64"
    })));
    assert_eq!(resp["output_encoding"], "base64");
    // Should be valid base64
    assert!(resp["digest"].as_str().unwrap().len() > 0);
}

#[test]
fn hash_compute_unknown_algorithm() {
    let resp = harness::invoke("hash_compute", json!({
        "algorithm": "md2",
        "data": "hello",
        "data_encoding": "text"
    }));
    harness::expect_error(&resp, "unknown_algorithm");
}

#[test]
fn hash_scan_finds_sha256() {
    // SHA256("hello") = 2cf24dba...
    let resp = harness::expect_success(harness::invoke("hash_scan", json!({
        "data": "hello",
        "data_encoding": "text",
        "known_hash": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        "hash_encoding": "hex"
    })));
    assert_eq!(resp["matches"], true);
    assert_eq!(resp["algorithm"], "sha256");
}

#[test]
fn hash_scan_no_match() {
    let resp = harness::invoke("hash_scan", json!({
        "data": "hello",
        "data_encoding": "text",
        "known_hash": "deadbeefdeadbeefdeadbeefdeadbeef",
        "hash_encoding": "hex"
    }));
    harness::expect_error(&resp, "no_match");
}
