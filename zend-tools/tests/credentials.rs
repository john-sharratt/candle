mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

#[test]
fn credential_save_http_bearer() {
    let resp = harness::expect_success(harness::invoke(
        "credential_save",
        json!({"name": "bearer-tok", "type": "http_bearer", "secret": "tok123"}),
    ));
    assert_eq!(resp["created"], true);
    let id = resp["id"].as_str().unwrap();
    assert!(!id.is_empty());
}

#[test]
fn credential_save_duplicate_name() {
    let ctx = ctx();
    harness::expect_success(harness::invoke_with_ctx(
        "credential_save",
        json!({"name": "dup-cred", "type": "api_key", "secret": "s1"}),
        &ctx,
    ));
    let resp = harness::invoke_with_ctx(
        "credential_save",
        json!({"name": "dup-cred", "type": "api_key", "secret": "s2"}),
        &ctx,
    );
    harness::expect_error(&resp, "duplicate_name");
}

#[test]
fn credential_list_type_filter() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "credential_save",
        json!({"name": "k1", "type": "api_key", "secret": "s1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "credential_save",
        json!({"name": "k2", "type": "api_key", "secret": "s2"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "credential_save",
        json!({"name": "t1", "type": "totp_secret", "secret": "JBSWY3DPEHPK3PXP"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "credential_list",
        json!({"type": "api_key"}),
        &ctx,
    ));
    let creds = resp["credentials"].as_array().unwrap();
    assert_eq!(creds.len(), 2);
    for c in creds {
        assert_eq!(c["type"], "api_key");
    }
}

#[test]
fn credential_delete_not_found_error() {
    let resp = harness::invoke("credential_delete", json!({"name": "cred_does_not_exist"}));
    harness::expect_error(&resp, "not_found");
}

#[test]
fn credential_save_ssh_key_missing_username() {
    let resp = harness::invoke(
        "credential_save",
        json!({"name": "my-ssh-key", "type": "ssh_key", "secret": "-----BEGIN..."}),
    );
    harness::expect_error(&resp, "missing_field");
}

#[test]
fn credential_save_http_header_missing_header_name() {
    let resp = harness::invoke(
        "credential_save",
        json!({"name": "my-header", "type": "http_header", "secret": "val"}),
    );
    harness::expect_error(&resp, "missing_field");
}

#[test]
fn credential_save_list_delete() {
    let ctx = ctx();
    let saved = harness::expect_success(harness::invoke_with_ctx("credential_save", json!({
        "name": "my-api-key",
        "type": "api_key",
        "secret": "sk-abc123"
    }), &ctx));
    let id = saved["id"].as_str().unwrap().to_string();
    assert!(!id.is_empty());
    assert_eq!(saved["created"], true);

    let list_resp = harness::expect_success(harness::invoke_with_ctx("credential_list", json!({}), &ctx));
    let creds = list_resp["credentials"].as_array().unwrap();
    assert_eq!(creds.len(), 1);
    assert_eq!(creds[0]["name"], "my-api-key");

    let del = harness::expect_success(harness::invoke_with_ctx("credential_delete", json!({"name": "my-api-key"}), &ctx));
    assert_eq!(del["deleted"], true);

    let list2 = harness::expect_success(harness::invoke_with_ctx("credential_list", json!({}), &ctx));
    assert_eq!(list2["credentials"].as_array().unwrap().len(), 0);
}

#[test]
fn credential_delete_not_found() {
    let resp = harness::invoke("credential_delete", json!({"name": "nonexistent"}));
    harness::expect_error(&resp, "not_found");
}

#[test]
fn credential_save_http_header_requires_header_name() {
    let resp = harness::invoke("credential_save", json!({
        "name": "my-header",
        "type": "http_header",
        "secret": "val"
    }));
    harness::expect_error(&resp, "missing_field");
}

#[test]
fn credential_save_ssh_password_requires_username() {
    let resp = harness::invoke("credential_save", json!({
        "name": "my-ssh",
        "type": "ssh_password",
        "secret": "pass"
    }));
    harness::expect_error(&resp, "missing_field");
}

#[test]
fn credential_save_invalid_type_rejected() {
    let resp = harness::invoke("credential_save", json!({
        "name": "bad-type",
        "type": "completely_unknown_type",
        "secret": "s"
    }));
    harness::expect_error(&resp, "invalid_credential_type");
}

#[test]
fn credential_save_ssh_key_invalid_pem_rejected() {
    let resp = harness::invoke("credential_save", json!({
        "name": "bad-key",
        "type": "ssh_key",
        "username": "admin",
        "secret": "not-a-key-just-random-garbage"
    }));
    harness::expect_error(&resp, "invalid_key");
}

#[test]
fn credential_save_ssh_key_valid_pem_accepted() {
    let resp = harness::expect_success(harness::invoke("credential_save", json!({
        "name": "valid-ssh-key",
        "type": "ssh_key",
        "username": "admin",
        "secret": "-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----"
    })));
    assert_eq!(resp["created"], true);
}

#[test]
fn credential_list_by_type() {
    let ctx = ctx();
    harness::invoke_with_ctx("credential_save", json!({
        "name": "key1", "type": "api_key", "secret": "s1"
    }), &ctx);
    harness::invoke_with_ctx("credential_save", json!({
        "name": "totp1", "type": "totp_secret", "secret": "JBSWY3DPEHPK3PXP"
    }), &ctx);

    let resp = harness::expect_success(harness::invoke_with_ctx("credential_list", json!({"type": "api_key"}), &ctx));
    let creds = resp["credentials"].as_array().unwrap();
    assert_eq!(creds.len(), 1);
    assert_eq!(creds[0]["name"], "key1");
}
