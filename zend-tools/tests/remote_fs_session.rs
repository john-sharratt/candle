mod harness;

use serde_json::json;

#[test]
fn remote_fs_session_not_sftp() {
    // The scheme is rejected before the credential is ever looked up, so an
    // unsupported URI reports `not_supported` regardless of the credential.
    let resp = harness::invoke(
        "remote_fs_session_open",
        json!({"uri": "ftp://host/path", "credential_name": "any"}),
    );
    harness::expect_error(&resp, "not_supported");
}

#[test]
fn remote_fs_session_open_missing_credential() {
    let resp = harness::invoke(
        "remote_fs_session_open",
        json!({"uri": "sftp://host/path", "credential_name": "bogus_cred"}),
    );
    harness::expect_error(&resp, "credential_not_found");
}

#[test]
fn remote_fs_session_list_empty() {
    let resp = harness::expect_success(harness::invoke("remote_fs_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn remote_fs_session_close_noop() {
    let resp = harness::expect_success(harness::invoke(
        "remote_fs_session_close",
        json!({"session_id": "x"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn remote_fs_session_nfs_not_supported() {
    let resp = harness::invoke(
        "remote_fs_session_open",
        json!({"uri": "nfs://host/path", "credential_name": "any"}),
    );
    harness::expect_error(&resp, "not_supported");
}

#[test]
fn remote_fs_session_smb_not_supported() {
    let resp = harness::invoke(
        "remote_fs_session_open",
        json!({"uri": "smb://fileserver/share", "credential_name": "any"}),
    );
    harness::expect_error(&resp, "not_supported");
}

#[test]
fn remote_fs_session_close_returns_false_for_missing() {
    let resp = harness::expect_success(harness::invoke(
        "remote_fs_session_close",
        json!({"session_id": "sess_definitely_missing"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn remote_fs_stat_session_not_found() {
    let resp = harness::invoke(
        "remote_fs_session_stat",
        json!({"session_id": "sess_missing", "path": "/etc/passwd"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn remote_fs_list_dir_session_not_found() {
    let resp = harness::invoke(
        "remote_fs_session_list_dir",
        json!({"session_id": "sess_missing", "path": "/"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn remote_fs_get_session_not_found() {
    let resp = harness::invoke(
        "remote_fs_session_get",
        json!({"session_id": "sess_missing", "remote_path": "/etc/passwd"}),
    );
    harness::expect_error(&resp, "session_not_found");
}
