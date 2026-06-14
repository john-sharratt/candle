mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn make_sql_cred(ctx: &ToolContext) -> &'static str {
    harness::expect_success(harness::invoke_with_ctx(
        "credential_save",
        json!({
            "name": "test-sql",
            "type": "sql_password",
            "username": "testuser",
            "secret": "testpass"
        }),
        ctx,
    ));
    "test-sql"
}

#[test]
fn sql_session_create_table_and_query() {
    let ctx = ToolContext::new();
    let cred = make_sql_cred(&ctx);

    let open = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": cred, "database": ":memory:"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();
    assert_eq!(open["driver"], "sqlite");

    harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"}),
        &ctx,
    ));

    let ins = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "INSERT INTO t (name) VALUES ('alice')"}),
        &ctx,
    ));
    assert_eq!(ins["rows_affected"], 1);
    assert_eq!(ins["last_insert_rowid"], 1);

    let sel = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "SELECT id, name FROM t"}),
        &ctx,
    ));
    let rows = sel["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][1], "alice");
    assert_eq!(sel["columns"][0], "id");
    assert_eq!(sel["columns"][1], "name");
}

#[test]
fn sql_session_open_list_close() {
    let ctx = ToolContext::new();
    let cred = make_sql_cred(&ctx);

    let open = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": cred, "database": ":memory:"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();

    let list = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);

    harness::expect_success(harness::invoke_with_ctx(
        "sql_session_close",
        json!({"session_id": sid}),
        &ctx,
    ));

    let list2 = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list2["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn sql_session_parameterized_query() {
    let ctx = ToolContext::new();
    let cred = make_sql_cred(&ctx);

    let open = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": cred, "database": ":memory:"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap();

    harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "CREATE TABLE vals (x INTEGER, y TEXT)"}),
        &ctx,
    ));

    let ins = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "INSERT INTO vals VALUES (?, ?)", "params": [42, "hello"]}),
        &ctx,
    ));
    assert_eq!(ins["rows_affected"], 1);

    let sel = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "SELECT y FROM vals WHERE x = ?", "params": [42]}),
        &ctx,
    ));
    assert_eq!(sel["rows"].as_array().unwrap()[0][0], "hello");
}

#[test]
fn sql_session_not_found() {
    let resp = harness::invoke(
        "sql_session_query",
        json!({"session_id": "sess_nonexistent", "query": "SELECT 1"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn sql_session_missing_credential() {
    let resp = harness::invoke("sql_session_open", json!({"credential_name": "cred_bogus"}));
    harness::expect_error(&resp, "credential_not_found");
}

#[test]
fn sql_session_wrong_credential_type() {
    let ctx = ToolContext::new();
    harness::expect_success(harness::invoke_with_ctx(
        "credential_save",
        json!({"name": "api-k", "type": "api_key", "secret": "xyz"}),
        &ctx,
    ));
    let resp = harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": "api-k", "database": ":memory:"}),
        &ctx,
    );
    harness::expect_error(&resp, "invalid_credential_type");
}

#[test]
fn sql_session_close_nonexistent_returns_false() {
    let resp = harness::expect_success(harness::invoke(
        "sql_session_close",
        json!({"session_id": "sess_nonexistent"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn sql_session_multiple_rows() {
    let ctx = ToolContext::new();
    let cred = make_sql_cred(&ctx);

    let open = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": cred, "database": ":memory:"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap();

    harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "CREATE TABLE items (val TEXT)"}),
        &ctx,
    ));
    for i in 0..5 {
        harness::expect_success(harness::invoke_with_ctx(
            "sql_session_query",
            json!({"session_id": sid, "query": "INSERT INTO items VALUES (?)", "params": [i.to_string()]}),
            &ctx,
        ));
    }

    let sel = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "SELECT val FROM items ORDER BY val"}),
        &ctx,
    ));
    assert_eq!(sel["rows"].as_array().unwrap().len(), 5);
}

#[test]
fn sql_session_syntax_error() {
    let ctx = ToolContext::new();
    let cred = make_sql_cred(&ctx);
    let open = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": cred, "database": ":memory:"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap();

    let resp = harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "NOT VALID SQL @@##"}),
        &ctx,
    );
    harness::expect_error(&resp, "query_failed");
}

#[test]
fn sql_session_null_params() {
    let ctx = ToolContext::new();
    let cred = make_sql_cred(&ctx);
    let open = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_open",
        json!({"credential_name": cred, "database": ":memory:"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap();

    harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "CREATE TABLE n (x TEXT)"}),
        &ctx,
    ));
    let ins = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "INSERT INTO n VALUES (?)", "params": [null]}),
        &ctx,
    ));
    assert_eq!(ins["rows_affected"], 1);

    let sel = harness::expect_success(harness::invoke_with_ctx(
        "sql_session_query",
        json!({"session_id": sid, "query": "SELECT x FROM n"}),
        &ctx,
    ));
    assert!(sel["rows"].as_array().unwrap()[0][0].is_null());
}
