//! Static registry of all 93 tool implementations.
//!
//! # How a tool is registered
//!
//! Each tool module defines:
//! ```text
//! pub const MY_TOOL: RegisteredTool = RegisteredTool::new::<MyToolImpl>();
//! ```
//! The const is a zero-cost type-erasure: `RegisteredTool::new` captures three
//! function pointers (`schema`, `run`, `confirmation`) at compile time, erasing
//! the concrete `Request`/`Response`/`Error` types so the registry can hold a
//! flat `&[RegisteredTool]`.
//!
//! # Dispatch pipeline
//!
//! `tool_run::<T>` does the following in order:
//! 1. `serde_json::from_value` — JSON → `T::Request` (returns `invalid_arguments` on failure)
//! 2. `req.validate()` — field constraints (returns `invalid_arguments` with per-field messages)
//! 3. `T::run(ctx, req)` — actual execution
//! 4. `serde_json::to_value(resp)` — serialize (returns `internal_error` only if this fails,
//!    which should never happen for well-formed `Serialize` impls)
//!
//! # Error codes from this layer
//!
//! | Code | Cause |
//! |------|-------|
//! | `invalid_arguments` | Serde parse failure or validator constraint violation |
//! | `unknown_tool` | No entry in [`all_tools`] for the given name |
//! | `internal_error` | Response serialization failure |

use serde_json::{json, Value};
use validator::Validate;

use crate::context::ToolContext;
use crate::tool::{ConfirmationDetails, Tool, ToolError};

/// A type-erased **execution** registration for the static tool table. Tool
/// *definitions* — name (shared here as the execution binding), description, the
/// `parameters` schema, high-risk flag, and calibration examples — live in the
/// bundled `src/prompts/tools/*.yaml` (see `zend::tool_def`); this table carries
/// only what running a call needs.
#[derive(Copy, Clone)]
pub struct RegisteredTool {
    /// The tool's canonical name — the key a model call and a tool definition
    /// bind to this executor by.
    pub name: &'static str,
    /// Dispatch entry — parses, validates, runs the tool, encodes result.
    /// The returned `Value` is what the orchestrator places inside the
    /// `<tool_response>` block (success payload OR an `{"error": ...}` shape).
    pub run: fn(&ToolContext, &Value) -> Value,
    /// Returns confirmation details if this tool wants a user prompt.
    /// Returns `None` if either no confirmation is needed *or* the args
    /// could not be parsed (in which case `run` will surface the error).
    pub confirmation: fn(&Value) -> Option<ConfirmationDetails>,
}

impl RegisteredTool {
    /// Build an execution registration for a `Tool` impl.
    pub const fn new<T: Tool>() -> Self {
        Self {
            name: T::NAME,
            run: tool_run::<T>,
            confirmation: tool_confirmation::<T>,
        }
    }
}

// ── Type-erasing shims ────────────────────────────────────────────────────────

fn tool_run<T: Tool>(ctx: &ToolContext, args: &Value) -> Value {
    let req: T::Request = match serde_json::from_value(args.clone()) {
        Ok(r) => r,
        Err(e) => return invalid_arguments(&e.to_string()),
    };
    if let Err(e) = req.validate() {
        return invalid_arguments(&format_validation_errors(&e));
    }
    match T::run(ctx, req) {
        Ok(resp) => serde_json::to_value(&resp).unwrap_or_else(|e| {
            // Serialization should be infallible for well-formed types,
            // but surface it cleanly if it isn't.
            json!({
                "error": "internal_error",
                "detail": format!("response serialization failed: {e}"),
            })
        }),
        Err(e) => json!({
            "error": e.code(),
            "detail": e.detail(),
        }),
    }
}

fn tool_confirmation<T: Tool>(args: &Value) -> Option<ConfirmationDetails> {
    let req: T::Request = serde_json::from_value(args.clone()).ok()?;
    if req.validate().is_err() {
        return None;
    }
    T::confirmation(&req)
}

// ── Error formatters ──────────────────────────────────────────────────────────

fn invalid_arguments(detail: &str) -> Value {
    json!({
        "error": "invalid_arguments",
        "detail": detail,
    })
}

fn format_validation_errors(errs: &validator::ValidationErrors) -> String {
    // Produce a deterministic, LLM-actionable error string.
    // Example: "max_results: must be <= 10; query: must not be empty"
    let mut parts: Vec<String> = errs
        .field_errors()
        .iter()
        .flat_map(|(field, fes)| {
            fes.iter().map(move |fe| {
                let msg = fe
                    .message
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| fe.code.to_string());
                format!("{field}: {msg}")
            })
        })
        .collect();
    parts.sort();
    parts.join("; ")
}

// ── Static table of all registered tools ──────────────────────────────────────

/// Every tool known to the registry. Built by [`register_all`].
pub fn all_tools() -> &'static [RegisteredTool] {
    register_all()
}

/// Look up a tool by its canonical name or any registered alias.
pub fn find(name: &str) -> Option<&'static RegisteredTool> {
    if let Some(t) = all_tools().iter().find(|t| t.name == name) {
        return Some(t);
    }
    let canon = ALIAS_GROUPS
        .iter()
        .find(|(_, aliases)| aliases.contains(&name))?
        .0;
    all_tools().iter().find(|t| t.name == canon)
}

/// Alternate names the model may emit for a tool, beyond its canonical
/// [`Tool::NAME`]. Drawn from prior-training conventions, other models'/agents'
/// tool names (e.g. Qwen-Agent's `code_interpreter`), and common abbreviations.
/// [`find`] resolves any alias to the canonical tool so an invocation under a
/// variant name still runs the right tool. **Aliases are not shown in the system
/// prompt** — only the canonical name is. Every alias is globally unique and
/// distinct from every canonical name (enforced by the `aliases_unique` test).
static ALIAS_GROUPS: &[(&str, &[&str])] = &[
    (
        "datetime",
        &[
            "get_current_time",
            "current_time",
            "get_time",
            "get_date",
            "current_date",
            "get_datetime",
            "now",
            "time_now",
        ],
    ),
    (
        "calculator",
        &[
            "calc",
            "compute",
            "evaluate",
            "eval",
            "do_math",
            "arithmetic",
        ],
    ),
    (
        "unit_convert",
        &[
            "convert",
            "convert_units",
            "unit_conversion",
            "convert_unit",
        ],
    ),
    (
        "random",
        &[
            "random_number",
            "rng",
            "roll_dice",
            "random_choice",
            "get_random",
            "generate_random",
            "coin_flip",
            "roll",
        ],
    ),
    (
        "web_search",
        &[
            "search",
            "search_web",
            "google",
            "internet_search",
            "search_internet",
            "web_query",
        ],
    ),
    (
        "web_fetch",
        &[
            "fetch",
            "fetch_url",
            "get_url",
            "fetch_page",
            "open_url",
            "get_webpage",
            "browse",
            "read_url",
        ],
    ),
    (
        "weather",
        &[
            "get_weather",
            "weather_forecast",
            "current_weather",
            "check_weather",
            "forecast",
        ],
    ),
    (
        "write",
        &[
            "file_write",
            "file_create",
            "create_file",
            "write_file",
            "save_file",
            "new_file",
            "fs_write",
        ],
    ),
    (
        "file_read",
        &[
            "read_file",
            "read",
            "cat",
            "get_file",
            "open_file",
            "view_file",
            "fs_read",
        ],
    ),
    (
        "file_edit",
        &[
            "edit",
            "modify",
            "edit_file",
            "modify_file",
            "update_file",
            "str_replace",
            "replace_in_file",
            "patch_file",
        ],
    ),
    (
        "file_list",
        &[
            "list_files",
            "ls",
            "dir",
            "list_dir",
            "list_directory",
            "fs_list",
        ],
    ),
    (
        "file_delete",
        &["delete_file", "rm", "remove_file", "fs_delete", "unlink"],
    ),
    (
        "file_present",
        &["show_file", "present_file", "display_file"],
    ),
    (
        "notes_write",
        &[
            "write_note",
            "save_note",
            "create_note",
            "add_note",
            "note_write",
            "store_note",
        ],
    ),
    (
        "notes_read",
        &["read_note", "get_note", "note_read", "fetch_note"],
    ),
    (
        "notes_search",
        &["search_notes", "find_notes", "note_search", "query_notes"],
    ),
    ("notes_list", &["list_notes", "note_list", "browse_notes"]),
    (
        "credential_save",
        &[
            "save_credential",
            "store_credential",
            "add_credential",
            "credential_store",
            "save_secret",
        ],
    ),
    (
        "cred_list",
        &[
            "list_credentials",
            "cred",
            "credlist",
            "credentials_list",
            "get_credentials",
            "list_creds",
        ],
    ),
    (
        "credential_delete",
        &[
            "delete_credential",
            "remove_credential",
            "revoke_credential",
            "delete_cred",
        ],
    ),
    (
        "ssh_open",
        &[
            "ssh",
            "ssh_connect",
            "open_ssh",
            "ssh_session",
            "connect_ssh",
            "ssh_login",
        ],
    ),
    (
        "ssh_session_exec",
        &[
            "ssh_exec",
            "ssh_run",
            "ssh_command",
            "exec_ssh",
            "run_ssh",
            "ssh_run_command",
        ],
    ),
    (
        "ssh_session_exec_async",
        &["ssh_exec_async", "ssh_run_async", "ssh_async"],
    ),
    (
        "ssh_session_poll",
        &["ssh_poll", "poll_ssh", "ssh_poll_output"],
    ),
    (
        "ssh_session_list",
        &["ssh_list", "list_ssh", "list_ssh_sessions", "ssh_sessions"],
    ),
    (
        "ssh_session_close",
        &["ssh_close", "close_ssh", "disconnect_ssh", "ssh_disconnect"],
    ),
    (
        "telnet_session_open",
        &[
            "telnet_open",
            "open_telnet",
            "telnet_connect",
            "connect_telnet",
        ],
    ),
    (
        "telnet_send",
        &[
            "telnet",
            "telnet_send_command",
            "telnet_command",
            "telnet_write",
        ],
    ),
    (
        "telnet_session_list",
        &["telnet_list", "list_telnet", "telnet_sessions"],
    ),
    (
        "telnet_session_close",
        &["telnet_close", "close_telnet", "disconnect_telnet"],
    ),
    (
        "http_session_open",
        &[
            "http_open",
            "open_http",
            "http_connect",
            "http_client",
            "create_http_session",
        ],
    ),
    (
        "http_request",
        &[
            "http",
            "http_call",
            "api_request",
            "send_request",
            "http_req",
            "rest_call",
        ],
    ),
    (
        "http_session_list",
        &["http_list", "list_http", "http_sessions"],
    ),
    (
        "http_session_close",
        &["http_close", "close_http", "disconnect_http"],
    ),
    (
        "tcp_session_open",
        &["tcp_open", "open_tcp", "tcp_connect", "connect_tcp"],
    ),
    ("tcp_session_send", &["tcp_send", "send_tcp", "tcp_write"]),
    ("tcp_session_recv", &["tcp_recv", "recv_tcp", "tcp_read"]),
    (
        "tcp_session_list",
        &["tcp_list", "list_tcp", "tcp_sessions"],
    ),
    ("tcp_session_close", &["tcp_close", "close_tcp"]),
    (
        "udp_session_open",
        &["udp_open", "open_udp", "udp_bind", "bind_udp"],
    ),
    ("udp_session_send", &["udp_send", "send_udp"]),
    ("udp_session_recv", &["udp_recv", "recv_udp", "udp_read"]),
    (
        "udp_session_list",
        &["udp_list", "list_udp", "udp_sessions"],
    ),
    ("udp_session_close", &["udp_close", "close_udp"]),
    (
        "tls_session_open",
        &["tls_open", "open_tls", "tls_connect", "connect_tls"],
    ),
    ("tls_session_send", &["tls_send", "send_tls", "tls_write"]),
    ("tls_session_recv", &["tls_recv", "recv_tls", "tls_read"]),
    (
        "tls_session_list",
        &["tls_list", "list_tls", "tls_sessions"],
    ),
    ("tls_session_close", &["tls_close", "close_tls"]),
    (
        "sql_session_open",
        &[
            "sql_open",
            "open_db",
            "db_connect",
            "sql_connect",
            "connect_db",
            "open_database",
        ],
    ),
    (
        "sql_session_query",
        &[
            "sql_query",
            "query",
            "sql",
            "run_query",
            "execute_sql",
            "db_query",
            "exec_sql",
        ],
    ),
    (
        "sql_session_list",
        &["sql_list", "list_db", "db_sessions", "sql_sessions"],
    ),
    (
        "sql_session_close",
        &["sql_close", "close_db", "disconnect_db"],
    ),
    (
        "remote_fs_session_open",
        &[
            "sftp_open",
            "open_sftp",
            "sftp_connect",
            "remote_fs_open",
            "connect_sftp",
        ],
    ),
    (
        "remote_fs_session_list_dir",
        &[
            "sftp_list_dir",
            "sftp_ls",
            "remote_ls",
            "sftp_listdir",
            "list_remote_dir",
        ],
    ),
    (
        "remote_fs_session_stat",
        &["sftp_stat", "remote_stat", "stat_remote"],
    ),
    (
        "remote_fs_session_get",
        &[
            "sftp_get",
            "sftp_download",
            "download_file",
            "remote_get",
            "sftp_download_file",
        ],
    ),
    (
        "remote_fs_session_put",
        &[
            "sftp_upload",
            "sftp_put",
            "upload_file",
            "remote_put",
            "sftp_upload_file",
        ],
    ),
    (
        "remote_fs_session_delete",
        &["sftp_delete", "remote_delete", "sftp_rm", "remote_rm"],
    ),
    (
        "remote_fs_session_mkdir",
        &["sftp_mkdir", "remote_mkdir", "make_remote_dir"],
    ),
    (
        "remote_fs_session_rename",
        &["sftp_rename", "remote_rename", "move_remote", "remote_move"],
    ),
    (
        "remote_fs_session_list",
        &[
            "sftp_list",
            "list_sftp",
            "sftp_sessions",
            "remote_fs_sessions",
        ],
    ),
    (
        "remote_fs_session_close",
        &["sftp_close", "close_sftp", "disconnect_sftp"],
    ),
    (
        "dns_lookup",
        &[
            "dns",
            "nslookup",
            "dig",
            "resolve",
            "resolve_host",
            "dns_resolve",
            "lookup_dns",
        ],
    ),
    ("ping_icmp", &["ping", "icmp_ping", "ping_host"]),
    (
        "trace_route",
        &["traceroute", "tracert", "trace", "trace_path"],
    ),
    (
        "port_scan",
        &["scan_ports", "portscan", "port_check", "check_ports"],
    ),
    (
        "ip_scan",
        &[
            "scan_ips",
            "network_scan",
            "subnet_scan",
            "scan_subnet",
            "scan_network",
        ],
    ),
    (
        "host_info",
        &["hostinfo", "whois", "host_lookup", "profile_host"],
    ),
    (
        "hash_scan",
        &[
            "identify_hash",
            "hash_identify",
            "detect_hash",
            "recognize_hash",
        ],
    ),
    (
        "hash_compute",
        &["hash", "compute_hash", "digest", "checksum", "hash_data"],
    ),
    (
        "totp",
        &[
            "totp_generate",
            "generate_totp",
            "get_otp",
            "otp",
            "get_totp",
            "totp_code",
            "mfa_code",
            "two_factor_code",
        ],
    ),
    ("aead_encrypt", &["encrypt", "aead_enc", "encrypt_aead"]),
    ("aead_decrypt", &["decrypt", "aead_dec", "decrypt_aead"]),
    ("hmac_compute", &["hmac", "compute_hmac", "hmac_sign"]),
    (
        "signature_verify",
        &["verify_signature", "verify_sig", "verify"],
    ),
    (
        "signature_sign",
        &["sign", "sign_message", "sign_data", "create_signature"],
    ),
    ("kdf_derive", &["derive_key", "kdf", "derive"]),
    ("hkdf_extract", &["hkdf", "hkdf_prk"]),
    ("hkdf_expand_label", &["hkdf_expand", "expand_label"]),
    ("hash_state_init", &["hash_init", "init_hash", "start_hash"]),
    (
        "hash_state_update",
        &["hash_update", "update_hash", "feed_hash"],
    ),
    (
        "hash_state_finalize",
        &[
            "hash_final",
            "hash_finalize",
            "finalize_hash",
            "hash_digest",
        ],
    ),
    (
        "bytes_transcode",
        &["transcode", "encode", "decode", "convert_bytes", "reencode"],
    ),
    ("bytes_pack", &["pack", "struct_pack", "pack_bytes"]),
    ("bytes_unpack", &["unpack", "struct_unpack", "unpack_bytes"]),
    ("bytes_xor", &["xor", "xor_bytes"]),
    (
        "code_run",
        &[
            "run_code",
            "execute_code",
            "run",
            "exec",
            "code_interpreter",
            "python",
            "run_python",
            "eval_code",
        ],
    ),
    (
        "code_session_open",
        &[
            "code_open",
            "open_code",
            "start_code",
            "open_repl",
            "code_session_start",
        ],
    ),
    (
        "code_session_exec",
        &[
            "code_exec",
            "exec_code",
            "run_in_session",
            "code_session_run",
        ],
    ),
    (
        "code_session_list",
        &["code_sessions", "list_code_sessions", "list_code_sandboxes"],
    ),
    (
        "code_session_close",
        &["code_close", "close_code", "close_repl", "end_code_session"],
    ),
    (
        "sub_run",
        &[
            "subagent",
            "run_subagent",
            "delegate",
            "spawn_agent",
            "sub_agent",
            "nested_agent",
        ],
    ),
];

use crate::tools::{
    bytes::{BYTES_PACK, BYTES_TRANSCODE, BYTES_UNPACK, BYTES_XOR},
    calculator::REGISTRATION as CALC,
    code::{CODE_RUN, CODE_SESSION_CLOSE, CODE_SESSION_EXEC, CODE_SESSION_LIST, CODE_SESSION_OPEN},
    credentials::{CREDENTIAL_DELETE, CREDENTIAL_LIST, CREDENTIAL_SAVE},
    crypto::{
        AEAD_DECRYPT, AEAD_ENCRYPT, HKDF_EXPAND_LABEL, HKDF_EXTRACT, HMAC_COMPUTE, KDF_DERIVE,
        SIGNATURE_SIGN, SIGNATURE_VERIFY,
    },
    datetime::REGISTRATION as DATETIME,
    file::{FILE_DELETE, FILE_EDIT, FILE_LIST, FILE_PRESENT, FILE_READ, FILE_WRITE},
    hash::{HASH_COMPUTE, HASH_SCAN},
    hash_state::{HASH_STATE_FINALIZE, HASH_STATE_INIT, HASH_STATE_UPDATE},
    http_session::{
        HTTP_SESSION_CLOSE, HTTP_SESSION_LIST, HTTP_SESSION_OPEN, HTTP_SESSION_REQUEST,
    },
    network_diag::{DNS_LOOKUP, HOST_INFO, IP_SCAN, PING_ICMP, PORT_SCAN, TRACE_ROUTE},
    notes::{NOTES_LIST, NOTES_READ, NOTES_SEARCH, NOTES_WRITE},
    random::REGISTRATION as RANDOM,
    remote_fs::{
        REMOTE_FS_SESSION_CLOSE, REMOTE_FS_SESSION_DELETE, REMOTE_FS_SESSION_GET,
        REMOTE_FS_SESSION_LIST, REMOTE_FS_SESSION_LIST_DIR, REMOTE_FS_SESSION_MKDIR,
        REMOTE_FS_SESSION_OPEN, REMOTE_FS_SESSION_PUT, REMOTE_FS_SESSION_RENAME,
        REMOTE_FS_SESSION_STAT,
    },
    sql_session::{SQL_SESSION_CLOSE, SQL_SESSION_LIST, SQL_SESSION_OPEN, SQL_SESSION_QUERY},
    ssh::{
        SSH_SESSION_CLOSE, SSH_SESSION_EXEC, SSH_SESSION_EXEC_ASYNC, SSH_SESSION_LIST,
        SSH_SESSION_OPEN, SSH_SESSION_POLL,
    },
    subagent::REGISTRATION as SUBAGENT,
    tcp_session::{
        TCP_SESSION_CLOSE, TCP_SESSION_LIST, TCP_SESSION_OPEN, TCP_SESSION_RECV, TCP_SESSION_SEND,
    },
    telnet::{TELNET_SESSION_CLOSE, TELNET_SESSION_LIST, TELNET_SESSION_OPEN, TELNET_SESSION_SEND},
    tls_session::{
        TLS_SESSION_CLOSE, TLS_SESSION_LIST, TLS_SESSION_OPEN, TLS_SESSION_RECV, TLS_SESSION_SEND,
    },
    totp::REGISTRATION as TOTP,
    udp_session::{
        UDP_SESSION_CLOSE, UDP_SESSION_LIST, UDP_SESSION_OPEN, UDP_SESSION_RECV, UDP_SESSION_SEND,
    },
    unit_convert::REGISTRATION as UNIT_CONVERT,
    weather::REGISTRATION as WEATHER,
    web_fetch::REGISTRATION as WEB_FETCH,
    web_search::REGISTRATION as WEB_SEARCH,
};

fn register_all() -> &'static [RegisteredTool] {
    // High-risk (`.risky()`) = side effects or reaches outside the host:
    // mutation, code execution, remote/network access, credentials, scanning.
    // These are the tools omitted in "Restricted" mode. Safe tools (local
    // read-only + pure compute) carry no marker.
    static TOOLS: &[RegisteredTool] = &[
        // Shared tools (7) — all safe (compute + web read)
        DATETIME,
        CALC,
        UNIT_CONVERT,
        RANDOM,
        WEB_SEARCH,
        WEB_FETCH,
        WEATHER,
        // File tools (6) — reads safe, mutations high-risk
        FILE_WRITE,
        FILE_READ,
        FILE_EDIT,
        FILE_LIST,
        FILE_DELETE,
        FILE_PRESENT,
        // Notes tools (4) — reads safe, write high-risk
        NOTES_WRITE,
        NOTES_READ,
        NOTES_SEARCH,
        NOTES_LIST,
        // Credential tools (3) — all high-risk (secrets)
        CREDENTIAL_SAVE,
        CREDENTIAL_LIST,
        CREDENTIAL_DELETE,
        // SSH tools (6) — all high-risk (remote exec)
        SSH_SESSION_OPEN,
        SSH_SESSION_EXEC,
        SSH_SESSION_EXEC_ASYNC,
        SSH_SESSION_POLL,
        SSH_SESSION_LIST,
        SSH_SESSION_CLOSE,
        // Telnet tools (4) — all high-risk (remote access)
        TELNET_SESSION_OPEN,
        TELNET_SESSION_SEND,
        TELNET_SESSION_LIST,
        TELNET_SESSION_CLOSE,
        // HTTP session tools (4) — all high-risk (network)
        HTTP_SESSION_OPEN,
        HTTP_SESSION_REQUEST,
        HTTP_SESSION_LIST,
        HTTP_SESSION_CLOSE,
        // TCP tools (5) — all high-risk (raw sockets)
        TCP_SESSION_OPEN,
        TCP_SESSION_SEND,
        TCP_SESSION_RECV,
        TCP_SESSION_LIST,
        TCP_SESSION_CLOSE,
        // UDP tools (5) — all high-risk (raw sockets)
        UDP_SESSION_OPEN,
        UDP_SESSION_SEND,
        UDP_SESSION_RECV,
        UDP_SESSION_LIST,
        UDP_SESSION_CLOSE,
        // TLS tools (5) — all high-risk (raw sockets)
        TLS_SESSION_OPEN,
        TLS_SESSION_SEND,
        TLS_SESSION_RECV,
        TLS_SESSION_LIST,
        TLS_SESSION_CLOSE,
        // SQL tools (4) — all high-risk (database access)
        SQL_SESSION_OPEN,
        SQL_SESSION_QUERY,
        SQL_SESSION_LIST,
        SQL_SESSION_CLOSE,
        // Remote FS tools (10) — all high-risk (remote files)
        REMOTE_FS_SESSION_OPEN,
        REMOTE_FS_SESSION_LIST_DIR,
        REMOTE_FS_SESSION_STAT,
        REMOTE_FS_SESSION_GET,
        REMOTE_FS_SESSION_PUT,
        REMOTE_FS_SESSION_DELETE,
        REMOTE_FS_SESSION_MKDIR,
        REMOTE_FS_SESSION_RENAME,
        REMOTE_FS_SESSION_LIST,
        REMOTE_FS_SESSION_CLOSE,
        // Network diagnostics (6) — lookups safe, scanning high-risk
        DNS_LOOKUP,
        PING_ICMP,
        TRACE_ROUTE,
        PORT_SCAN,
        IP_SCAN,
        HOST_INFO,
        // Security utilities (3) — all safe (compute)
        HASH_SCAN,
        HASH_COMPUTE,
        TOTP,
        // Crypto primitives (8) — all safe (pure compute, no external effect)
        AEAD_ENCRYPT,
        AEAD_DECRYPT,
        HMAC_COMPUTE,
        SIGNATURE_VERIFY,
        SIGNATURE_SIGN,
        KDF_DERIVE,
        HKDF_EXTRACT,
        HKDF_EXPAND_LABEL,
        // Hash state tools (3) — all safe (compute)
        HASH_STATE_INIT,
        HASH_STATE_UPDATE,
        HASH_STATE_FINALIZE,
        // Byte tools (4) — all safe (compute)
        BYTES_TRANSCODE,
        BYTES_PACK,
        BYTES_UNPACK,
        BYTES_XOR,
        // Code execution (5) — JavaScript on the embedded sandboxed engine
        CODE_RUN,
        CODE_SESSION_OPEN,
        CODE_SESSION_EXEC,
        CODE_SESSION_LIST,
        CODE_SESSION_CLOSE,
        // Subagent (1) — high-risk (delegated agency)
        SUBAGENT,
    ];
    TOOLS
}

#[cfg(test)]
mod alias_tests {
    use super::*;

    /// Every canonical name is unique, every alias group points at a real tool,
    /// and every alias is globally unique (no collision with a canonical name or
    /// another alias). Keeps the variant coverage unambiguous.
    #[test]
    fn aliases_unique() {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for t in all_tools() {
            assert!(
                seen.insert(t.name),
                "duplicate canonical tool name {:?}",
                t.name
            );
        }
        for (canon, aliases) in ALIAS_GROUPS {
            let resolved = all_tools().iter().find(|t| t.name == *canon);
            assert!(
                resolved.is_some(),
                "alias group canonical {canon:?} is not a registered tool"
            );
            for a in *aliases {
                assert!(
                    seen.insert(a),
                    "alias {a:?} (group {canon:?}) collides with a tool name or another alias",
                );
                assert_eq!(
                    find(a).map(|t| t.name),
                    Some(*canon),
                    "alias {a:?} must resolve to {canon:?}"
                );
            }
        }
    }
}
