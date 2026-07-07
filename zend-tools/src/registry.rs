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

/// A type-erased registration for the static tool table.
#[derive(Copy, Clone)]
pub struct RegisteredTool {
    pub name: &'static str,
    pub description: &'static str,
    /// Returns the JSON schema for this tool's `Request` type.
    pub schema: fn() -> Value,
    /// Dispatch entry — parses, validates, runs the tool, encodes result.
    /// The returned `Value` is what the orchestrator places inside the
    /// `<tool_response>` block (success payload OR an `{"error": ...}` shape).
    pub run: fn(&ToolContext, &Value) -> Value,
    /// Returns confirmation details if this tool wants a user prompt.
    /// Returns `None` if either no confirmation is needed *or* the args
    /// could not be parsed (in which case `run` will surface the error).
    pub confirmation: fn(&Value) -> Option<ConfirmationDetails>,
    /// Whether this tool has side effects or reaches outside the host —
    /// mutation, code execution, remote/network access, credentials, or
    /// scanning. The "Restricted" tools mode omits every high-risk tool (and
    /// uses a tool summary built from the safe subset); "None" omits all tools;
    /// "Comprehensive" keeps the full catalog. The full policy is the set of
    /// `.risky()` calls in [`register_all`].
    pub high_risk: bool,
    /// Eight example user prompts whose intent this tool satisfies. Decoded under
    /// forced tool-selection in the "Calibrating sections" load phase to seed the
    /// per-tool wide-Q (`Q·Q`) reference substrate. Eight per tool (not four) so
    /// leave-one-out holdout has more support; the eight are authored to be
    /// genuinely varied (phrasing, values, sub-scenario) rather than near-
    /// duplicates, so a held-out example still tests generalization. Authored per
    /// tool in [`register_all`]; empty strings are skipped.
    pub examples: [&'static str; 8],
}

impl RegisteredTool {
    /// Build a registration for a `Tool` impl. Safe (no side effects) by
    /// default; mark side-effecting/outward-facing tools with [`Self::risky`].
    pub const fn new<T: Tool>() -> Self {
        Self {
            name: T::NAME,
            description: T::DESCRIPTION,
            schema: tool_schema::<T>,
            run: tool_run::<T>,
            confirmation: tool_confirmation::<T>,
            high_risk: false,
            examples: [""; 8],
        }
    }

    /// Mark this registration high-risk (omitted in "Restricted" tools mode).
    /// Const so it composes into the `static` registry table.
    pub const fn risky(mut self) -> Self {
        self.high_risk = true;
        self
    }

    /// Attach the eight example prompts for this tool (authored in [`register_all`]).
    /// Each is a realistic user request whose intent the tool satisfies.
    pub const fn examples(mut self, examples: [&'static str; 8]) -> Self {
        self.examples = examples;
        self
    }
}

// ── Type-erasing shims ────────────────────────────────────────────────────────

fn tool_schema<T: Tool>() -> Value {
    let schema = schemars::gen::SchemaGenerator::default().into_root_schema_for::<T::Request>();
    serde_json::to_value(&schema).unwrap_or(Value::Null)
}

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

/// The catalog category a tool belongs to — the deterministic grouping the
/// tool-catalog summary renders under `## <category>`. One arm per family,
/// mirroring the layout of [`register_all`]; every registered tool name maps to
/// exactly one category, so the summary is a pure function of the catalog (no
/// model call). A name with no arm falls through to `"Other"`.
pub fn category_for(name: &str) -> &'static str {
    match name {
        "datetime" | "calculator" | "unit_convert" | "random" | "web_search" | "web_fetch"
        | "weather" => "Utilities & web",
        "write" | "file_read" | "file_edit" | "file_list" | "file_delete" | "file_present" => {
            "Files"
        }
        "notes_write" | "notes_read" | "notes_search" | "notes_list" => "Notes",
        "credential_save" | "cred_list" | "credential_delete" => "Credentials",
        "ssh_open"
        | "ssh_session_exec"
        | "ssh_session_exec_async"
        | "ssh_session_poll"
        | "ssh_session_list"
        | "ssh_session_close" => "SSH sessions",
        "telnet_session_open" | "telnet_send" | "telnet_session_list" | "telnet_session_close" => {
            "Telnet sessions"
        }
        "http_session_open" | "http_request" | "http_session_list" | "http_session_close" => {
            "HTTP sessions"
        }
        "tcp_session_open" | "tcp_session_send" | "tcp_session_recv" | "tcp_session_list"
        | "tcp_session_close" => "TCP sessions",
        "udp_session_open" | "udp_session_send" | "udp_session_recv" | "udp_session_list"
        | "udp_session_close" => "UDP sessions",
        "tls_session_open" | "tls_session_send" | "tls_session_recv" | "tls_session_list"
        | "tls_session_close" => "TLS sessions",
        "sql_session_open" | "sql_session_query" | "sql_session_list" | "sql_session_close" => {
            "SQL sessions"
        }
        "remote_fs_session_open"
        | "remote_fs_session_list_dir"
        | "remote_fs_session_get"
        | "remote_fs_session_put"
        | "remote_fs_session_delete"
        | "remote_fs_session_mkdir"
        | "remote_fs_session_stat"
        | "remote_fs_session_rename"
        | "remote_fs_session_list"
        | "remote_fs_session_close" => "Remote filesystem",
        "dns_lookup" | "ping_icmp" | "trace_route" | "port_scan" | "ip_scan" | "host_info" => {
            "Network diagnostics"
        }
        "hash_scan" | "hash_compute" | "totp" => "Security utilities",
        "aead_encrypt" | "aead_decrypt" | "hmac_compute" | "signature_verify"
        | "signature_sign" | "kdf_derive" | "hkdf_extract" | "hkdf_expand_label" => "Cryptography",
        "hash_state_init" | "hash_state_update" | "hash_state_finalize" => "Hash streaming",
        "bytes_transcode" | "bytes_pack" | "bytes_unpack" | "bytes_xor" => "Byte encoding",
        "code_run" | "code_session_open" | "code_session_exec" | "code_session_list"
        | "code_session_close" => "Code execution",
        "sub_run" => "Subagent",
        _ => "Other",
    }
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
        DATETIME.examples([
            "what time is it in Tokyo right now",
            "what's today's date",
            "what's the current time in America/New_York",
            "give me the current UTC date and time",
            "what day of the week is it today",
            "what's the current local time in Asia/Dubai",
            "tell me the current Unix timestamp",
            "if it's 3pm here in Berlin, what time is it in Sydney",
        ]),
        CALC.examples([
            "what is 37% of 84193",
            "evaluate (128 + 47) * 2^7 - 913",
            "compute the square root of 74183",
            "what is 1234 multiplied by 5678",
            "what is 76234 multiplied by 88911",
            "log base 2 of 95443",
            "what's the square root of 2750000",
            "how much is 918273 divided by 4657, to four decimal places",
        ]),
        UNIT_CONVERT.examples([
            "convert 100 kilometers to miles",
            "how many ounces are in 2.5 liters",
            "convert 72 fahrenheit to celsius",
            "how many feet are in 3 meters",
            "turn 5 pounds into kilograms",
            "convert 90 miles per hour to meters per second",
            "how many gigabytes is 4500 megabytes",
            "how many seconds are in 6.25 hours",
        ]),
        RANDOM.examples([
            "pick a random number between 1 and 100",
            "flip a coin for me",
            "choose randomly between pizza, sushi, and tacos",
            "roll two six-sided dice",
            "shuffle the list Alice, Bob, Carol, Dave and give me the order",
            "generate a random hex color",
            "give me a random integer from 1000 to 9999 for a PIN",
            "pick one winner at random from these five names",
        ]),
        WEB_SEARCH.examples([
            "search the web for the latest Rust async runtime benchmarks",
            "find recent news about the James Webb telescope",
            "look up the current population of Canada",
            "search for how to configure nginx as a reverse proxy",
            "who won the Formula 1 race last weekend",
            "what are the reviews saying about the new Framework laptop",
            "find the official documentation for the tokio crate",
            "search for vegetarian ramen recipes",
        ]),
        WEB_FETCH.examples([
            "fetch the contents of https://example.com",
            "get the text from https://www.rust-lang.org/learn",
            "download the page at https://news.ycombinator.com",
            "retrieve https://api.github.com/repos/rust-lang/rust",
            "summarize the article at https://blog.rust-lang.org/2024/01/01/announcement.html",
            "pull the raw README from https://raw.githubusercontent.com/tokio-rs/tokio/master/README.md",
            "grab the JSON from https://httpbin.org/json",
            "read what's on https://en.wikipedia.org/wiki/Ferris_wheel",
        ]),
        WEATHER.examples([
            "what's the weather in London today",
            "will it rain in Seattle tomorrow",
            "give me the 5-day forecast for Denver",
            "how hot is it in Phoenix right now",
            "do I need a jacket in Paris this evening",
            "what's the humidity like in Singapore",
            "is it snowing in Reykjavik",
            "what's the wind speed at the coast in Cape Town",
        ]),
        // File tools (6) — reads safe, mutations high-risk
        FILE_WRITE.risky().examples([
            "create the file /workspace/config.toml with a basic logging section",
            "write 'hello world' to /workspace/notes/greeting.txt",
            "create /workspace/data/users.json containing exactly an empty JSON object: {}",
            "create /workspace/README.md with the title '# Acme API'",
            "drop a .gitignore at /workspace/.gitignore that ignores target/ and *.log",
            "can you save a shell script to /workspace/scripts/deploy.sh that just echoes \"deploying\"",
            "overwrite /workspace/VERSION so it contains only the text 2.4.1",
            "put a Dockerfile in /workspace/Dockerfile starting FROM rust:1.79 as the base image",
        ]),
        FILE_READ.examples([
            "show me the contents of /workspace/Cargo.toml",
            "read the file /workspace/src/main.rs",
            "what's in /workspace/.gitignore",
            "open /workspace/config/settings.yaml and show it",
            "cat /workspace/docs/architecture.md for me",
            "I need to see what's inside /workspace/scripts/build.sh",
            "pull up the contents of /workspace/tests/fixtures/sample.json",
            "grab the text of /workspace/LICENSE and print it",
        ]),
        FILE_EDIT.risky().examples([
            "in /workspace/Cargo.toml change the line `version = \"0.1.0\"` to `version = \"0.2.0\"`",
            "in /workspace/src/config.rs replace the string `localhost` with `0.0.0.0`",
            "update the port from 8080 to 9090 in /workspace/config.toml",
            "rename the function foo to bar in /workspace/src/utils.rs",
            "in /workspace/.env swap DEBUG=true for DEBUG=false",
            "change the log level from \"info\" to \"debug\" in /workspace/logging.yaml",
            "in /workspace/docker-compose.yml bump the image tag from :1.2 to :1.3",
            "fix the typo `recieve` to `receive` in /workspace/src/handlers/mod.rs",
        ]),
        FILE_LIST.examples([
            "list the files in /workspace/src",
            "what files are in /workspace",
            "show me everything under /workspace/tests",
            "list the files under /workspace/benches",
            "what's in the /workspace/migrations directory",
            "give me a directory listing of /workspace/assets/images",
            "which files live under /workspace/config",
            "enumerate the contents of /workspace/scripts",
        ]),
        FILE_DELETE.risky().examples([
            "delete the file /workspace/tmp/scratch.txt",
            "remove /workspace/config.bak",
            "delete /workspace/build/output.log",
            "get rid of the file /workspace/notes/draft.md",
            "can you clean up /workspace/target/debug/incremental.tmp",
            "I don't need /workspace/old/legacy.py anymore, delete it",
            "wipe /workspace/cache/session.dat",
            "trash the file at /workspace/downloads/duplicate(1).zip",
        ]),
        FILE_PRESENT.examples([
            "show me the file /workspace/Cargo.lock",
            "display /workspace/src/main.rs in the chat",
            "surface /workspace/config/prod.yaml and /workspace/.env for me to look at",
            "present the file /workspace/README.md to the user",
            "pop /workspace/docs/api-reference.md into the conversation",
            "can you bring up /workspace/src/lib.rs and /workspace/src/error.rs side by side",
            "put /workspace/CHANGELOG.md in front of me",
            "attach /workspace/reports/coverage.html to the chat so I can view it",
        ]),
        // Notes tools (4) — reads safe, write high-risk
        NOTES_WRITE.risky().examples([
            "save a note titled 'auth-refactor' with content 'refactor the auth module to use the new token store'",
            "save a note titled 'meeting' with content 'team sync at 3pm Thursday in room 4'",
            "save a note titled 'parser-bug' with content 'the tokenizer drops trailing commas in nested arrays'",
            "save a note titled 'api-key-rotation' with content 'revoke old key, generate new, update configs, redeploy'",
            "jot down a note called 'gift-ideas' saying 'headphones for Sam, cookbook for Alex'",
            "remember this: under the title 'server-ip' store '10.0.0.42 is the new prod box'",
            "make a note 'grocery-list' with the body 'eggs, oat milk, spinach, coffee beans'",
            "note titled 'interview-feedback': candidate was strong on systems, weak on frontend",
        ]),
        NOTES_READ.examples([
            "read the note titled deployment-runbook",
            "show me the note titled todo",
            "read the note titled standup-2026-06-15",
            "open the note titled database-schema",
            "pull up my note called onboarding-checklist",
            "what did I write in the note titled retro-notes",
            "fetch the contents of the note keyed vacation-plans",
            "let me see the note titled 'ssl-cert-renewal'",
        ]),
        NOTES_SEARCH.examples([
            "search my notes for anything about kubernetes",
            "find notes mentioning the payment gateway",
            "look through my notes for the wifi setup",
            "search notes for 'release checklist'",
            "do I have any notes tagged 'urgent'",
            "which of my notes talk about the database migration",
            "hunt through my notes for anything referencing Postgres",
            "find the note where I wrote down the office door code",
        ]),
        NOTES_LIST.examples([
            "list all my notes",
            "what notes do I have",
            "show me every note I've saved",
            "give me an index of my notes",
            "which notes start with the prefix 'meeting-'",
            "show me the notes tagged 'work' along with their timestamps",
            "catalog my notes so I can see what exists before opening one",
            "how many notes have I stored and what are their keys",
        ]),
        // Credential tools (3) — all high-risk (secrets)
        CREDENTIAL_SAVE.risky().examples([
            "save the credential named github-pat with value ghp_aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3z",
            "store an API token under the name openai-key with value sk-proj-abc123def456",
            "save the password for the analytics DB under the name analytics-db, value Hunter2!swordfish",
            "remember the credential named stripe-secret with value sk_live_51HxYzAbCdEfG",
            "can you stash my AWS secret access key as aws-prod, the value is wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "add a new credential called cloudflare-token set to v1.0-9zQwErTyUiOpAsDfGhJkL",
            "keep the vault-unseal-key credential around with value hvs-EXAMPLE-not-a-real-token",
            "I want to store the SMTP app password mailgun-smtp — it's key-3ax6xnjp29jd6fds4gc373sgvjxteol0",
        ]),
        CREDENTIAL_LIST.risky().examples([
            "list all my saved credentials",
            "what credentials do I have stored",
            "show me the names of my saved API keys",
            "which credentials are available for this session",
            "do I already have a credential saved for the staging database?",
            "give me a rundown of every secret name in the credential store",
            "before I connect, remind me what login credentials are on file",
            "enumerate the credential entries I've set up so far",
        ]),
        CREDENTIAL_DELETE.risky().examples([
            "delete the saved credential named old-api-key",
            "revoke the SSH key credential named staging-ssh-key",
            "remove the database password credential named dev-db-password",
            "forget the credential called expired-token",
            "drop the credential named legacy-smtp-password, we don't use it anymore",
            "can you wipe the stored value for jenkins-token?",
            "purge the credential entry named test-account-key from the store",
            "get rid of the saved secret called demo-webhook-secret",
        ]),
        // SSH tools (6) — all high-risk (remote exec)
        SSH_SESSION_OPEN.risky().examples([
            "open an SSH session to 10.0.0.5 using the prod-server credential",
            "connect over SSH to deploy@example.com using credential deploy-key",
            "start an SSH session to staging.example.com port 22 with credential staging-key",
            "open an SSH session to 192.168.1.20 as root using credential root-key",
            "can you SSH into db01.internal.net on port 2222 with the dba-key credential?",
            "establish an SSH connection to ubuntu@172.31.4.18 using credential ec2-pem",
            "log into the build box at 10.10.5.7 over SSH with the ci-runner credential",
            "SSH to backup@nas.lan using the credential nas-admin",
        ]),
        SSH_SESSION_EXEC.risky().examples([
            "run `df -h` on SSH session ssh-7f3a",
            "execute `systemctl status nginx` on SSH session ssh-7f3a",
            "check the uptime on SSH session ssh-2b1c",
            "list the running docker containers on SSH session ssh-9d4e",
            "on SSH session ssh-4c8d, run `tail -n 50 /var/log/auth.log`",
            "grab the free memory by running `free -m` over SSH session ssh-1a2b",
            "execute `whoami && hostname` on SSH session ssh-6e0f",
            "run `cat /etc/os-release` on the SSH session ssh-3d5a and show me the output",
        ]),
        SSH_SESSION_EXEC_ASYNC.risky().examples([
            "run `pg_dump mydb > /backups/db.sql` on SSH session ssh-7f3a in the background",
            "run `make -j8 release` on SSH session ssh-2b1c without waiting",
            "run `./migrate.sh --all` asynchronously on SSH session ssh-9d4e",
            "run `stress-ng --cpu 4 --timeout 60s` on SSH session ssh-7f3a and don't block",
            "kick off `rsync -avz /data/ remote:/data/` on SSH session ssh-5b3c in the background",
            "fire off `apt-get -y dist-upgrade` on SSH session ssh-0e1d and don't wait for it",
            "start `find / -name '*.core' -delete` async on SSH session ssh-8f2a",
            "launch `ffmpeg -i in.mov out.mp4` on SSH session ssh-c4d7 without blocking",
        ]),
        SSH_SESSION_POLL.risky().examples([
            "check whether async process proc_8a3f2c1d4e5b has finished",
            "poll async process proc_4b7e9a02c1f3 for its output so far",
            "is the background job proc_1f2e3d4c5a6b done yet",
            "get the latest stdout from async process proc_9c8b7a6e5d4f",
            "has the async task proc_2d6f8b0a3c7e exited, and what was its status?",
            "show me any new output from background process proc_7e5c1a9b2d0f",
            "what's the current state of the async SSH job proc_0a1b2c3d4e5f",
            "poll proc_6f4e2d0c8b7a and tell me if it's still running",
        ]),
        SSH_SESSION_LIST.risky().examples([
            "list my open SSH sessions",
            "which SSH connections do I currently have open",
            "show me the active SSH sessions",
            "how many SSH sessions are running",
            "give me the ids of every SSH session I've opened",
            "are there any SSH sessions still connected right now?",
            "enumerate my current SSH remote sessions",
            "what SSH sessions are on the books at the moment",
        ]),
        SSH_SESSION_CLOSE.risky().examples([
            "close SSH session ssh-7f3a",
            "disconnect SSH session ssh-2b1c",
            "tear down SSH session ssh-9d4e",
            "end SSH session ssh-7f3a",
            "hang up the SSH session ssh-5f1a, I'm done with it",
            "please shut down SSH session ssh-b3c8",
            "drop the SSH connection ssh-0d2e",
            "terminate SSH session ssh-e7a4",
        ]),
        // Telnet tools (4) — all high-risk (remote access)
        TELNET_SESSION_OPEN.risky().examples([
            "open a telnet connection to the router at 192.168.0.1 port 23",
            "connect via telnet to the switch console at 10.0.0.30 port 23",
            "start a telnet session to the device at 172.16.0.5 port 23",
            "telnet to the serial console at 192.168.1.100 port 2000",
            "open a telnet session to the firewall at 10.10.10.1 on port 23",
            "I need a telnet session opened to bbs.example.net port 23",
            "spin up a telnet connection to the PDU at 172.16.4.9 port 23",
            "establish a telnet link to the mainframe host at 10.20.30.40 port 992",
        ]),
        TELNET_SESSION_SEND.risky().examples([
            "send `show running-config` to telnet session telnet-1 and wait for the prompt",
            "send `enable` on telnet session telnet-1",
            "send `reload` to telnet session telnet-2",
            "issue `show version` on telnet session telnet-1",
            "on telnet session telnet-3, type `configure terminal` and wait for the prompt",
            "could you send `write memory` over telnet session telnet-2?",
            "push the command `show ip interface brief` to telnet session telnet-4",
            "run `logout` on telnet session telnet-1",
        ]),
        TELNET_SESSION_LIST.risky().examples([
            "list my open telnet sessions",
            "which telnet connections are active",
            "show the telnet sessions I have open",
            "how many telnet sessions are running",
            "give me a rundown of every active telnet session",
            "what telnet session ids are currently open?",
            "enumerate the telnet connections still alive",
            "do I still have any telnet sessions going?",
        ]),
        TELNET_SESSION_CLOSE.risky().examples([
            "close telnet session telnet-1",
            "disconnect telnet session telnet-2",
            "end telnet session telnet-1",
            "tear down telnet session telnet-3",
            "please hang up telnet session telnet-4",
            "kill the telnet session telnet-2 now",
            "I'm done with telnet session telnet-5, drop it",
            "terminate telnet session telnet-6",
        ]),
        // HTTP session tools (4) — all high-risk (network)
        HTTP_SESSION_OPEN.risky().examples([
            "open an HTTP session with base URL https://api.example.com",
            "open an HTTP session for https://api.github.com with an Authorization header of `Bearer ghp_abc123`",
            "create an HTTP session for https://api.stripe.com with a default JSON content-type header",
            "open an HTTP session with base URL https://httpbin.org",
            "start an HTTP session pointed at https://api.openweathermap.org for me",
            "can you open an HTTP client session to https://gitlab.com/api/v4 with header `PRIVATE-TOKEN: glpat-xyz789`?",
            "set up an HTTP session for https://jsonplaceholder.typicode.com",
            "spin up an HTTP session against https://api.twilio.com with an `Accept: application/json` default header",
        ]),
        HTTP_SESSION_REQUEST.risky().examples([
            "send a GET request to /users on HTTP session http-1",
            "POST the JSON body {\"item\":\"widget\",\"qty\":3} to /orders on HTTP session http-1",
            "make a DELETE request to /items/42 on HTTP session http-2",
            "send a PATCH to /config with body {\"debug\":true} on HTTP session http-1",
            "do a PUT to /profile/7 with body {\"name\":\"Ada\"} on HTTP session http-3",
            "fire off a GET to /health on HTTP session http-2",
            "on HTTP session http-4, POST {\"email\":\"a@b.com\"} to /subscribe",
            "hit /search?q=candle with a GET on HTTP session http-1",
        ]),
        HTTP_SESSION_LIST.risky().examples([
            "list my open HTTP sessions",
            "which HTTP sessions do I have",
            "show the active HTTP client sessions",
            "what HTTP sessions are open right now",
            "enumerate every open HTTP session id",
            "are there any HTTP client sessions still around?",
            "give me the list of live HTTP sessions",
            "how many HTTP sessions am I holding open?",
        ]),
        HTTP_SESSION_CLOSE.risky().examples([
            "close HTTP session http-1",
            "release HTTP session http-2",
            "end HTTP session http-1",
            "tear down HTTP session http-3",
            "shut down HTTP session http-4 please",
            "I'm finished with HTTP session http-5, close it out",
            "drop the HTTP session http-2",
            "dispose of HTTP session http-6",
        ]),
        // TCP tools (5) — all high-risk (raw sockets)
        TCP_SESSION_OPEN.risky().examples([
            "open a raw TCP connection to 10.0.0.8 on port 6379",
            "connect over TCP to example.com:8080",
            "start a TCP session to 10.0.0.20 on port 5432",
            "open a TCP socket to 192.168.1.50 port 9000",
            "can you open a TCP connection to db.internal:27017?",
            "establish a raw TCP session to 172.16.9.4 on port 11211",
            "dial a TCP socket out to smtp.mailgun.org port 587",
            "open a TCP connection to 10.4.4.4 port 4444",
        ]),
        TCP_SESSION_SEND.risky().examples([
            "send `PING\\r\\n` to TCP session tcp-1",
            "write the raw bytes deadbeef to TCP session tcp-1",
            "send `GET / HTTP/1.0\\r\\n\\r\\n` to TCP session tcp-2",
            "transmit `INFO\\r\\n` to TCP session tcp-1",
            "write the hex bytes 0badf00d to TCP session tcp-3",
            "push `QUIT\\r\\n` down TCP session tcp-2",
            "can you send `EHLO zen\\r\\n` over TCP session tcp-4?",
            "send the raw bytes cafebabe00ff to TCP session tcp-1",
        ]),
        TCP_SESSION_RECV.risky().examples([
            "receive 512 bytes from TCP session tcp-1",
            "receive 1024 bytes from TCP session tcp-1",
            "drain TCP session tcp-2 for 2 seconds",
            "read for up to 5 seconds from TCP session tcp-1",
            "grab 256 bytes off TCP session tcp-3",
            "pull whatever's waiting on TCP session tcp-2 for the next 3 seconds",
            "read 4096 bytes from TCP session tcp-4",
            "block up to 10 seconds reading from TCP session tcp-1",
        ]),
        TCP_SESSION_LIST.risky().examples([
            "list my open TCP connections",
            "which TCP sockets are open",
            "show the active TCP sessions",
            "how many TCP connections do I have",
            "enumerate the live TCP session ids",
            "are any raw TCP sockets still connected?",
            "give me a list of every open TCP session",
            "what TCP connections am I currently holding?",
        ]),
        TCP_SESSION_CLOSE.risky().examples([
            "close TCP session tcp-1",
            "shut down TCP session tcp-2",
            "end TCP session tcp-1",
            "tear down TCP session tcp-3",
            "drop the TCP socket tcp-4",
            "I'm done with TCP session tcp-5, close it",
            "kill TCP session tcp-2 now",
            "close the TCP session whose id is tcp-6",
        ]),
        // UDP tools (5) — all high-risk (raw sockets)
        UDP_SESSION_OPEN.risky().examples([
            "open a UDP socket to 8.8.8.8 port 53",
            "open a UDP session to the syslog server at 10.0.0.50 port 514",
            "open a UDP socket to send datagrams to 10.0.0.9 port 5000",
            "open a UDP session to the NTP server at 192.168.1.1 port 123",
            "can you open a UDP socket bound to the DHCP server 172.16.0.1 on port 67?",
            "I need a UDP session to the game server at game.example.com:27015",
            "spin up a UDP datagram socket pointed at 203.0.113.7 port 1194 for the VPN",
            "open UDP to the StatsD collector at 10.10.5.5 port 8125",
        ]),
        UDP_SESSION_SEND.risky().examples([
            "send the bytes cafebabe to UDP session udp-1",
            "send the text `ping` to UDP session udp-1",
            "send the bytes 0102030405 to UDP session udp-2",
            "send the heartbeat string `hb` to UDP session udp-1",
            "fire the hex datagram deadbeef00ff over UDP session udp-3",
            "can you push the text `DISCOVER` out on UDP session udp-2?",
            "transmit the bytes 1b0000000000000000 as a UDP datagram on udp-1",
            "blast the string `STATUS?` to UDP session udp-4",
        ]),
        UDP_SESSION_RECV.risky().examples([
            "read the next datagram from UDP session udp-1",
            "receive the response packet on UDP session udp-1",
            "get the incoming datagram and sender address from UDP session udp-2",
            "read one packet from UDP session udp-1",
            "grab the next UDP datagram off session udp-3 and tell me who sent it",
            "receive the next inbound datagram on UDP session udp-2",
            "pull one datagram from UDP session udp-4 along with the source address",
            "wait for the next UDP reply on session udp-1",
        ]),
        UDP_SESSION_LIST.risky().examples([
            "list my open UDP sockets",
            "which UDP sessions are active",
            "show the open UDP connections",
            "how many UDP sockets do I have",
            "give me a rundown of every bound UDP datagram socket and its default peer",
            "what UDP sessions are currently open right now",
            "enumerate the active UDP sockets in this conversation",
            "are there any UDP sessions still open",
        ]),
        UDP_SESSION_CLOSE.risky().examples([
            "close UDP session udp-1",
            "end UDP session udp-2",
            "tear down UDP session udp-1",
            "shut down UDP session udp-3",
            "free up the UDP socket on session udp-4",
            "we're done with UDP session udp-2, please release it",
            "drop the UDP datagram socket udp-5",
            "can you close out UDP session udp-6 and free its port?",
        ]),
        // TLS tools (5) — all high-risk (raw sockets)
        TLS_SESSION_OPEN.risky().examples([
            "open a TLS connection to imaps://mail.example.com:993",
            "start a TLS session to ldaps://ldap.example.com:636",
            "connect over TLS to smtps://smtp.example.com:465",
            "open a TLS connection to mqtts://broker.example.com:8883",
            "establish a TLS session to pop3s://pop.example.org:995",
            "can you open TLS to the FTPS control channel at ftps://files.example.net:990?",
            "start an encrypted TLS connection to the NNTPS server nntps://news.example.com:563",
            "open a TLS session to the syslog-tls endpoint at logs.example.com:6514",
        ]),
        TLS_SESSION_SEND.risky().examples([
            "send the IMAP `a1 LOGIN user pass` command to TLS session tls-1",
            "send the hex bytes 300c020101600702010304008000 to TLS session tls-1",
            "send `EHLO example.com` to TLS session tls-2",
            "send the hex bytes 101000044d51545404020000 to TLS session tls-1",
            "over TLS session tls-3, send the POP3 command `USER alice\\r\\n`",
            "write the hex bytes 820100 to TLS session tls-4",
            "push `CAPABILITY\\r\\n` down TLS session tls-1",
            "can you send the LDAP unbind hex 30050201034200 to TLS session tls-2?",
        ]),
        TLS_SESSION_RECV.risky().examples([
            "receive 512 bytes from TLS session tls-1",
            "read for up to 3 seconds from TLS session tls-1",
            "receive 1024 bytes from TLS session tls-2",
            "drain TLS session tls-1 for 2 seconds",
            "read up to 4096 bytes of decrypted data from TLS session tls-3",
            "wait 5 seconds for the TLS server greeting on session tls-1",
            "grab the next 256 bytes off TLS session tls-4",
            "can you read the TLS response on session tls-2 for up to 1 second?",
        ]),
        TLS_SESSION_LIST.risky().examples([
            "list my open TLS sessions",
            "which TLS connections are active",
            "show the open TLS sockets",
            "how many TLS sessions do I have",
            "show me every TLS channel and its peer certificate identity",
            "what encrypted TLS sessions are open at the moment",
            "enumerate the active TLS connections and their hosts",
            "are any TLS sessions still connected",
        ]),
        TLS_SESSION_CLOSE.risky().examples([
            "close TLS session tls-1",
            "end TLS session tls-2",
            "tear down TLS session tls-1",
            "shut down TLS session tls-3",
            "send close-notify and shut TLS session tls-4",
            "we're finished with TLS session tls-2, please close it",
            "drop the encrypted TLS channel tls-5",
            "can you close out TLS session tls-6?",
        ]),
        // SQL tools (4) — all high-risk (database access)
        SQL_SESSION_OPEN.risky().examples([
            "open a SQL session against the database at ./data/app.db",
            "connect to the SQLite database analytics.db",
            "start a SQL session for the local users database",
            "open the SQLite file inventory.db as a session",
            "open an in-memory SQL session using :memory:",
            "can you connect to the SQLite database at /var/lib/metrics/telemetry.db?",
            "start a SQL session on the orders database at ./db/orders.sqlite",
            "open a database session for reporting.db",
        ]),
        SQL_SESSION_QUERY.risky().examples([
            "run `SELECT * FROM users LIMIT 10` on SQL session sql-1",
            "run `SELECT * FROM orders WHERE date = date('now')` on SQL session sql-1",
            "execute `UPDATE settings SET value = 'on' WHERE key = 'feature'` on SQL session sql-2",
            "run `SELECT COUNT(*) FROM events` on SQL session sql-1",
            "execute `INSERT INTO logs (level, msg) VALUES ('warn', 'disk low')` on SQL session sql-3",
            "run `SELECT name FROM sqlite_master WHERE type = 'table'` on SQL session sql-1",
            "delete stale rows with `DELETE FROM sessions WHERE expires_at < date('now')` on SQL session sql-2",
            "run `SELECT user_id, SUM(amount) FROM payments GROUP BY user_id` on SQL session sql-4",
        ]),
        SQL_SESSION_LIST.risky().examples([
            "list my open SQL sessions",
            "which database sessions are active",
            "show the open SQL connections",
            "how many SQL sessions do I have",
            "show me every SQL session with its database path and connection state",
            "what database sessions are currently open",
            "enumerate the active SQL connections in this conversation",
            "do I still have any SQL sessions open",
        ]),
        SQL_SESSION_CLOSE.risky().examples([
            "close SQL session sql-1",
            "release SQL session sql-2",
            "end SQL session sql-1",
            "tear down SQL session sql-3",
            "free the SQLite connection on SQL session sql-4",
            "we're done querying, please close SQL session sql-2",
            "drop the database session sql-5",
            "can you shut down SQL session sql-6 and release its handle?",
        ]),
        // Remote FS tools (10) — all high-risk (remote files)
        REMOTE_FS_SESSION_OPEN.risky().examples([
            "open a remote filesystem session to sftp://deploy@example.com using credential deploy-key",
            "open an SFTP session to 10.0.0.5 using credential prod-key",
            "open a remote FS session to sftp://backup@nas.local using credential nas-key",
            "open an SFTP session to prod.example.com using credential prod-key",
            "connect a new remote-fs session to sftp://ci@build-01.internal:2222 with credential ci-runner",
            "can you spin up an SFTP session to 172.16.4.10 using the staging-key credential?",
            "I need a remote filesystem connection to sftp://ops@db-primary.corp using credential vault-ssh",
            "start a remote FS session against archive.example.org on port 22 with credential archive-ro",
        ]),
        REMOTE_FS_SESSION_LIST_DIR.risky().examples([
            "list the files in /var/www on remote-fs session rfs-1",
            "show the contents of /home/deploy on remote-fs session rfs-1",
            "what's in /etc on remote-fs session rfs-2",
            "list the directory /opt/app/logs on remote-fs session rfs-1",
            "ls /srv/data/incoming on remote-fs session rfs-3",
            "can you show me everything under /var/lib/postgresql on remote-fs session rfs-2?",
            "enumerate the entries in /home/ci/artifacts on remote-fs session rfs-4",
            "what files are sitting in /mnt/backup/nightly on remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_STAT.risky().examples([
            "stat /etc/nginx/nginx.conf on remote-fs session rfs-1",
            "get the size and permissions of /var/log/syslog on remote-fs session rfs-1",
            "check the metadata of /opt/app/config.yaml on remote-fs session rfs-2",
            "is /tmp/lock a directory on remote-fs session rfs-1",
            "when was /srv/data/dump.sql last modified on remote-fs session rfs-3?",
            "show the ownership and mode of /usr/local/bin/deploy.sh on remote-fs session rfs-2",
            "how big is /var/lib/mysql/ibdata1 on remote-fs session rfs-4",
            "does /home/deploy/.ssh/authorized_keys exist on remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_GET.risky().examples([
            "download /var/log/app.log from remote-fs session rfs-1",
            "fetch /etc/hosts from remote-fs session rfs-1",
            "pull /opt/app/config.yaml from remote-fs session rfs-2",
            "get /backups/latest.sql from remote-fs session rfs-1",
            "grab /var/log/nginx/error.log off remote-fs session rfs-3 for me",
            "can you retrieve /srv/certs/fullchain.pem from remote-fs session rfs-2?",
            "download /home/ci/artifacts/build-42.tar.gz from remote-fs session rfs-4",
            "copy /etc/systemd/system/zend.service down from remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_PUT.risky().examples([
            "upload config.toml to /opt/app/config.toml on remote-fs session rfs-1",
            "push deploy.sh to /usr/local/bin/deploy.sh on remote-fs session rfs-1",
            "send build.tar.gz to /var/www/releases/build.tar.gz on remote-fs session rfs-2",
            "copy notes.txt to /home/deploy/notes.txt on remote-fs session rfs-1",
            "put local patch.diff at /tmp/patch.diff on remote-fs session rfs-3",
            "can you upload nginx.conf to /etc/nginx/nginx.conf on remote-fs session rfs-2?",
            "transfer secrets.env up to /opt/app/.env on remote-fs session rfs-4",
            "write ca-bundle.crt to /usr/local/share/ca-certificates/ca-bundle.crt on remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_DELETE.risky().examples([
            "delete /tmp/old.log on remote-fs session rfs-1",
            "remove /var/www/stale.html on remote-fs session rfs-1",
            "delete /backups/2019.sql on remote-fs session rfs-2",
            "remove /opt/app/debug.log on remote-fs session rfs-1",
            "get rid of /srv/cache/session-tmp.dat on remote-fs session rfs-3",
            "can you delete /home/ci/artifacts/build-01.tar.gz on remote-fs session rfs-4?",
            "erase /var/log/nginx/access.log.1 on remote-fs session rfs-2",
            "rm /etc/cron.d/stale-job on remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_MKDIR.risky().examples([
            "create the directory /opt/app/releases on remote-fs session rfs-1",
            "make the folder /var/www/uploads on remote-fs session rfs-1",
            "mkdir /home/deploy/tmp on remote-fs session rfs-2",
            "create /data/2026 on remote-fs session rfs-1",
            "can you make a /srv/backups/weekly directory on remote-fs session rfs-3?",
            "set up the folder /opt/app/staging/incoming on remote-fs session rfs-4",
            "new directory /var/lib/zend/snapshots on remote-fs session rfs-2",
            "create /home/ci/artifacts/2026-07 on remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_RENAME.risky().examples([
            "rename /opt/app/current to /opt/app/previous on remote-fs session rfs-1",
            "move /tmp/upload.bin to /data/final.bin on remote-fs session rfs-1",
            "rename /var/www/index.html.new to /var/www/index.html on remote-fs session rfs-2",
            "move /home/deploy/draft.md to /home/deploy/final.md on remote-fs session rfs-1",
            "can you rename /srv/data/dump.sql.tmp to /srv/data/dump.sql on remote-fs session rfs-3?",
            "relocate /tmp/build.tar.gz to /var/www/releases/build.tar.gz on remote-fs session rfs-4",
            "swap /etc/nginx/nginx.conf over to /etc/nginx/nginx.conf.bak on remote-fs session rfs-2",
            "move /home/ci/artifacts/staging to /home/ci/artifacts/archived on remote-fs session rfs-1",
        ]),
        REMOTE_FS_SESSION_LIST.risky().examples([
            "list my open remote filesystem sessions",
            "which SFTP sessions are active",
            "show the open remote FS connections",
            "how many remote filesystem sessions do I have",
            "what remote-fs sessions are currently connected?",
            "give me a rundown of every open SFTP session and its host",
            "are there any remote filesystem sessions still open right now",
            "enumerate my active remote FS session ids",
        ]),
        REMOTE_FS_SESSION_CLOSE.risky().examples([
            "close remote-fs session rfs-1",
            "disconnect remote-fs session rfs-2",
            "end remote-fs session rfs-1",
            "tear down remote-fs session rfs-3",
            "can you shut down remote-fs session rfs-4?",
            "hang up the connection for remote-fs session rfs-2",
            "please close out remote-fs session rfs-5 now",
            "drop remote-fs session rfs-1, I'm done with it",
        ]),
        // Network diagnostics (6) — lookups safe, scanning high-risk
        DNS_LOOKUP.examples([
            "look up the A records for example.com",
            "what's the IP address of rust-lang.org",
            "resolve the DNS for github.com",
            "look up the IP address of cloudflare.com",
            "resolve the IPv4 address for api.stripe.com",
            "what does the AAAA record for ipv6.google.com resolve to?",
            "do a reverse DNS lookup on 8.8.4.4",
            "reverse-resolve the IP 208.67.222.222 to a hostname",
        ]),
        PING_ICMP.examples([
            "ping 8.8.8.8",
            "check if example.com is reachable",
            "ping the gateway at 192.168.1.1",
            "measure the round-trip time to cloudflare.com",
            "is 10.0.0.1 responding to pings?",
            "send 10 ICMP echoes to 1.1.1.1 and report packet loss",
            "can you tell if my NAS at 192.168.1.50 is alive?",
            "what's the latency to dns.quad9.net",
        ]),
        TRACE_ROUTE.risky().examples([
            "trace the route to google.com",
            "show me the network path to 1.1.1.1",
            "traceroute to example.com",
            "map the hops to github.com",
            "where does traffic to 8.8.8.8 get slow?",
            "run a traceroute to cdn.jsdelivr.net and list every hop",
            "trace the path to 203.0.113.42",
            "which routers sit between me and aws.amazon.com?",
        ]),
        PORT_SCAN.risky().examples([
            "check if port 22 is open on 10.0.0.5",
            "scan ports 80, 443, and 8080 on example.com",
            "is the SSH port open on 192.168.1.20",
            "scan ports 22, 80, and 443 on 10.0.0.8",
            "scan ports 21, 25, 110, and 143 on 172.16.0.9",
            "is the Postgres port 5432 listening on db.internal?",
            "check whether 3389 is reachable on 192.168.1.100",
            "probe ports 25, 465, and 587 on mail.example.org",
        ]),
        IP_SCAN.risky().examples([
            "scan the subnet 192.168.1.0/24 for live hosts",
            "find which hosts are up on 10.0.0.0/24",
            "sweep the 172.16.0.0/24 network for active machines",
            "discover live hosts on 192.168.0.0/24",
            "which addresses respond in the 10.10.5.0/25 range?",
            "do a host sweep of 192.168.100.0/24 and list everything alive",
            "scan the subnet 172.20.5.0/24 for reachable hosts",
            "map out active machines across the 10.0.42.0/24 subnet",
        ]),
        HOST_INFO.examples([
            "give me a full profile of example.com",
            "what can you tell me about the host github.com",
            "profile the server at 10.0.0.5",
            "gather host info for rust-lang.org",
            "pull together everything you can find on 1.1.1.1",
            "what's the full rundown on mail.protonmail.com?",
            "build an identification profile for the host at 208.67.220.220",
            "tell me about the machine behind cloudflare.com",
        ]),
        // Security utilities (3) — all safe (compute)
        HASH_SCAN.examples([
            "what algorithm hashes the string 'password' to 5f4dcc3b5aa765d61d8327deb882cf99",
            "which hash function turns 'hello' into 5d41402abc4b2a76b9719d911017c592",
            "identify the algorithm that hashes 'admin' to 21232f297a57a5a743894a0e4a801fc3",
            "what hash produces e10adc3949ba59abbe56e057f20f883e from the input '123456'",
            "figure out which algorithm maps 'letmein' to 0d107d09f5bbe40cade3de5c71e9e9b7",
            "the string 'test' hashes to a94a8fe5ccb19ba61c4c0873d391e987982fbbd3 — what algorithm is that?",
            "identify the hash that turns 'root' into 4813494d137e1631bba301d5acab6e7bb7aa74ce1185d456565ef51d737677b2",
            "which digest function produces 098f6bcd4621d373cade4e832627b4f6 from 'test'",
        ]),
        HASH_COMPUTE.examples([
            "compute the SHA-256 of the string hello world",
            "compute the MD5 hash of the string 'correct horse battery staple'",
            "compute the SHA-1 digest of the string 'commit message v1'",
            "hash the string 'foo' with SHA-512",
            "give me the SHA-256 digest of 'the quick brown fox jumps over the lazy dog'",
            "what's the MD5 of 'release-2026.07'?",
            "hash 'user:alice|role:admin' with SHA-512",
            "compute a SHA-1 checksum for the string 'v3.1.4-rc2'",
        ]),
        TOTP.examples([
            "generate the current TOTP code for the credential named github",
            "what's the current 6-digit 2FA code for the credential named aws-root",
            "give me the current one-time password for the credential named gitlab",
            "compute the TOTP for the credential named okta",
            "I need the current authenticator code for the credential named cloudflare",
            "read off the live 2FA digits for the credential named vault-admin",
            "what one-time passcode does the credential named bitwarden show right now?",
            "grab the current TOTP for the credential named npm-registry",
        ]),
        // Crypto primitives (8) — all safe (pure compute, no external effect)
        AEAD_ENCRYPT.examples([
            "encrypt the text 'meet at noon' with AES-256-GCM, key 0011223344556677889900aabbccddeeff00112233445566778899aabbccddee, nonce 0102030405060708090a0b0c",
            "encrypt the text 'top secret' with ChaCha20-Poly1305, key 0011223344556677889900aabbccddeeff00112233445566778899aabbccddee, nonce 000102030405060708090a0b",
            "encrypt the text 'launch code 7' with AES-128-GCM, key 000102030405060708090a0b0c0d0e0f, nonce 0102030405060708090a0b0c",
            "encrypt the text 'confidential memo' with AES-128-GCM, key aabbccddeeff00112233445566778899, nonce 00112233445566778899aabb",
            "seal 'wire the funds' using AES-256-GCM with key cafebabedeadbeef0011223344556677cafebabedeadbeef0011223344556677 and nonce a1b2c3d4e5f60718293a4b5c",
            "can you AES-128-GCM encrypt 'ping ok' with the key 112233445566778899aabbccddeeff00 and nonce ffeeddccbbaa99887766554433",
            "encrypt 'session token grant' with ChaCha20-Poly1305, key ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100, nonce 0b0a090807060504030201ff",
            "run AES-256-GCM over the message 'rotate the master key' with key 1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100 and nonce 00112233445566778899aabb",
        ]),
        AEAD_DECRYPT.examples([
            "decrypt the AES-128-GCM ciphertext 3a7f9c2e1b8d with key 000102030405060708090a0b0c0d0e0f, nonce 0102030405060708090a0b0c, and no AAD",
            "decrypt the ChaCha20-Poly1305 ciphertext 9f3c7a2e with key 0011223344556677889900aabbccddeeff00112233445566778899aabbccddee, nonce 000102030405060708090a0b, no AAD",
            "decrypt the AES-256-GCM ciphertext 5e1a3f9c with key 0011223344556677889900aabbccddeeff00112233445566778899aabbccddee, nonce 0102030405060708090a0b0c, and no AAD",
            "decrypt the AES-128-GCM ciphertext c0ffee11 with key aabbccddeeff00112233445566778899, nonce 00112233445566778899aabb, no AAD",
            "open the AES-256-GCM ciphertext 7b2d4e6f8a0c with key 1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100, nonce a1b2c3d4e5f60718293a4b5c, no AAD",
            "can you decrypt ChaCha20-Poly1305 blob feedface90ab using key ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100 and nonce 0b0a090807060504030201ff with no AAD",
            "recover the plaintext from AES-128-GCM ciphertext 1234abcd5678 with key 112233445566778899aabbccddeeff00, nonce ffeeddccbbaa99887766554433, and no AAD",
            "decrypt the AES-256-GCM ciphertext 9a8b7c6d5e4f302112034455667788990a1b2c3d4e5f60718293a4b5c6d7e8f90 with key cafebabedeadbeef0011223344556677cafebabedeadbeef0011223344556677, nonce 00112233445566778899aabb, no AAD",
        ]),
        HMAC_COMPUTE.examples([
            "compute the HMAC-SHA256 of the message 'transfer 500 to alice' with key s3cr3t-key",
            "generate an HMAC-SHA256 tag for the data 'order#42' with key my-signing-key",
            "what's the HMAC-SHA1 of the string 'payload' with key topsecret",
            "compute the HMAC-SHA512 of the message 'hello' with key abc123def456",
            "give me the HMAC-SHA256 of 'GET /api/v2/balance' keyed with webhook-secret-9f",
            "HMAC-SHA512 the message 'nonce=8823;amount=1000' using the key hunter2-rotated",
            "what HMAC-SHA1 tag does 'cookie=session42' produce with key legacy-mac-key",
            "authenticate 'invoice-2026-0714' with HMAC-SHA256 under the key billing-hmac-key",
        ]),
        SIGNATURE_VERIFY.examples([
            "verify the ed25519 signature 3a7f9c2e1b8d4f60a5e8c1d2b3f4a69708172635445566778899aabbccddeeff over the message 'release v1.2' with public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "check whether the ed25519 signature 9f8e7d6c5b4a392817063a4b5c6d7e8f0091a2b3c4d5e6f708192a3b4c5d6e7f over 'deploy approved' is valid with public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "verify the ed25519 signature c0ffee11d00dfeedbaadf00d1234567890abcdef00112233445566778899aabb over 'contract terms' with public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "confirm the ed25519 signature feedface8badf00dcafebabe0123456789abcdef112233445566778899aabbcc over 'audit log' with public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "is the ed25519 signature 0011223344556677889900aabbccddeeff112233445566778899aabbccddeeff0 over 'firmware image v9' authentic given public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "validate the ed25519 signature a5b4c3d2e1f00918273645546372819aabbccddee0011223344556677889900ab on the message 'grant admin token' using public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "verify signature 7788990011223344556677889900aabbccddeeff00112233445566778899aabb over 'ceasefire at dawn' with the ed25519 public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
            "tell me if the ed25519 signature 1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809 matches 'ledger close 2026-07' under public key -----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----",
        ]),
        SIGNATURE_SIGN.examples([
            "sign the message 'release v1.2.0' with the ed25519 credential release-signing-key",
            "produce an ed25519 signature over 'deploy approved' using credential deploy-key",
            "sign the data 'order#42 confirmed' using the p256_sha256 algorithm with credential 'signing-key'",
            "create an ed25519 signature for the message 'audit entry 7' using credential audit-key",
            "sign 'firmware image v9 checksum ok' with the p256_sha256 credential firmware-key",
            "can you ed25519-sign the message 'ceasefire at dawn' using credential ops-key",
            "generate a p256_sha256 signature over 'ledger close 2026-07' with credential 'ledger-key'",
            "sign the string 'grant admin token' using the ed25519 credential admin-issuer-key",
        ]),
        KDF_DERIVE.examples([
            "derive a 32-byte key from the password 'hunter2' with PBKDF2-HMAC-SHA256, hex salt 00112233445566778899aabbccddeeff",
            "derive a 32-byte key from the passphrase 'correct horse battery' with scrypt, hex salt a1b2c3d4e5f60718",
            "derive a 32-byte key from the password 's3cr3t' with Argon2id, hex salt deadbeefcafebabe",
            "derive a 64-byte key from the password 'p@ssw0rd' with PBKDF2-HMAC-SHA256, hex salt 0f1e2d3c4b5a6978",
            "stretch the passphrase 'summer sky lantern' into a 16-byte key using scrypt with hex salt 5a5a5a5a12345678",
            "give me a 32-byte Argon2id key from the password 'Tr0ub4dour&3' with hex salt 99887766554433221100ffee",
            "derive a 48-byte key with PBKDF2-HMAC-SHA256 from the password 'letmein-2026' and hex salt cafef00dbaadf00d",
            "turn the passphrase 'north star river' into a 64-byte scrypt key using hex salt 0123456789abcdeffedcba98",
        ]),
        HKDF_EXTRACT.examples([
            "HKDF-extract with SHA-256 from the hex IKM 0b0b0b0b0b0b0b0b0b0b0b and hex salt 000102030405060708090a0b0c",
            "HKDF-extract with SHA-256: condense the hex shared secret deadbeefcafebabedeadbeefcafebabe into a PRK using hex salt 0102030405",
            "HKDF-extract with SHA-256 from the hex IKM aabbccddeeff00112233445566778899 and an empty salt",
            "HKDF-extract with SHA-256 from the hex Diffie-Hellman output 1122334455667788 using hex salt 99aabbccdd",
            "run HKDF-extract with SHA-256 over the hex IKM ccddeeff00112233445566778899aabb with hex salt fedcba9876543210",
            "extract a PRK with SHA-256 from the hex ECDH secret 5a5a5a5a6b6b6b6b7c7c7c7c8d8d8d8d and hex salt a0b1c2d3",
            "HKDF-extract SHA-256: take the hex IKM 00ff00ff00ff00ff and salt it with hex 1234567890",
            "condense the hex key material 9988776655443322110000112233445566 into a PRK via HKDF-extract SHA-256 with an empty salt",
        ]),
        HKDF_EXPAND_LABEL.examples([
            "HKDF-Expand-Label with SHA-256 the hex secret 33ad0a1c607ec03b09e6cd9893680ce2 with label 'c hs traffic' to 32 bytes",
            "derive a 16-byte TLS 1.3 key with SHA-256 from the hex PRK aabbccddeeff00112233445566778899 with label 'key'",
            "run HKDF-Expand-Label with SHA-256 on the hex secret 0102030405060708 with label 'finished' for 32 bytes",
            "expand the hex secret deadbeefcafebabe into a 12-byte 'iv' with HKDF-Expand-Label using SHA-256",
            "HKDF-Expand-Label with SHA-256 from the hex secret 112233445566778899aabbccddeeff00 with label 's hs traffic' to 32 bytes",
            "give me a 32-byte 'c ap traffic' secret via HKDF-Expand-Label SHA-256 from the hex PRK 5a5a5a5a6b6b6b6b7c7c7c7c8d8d8d8d",
            "run HKDF-Expand-Label SHA-256 on hex secret cafef00dbaadf00dcafef00dbaadf00d with label 'res master' for 32 bytes",
            "expand the hex secret 0011223344556677 into a 16-byte 'quic key' using HKDF-Expand-Label with SHA-256",
        ]),
        // Hash state tools (3) — all safe (compute)
        HASH_STATE_INIT.examples([
            "start a streaming SHA-256 hash",
            "begin an incremental SHA-512 computation",
            "open a new streaming hash with SHA-1",
            "initialize an incremental SHA-256 hash for feeding data in chunks",
            "spin up a fresh streaming MD5 hash",
            "I want to hash a big file piece by piece with SHA-512 — set up the state",
            "create a new incremental SHA-1 hashing session",
            "open a streaming SHA-256 context so I can push data as it arrives",
        ]),
        HASH_STATE_UPDATE.examples([
            "feed the hex 48656c6c6f into hash state hs-1",
            "update hash state hs-1 with the string 'more data'",
            "absorb the bytes deadbeef into hash state hs-2",
            "add the chunk 'final part' to hash state hs-1",
            "push the hex cafebabe0011 into hash state hs-4",
            "append the string 'row 2048 of the export' to hash state hs-3",
            "feed the next block 9988776655443322 into hash state hs-5",
            "keep hashing — add 'appended log line' to hash state hs-2",
        ]),
        HASH_STATE_FINALIZE.examples([
            "finalize hash state hs-1 and give me the digest",
            "conclude hash state hs-2 and emit the result",
            "finish hash state hs-1",
            "give me the final digest of hash state hs-3",
            "close out hash state hs-4 and return the hash",
            "wrap up hash state hs-5 and show me the hex digest",
            "I'm done streaming — finalize hash state hs-2",
            "seal hash state hs-6 and hand me the checksum",
        ]),
        // Byte tools (4) — all safe (compute)
        BYTES_TRANSCODE.examples([
            "convert the hex string 89504e470d0a1a0a0000000d49484452 to base64",
            "transcode the base64 value SGVsbG8sIFdvcmxkIQ== to hex",
            "re-encode the utf8 string 'the quick brown fox' to base64url",
            "convert the base64url string aGVsbG8td29ybGQ to hex",
            "turn the hex ffd8ffe000104a464946 into base64url",
            "decode the base64 value VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZyBhbmQga2VlcHMgb24gcnVubmluZw== to a utf8 string",
            "re-encode the utf8 text 'café menu' as hex",
            "convert the base64url string SGVsbG8gd29ybGQ to base64",
        ]),
        BYTES_PACK.examples([
            "pack the integers 3149642683, 1094795585, and 2882395258 as big-endian unsigned 32-bit ints",
            "pack the float 3.14 and the shorts 7 and 9 into bytes with format '>fHH'",
            "pack the values 46021 and 12873 as little-endian u16s",
            "pack the values 3405691582 and 48350 into a binary buffer with the format '>IH'",
            "pack the double 2.71828 with the format '<d'",
            "encode the signed bytes -47, 112, and -9 with the format '>bbb'",
            "pack the value 65535 and the float 1.5 as little-endian with format '<Hf'",
            "serialize the u64 4294967296 and the u8 200 using format '>QB'",
        ]),
        BYTES_UNPACK.examples([
            "unpack the hex-encoded bytes 3ac91f7e5b2048d1 as two little-endian u32s, format '<II'",
            "unpack the hex-encoded bytes a83f5c19 as two big-endian u16s, format '>HH'",
            "unpack the hex-encoded bytes 6f12833e as a little-endian float, format '<f'",
            "unpack the hex-encoded bytes 39d4a10800000000 as a signed 64-bit integer, format '<q'",
            "unpack the hex bytes 401921fb54442d18 as a big-endian double, format '>d'",
            "read the hex 9c2f04e7 as four unsigned bytes, format '>BBBB'",
            "decode the hex-encoded bytes b3f10a4c as a big-endian u32, format '>I'",
            "unpack the hex 7b19c4a891e3f5a2c47d1b60 into a u32 and an i64, format '<Iq'",
        ]),
        BYTES_XOR.examples([
            "xor the hex a1b2c3d4e5f60718293a4b5c6d7e8f90 and 0f1e2d3c4b5a69788796a5b4c3d2e1f0",
            "xor the hex blocks 3c8f1ad57e29b0c46f81d2a3905e7bc191e4072db8a6f53c1d4e8b09a7f26d3e and 5f4e3d2c1b0a998877665544332211009988776655443322110ffeeddccbbaa9",
            "xor the keystream 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d with the ciphertext deadbeefcafebabe0011223344556677",
            "xor the hex cafebabedeadbeef0123456789abcdef and fedcba9876543210ffeeddccbbaa9988",
            "xor 7d3f9a02c815be64f0a19d3c5e82740b against 2b6ef1c40953a7d8e14c60ba39f5127e",
            "what do you get when you XOR 1234567890abcdef and c73e91a45db8206f",
            "mask the hex 5e6f7a8b9c0d1e2f with the one-time pad a5b4c3d2e1f00918",
            "xor the 32-byte hex 00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f and ffeeddccbbaa998877665544332211000f1e2d3c4b5a69788796a5b4c3d2e1f0",
        ]),

        // Code execution (5) — JavaScript on the embedded sandboxed engine
        CODE_RUN.risky().examples([
            "run some JavaScript that prints the first 10 Fibonacci numbers",
            "evaluate this JavaScript expression that sums an array of numbers",
            "write and run JS that parses the JSON {\"name\":\"Ada\",\"age\":36,\"admin\":true} and prints its keys",
            "execute a JavaScript snippet that reverses the string 'hello world'",
            "run JS to compute the factorial of 12",
            "use JavaScript to filter the array [3, 8, 11, 14, 5, 20, 7] down to just the even numbers",
            "run a quick JavaScript to format a Unix timestamp as an ISO date string",
            "evaluate JavaScript that counts the vowels in 'sandbox engine'",
        ]),
        CODE_SESSION_OPEN.risky().examples([
            "start a persistent JavaScript session",
            "open a JS session so variables are kept between calls",
            "spin up a stateful JavaScript code session",
            "open a code session for javascript",
            "give me a persistent JS REPL so I can build up state across steps",
            "create a JavaScript sandbox session I can keep reusing",
            "open a long-lived js session",
            "start a code session (javascript) that remembers my definitions",
        ]),
        CODE_SESSION_EXEC.risky().examples([
            "run `let x = 42` in code session js-1",
            "execute `const arr = [1, 2, 3, 4]` in code session js-1",
            "run `function add(a, b) { return a + b; }` in code session js-2",
            "run `console.log(x + 1)` in code session js-1",
            "in code session js-3, run `arr.map(n => n * n)`",
            "execute `const total = arr.reduce((a, b) => a + b, 0)` in code session js-1",
            "run `add(x, 8)` in code session js-2",
            "evaluate `JSON.stringify({ x, total })` in code session js-1",
        ]),
        CODE_SESSION_LIST.risky().examples([
            "list my open code sessions",
            "which code execution sessions are running",
            "show the active code sandboxes",
            "how many code sessions do I have open",
            "enumerate my open JavaScript sessions",
            "what code session ids are currently alive?",
            "give me a rundown of every running code sandbox",
            "are there any code sessions still open?",
        ]),
        CODE_SESSION_CLOSE.risky().examples([
            "close code session sess_3f9a2c",
            "terminate the JavaScript session js-1",
            "end code execution session build-7",
            "shut down the code sandbox, session id abc123",
            "drop code session js-4, I'm done with it",
            "please close out code session sess_9d2e1a",
            "discard the state and end code session js-2",
            "kill the code session with id run-42",
        ]),
        // Subagent (1) — high-risk (delegated agency)
        SUBAGENT.risky().examples([
            "spawn a subagent to research the best Rust logging crate and report back",
            "delegate the task of finding every TODO comment under the src directory to a subagent",
            "have a subagent investigate why the build is failing",
            "launch a nested agent to draft the migration plan",
            "kick off a subagent to audit the codebase for unwrap() calls and list them",
            "delegate writing the unit tests for the parser module to a subagent",
            "have a subagent research the top three Rust web frameworks and recommend one for a REST API",
            "spawn an agent to trace where this config value is read across the repo",
        ]),
    ];
    TOOLS
}

#[cfg(test)]
mod alias_tests {
    use super::*;

    /// Every registered tool maps to a real category (never the `"Other"`
    /// fallback), so the deterministic tool-catalog summary groups the whole
    /// catalog. Fails loudly if a new tool is added without a `category_for` arm.
    #[test]
    fn every_tool_has_a_category() {
        for t in all_tools() {
            assert_ne!(
                category_for(t.name),
                "Other",
                "tool {:?} has no category_for arm",
                t.name
            );
        }
    }

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
