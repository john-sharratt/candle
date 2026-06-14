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
}

impl RegisteredTool {
    /// Build a registration for a `Tool` impl.
    pub const fn new<T: Tool>() -> Self {
        Self {
            name: T::NAME,
            description: T::DESCRIPTION,
            schema: tool_schema::<T>,
            run: tool_run::<T>,
            confirmation: tool_confirmation::<T>,
        }
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

/// Look up a tool by name.
pub fn find(name: &str) -> Option<&'static RegisteredTool> {
    all_tools().iter().find(|t| t.name == name)
}

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
    static TOOLS: &[RegisteredTool] = &[
        // Shared tools (7)
        DATETIME,
        CALC,
        UNIT_CONVERT,
        RANDOM,
        WEB_SEARCH,
        WEB_FETCH,
        WEATHER,
        // File tools (6)
        FILE_WRITE,
        FILE_READ,
        FILE_EDIT,
        FILE_LIST,
        FILE_DELETE,
        FILE_PRESENT,
        // Notes tools (4)
        NOTES_WRITE,
        NOTES_READ,
        NOTES_SEARCH,
        NOTES_LIST,
        // Credential tools (3)
        CREDENTIAL_SAVE,
        CREDENTIAL_LIST,
        CREDENTIAL_DELETE,
        // SSH tools (6)
        SSH_SESSION_OPEN,
        SSH_SESSION_EXEC,
        SSH_SESSION_EXEC_ASYNC,
        SSH_SESSION_POLL,
        SSH_SESSION_LIST,
        SSH_SESSION_CLOSE,
        // Telnet tools (4)
        TELNET_SESSION_OPEN,
        TELNET_SESSION_SEND,
        TELNET_SESSION_LIST,
        TELNET_SESSION_CLOSE,
        // HTTP session tools (4)
        HTTP_SESSION_OPEN,
        HTTP_SESSION_REQUEST,
        HTTP_SESSION_LIST,
        HTTP_SESSION_CLOSE,
        // TCP tools (5)
        TCP_SESSION_OPEN,
        TCP_SESSION_SEND,
        TCP_SESSION_RECV,
        TCP_SESSION_LIST,
        TCP_SESSION_CLOSE,
        // UDP tools (5)
        UDP_SESSION_OPEN,
        UDP_SESSION_SEND,
        UDP_SESSION_RECV,
        UDP_SESSION_LIST,
        UDP_SESSION_CLOSE,
        // TLS tools (5)
        TLS_SESSION_OPEN,
        TLS_SESSION_SEND,
        TLS_SESSION_RECV,
        TLS_SESSION_LIST,
        TLS_SESSION_CLOSE,
        // SQL tools (4)
        SQL_SESSION_OPEN,
        SQL_SESSION_QUERY,
        SQL_SESSION_LIST,
        SQL_SESSION_CLOSE,
        // Remote FS tools (10)
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
        // Network diagnostics (6)
        DNS_LOOKUP,
        PING_ICMP,
        TRACE_ROUTE,
        PORT_SCAN,
        IP_SCAN,
        HOST_INFO,
        // Security utilities (3)
        HASH_SCAN,
        HASH_COMPUTE,
        TOTP,
        // Crypto primitives (8)
        AEAD_ENCRYPT,
        AEAD_DECRYPT,
        HMAC_COMPUTE,
        SIGNATURE_VERIFY,
        SIGNATURE_SIGN,
        KDF_DERIVE,
        HKDF_EXTRACT,
        HKDF_EXPAND_LABEL,
        // Hash state tools (3)
        HASH_STATE_INIT,
        HASH_STATE_UPDATE,
        HASH_STATE_FINALIZE,
        // Byte tools (4)
        BYTES_TRANSCODE,
        BYTES_PACK,
        BYTES_UNPACK,
        BYTES_XOR,
        // Code execution (5)
        CODE_RUN,
        CODE_SESSION_OPEN,
        CODE_SESSION_EXEC,
        CODE_SESSION_LIST,
        CODE_SESSION_CLOSE,
        // Subagent (1)
        SUBAGENT,
    ];
    TOOLS
}
