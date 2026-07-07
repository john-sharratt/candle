//! All 93 tool implementations, organised into one module per logical group.
//!
//! Each tool module follows the same pattern:
//! - a `Request` struct (`Deserialize + JsonSchema + Validate`)
//! - a `Response` struct (`Serialize`)
//! - a group-level `Error` enum (`ToolError`) defined in `mod.rs`
//! - a zero-state unit struct implementing `Tool`
//! - a `pub const FOO: RegisteredTool = RegisteredTool::new::<FooImpl>();`
//!
//! Tools are gathered into the static table in `crate::registry::register_all`.
//!
//! # Group summary
//!
//! | Module | Tools | Notes |
//! |--------|-------|-------|
//! | `bytes` | `bytes_{transcode,pack,unpack,xor}` | struct.pack/unpack semantics |
//! | `calculator` | `calculator` | evalexpr; no eval code path |
//! | `code` | `code_run`, `code_session_*` | Python/Node REPL via subprocess |
//! | `credentials` | `credential_{save,list,delete}` | In-memory typed credential store |
//! | `crypto` | `aead_*`, `hmac_compute`, `signature_*`, `kdf_derive`, `hkdf_*` | RustCrypto |
//! | `datetime` | `datetime` | chrono + chrono-tz; stateless |
//! | `file` | `file_{write,read,edit,list,delete,present}` | VFS tools |
//! | `hash` | `hash_compute`, `hash_scan` | SHA2/SHA3/BLAKE3/MD5 |
//! | `hash_state` | `hash_state_{init,update,finalize}` | Streaming hash for large data |
//! | `http_session` | `http_session_{open,request,list,close}` | reqwest; cookie jar |
//! | `network_diag` | `dns_lookup`, `ping_icmp`, `trace_route`, `port_scan`, `ip_scan`, `host_info` | subprocess |
//! | `notes` | `notes_{write,read,search,list}` | In-memory KV with FTS |
//! | `random` | `random` | rand crate; real OS entropy |
//! | `remote_fs` | `remote_fs_session_*` (10 tools) | SFTP via ssh2 |
//! | `sql_session` | `sql_session_{open,query,list,close}` | rusqlite |
//! | `ssh` | `ssh_session_{open,exec,exec_async,poll,list,close}` | ssh2 |
//! | `subagent` | `sub_run` | Calls `SubagentRunner` trait impl injected by daemon |
//! | `tcp_session` | `tcp_session_{open,send,recv,list,close}` | Raw TCP; hex wire format |
//! | `telnet` | `telnet_session_{open,send,list,close}` | Raw TCP; prompt-regex |
//! | `tls_session` | `tls_session_{open,send,recv,list,close}` | native-tls |
//! | `totp` | `totp` | totp-rs; RFC 6238 |
//! | `udp_session` | `udp_session_{open,send,recv,list,close}` | UDP socket; hex wire format |
//! | `unit_convert` | `unit_convert` | Static unit table; temperature special-cased |
//! | `weather` | `weather` | Open-Meteo (no API key) |
//! | `web_fetch` | `web_fetch` | reqwest + readability extractor |
//! | `web_search` | `web_search` | Tavily API |

pub mod bytes;
pub mod calculator;
pub mod code;
pub mod credentials;
pub mod crypto;
pub mod datetime;
pub mod file;
pub mod hash;
pub mod hash_state;
pub mod http_session;
pub mod network_diag;
pub mod notes;
pub mod random;
pub mod remote_fs;
pub mod sql_session;
pub mod ssh;
pub mod subagent;
pub mod tcp_session;
pub mod telnet;
pub mod tls_session;
pub mod totp;
pub mod udp_session;
pub mod unit_convert;
pub mod weather;
pub mod web_fetch;
pub mod web_search;

use schemars::JsonSchema;
use serde::Serialize;

// Schema-only mirrors of the encoding names the per-module decoders accept. Tool
// request `*_encoding` fields stay `Option<String>` for decoding; these types are
// referenced via `#[schemars(with = "Option<…>")]` purely so the generated JSON
// schema carries a real `"enum"` of the allowed values. `BytesEncoding` lives in
// `bytes` next to `decode_bytes`.

/// Encoding format for crypto/hash byte inputs and outputs.
#[derive(JsonSchema, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DataEncoding {
    Text,
    Hex,
    Base64,
}

/// Encoding format for received session bytes.
#[derive(JsonSchema, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RecvEncoding {
    Auto,
    Hex,
}
