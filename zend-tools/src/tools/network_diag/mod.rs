//! Network diagnostic tools: `dns_lookup`, `ping_icmp`, `trace_route`,
//! `port_scan`, `ip_scan`, `host_info`.
//!
//! Thin wrappers around OS utilities (`nslookup`/`dig`, `ping`, `traceroute`,
//! `nc`/`nmap`, `arp-scan`, `host`) that invoke the subprocess and parse
//! structured output.
//!
//! # Error codes
//!
//! All six tools share [`DiagError`] with three codes:
//!
//! | Code | Cause |
//! |------|-------|
//! | `host_not_found` | DNS resolution returned no records |
//! | `operation_failed` | Subprocess error, parse failure, or OS-level denial |
//! | `not_supported` | Requested operation is not available on this platform |
//!
//! # Shared utilities
//!
//! - [`extract_ip`] — pulls the first IPv4 address out of subprocess output
//! - [`extract_rtt`] — extracts the first round-trip time value (ms) from output

use thiserror::Error;
use crate::ToolError;

pub mod dns_lookup;
pub mod ping_icmp;
pub mod trace_route;
pub mod port_scan;
pub mod ip_scan;
pub mod host_info;

pub use dns_lookup::DNS_LOOKUP;
pub use ping_icmp::PING_ICMP;
pub use trace_route::TRACE_ROUTE;
pub use port_scan::PORT_SCAN;
pub use ip_scan::IP_SCAN;
pub use host_info::HOST_INFO;

#[derive(Debug, Error)]
pub enum DiagError {
    #[error("host not found: {0}")]
    HostNotFound(String),
    #[error("operation failed: {0}")]
    Failed(String),
    #[error("not supported: {0}")]
    NotSupported(String),
}

impl ToolError for DiagError {
    fn code(&self) -> &'static str {
        match self {
            DiagError::HostNotFound(_) => "host_not_found",
            DiagError::Failed(_) => "operation_failed",
            DiagError::NotSupported(_) => "not_supported",
        }
    }
}

pub fn extract_ip(s: &str) -> Option<String> {
    let re = regex::Regex::new(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b").ok()?;
    re.find(s).map(|m| m.as_str().to_string())
}

pub fn extract_rtt(s: &str) -> Option<f64> {
    let re = regex::Regex::new(r"(\d+(?:\.\d+)?)\s*ms").ok()?;
    re.find(s)?.as_str().split_whitespace().next()?.parse().ok()
}
