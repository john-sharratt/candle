//! Asking origin for its branch tips without the git binary: protocol v2's
//! `ls-refs` over HTTPS, one small request, made every couple of seconds by
//! the daemon's origin watcher (`docs/zend_branch_ingest.md` §4.1).
//!
//! Only the bytes are built and read here — the request body, the
//! advertisement check, the answer, and the URL the exchange goes to. The
//! daemon carries them over HTTP and presents its own credentials.

pub mod ls_refs;
pub mod pkt_line;
pub mod url;

use thiserror::Error;

pub use ls_refs::{advertises_ls_refs, parse_response, request, BranchTip};
pub use url::{https_url, ProbeUrl};

/// Why an answer could not be read.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProbeError {
    /// The bytes are not what the protocol allows.
    #[error("the answer is malformed: {0}")]
    Malformed(String),
    /// The server answered with an `ERR` packet.
    #[error("origin refused the request: {0}")]
    Refused(String),
}
