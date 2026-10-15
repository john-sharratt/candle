//! zend's ingest-cancel latch.
//!
//! The engine keeps one latch per substrate; this is the one zend's substrate
//! carries. It exists before the engine does — handed to the model builder at
//! load — so a shutdown that arrives while the model is still loading reaches
//! the ingest the load is about to start, and every ingest loop (repo map, code
//! reading, branch priming, raw reads) polls it without holding the engine,
//! as does every long load step between its units. It is raised the moment
//! the process's stop is asked (`crate::tenant`), and again by
//! `ZendSession::shutdown`.
//! Another tenant on the same engine has its own, which this never touches.

use std::sync::OnceLock;

use candle_conversation::CancelToken;

/// The token zend's substrate is built with.
pub fn token() -> &'static CancelToken {
    static TOKEN: OnceLock<CancelToken> = OnceLock::new();
    TOKEN.get_or_init(CancelToken::new)
}

/// Whether a shutdown has asked zend's ingest to stop.
pub fn cancelled() -> bool {
    token().is_cancelled()
}
