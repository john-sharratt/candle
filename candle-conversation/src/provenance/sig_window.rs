//! A turn's stored signature window, shared and decoded only on demand.
//!
//! The substrate keeps every turn's wide-Q signatures encoded, one shared
//! buffer per turn. A belief scan needs each turn's length, its per-token width
//! and a residency fingerprint on every pass, and its decoded tokens only when
//! the turn has to be uploaded to the GPU gallery arena or scored on the host.
//! [`SigWindow`] carries the first three without decoding anything, so a
//! gallery resident on the device costs no host memory beyond the encoded
//! buffers the substrate already holds.
//!
//! Nothing decoded is kept between scans. A decoded window is a `WideQSig` and a
//! separate words allocation per token, several times the encoded size, so
//! keeping the corpus decoded would hold a second, larger copy of it on the
//! host — gigabytes at ~10 M tokens — growing with the corpus without bound.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::wide_sig::{
    active_fold, decode_wide_sigs, fold_admits, parse_wide_header, warn_fold_mismatch, WideQSig,
};

/// One turn's encoded signature window, validated and sized, not decoded.
#[derive(Clone, Debug)]
pub struct SigWindow {
    blob: Arc<[u8]>,
    header_len: usize,
    n_tokens: usize,
    width: usize,
}

/// Set once a refused fold has been reported, so a scan that meets the same
/// foreign records on every pass says so once rather than on every pass.
static FOLD_MISMATCH_REPORTED: AtomicBool = AtomicBool::new(false);

impl SigWindow {
    /// The window `blob` stores, or `None` for one a scan must not read: a
    /// foreign or truncated blob, an empty window, or a record folded under a
    /// different fold than this process's — the records
    /// [`decode_wide_sigs_for_scoring`](super::decode_wide_sigs_for_scoring)
    /// refuses.
    pub fn new(blob: Arc<[u8]>) -> Option<Self> {
        let h = parse_wide_header(&blob)?;
        if let Some(expected) = active_fold() {
            if !fold_admits(h.params, expected) {
                if !FOLD_MISMATCH_REPORTED.swap(true, Ordering::Relaxed) {
                    warn_fold_mismatch(h.params, expected);
                }
                return None;
            }
        }
        let width = h.n_heads as usize * h.wph;
        if h.n_tokens == 0 || width == 0 || blob.len() < h.header_len + h.n_tokens * width * 8 {
            return None;
        }
        Some(Self {
            header_len: h.header_len,
            n_tokens: h.n_tokens,
            width,
            blob,
        })
    }

    /// Tokens in the window.
    pub fn len(&self) -> usize {
        self.n_tokens
    }

    /// Always `false`: an empty window is refused by [`Self::new`].
    pub fn is_empty(&self) -> bool {
        self.n_tokens == 0
    }

    /// `u64` words per token — the same for every token of a window.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The window's residency fingerprint: the stored buffer's address, its
    /// length, and its first and last payload words.
    ///
    /// The substrate replaces a turn's buffer when it rewrites the blob and
    /// never mutates one in place, so the address changes exactly when the
    /// content can — a scan against unchanged turns sees the same fingerprint
    /// and uploads nothing. The words are the guard against a freed buffer's
    /// address being reused for a different window.
    pub fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        (self.blob.as_ptr() as usize).hash(&mut h);
        self.n_tokens.hash(&mut h);
        let payload = &self.blob[self.header_len..self.header_len + self.n_tokens * self.width * 8];
        payload[..8].hash(&mut h);
        payload[payload.len() - 8..].hash(&mut h);
        h.finish()
    }

    /// The window's tokens, decoded now. Nothing keeps the result: a caller
    /// holds it only as long as its upload or host scan needs it.
    pub fn decode(&self) -> Vec<WideQSig> {
        decode_wide_sigs(&self.blob).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::encode_wide_sigs;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    fn blob(sigs: &[WideQSig]) -> Arc<[u8]> {
        Arc::from(encode_wide_sigs(sigs))
    }

    /// The window reports its length and width from the header alone, and
    /// decodes to exactly the signatures that were encoded.
    #[test]
    fn a_window_is_sized_from_its_header_and_decodes_to_what_was_stored() {
        let sigs = vec![
            sig(0xAAAA_AAAA_AAAA_AAAA),
            sig(0x5555_5555_5555_5555),
            sig(7),
        ];
        let w = SigWindow::new(blob(&sigs)).expect("a valid window");
        assert_eq!(w.len(), 3);
        assert_eq!(w.width(), 24);
        assert!(!w.is_empty());
        assert_eq!(w.decode(), sigs);
    }

    /// The fingerprint is the buffer's: two reads of one stored buffer agree,
    /// and an equal window in another buffer — a rewritten blob — does not.
    #[test]
    fn the_fingerprint_follows_the_stored_buffer() {
        let sigs = vec![sig(1), sig(2)];
        let stored = blob(&sigs);
        let a = SigWindow::new(Arc::clone(&stored)).unwrap();
        let b = SigWindow::new(Arc::clone(&stored)).unwrap();
        assert_eq!(
            a.fingerprint(),
            b.fingerprint(),
            "one buffer, one fingerprint"
        );
        let rewritten = SigWindow::new(blob(&sigs)).unwrap();
        assert_ne!(
            a.fingerprint(),
            rewritten.fingerprint(),
            "a rewritten blob is a new buffer and must re-upload"
        );
    }

    /// A window a scan must not read is refused: empty, truncated or foreign.
    #[test]
    fn empty_truncated_and_foreign_blobs_are_refused() {
        assert!(SigWindow::new(blob(&[])).is_none(), "empty");
        let full = encode_wide_sigs(&[sig(3), sig(4)]);
        let truncated: Arc<[u8]> = Arc::from(&full[..full.len() - 1]);
        assert!(SigWindow::new(truncated).is_none(), "truncated");
        let foreign: Arc<[u8]> = Arc::from(&b"not a signature record"[..]);
        assert!(SigWindow::new(foreign).is_none(), "foreign");
    }
}
