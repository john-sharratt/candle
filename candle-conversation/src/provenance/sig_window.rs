//! A turn's stored signature window, sized from its header and decoded only on
//! demand.
//!
//! A belief scan needs each turn's length, its per-token width and a residency
//! fingerprint on every pass, and its decoded tokens only when the turn has to
//! be uploaded to the GPU gallery arena or scored on the host. [`SigWindow`]
//! carries the first three from the record's header ([`SigHead`]) and reaches
//! the tokens only when asked: from the bytes this process still holds, or by
//! reading the record back from the log ([`StoredSig`]). A gallery resident on
//! the device therefore costs no host memory at all.
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
    active_fold, decode_wide_sigs, fold_admits, parse_wide_header, warn_fold_mismatch, FoldParams,
    WideQSig,
};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::RecordLoc;
use crate::substrate::PayloadSource;

/// What a window's record header says: enough to size the window and judge
/// whether a scan may read it, without its tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigHead {
    header_len: usize,
    n_tokens: usize,
    width: usize,
    params: FoldParams,
}

impl SigHead {
    /// The header `bytes` begin with, or `None` for a foreign or unknown-version
    /// record. `bytes` need hold no more than the header.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let h = parse_wide_header(bytes)?;
        Some(Self {
            header_len: h.header_len,
            n_tokens: h.n_tokens,
            width: h.n_heads as usize * h.wph,
            params: h.params,
        })
    }

    /// The payload length a record with this header must have to hold every
    /// token it states.
    fn needs(&self) -> usize {
        self.header_len + self.n_tokens * self.width * 8
    }
}

/// A window's record in the log: what reads it back.
#[derive(Clone, Debug)]
pub struct StoredSig {
    /// The substrate's reader for stored records.
    pub source: Arc<PayloadSource>,
    /// The record holding the window.
    pub id: RecordIdentity,
    /// Where the substrate last placed it.
    pub loc: RecordLoc,
}

/// Where a window's bytes are.
#[derive(Clone, Debug)]
enum SigBytes {
    Held(Arc<[u8]>),
    Stored(StoredSig),
}

/// One turn's signature window, validated and sized, not decoded.
#[derive(Clone, Debug)]
pub struct SigWindow {
    bytes: SigBytes,
    head: SigHead,
    /// [`Self::fingerprint`], taken once: a scan reads it per window, per scan.
    fingerprint: u64,
}

/// Set once a refused fold has been reported, so a scan that meets the same
/// foreign records on every pass says so once rather than on every pass.
static FOLD_MISMATCH_REPORTED: AtomicBool = AtomicBool::new(false);

/// Set once an unreadable stored window has been reported.
static UNREADABLE_REPORTED: AtomicBool = AtomicBool::new(false);

impl SigWindow {
    /// The window `blob` holds, its fingerprint the buffer's address — a window
    /// nothing else stamps. `None` for one a scan must not read
    /// ([`Self::from_head`]).
    pub fn new(blob: Arc<[u8]>) -> Option<Self> {
        let head = SigHead::parse(&blob)?;
        let stamp = blob.as_ptr() as usize as u64;
        Self::held(head, blob, stamp)
    }

    /// The window held in `blob`, whose header is `head`, stamped `stamp`.
    pub fn held(head: SigHead, blob: Arc<[u8]>, stamp: u64) -> Option<Self> {
        let len = blob.len();
        Self::from_head(head, len, SigBytes::Held(blob), stamp)
    }

    /// The window in the log at `stored.loc`, whose header is `head`, stamped
    /// `stamp`.
    pub fn stored(head: SigHead, stored: StoredSig, stamp: u64) -> Option<Self> {
        let len = stored.loc.payload_len as usize;
        Self::from_head(head, len, SigBytes::Stored(stored), stamp)
    }

    /// `None` for a window a scan must not read: an empty one, one shorter
    /// than its header states, or one folded under a different fold than this
    /// process's — the records
    /// [`decode_wide_sigs_for_scoring`](super::decode_wide_sigs_for_scoring)
    /// refuses.
    fn from_head(head: SigHead, len: usize, bytes: SigBytes, stamp: u64) -> Option<Self> {
        if let Some(expected) = active_fold() {
            if !fold_admits(head.params, expected) {
                if !FOLD_MISMATCH_REPORTED.swap(true, Ordering::Relaxed) {
                    warn_fold_mismatch(head.params, expected);
                }
                return None;
            }
        }
        if head.n_tokens == 0 || head.width == 0 || len < head.needs() {
            return None;
        }
        let mut hasher = DefaultHasher::new();
        stamp.hash(&mut hasher);
        head.n_tokens.hash(&mut hasher);
        Some(Self {
            bytes,
            head,
            fingerprint: hasher.finish(),
        })
    }

    /// Tokens in the window.
    pub fn len(&self) -> usize {
        self.head.n_tokens
    }

    /// Always `false`: an empty window is refused.
    pub fn is_empty(&self) -> bool {
        self.head.n_tokens == 0
    }

    /// `u64` words per token — the same for every token of a window.
    pub fn width(&self) -> usize {
        self.head.width
    }

    /// The window's residency fingerprint, from the stamp the substrate gave
    /// its value when it was written and its length. A value is stamped anew
    /// exactly when it is rewritten — moving its record, or settling it from
    /// held to stored, leaves the stamp — so a scan against unchanged turns
    /// sees the same fingerprint and uploads nothing.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The window's tokens, decoded now. Nothing keeps the result: a caller
    /// holds it only as long as its upload or host scan needs it. Empty when a
    /// stored window cannot be read back.
    pub fn decode(&self) -> Vec<WideQSig> {
        self.with_bytes(|b| decode_wide_sigs(b).unwrap_or_default())
            .unwrap_or_default()
    }

    /// The window's words, token-major and [`Self::width`] to a token, read
    /// now into one buffer — what a pass over every token's bits reads, without
    /// a [`WideQSig`] per token. Empty when a stored window cannot be read back.
    pub fn words(&self) -> Vec<u64> {
        let (from, to) = (self.head.header_len, self.head.needs());
        self.with_bytes(|b| {
            b[from..to]
                .chunks_exact(8)
                .map(|w| u64::from_le_bytes(w.try_into().expect("an 8-byte chunk")))
                .collect()
        })
        .unwrap_or_default()
    }

    /// `f` over the window's bytes: the held buffer, or the record read back.
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        match &self.bytes {
            SigBytes::Held(blob) => Some(f(blob)),
            SigBytes::Stored(s) => match s.source.read(s.id, s.loc) {
                Ok(bytes) if bytes.len() >= self.head.needs() => Some(f(&bytes)),
                Ok(bytes) => {
                    report_unreadable(
                        s,
                        &format!("{} of {} bytes", bytes.len(), self.head.needs()),
                    );
                    None
                }
                Err(e) => {
                    report_unreadable(s, &e.to_string());
                    None
                }
            },
        }
    }
}

/// Say once that a stored window could not be read back: the scan goes on
/// without it, and every later failure is the same fault.
fn report_unreadable(s: &StoredSig, why: &str) {
    if !UNREADABLE_REPORTED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            target: "candle_conversation::provenance",
            origin = format!("{:x}", s.id.origin),
            oseq = s.id.oseq,
            segment = s.loc.segment.0,
            offset = s.loc.offset,
            "a stored signature window could not be read back ({why}); scans skip it"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::segment::SegmentId;
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
        let words = w.words();
        assert_eq!(words.len(), 72);
        assert_eq!(words[0], 0xAAAA_AAAA_AAAA_AAAA);
        assert_eq!(words[24], 0x5555_5555_5555_5555);
        assert_eq!(words[71], 7);
    }

    /// The header alone sizes a window: its first bytes parse to what the
    /// whole record states.
    #[test]
    fn the_head_parses_from_the_first_bytes_alone() {
        let b = blob(&[sig(1), sig(2), sig(3)]);
        let head = SigHead::parse(&b[..20]).expect("the header is in the first 20 bytes");
        assert_eq!(head, SigHead::parse(&b).unwrap());
        assert_eq!(head.n_tokens, 3);
        assert_eq!(head.width, 24);
        assert_eq!(head.needs(), b.len());
    }

    /// The fingerprint follows the stamp: one stamp, one fingerprint, whatever
    /// buffer holds the bytes; a new stamp is a new fingerprint.
    #[test]
    fn the_fingerprint_follows_the_stamp() {
        let sigs = vec![sig(1), sig(2)];
        let head = SigHead::parse(&blob(&sigs)).unwrap();
        let a = SigWindow::held(head, blob(&sigs), 7).unwrap();
        let b = SigWindow::held(head, blob(&sigs), 7).unwrap();
        assert_eq!(
            a.fingerprint(),
            b.fingerprint(),
            "one stamp, one fingerprint"
        );
        let rewritten = SigWindow::held(head, blob(&sigs), 8).unwrap();
        assert_ne!(
            a.fingerprint(),
            rewritten.fingerprint(),
            "a rewritten value is stamped anew and must re-upload"
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

    /// A stored window nothing can read back decodes to nothing rather than
    /// failing the scan.
    #[test]
    fn a_stored_window_with_no_reader_decodes_to_nothing() {
        let b = blob(&[sig(5)]);
        let head = SigHead::parse(&b).unwrap();
        let stored = StoredSig {
            source: Arc::new(PayloadSource::default()),
            id: RecordIdentity::NONE,
            loc: RecordLoc {
                segment: SegmentId(1),
                offset: 4096,
                payload_len: b.len() as u64,
                record_size: 8192,
            },
        };
        let w = SigWindow::stored(head, stored, 1).expect("sized from its header");
        assert_eq!(w.len(), 1);
        assert!(w.decode().is_empty());
        assert!(w.words().is_empty());
    }
}
