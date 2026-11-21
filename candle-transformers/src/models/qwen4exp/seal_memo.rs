//! The records a tail seal made, kept by their piece key so the next seal of
//! the same pages hands them back instead of reading the pages off the card.
//!
//! A reprojection seals the decoding turn's whole span —
//! `Qwen4ExpBatched::seal_positional_tail_span` — and pushes the records back
//! onto the rebuilt slot, where each is placed as a page under its
//! [`PieceKey`]. The next reprojection seals the same span again: its closed
//! pages are ones earlier seals made, and their rows are immutable, so
//! sealing one reads back exactly the record it was placed from. Each read is
//! a synchronous device copy per chunk per layer, so a turn's seal grew with
//! the turn and stalled the stream at every reprojection of a long decode —
//! 8.6 ms of each rebuild on BlackWorld (2026-11-21), O(turn) per
//! reprojection. A page placed from a record carries the record's key
//! (`ResidentPage::source`); a seal finding every layer's page placed from
//! one key it holds here takes the record as it was. Only closed pages'
//! records are kept, so a page is read back twice in its life — as the live
//! tail, and at the first seal after it is placed — and never again.
//!
//! Bounded by bytes, least recently used first out: what it holds is the
//! spans of the turns decoding now.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::models::piece_key::PieceKey;

/// Bytes of records kept at most.
pub const SEAL_MEMO_BYTES: usize = 64 << 20;

/// See the module documentation.
#[derive(Debug)]
pub struct SealMemo {
    inner: Mutex<Memo>,
    cap: usize,
}

#[derive(Debug, Default)]
struct Memo {
    records: HashMap<PieceKey, Arc<[u8]>>,
    /// Keys oldest use first; a key used again moves to the back.
    order: VecDeque<PieceKey>,
    bytes: usize,
}

impl Default for SealMemo {
    fn default() -> Self {
        Self::with_cap(SEAL_MEMO_BYTES)
    }
}

impl SealMemo {
    /// A memo holding at most `cap` bytes of records.
    pub fn with_cap(cap: usize) -> Self {
        Self {
            inner: Mutex::new(Memo::default()),
            cap,
        }
    }

    /// The record kept under `key`, marked used.
    pub fn get(&self, key: &PieceKey) -> Option<Arc<[u8]>> {
        let mut m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let record = m.records.get(key).cloned()?;
        m.touch(key);
        Some(record)
    }

    /// Keep `record` under its key; the least recently used go past the cap.
    /// A record larger than the cap is not kept.
    pub fn put(&self, record: &[u8]) {
        if record.len() > self.cap {
            return;
        }
        let key = PieceKey::of(record);
        let mut m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if m.records.contains_key(&key) {
            m.touch(&key);
            return;
        }
        while m.bytes + record.len() > self.cap {
            let Some(old) = m.order.pop_front() else {
                break;
            };
            if let Some(r) = m.records.remove(&old) {
                m.bytes -= r.len();
            }
        }
        m.bytes += record.len();
        m.records.insert(key, Arc::from(record));
        m.order.push_back(key);
    }

    /// Bytes held — for a test and the memory report.
    pub fn bytes(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }
}

impl Memo {
    fn touch(&mut self, key: &PieceKey) {
        if let Some(at) = self.order.iter().position(|k| k == key) {
            self.order.remove(at);
        }
        self.order.push_back(*key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A kept record comes back under its own key, and the least recently
    /// used goes first past the cap**; one larger than the cap is not kept.
    #[test]
    fn records_come_back_by_key_and_the_oldest_goes_first() {
        let memo = SealMemo::with_cap(10);
        memo.put(b"aaaa");
        memo.put(b"bbbb");
        assert_eq!(
            memo.get(&PieceKey::of(b"aaaa")).as_deref(),
            Some(&b"aaaa"[..])
        );
        // `bbbb` is now the least recently used.
        memo.put(b"cccc");
        assert_eq!(memo.get(&PieceKey::of(b"bbbb")), None);
        assert_eq!(
            memo.get(&PieceKey::of(b"aaaa")).as_deref(),
            Some(&b"aaaa"[..])
        );
        assert_eq!(
            memo.get(&PieceKey::of(b"cccc")).as_deref(),
            Some(&b"cccc"[..])
        );
        assert_eq!(memo.bytes(), 8);
        memo.put(b"a record past the cap");
        assert_eq!(memo.get(&PieceKey::of(b"a record past the cap")), None);
        memo.put(b"aaaa");
        assert_eq!(memo.bytes(), 8, "a record kept twice is kept once");
    }
}
