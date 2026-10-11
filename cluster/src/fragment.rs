//! `EVENT` fragmentation (§4.2, §4.3): a record above 1 MiB travels as
//! consecutive fragments, so no single transfer holds the socket. The receiver
//! reassembles by `(origin, oseq)` from fragments relayed as they arrive. A
//! socket delivers in order, so one record's fragments arrive in offset order;
//! a gap, an overlap or a changed total is refused. Partial records of a socket
//! that drops are discarded — the sender sends the whole record again (§6.3).

use std::collections::HashMap;

use thiserror::Error;

use crate::frame::{Event, MAX_FRAGMENT};

/// The fragments of one record, in order.
pub fn split(origin: u64, oseq: u64, record: &[u8], reply_to: Option<u64>) -> Vec<Event> {
    assert!(!record.is_empty(), "a record is never empty");
    let total = u32::try_from(record.len()).expect("a record is under 4 GiB");
    record
        .chunks(MAX_FRAGMENT)
        .enumerate()
        .map(|(i, bytes)| Event {
            origin,
            oseq,
            total,
            offset: (i * MAX_FRAGMENT) as u32,
            reply_to,
            bytes: bytes.to_vec(),
        })
        .collect()
}

/// How many fragments a record of `len` bytes takes.
pub fn fragments(len: usize) -> usize {
    len.div_ceil(MAX_FRAGMENT)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "fragment of ({origin}, {oseq}) at {offset} does not continue the {have} bytes held of {total}"
)]
pub struct OutOfOrder {
    pub origin: u64,
    pub oseq: u64,
    pub offset: u32,
    pub have: usize,
    pub total: u32,
}

/// A whole record, reassembled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Whole {
    pub origin: u64,
    pub oseq: u64,
    pub reply_to: Option<u64>,
    pub bytes: Vec<u8>,
}

/// One socket's partial records.
#[derive(Debug, Default)]
pub struct Reassembly {
    partial: HashMap<(u64, u64), (u32, Vec<u8>)>,
}

impl Reassembly {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a fragment; a record's last fragment returns it whole.
    pub fn accept(&mut self, event: Event) -> Result<Option<Whole>, OutOfOrder> {
        let key = (event.origin, event.oseq);
        let (total, held) = self
            .partial
            .entry(key)
            .or_insert_with(|| (event.total, Vec::new()));
        if event.total != *total || event.offset as usize != held.len() {
            let err = OutOfOrder {
                origin: event.origin,
                oseq: event.oseq,
                offset: event.offset,
                have: held.len(),
                total: *total,
            };
            self.partial.remove(&key);
            return Err(err);
        }
        held.extend_from_slice(&event.bytes);
        if held.len() < *total as usize {
            return Ok(None);
        }
        let (_, bytes) = self.partial.remove(&key).expect("just held");
        Ok(Some(Whole {
            origin: event.origin,
            oseq: event.oseq,
            reply_to: event.reply_to,
            bytes,
        }))
    }

    /// The socket dropped: every partial record is discarded.
    pub fn clear(&mut self) {
        self.partial.clear();
    }

    pub fn pending(&self) -> usize {
        self.partial.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 % 251) as u8).collect()
    }

    #[test]
    fn a_small_record_is_one_fragment() {
        let f = split(1, 2, &[9, 8], Some(5));
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].total, f[0].offset, f[0].reply_to), (2, 0, Some(5)));
        assert!(f[0].is_last());
    }

    /// A 63 MB record is 63 fragments of at most 1 MiB, and reassembles to the
    /// same bytes even with another record's fragments between them.
    #[test]
    fn a_large_record_splits_and_reassembles_interleaved() {
        let big = record(63_000_000);
        let small = record(3 << 20);
        let a = split(7, 1, &big, None);
        let b = split(8, 4, &small, None);
        assert_eq!(a.len(), fragments(big.len()));
        assert_eq!(a.len(), 61);
        assert!(a.iter().all(|e| e.bytes.len() <= MAX_FRAGMENT));
        let mut r = Reassembly::new();
        let mut done = Vec::new();
        let mut bi = b.into_iter();
        for frag in a {
            if let Some(w) = r.accept(frag).unwrap() {
                done.push(w);
            }
            if let Some(f) = bi.next() {
                if let Some(w) = r.accept(f).unwrap() {
                    done.push(w);
                }
            }
        }
        assert_eq!(done.len(), 2);
        assert_eq!((done[0].origin, done[0].bytes.len()), (8, small.len()));
        assert_eq!(done[0].bytes, small);
        assert_eq!(done[1].bytes, big);
        assert_eq!(r.pending(), 0);
    }

    #[test]
    fn a_gap_or_a_changed_total_is_refused() {
        let frags = split(1, 1, &record(3 << 20), None);
        let mut r = Reassembly::new();
        r.accept(frags[0].clone()).unwrap();
        assert!(r.accept(frags[2].clone()).is_err());
        assert_eq!(r.pending(), 0);
        r.accept(frags[0].clone()).unwrap();
        let mut changed = frags[1].clone();
        changed.total += 1;
        assert!(r.accept(changed).is_err());
    }

    #[test]
    fn a_dropped_socket_discards_its_partials() {
        let frags = split(1, 1, &record(2 << 20), None);
        let mut r = Reassembly::new();
        r.accept(frags[0].clone()).unwrap();
        r.clear();
        assert_eq!(r.pending(), 0);
        // The whole record sent again reassembles.
        for f in &frags[..frags.len() - 1] {
            assert_eq!(r.accept(f.clone()).unwrap(), None);
        }
        assert!(r.accept(frags.last().unwrap().clone()).unwrap().is_some());
    }
}
