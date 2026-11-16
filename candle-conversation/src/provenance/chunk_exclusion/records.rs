//! The keyed records an incremental exclusion keeps in the substrate
//! (`docs/incremental_chunk_exclusion.md` §4.2): one per sample question,
//! holding its entries, and one per group, holding the cut and the cursor.
//! Every member installs the group record's cut as it applies it.
//!
//! Both are little-endian byte layouts, carried base64-encoded in the custom
//! object's JSON as `{"b": "<base64>"}`.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};

use super::stats::{EntryStats, FileId};
use crate::persistence::streams::StreamId;

/// Every key the exclusion keeps begins with this.
pub const PREFIX: &str = "chunk_exclusion/";

/// Layout version both records open with.
const VERSION: u8 = 1;

/// The group record's key.
pub fn group_key(group: u32) -> String {
    format!("{PREFIX}{group}")
}

/// The prefix every question record of `group` sits under.
pub fn questions_prefix(group: u32) -> String {
    format!("{PREFIX}{group}/q/")
}

/// Sample question `sid`'s record key.
pub fn question_key(group: u32, sid: StreamId) -> String {
    format!("{}{}", questions_prefix(group), sid.0)
}

/// The group a group record's key names; `None` for any other key.
pub fn group_of_key(key: &str) -> Option<u32> {
    key.strip_prefix(PREFIX)?.parse().ok()
}

/// The question a question record's key names.
pub fn question_of_key(key: &str) -> Option<StreamId> {
    let rest = key.strip_prefix(PREFIX)?;
    let (_, sid) = rest.split_once("/q/")?;
    sid.parse().ok().map(StreamId)
}

/// One entry as a question record keeps it: its statistics, the gallery token
/// that won it when one has been found, and whether its leaders wait for a
/// rescan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub stats: EntryStats,
    /// `(stream, position)` of the leading file's best-agreeing token.
    pub win: Option<(StreamId, u32)>,
    /// A leader was removed; the entry is left out of the cut until rescanned.
    pub dirty: bool,
}

/// Bytes an entry takes: a1, a2 (u16), flags (u8), c1, c2, sum, sumsq, win
/// stream (u64), win position (u32).
const ENTRY_BYTES: usize = 2 + 2 + 1 + 8 * 5 + 4;

/// A sample question's record: its owner file, the gallery count each
/// competition's sums were taken at, and its entries per competition, token
/// major, fold group minor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuestionRecord {
    pub owner: FileId,
    pub n: [u64; 2],
    pub entries: [Vec<Entry>; 2],
}

impl QuestionRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(
            1 + 8 * 3 + 8 + ENTRY_BYTES * (self.entries[0].len() + self.entries[1].len()),
        );
        b.push(VERSION);
        b.extend_from_slice(&self.owner.to_le_bytes());
        b.extend_from_slice(&self.n[0].to_le_bytes());
        b.extend_from_slice(&self.n[1].to_le_bytes());
        for comp in &self.entries {
            b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            for e in comp {
                let s = &e.stats;
                b.extend_from_slice(&(s.a1 as u16).to_le_bytes());
                b.extend_from_slice(&(s.a2 as u16).to_le_bytes());
                b.push(u8::from(e.dirty) | (u8::from(e.win.is_some()) << 1));
                b.extend_from_slice(&s.c1.to_le_bytes());
                b.extend_from_slice(&s.c2.to_le_bytes());
                b.extend_from_slice(&s.sum.to_le_bytes());
                b.extend_from_slice(&s.sumsq.to_le_bytes());
                let (ws, wp) = e.win.map_or((0, 0), |(s, p)| (s.0, p));
                b.extend_from_slice(&ws.to_le_bytes());
                b.extend_from_slice(&wp.to_le_bytes());
            }
        }
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        if r.u8()? != VERSION {
            return None;
        }
        let owner = r.u64()?;
        let n = [r.u64()?, r.u64()?];
        let mut entries: [Vec<Entry>; 2] = [Vec::new(), Vec::new()];
        for comp in &mut entries {
            let count = r.u32()? as usize;
            comp.reserve(count);
            for _ in 0..count {
                let a1 = u32::from(r.u16()?);
                let a2 = u32::from(r.u16()?);
                let flags = r.u8()?;
                let c1 = r.u64()?;
                let c2 = r.u64()?;
                let sum = r.u64()?;
                let sumsq = r.u64()?;
                let ws = r.u64()?;
                let wp = r.u32()?;
                comp.push(Entry {
                    stats: EntryStats {
                        a1,
                        c1,
                        a2,
                        c2,
                        sum,
                        sumsq,
                    },
                    win: (flags & 2 != 0).then_some((StreamId(ws), wp)),
                    dirty: flags & 1 != 0,
                });
            }
        }
        r.0.is_empty()
            .then_some(QuestionRecord { owner, n, entries })
    }
}

/// The group record: the cursor, each competition's gallery count, and the cut
/// chunks of both competitions as `(stream, start, end)` token ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CutRecord {
    pub cursor: u64,
    pub n: [u64; 2],
    pub cut: Vec<(StreamId, u32, u32)>,
}

impl CutRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(1 + 8 * 3 + 4 + 16 * self.cut.len());
        b.push(VERSION);
        b.extend_from_slice(&self.cursor.to_le_bytes());
        b.extend_from_slice(&self.n[0].to_le_bytes());
        b.extend_from_slice(&self.n[1].to_le_bytes());
        b.extend_from_slice(&(self.cut.len() as u32).to_le_bytes());
        for &(sid, start, end) in &self.cut {
            b.extend_from_slice(&sid.0.to_le_bytes());
            b.extend_from_slice(&start.to_le_bytes());
            b.extend_from_slice(&end.to_le_bytes());
        }
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        if r.u8()? != VERSION {
            return None;
        }
        let cursor = r.u64()?;
        let n = [r.u64()?, r.u64()?];
        let count = r.u32()? as usize;
        let mut cut = Vec::with_capacity(count);
        for _ in 0..count {
            cut.push((StreamId(r.u64()?), r.u32()?, r.u32()?));
        }
        r.0.is_empty().then_some(CutRecord { cursor, n, cut })
    }
}

/// `bytes` as a custom object's JSON.
pub fn to_json(bytes: &[u8]) -> Value {
    json!({ "b": STANDARD.encode(bytes) })
}

/// The bytes a custom object's JSON carries.
pub fn from_json(value: &Value) -> Option<Vec<u8>> {
    STANDARD.decode(value.get("b")?.as_str()?).ok()
}

/// A little-endian cursor over a record's bytes.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take().map(u16::from_le_bytes)
    }
    fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }
    fn u64(&mut self) -> Option<u64> {
        self.take().map(u64::from_le_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Keys name their group and question**, and nothing else parses as one.
    #[test]
    fn keys_name_their_group_and_question() {
        assert_eq!(group_key(7), "chunk_exclusion/7");
        assert_eq!(question_key(7, StreamId(42)), "chunk_exclusion/7/q/42");
        assert_eq!(group_of_key("chunk_exclusion/7"), Some(7));
        assert_eq!(group_of_key("chunk_exclusion/7/q/42"), None);
        assert_eq!(group_of_key("code_read/binary/x"), None);
        assert_eq!(
            question_of_key("chunk_exclusion/7/q/42"),
            Some(StreamId(42))
        );
        assert_eq!(question_of_key("chunk_exclusion/7"), None);
    }

    /// **A question record is these bytes**, and reads back as itself.
    #[test]
    fn a_question_record_is_these_bytes() {
        let rec = QuestionRecord {
            owner: 0x0102,
            n: [3, 4],
            entries: [
                vec![Entry {
                    stats: EntryStats {
                        a1: 0x0105,
                        c1: 9,
                        a2: 0x0100,
                        c2: 8,
                        sum: 0x10,
                        sumsq: 0x20,
                    },
                    win: Some((StreamId(0x33), 0x44)),
                    dirty: true,
                }],
                Vec::new(),
            ],
        };
        let bytes = rec.encode();
        let mut want = vec![1u8];
        want.extend_from_slice(&[0x02, 0x01, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[4, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[0x05, 0x01, 0x00, 0x01, 0b11]);
        want.extend_from_slice(&[9, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[8, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x10, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x20, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x33, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x44, 0, 0, 0]);
        want.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(bytes, want);
        assert_eq!(QuestionRecord::decode(&bytes), Some(rec));
        assert_eq!(QuestionRecord::decode(&bytes[..bytes.len() - 1]), None);
    }

    /// **A group record is these bytes**, and reads back as itself through its
    /// JSON.
    #[test]
    fn a_group_record_is_these_bytes() {
        let rec = CutRecord {
            cursor: 0x0A0B,
            n: [5, 6],
            cut: vec![(StreamId(2), 32, 64)],
        };
        let bytes = rec.encode();
        let mut want = vec![1u8];
        want.extend_from_slice(&[0x0B, 0x0A, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[5, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[6, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[32, 0, 0, 0, 64, 0, 0, 0]);
        assert_eq!(bytes, want);
        let json = to_json(&bytes);
        assert_eq!(CutRecord::decode(&from_json(&json).unwrap()), Some(rec));
        assert_eq!(CutRecord::decode(&[2]), None);
    }
}
