//! The records a worker's handing carries (`crate::worker_hop`,
//! `docs/claude_zend_gateway.md` §6.3).
//!
//! A member taking a worker must hold its conversation through the handing
//! member's writes: the round's call turn, its commit, the round's resume
//! entry and the worker's record. Replication carries them, but a member far
//! behind on the handing member's lane takes longer than a handing waits. So
//! it answers with what it lacks — the ranges of the handing member's own
//! `oseq`s, through the point the handing names — and the handing member,
//! when what it lacks is small enough, sends those records with the handing
//! again.
//!
//! **What is sent** is chosen by size ([`choose`]): every record the taker
//! lacks, when they fit in [`PUSH_BUDGET`]; else the conversation's own — its
//! turns' streams and the records no stream holds, its resume entry and the
//! worker's record among them — when those fit; else nothing, and the worker
//! stays.
//!
//! **On the wire** the records follow the handing, under one signature: the
//! handing's JSON, then — when it carries records — a newline and each record
//! as its `oseq` and length, little-endian `u64` and `u32`, and its bytes as
//! they travel. Compact JSON never holds a raw newline, so the first one ends
//! the handing.

use std::collections::BTreeSet;

use candle_conversation::persistence::listing::ListedRecord;
use candle_conversation::persistence::replica::OriginApplied;
use candle_conversation::projection::{Conversation, TimelineId};

/// The most a handing carries of the records its taker lacks.
pub const PUSH_BUDGET: u64 = 30 << 20;

/// The most ranges a taker names: the newest, which a handing's own writes
/// are among.
pub const MOST_RANGES: usize = 256;

/// The most `oseq`s a handing member lists to choose from: past this the
/// taker is so far behind that what it lacks is no handing's to carry.
pub const MOST_LISTED: u64 = 1 << 16;

/// What separates the handing from the records after it.
const SEPARATOR: u8 = b'\n';

/// The ranges of `1..=written` that `held` — what a member holds of the
/// handing member's `oseq`s — does not cover, as inclusive `(start, end)`:
/// the newest [`MOST_RANGES`] of them.
pub fn missing(held: Option<&OriginApplied>, written: u64) -> Vec<(u64, u64)> {
    let gaps = match held {
        Some(held) => held.uncovered(1, written),
        None if written > 0 => vec![(1, written)],
        None => Vec::new(),
    };
    let skip = gaps.len().saturating_sub(MOST_RANGES);
    gaps.into_iter().skip(skip).collect()
}

/// How many `oseq`s `ranges` name.
pub fn count(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|(s, e)| e.saturating_sub(*s) + 1).sum()
}

/// Why a handing carries nothing: what the taker lacks, all of it and the
/// conversation's own, is over the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooBig {
    pub all: u64,
    pub ours: u64,
}

/// Whether a record of `stream_id` is the conversation's: one of its turns'
/// streams, or no stream at all — its resume entry and the worker's record are
/// among those.
fn ours(stream_id: u64, streams: &BTreeSet<u64>) -> bool {
    stream_id == 0 || streams.contains(&stream_id)
}

/// The `oseq`s to carry of `listed`, the records the taker lacks, given the
/// conversation's turn `streams`: see the module documentation.
pub fn choose(
    listed: &[ListedRecord],
    streams: &BTreeSet<u64>,
    budget: u64,
) -> Result<Vec<u64>, TooBig> {
    let size = |r: &&ListedRecord| u64::from(r.len);
    let all: u64 = listed.iter().map(|r| size(&r)).sum();
    if all <= budget {
        return Ok(listed.iter().map(|r| r.oseq).collect());
    }
    let mine: Vec<&ListedRecord> = listed
        .iter()
        .filter(|r| ours(r.stream_id, streams))
        .collect();
    let ours_bytes: u64 = mine.iter().map(size).sum();
    if ours_bytes <= budget {
        return Ok(mine.iter().map(|r| r.oseq).collect());
    }
    Err(TooBig {
        all,
        ours: ours_bytes,
    })
}

/// The conversation's turns `listed` holds a record of, ascending, given each
/// turn as `(index, stream id)`.
pub fn turns_of(listed: &[ListedRecord], turns: &[(u32, u64)]) -> Vec<u32> {
    let streams: BTreeSet<u64> = listed.iter().map(|r| r.stream_id).collect();
    let found: BTreeSet<u32> = turns
        .iter()
        .filter(|(_, sid)| streams.contains(sid))
        .map(|(idx, _)| *idx)
        .collect();
    found.into_iter().collect()
}

/// What a handing carries: the records, the conversation's turns they hold,
/// and their size as they travel.
#[derive(Debug, Default)]
pub struct Carrying {
    pub records: Vec<(u64, Vec<u8>)>,
    pub turns: Vec<u32>,
    pub bytes: u64,
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / f64::from(1 << 20))
}

/// The records of `origin` — this member — in `missing`, the ranges a member
/// taking `timeline`'s worker lacks, that a handing carries to it: chosen by
/// [`choose`], read as they travel. `Err` says why it carries none.
pub fn gather(
    conversation: &Conversation,
    timeline: TimelineId,
    origin: u64,
    missing: &[(u64, u64)],
) -> Result<Carrying, String> {
    let lacked = count(missing);
    if lacked > MOST_LISTED {
        return Err(format!(
            "it lacks {lacked} of this member's records, past the {MOST_LISTED} a handing looks through"
        ));
    }
    let listed: Vec<ListedRecord> = missing
        .iter()
        .flat_map(|&(from, to)| conversation.list_held(origin, from, to).records)
        .collect();
    let turns = conversation.turn_streams_by(timeline, origin);
    let streams: BTreeSet<u64> = turns.iter().map(|(_, sid)| *sid).collect();
    let oseqs = choose(&listed, &streams, PUSH_BUDGET).map_err(|too| {
        format!(
            "it lacks {} of this member's records, {} of them this conversation's — over the {} a handing carries",
            mb(too.all),
            mb(too.ours),
            mb(PUSH_BUDGET)
        )
    })?;
    let chosen: BTreeSet<u64> = oseqs.iter().copied().collect();
    let carried: Vec<ListedRecord> = listed
        .into_iter()
        .filter(|r| chosen.contains(&r.oseq))
        .collect();
    let records: Vec<(u64, Vec<u8>)> = oseqs
        .iter()
        .zip(conversation.read_events(origin, &oseqs))
        .filter_map(|(&oseq, bytes)| Some((oseq, bytes?)))
        .collect();
    Ok(Carrying {
        bytes: records.iter().map(|(_, b)| b.len() as u64).sum(),
        turns: turns_of(&carried, &turns),
        records,
    })
}

impl Carrying {
    /// What it carries, as a log line reads it.
    pub fn told(&self) -> String {
        format!(
            "{} records, {}, turns {:?}",
            self.records.len(),
            mb(self.bytes),
            self.turns
        )
    }
}

/// A handing's body: its JSON `handing`, then `records`.
pub fn encode(handing: &[u8], records: &[(u64, Vec<u8>)]) -> Vec<u8> {
    debug_assert!(!handing.contains(&SEPARATOR));
    if records.is_empty() {
        return handing.to_vec();
    }
    let size: usize = records.iter().map(|(_, b)| 12 + b.len()).sum();
    let mut body = Vec::with_capacity(handing.len() + 1 + size);
    body.extend_from_slice(handing);
    body.push(SEPARATOR);
    for (oseq, bytes) in records {
        body.extend_from_slice(&oseq.to_le_bytes());
        body.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(bytes);
    }
    body
}

/// A handing body's JSON and the records after it.
pub fn decode(body: &[u8]) -> Result<(&[u8], Vec<(u64, &[u8])>), String> {
    let Some(at) = body.iter().position(|b| *b == SEPARATOR) else {
        return Ok((body, Vec::new()));
    };
    let (handing, mut rest) = (&body[..at], &body[at + 1..]);
    let mut records = Vec::new();
    while !rest.is_empty() {
        if rest.len() < 12 {
            return Err(format!(
                "a carried record's head is {} bytes of 12",
                rest.len()
            ));
        }
        let oseq = u64::from_le_bytes(rest[..8].try_into().expect("eight bytes"));
        let len = u32::from_le_bytes(rest[8..12].try_into().expect("four bytes")) as usize;
        rest = &rest[12..];
        if rest.len() < len {
            return Err(format!(
                "carried record {oseq} is {len} bytes, and {} are left",
                rest.len()
            ));
        }
        records.push((oseq, &rest[..len]));
        rest = &rest[len..];
    }
    Ok((handing, records))
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::SubstratePersistence;
    use candle_conversation::substrate::Substrate;
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    /// **What is carried is each lacked record as it travels**, read off this
    /// member's log; a taker too far behind is carried nothing.
    #[test]
    fn what_a_taker_lacks_is_read_as_it_travels() {
        let dir = TempDir::new().unwrap();
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let conv = Conversation::from_parts(substrate, sp);
        for k in 0..3 {
            conv.put_cluster_object(&format!("cluster/k{k}"), json!(k))
                .unwrap();
        }
        conv.wait_written(conv.write_ticket());
        let (node, through) = (conv.node(), conv.own_through());
        assert!(through >= 3, "{through}");
        let timeline = TimelineId::from_raw(1).unwrap();
        let carrying = gather(&conv, timeline, node, &[(1, through)]).unwrap();
        assert!(carrying.records.len() >= 3, "{carrying:?}");
        for (oseq, bytes) in &carrying.records {
            assert_eq!(Some(bytes), conv.read_event(node, *oseq).as_ref());
        }
        assert_eq!(
            carrying.bytes,
            carrying
                .records
                .iter()
                .map(|(_, b)| b.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(carrying.turns, Vec::<u32>::new(), "no turn of the timeline");
        assert!(gather(&conv, timeline, node, &[(1, MOST_LISTED + 1)])
            .unwrap_err()
            .contains("past the"));
    }

    fn listed(oseq: u64, stream_id: u64, len: u32) -> ListedRecord {
        ListedRecord {
            oseq,
            record_type: 0,
            stream_id,
            chunk_index: 0,
            len,
            digest: [0; 16],
        }
    }

    /// **The taker names what it lacks of the handing's writes**: every
    /// uncovered range through the point the handing names, the newest when
    /// there are many, and nothing past that point.
    #[test]
    fn the_taker_names_the_ranges_it_lacks() {
        let held = OriginApplied::from_parts(10, [(15, 20), (25, 25)]);
        assert_eq!(missing(Some(&held), 30), vec![(11, 14), (21, 24), (26, 30)]);
        assert_eq!(missing(Some(&held), 18), vec![(11, 14)]);
        assert_eq!(missing(Some(&held), 10), vec![]);
        assert_eq!(missing(None, 4), vec![(1, 4)]);
        assert_eq!(missing(None, 0), vec![]);
        assert_eq!(count(&missing(Some(&held), 30)), 4 + 4 + 5);

        let holey = OriginApplied::from_parts(0, (1..=300).map(|i| (i * 2, i * 2)));
        let named = missing(Some(&holey), 601);
        assert_eq!(named.len(), MOST_RANGES);
        assert_eq!(named.last(), Some(&(601, 601)), "the newest kept");
        assert_eq!(named[0], (91, 91));
    }

    /// **All of it when it fits, the conversation's own when only that fits,
    /// and nothing otherwise.**
    #[test]
    fn what_is_carried_is_chosen_by_size() {
        let streams = BTreeSet::from([7]);
        let records = [listed(1, 7, 10), listed(2, 0, 5), listed(3, 9, 40)];
        assert_eq!(choose(&records, &streams, 55), Ok(vec![1, 2, 3]));
        assert_eq!(
            choose(&records, &streams, 54),
            Ok(vec![1, 2]),
            "another conversation's turn left behind"
        );
        assert_eq!(
            choose(&records, &streams, 14),
            Err(TooBig { all: 55, ours: 15 })
        );
        assert_eq!(choose(&[], &streams, 0), Ok(vec![]));
    }

    #[test]
    fn the_turns_lacked_are_named_by_their_streams() {
        let turns = [(3, 70), (4, 71), (5, 72)];
        let records = [
            listed(1, 72, 1),
            listed(2, 0, 1),
            listed(3, 70, 1),
            listed(4, 72, 1),
        ];
        assert_eq!(turns_of(&records, &turns), vec![3, 5]);
        assert_eq!(turns_of(&[], &turns), vec![]);
    }

    /// **A handing's body is its JSON, then its records, byte for byte**, and
    /// one with no records is the JSON alone.
    #[test]
    fn a_body_carries_its_records_after_the_handing() {
        let records = vec![(5u64, vec![0xaa, 0xbb]), (0x0102, vec![])];
        let body = encode(br#"{"a":1}"#, &records);
        assert_eq!(
            body,
            [
                &br#"{"a":1}"#[..],
                b"\n",
                &[5, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0xaa, 0xbb],
                &[2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
            .concat()
        );
        let (handing, carried) = decode(&body).unwrap();
        assert_eq!(handing, br#"{"a":1}"#);
        assert_eq!(carried, vec![(5, &[0xaa, 0xbb][..]), (0x0102, &[][..])]);

        assert_eq!(encode(br#"{"a":1}"#, &[]), br#"{"a":1}"#);
        assert_eq!(decode(br#"{"a":1}"#).unwrap(), (&br#"{"a":1}"#[..], vec![]));
    }

    #[test]
    fn a_body_cut_short_is_refused() {
        let body = encode(b"{}", &[(5, vec![1, 2, 3])]);
        assert!(decode(&body[..body.len() - 1])
            .unwrap_err()
            .contains("carried record 5 is 3 bytes, and 2 are left"));
        assert!(decode(&body[..8]).unwrap_err().contains("head is 5 bytes"));
    }
}
