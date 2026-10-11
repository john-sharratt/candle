//! `substrate/VOTES` (§5.9): a file preallocated at first open, of fixed-size
//! slots, one per Paxos key. Each slot is two halves; a vote is written in
//! place to the half not holding the current one, with a sequence number and
//! a checksum, then synced. A write torn by a crash leaves the previous vote
//! readable, never a half-written one. The file never grows.
//!
//! A half, little-endian:
//!
//! ```text
//!    0  u32 magic "VOTE"
//!    4  u64 seq
//!   12  u16 key_len              0 = the slot is free
//!   14  [128] key
//!  142  u64 promised.round · u64 promised.node
//!  158  u8 accepted              0 or 1
//!  159  u64 accepted.round · u64 accepted.node
//!  175  u16 value_len
//!  177  [800] value
//!  977  zero padding
//! 1020  u32 CRC-32 of bytes 0..1020
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::{Vote, VoteStore};
use crate::frame::{AcceptedValue, Ballot};

/// The journal's size when the caller names none: 64 MiB, 32,768 keys.
pub const DEFAULT_SIZE: u64 = 64 << 20;
pub const HALF: usize = 1024;
const SLOT: usize = 2 * HALF;
pub const MAX_KEY: usize = 128;
pub const MAX_VALUE: usize = 800;
const MAGIC: u32 = u32::from_le_bytes(*b"VOTE");
const CRC_AT: usize = HALF - 4;

/// A slot's state in memory: which half holds its current record, and that
/// record's sequence number.
#[derive(Clone, Copy, Default)]
struct SlotState {
    current: Option<usize>,
    seq: u64,
}

pub struct FileJournal {
    file: File,
    slots: Vec<SlotState>,
    index: HashMap<Vec<u8>, usize>,
    free: Vec<usize>,
}

fn encode(seq: u64, key: &[u8], vote: &Vote) -> [u8; HALF] {
    let mut h = [0u8; HALF];
    h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    h[4..12].copy_from_slice(&seq.to_le_bytes());
    h[12..14].copy_from_slice(&(key.len() as u16).to_le_bytes());
    h[14..14 + key.len()].copy_from_slice(key);
    h[142..150].copy_from_slice(&vote.promised.round.to_le_bytes());
    h[150..158].copy_from_slice(&vote.promised.node.to_le_bytes());
    if let Some(a) = &vote.accepted {
        h[158] = 1;
        h[159..167].copy_from_slice(&a.ballot.round.to_le_bytes());
        h[167..175].copy_from_slice(&a.ballot.node.to_le_bytes());
        h[175..177].copy_from_slice(&(a.value.len() as u16).to_le_bytes());
        h[177..177 + a.value.len()].copy_from_slice(&a.value);
    }
    let crc = crc32fast::hash(&h[..CRC_AT]);
    h[CRC_AT..].copy_from_slice(&crc.to_le_bytes());
    h
}

fn u64_at(h: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(h[at..at + 8].try_into().unwrap())
}

fn u16_at(h: &[u8], at: usize) -> usize {
    u16::from_le_bytes(h[at..at + 2].try_into().unwrap()) as usize
}

/// A valid half's sequence number, key and vote; `None` for one never written
/// or torn.
fn decode(h: &[u8]) -> Option<(u64, Vec<u8>, Vote)> {
    if u32::from_le_bytes(h[0..4].try_into().unwrap()) != MAGIC {
        return None;
    }
    let crc = u32::from_le_bytes(h[CRC_AT..].try_into().unwrap());
    if crc32fast::hash(&h[..CRC_AT]) != crc {
        return None;
    }
    let key_len = u16_at(h, 12);
    let value_len = u16_at(h, 175);
    if key_len > MAX_KEY || value_len > MAX_VALUE || h[158] > 1 {
        return None;
    }
    let accepted = (h[158] == 1).then(|| AcceptedValue {
        ballot: Ballot {
            round: u64_at(h, 159),
            node: u64_at(h, 167),
        },
        value: h[177..177 + value_len].to_vec(),
    });
    Some((
        u64_at(h, 4),
        h[14..14 + key_len].to_vec(),
        Vote {
            promised: Ballot {
                round: u64_at(h, 142),
                node: u64_at(h, 150),
            },
            accepted,
        },
    ))
}

fn full() -> io::Error {
    io::Error::other("the vote journal has no free slot")
}

impl FileJournal {
    /// Open the journal at `path`, creating and preallocating it to `size`
    /// bytes at first open.
    pub fn open(path: &Path, size: u64) -> io::Result<Self> {
        let created = !path.exists();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        if created {
            file.set_len(size)?;
            file.sync_all()?;
        }
        let len = file.metadata()?.len() as usize;
        let n = len / SLOT;
        let mut j = Self {
            file,
            slots: vec![SlotState::default(); n],
            index: HashMap::new(),
            free: Vec::new(),
        };
        let mut slot = vec![0u8; SLOT];
        j.file.seek(SeekFrom::Start(0))?;
        for s in 0..n {
            j.file.read_exact(&mut slot)?;
            let halves = [decode(&slot[..HALF]), decode(&slot[HALF..])];
            let best = halves
                .iter()
                .enumerate()
                .filter_map(|(i, h)| h.as_ref().map(|h| (i, h)))
                .max_by_key(|(_, h)| h.0);
            match best {
                Some((half, (seq, key, _))) => {
                    j.slots[s] = SlotState {
                        current: Some(half),
                        seq: *seq,
                    };
                    if key.is_empty() {
                        j.free.push(s);
                    } else {
                        j.index.insert(key.clone(), s);
                    }
                }
                None => j.free.push(s),
            }
        }
        // Allocate from the front.
        j.free.reverse();
        Ok(j)
    }

    /// How many keys the journal can hold at once.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn write_half(&mut self, slot: usize, key: &[u8], vote: &Vote) -> io::Result<()> {
        let state = self.slots[slot];
        let half = match state.current {
            Some(0) => 1,
            _ => 0,
        };
        let seq = state.seq + 1;
        let bytes = encode(seq, key, vote);
        self.file
            .seek(SeekFrom::Start((slot * SLOT + half * HALF) as u64))?;
        self.file.write_all(&bytes)?;
        self.file.sync_data()?;
        self.slots[slot] = SlotState {
            current: Some(half),
            seq,
        };
        Ok(())
    }

    /// The byte offset of `key`'s slot, for tests that tear a write.
    #[cfg(test)]
    pub(super) fn slot_of(&self, key: &[u8]) -> Option<(usize, Option<usize>)> {
        self.index
            .get(key)
            .map(|s| (*s * SLOT, self.slots[*s].current))
    }

    #[cfg(test)]
    pub(super) fn file(&mut self) -> &mut File {
        &mut self.file
    }
}

impl VoteStore for FileJournal {
    fn load(&mut self) -> io::Result<Vec<(Vec<u8>, Vote)>> {
        let mut out = Vec::with_capacity(self.index.len());
        let mut half = vec![0u8; HALF];
        for (key, slot) in &self.index {
            let current = self.slots[*slot]
                .current
                .expect("indexed slots hold a record");
            self.file
                .seek(SeekFrom::Start((slot * SLOT + current * HALF) as u64))?;
            self.file.read_exact(&mut half)?;
            let (_, _, vote) = decode(&half).expect("read back what open validated");
            out.push((key.clone(), vote));
        }
        Ok(out)
    }

    fn write(&mut self, key: &[u8], vote: &Vote) -> io::Result<()> {
        if key.is_empty() || key.len() > MAX_KEY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a Paxos key is 1 to {MAX_KEY} bytes, not {}", key.len()),
            ));
        }
        if vote
            .accepted
            .as_ref()
            .is_some_and(|a| a.value.len() > MAX_VALUE)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a Paxos value is at most {MAX_VALUE} bytes"),
            ));
        }
        let slot = match self.index.get(key) {
            Some(s) => *s,
            None => {
                let s = self.free.pop().ok_or_else(full)?;
                self.index.insert(key.to_vec(), s);
                s
            }
        };
        self.write_half(slot, key, vote)
    }

    fn free(&mut self, key: &[u8]) -> io::Result<()> {
        let Some(slot) = self.index.remove(key) else {
            return Ok(());
        };
        self.write_half(slot, &[], &Vote::default())?;
        self.free.push(slot);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn vote(round: u64, value: Option<&[u8]>) -> Vote {
        Vote {
            promised: Ballot { round, node: 7 },
            accepted: value.map(|v| AcceptedValue {
                ballot: Ballot { round, node: 7 },
                value: v.to_vec(),
            }),
        }
    }

    /// A half encodes to exact bytes at its documented offsets.
    #[test]
    fn a_half_encodes_to_its_layout() {
        let h = encode(5, b"world/w1", &vote(3, Some(b"v")));
        assert_eq!(&h[0..4], b"VOTE");
        assert_eq!(&h[4..12], &5u64.to_le_bytes());
        assert_eq!(&h[12..14], &[8, 0]);
        assert_eq!(&h[14..22], b"world/w1");
        assert_eq!(&h[142..150], &3u64.to_le_bytes());
        assert_eq!(&h[150..158], &7u64.to_le_bytes());
        assert_eq!(h[158], 1);
        assert_eq!(&h[175..177], &[1, 0]);
        assert_eq!(h[177], b'v');
        assert!(h[178..1020].iter().all(|b| *b == 0));
        assert_eq!(&h[1020..], &crc32fast::hash(&h[..1020]).to_le_bytes());
        assert_eq!(
            decode(&h).unwrap(),
            (5, b"world/w1".to_vec(), vote(3, Some(b"v")))
        );
    }

    /// The file is preallocated at first open and never grows; votes survive
    /// a reopen.
    #[test]
    fn votes_survive_a_reopen_and_the_file_never_grows() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("VOTES");
        let mut j = FileJournal::open(&path, 64 * 1024).unwrap();
        assert_eq!(j.capacity(), 32);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 64 * 1024);
        for i in 0..32u64 {
            j.write(format!("k{i}").as_bytes(), &vote(i, None)).unwrap();
        }
        assert!(j.write(b"one-too-many", &vote(1, None)).is_err());
        j.write(b"k3", &vote(9, Some(b"x"))).unwrap();
        j.free(b"k4").unwrap();
        j.write(b"fresh", &vote(1, None)).unwrap();
        drop(j);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 64 * 1024);
        let mut j = FileJournal::open(&path, 64 * 1024).unwrap();
        let votes: HashMap<Vec<u8>, Vote> = j.load().unwrap().into_iter().collect();
        assert_eq!(votes.len(), 32);
        assert_eq!(votes[b"k3".as_slice()], vote(9, Some(b"x")));
        assert!(!votes.contains_key(b"k4".as_slice()));
        assert_eq!(votes[b"fresh".as_slice()], vote(1, None));
    }

    /// A write torn by a crash leaves the previous vote readable.
    #[test]
    fn a_torn_write_leaves_the_previous_vote() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("VOTES");
        let mut j = FileJournal::open(&path, 16 * 1024).unwrap();
        j.write(b"gen/4", &vote(1, None)).unwrap();
        j.write(b"gen/4", &vote(2, Some(b"accepted"))).unwrap();
        // The next write goes to the other half: tear it half way.
        let (at, current) = j.slot_of(b"gen/4").unwrap();
        let next = if current == Some(0) { 1 } else { 0 };
        let torn = encode(3, b"gen/4", &vote(3, Some(b"newer")));
        let f = j.file();
        f.seek(SeekFrom::Start((at + next * HALF) as u64)).unwrap();
        f.write_all(&torn[..HALF / 2]).unwrap();
        f.sync_data().unwrap();
        drop(j);
        let mut j = FileJournal::open(&path, 16 * 1024).unwrap();
        assert_eq!(
            j.load().unwrap(),
            vec![(b"gen/4".to_vec(), vote(2, Some(b"accepted")))]
        );
        // And the journal carries on writing past the tear.
        j.write(b"gen/4", &vote(4, Some(b"later"))).unwrap();
        drop(j);
        let mut j = FileJournal::open(&path, 16 * 1024).unwrap();
        assert_eq!(
            j.load().unwrap(),
            vec![(b"gen/4".to_vec(), vote(4, Some(b"later")))]
        );
    }

    #[test]
    fn an_oversized_key_or_value_is_refused() {
        let dir = TempDir::new().unwrap();
        let mut j = FileJournal::open(&dir.path().join("VOTES"), 8 * 1024).unwrap();
        assert!(j.write(&[1; MAX_KEY + 1], &vote(1, None)).is_err());
        assert!(j.write(b"k", &vote(1, Some(&[0; MAX_VALUE + 1]))).is_err());
        assert!(j.write(b"k", &vote(1, Some(&[0; MAX_VALUE]))).is_ok());
    }
}
