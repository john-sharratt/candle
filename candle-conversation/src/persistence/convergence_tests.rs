//! **Two members converge** (`docs/replicated_substrate.md` §6.9, §15 chunk 1):
//! two substrates write conversations, sections, tombstones and file events
//! through the real write paths, competing on the same keys; every record each
//! wrote is shipped to the other in a seeded random order, in batches, with
//! duplicates and a reopen part-way through; and the two end holding the same
//! replicated state — every index, value, stamp and turn order — and the same
//! file events, before and after a compaction.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::json;
use tempfile::TempDir;

use super::content_hash::{hash_bytes, ContentHash};
use super::manifest::encode_label_payload;
use super::record::{ChunkPayload, CustomObjectPayload, RecordType, SectionGeneration};
use super::resume::encode_token_ids;
use super::streams::{ContentAddress, SectionDecl, StreamDecl, StreamId, TurnDecl};
use super::vfs::{VfsAppend, VfsEventId, VfsKill, VfsWrite};
use super::SubstratePersistence;
use crate::projection::TimelineId;
use crate::substrate::{Substrate, TurnMark};

/// A conversation both members write.
const SHARED: u64 = 0x7000_0001;
/// A conversation one member writes and later retires a turn of.
const RETIRED_TURN: u64 = 0x7000_0002;
/// A conversation one member writes and the other deletes.
const DELETED: u64 = 0x7000_0003;
const N_LAYERS: usize = 1;

struct Member {
    dir: TempDir,
    substrate: Substrate,
    sp: SubstratePersistence,
}

impl Member {
    fn new() -> Member {
        let dir = TempDir::new().unwrap();
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        Member { dir, substrate, sp }
    }

    fn reopen(&mut self) {
        self.sp.commit().unwrap();
        let mut substrate = Substrate::new();
        // The handle holds the writer lock: release it before the next open.
        self.sp.release_writer().unwrap();
        self.sp =
            SubstratePersistence::open_in_with_substrate(self.dir.path(), &mut substrate).unwrap();
        self.substrate = substrate;
    }

    fn node(&self) -> u64 {
        self.sp.node()
    }

    /// Turn `turn_index` of `timeline`: its declaration, tokens and chunk.
    fn turn(&mut self, timeline: u64, turn_index: u32, token: u32) {
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: timeline,
            turn_index,
            turn_id_day: 0,
            turn_id_seq: turn_index + 1,
            role: 1,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = self.sp.declare_stream(&decl).unwrap().value;
        self.sp
            .append_tokens(sid, &encode_token_ids(&[token]))
            .unwrap();
        self.chunk(sid, token as u8);
    }

    fn chunk(&mut self, sid: StreamId, seed: u8) {
        let payload = ChunkPayload {
            offset: 0,
            k_formats: vec![4],
            v_formats: vec![5],
            k_pal: vec![seed],
            v_pal: vec![seed],
            k_scale: vec![1.0],
            v_scale: vec![0.5],
            kv_bytes: (0..64u32).map(|i| (i as u8).wrapping_add(seed)).collect(),
        };
        self.sp.write_chunk(sid, 0, 1, 4, None, &payload).unwrap();
    }

    fn label(&mut self, timeline: u64, title: &str) {
        let payload = encode_label_payload(timeline, "shared", title, &BTreeMap::new());
        self.sp
            .append_record(RecordType::Label, 0, 0, 0, 0, 0, &payload)
            .unwrap();
    }

    /// Every record this member wrote, as the bytes it sends.
    fn events(&mut self) -> Vec<Vec<u8>> {
        let node = self.node();
        let through = self.sp.applied_through(node);
        (1..=through)
            .filter_map(|oseq| self.sp.read_event(node, oseq).unwrap())
            .collect()
    }

    fn apply(&mut self, batch: &[Vec<u8>]) {
        let refs: Vec<&[u8]> = batch.iter().map(Vec::as_slice).collect();
        self.sp
            .apply_remote(&mut self.substrate, &refs, N_LAYERS)
            .unwrap();
    }

    /// The replicated state and the shared conversation's file events, as a
    /// fresh open reads them.
    fn state(&mut self) -> (String, Vec<String>) {
        self.reopen();
        let files = self
            .sp
            .vfs_events(SHARED)
            .unwrap()
            .into_iter()
            .map(|e| format!("{:x}/{} {} {}", e.origin, e.seq, e.key, e.body))
            .collect();
        (self.substrate.replicated_state(), files)
    }
}

fn section() -> StreamDecl {
    StreamDecl::PromptSection(SectionDecl {
        address: ContentAddress {
            prefix_hash: ContentHash { lo: 5, hi: 0 },
            section_hash: hash_bytes(b"prelude"),
        },
        debug_name: "prelude".into(),
    })
}

fn appends(key: &str, ns: &[u64]) -> Vec<VfsAppend> {
    ns.iter()
        .map(|n| VfsAppend {
            repo: "candle".into(),
            key: key.into(),
            body: json!({ "n": n }),
        })
        .collect()
}

/// What each member writes, competing on every key they share.
fn write(a: &mut Member, b: &mut Member) {
    // Two writers' turn 0 of one conversation, and the second's turn 1.
    a.turn(SHARED, 0, 10);
    b.turn(SHARED, 0, 20);
    b.turn(SHARED, 1, 21);
    a.label(SHARED, "from a");
    b.label(SHARED, "from b");
    a.sp.write_debug_id(SHARED, "a's", a.sp.mint()).unwrap();
    b.sp.write_debug_id(SHARED, "b's", b.sp.mint()).unwrap();
    for (m, n) in [(&mut *a, 1u64), (&mut *b, 2)] {
        let id = m.sp.mint();
        m.sp.write_custom_object(&CustomObjectPayload::new("flag", json!(n)), id)
            .unwrap();
    }

    // A turn its writer retires.
    a.turn(RETIRED_TURN, 0, 30);
    a.turn(RETIRED_TURN, 1, 31);
    let mark = TurnMark {
        timeline: TimelineId::from_raw(RETIRED_TURN).unwrap(),
        turn_index: 1,
        writer: a.node(),
    };
    a.sp.write_turn_tombstone(mark, None).unwrap();

    // A conversation one member writes and the other deletes.
    b.turn(DELETED, 0, 40);
    b.label(DELETED, "doomed");
    a.sp.write_tombstone(DELETED, None).unwrap();

    // One section prefilled by both; the second ends its generation and
    // prefills it again.
    for m in [&mut *a, &mut *b] {
        let sid = m.sp.declare_stream(&section()).unwrap().value;
        m.chunk(sid, 50);
    }
    let first = b.sp.declare_stream(&section()).unwrap();
    b.chunk(first.value, 51);
    b.sp.write_section_tombstone(
        first.value.0,
        Some(SectionGeneration {
            writer: b.node(),
            oseq: first.id.oseq,
        }),
        None,
    )
    .unwrap();
    let again = b.sp.declare_stream(&section()).unwrap().value;
    b.chunk(again, 52);

    // File events on one key from both, and one member's kill of the
    // other's first event.
    a.sp.write_vfs(
        SHARED,
        &VfsWrite {
            tombstones: Vec::new(),
            events: appends("a.rs", &[1, 2]),
        },
    )
    .unwrap();
    let a_first = VfsEventId {
        origin: a.node(),
        seq: 0,
    };
    b.sp.write_vfs(
        SHARED,
        &VfsWrite {
            tombstones: vec![VfsKill {
                repo: "candle".into(),
                key: "a.rs".into(),
                kills: vec![a_first],
            }],
            events: appends("a.rs", &[3]),
        },
    )
    .unwrap();
    a.sp.commit().unwrap();
    b.sp.commit().unwrap();
}

/// Ship `events` to `to` in a shuffled order, in batches of one to four, each
/// batch followed now and then by a duplicate of an earlier event, with a
/// reopen half-way.
fn ship(events: &[Vec<u8>], to: &mut Member, rng: &mut StdRng) {
    let mut order: Vec<usize> = (0..events.len()).collect();
    order.shuffle(rng);
    let mut sent = 0;
    while sent < order.len() {
        let n = rng.random_range(1..=4).min(order.len() - sent);
        let mut batch: Vec<Vec<u8>> = order[sent..sent + n]
            .iter()
            .map(|&i| events[i].clone())
            .collect();
        if sent > 0 && rng.random_bool(0.3) {
            batch.push(events[order[rng.random_range(0..sent)]].clone());
        }
        to.apply(&batch);
        let half = order.len() / 2;
        if sent < half && sent + n >= half {
            to.reopen();
        }
        sent += n;
    }
}

#[test]
fn two_members_converge_whatever_the_order_of_arrival() {
    for seed in [1u64, 2, 3, 4] {
        let mut a = Member::new();
        let mut b = Member::new();
        write(&mut a, &mut b);
        let from_a = a.events();
        let from_b = b.events();
        let mut rng = StdRng::seed_from_u64(seed);
        ship(&from_a, &mut b, &mut rng);
        ship(&from_b, &mut a, &mut rng);

        let (state_a, files_a) = a.state();
        let (state_b, files_b) = b.state();
        assert_eq!(state_a, state_b, "seed {seed}");
        assert_eq!(files_a, files_b, "seed {seed}");
        assert_eq!(
            files_a.len(),
            2,
            "a's second event and b's; b killed a's first"
        );

        // A compaction on either side changes nothing the other can see.
        b.sp.compact(&mut b.substrate, None).unwrap();
        let (state_b, files_b) = b.state();
        assert_eq!(state_a, state_b, "seed {seed}, after compaction");
        assert_eq!(files_a, files_b, "seed {seed}, after compaction");
    }
}
