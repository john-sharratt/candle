//! The substrate's stream index: every stream's runtime state, kept per
//! writer, and the logical view every caller reads it through
//! (`docs/replicated_substrate.md` §6.6).
//!
//! **Physically, a stream is `(stream_id, writer)`.** Two members that each
//! wrote turn *N* of one timeline wrote the same stream id —
//! `turn_stream_id(timeline, N)` is a function of the timeline and the index
//! alone — so keying the index by stream id would interleave their K/V under
//! one key. Each record carries its writer's origin in its header, and that is
//! the second half of the key. A stream's records are all written by the
//! member that declared it, at the turn's seal.
//!
//! **Logically, a stream is what its id names.** A timeline's declared turns
//! are ordered by their declaration's version stamp, and the turn at index *N*
//! is the one at that place in the order, whoever wrote it. A lookup by
//! `turn_stream_id(timeline, N)` resolves to that turn's stream, so the call
//! sites that derive a turn's stream id from its index stay as they are. The
//! order is a function of the declarations held, never of the order they
//! arrived in.
//!
//! A turn never sits below its own index: a timeline whose declaration was lost
//! in a crash keeps a gap where it was, and the turns above stay where they
//! were written. So the position of each turn is `max(previous + 1, its own
//! index)` down the stamp order — for one writer exactly the index it wrote,
//! and for two the interleaving their stamps give.
//!
//! A section is content-addressed, so two members that prefilled it wrote one
//! id with different bytes. Both are kept, and the id resolves to the
//! declaration with the greatest stamp.

use std::collections::{BTreeMap, HashMap};

use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::identity::{RecordIdentity, VersionStamp};
use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};
use crate::projection::TurnIndex;

use super::stamps::take;
use super::StreamRuntime;

/// A declared turn's stream, as callers address it.
#[derive(Debug, Clone, Copy)]
pub struct TurnStream<'a> {
    /// The writer's stream the turn's records are filed under.
    pub key: StreamKey,
    /// The id the turn is addressed by — its position's, which is not the id
    /// its records carry when another writer's turn placed it higher.
    pub stream_id: StreamId,
    /// The turn's position in its timeline.
    pub index: TurnIndex,
    /// The turn's declaration, as its writer wrote it: its `turn_index` is
    /// the index the writer gave it, which `index` supersedes.
    pub decl: &'a TurnDecl,
    pub runtime: &'a StreamRuntime,
}

/// One writer's stream: the stream id and the origin of the member that
/// wrote it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamKey {
    pub stream_id: StreamId,
    pub writer: u64,
}

impl StreamKey {
    /// The stream a record with identity `id` belongs to.
    pub fn of(stream_id: StreamId, id: &RecordIdentity) -> StreamKey {
        StreamKey {
            stream_id,
            writer: id.origin,
        }
    }
}

/// Where a declared turn sorts in its timeline: its declaration's stamp, then
/// its own index (records written before identities all carry the zero stamp,
/// and among them the index is the order they were written in), then its
/// writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TurnRank {
    stamp: VersionStamp,
    turn_index: u32,
    writer: u64,
}

/// See the module documentation.
#[derive(Debug, Default)]
pub struct StreamIndex {
    /// Every stream, by id and then by writer.
    streams: HashMap<StreamId, BTreeMap<u64, StreamRuntime>>,
    /// Each timeline's declared turns, in rank order.
    timelines: HashMap<u64, BTreeMap<TurnRank, StreamKey>>,
    /// The timeline and rank of every declared turn.
    ranks: HashMap<StreamKey, (u64, TurnRank)>,
    /// Where every declared turn sits in its timeline.
    positions: HashMap<StreamKey, u32>,
    /// The stream id of each turn position, to the stream at that position.
    logical: HashMap<StreamId, StreamKey>,
}

impl StreamIndex {
    /// The stream `stream_id` names: the turn at that position in its
    /// timeline, or the section or undeclared stream of that id.
    pub fn get(&self, stream_id: StreamId) -> Option<&StreamRuntime> {
        self.resolve(stream_id).and_then(|key| self.physical(key))
    }

    /// The writer's stream `stream_id` names — see [`Self::get`].
    ///
    /// A declared turn is reached only through its position: an id whose
    /// position holds no turn names nothing, even when some writer's turn
    /// further up the timeline was written under it.
    pub fn resolve(&self, stream_id: StreamId) -> Option<StreamKey> {
        if let Some(key) = self.logical.get(&stream_id) {
            return Some(*key);
        }
        let writers = self.streams.get(&stream_id)?;
        writers
            .iter()
            .filter(|(_, rt)| !matches!(rt.decl, Some(StreamDecl::Turn(_))))
            .max_by_key(|(writer, rt)| (rt.ids.decl.stamp(), **writer))
            .map(|(writer, _)| StreamKey {
                stream_id,
                writer: *writer,
            })
    }

    /// One writer's stream.
    pub fn physical(&self, key: StreamKey) -> Option<&StreamRuntime> {
        self.streams.get(&key.stream_id)?.get(&key.writer)
    }

    /// One writer's stream, created empty if the index has none.
    pub fn entry(&mut self, key: StreamKey) -> &mut StreamRuntime {
        self.streams
            .entry(key.stream_id)
            .or_default()
            .entry(key.writer)
            .or_default()
    }

    /// Install `decl` on `key`'s stream, written as record `id`, when it
    /// supersedes the declaration held. `None` when it does not; otherwise the
    /// logical ids whose stream changed, which every memo keyed by logical id
    /// must drop.
    pub fn declare(
        &mut self,
        key: StreamKey,
        decl: StreamDecl,
        id: RecordIdentity,
    ) -> Option<Vec<StreamId>> {
        let rt = self.entry(key);
        if !take(&mut rt.ids.decl, id) {
            return None;
        }
        let turn = match &decl {
            StreamDecl::Turn(t) => Some((t.timeline_id, t.turn_index)),
            StreamDecl::PromptSection(_) => None,
        };
        rt.decl = Some(decl);
        let mut changed = Vec::new();
        if let Some((timeline, rank)) = self.unrank(key, &mut changed) {
            changed.extend(self.place(timeline, rank));
        }
        if let Some((timeline, turn_index)) = turn {
            let rank = TurnRank {
                stamp: id.stamp(),
                turn_index,
                writer: key.writer,
            };
            self.ranks.insert(key, (timeline, rank));
            let order = self.timelines.entry(timeline).or_default();
            order.insert(rank, key);
            changed.extend(self.place(timeline, rank));
        }
        changed.sort_unstable();
        changed.dedup();
        Some(changed)
    }

    /// Take `key`'s declaration away, returning the logical ids whose stream
    /// changed. What the stream holds besides stays.
    pub fn undeclare(&mut self, key: StreamKey) -> Vec<StreamId> {
        let mut changed = vec![key.stream_id];
        if let Some((timeline, rank)) = self.unrank(key, &mut changed) {
            changed.extend(self.place(timeline, rank));
        }
        if let Some(rt) = self
            .streams
            .get_mut(&key.stream_id)
            .and_then(|w| w.get_mut(&key.writer))
        {
            rt.decl = None;
            rt.ids.decl = RecordIdentity::NONE;
        }
        self.drop_if_empty(key);
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// Drop every value of `key`'s stream whose record `keep` refuses — the
    /// declaration aside — and the stream itself once it holds nothing.
    /// Returns whether anything was dropped.
    pub fn prune(&mut self, key: StreamKey, keep: impl Fn(&RecordIdentity) -> bool) -> bool {
        let Some(rt) = self
            .streams
            .get_mut(&key.stream_id)
            .and_then(|w| w.get_mut(&key.writer))
        else {
            return false;
        };
        let mut dropped = false;
        let ids = &mut rt.ids;
        let gone: Vec<u64> = ids
            .chunks
            .iter()
            .filter(|(_, id)| !keep(id))
            .map(|(idx, _)| *idx)
            .collect();
        for idx in gone {
            ids.chunks.remove(&idx);
            rt.chunks.remove(&idx);
            dropped = true;
        }
        if rt.tokens.is_some() && !keep(&ids.tokens) {
            rt.tokens = None;
            ids.tokens = RecordIdentity::NONE;
            dropped = true;
        }
        if rt.projection_events.is_some() && !keep(&ids.projection_events) {
            rt.projection_events = None;
            ids.projection_events = RecordIdentity::NONE;
            dropped = true;
        }
        if rt.wide_q_sigs.is_some() && !keep(&ids.wide_q_sigs) {
            rt.wide_q_sigs = None;
            ids.wide_q_sigs = RecordIdentity::NONE;
            dropped = true;
        }
        if rt.index_page.is_some() && !keep(&ids.index_page) {
            rt.index_page = None;
            ids.index_page = RecordIdentity::NONE;
            dropped = true;
        }
        if rt.committed_through.is_some() && !keep(&ids.commit) {
            rt.committed_through = None;
            ids.commit = RecordIdentity::NONE;
            dropped = true;
        }
        self.drop_if_empty(key);
        dropped
    }

    /// Forget `key`'s stream when it holds nothing at all.
    fn drop_if_empty(&mut self, key: StreamKey) {
        let Some(writers) = self.streams.get_mut(&key.stream_id) else {
            return;
        };
        if writers
            .get(&key.writer)
            .is_some_and(StreamRuntime::is_empty)
        {
            writers.remove(&key.writer);
        }
        if writers.is_empty() {
            self.streams.remove(&key.stream_id);
        }
    }

    /// Drop one writer's stream, returning the logical ids whose stream
    /// changed.
    pub fn remove_writer(&mut self, key: StreamKey) -> Vec<StreamId> {
        let Some(writers) = self.streams.get_mut(&key.stream_id) else {
            return Vec::new();
        };
        if writers.remove(&key.writer).is_none() {
            return Vec::new();
        }
        if writers.is_empty() {
            self.streams.remove(&key.stream_id);
        }
        let mut changed = vec![key.stream_id];
        if let Some((timeline, rank)) = self.unrank(key, &mut changed) {
            changed.extend(self.place(timeline, rank));
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// Take `key` out of its timeline's order, answering which timeline and
    /// rank that was and pushing the id it vacates onto `changed`. The caller
    /// re-places the timeline from that rank.
    fn unrank(&mut self, key: StreamKey, changed: &mut Vec<StreamId>) -> Option<(u64, TurnRank)> {
        let (timeline, rank) = self.ranks.remove(&key)?;
        if let Some(order) = self.timelines.get_mut(&timeline) {
            order.remove(&rank);
            if order.is_empty() {
                self.timelines.remove(&timeline);
            }
        }
        if let Some(pos) = self.positions.remove(&key) {
            let sid = turn_stream_id(timeline, pos);
            if self.logical.get(&sid) == Some(&key) {
                self.logical.remove(&sid);
                changed.push(sid);
            }
        }
        Some((timeline, rank))
    }

    /// Recompute the positions of `timeline`'s turns from rank `from` up — a
    /// turn was inserted there or taken out — returning the logical ids whose
    /// stream changed.
    ///
    /// A turn's position depends only on the position of the turn before it,
    /// so the first turn past `from` that keeps its place ends the walk: every
    /// turn above it keeps its place too. One writer's turns never move, so an
    /// open that meets a timeline's declarations in any order places each in
    /// constant work.
    fn place(&mut self, timeline: u64, from: TurnRank) -> Vec<StreamId> {
        let mut changed = Vec::new();
        let Some(order) = self.timelines.get(&timeline) else {
            return changed;
        };
        let mut next: u32 = order
            .range(..from)
            .next_back()
            .and_then(|(_, key)| self.positions.get(key))
            .map_or(0, |pos| pos + 1);
        for (rank, key) in order.range(from..) {
            let pos = next.max(rank.turn_index);
            next = pos + 1;
            let sid = turn_stream_id(timeline, pos);
            if self.positions.get(key) == Some(&pos) && self.logical.get(&sid) == Some(key) {
                if *rank == from {
                    continue;
                }
                break;
            }
            if let Some(old) = self.positions.insert(*key, pos) {
                let old_sid = turn_stream_id(timeline, old);
                if self.logical.get(&old_sid) == Some(key) {
                    self.logical.remove(&old_sid);
                    changed.push(old_sid);
                }
            }
            self.logical.insert(sid, *key);
            changed.push(sid);
        }
        changed
    }

    /// `key`'s stream as a [`TurnStream`], when it is a declared turn.
    pub fn turn(&self, key: StreamKey) -> Option<TurnStream<'_>> {
        let runtime = self.physical(key)?;
        let Some(StreamDecl::Turn(decl)) = &runtime.decl else {
            return None;
        };
        let pos = self.position(key)?;
        Some(TurnStream {
            key,
            stream_id: turn_stream_id(decl.timeline_id, pos),
            index: TurnIndex(pos),
            decl,
            runtime,
        })
    }

    /// Where `key`'s turn sits in its timeline; `None` for a stream that is
    /// not a declared turn.
    pub fn position(&self, key: StreamKey) -> Option<u32> {
        self.positions.get(&key).copied()
    }

    /// The id callers name `key`'s stream by: its position's for a declared
    /// turn, its own otherwise.
    pub fn logical_id(&self, key: StreamKey) -> StreamId {
        match self.ranks.get(&key) {
            Some((timeline, _)) => {
                turn_stream_id(*timeline, self.positions.get(&key).copied().unwrap_or(0))
            }
            None => key.stream_id,
        }
    }

    /// `timeline`'s declared turns in order, each with its position.
    pub fn turns_of(&self, timeline: u64) -> impl Iterator<Item = (u32, StreamKey)> + '_ {
        self.timelines
            .get(&timeline)
            .into_iter()
            .flat_map(|order| order.values())
            .filter_map(|key| self.positions.get(key).map(|pos| (*pos, *key)))
    }

    /// Every timeline with a declared turn.
    pub fn timelines(&self) -> impl Iterator<Item = u64> + '_ {
        self.timelines.keys().copied()
    }

    /// Every writer's stream.
    pub fn iter_physical(&self) -> impl Iterator<Item = (StreamKey, &StreamRuntime)> + '_ {
        self.streams.iter().flat_map(|(sid, writers)| {
            writers.iter().map(move |(writer, rt)| {
                (
                    StreamKey {
                        stream_id: *sid,
                        writer: *writer,
                    },
                    rt,
                )
            })
        })
    }

    /// Every stream an id names, under that id: each declared turn under its
    /// position's id, and each other id's resolved stream.
    pub fn iter_logical(&self) -> impl Iterator<Item = (StreamId, &StreamRuntime)> + '_ {
        self.iter_physical().filter_map(|(key, rt)| {
            if self.ranks.contains_key(&key) {
                return Some((self.logical_id(key), rt));
            }
            (self.resolve(key.stream_id) == Some(key)).then_some((key.stream_id, rt))
        })
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.streams.clear();
        self.timelines.clear();
        self.ranks.clear();
        self.positions.clear();
        self.logical.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::ContentHash;
    use crate::persistence::manifest::RecordLoc;
    use crate::persistence::segment::SegmentId;
    use crate::persistence::streams::{ContentAddress, SectionDecl};

    const TL: u64 = 77;

    fn id(origin: u64, hlc: u64) -> RecordIdentity {
        RecordIdentity {
            origin,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        }
    }

    fn turn(turn_index: u32) -> StreamDecl {
        StreamDecl::Turn(TurnDecl {
            timeline_id: TL,
            turn_index,
            turn_id_day: 0,
            turn_id_seq: turn_index + 1,
            role: 1,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        })
    }

    fn section() -> StreamDecl {
        StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 1, hi: 2 },
                section_hash: ContentHash { lo: 3, hi: 4 },
            },
            debug_name: "sys".into(),
        })
    }

    fn sid(n: u32) -> StreamId {
        turn_stream_id(TL, n)
    }

    /// Declare `turn_index` as written by `origin` at `hlc`, with its tokens
    /// marked by `mark` so a lookup can say whose stream it reached.
    fn write(index: &mut StreamIndex, turn_index: u32, origin: u64, hlc: u64, mark: u64) {
        let key = StreamKey {
            stream_id: sid(turn_index),
            writer: origin,
        };
        index.declare(key, turn(turn_index), id(origin, hlc));
        index.entry(key).committed_through = Some(mark);
    }

    fn mark_at(index: &StreamIndex, pos: u32) -> Option<u64> {
        index.get(sid(pos)).and_then(|rt| rt.committed_through)
    }

    fn order(index: &StreamIndex) -> Vec<(u32, u64, u64)> {
        index
            .turns_of(TL)
            .map(|(pos, key)| {
                let rt = index.physical(key).unwrap();
                let StreamDecl::Turn(t) = rt.decl.as_ref().unwrap() else {
                    unreachable!()
                };
                (pos, t.turn_index as u64, key.writer)
            })
            .collect()
    }

    /// One writer's turns sit at the indices it wrote, and every lookup
    /// reaches its own stream.
    #[test]
    fn a_single_writer_keeps_its_indices() {
        let mut index = StreamIndex::default();
        for n in 0..4 {
            write(&mut index, n, 5, 10 + n as u64, 100 + n as u64);
        }
        assert_eq!(
            order(&index),
            vec![(0, 0, 5), (1, 1, 5), (2, 2, 5), (3, 3, 5)]
        );
        for n in 0..4 {
            assert_eq!(mark_at(&index, n), Some(100 + n as u64));
        }
        assert_eq!(index.get(sid(4)).map(|_| ()), None);
    }

    /// Two writers' turn 2 share a stream id and both are kept: the one with
    /// the lesser stamp sits at 2, the other above it, and turn 3 of the first
    /// writer — written after both — above that.
    #[test]
    fn two_writers_turn_n_interleave_by_stamp() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        write(&mut index, 1, 5, 11, 101);
        write(&mut index, 2, 5, 12, 102);
        write(&mut index, 2, 9, 13, 902);
        assert_eq!(
            order(&index),
            vec![(0, 0, 5), (1, 1, 5), (2, 2, 5), (3, 2, 9)]
        );
        assert_eq!(mark_at(&index, 2), Some(102));
        assert_eq!(mark_at(&index, 3), Some(902), "the second turn 2 is at 3");
        assert_eq!(index.iter_physical().count(), 4);
    }

    /// The order is a function of the declarations held: every arrival order
    /// of the same set places every turn the same.
    #[test]
    fn the_order_is_the_same_in_any_arrival_order() {
        let writes: [(u32, u64, u64, u64); 5] = [
            (0, 5, 10, 100),
            (1, 5, 11, 101),
            (1, 9, 12, 901),
            (2, 9, 13, 902),
            (2, 5, 14, 102),
        ];
        let mut expected: Option<(Vec<(u32, u64, u64)>, Vec<Option<u64>>)> = None;
        let mut perm: Vec<usize> = (0..writes.len()).collect();
        // Every permutation, by Heap's algorithm.
        let mut c = vec![0usize; perm.len()];
        let mut run = |perm: &[usize]| {
            let mut index = StreamIndex::default();
            for &i in perm {
                let (n, origin, hlc, mark) = writes[i];
                write(&mut index, n, origin, hlc, mark);
            }
            let got = (order(&index), (0..6).map(|p| mark_at(&index, p)).collect());
            match &expected {
                None => expected = Some(got),
                Some(e) => assert_eq!(&got, e, "arrival order {perm:?}"),
            }
        };
        run(&perm);
        let mut i = 0;
        while i < perm.len() {
            if c[i] < i {
                if i % 2 == 0 {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                run(&perm);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        let (placed, marks) = expected.unwrap();
        assert_eq!(
            placed,
            vec![(0, 0, 5), (1, 1, 5), (2, 1, 9), (3, 2, 9), (4, 2, 5)]
        );
        assert_eq!(
            marks,
            vec![Some(100), Some(101), Some(901), Some(902), Some(102), None]
        );
    }

    /// A lost declaration leaves a gap: the turns above it stay at the
    /// indices they were written at, and the gap's id names nothing.
    #[test]
    fn a_lost_declaration_leaves_a_gap() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        write(&mut index, 1, 5, 11, 101);
        write(&mut index, 3, 5, 13, 103);
        assert_eq!(order(&index), vec![(0, 0, 5), (1, 1, 5), (3, 3, 5)]);
        assert_eq!(mark_at(&index, 2), None);
        assert_eq!(mark_at(&index, 3), Some(103));
    }

    /// A declared turn is reached only through its position: when another
    /// writer's turn sits at the index a stream id names, the turn written
    /// under that id is not what the id resolves to.
    #[test]
    fn a_declared_turn_is_reached_only_through_its_position() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        write(&mut index, 5, 5, 11, 105);
        write(&mut index, 3, 9, 12, 903);
        // 0 at 0, 5 at 5, and the other writer's 3 above them at 6: the gap
        // at 3 names nothing, though a stream of id 3 exists.
        assert_eq!(order(&index), vec![(0, 0, 5), (5, 5, 5), (6, 3, 9)]);
        assert_eq!(mark_at(&index, 3), None);
        assert_eq!(mark_at(&index, 6), Some(903));
    }

    /// Records under a stream id with no declaration yet are kept under their
    /// writer and reached by the id; the declaration then places the stream.
    #[test]
    fn records_before_their_declaration_wait_under_their_writer() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        let early = StreamKey {
            stream_id: sid(1),
            writer: 9,
        };
        index.entry(early).committed_through = Some(901);
        assert_eq!(mark_at(&index, 1), Some(901));
        let changed = index.declare(early, turn(1), id(9, 20)).unwrap();
        assert_eq!(changed, vec![sid(1)]);
        assert_eq!(order(&index), vec![(0, 0, 5), (1, 1, 9)]);
        assert_eq!(mark_at(&index, 1), Some(901));
    }

    /// A turn arriving ranked below one already placed moves it up, and the
    /// ids whose stream changed are reported.
    #[test]
    fn a_turn_ranked_below_reports_every_moved_id() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        write(&mut index, 1, 9, 30, 901);
        let key = StreamKey {
            stream_id: sid(1),
            writer: 5,
        };
        let changed = index.declare(key, turn(1), id(5, 20)).unwrap();
        let mut want = vec![sid(1), sid(2)];
        want.sort_unstable();
        assert_eq!(changed, want);
        assert_eq!(mark_at(&index, 2), Some(901));
    }

    /// A declaration ranked below the one held changes nothing.
    #[test]
    fn a_lesser_declaration_is_refused() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        let key = StreamKey {
            stream_id: sid(0),
            writer: 5,
        };
        assert!(index.declare(key, turn(0), id(5, 9)).is_none());
        assert_eq!(order(&index), vec![(0, 0, 5)]);
    }

    /// Two writers' prefill of one section: both are kept, and the id
    /// resolves to the greater declaration.
    #[test]
    fn a_section_resolves_to_the_greatest_declaration() {
        let mut index = StreamIndex::default();
        let s = StreamId(4242);
        let a = StreamKey {
            stream_id: s,
            writer: 5,
        };
        let b = StreamKey {
            stream_id: s,
            writer: 9,
        };
        index.declare(b, section(), id(9, 30));
        index.entry(b).committed_through = Some(9);
        index.declare(a, section(), id(5, 20));
        index.entry(a).committed_through = Some(5);
        assert_eq!(index.resolve(s), Some(b));
        assert_eq!(index.get(s).unwrap().committed_through, Some(9));
        let logical: Vec<StreamId> = index.iter_logical().map(|(id, _)| id).collect();
        assert_eq!(
            logical,
            vec![s],
            "the losing prefill is not a second stream"
        );
        assert_eq!(index.iter_physical().count(), 2);
    }

    /// Removing one writer's stream re-places the turns of its timeline and
    /// leaves the other writer's stream of the same id.
    #[test]
    fn removing_a_writers_stream_re_places_its_timeline() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        write(&mut index, 1, 5, 11, 101);
        write(&mut index, 1, 9, 12, 901);
        assert_eq!(mark_at(&index, 2), Some(901));
        let changed = index.remove_writer(StreamKey {
            stream_id: sid(1),
            writer: 5,
        });
        assert!(changed.contains(&sid(1)));
        assert!(changed.contains(&sid(2)));
        assert_eq!(order(&index), vec![(0, 0, 5), (1, 1, 9)]);
        assert_eq!(mark_at(&index, 1), Some(901));
        assert_eq!(mark_at(&index, 2), None);
    }

    /// Pruning drops the values a predicate refuses, and the stream with them
    /// once nothing is left; undeclaring keeps the values.
    #[test]
    fn pruning_and_undeclaring_a_stream() {
        let mut index = StreamIndex::default();
        let key = StreamKey {
            stream_id: StreamId(4242),
            writer: 5,
        };
        index.declare(key, section(), id(5, 20));
        let rt = index.entry(key);
        rt.committed_through = Some(3);
        rt.ids.commit = RecordIdentity {
            epoch: 20,
            ..id(5, 21)
        };
        rt.tokens = Some(RecordLoc {
            segment: SegmentId(1),
            offset: 0,
            payload_len: 8,
            record_size: 4096,
        });
        rt.ids.tokens = RecordIdentity {
            epoch: 7,
            ..id(5, 8)
        };
        assert!(index.prune(key, |held| held.epoch >= 20));
        let rt = index.physical(key).unwrap();
        assert!(rt.tokens.is_none());
        assert_eq!(rt.committed_through, Some(3));
        assert!(
            !index.prune(key, |held| held.epoch >= 20),
            "nothing left to drop"
        );
        index.undeclare(key);
        assert!(index.physical(key).unwrap().decl.is_none());
        assert!(index.prune(key, |_| false));
        assert!(
            index.physical(key).is_none(),
            "an empty stream is forgotten"
        );
    }

    /// The logical view names each declared turn by its position.
    #[test]
    fn the_logical_view_names_turns_by_position() {
        let mut index = StreamIndex::default();
        write(&mut index, 0, 5, 10, 100);
        write(&mut index, 0, 9, 11, 900);
        let mut got: Vec<(StreamId, Option<u64>)> = index
            .iter_logical()
            .map(|(id, rt)| (id, rt.committed_through))
            .collect();
        got.sort_unstable();
        let mut want = vec![(sid(0), Some(100)), (sid(1), Some(900))];
        want.sort_unstable();
        assert_eq!(got, want);
    }
}
