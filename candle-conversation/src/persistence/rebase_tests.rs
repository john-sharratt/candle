//! **Writing while cut off, and rejoining** (`docs/replicated_substrate.md`
//! §6.7, §14.1 "generations"): what a member writes cut off carries the
//! provisional generation and sorts above everything it holds; a rejoin takes
//! a decided generation and fixes it for every one of those records, wherever
//! the member reads them back — at open, though the `Rebase` record comes
//! after the records it fixes, after every rewrite, and when it sends them.

use tempfile::TempDir;

use super::content_hash::{hash_bytes, ContentHash};
use super::identity::{LocalGenerations, RecordIdentity, GEN_DETACHED};
use super::maintenance::MaintenanceOp;
use super::rebase::RebaseEntry;
use super::record::decode_header;
use super::segment::SegmentId;
use super::streams::{ContentAddress, SectionDecl, StreamDecl, StreamId};
use super::SubstratePersistence;
use crate::substrate::{StreamKey, Substrate};

fn open(dir: &TempDir) -> (Substrate, SubstratePersistence) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    (substrate, sp)
}

fn section(name: &str) -> StreamDecl {
    StreamDecl::PromptSection(SectionDecl {
        address: ContentAddress {
            prefix_hash: ContentHash { lo: 1, hi: 0 },
            section_hash: hash_bytes(name.as_bytes()),
        },
        debug_name: name.to_string(),
    })
}

/// Declare section `name`, returning its stream and the identity the
/// declaration was written with.
fn declare(sp: &mut SubstratePersistence, name: &str) -> (StreamId, RecordIdentity) {
    let at = sp.declare_stream(&section(name)).unwrap();
    (at.value, at.id)
}

/// The identity the substrate holds for `stream`'s declaration by `sp`'s node.
fn held(substrate: &Substrate, sp: &SubstratePersistence, stream: StreamId) -> RecordIdentity {
    substrate
        .stream_at(StreamKey {
            stream_id: stream,
            writer: sp.node(),
        })
        .unwrap()
        .ids
        .decl
}

/// The generation the record `id` carries when read back to send.
fn sent_gen(sp: &mut SubstratePersistence, id: RecordIdentity) -> u64 {
    let bytes = sp.read_event(id.origin, id.oseq).unwrap().unwrap();
    decode_header(&bytes).unwrap().0.gen
}

/// **A cut-off member's records are detached**, sort above everything it
/// holds, and it is still cut off when it reopens.
#[test]
fn a_cut_off_member_writes_detached_and_reopens_cut_off() {
    let dir = TempDir::new().unwrap();
    let (joined, detached) = {
        let (_, mut sp) = open(&dir);
        let joined = declare(&mut sp, "joined");
        sp.detach();
        assert!(sp.is_detached());
        let detached = declare(&mut sp, "detached");
        assert_eq!(joined.1.gen, 0);
        assert_eq!(detached.1.gen, GEN_DETACHED);
        assert!(detached.1.stamp() > joined.1.stamp());
        sp.commit().unwrap();
        (joined, detached)
    };
    let (substrate, mut sp) = open(&dir);
    assert!(sp.is_detached(), "it has not rejoined");
    assert_eq!(held(&substrate, &sp, detached.0).gen, GEN_DETACHED);
    assert_eq!(held(&substrate, &sp, joined.0).gen, 0);
    assert_eq!(declare(&mut sp, "later").1.gen, GEN_DETACHED);
}

/// **A rejoin fixes the generation of everything written cut off** — at
/// open, though its `Rebase` comes after the records it fixes; after a
/// segment rewrite and a compaction; and on every record read back to send —
/// and the member mints in that generation from then on.
#[test]
fn a_rejoin_fixes_the_generation_wherever_records_are_read_back() {
    let dir = TempDir::new().unwrap();
    let generations = LocalGenerations::new();
    let (first, second) = {
        let (_, mut sp) = open(&dir);
        declare(&mut sp, "joined");
        sp.detach();
        let first = declare(&mut sp, "one");
        let second = declare(&mut sp, "two");
        let entry = sp.rejoin(&generations).unwrap().unwrap();
        assert_eq!(
            entry,
            RebaseEntry {
                origin: sp.node(),
                from: first.1.oseq,
                through: second.1.oseq,
                gen: 1,
            }
        );
        assert!(!sp.is_detached());
        assert_eq!(declare(&mut sp, "after").1.gen, 1);
        assert_eq!(sent_gen(&mut sp, first.1), 1);
        assert_eq!(sent_gen(&mut sp, second.1), 1);
        sp.commit().unwrap();
        (first, second)
    };
    let check = |substrate: &Substrate, sp: &mut SubstratePersistence, context: &str| {
        assert!(!sp.is_detached(), "{context}");
        for (stream, id) in [first, second] {
            let fixed = RecordIdentity { gen: 1, ..id };
            assert_eq!(held(substrate, sp, stream), fixed, "{context}");
            assert_eq!(sent_gen(sp, id), 1, "{context}");
        }
    };
    let (mut substrate, mut sp) = open(&dir);
    check(&substrate, &mut sp, "reopened");
    sp.seal_active().unwrap();
    sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
        .unwrap();
    check(&substrate, &mut sp, "segment rewritten");
    drop(sp);
    let (mut substrate, mut sp) = open(&dir);
    check(&substrate, &mut sp, "reopened after the segment rewrite");
    sp.compact(&mut substrate, None).unwrap();
    check(&substrate, &mut sp, "compacted");
    drop(sp);
    let (substrate, mut sp) = open(&dir);
    check(&substrate, &mut sp, "reopened after compaction");
    assert_eq!(sp.rebases().entries().len(), 1);
}

/// **A member that wrote nothing cut off has nothing to place** and takes no
/// generation.
#[test]
fn a_member_that_wrote_nothing_cut_off_takes_no_generation() {
    let dir = TempDir::new().unwrap();
    let generations = LocalGenerations::new();
    let (_, mut sp) = open(&dir);
    declare(&mut sp, "joined");
    sp.detach();
    assert_eq!(sp.rejoin(&generations).unwrap(), None);
    assert!(!sp.is_detached());
    assert_eq!(generations.decided(), 0);
    assert_eq!(declare(&mut sp, "after").1.gen, 0);
}

/// **Two members cut off together take generations in the order they
/// rejoin**, so the second's records sort after the first's on every member.
#[test]
fn members_cut_off_together_rejoin_in_order() {
    let generations = LocalGenerations::new();
    let mut sent = Vec::new();
    for _ in 0..2 {
        let dir = TempDir::new().unwrap();
        let (_, mut sp) = open(&dir);
        sp.detach();
        let (_, id) = declare(&mut sp, "offline");
        sp.rejoin(&generations).unwrap().unwrap();
        let bytes = sp.read_event(id.origin, id.oseq).unwrap().unwrap();
        sent.push(decode_header(&bytes).unwrap().0.identity());
    }
    assert_eq!((sent[0].gen, sent[1].gen), (1, 2));
    assert!(sent[1].stamp() > sent[0].stamp());
}
