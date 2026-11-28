//! **A compaction keeps every chunk the substrate locates**, whatever order
//! the section's generations were declared in.
//!
//! A section's records carry the generation they were written under in their
//! identity's `epoch` — the `oseq` of the declaration that started it — and the
//! substrate admits a section record only of the declaration it holds or a
//! later one. The log holds a section's records in the order they were
//! written; a compaction writes the declaration it holds first and the chunks
//! after it. A section declared again after its chunks were written holds a
//! declaration newer than those chunks' generation, and the compaction's
//! re-apply must still locate them, or settling the rewrite drops their
//! locations and maintenance deletes their bytes.

use tempfile::TempDir;

use super::content_hash::{hash_bytes, ContentHash};
use super::record::ChunkPayload;
use super::streams::{ContentAddress, SectionDecl, StreamDecl, StreamId};
use super::{PersistenceError, SubstratePersistence};
use crate::substrate::Substrate;

fn section() -> StreamDecl {
    StreamDecl::PromptSection(SectionDecl {
        address: ContentAddress {
            prefix_hash: ContentHash { lo: 9, hi: 0 },
            section_hash: hash_bytes(b"a file's section"),
        },
        debug_name: "shared/files/candle/a.rs@b#1-10".into(),
    })
}

fn chunk(seed: u8) -> ChunkPayload {
    ChunkPayload {
        offset: 0,
        k_formats: vec![4],
        v_formats: vec![5],
        k_pal: vec![seed],
        v_pal: vec![seed],
        k_scale: vec![1.0],
        v_scale: vec![0.5],
        kv_bytes: (0..64u32).map(|i| (i as u8).wrapping_add(seed)).collect(),
    }
}

fn reopen(dir: &TempDir, sp: &mut SubstratePersistence) -> Substrate {
    sp.commit().unwrap();
    sp.release_writer().unwrap();
    let mut substrate = Substrate::new();
    *sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    substrate
}

fn located(substrate: &Substrate, sid: StreamId) -> usize {
    substrate.stream_of(sid).map_or(0, |s| s.chunks.len())
}

/// **A section written while the member is cut off, with no rejoin since,
/// survives a compaction.** Its records carry the provisional generation; the
/// open holds them back and applies them after every other record, and the
/// compaction's re-apply must locate them all the same.
#[test]
fn a_section_written_detached_survives_a_compaction() {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let attached = StreamDecl::PromptSection(SectionDecl {
        address: ContentAddress {
            prefix_hash: ContentHash { lo: 9, hi: 0 },
            section_hash: hash_bytes(b"written before the cut"),
        },
        debug_name: "shared/files/candle/before.rs@b#1-10".into(),
    });
    let before = sp.declare_stream(&attached).unwrap().value;
    sp.write_chunk(before, 0, 32, 4, None, &chunk(1)).unwrap();
    sp.detach();
    let sid = sp.declare_stream(&section()).unwrap().value;
    sp.write_chunk(sid, 0, 32, 4, None, &chunk(2)).unwrap();
    sp.write_chunk(sid, 1, 32, 4, None, &chunk(3)).unwrap();

    let mut substrate = reopen(&dir, &mut sp);
    assert!(sp.is_detached(), "no rejoin: still cut off");
    assert_eq!(located(&substrate, before), 1);
    assert_eq!(located(&substrate, sid), 2, "the log read in written order");

    sp.compact(&mut substrate, None).unwrap();
    assert_eq!(located(&substrate, before), 1);
    assert_eq!(located(&substrate, sid), 2, "re-applied after the rewrite");

    let substrate = reopen(&dir, &mut sp);
    assert_eq!(located(&substrate, before), 1);
    assert_eq!(located(&substrate, sid), 2, "the compacted log read back");
}

/// **A rewrite the substrate did not take back is never settled.** With the
/// re-apply withheld — the state the 2026-11-28 compaction left 6,221 streams
/// in — the stream is named untaken, the compaction refuses, the replaced
/// segment is kept as a sealed segment of the log, and the chunks are located
/// on the next open as they were before.
#[test]
fn a_rewrite_not_taken_back_keeps_the_replaced_segments() {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let sid = sp.declare_stream(&section()).unwrap().value;
    sp.write_chunk(sid, 0, 32, 4, None, &chunk(1)).unwrap();
    sp.write_chunk(sid, 1, 32, 4, None, &chunk(2)).unwrap();
    let substrate = reopen(&dir, &mut sp);

    let rewritten = sp.rewrite(&substrate, None, None).unwrap();
    let untaken = rewritten.untaken(&substrate);
    assert_eq!(untaken.len(), 1);
    assert_eq!(
        (
            untaken[0].key.stream_id,
            untaken[0].carried,
            untaken[0].taken
        ),
        (sid, 2, 0)
    );
    assert!(matches!(
        sp.refuse_untaken(&rewritten, &untaken),
        Err(PersistenceError::RewriteUntaken {
            streams: 1,
            chunks: 2
        })
    ));
    let on_disk = std::fs::read_dir(sp.dir()).unwrap().count();
    assert!(
        on_disk > rewritten.old_segments().len(),
        "the replaced segments are kept beside the compacted one"
    );
    assert_eq!(located(&substrate, sid), 2, "still located where it was");

    let substrate = reopen(&dir, &mut sp);
    assert_eq!(located(&substrate, sid), 2, "nothing lost on the next open");
}

/// **A rewrite the substrate took back whole names nothing untaken.**
#[test]
fn a_rewrite_taken_back_whole_names_nothing() {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let sid = sp.declare_stream(&section()).unwrap().value;
    sp.write_chunk(sid, 0, 32, 4, None, &chunk(1)).unwrap();
    let mut substrate = reopen(&dir, &mut sp);

    let rewritten = sp.rewrite(&substrate, None, None).unwrap();
    rewritten.reapply(&mut substrate);
    assert_eq!(rewritten.untaken(&substrate), Vec::new());
}

/// **A section's chunks survive a compaction**: located on the reopen, after
/// the re-apply and on the compacted log's open.
#[test]
fn a_sections_chunks_survive_a_compaction() {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let sid = sp.declare_stream(&section()).unwrap().value;
    sp.write_chunk(sid, 0, 32, 4, None, &chunk(1)).unwrap();
    sp.write_chunk(sid, 1, 32, 4, None, &chunk(2)).unwrap();

    let mut substrate = reopen(&dir, &mut sp);
    assert_eq!(located(&substrate, sid), 2, "the log read in written order");

    sp.compact(&mut substrate, None).unwrap();
    assert_eq!(located(&substrate, sid), 2, "re-applied after the rewrite");

    let substrate = reopen(&dir, &mut sp);
    assert_eq!(located(&substrate, sid), 2, "the compacted log read back");
}
