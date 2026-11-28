//! A cut keeps everything a store held before a run and nothing that run
//! wrote — against a real store, written, closed, cut and opened again.

use std::time::Duration;

use tempfile::TempDir;

use super::*;
use crate::persistence::content_hash::{hash_bytes, ContentHash};
use crate::persistence::identity::now_ms;
use crate::persistence::record::ChunkPayload;
use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl, StreamId};
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

fn member() -> StreamDecl {
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

fn open(dir: &TempDir) -> (SubstratePersistence, Substrate) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    (sp, substrate)
}

fn close(mut sp: SubstratePersistence) {
    sp.commit().unwrap();
    sp.release_writer().unwrap();
}

fn located(substrate: &Substrate, sid: StreamId) -> usize {
    substrate.stream_of(sid).map_or(0, |s| s.chunks.len())
}

/// Write a member's two chunks in one run; return the store's node, the
/// member's stream and the first wall-clock millisecond after that run.
fn first_run(dir: &TempDir) -> (u64, StreamId, u64) {
    let (mut sp, _) = open(dir);
    let sid = sp.declare_stream(&member()).unwrap().value;
    sp.write_chunk(sid, 0, 32, 4, None, &chunk(1)).unwrap();
    sp.write_chunk(sid, 1, 32, 4, None, &chunk(2)).unwrap();
    let node = sp.node();
    close(sp);
    std::thread::sleep(Duration::from_millis(5));
    let from = now_ms();
    std::thread::sleep(Duration::from_millis(5));
    (node, sid, from)
}

/// **A tombstone a later run wrote is cut, and the member stands again.**
/// The second run retires the member as the 2026-11-28 sweep did; the store
/// opened after it locates nothing of it, and the store cut back to before
/// that run locates both chunks.
#[test]
fn a_later_runs_tombstone_is_cut_and_the_member_stands_again() {
    let dir = TempDir::new().unwrap();
    let (node, sid, from) = first_run(&dir);

    let (mut sp, mut substrate) = open(&dir);
    assert_eq!(located(&substrate, sid), 2);
    let generation = substrate.tombstone_section(sid);
    sp.write_section_tombstone(sid.0, generation, Some("shared journal section superseded"))
        .unwrap();
    let store = sp.dir().to_path_buf();
    close(sp);
    let (sp, substrate) = open(&dir);
    assert_eq!(located(&substrate, sid), 0, "the tombstone stands");
    close(sp);

    let cut = plan(&store, node, from).unwrap().expect("a run to cut");
    assert!(cut.set_aside.is_empty(), "one segment, cut inside");
    assert!(cut.truncate.is_some());
    assert_eq!(cut.cut.get("SectionTombstone"), Some(&1));
    assert!(
        cut.last_kept_ms < from,
        "the kept records are the first run's"
    );
    apply(&cut).unwrap();

    let (sp, substrate) = open(&dir);
    assert_eq!(located(&substrate, sid), 2, "the member stands again");
    close(sp);
}

/// **A store with nothing that late is left as it is.**
#[test]
fn nothing_that_late_plans_no_cut() {
    let dir = TempDir::new().unwrap();
    let (node, _, _) = first_run(&dir);
    let store = dir.path().join(crate::persistence::SUBSTRATE_DIR);
    assert_eq!(plan(&store, node, now_ms() + 60_000).unwrap(), None);
}

/// **A segment set aside is no segment of the store.** Its name keeps the
/// segment's and adds the suffix, which the segment scan does not read.
#[test]
fn a_set_aside_segment_is_not_read_as_one() {
    let dir = TempDir::new().unwrap();
    let seg = dir.path().join("seg-0000000108.log");
    std::fs::write(&seg, b"").unwrap();
    let aside = set_aside_path(&seg);
    assert_eq!(aside, dir.path().join("seg-0000000108.log.cut"));
    std::fs::rename(&seg, &aside).unwrap();
    assert!(scan_segment_files(dir.path()).unwrap().is_empty());
}
