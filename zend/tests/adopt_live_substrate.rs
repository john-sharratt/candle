//! **A copy of the live substrate keeps its state through a writable open**
//! (`docs/replicated_substrate.md` §15, chunk 1's exit): the store this daemon
//! serves reopens to the same conversations, the same labels and the same turns
//! at the same indices, with every record an origin writes stamped as this
//! node's own. A store written before records carried an identity is adopted by
//! that open; the live one was, at the daemon's first start on this build
//! (2026-10-11), and this test ran the adoption on a copy of it first.
//!
//! The live store is the one this checkout's daemon serves: its workspace is
//! the directory that holds the checkout. The copy is taken while the daemon
//! runs — every segment is append-only and the open recovers a torn tail —
//! into `target/tmp`, and removed at the end.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use candle_conversation::persistence::record::RecordType;
use candle_conversation::persistence::resume::{derive_layer_count, recovered_turn_decls};
use candle_conversation::persistence::writer_lock::LOCK_FILE;
use candle_conversation::persistence::{Admit, SubstratePersistence};
use candle_conversation::projection::{Conversation, TurnIndex};
use candle_conversation::substrate::Substrate;

/// The daemon's workspace: the directory that holds this checkout.
fn live_workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the checkout sits in a workspace")
        .to_path_buf()
}

/// Copy `from` to `to`, all but the writer lock, which the running daemon
/// holds and which a copy has no use for.
fn copy_store(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() == LOCK_FILE {
            continue;
        }
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_store(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// One live conversation as a reload holds it: its id, conv_id, label,
/// archived flag, and a hash of each turn's tokens at its index. A deleted
/// conversation is dead at read whether or not a rewrite has dropped its
/// records yet, so it is not one.
#[derive(Debug, PartialEq, Eq)]
struct Held {
    timeline: u64,
    conv_id: Option<String>,
    label: Option<String>,
    archived: bool,
    turns: Vec<u64>,
}

/// Every conversation the store at `ws` reloads to — opened read-only, which
/// adopts nothing, or writable, which adopts a store that needs it.
fn reload(ws: &Path, writable: bool) -> Vec<Held> {
    let mut substrate = Substrate::new();
    let sp = if writable {
        SubstratePersistence::open_in_with_substrate(ws, &mut substrate).unwrap()
    } else {
        SubstratePersistence::open_in_with_substrate_read_only(ws, &mut substrate).unwrap()
    };
    let decls = recovered_turn_decls(&substrate);
    let n_layers = derive_layer_count(&substrate, &decls).unwrap_or(1);
    let conv = Conversation::from_parts(substrate, sp);
    conv.reconstruct_from_log(n_layers, None).unwrap();
    let read = conv.read();
    let mut held: Vec<Held> = conv
        .timeline_ids()
        .into_iter()
        .filter(|tl| !read.is_timeline_dead(tl.raw()))
        .map(|tl| {
            let entry = read.timeline_entry(tl).unwrap();
            let turns = (0..read.turn_count(tl))
                .map(|i| {
                    let mut h = DefaultHasher::new();
                    read.token_ids_of(tl, TurnIndex(i)).hash(&mut h);
                    h.finish()
                })
                .collect();
            Held {
                timeline: tl.raw(),
                conv_id: entry.conv_id.clone(),
                label: entry.label.clone(),
                archived: entry.archived,
                turns,
            }
        })
        .collect();
    held.sort_by_key(|h| h.timeline);
    held
}

#[test]
#[ignore = "copies the live substrate — 160 GB on the dev box — and adopts the copy: run by \
            name, with free disk for two copies"]
fn a_copy_of_the_live_substrate_adopts_and_keeps_its_state() {
    let ws = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("live_adoption");
    if ws.exists() {
        fs::remove_dir_all(&ws).unwrap();
    }
    copy_store(&live_workspace().join("substrate"), &ws.join("substrate"));

    let before = reload(&ws, false);
    let after = reload(&ws, true);
    assert!(!before.is_empty(), "the live store holds conversations");
    let ids = |held: &[Held]| held.iter().map(|h| h.timeline).collect::<Vec<_>>();
    let lost: Vec<&Held> = before
        .iter()
        .filter(|b| !ids(&after).contains(&b.timeline))
        .collect();
    let gained: Vec<&Held> = after
        .iter()
        .filter(|a| !ids(&before).contains(&a.timeline))
        .collect();
    assert!(
        lost.is_empty() && gained.is_empty(),
        "conversations lost by the adoption: {lost:#?}\ngained: {gained:#?}"
    );
    for (b, a) in before.iter().zip(&after) {
        assert_eq!(b, a, "timeline {:x}", b.timeline);
    }

    // Adopted: every record an origin writes carries an identity, and a
    // further open adopts nothing.
    let mut substrate = Substrate::new();
    let mut unstamped = Vec::new();
    let mut sp = SubstratePersistence::open_in_with_substrate_and_sink(
        &ws,
        &mut substrate,
        Admit::Unbound,
        |e| {
            let h = &e.record.header;
            let housekeeping = matches!(
                h.record_type,
                RecordType::HeaderIndex
                    | RecordType::ModelKey
                    | RecordType::Template
                    | RecordType::Tokenizer
                    | RecordType::BranchCheckpoint
                    | RecordType::Replica
                    | RecordType::Deleted
                    | RecordType::Rebase
                    | RecordType::Tombstone
            );
            if h.identity().is_none() && !housekeeping {
                unstamped.push(h.record_type);
            }
        },
        &|_, _, _| {},
    )
    .unwrap();
    assert!(
        unstamped.is_empty(),
        "unstamped after adoption: {unstamped:?}"
    );
    // Every conversation's file events read back in the named form.
    for held in &after {
        sp.vfs_events(held.timeline).unwrap();
    }
    drop(sp);
    fs::remove_dir_all(&ws).unwrap();
}
