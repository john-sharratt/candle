//! **A leak is named by the scope that made it**: blocks kept under one scope
//! are counted there to the byte, blocks made and freed under another leave it
//! at nothing, and a block freed on another thread is still taken off the scope
//! that made it.
//!
//! Its own test binary, so the counting allocator is this process's global
//! allocator exactly as it is in motherd's. Each test uses scopes no other test
//! here does, so they can run side by side.

use std::hint::black_box;
use std::mem::size_of;
use std::thread;

use heap_census::counting::CENSUS;
use heap_census::{enter, Counting, Scope};

#[global_allocator]
static HEAP: Counting = Counting;

/// About the block BlackWorld's growth is made of.
const SIZE: usize = 100_000;
const KEPT: usize = 500;

fn scope_bytes(s: Scope) -> u64 {
    CENSUS.counts().scope_bytes(s as usize)
}

/// **Kept under one scope: counted there exactly** — every buffer and the
/// vector holding them, nothing else — **and churn under another leaves it at
/// zero.**
#[test]
fn a_kept_leak_is_named_by_its_scope_and_churn_is_not() {
    let (kept_before, churn_before) = (scope_bytes(Scope::RemoteApply), scope_bytes(Scope::Adoption));
    let kept: Vec<Vec<u8>> = {
        let _s = enter(Scope::RemoteApply);
        (0..KEPT).map(|i| vec![(i % 251) as u8; SIZE]).collect()
    };
    {
        let _s = enter(Scope::Adoption);
        for round in 0..20 {
            let churned: Vec<Vec<u8>> = (0..KEPT).map(|i| vec![((i + round) % 251) as u8; SIZE]).collect();
            black_box(&churned);
        }
    }
    let outer = (KEPT * size_of::<Vec<u8>>()) as u64;
    assert_eq!(
        scope_bytes(Scope::RemoteApply) - kept_before,
        (KEPT * SIZE) as u64 + outer
    );
    assert_eq!(scope_bytes(Scope::Adoption), churn_before);
    drop(kept);
    assert_eq!(scope_bytes(Scope::RemoteApply), kept_before);
}

/// **A block freed on another thread is taken off the scope that made it**,
/// though that thread is in no scope at all.
#[test]
fn a_free_on_another_thread_leaves_the_scope_it_was_made_in() {
    let before = scope_bytes(Scope::Summariser);
    let made: Vec<Vec<u8>> = {
        let _s = enter(Scope::Summariser);
        (0..KEPT).map(|_| vec![7u8; SIZE]).collect()
    };
    assert!(scope_bytes(Scope::Summariser) > before);
    thread::spawn(move || drop(made)).join().unwrap();
    assert_eq!(scope_bytes(Scope::Summariser), before);
}

/// **A block grown across the labelled size and back is taken off its scope
/// in full**: nothing is left behind by the header coming and going.
#[test]
fn a_block_grown_and_shrunk_across_the_labelled_size_balances() {
    let before = scope_bytes(Scope::Journals);
    {
        let _s = enter(Scope::Journals);
        let mut v: Vec<u8> = Vec::with_capacity(16);
        for i in 0..1_000_000u32 {
            v.push(i as u8);
        }
        assert!(scope_bytes(Scope::Journals) >= before + v.capacity() as u64);
        v.truncate(100);
        v.shrink_to_fit();
        assert_eq!(scope_bytes(Scope::Journals), before, "100 bytes carry no scope");
        black_box(&v);
    }
    assert_eq!(scope_bytes(Scope::Journals), before);
}
