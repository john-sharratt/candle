//! **A leak is named**: buffers allocated and kept are counted in their size
//! class, the process's commit grows with them, and the verdict over the window
//! is an owner in that class.
//!
//! Its own test binary, so the counting allocator is this process's global
//! allocator exactly as it is in motherd's.

use std::hint::black_box;

use heap_census::reading::Reading;
use heap_census::verdict::{judge, Verdict};
use heap_census::Counting;

#[global_allocator]
static HEAP: Counting = Counting;

/// The buffers kept: about a replicated chunk record's size, the size BlackWorld
/// grows by per record applied.
const KEPT: usize = 2_000;
const SIZE: usize = 30_000;
const LEAKED: u64 = (KEPT * SIZE) as u64;

/// The threshold a 30-minute window is judged at in production is 2 GiB; this
/// leak is 60 MB, so the window here is judged at 32 MiB.
const THRESHOLD: u64 = 32 << 20;

#[test]
fn kept_buffers_are_named_as_an_owner_in_their_class() {
    let before = Reading::take(0).expect("this platform reads its process");
    // Filled, not merely allocated, so every page is touched and committed.
    let kept: Vec<Vec<u8>> = (0..KEPT).map(|i| vec![(i % 251) as u8; SIZE]).collect();
    let after = Reading::take(30 * 60 * 1000).unwrap();

    let class_15 = |r: &Reading| r.classes.expect("the allocator is installed")[15];
    let grew = class_15(&after) - class_15(&before);
    assert!(
        (LEAKED..LEAKED + 64 * 32_768).contains(&grew),
        "class 15 grew {grew} B for {LEAKED} B kept"
    );
    let private_grew = after.private_bytes as i64 - before.private_bytes as i64;
    assert!(
        private_grew >= LEAKED as i64 * 9 / 10,
        "the commit charge grew {private_grew} B for {LEAKED} B kept"
    );

    let g = judge(&before, &after, THRESHOLD);
    match &g.verdict {
        Verdict::Owner { classes, .. } => {
            assert_eq!(classes[0].class, 15, "{g:?}");
            assert!(classes[0].grew_bytes >= LEAKED as i64, "{g:?}");
        }
        other => panic!("a kept 60 MB was judged {other:?}: {g:?}"),
    }
    black_box(&kept);
}
