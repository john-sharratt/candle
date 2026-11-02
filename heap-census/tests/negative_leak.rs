//! **Churn is not a leak**: the same buffers allocated and freed over and over
//! leave nothing live, the commit charge does not grow with the rounds, and the
//! verdict is never an owner — however much passed through.
//!
//! Its own test binary, so the counting allocator is this process's global
//! allocator exactly as it is in motherd's.

use std::hint::black_box;

use heap_census::reading::Reading;
use heap_census::verdict::{judge, Verdict};
use heap_census::Counting;

#[global_allocator]
static HEAP: Counting = Counting;

const ROUNDS: usize = 20;
const PER_ROUND: usize = 2_000;
const SIZE: usize = 30_000;
const ROUND_BYTES: u64 = (PER_ROUND * SIZE) as u64;

const THRESHOLD: u64 = 32 << 20;

#[test]
fn churned_buffers_leave_nothing_behind() {
    let before = Reading::take(0).expect("this platform reads its process");
    for round in 0..ROUNDS {
        let held: Vec<Vec<u8>> = (0..PER_ROUND)
            .map(|i| vec![((i + round) % 251) as u8; SIZE])
            .collect();
        black_box(&held);
    }
    let after = Reading::take(30 * 60 * 1000).unwrap();

    let live = |r: &Reading| r.live_bytes().expect("the allocator is installed") as i64;
    let live_grew = live(&after) - live(&before);
    assert!(
        live_grew.abs() < 1 << 20,
        "{live_grew} B live after {ROUNDS} rounds freed in full"
    );
    let churned = after.allocated_total.unwrap() - before.allocated_total.unwrap();
    assert!(
        churned >= ROUNDS as u64 * ROUND_BYTES,
        "{churned} B allocated across {ROUNDS} rounds of {ROUND_BYTES} B"
    );
    // A heap may keep one round's worth committed for the next; twenty rounds
    // of a leak would be twenty.
    let private_grew = after.private_bytes as i64 - before.private_bytes as i64;
    assert!(
        private_grew < 2 * ROUND_BYTES as i64,
        "the commit charge grew {private_grew} B over {ROUNDS} rounds of {ROUND_BYTES} B freed"
    );

    let g = judge(&before, &after, THRESHOLD);
    assert!(
        !matches!(g.verdict, Verdict::Owner { .. }),
        "churn was judged an owner: {g:?}"
    );
}
