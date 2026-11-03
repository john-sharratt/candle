//! The calling thread's census state — its shard and its scope — in one
//! `const`-initialised thread-local word, so an allocation reads both with one
//! TLS access. Two thread-locals cost a labelled allocation two; measured, that
//! and the layout checks were most of a 1.17× allocator.
//!
//! The word holds the scope in its low byte and the shard plus one above it,
//! zero meaning no shard chosen yet. No destructor, so reading it allocates
//! nothing and it is never torn down while its thread runs.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use super::shards::SHARDS;

/// The next shard a thread is given.
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

const SCOPE_MASK: u32 = 0xFF;
const SHARD_SHIFT: u32 = 8;

thread_local! {
    static STATE: Cell<u32> = const { Cell::new(0) };
}

/// The calling thread's shard and scope, choosing the shard on its first
/// call. A thread past its thread-locals' lifetime counts on shard 0,
/// unscoped.
#[inline(always)]
pub(crate) fn shard_and_scope() -> (usize, u8) {
    STATE
        .try_with(|s| {
            let word = s.get();
            let scope = (word & SCOPE_MASK) as u8;
            let held = word >> SHARD_SHIFT;
            if held != 0 {
                return ((held - 1) as usize, scope);
            }
            let given = NEXT_SHARD.fetch_add(1, Relaxed) % SHARDS;
            s.set(((given as u32 + 1) << SHARD_SHIFT) | scope as u32);
            (given, scope)
        })
        .unwrap_or((0, 0))
}

/// Set the calling thread's scope, keeping its shard; the scope it had.
#[inline(always)]
pub(crate) fn replace_scope(scope: u8) -> u8 {
    STATE
        .try_with(|s| {
            let word = s.get();
            s.set((word & !SCOPE_MASK) | scope as u32);
            (word & SCOPE_MASK) as u8
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    /// **A thread's shard is chosen once and kept across scope changes**, and
    /// its scope is its own.
    #[test]
    fn the_shard_survives_a_scope_change() {
        thread::spawn(|| {
            let (shard, scope) = shard_and_scope();
            assert_eq!(scope, 0);
            assert_eq!(replace_scope(7), 0);
            assert_eq!(shard_and_scope(), (shard, 7));
            assert_eq!(replace_scope(3), 7);
            assert_eq!(shard_and_scope(), (shard, 3));
        })
        .join()
        .unwrap();
    }

    /// **A scope set before the first allocation keeps the shard unchosen**
    /// until that allocation chooses one.
    #[test]
    fn a_scope_set_first_leaves_the_shard_to_be_chosen() {
        thread::spawn(|| {
            assert_eq!(replace_scope(9), 0);
            let (shard, scope) = shard_and_scope();
            assert_eq!(scope, 9);
            assert!(shard < SHARDS);
            assert_eq!(shard_and_scope(), (shard, 9));
        })
        .join()
        .unwrap();
    }
}
