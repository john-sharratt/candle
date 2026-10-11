//! The engine's maps: `ahash` with fixed keys rather than per-process random
//! ones, so iteration order — which rounds a tick retries first — is the same
//! on every run and a seeded simulation replays exactly. The keys are Paxos
//! keys from admitted members, not attacker-chosen input.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use ahash::{AHasher, RandomState};

/// Builds `ahash` hashers with the same keys in every process.
#[derive(Clone, Copy, Default)]
pub struct Fixed;

impl BuildHasher for Fixed {
    type Hasher = AHasher;

    fn build_hasher(&self) -> AHasher {
        RandomState::with_seeds(
            0x243f_6a88_85a3_08d3,
            0x1319_8a2e_0370_7344,
            0xa409_3822_299f_31d0,
            0x082e_fa98_ec4e_6c89,
        )
        .build_hasher()
    }
}

pub type Map<K, V> = HashMap<K, V, Fixed>;
pub type Set<T> = HashSet<T, Fixed>;
