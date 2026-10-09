//! What operations are called: "Operation Iron Lantern".
//!
//! Two words from lists in the world's own register — towers, vaults, machines,
//! weather on a dead planet — stepped through by co-prime strides so neighbouring
//! operations do not share a word, and skipped past any name already in use.

const ADJECTIVES: [&str; 32] = [
    "Iron",
    "Silent",
    "Hollow",
    "Amber",
    "Cold",
    "Broken",
    "Quiet",
    "Ashen",
    "Long",
    "Pale",
    "Copper",
    "Buried",
    "Last",
    "Slow",
    "Grey",
    "Burning",
    "Patient",
    "Sealed",
    "Distant",
    "Bitter",
    "Steady",
    "Hidden",
    "Shattered",
    "Lean",
    "Narrow",
    "Waking",
    "Rusted",
    "Bright",
    "Deep",
    "Second",
    "Wintered",
    "Open",
];

const NOUNS: [&str; 31] = [
    "Lantern",
    "Ledger",
    "Bearing",
    "Harbour",
    "Signal",
    "Tower",
    "Furnace",
    "Archive",
    "Meridian",
    "Shutter",
    "Vault",
    "Compass",
    "Anchor",
    "Relay",
    "Cinder",
    "Gate",
    "Tally",
    "Bulwark",
    "Lattice",
    "Ember",
    "Sentinel",
    "Chorus",
    "Keystone",
    "Threshold",
    "Watch",
    "Breaker",
    "Census",
    "Orchard",
    "Spindle",
    "Beacon",
    "Hinge",
];

/// The name of operation `id`, avoiding every name `taken` says is in use.
pub fn name_for(id: u64, taken: &dyn Fn(&str) -> bool) -> String {
    let all = (ADJECTIVES.len() * NOUNS.len()) as u64;
    let start = id.saturating_sub(1);
    for k in 0..all {
        let i = (start + k) % all;
        // The adjective steps by 7 (co-prime with 32) and the noun by 1 through
        // 31; with 32 and 31 co-prime, the pair walks every combination before
        // it repeats.
        let adjective = ADJECTIVES[((i * 7) % ADJECTIVES.len() as u64) as usize];
        let noun = NOUNS[(i % NOUNS.len() as u64) as usize];
        let name = format!("Operation {adjective} {noun}");
        if !taken(&name) {
            return name;
        }
    }
    format!("Operation {id}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn the_first_names_are_fixed_and_neighbours_share_no_word() {
        let none = |_: &str| false;
        assert_eq!(name_for(1, &none), "Operation Iron Lantern");
        assert_eq!(name_for(2, &none), "Operation Ashen Ledger");
        assert_eq!(name_for(3, &none), "Operation Grey Bearing");
    }

    #[test]
    fn a_taken_name_is_skipped_and_every_pair_is_reachable() {
        let first = name_for(1, &|_| false);
        assert_ne!(name_for(1, &|n| n == first), first);
        let mut seen = HashSet::new();
        for id in 1..=(32 * 31) {
            assert!(seen.insert(name_for(id, &|_| false)), "repeated at {id}");
        }
    }
}
