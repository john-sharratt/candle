//! Weighted rendezvous (§9.5): who proposes for a lease, and where an
//! authority lends a character.
//!
//! ```text
//! owner(k) = argmax over members m of   -w_m / ln(u(k, m))
//!            u(k, m) = (SHA-256(k ‖ m.node) as u64 + 1) / 2^64     ∈ (0, 1]
//! ```
//!
//! `m.node` is its eight little-endian bytes and "as u64" the digest's first
//! eight bytes, little-endian. Every member computes the owner itself; a member
//! joining or leaving moves only the keys whose winner changed; each member's
//! share of the keys tends to its share of the weight.

use sha2::{Digest, Sha256};

/// A member as rendezvous weighs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weighted {
    pub node: u64,
    pub weight: f64,
}

/// `-w / ln(u(k, node))`.
pub fn score(key: &[u8], node: u64, weight: f64) -> f64 {
    let mut h = Sha256::new();
    h.update(key);
    h.update(node.to_le_bytes());
    let digest = h.finalize();
    let x = u64::from_le_bytes(digest[..8].try_into().unwrap());
    let u = (x as f64 + 1.0) / 18_446_744_073_709_551_616.0;
    -weight / u.ln()
}

/// The member that owns `key`, or `None` with no members. Ties — vanishingly
/// rare — go to the lower node, so every member agrees.
///
/// **A member of weight 0 owns nothing**, even alone: it was admitted saying
/// rendezvous may not name it (`Identity::leaseholder`), and a scan that let it
/// win by default would hand it a world the moment every other member is away.
pub fn owner(key: &[u8], members: &[Weighted]) -> Option<u64> {
    members
        .iter()
        .filter(|m| m.weight > 0.0)
        .map(|m| (score(key, m.node, m.weight), m.node))
        .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
        .map(|(_, node)| node)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn members(weights: &[(u64, f64)]) -> Vec<Weighted> {
        weights
            .iter()
            .map(|(node, weight)| Weighted {
                node: *node,
                weight: *weight,
            })
            .collect()
    }

    /// A fixed key and node score a fixed value: the digest of `"w1"` then
    /// node 7's eight bytes, through the documented formula.
    #[test]
    fn a_fixed_key_scores_a_fixed_value() {
        let mut h = Sha256::new();
        h.update(b"w1");
        h.update(7u64.to_le_bytes());
        let d = h.finalize();
        let x = u64::from_le_bytes(d[..8].try_into().unwrap());
        let want = -2.0 / ((x as f64 + 1.0) / 2f64.powi(64)).ln();
        assert_eq!(score(b"w1", 7, 2.0), want);
        assert!(want > 0.0);
    }

    /// Ownership is deterministic, and a join or a leave moves only the keys
    /// whose winner changed.
    #[test]
    fn only_keys_whose_winner_changed_move() {
        let before = members(&[(1, 1.0), (2, 1.0), (3, 1.0)]);
        let joined = members(&[(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)]);
        let left = members(&[(1, 1.0), (3, 1.0)]);
        for i in 0..2000u32 {
            let k = i.to_le_bytes();
            let was = owner(&k, &before).unwrap();
            assert_eq!(owner(&k, &before), Some(was));
            let now = owner(&k, &joined).unwrap();
            assert!(
                now == was || now == 4,
                "a join moved {i} from {was} to {now}"
            );
            let after = owner(&k, &left).unwrap();
            assert!(after == was || was == 2, "a leave moved {i} from {was}");
        }
        assert_eq!(owner(b"k", &[]), None);
    }

    /// **A member of weight 0 owns no key**, beside others or alone.
    #[test]
    fn a_member_of_weight_zero_owns_nothing() {
        let ms = members(&[(1, 1.0), (2, 0.0)]);
        for i in 0..2000u32 {
            assert_eq!(owner(&i.to_le_bytes(), &ms), Some(1));
        }
        assert_eq!(owner(b"k", &members(&[(2, 0.0)])), None);
    }

    /// Shares tend to the weights.
    #[test]
    fn shares_follow_the_weights() {
        let ms = members(&[(1, 1.0), (2, 2.0), (3, 5.0)]);
        let mut count: HashMap<u64, u32> = HashMap::new();
        let n = 40_000u32;
        for i in 0..n {
            *count
                .entry(owner(&i.to_le_bytes(), &ms).unwrap())
                .or_default() += 1;
        }
        for (node, share) in [(1u64, 1.0 / 8.0), (2, 2.0 / 8.0), (3, 5.0 / 8.0)] {
            let got = count[&node] as f64 / n as f64;
            assert!((got - share).abs() < 0.02, "node {node}: {got} vs {share}");
        }
    }
}
