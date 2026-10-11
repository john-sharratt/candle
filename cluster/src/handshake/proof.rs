//! The handshake's two proofs (§3.3):
//!
//! ```text
//! proof_g = HMAC-SHA256(K, "tokera-cluster/1 gateway" ‖ app ‖ node ‖ nonce_m ‖ nonce_g)
//! proof_m = HMAC-SHA256(K, "tokera-cluster/1 member"  ‖ app ‖ node ‖ nonce_g ‖ nonce_m)
//! ```
//!
//! `app` is its UTF-8 bytes and `node` eight little-endian bytes. The labels
//! bind direction and protocol version, so a member's proof can never be played
//! back as the gateway's. Proofs are compared in constant time.

use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::key::GroupKey;

const GATEWAY_LABEL: &[u8] = b"tokera-cluster/1 gateway";
const MEMBER_LABEL: &[u8] = b"tokera-cluster/1 member";

type HmacSha256 = Hmac<Sha256>;

fn mac(
    key: &GroupKey,
    label: &[u8],
    app: &str,
    node: u64,
    first: &[u8; 32],
    second: &[u8; 32],
) -> HmacSha256 {
    let mut m = HmacSha256::new_from_slice(key.bytes()).expect("HMAC takes a key of any length");
    m.update(label);
    m.update(app.as_bytes());
    m.update(&node.to_le_bytes());
    m.update(first);
    m.update(second);
    m
}

/// The gateway's proof, over the member's nonce then its own.
pub fn gateway_proof(
    key: &GroupKey,
    app: &str,
    node: u64,
    nonce_m: &[u8; 32],
    nonce_g: &[u8; 32],
) -> [u8; 32] {
    mac(key, GATEWAY_LABEL, app, node, nonce_m, nonce_g)
        .finalize()
        .into_bytes()
        .into()
}

/// The member's proof, over the gateway's nonce then its own.
pub fn member_proof(
    key: &GroupKey,
    app: &str,
    node: u64,
    nonce_g: &[u8; 32],
    nonce_m: &[u8; 32],
) -> [u8; 32] {
    mac(key, MEMBER_LABEL, app, node, nonce_g, nonce_m)
        .finalize()
        .into_bytes()
        .into()
}

/// Whether `proof` is the gateway's, compared in constant time.
pub fn verify_gateway(
    key: &GroupKey,
    app: &str,
    node: u64,
    nonce_m: &[u8; 32],
    nonce_g: &[u8; 32],
    proof: &[u8; 32],
) -> bool {
    mac(key, GATEWAY_LABEL, app, node, nonce_m, nonce_g)
        .verify_slice(proof)
        .is_ok()
}

/// Whether `proof` is the member's, compared in constant time.
pub fn verify_member(
    key: &GroupKey,
    app: &str,
    node: u64,
    nonce_g: &[u8; 32],
    nonce_m: &[u8; 32],
    proof: &[u8; 32],
) -> bool {
    mac(key, MEMBER_LABEL, app, node, nonce_g, nonce_m)
        .verify_slice(proof)
        .is_ok()
}

/// 32 bytes from the OS-seeded generator.
pub fn nonce() -> [u8; 32] {
    let mut n = [0u8; 32];
    rand::fill(&mut n);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> GroupKey {
        GroupKey::from_bytes([0x42; 32])
    }

    const NM: [u8; 32] = [0x11; 32];
    const NG: [u8; 32] = [0x22; 32];

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// The proofs for a fixed key and fixed nonces are fixed bytes, computed
    /// independently of this crate with .NET's `HMACSHA256` over the
    /// documented message: key `0x42 × 32`, then the label, `zend`,
    /// `08 07 06 05 04 03 02 01`, and the two nonces in the proof's order.
    #[test]
    fn proofs_for_fixed_inputs_are_fixed_bytes() {
        let g = gateway_proof(&key(), "zend", 0x0102_0304_0506_0708, &NM, &NG);
        let m = member_proof(&key(), "zend", 0x0102_0304_0506_0708, &NG, &NM);
        assert_eq!(hex(&g), GATEWAY_PROOF);
        assert_eq!(hex(&m), MEMBER_PROOF);
        assert_ne!(g, m);
    }

    const GATEWAY_PROOF: &str = "59a3a0f07415f7644844d9d0eb063f0fae3c96718882a7d2f5cd27010a7c8caf";
    const MEMBER_PROOF: &str = "a088e23f8034926b086566819b47c0b146bc5e91e029c6bc5e6d35e68d97bbc9";

    /// A proof made for one direction never passes for the other, nor for
    /// another app, node, nonce or key.
    #[test]
    fn a_swapped_label_or_any_changed_input_fails() {
        let node = 9;
        let g = gateway_proof(&key(), "zend", node, &NM, &NG);
        let m = member_proof(&key(), "zend", node, &NG, &NM);
        assert!(verify_gateway(&key(), "zend", node, &NM, &NG, &g));
        assert!(verify_member(&key(), "zend", node, &NG, &NM, &m));
        assert!(!verify_gateway(&key(), "zend", node, &NM, &NG, &m));
        assert!(!verify_member(&key(), "zend", node, &NG, &NM, &g));
        assert!(!verify_gateway(&key(), "npcd", node, &NM, &NG, &g));
        assert!(!verify_gateway(&key(), "zend", node + 1, &NM, &NG, &g));
        assert!(!verify_gateway(&key(), "zend", node, &NG, &NM, &g));
        let other = GroupKey::from_bytes([0x43; 32]);
        assert!(!verify_gateway(&other, "zend", node, &NM, &NG, &g));
    }

    #[test]
    fn nonces_differ() {
        assert_ne!(nonce(), nonce());
    }
}
