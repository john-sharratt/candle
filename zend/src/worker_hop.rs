//! Which member a worker's turn moves to, and how the member it moves to
//! knows a member sent it (`docs/claude_zend_gateway.md` §6.3).
//!
//! A worker's turn runs on the member that took the call starting it, and the
//! call is placed by round robin. So a worker started while the fast card was
//! busy or down ran every round on the slow one, at a tenth of the speed of
//! the card beside it.
//!
//! So at every tool-round boundary — its call turn sealed and committed, its
//! tools not yet run — a worker's turn asks whether a markedly faster member is
//! ready. If one is, the turn hands itself to that member, which runs the
//! round's tools and carries the worker on (`workers::continue_handed`). Moving
//! off a card that went away needs nothing: the next call lands elsewhere.
//!
//! **Markedly faster** is [`HOP_RATIO`] times this member's weight, the peak
//! decode rate each member is admitted at (`MEMBERS`). That keeps two cards of
//! one class from trading a worker back and forth, and a card hands a worker
//! to one of its own class never.
//!
//! **A handing is signed.** The gateway stamps a request on a member's stream
//! as that member's, but the stream also carries what clients sent the
//! member's own port. So the handing member signs the request it sends with a
//! key every member derives from the group key ([`PURPOSE`]), with the time it
//! signed; the member taking it believes a request that verifies, signed in
//! the last [`FRESH_MS`].

use std::cmp::Reverse;
use std::sync::{Arc, OnceLock};

use cluster::frame::{MemberInfo, MemberState};
use cluster::member::AppMember;
use hmac::{Hmac, Mac};
use sha2::Sha256;

/// How much faster a peer must be, by weight, for a worker to move to it.
pub const HOP_RATIO: u32 = 2;

/// What the handing key is derived for (`GroupKey::derive`).
pub const PURPOSE: &str = "zend worker handoff";

/// The header a handing's signature travels in: the body's HMAC under the
/// handing key, in hex.
pub const PROOF_HEADER: &str = "x-zend-handoff-proof";

/// How old a signed handing may be when it arrives.
pub const FRESH_MS: u64 = 120_000;

type HmacSha256 = Hmac<Sha256>;

struct Hop {
    member: Arc<AppMember>,
    key: [u8; 32],
}

static HOP: OnceLock<Hop> = OnceLock::new();

/// Let workers' turns weigh this member's peers, and sign and verify handings
/// with `key`, derived for [`PURPOSE`].
pub fn install(member: Arc<AppMember>, key: [u8; 32]) {
    let _ = HOP.set(Hop { member, key });
}

/// This member, when it is in a cluster.
pub fn member() -> Option<Arc<AppMember>> {
    HOP.get().map(|h| Arc::clone(&h.member))
}

/// This member's node, when it is in a cluster.
pub fn own_node() -> Option<u64> {
    HOP.get().map(|h| h.member.node())
}

/// The ready member of `group` at least [`HOP_RATIO`] times `me`'s weight —
/// the fastest, when several are. `None` when `me` is not listed or has no
/// weight yet, or nobody qualifies.
pub fn faster_peer(group: &[MemberInfo], me: u64) -> Option<u64> {
    let mine = group.iter().find(|m| m.node == me)?.weight;
    if mine == 0 {
        return None;
    }
    group
        .iter()
        .filter(|m| m.node != me && m.state == MemberState::Ready)
        .filter(|m| u64::from(m.weight) >= u64::from(mine) * u64::from(HOP_RATIO))
        .max_by_key(|m| (m.weight, Reverse(m.node)))
        .map(|m| m.node)
}

/// The member a worker's turn on this one should move to now, if any.
pub fn target() -> Option<u64> {
    let hop = HOP.get()?;
    faster_peer(&hop.member.group(), hop.member.node())
}

/// Whether a worker whose record is `running` with `runner`, read on member
/// `me`, is another member's to run: then this one neither beats it nor
/// finishes its round.
pub fn runs_elsewhere(running: bool, runner: Option<u64>, me: Option<u64>) -> bool {
    running && runner.is_some() && runner != me
}

fn mac(key: &[u8; 32], body: &[u8]) -> HmacSha256 {
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC takes a key of any length");
    m.update(body);
    m
}

/// `body`'s signature under `key`, in hex.
pub fn sign_with(key: &[u8; 32], body: &[u8]) -> String {
    mac(key, body)
        .finalize()
        .into_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Whether `proof` is `body`'s signature under `key`, compared in constant
/// time.
pub fn verify_with(key: &[u8; 32], body: &[u8], proof: &str) -> bool {
    let Some(bytes) = unhex(proof) else {
        return false;
    };
    mac(key, body).verify_slice(&bytes).is_ok()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

/// `body`'s signature under this member's handing key; `None` outside a
/// cluster.
pub fn sign(body: &[u8]) -> Option<String> {
    HOP.get().map(|h| sign_with(&h.key, body))
}

/// Whether `proof` is `body`'s signature under this member's handing key.
pub fn verify(body: &[u8], proof: &str) -> bool {
    HOP.get().is_some_and(|h| verify_with(&h.key, body, proof))
}

/// Whether a handing signed at `signed_ms` is fresh at `now_ms`.
pub fn fresh(signed_ms: u64, now_ms: u64) -> bool {
    now_ms.abs_diff(signed_ms) <= FRESH_MS
}

#[cfg(test)]
mod tests {
    use cluster::frame::BuildId;

    use super::*;

    fn member(node: u64, weight: u32, state: MemberState) -> MemberInfo {
        MemberInfo {
            node,
            host: format!("m{node}"),
            app: "zend".into(),
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            weight,
            state,
            high_water: 0,
        }
    }

    /// **A worker moves only to a ready member markedly faster than this
    /// one**, the fastest of them; never to one of its own class, a loading
    /// one, or anywhere when this member is unweighed.
    #[test]
    fn a_worker_moves_only_to_a_markedly_faster_ready_member() {
        let ww = member(2, 15, MemberState::Ready);
        let bw = member(1, 100, MemberState::Ready);
        assert_eq!(faster_peer(&[bw.clone(), ww.clone()], 2), Some(1));
        assert_eq!(
            faster_peer(&[bw.clone(), ww.clone()], 1),
            None,
            "never back down"
        );
        let twin = member(3, 25, MemberState::Ready);
        assert_eq!(
            faster_peer(&[ww.clone(), twin.clone()], 2),
            None,
            "under twice"
        );
        let loading = member(1, 100, MemberState::Loading);
        assert_eq!(faster_peer(&[loading, ww.clone()], 2), None);
        let fastest = member(4, 300, MemberState::Ready);
        assert_eq!(faster_peer(&[bw.clone(), fastest, ww.clone()], 2), Some(4));
        let unweighed = member(2, 0, MemberState::Ready);
        assert_eq!(faster_peer(&[bw.clone(), unweighed], 2), None);
        assert_eq!(faster_peer(&[bw], 9), None, "not listed");
    }

    /// **A worker is another member's only while it runs there**: one this
    /// member runs, one not running, and one run outside any cluster are this
    /// member's to finish.
    #[test]
    fn a_worker_runs_elsewhere_only_while_another_member_runs_it() {
        assert!(runs_elsewhere(true, Some(1), Some(2)));
        assert!(!runs_elsewhere(true, Some(2), Some(2)));
        assert!(!runs_elsewhere(false, Some(1), Some(2)));
        assert!(!runs_elsewhere(true, None, Some(2)));
    }

    /// **A handing verifies only with its own signature under the members'
    /// key**: another body, another key, or a mangled proof does not, and it
    /// must be fresh.
    #[test]
    fn a_handing_verifies_only_as_signed_by_a_member() {
        let key = [7u8; 32];
        let body = br#"{"owner":"o","call_turn":3}"#;
        let proof = sign_with(&key, body);
        assert_eq!(proof.len(), 64);
        assert!(verify_with(&key, body, &proof));
        assert!(!verify_with(
            &key,
            br#"{"owner":"o","call_turn":4}"#,
            &proof
        ));
        assert!(!verify_with(&[8u8; 32], body, &proof));
        assert!(!verify_with(&key, body, &proof[..62]));
        assert!(!verify_with(&key, body, "zz"));
        assert!(fresh(1_000_000, 1_000_000 + FRESH_MS));
        assert!(!fresh(1_000_000, 1_000_000 + FRESH_MS + 1));
        assert!(
            fresh(1_000_000 + 5, 1_000_000),
            "a peer's clock a little ahead"
        );
    }
}
