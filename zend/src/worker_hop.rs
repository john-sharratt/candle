//! Which member a worker's turn moves to (`docs/claude_zend_gateway.md` §6.2).
//!
//! A worker's turn runs on the member that took the call starting it, and the
//! call is placed by round robin. So a worker started while the fast card was
//! busy, or down, ran every round on the slow one: on 2026-10-31 a cleanup
//! worker ran 7–16 tokens a second on WhiteWorld's 3090 for 23 minutes while
//! BlackWorld, at 42–108, stood idle beside it.
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

use std::sync::{Arc, OnceLock};

use cluster::frame::{MemberInfo, MemberState};
use cluster::member::AppMember;

/// How much faster a peer must be, by weight, for a worker to move to it.
pub const HOP_RATIO: u32 = 2;

static MEMBER: OnceLock<Arc<AppMember>> = OnceLock::new();

/// Let workers' turns weigh this member's peers.
pub fn install(member: Arc<AppMember>) {
    let _ = MEMBER.set(member);
}

/// This member, when it is in a cluster.
pub fn member() -> Option<Arc<AppMember>> {
    MEMBER.get().cloned()
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
        .max_by_key(|m| (m.weight, std::cmp::Reverse(m.node)))
        .map(|m| m.node)
}

/// The member a worker's turn on this one should move to now, if any.
pub fn target() -> Option<u64> {
    let member = MEMBER.get()?;
    faster_peer(&member.group(), member.node())
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
}
