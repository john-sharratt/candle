//! A tenant whose app has no quorum says so in `/v1/status`, for a console to
//! show its loading page.
//!
//! An app's members commit through a majority of the group's largest size:
//! `⌊max_members / 2⌋ + 1`, the ready members seen now counted when there are
//! more. In a group of two the voting gateway's tie-break is the second vote,
//! so one member and the gateway decide (`docs/replicated_substrate.md` §5.9).
//! Short of a quorum, a route that commits — opening a conversation, listing
//! them, a turn sealing — waits instead of failing, so a console reading the
//! tenant as ready opens onto a page whose every action hangs. One member of a
//! larger group left alone, or one of two whose gateway's tie-break has stopped
//! answering, is the case.
//!
//! **The tenant stays `ready`.** A member short of quorum is still the one
//! serving its app, and the guards on stopping a service read exactly that
//! (`keep::guard::served_alone`, the deploy's `only_here`): read as loading, the
//! member alone would be the one nothing stopped them taking down. So the wait
//! is its own field, `quorum_wait` — a load in [`LoadingView`]'s shape on the
//! [`QUORUM_STEP`] — which a console shows as its loading page.

use cluster::replicate::QuorumView;
use serde_json::to_value;

use crate::loading::LoadingView;
use crate::tenant::TenantStatus;

/// The step a tenant short of quorum waits on, by id.
pub const QUORUM_STEP: &str = "quorum";

/// The field of a tenant's line that carries the wait.
pub const QUORUM_WAIT: &str = "quorum_wait";

/// The step as the loading page prints it.
const LABEL: &str = "Waiting for the cluster";

/// The votes a group of `size` needs to commit.
pub fn needed(size: u32) -> u32 {
    size / 2 + 1
}

/// The votes `quorum`'s group has now: its ready members, and in a group of
/// two the voting gateway's tie-break while it answers.
pub fn votes(quorum: &QuorumView) -> u32 {
    let size = quorum.max_members.max(quorum.ready);
    quorum.ready + u32::from(quorum.voter && size == 2)
}

/// The wait of app `app`, whose group as this member counts it is `quorum`,
/// when it has fewer votes than it needs; `None` at quorum.
pub fn quorum_wait(app: &str, quorum: &QuorumView) -> Option<LoadingView> {
    let need = needed(quorum.max_members.max(quorum.ready));
    let have = votes(quorum);
    (have < need).then(|| LoadingView {
        current: QUORUM_STEP,
        label: LABEL,
        progress: have as f32 / need as f32,
        completed: Vec::new(),
        completed_labels: Vec::new(),
        progressed: u64::from(have),
        total: u64::from(need),
        unit: "votes".into(),
        detail: if quorum.max_members.max(quorum.ready) == 2 {
            format!(
                "{have} of the {need} votes {app} needs to commit: the gateway's tie-break \
                 is not answering — it serves again once the gateway answers or the other \
                 member joins"
            )
        } else {
            format!(
                "{have} of the {need} votes {app} needs to commit are ready members — it \
                 serves again once another member joins"
            )
        },
        elapsed_ms: 0,
        prefill_tokens: 0,
        prefill_tps: None,
        waiting_for: Vec::new(),
    })
}

/// `line`, the tenant `app`'s, with its [`QUORUM_WAIT`] when its app is short
/// of quorum and the tenant is otherwise ready. A tenant still loading shows
/// its own load; one with no cluster (`None`) has no quorum to wait for.
pub fn with_quorum_wait(
    mut line: TenantStatus,
    app: &str,
    quorum: Option<&QuorumView>,
) -> TenantStatus {
    if line.state != "ready" {
        return line;
    }
    if let Some(wait) = quorum.and_then(|q| quorum_wait(app, q)) {
        if let Ok(value) = to_value(wait) {
            line.extra.insert(QUORUM_WAIT.to_string(), value);
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::test_support::loading;
    use serde_json::json;

    fn group(max_members: u32, ready: u32, voter: bool) -> QuorumView {
        QuorumView {
            max_members,
            ready,
            voter,
            downgrade: None,
        }
    }

    #[test]
    fn the_quorum_is_a_majority_of_the_group() {
        assert_eq!((needed(1), needed(2), needed(3), needed(4)), (1, 2, 2, 3));
    }

    /// **The gateway's tie-break is the second vote of a group of two, and no
    /// other**: a quorum of two is also three members' majority, which one
    /// member and a gateway must not make.
    #[test]
    fn the_tie_break_counts_only_in_a_group_of_two() {
        assert_eq!(votes(&group(2, 1, true)), 2);
        assert_eq!(votes(&group(2, 1, false)), 1);
        assert_eq!(votes(&group(3, 1, true)), 1);
        assert_eq!(
            votes(&group(1, 2, true)),
            3,
            "two seen is a group of two, in which the gateway votes too"
        );
    }

    /// **One of two ready with no tie-break is short of quorum**: the tenant
    /// stays ready — it is the member serving — and its line carries the wait,
    /// saying how many votes it has and needs, and that the gateway is silent.
    #[test]
    fn a_tenant_short_of_quorum_carries_the_wait() {
        let line = with_quorum_wait(TenantStatus::of(None), "zend", Some(&group(2, 1, false)));
        assert_eq!(line.state, "ready");
        assert!(line.loading.is_none());
        let wait = &line.extra[QUORUM_WAIT];
        assert_eq!(wait["current"], json!(QUORUM_STEP));
        assert_eq!(wait["label"], json!(LABEL));
        assert_eq!(
            (&wait["progressed"], &wait["total"], &wait["unit"]),
            (&json!(1), &json!(2), &json!("votes"))
        );
        assert_eq!(
            wait["detail"],
            json!(
                "1 of the 2 votes zend needs to commit: the gateway's tie-break is not \
                 answering — it serves again once the gateway answers or the other member \
                 joins"
            )
        );
        let alone = with_quorum_wait(TenantStatus::of(None), "zend", Some(&group(3, 1, true)));
        assert_eq!(
            alone.extra[QUORUM_WAIT]["detail"],
            json!(
                "1 of the 2 votes zend needs to commit are ready members — it serves again \
                 once another member joins"
            )
        );
    }

    /// A group at quorum — one of two with the gateway's tie-break among them —
    /// a member alone in a group of one, and a tenant with no cluster carry no
    /// wait.
    #[test]
    fn a_tenant_at_quorum_carries_no_wait() {
        for q in [
            Some(group(2, 1, true)),
            Some(group(2, 2, false)),
            Some(group(3, 2, false)),
            Some(group(1, 1, false)),
            None,
        ] {
            let line = with_quorum_wait(TenantStatus::of(None), "zend", q.as_ref());
            assert!(!line.extra.contains_key(QUORUM_WAIT), "{q:?}");
        }
    }

    /// A tenant still loading shows its own load, not the wait.
    #[test]
    fn a_loading_tenant_shows_its_own_load() {
        let own = loading("model");
        let line = with_quorum_wait(
            TenantStatus::of(Some(own.clone())),
            "zend",
            Some(&group(2, 1, false)),
        );
        assert_eq!(line.loading, Some(own));
        assert!(!line.extra.contains_key(QUORUM_WAIT));
    }
}
