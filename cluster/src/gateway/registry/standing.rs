//! Which of an app's members a request may be routed to, and why each other
//! one is passed over (§4.4), as one line for the placement log. The hub logs
//! it when it changes, so a member that drops out of the routing — lagging,
//! syncing, silent, closing, not ready — is named with its reasons at the
//! moment it does, not inferred later from where the requests went.

use super::{MemberView, Registry};
use crate::frame::MemberState;

/// Why `m` is routed nothing: empty while it is routable and reporting.
pub fn passed_over(m: &MemberView) -> Vec<&'static str> {
    let mut why = Vec::new();
    match m.state() {
        MemberState::Ready => {}
        MemberState::Loading => why.push("loading"),
        MemberState::Upgrading => why.push("upgrading"),
        MemberState::Fault => why.push("fault"),
        MemberState::Draining => why.push("draining"),
        MemberState::Disabled => why.push("disabled"),
    }
    if m.lagging {
        why.push("lagging");
    }
    if m.syncing {
        why.push("syncing");
    }
    if m.closing {
        why.push("closing");
    }
    if m.silent {
        why.push("silent");
    }
    why
}

/// Each member's standing, in node order: `host node: routable`, or
/// `host node: passed over (reasons)`.
pub fn standing<'a>(members: impl Iterator<Item = &'a MemberView>) -> String {
    members
        .map(|m| {
            let why = passed_over(m);
            if why.is_empty() {
                format!("{} {:016x}: routable", m.host, m.node)
            } else {
                format!(
                    "{} {:016x}: passed over ({})",
                    m.host,
                    m.node,
                    why.join(", ")
                )
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

impl Registry {
    /// `app`'s standing ([`standing`]) when it differs from the last one this
    /// returned, `None` while it is unchanged.
    pub fn standing_changed(&mut self, app: &str) -> Option<String> {
        let a = self.apps.get_mut(app)?;
        let now = standing(a.members.values());
        if now == a.standing {
            return None;
        }
        a.standing = now.clone();
        Some(now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Capacity;
    use crate::gateway::member_fixture::member;

    fn ready(node: u64, host: &str) -> MemberView {
        let mut m = member(node, node);
        m.host = host.into();
        m.capacity = Some(Capacity {
            state: MemberState::Ready,
            ..Capacity::default()
        });
        m
    }

    /// **Every reason a member is routed nothing is named**, and a routable
    /// member names none.
    #[test]
    fn a_passed_over_member_names_every_reason() {
        let mut m = ready(1, "bw");
        assert!(passed_over(&m).is_empty());
        m.syncing = true;
        m.silent = true;
        assert_eq!(passed_over(&m), vec!["syncing", "silent"]);
        m.capacity = None;
        m.lagging = true;
        m.closing = true;
        assert_eq!(
            passed_over(&m),
            vec!["loading", "lagging", "syncing", "closing", "silent"]
        );
    }

    /// **The standing is logged once per change**: the first look and each
    /// change answer the line, an unchanged look answers nothing.
    #[test]
    fn the_standing_is_answered_only_when_it_changes() {
        let mut r = Registry::new();
        r.join(ready(0xa0, "BlackWorld"));
        r.join(ready(0x89, "WhiteWorld"));
        assert_eq!(
            r.standing_changed("zend").as_deref(),
            Some(
                "WhiteWorld 0000000000000089: routable; \
                 BlackWorld 00000000000000a0: routable"
            )
        );
        assert_eq!(r.standing_changed("zend"), None);
        r.set_syncing("zend", 0xa0, true);
        assert_eq!(
            r.standing_changed("zend").as_deref(),
            Some(
                "WhiteWorld 0000000000000089: routable; \
                 BlackWorld 00000000000000a0: passed over (syncing)"
            )
        );
        assert_eq!(r.standing_changed("zend"), None);
        assert_eq!(r.standing_changed("npcd"), None, "no such app here");
    }
}
