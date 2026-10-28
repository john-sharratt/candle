//! Who may use which tools mode.
//!
//! The tools dial decides how far a conversation's tools reach, and its top
//! setting reaches a long way: Comprehensive offers the high-risk tools — the
//! network, credentials, code, the git writers and programs on this host. It is
//! for the estate's admins. Everyone else — signed in or not — gets Restricted
//! by default and may go no further than it.
//!
//! Who the caller is, and their role, is the host's rule for every tenant
//! (`host::access`): the gateway's forwarded identity, believed only from a
//! trusted peer, resolved against the estate's table. A request with no
//! identity is [`Role::Unauthenticated`], and gets the least a person gets.
//!
//! # A mode the caller may not use
//!
//! Asked for by a non-admin, Comprehensive is **served as Restricted**, not
//! refused: the turn still runs, on the tools the caller is entitled to. A
//! refusal would cost the whole turn for a dial set too high, and the GUI never
//! offers that mode to a non-admin in the first place.

use web::auth::Role;
use zend_tools::{Capability, Grants};

use crate::types::ToolMode;

/// Whether `role` may run a turn in `mode`.
pub fn allows(role: Role, mode: ToolMode) -> bool {
    match mode {
        ToolMode::None | ToolMode::Restricted => true,
        ToolMode::Comprehensive => role.at_least(Role::Admin),
    }
}

/// The modes `role` may choose, in dial order.
pub fn allowed_modes(role: Role) -> Vec<ToolMode> {
    ToolMode::ALL
        .into_iter()
        .filter(|m| allows(role, *m))
        .collect()
}

/// The mode a turn runs in when the caller names none.
pub fn default_mode(role: Role) -> ToolMode {
    if role.at_least(Role::Admin) {
        ToolMode::Comprehensive
    } else {
        ToolMode::Restricted
    }
}

/// What a round of tools in `mode` may do outside the conversation — the
/// [`Grants`] its [`ToolContext`](zend_tools::ToolContext) carries, which the
/// registry checks at dispatch and the network, process, VM, database and
/// disk primitives check again where the action happens.
///
/// - None and Restricted grant nothing: their tools answer from the
///   conversation and the workspace as read, and their file changes stay in
///   the conversation's overlay, which needs no grant — it never writes the
///   disk. No git writer and no program on this host runs without one.
/// - Comprehensive grants everything: the network, stored credentials, the JS
///   sandbox, the disk (which the git writers' checkout runs and SQLite need)
///   and programs on this host.
pub fn grants(mode: ToolMode) -> Grants {
    match mode {
        ToolMode::None | ToolMode::Restricted => Grants::NONE,
        ToolMode::Comprehensive => Grants::ALL,
    }
}

/// Whether a tool needing `requires` is offered in `mode`: its needs are
/// within the mode's grants, and — below Comprehensive — it is not marked
/// high-risk. A mode never offers a tool it would refuse.
pub fn offers(mode: ToolMode, requires: &[Capability], high_risk: bool) -> bool {
    let within = grants(mode).require_all(requires).is_ok();
    match mode {
        ToolMode::None => false,
        ToolMode::Restricted => within && !high_risk,
        ToolMode::Comprehensive => within,
    }
}

/// The mode a turn actually runs in: the one asked for when the caller may use
/// it, the caller's default when none was asked for, and Restricted when the
/// one asked for is above the caller.
pub fn effective_mode(role: Role, requested: Option<ToolMode>) -> ToolMode {
    match requested {
        None => default_mode(role),
        Some(mode) if allows(role, mode) => mode,
        Some(mode) => {
            tracing::warn!(
                %role,
                requested = mode.id(),
                "tools mode above the caller's role — running the turn as restricted",
            );
            ToolMode::Restricted
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADMINS: [Role; 2] = [Role::Admin, Role::Creator];
    const OTHERS: [Role; 3] = [Role::Unauthenticated, Role::Guest, Role::User];

    #[test]
    fn admins_may_use_every_mode_and_default_to_comprehensive() {
        for role in ADMINS {
            assert_eq!(allowed_modes(role), ToolMode::ALL);
            assert_eq!(default_mode(role), ToolMode::Comprehensive);
            assert_eq!(effective_mode(role, None), ToolMode::Comprehensive);
            assert_eq!(
                effective_mode(role, Some(ToolMode::Restricted)),
                ToolMode::Restricted
            );
        }
    }

    /// **Everyone else gets Restricted, and cannot climb above it.** Asking for
    /// Comprehensive runs the turn restricted rather than refusing it.
    #[test]
    fn everyone_else_defaults_to_restricted_and_is_held_there() {
        for role in OTHERS {
            assert_eq!(allowed_modes(role), [ToolMode::None, ToolMode::Restricted]);
            assert_eq!(effective_mode(role, None), ToolMode::Restricted);
            assert_eq!(
                effective_mode(role, Some(ToolMode::Comprehensive)),
                ToolMode::Restricted
            );
            assert_eq!(effective_mode(role, Some(ToolMode::None)), ToolMode::None);
        }
    }

    /// **Comprehensive holds everything, and nothing below it may reach past
    /// the conversation.** A non-admin is held to Restricted, so a non-admin's
    /// round runs with no grant at all.
    #[test]
    fn each_mode_grants_what_it_offers_and_no_more() {
        assert_eq!(grants(ToolMode::None), Grants::NONE);
        assert_eq!(grants(ToolMode::Restricted), Grants::NONE);
        assert_eq!(grants(ToolMode::Comprehensive), Grants::ALL);
        for role in OTHERS {
            for asked in ToolMode::ALL {
                assert_eq!(grants(effective_mode(role, Some(asked))), Grants::NONE);
            }
            assert_eq!(grants(effective_mode(role, None)), Grants::NONE);
        }
    }

    /// A mode offers exactly the tools its grants cover: everything in
    /// Comprehensive, no tool needing a grant and no high-risk tool in
    /// Restricted, and nothing at all in None.
    #[test]
    fn a_mode_offers_only_what_it_would_run() {
        for cap in Capability::ALL {
            assert!(offers(ToolMode::Comprehensive, &[cap], true), "{cap}");
            assert!(!offers(ToolMode::Restricted, &[cap], false), "{cap}");
        }
        assert!(offers(ToolMode::Comprehensive, &[], false));
        assert!(offers(ToolMode::Restricted, &[], false));
        assert!(!offers(ToolMode::Restricted, &[], true));
        assert!(!offers(ToolMode::None, &[], false));
    }
}
