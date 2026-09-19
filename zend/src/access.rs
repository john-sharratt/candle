//! Who may use which tools mode.
//!
//! The tools dial decides how far a conversation's tools reach, and its top two
//! settings reach a long way: Comprehensive offers the high-risk tools, and
//! Mutable lets the file tools change the project on disk. Both are for the
//! estate's admins. Everyone else — signed in or not — gets Restricted by
//! default and may go no further than it.
//!
//! # Where the caller comes from
//!
//! The gateway in front of zend signs people in and forwards who they are on
//! the `x-tokera-*` headers ([`web::auth::forwarded`]); the role is resolved
//! from [`ZEND_ROLES`], this deployment's table. The headers are believed only
//! from a trusted peer ([`Gateways`]): a caller that reaches zend's port
//! directly is anonymous whatever it claims. A request with no identity is
//! [`Role::Unauthenticated`], and gets the least a person gets.
//!
//! # A mode the caller may not use
//!
//! Asked for by a non-admin, Comprehensive or Mutable is **served as
//! Restricted**, not refused: the turn still runs, on the tools the caller is
//! entitled to. A refusal would cost the whole turn for a dial set too high,
//! and the GUI never offers those modes to a non-admin in the first place.

use std::fmt::{self, Display, Formatter};
use std::iter;
use std::net::IpAddr;

use axum::http::HeaderMap;
use web::auth::forwarded::identify;
use web::auth::{Role, Roles};
use zend_tools::{Capability, Grants};

use crate::types::ToolMode;

/// This deployment's role table: who is an admin. Configuration, embedded at
/// build time — there is deliberately no API that grants a role.
pub const ZEND_ROLES: &str = include_str!("../zend.roles.yaml");

/// Parse [`ZEND_ROLES`]. A table that does not parse is a build defect, found
/// by the test below rather than at startup.
pub fn roles() -> Roles {
    serde_yaml::from_str(ZEND_ROLES).expect("zend.roles.yaml parses")
}

/// The peers whose `x-tokera-*` headers are believed: loopback, the address
/// zend binds, and each `--gateway`.
///
/// The gateway strips any identity a client sends and sets its own, but that
/// protects only requests that pass through it. zend listens on a LAN address,
/// and any machine that can reach that port directly could otherwise send
/// `x-tokera-email` naming an admin and be served Mutable. The gateway on this
/// box connects from the bound address itself, so that is trusted by default.
#[derive(Debug, Clone, Default)]
pub struct Gateways(Vec<IpAddr>);

impl Gateways {
    pub fn new(bind: IpAddr, gateways: &[IpAddr]) -> Self {
        let mut peers: Vec<IpAddr> = Vec::new();
        for ip in iter::once(bind).chain(gateways.iter().copied()) {
            let ip = canonical(ip);
            if !ip.is_unspecified() && !peers.contains(&ip) {
                peers.push(ip);
            }
        }
        Self(peers)
    }

    /// Whether identity headers from `peer` are believed.
    pub fn trusts(&self, peer: IpAddr) -> bool {
        let peer = canonical(peer);
        peer.is_loopback() || self.0.contains(&peer)
    }
}

impl Display for Gateways {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("loopback")?;
        for ip in &self.0 {
            write!(f, ", {ip}")?;
        }
        Ok(())
    }
}

/// An IPv4 peer seen on a dual-stack socket arrives as `::ffff:a.b.c.d`.
fn canonical(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        v4 => v4,
    }
}

/// The caller's role, from the gateway's headers — believed only when `peer`
/// is a trusted gateway. A request from anywhere else, or with no known peer,
/// is [`Role::Unauthenticated`] whatever it claims.
pub fn role(headers: &HeaderMap, peer: Option<IpAddr>, gateways: &Gateways, roles: &Roles) -> Role {
    let claimed = identify(headers).ok();
    match peer {
        Some(p) if gateways.trusts(p) => roles.of(claimed.as_ref()),
        _ => {
            if let Some(id) = &claimed {
                tracing::warn!(
                    peer = ?peer,
                    claimed = %id.email,
                    "identity headers from a peer that is not a trusted gateway — ignored; \
                     the caller is anonymous",
                );
            }
            Role::Unauthenticated
        }
    }
}

/// Whether `role` may run a turn in `mode`.
pub fn allows(role: Role, mode: ToolMode) -> bool {
    match mode {
        ToolMode::None | ToolMode::Restricted => true,
        ToolMode::Comprehensive | ToolMode::Mutable => role.at_least(Role::Admin),
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
///   conversation, the overlay and the workspace as read.
/// - Comprehensive grants the network and stored credentials. Its file changes
///   stay in the overlay, so it grants neither the disk nor execution on this
///   host: code or a program that runs here reaches the real filesystem,
///   which no overlay can stand in front of.
/// - Mutable grants everything, the disk and execution included.
pub fn grants(mode: ToolMode) -> Grants {
    match mode {
        ToolMode::None | ToolMode::Restricted => Grants::NONE,
        ToolMode::Comprehensive => Grants::NONE
            .with(Capability::Network)
            .with(Capability::Secrets),
        ToolMode::Mutable => Grants::ALL,
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
        ToolMode::Comprehensive | ToolMode::Mutable => within,
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
    const OTHERS: [Role; 2] = [Role::Unauthenticated, Role::User];

    #[test]
    fn the_embedded_table_parses_and_names_someone() {
        assert!(!roles().is_empty(), "zend.roles.yaml names no admin");
    }

    #[test]
    fn admins_may_use_every_mode_and_default_to_comprehensive() {
        for role in ADMINS {
            assert_eq!(allowed_modes(role), ToolMode::ALL);
            assert_eq!(default_mode(role), ToolMode::Comprehensive);
            assert_eq!(effective_mode(role, None), ToolMode::Comprehensive);
            assert_eq!(
                effective_mode(role, Some(ToolMode::Mutable)),
                ToolMode::Mutable
            );
        }
    }

    /// **Everyone else gets Restricted, and cannot climb above it.** Asking for
    /// Comprehensive or Mutable runs the turn restricted rather than refusing it.
    #[test]
    fn everyone_else_defaults_to_restricted_and_is_held_there() {
        for role in OTHERS {
            assert_eq!(allowed_modes(role), [ToolMode::None, ToolMode::Restricted]);
            assert_eq!(effective_mode(role, None), ToolMode::Restricted);
            for above in [ToolMode::Comprehensive, ToolMode::Mutable] {
                assert_eq!(effective_mode(role, Some(above)), ToolMode::Restricted);
            }
            assert_eq!(effective_mode(role, Some(ToolMode::None)), ToolMode::None);
        }
    }

    /// **Only Mutable may touch the disk or run code here, and nothing below
    /// Comprehensive may reach past the conversation.** A non-admin is held to
    /// Restricted, so a non-admin's round runs with no grant at all.
    #[test]
    fn each_mode_grants_what_it_offers_and_no_more() {
        assert_eq!(grants(ToolMode::None), Grants::NONE);
        assert_eq!(grants(ToolMode::Restricted), Grants::NONE);
        let comprehensive = grants(ToolMode::Comprehensive);
        for cap in [Capability::DiskWrite, Capability::Exec] {
            assert!(!comprehensive.has(cap), "comprehensive holds {cap}");
        }
        for cap in [Capability::Network, Capability::Secrets] {
            assert!(comprehensive.has(cap), "comprehensive lacks {cap}");
        }
        assert_eq!(grants(ToolMode::Mutable), Grants::ALL);
        for role in OTHERS {
            for asked in ToolMode::ALL {
                assert_eq!(grants(effective_mode(role, Some(asked))), Grants::NONE);
            }
            assert_eq!(grants(effective_mode(role, None)), Grants::NONE);
        }
    }

    /// A mode offers exactly the tools its grants cover: code execution only
    /// in Mutable, the network from Comprehensive up, high-risk tools never
    /// in Restricted, and nothing at all in None.
    #[test]
    fn a_mode_offers_only_what_it_would_run() {
        let exec = [Capability::Exec];
        let net = [Capability::Network];
        assert!(offers(ToolMode::Mutable, &exec, true));
        assert!(!offers(ToolMode::Comprehensive, &exec, true));
        assert!(offers(ToolMode::Comprehensive, &net, true));
        assert!(!offers(ToolMode::Restricted, &net, false));
        assert!(offers(ToolMode::Restricted, &[], false));
        assert!(!offers(ToolMode::Restricted, &[], true));
        assert!(!offers(ToolMode::None, &[], false));
    }

    /// The role comes from the gateway's headers; a request without them is
    /// anonymous and gets no admin mode, whatever it asks for.
    #[test]
    fn the_role_is_read_from_the_gateways_headers() {
        let table: Roles = serde_yaml::from_str("admins:\n  - email: admin@example.com\n").unwrap();
        let gw = Gateways::new(ip("192.168.0.5"), &[]);
        let from_gw = Some(ip("192.168.0.5"));
        let mut h = HeaderMap::new();
        assert_eq!(role(&h, from_gw, &gw, &table), Role::Unauthenticated);
        h.insert("x-tokera-user", "g-1".parse().unwrap());
        h.insert("x-tokera-provider", "google".parse().unwrap());
        h.insert("x-tokera-email", "someone@example.com".parse().unwrap());
        assert_eq!(role(&h, from_gw, &gw, &table), Role::User);
        h.insert("x-tokera-email", "admin@example.com".parse().unwrap());
        assert_eq!(role(&h, from_gw, &gw, &table), Role::Admin);
    }

    /// **A peer that is not the gateway cannot claim an identity.** Another
    /// machine on the LAN sending the admin's headers straight to zend's port
    /// is anonymous, as is a request whose peer is unknown.
    #[test]
    fn identity_from_any_other_peer_is_ignored() {
        let table: Roles = serde_yaml::from_str("admins:\n  - email: admin@example.com\n").unwrap();
        let gw = Gateways::new(ip("192.168.0.5"), &[ip("10.0.0.9")]);
        let mut h = HeaderMap::new();
        h.insert("x-tokera-user", "g-1".parse().unwrap());
        h.insert("x-tokera-provider", "google".parse().unwrap());
        h.insert("x-tokera-email", "admin@example.com".parse().unwrap());
        for forger in [Some(ip("192.168.0.77")), Some(ip("10.0.0.8")), None] {
            assert_eq!(
                role(&h, forger, &gw, &table),
                Role::Unauthenticated,
                "{forger:?}"
            );
        }
        for trusted in [
            "127.0.0.1",
            "::1",
            "192.168.0.5",
            "::ffff:192.168.0.5",
            "10.0.0.9",
        ] {
            assert_eq!(
                role(&h, Some(ip(trusted)), &gw, &table),
                Role::Admin,
                "{trusted}"
            );
        }
    }

    /// Binding every interface trusts no address for it — only loopback and
    /// the named gateways.
    #[test]
    fn an_unspecified_bind_is_not_a_trusted_peer() {
        let gw = Gateways::new(ip("0.0.0.0"), &[]);
        assert!(!gw.trusts(ip("0.0.0.0")));
        assert!(!gw.trusts(ip("192.168.0.5")));
        assert!(gw.trusts(ip("127.0.0.1")));
        assert!(Gateways::default().trusts(ip("127.0.0.1")));
        assert!(!Gateways::default().trusts(ip("192.168.0.5")));
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }
}
