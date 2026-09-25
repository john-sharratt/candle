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
//!
//! # `--local-signin` — standing in for a gateway that isn't there
//!
//! A developer running zend directly, with no Tokera gateway in front of it,
//! has no way to arrive with `x-tokera-*` headers and so is always
//! [`Role::Unauthenticated`] — there is deliberately no API that grants a
//! role. `--local-signin <email>` is the one boot-time way out: it resolves
//! `email` against [`ZEND_ROLES`] and applies **only** to a request whose
//! peer is genuine loopback (this machine, not a configured `--gateway`) and
//! that carries no forwarded identity headers of its own. Any real forwarded
//! identity, from any trusted peer, always takes precedence. It is still a
//! widening of trust on this box — any other local process or user account
//! can now also reach zend's port and be recognized as that email — which is
//! why it is off by default and logged loudly at startup when set.

use std::fmt::{self, Display, Formatter};
use std::iter;
use std::net::IpAddr;

use axum::http::HeaderMap;
use web::auth::forwarded::identify;
use web::auth::session::Identity;
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

/// Whether `peer` is this machine, not merely a configured `--gateway`.
///
/// `--local-signin` stands in for a gateway on the box that would otherwise
/// run one, so it must not reach further than loopback already does — a
/// `--gateway` IP is a *different* machine the operator has chosen to trust
/// for forwarded headers, and letting a flag on this process grant an
/// identity to callers arriving over the network would widen that trust
/// silently.
fn is_loopback(peer: IpAddr) -> bool {
    canonical(peer).is_loopback()
}

/// The synthetic identity `--local-signin <email>` presents on behalf of a
/// loopback caller that sent no `x-tokera-*` headers of its own.
///
/// `sub` is a fixed placeholder rather than empty: [`Roles::of`] treats an
/// empty subject as no identity at all (mirroring the real gateway, which
/// never forwards one), and this identity is deliberately real. The table is
/// keyed on email for this path — `zend.roles.yaml` names people by email —
/// so the placeholder subject never needs to match anything itself.
fn local_identity(email: &str) -> Identity {
    Identity {
        provider: "local-signin".to_string(),
        sub: "local-signin".to_string(),
        email: email.to_string(),
        name: String::new(),
        picture: String::new(),
        exp: 0,
    }
}

/// The caller's role, from the gateway's headers — believed only when `peer`
/// is a trusted gateway. A request from anywhere else, or with no known peer,
/// is [`Role::Unauthenticated`] whatever it claims.
///
/// `local_signin` is `--local-signin`'s email, if the daemon was started with
/// it. It applies only when every one of these holds: the peer is truly
/// loopback (not merely a trusted `--gateway`), and the request carries no
/// `x-tokera-*` headers at all — a forwarded identity, however it resolves,
/// always wins over the flag.
pub fn role(
    headers: &HeaderMap,
    peer: Option<IpAddr>,
    gateways: &Gateways,
    roles: &Roles,
    local_signin: Option<&str>,
) -> Role {
    let claimed = identify(headers).ok();
    match peer {
        Some(p) if gateways.trusts(p) => {
            if claimed.is_none() {
                if let Some(email) = local_signin {
                    if is_loopback(p) {
                        return roles.of(Some(&local_identity(email)));
                    }
                }
            }
            roles.of(claimed.as_ref())
        }
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
/// - Comprehensive grants the network, stored credentials and the JS sandbox.
///   Its file changes stay in the overlay, so it grants neither the disk nor
///   execution on this host: a program that runs here reaches the real
///   filesystem, which no overlay can stand in front of. The sandbox can —
///   its only filesystem is the context's file store, the overlay itself.
/// - Mutable grants everything, the disk and execution included.
pub fn grants(mode: ToolMode) -> Grants {
    match mode {
        ToolMode::None | ToolMode::Restricted => Grants::NONE,
        ToolMode::Comprehensive => Grants::NONE
            .with(Capability::Network)
            .with(Capability::Sandbox)
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
        for cap in [
            Capability::Network,
            Capability::Sandbox,
            Capability::Secrets,
        ] {
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
        assert_eq!(role(&h, from_gw, &gw, &table, None), Role::Unauthenticated);
        h.insert("x-tokera-user", "g-1".parse().unwrap());
        h.insert("x-tokera-provider", "google".parse().unwrap());
        h.insert("x-tokera-email", "someone@example.com".parse().unwrap());
        assert_eq!(role(&h, from_gw, &gw, &table, None), Role::User);
        h.insert("x-tokera-email", "admin@example.com".parse().unwrap());
        assert_eq!(role(&h, from_gw, &gw, &table, None), Role::Admin);
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
                role(&h, forger, &gw, &table, None),
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
                role(&h, Some(ip(trusted)), &gw, &table, None),
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

    /// **The whole point of `--local-signin`.** A loopback caller with no
    /// forwarded identity resolves the flag's email against the roles table,
    /// same as a real gateway would have resolved it.
    #[test]
    fn local_signin_resolves_a_loopback_caller_with_no_headers() {
        let table: Roles = serde_yaml::from_str("creators:\n  - email: me@example.com\n").unwrap();
        let gw = Gateways::default();
        let h = HeaderMap::new();
        assert_eq!(
            role(
                &h,
                Some(ip("127.0.0.1")),
                &gw,
                &table,
                Some("me@example.com")
            ),
            Role::Creator
        );
        assert_eq!(
            role(&h, Some(ip("::1")), &gw, &table, Some("me@example.com")),
            Role::Creator
        );
    }

    /// An email the flag names that is not in the table is a real, signed-in
    /// nobody — `User`, not an error and not `Unauthenticated`.
    #[test]
    fn local_signin_for_an_unlisted_email_is_a_plain_user() {
        let table: Roles = serde_yaml::from_str("creators:\n  - email: me@example.com\n").unwrap();
        let gw = Gateways::default();
        let h = HeaderMap::new();
        assert_eq!(
            role(
                &h,
                Some(ip("127.0.0.1")),
                &gw,
                &table,
                Some("nobody@example.com")
            ),
            Role::User
        );
    }

    /// A real forwarded identity always wins — the flag never overrides
    /// headers the request actually carried, whatever they resolve to.
    #[test]
    fn local_signin_never_overrides_a_forwarded_identity() {
        let table: Roles = serde_yaml::from_str(
            "creators:\n  - email: me@example.com\nadmins:\n  - email: other@example.com\n",
        )
        .unwrap();
        let gw = Gateways::default();
        let mut h = HeaderMap::new();
        h.insert("x-tokera-user", "g-1".parse().unwrap());
        h.insert("x-tokera-provider", "google".parse().unwrap());
        h.insert("x-tokera-email", "other@example.com".parse().unwrap());
        assert_eq!(
            role(
                &h,
                Some(ip("127.0.0.1")),
                &gw,
                &table,
                Some("me@example.com")
            ),
            Role::Admin,
            "the forwarded admin identity must win over the flag's creator email",
        );
    }

    /// **The flag must not reach past loopback.** A `--gateway` peer is a
    /// different machine the operator chose to trust for *forwarded* headers;
    /// it must not also gain the local flag's identity when it sends none.
    #[test]
    fn local_signin_does_not_apply_to_a_trusted_gateway_peer() {
        let table: Roles = serde_yaml::from_str("creators:\n  - email: me@example.com\n").unwrap();
        let gw = Gateways::new(ip("192.168.0.5"), &[ip("10.0.0.9")]);
        let h = HeaderMap::new();
        for peer in ["192.168.0.5", "10.0.0.9"] {
            assert_eq!(
                role(&h, Some(ip(peer)), &gw, &table, Some("me@example.com")),
                Role::Unauthenticated,
                "{peer}"
            );
        }
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }
}
