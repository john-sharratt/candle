//! Which peers may name the caller.
//!
//! The gateway signs people in and forwards who they are on `x-tokera-*`
//! ([`super::forwarded`]). It strips any identity a client sends and sets its
//! own, but that protects only requests that pass through it. A daemon listens
//! on a LAN address, and any machine that can reach that port directly could
//! otherwise send `x-tokera-email` naming an admin. So a server that reads the
//! headers believes them only from a trusted peer — loopback, the address it
//! binds, and each named gateway — and clears them from anyone else.

use std::fmt::{self, Display, Formatter};
use std::iter;
use std::net::IpAddr;

/// The peers whose `x-tokera-*` headers are believed: loopback, the address the
/// server binds, and each named gateway. The gateway on the same box connects
/// from the bound address itself, so that is trusted by default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

    /// The trusted addresses, loopback aside.
    pub fn peers(&self) -> &[IpAddr] {
        &self.0
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
pub fn canonical(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        v4 => v4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    /// Loopback, the bound address in either spelling, and each named gateway
    /// are believed; nothing else is.
    #[test]
    fn only_loopback_the_bind_and_named_gateways_are_trusted() {
        let gw = Gateways::new(ip("192.168.0.5"), &[ip("10.0.0.9")]);
        for trusted in [
            "127.0.0.1",
            "::1",
            "192.168.0.5",
            "::ffff:192.168.0.5",
            "10.0.0.9",
        ] {
            assert!(gw.trusts(ip(trusted)), "{trusted}");
        }
        for forger in ["192.168.0.77", "10.0.0.8"] {
            assert!(!gw.trusts(ip(forger)), "{forger}");
        }
        assert_eq!(gw.to_string(), "loopback, 192.168.0.5, 10.0.0.9");
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
        assert!(gw.peers().is_empty());
    }
}
