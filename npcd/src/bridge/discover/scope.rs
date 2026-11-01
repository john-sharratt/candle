//! **Discovery never leaves the house** (`docs/zen_moon_mind.md` §13): the
//! networks a survey may find anything on are the member's own subnets, and
//! only those that are private.
//!
//! A finding's address is held to this before it is a finding, and a device's
//! own description is fetched only from an address that passes it — so nothing
//! on the internet can be found, and nothing found can be onboarded.

use std::net::Ipv4Addr;

/// One subnet this member stands on: its own address and the netmask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalNet {
    pub own: Ipv4Addr,
    pub mask: Ipv4Addr,
}

impl LocalNet {
    /// Whether `ip` is another host on this subnet: inside it, not the
    /// network's own or broadcast address, and not this member.
    pub fn holds(&self, ip: Ipv4Addr) -> bool {
        let mask = u32::from(self.mask);
        let net = u32::from(self.own) & mask;
        let host = u32::from(ip);
        host & mask == net && host != net && host != net | !mask && ip != self.own
    }
}

/// Whether `ip` is a private address — RFC 1918 — the only kind a house
/// network is on.
pub fn is_private(ip: Ipv4Addr) -> bool {
    ip.is_private()
}

/// The subnets a survey may find things on: each private one this member
/// stands on, from its interfaces. A public interface is no house network,
/// and is left out.
pub fn local_nets(interfaces: impl IntoIterator<Item = (Ipv4Addr, Ipv4Addr)>) -> Vec<LocalNet> {
    interfaces
        .into_iter()
        .filter(|(own, _)| is_private(*own))
        .map(|(own, mask)| LocalNet { own, mask })
        .collect()
}

/// Whether `ip` is on one of `nets` — the test every address in a survey
/// passes or is dropped at.
pub fn in_house(nets: &[LocalNet], ip: Ipv4Addr) -> bool {
    is_private(ip) && nets.iter().any(|n| n.holds(ip))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> Ipv4Addr {
        s.parse().unwrap()
    }

    fn home() -> LocalNet {
        LocalNet {
            own: ip("192.168.0.237"),
            mask: ip("255.255.255.0"),
        }
    }

    /// **Only the member's own private subnet is the house**: a neighbour on
    /// it is, and the internet, another private subnet, the member itself, and
    /// the subnet's own and broadcast addresses are not.
    #[test]
    fn only_the_members_own_subnet_is_the_house() {
        let nets = [home()];
        assert!(in_house(&nets, ip("192.168.0.89")));
        assert!(in_house(&nets, ip("192.168.0.1")));
        assert!(!in_house(&nets, ip("8.8.8.8")), "the internet");
        assert!(!in_house(&nets, ip("142.250.70.14")), "the internet");
        assert!(!in_house(&nets, ip("192.168.1.20")), "another subnet");
        assert!(!in_house(&nets, ip("10.0.0.5")), "another private network");
        assert!(!in_house(&nets, ip("192.168.0.237")), "the member itself");
        assert!(!in_house(&nets, ip("192.168.0.0")), "the network address");
        assert!(
            !in_house(&nets, ip("192.168.0.255")),
            "the broadcast address"
        );
        assert!(!in_house(&nets, ip("239.255.255.250")), "multicast");
    }

    /// **A public interface is no house network.**
    #[test]
    fn a_public_interface_is_left_out() {
        let nets = local_nets([
            (ip("192.168.0.237"), ip("255.255.255.0")),
            (ip("203.0.113.7"), ip("255.255.255.0")),
            (ip("127.0.0.1"), ip("255.0.0.0")),
        ]);
        assert_eq!(nets, [home()]);
        assert!(!in_house(&nets, ip("203.0.113.9")));
    }
}
