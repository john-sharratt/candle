//! The house network's edge (`docs/zen_moon_mind.md` §13): the private subnets
//! the member running a bridge stands on, and the one test every address is
//! held to — on one of those subnets, or refused. A bridge reaches the house
//! and nothing beyond it.

use std::net::Ipv4Addr;

/// One of the member's own private subnets: the address it has there, and the
/// mask that says which bits are the subnet's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalNet {
    pub own: Ipv4Addr,
    pub mask: Ipv4Addr,
}

impl LocalNet {
    /// Whether `ip` is on this subnet — its subnet bits match the member's.
    pub fn holds(&self, ip: Ipv4Addr) -> bool {
        let net = |a: Ipv4Addr| u32::from(a) & u32::from(self.mask);
        net(self.own) == net(ip)
    }
}

/// Whether `ip` is a private address — RFC 1918 — the only kind a house network
/// is on.
pub fn is_private(ip: Ipv4Addr) -> bool {
    ip.is_private()
}

/// The subnets a bridge may reach: each private one this member stands on, from
/// its interfaces. A public interface is no house network, and is left out.
pub fn local_nets(interfaces: impl IntoIterator<Item = (Ipv4Addr, Ipv4Addr)>) -> Vec<LocalNet> {
    interfaces
        .into_iter()
        .filter(|(own, _)| is_private(*own))
        .map(|(own, mask)| LocalNet { own, mask })
        .collect()
}

/// This member's private subnets, from its interfaces, now.
pub fn nets() -> Vec<LocalNet> {
    let interfaces = if_addrs::get_if_addrs().unwrap_or_default();
    local_nets(interfaces.into_iter().filter_map(|i| match i.addr {
        if_addrs::IfAddr::V4(v4) => Some((v4.ip, v4.netmask)),
        if_addrs::IfAddr::V6(_) => None,
    }))
}

/// Whether `ip` is in the house: private, and on one of `nets`. The test every
/// device a bridge touches passes, or the bridge refuses it.
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

    /// **Only an address on the member's own private subnet is in the house**:
    /// another private subnet, and any public address, are out.
    #[test]
    fn only_the_members_own_subnet_is_the_house() {
        let home = [home()];
        assert!(in_house(&home, ip("192.168.0.89")));
        assert!(in_house(&home, ip("192.168.0.1")));
        assert!(!in_house(&home, ip("192.168.1.5")), "another subnet");
        assert!(!in_house(&home, ip("10.0.0.5")), "another private net");
        assert!(!in_house(&home, ip("8.8.8.8")), "public");
    }

    /// **A public interface is no house network** and is dropped from the
    /// scope, so nothing on it is ever reached.
    #[test]
    fn a_public_interface_is_left_out() {
        let nets = local_nets([
            (ip("192.168.0.237"), ip("255.255.255.0")),
            (ip("203.0.113.7"), ip("255.255.255.0")),
        ]);
        assert_eq!(nets, [home()]);
    }
}
