//! The addresses a dial may go out from (`docs/replicated_substrate.md` §3.2,
//! paths). A machine with one address lets the system choose. A machine with
//! two on one network — a wired and a wireless link to the same LAN — dials
//! each destination once per address, bound to it, so a dead link is a path
//! that fails rather than a route the system keeps taking: a connection the
//! system routes is sent on its preferred link whether or not that link works.

use std::net::{IpAddr, UdpSocket};

use if_addrs::{get_if_addrs, IfAddr};

/// One of this machine's addresses, with the length of its network prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Local {
    pub ip: IpAddr,
    pub prefix: u8,
}

impl Local {
    /// Whether `ip` is on this address's network.
    fn holds(&self, ip: IpAddr) -> bool {
        match (self.ip, ip) {
            (IpAddr::V4(a), IpAddr::V4(b)) => {
                let mask = u32::MAX
                    .checked_shl(32 - u32::from(self.prefix))
                    .unwrap_or(0);
                u32::from(a) & mask == u32::from(b) & mask
            }
            (IpAddr::V6(a), IpAddr::V6(b)) => {
                let mask = u128::MAX
                    .checked_shl(128 - u32::from(self.prefix))
                    .unwrap_or(0);
                u128::from(a) & mask == u128::from(b) & mask
            }
            _ => false,
        }
    }
}

fn link_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_link_local(),
        IpAddr::V6(v6) => v6.segments()[0] & 0xffc0 == 0xfe80,
    }
}

/// This machine's addresses a dial may bind: on an interface that is up, not
/// loopback, not link-local.
pub fn locals() -> Vec<Local> {
    let Ok(interfaces) = get_if_addrs() else {
        return Vec::new();
    };
    interfaces
        .into_iter()
        .filter(|i| i.is_oper_up() && !i.is_loopback())
        .map(|i| match i.addr {
            IfAddr::V4(a) => Local {
                ip: IpAddr::V4(a.ip),
                prefix: a.prefixlen,
            },
            IfAddr::V6(a) => Local {
                ip: IpAddr::V6(a.ip),
                prefix: a.prefixlen,
            },
        })
        .filter(|l| !link_local(l.ip))
        .collect()
}

/// Whether `ip` is this machine's own: loopback, or an address a socket can
/// bind. Asked of the system rather than read from the interface list, so an
/// address on a link that is down still reads as this machine's — a dial to
/// it never leaves the machine.
pub fn is_own(ip: IpAddr) -> bool {
    ip.is_loopback() || std::net::TcpListener::bind((ip, 0)).is_ok()
}

/// The address the system would send to `dest` from: a UDP socket connected
/// to it, which sends nothing, reads its route.
pub fn preferred(dest: IpAddr) -> Option<IpAddr> {
    let any: IpAddr = match dest {
        IpAddr::V4(_) => IpAddr::from([0u8; 4]),
        IpAddr::V6(_) => IpAddr::from([0u16; 8]),
    };
    let socket = UdpSocket::bind((any, 0)).ok()?;
    socket.connect((dest, 9)).ok()?;
    Some(socket.local_addr().ok()?.ip())
}

/// The sources to dial `dest` from, given this machine's addresses and the one
/// the system would choose: `None` lets the system choose. A destination on
/// this machine, or a machine with one candidate, is dialled unbound.
/// Otherwise each address of `dest`'s family is a source — only those on
/// `dest`'s network when there are any — the system's choice first, then the
/// rest in the order the system lists them: a machine whose route works dials
/// as it always has, and the others are tried when it does not.
pub fn choose(
    dest: IpAddr,
    locals: &[Local],
    own: bool,
    preferred: Option<IpAddr>,
) -> Vec<Option<IpAddr>> {
    if own {
        return vec![None];
    }
    let family: Vec<&Local> = locals
        .iter()
        .filter(|l| l.ip.is_ipv4() == dest.is_ipv4())
        .collect();
    let near: Vec<&Local> = family.iter().copied().filter(|l| l.holds(dest)).collect();
    let mut pick = if near.is_empty() { family } else { near };
    if pick.len() <= 1 {
        return vec![None];
    }
    // Stable: the system's choice first, the rest in their order.
    pick.sort_by_key(|l| Some(l.ip) != preferred);
    pick.into_iter().map(|l| Some(l.ip)).collect()
}

/// [`choose`] against this machine as it stands.
pub fn sources_for(dest: IpAddr) -> Vec<Option<IpAddr>> {
    choose(dest, &locals(), is_own(dest), preferred(dest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(a: [u8; 4], prefix: u8) -> Local {
        Local {
            ip: IpAddr::from(a),
            prefix,
        }
    }

    /// BlackWorld: a wired .4 and a wireless .5 on one /24, and a VPN address
    /// on another network.
    fn bw() -> Vec<Local> {
        vec![
            v4([192, 168, 0, 4], 24),
            v4([192, 168, 0, 5], 24),
            v4([10, 8, 0, 2], 24),
        ]
    }

    fn ip(a: [u8; 4]) -> IpAddr {
        IpAddr::from(a)
    }

    #[test]
    fn a_lan_destination_is_dialled_from_each_address_on_its_network() {
        assert_eq!(
            choose(ip([192, 168, 0, 6]), &bw(), false, None),
            [Some(ip([192, 168, 0, 4])), Some(ip([192, 168, 0, 5]))]
        );
    }

    /// BlackWorld's system routes the LAN by its wired link: that is tried
    /// first, then the wireless one. Were the system to prefer the wireless
    /// one, it would lead.
    #[test]
    fn the_systems_choice_is_dialled_first() {
        let lan = ip([192, 168, 0, 6]);
        assert_eq!(
            choose(lan, &bw(), false, Some(ip([192, 168, 0, 5]))),
            [Some(ip([192, 168, 0, 5])), Some(ip([192, 168, 0, 4]))]
        );
        // A choice that is no candidate changes nothing.
        assert_eq!(
            choose(lan, &bw(), false, Some(ip([10, 8, 0, 2]))),
            [Some(ip([192, 168, 0, 4])), Some(ip([192, 168, 0, 5]))]
        );
    }

    /// WhiteWorld: one wired link and a virtual switch's address. To the
    /// internet both are candidates, the system's route first, so a dial goes
    /// out as it always did and the virtual address is only a fallback.
    #[test]
    fn a_remote_destination_is_dialled_from_every_address_the_route_first() {
        let ww = vec![v4([172, 28, 0, 1], 20), v4([192, 168, 0, 6], 24)];
        assert_eq!(
            choose(
                ip([60, 227, 70, 132]),
                &ww,
                false,
                Some(ip([192, 168, 0, 6]))
            ),
            [Some(ip([192, 168, 0, 6])), Some(ip([172, 28, 0, 1]))]
        );
        let from = choose(ip([60, 227, 70, 132]), &bw(), false, None);
        assert_eq!(from.len(), 3);
        assert!(from.iter().all(Option::is_some));
    }

    /// A destination on this machine never leaves it, whatever its links do:
    /// BlackWorld dialling its own .4 while that link is dead.
    #[test]
    fn this_machines_own_address_is_dialled_unbound() {
        assert_eq!(choose(ip([192, 168, 0, 4]), &bw(), true, None), [None]);
        assert!(is_own(ip([127, 0, 0, 1])));
        // TEST-NET-1 is never assigned to a machine.
        assert!(!is_own(ip([192, 0, 2, 1])));
    }

    /// The system's route to loopback is loopback.
    #[test]
    fn the_preferred_source_is_read_from_the_route() {
        assert_eq!(preferred(ip([127, 0, 0, 1])), Some(ip([127, 0, 0, 1])));
    }

    /// One address, or one on the destination's family: the system chooses,
    /// as a machine with one link always has.
    #[test]
    fn one_candidate_is_left_to_the_system() {
        let ww = vec![v4([192, 168, 0, 6], 24)];
        assert_eq!(choose(ip([192, 168, 0, 5]), &ww, false, None), [None]);
        assert_eq!(choose(ip([1, 1, 1, 1]), &ww, false, None), [None]);
        let v6 = IpAddr::from([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]);
        assert_eq!(choose(v6, &bw(), false, None), [None]);
        assert_eq!(choose(v6, &[], false, None), [None]);
    }

    #[test]
    fn a_network_holds_the_addresses_under_its_prefix() {
        let l = v4([192, 168, 0, 5], 24);
        assert!(l.holds(IpAddr::from([192, 168, 0, 255])));
        assert!(!l.holds(IpAddr::from([192, 168, 1, 1])));
        assert!(v4([10, 0, 0, 1], 0).holds(IpAddr::from([8, 8, 8, 8])));
        assert!(!v4([10, 0, 0, 1], 32).holds(IpAddr::from([10, 0, 0, 2])));
        let six = Local {
            ip: IpAddr::from([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]),
            prefix: 64,
        };
        assert!(six.holds(IpAddr::from([0x2001, 0xdb8, 0, 0, 9, 9, 9, 9])));
        assert!(!six.holds(IpAddr::from([192, 168, 0, 5])));
    }
}
