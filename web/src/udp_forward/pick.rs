//! Which member a client's flow is given to (`docs/battlecities_app.md` §5):
//! a hash of the client's address and port over the members that serve, so
//! players spread over them and the same client lands on the same member
//! while the members stand.

use std::net::{IpAddr, SocketAddr};

/// FNV-1a's offset basis and prime, 64-bit.
const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

fn fold(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, b| (h ^ u64::from(*b)).wrapping_mul(PRIME))
}

/// The hash of `client`: FNV-1a over its address's bytes, then its port's,
/// big end first. The same on every gateway and every build.
pub fn hash(client: SocketAddr) -> u64 {
    let of_ip = match client.ip() {
        IpAddr::V4(ip) => fold(BASIS, &ip.octets()),
        IpAddr::V6(ip) => fold(BASIS, &ip.octets()),
    };
    fold(of_ip, &client.port().to_be_bytes())
}

/// The member of `serving` — each a node and the address its machine is
/// reached at, in node order — that `client`'s flow is given to. A member
/// whose address is not known is not one a datagram can be sent to.
pub fn pick(serving: &[(u64, IpAddr)], client: SocketAddr) -> Option<(u64, IpAddr)> {
    let reachable: Vec<(u64, IpAddr)> = serving
        .iter()
        .copied()
        .filter(|(_, at)| !at.is_unspecified())
        .collect();
    let n = u64::try_from(reachable.len()).ok().filter(|n| *n > 0)?;
    reachable.get((hash(client) % n) as usize).copied()
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;

    fn client(last: u8, port: u16) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::new(203, 0, 113, last), port))
    }

    fn at(last: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, last))
    }

    /// **The hash is FNV-1a**, to the bit: these are its values for the six
    /// bytes of each address and port.
    #[test]
    fn a_clients_hash_is_fnv1a_of_its_address_and_port() {
        // FNV-1a of no bytes is the offset basis; of the one byte 0x61, the
        // published 0xaf63dc4c8601ec8c.
        assert_eq!(fold(BASIS, &[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fold(BASIS, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(
            hash(client(9, 40_000)),
            fold(BASIS, &[203, 0, 113, 9, 0x9c, 0x40])
        );
        assert_ne!(hash(client(9, 40_000)), hash(client(9, 40_001)));
        assert_ne!(hash(client(9, 40_000)), hash(client(10, 40_000)));
    }

    /// **The same client lands on the same member, and clients spread**: of
    /// a thousand ports from one address, each of three members takes a
    /// share.
    #[test]
    fn clients_spread_over_the_members_and_stay_where_they_land() {
        let serving = [(1, at(3)), (2, at(4)), (3, at(5))];
        let first = pick(&serving, client(9, 40_000));
        assert_eq!(first, pick(&serving, client(9, 40_000)));
        let mut taken = [0usize; 3];
        for port in 40_000..41_000 {
            let (node, _) = pick(&serving, client(9, port)).unwrap();
            taken[(node - 1) as usize] += 1;
        }
        assert_eq!(taken.iter().sum::<usize>(), 1_000);
        assert!(taken.iter().all(|n| *n > 200), "{taken:?}");
    }

    /// One member takes every flow; none takes none; a member whose address
    /// is not known is passed over.
    #[test]
    fn a_flow_goes_only_to_a_member_that_can_be_reached() {
        assert_eq!(pick(&[], client(9, 1)), None);
        assert_eq!(pick(&[(7, at(3))], client(9, 1)), Some((7, at(3))));
        let unknown = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
        assert_eq!(pick(&[(7, unknown)], client(9, 1)), None);
        assert_eq!(
            pick(&[(7, unknown), (8, at(4))], client(9, 1)),
            Some((8, at(4)))
        );
    }
}
