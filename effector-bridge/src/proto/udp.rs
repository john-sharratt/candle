//! A UDP broadcast on the house's own subnets, and the magic packet that wakes
//! a sleeping machine — the whole of wake-on-LAN's wire.
//!
//! A magic packet is six `0xFF` bytes, then the card's six-byte hardware
//! address sixteen times over. It is a broadcast on the LAN and goes nowhere
//! else: there is no address to reach beyond the house.

use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};

use anyhow::{bail, Result};

use crate::scope::LocalNet;

/// The discard port a magic packet is conventionally sent to; any works, since
/// the card wakes on the frame, not the port.
pub const WAKE_PORT: u16 = 9;

/// The six bytes of a hardware address written `aa:bb:cc:dd:ee:ff` or with `-`
/// — or `None` when it is not six hex octets.
pub fn mac_bytes(mac: &str) -> Option<[u8; 6]> {
    let octets: Vec<u8> = mac
        .split([':', '-'])
        .filter_map(|o| u8::from_str_radix(o.trim(), 16).ok())
        .collect();
    (octets.len() == 6).then(|| {
        let mut out = [0u8; 6];
        out.copy_from_slice(&octets);
        out
    })
}

/// The 102-byte magic packet that wakes the card with hardware address `mac`:
/// `0xFF` six times, then the address sixteen times.
pub fn magic(mac: &str) -> Option<[u8; 102]> {
    let addr = mac_bytes(mac)?;
    let mut packet = [0xFFu8; 102];
    for i in 0..16 {
        packet[6 + i * 6..6 + i * 6 + 6].copy_from_slice(&addr);
    }
    Some(packet)
}

/// A subnet's broadcast address — every host bit set.
pub fn broadcast_addr(net: &LocalNet) -> Ipv4Addr {
    Ipv4Addr::from(u32::from(net.own) | !u32::from(net.mask))
}

/// Broadcast `packet` on each of `nets` at `port` — the house's own subnets,
/// nowhere else. Returns how many subnets it went out on.
pub fn broadcast(packet: &[u8], nets: &[LocalNet], port: u16) -> Result<usize> {
    if nets.is_empty() {
        bail!("this member stands on no house network to broadcast on");
    }
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    socket.set_broadcast(true)?;
    let mut sent = 0;
    for net in nets {
        let to = SocketAddrV4::new(broadcast_addr(net), port);
        if socket.send_to(packet, to).is_ok() {
            sent += 1;
        }
    }
    if sent == 0 {
        bail!("the broadcast could not be sent on any house network");
    }
    Ok(sent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A magic packet is six 0xFF bytes then the address sixteen times** —
    /// 102 bytes — and an address reads with colons or dashes.
    #[test]
    fn a_magic_packet_is_built_to_the_letter() {
        assert_eq!(
            mac_bytes("f0:f6:c1:a0:79:b4"),
            Some([0xf0, 0xf6, 0xc1, 0xa0, 0x79, 0xb4])
        );
        assert_eq!(
            mac_bytes("F0-F6-C1-A0-79-B4"),
            Some([0xf0, 0xf6, 0xc1, 0xa0, 0x79, 0xb4])
        );
        assert_eq!(mac_bytes("f0:f6:c1:a0:79"), None, "five octets");
        assert_eq!(mac_bytes("nope"), None);
        let p = magic("01:02:03:04:05:06").unwrap();
        assert_eq!(&p[..6], &[0xff; 6]);
        for i in 0..16 {
            assert_eq!(&p[6 + i * 6..6 + i * 6 + 6], &[1, 2, 3, 4, 5, 6]);
        }
        assert_eq!(p.len(), 102);
        assert!(magic("bad").is_none());
    }

    /// **The broadcast address sets every host bit** of the subnet.
    #[test]
    fn the_broadcast_address_is_the_subnets_last() {
        let net = LocalNet {
            own: "192.168.0.237".parse().unwrap(),
            mask: "255.255.255.0".parse().unwrap(),
        };
        assert_eq!(
            broadcast_addr(&net),
            "192.168.0.255".parse::<Ipv4Addr>().unwrap()
        );
    }

    #[test]
    fn with_no_house_network_it_is_refused() {
        assert!(broadcast(&magic("01:02:03:04:05:06").unwrap(), &[], WAKE_PORT).is_err());
    }
}
