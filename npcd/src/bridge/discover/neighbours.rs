//! The member's neighbour table: the addresses on its networks it has already
//! exchanged traffic with, and their hardware addresses. Read from the system's
//! own listing — `arp -a` on Windows, `ip neigh` elsewhere — which asks nothing
//! of any device.

use std::net::Ipv4Addr;

use anyhow::{bail, Result};
use tokio::process::Command;

/// One neighbour: its address and its hardware address, as `aa:bb:cc:dd:ee:ff`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Neighbour {
    pub ip: Ipv4Addr,
    pub hardware: String,
}

/// A hardware address written with any separator, lower case with colons —
/// `F0-F6-C1-A0-79-B4` is `f0:f6:c1:a0:79:b4`. `None` for anything that is not
/// six octets, and for the broadcast and multicast addresses no device has.
pub fn hardware(text: &str) -> Option<String> {
    let octets: Vec<u8> = text
        .split([':', '-'])
        .map(|o| u8::from_str_radix(o, 16).ok().filter(|_| o.len() == 2))
        .collect::<Option<Vec<u8>>>()?;
    if octets.len() != 6 || octets[0] & 1 == 1 {
        return None;
    }
    Some(
        octets
            .iter()
            .map(|o| format!("{o:02x}"))
            .collect::<Vec<_>>()
            .join(":"),
    )
}

/// The neighbours in `arp -a`'s listing (Windows): `  <ip>  <hw>  dynamic`
/// rows, the multicast and broadcast entries among them dropped.
pub fn parse_arp(text: &str) -> Vec<Neighbour> {
    text.lines()
        .filter_map(|line| {
            let mut cols = line.split_whitespace();
            let ip: Ipv4Addr = cols.next()?.parse().ok()?;
            let hardware = hardware(cols.next()?)?;
            Some(Neighbour { ip, hardware })
        })
        .collect()
}

/// The neighbours in `ip neigh`'s listing (Linux):
/// `<ip> dev <if> lladdr <hw> <state>`, the failed and incomplete ones dropped.
pub fn parse_ip_neigh(text: &str) -> Vec<Neighbour> {
    text.lines()
        .filter_map(|line| {
            let cols: Vec<&str> = line.split_whitespace().collect();
            let ip: Ipv4Addr = cols.first()?.parse().ok()?;
            let at = cols.iter().position(|c| *c == "lladdr")?;
            let hardware = hardware(cols.get(at + 1)?)?;
            Some(Neighbour { ip, hardware })
        })
        .collect()
}

/// This member's neighbour table, read now.
pub async fn read() -> Result<Vec<Neighbour>> {
    let (program, arg) = match cfg!(windows) {
        true => ("arp", "-a"),
        false => ("ip", "neigh"),
    };
    let out = Command::new(program).arg(arg).output().await?;
    if !out.status.success() {
        bail!("`{program}` exited {}", out.status);
    }
    let listing = String::from_utf8_lossy(&out.stdout);
    Ok(match cfg!(windows) {
        true => parse_arp(&listing),
        false => parse_ip_neigh(&listing),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(ip: &str, hw: &str) -> Neighbour {
        Neighbour {
            ip: ip.parse().unwrap(),
            hardware: hw.into(),
        }
    }

    /// **`arp -a` reads as its neighbours**, the multicast and broadcast rows
    /// and the interface headers left out.
    #[test]
    fn arp_a_reads_as_its_neighbours() {
        let listing = "\r\nInterface: 192.168.0.237 --- 0xe\r\n  Internet Address      Physical Address      Type\r\n  192.168.0.1           d8-d8-e5-9c-fe-35     dynamic   \r\n  192.168.0.89          f0-f6-c1-a0-79-b4     dynamic   \r\n  192.168.0.255         ff-ff-ff-ff-ff-ff     static    \r\n  224.0.0.251           01-00-5e-00-00-fb     static    \r\n  239.255.255.250       01-00-5e-7f-ff-fa     static    \r\n";
        assert_eq!(
            parse_arp(listing),
            [
                n("192.168.0.1", "d8:d8:e5:9c:fe:35"),
                n("192.168.0.89", "f0:f6:c1:a0:79:b4")
            ]
        );
    }

    /// **`ip neigh` reads the same**, the failed entries left out.
    #[test]
    fn ip_neigh_reads_as_its_neighbours() {
        let listing = "192.168.0.1 dev eth0 lladdr d8:d8:e5:9c:fe:35 REACHABLE\n192.168.0.50 dev eth0  FAILED\n192.168.0.61 dev eth0 lladdr 00:09:b0:39:af:87 STALE\n";
        assert_eq!(
            parse_ip_neigh(listing),
            [
                n("192.168.0.1", "d8:d8:e5:9c:fe:35"),
                n("192.168.0.61", "00:09:b0:39:af:87")
            ]
        );
    }

    #[test]
    fn a_hardware_address_is_six_octets_of_a_device() {
        assert_eq!(
            hardware("F0-F6-C1-A0-79-B4").as_deref(),
            Some("f0:f6:c1:a0:79:b4")
        );
        assert_eq!(hardware("ff-ff-ff-ff-ff-ff"), None, "broadcast");
        assert_eq!(hardware("01-00-5e-00-00-fb"), None, "multicast");
        assert_eq!(hardware("f0-f6-c1-a0-79"), None);
        assert_eq!(hardware("f0-f6-c1-a0-79-b"), None);
    }
}
