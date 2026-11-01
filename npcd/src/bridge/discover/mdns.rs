//! mDNS: what announces itself on the house network as a kind of service — a
//! Cast device, a Sonos player, a Spotify Connect receiver, an AirPlay target.
//!
//! One query to the link-local multicast group, which no router forwards, from
//! an ordinary port, so each responder answers this member directly (RFC 6762
//! §6.7). The answers name the device and carry its own key-value notes — a
//! Cast device's `fn` (its friendly name) and `md` (its model).

use std::collections::BTreeMap;
use std::io::{self, ErrorKind};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

/// The mDNS multicast group and port.
const GROUP: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(224, 0, 0, 251), 5353);

/// The service types a survey asks for.
pub const SERVICES: &[&str] = &[
    "_googlecast._tcp.local",
    "_sonos._tcp.local",
    "_spotify-connect._tcp.local",
    "_airplay._tcp.local",
];

const PTR: u16 = 12;
const TXT: u16 = 16;
const SRV: u16 = 33;
const IN: u16 = 1;

/// A name in DNS wire form: length-prefixed labels, a zero to end.
fn name(out: &mut Vec<u8>, dotted: &str) {
    for label in dotted.split('.').filter(|l| !l.is_empty()) {
        out.push(label.len() as u8);
        out.extend_from_slice(label.as_bytes());
    }
    out.push(0);
}

/// The query: one PTR question per service type.
pub fn query(services: &[&str]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 0];
    out.extend_from_slice(&(services.len() as u16).to_be_bytes());
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    for s in services {
        name(&mut out, s);
        out.extend_from_slice(&PTR.to_be_bytes());
        out.extend_from_slice(&IN.to_be_bytes());
    }
    out
}

/// Read a name at `at`, following compression pointers; the name and where
/// the record goes on after it.
fn read_name(packet: &[u8], mut at: usize) -> Option<(String, usize)> {
    let mut labels = Vec::new();
    let mut end = None;
    for _ in 0..128 {
        let len = *packet.get(at)? as usize;
        if len == 0 {
            return Some((labels.join("."), end.unwrap_or(at + 1)));
        }
        if len & 0xc0 == 0xc0 {
            let to = ((len & 0x3f) << 8) | *packet.get(at + 1)? as usize;
            end.get_or_insert(at + 2);
            at = to;
            continue;
        }
        let label = packet.get(at + 1..at + 1 + len)?;
        labels.push(String::from_utf8_lossy(label).into_owned());
        at += 1 + len;
    }
    None
}

fn u16_at(packet: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*packet.get(at)?, *packet.get(at + 1)?]))
}

/// What one responder said about itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Announced {
    /// The service types it offers, from the PTR answers' owners.
    pub services: Vec<String>,
    /// Its instance names — `Living-Room-abc123._googlecast._tcp.local`.
    pub instances: Vec<String>,
    /// The host it names for itself, from an SRV answer.
    pub host: Option<String>,
    /// Its TXT notes, each `key=value`.
    pub notes: BTreeMap<String, String>,
}

/// What a response packet says: PTR, SRV and TXT records, in every section.
pub fn parse(packet: &[u8]) -> Option<Announced> {
    if packet.len() < 12 || packet[2] & 0x80 == 0 {
        return None;
    }
    let count = |i: usize| u16_at(packet, 4 + 2 * i).map(usize::from);
    let questions = count(0)?;
    let records = count(1)? + count(2)? + count(3)?;
    let mut at = 12;
    for _ in 0..questions {
        at = read_name(packet, at)?.1 + 4;
    }
    let mut out = Announced::default();
    for _ in 0..records {
        let (owner, after) = read_name(packet, at)?;
        let kind = u16_at(packet, after)?;
        let len = u16_at(packet, after + 8)? as usize;
        let data = after + 10;
        packet.get(data..data + len)?;
        match kind {
            PTR => {
                if !out.services.contains(&owner) {
                    out.services.push(owner);
                }
                let (instance, _) = read_name(packet, data)?;
                if !out.instances.contains(&instance) {
                    out.instances.push(instance);
                }
            }
            SRV => {
                out.host = Some(read_name(packet, data + 6)?.0);
            }
            TXT => {
                let mut i = data;
                while i < data + len {
                    let n = packet[i] as usize;
                    let note = String::from_utf8_lossy(packet.get(i + 1..i + 1 + n)?);
                    if let Some((k, v)) = note.split_once('=') {
                        out.notes.insert(k.to_string(), v.to_string());
                    }
                    i += 1 + n;
                }
            }
            _ => {}
        }
        at = data + len;
    }
    Some(out)
}

/// Ask from `own`, the member's address on the house network, and gather what
/// each responder says within `wait`, by the address it answered from.
pub fn ask(own: Ipv4Addr, wait: Duration) -> io::Result<BTreeMap<Ipv4Addr, Announced>> {
    let socket = UdpSocket::bind(SocketAddrV4::new(own, 0))?;
    socket.set_multicast_ttl_v4(1)?;
    socket.send_to(&query(SERVICES), GROUP)?;
    let deadline = Instant::now() + wait;
    let mut heard: BTreeMap<Ipv4Addr, Announced> = BTreeMap::new();
    let mut buf = [0u8; 9000];
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        socket.set_read_timeout(Some(left))?;
        match socket.recv_from(&mut buf) {
            Ok((n, SocketAddr::V4(from))) => {
                let Some(said) = parse(&buf[..n]) else {
                    continue;
                };
                let entry = heard.entry(*from.ip()).or_default();
                for s in said.services {
                    if !entry.services.contains(&s) {
                        entry.services.push(s);
                    }
                }
                for i in said.instances {
                    if !entry.instances.contains(&i) {
                        entry.instances.push(i);
                    }
                }
                entry.host = entry.host.take().or(said.host);
                entry.notes.extend(said.notes);
            }
            Ok(_) => {}
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(heard)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The query is one PTR question per service, in DNS wire form.**
    #[test]
    fn the_query_asks_for_each_service() {
        assert_eq!(
            query(&["_sonos._tcp.local"]),
            [
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, //
                6, b'_', b's', b'o', b'n', b'o', b's', //
                4, b'_', b't', b'c', b'p', //
                5, b'l', b'o', b'c', b'a', b'l', 0, //
                0, 12, 0, 1
            ]
        );
    }

    /// A response as a Cast device sends it: a PTR answer for the service, its
    /// instance compressed against it, a TXT with `fn` and `md`, and an SRV.
    fn cast_response() -> Vec<u8> {
        let mut p = vec![0, 0, 0x84, 0, 0, 0, 0, 3, 0, 0, 0, 0];
        // PTR: _googlecast._tcp.local -> Chromecast-ab12._googlecast._tcp.local
        let service_at = p.len();
        name(&mut p, "_googlecast._tcp.local");
        p.extend_from_slice(&[0, 12, 0, 1, 0, 0, 0, 120]);
        let instance = {
            let mut d = vec![15];
            d.extend_from_slice(b"Chromecast-ab12");
            d.extend_from_slice(&[0xc0, service_at as u8]);
            d
        };
        p.extend_from_slice(&(instance.len() as u16).to_be_bytes());
        let instance_at = p.len();
        p.extend_from_slice(&instance);
        // TXT on the instance: fn=Family Room TV, md=Chromecast
        p.extend_from_slice(&[0xc0, instance_at as u8, 0, 16, 0x80, 1, 0, 0, 0, 120]);
        let notes: &[&[u8]] = &[b"fn=Family Room TV", b"md=Chromecast"];
        let len: usize = notes.iter().map(|n| n.len() + 1).sum();
        p.extend_from_slice(&(len as u16).to_be_bytes());
        for n in notes {
            p.push(n.len() as u8);
            p.extend_from_slice(n);
        }
        // SRV on the instance: priority, weight, port 8009, target host
        p.extend_from_slice(&[0xc0, instance_at as u8, 0, 33, 0x80, 1, 0, 0, 0, 120]);
        let mut srv = vec![0, 0, 0, 0, 0x1f, 0x49];
        name(&mut srv, "ab12.local");
        p.extend_from_slice(&(srv.len() as u16).to_be_bytes());
        p.extend_from_slice(&srv);
        p
    }

    /// **A Cast device's answer names its service, instance, host and notes.**
    #[test]
    fn a_cast_answer_says_what_the_device_is() {
        let said = parse(&cast_response()).expect("a response");
        assert_eq!(said.services, ["_googlecast._tcp.local"]);
        assert_eq!(said.instances, ["Chromecast-ab12._googlecast._tcp.local"]);
        assert_eq!(said.host.as_deref(), Some("ab12.local"));
        assert_eq!(said.notes["fn"], "Family Room TV");
        assert_eq!(said.notes["md"], "Chromecast");
        assert_eq!(parse(&query(SERVICES)), None, "a query is no answer");
        assert_eq!(parse(&[0, 0, 0x84]), None, "too short");
    }
}
