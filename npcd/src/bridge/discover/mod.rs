//! A survey of the house network (`docs/zen_moon_mind.md` §13): every device
//! on the member's own subnets, from what devices openly announce and no
//! further — the neighbour table, the router's names, SSDP and mDNS — merged
//! into one [`Finding`] per device.
//!
//! **It never leaves the house** ([`scope`]): an address not on one of the
//! member's private subnets is dropped before it is a finding, a description
//! is fetched only from one that is, and nothing walks an address range.

pub mod mdns;
pub mod neighbours;
pub mod scope;
pub mod ssdp;

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use anyhow::{bail, Result};
use dns_lookup::lookup_addr;
use futures::future::join_all;
use if_addrs::{get_if_addrs, IfAddr};
use reqwest::redirect::Policy;
use reqwest::Client;
use tokio::task::spawn_blocking;

use super::finding::{profile_of, Finding};
use mdns::Announced;
use neighbours::Neighbour;
use scope::{in_house, local_nets, LocalNet};
use ssdp::{Announce, Description};

/// How long a survey listens for answers to its multicast questions.
const LISTEN: Duration = Duration::from_millis(2500);

/// How long a device is given to serve its description.
const DESCRIBE: Duration = Duration::from_secs(3);

/// What one device answered to the SSDP search, and its description.
pub struct Upnp {
    pub announce: Announce,
    pub description: Option<Description>,
}

/// The router's short name for an address: `SonosZP.modem` is `SonosZP`.
fn short(name: &str) -> String {
    name.split('.').next().unwrap_or(name).to_string()
}

/// One finding per device on `nets`, from what each source said about it.
///
/// A device is one hardware address: the neighbour table says which, and an
/// address it does not hold is no device a survey can name for good. Names run
/// best first — what the device calls itself over mDNS (`fn`), then over UPnP,
/// then the router's name.
pub fn merge(
    nets: &[LocalNet],
    neighbours: &[Neighbour],
    upnp: &BTreeMap<Ipv4Addr, Vec<Upnp>>,
    mdns: &BTreeMap<Ipv4Addr, Announced>,
    names: &BTreeMap<Ipv4Addr, String>,
) -> Vec<Finding> {
    let mut out: Vec<Finding> = neighbours
        .iter()
        .filter(|n| in_house(nets, n.ip))
        .map(|n| {
            let mut f = Finding {
                address: n.ip.to_string(),
                hardware: n.hardware.clone(),
                ..Finding::default()
            };
            let mut name = |s: &str| {
                let s = s.trim();
                if !s.is_empty() && !f.names.iter().any(|x| x == s) {
                    f.names.push(s.to_string());
                }
            };
            if let Some(m) = mdns.get(&n.ip) {
                if let Some(fname) = m.notes.get("fn") {
                    name(fname);
                }
            }
            for u in upnp.get(&n.ip).into_iter().flatten() {
                if let Some(d) = &u.description {
                    name(&d.friendly);
                }
            }
            if let Some(r) = names.get(&n.ip) {
                name(&short(r));
            }
            if let Some(m) = mdns.get(&n.ip) {
                f.services.extend(m.services.iter().cloned());
                if let Some(md) = m.notes.get("md") {
                    f.model = md.clone();
                }
            }
            for u in upnp.get(&n.ip).into_iter().flatten() {
                if let Some(d) = &u.description {
                    if f.maker.is_empty() {
                        f.maker = d.maker.clone();
                    }
                    if f.model.is_empty() {
                        f.model = d.model.clone();
                    }
                    if !d.device_type.is_empty() {
                        f.services.push(d.device_type.clone());
                    }
                }
            }
            f.services.sort();
            f.services.dedup();
            f.profile = profile_of(&f).to_string();
            f
        })
        .collect();
    out.sort_by_key(|f| f.address.parse::<Ipv4Addr>().ok());
    out
}

/// This member's private subnets, from its interfaces.
pub fn nets() -> Vec<LocalNet> {
    let interfaces = get_if_addrs().unwrap_or_default();
    local_nets(interfaces.into_iter().filter_map(|i| match i.addr {
        IfAddr::V4(v4) => Some((v4.ip, v4.netmask)),
        IfAddr::V6(_) => None,
    }))
}

/// Survey the house network, now: every device on the member's own subnets.
pub async fn survey() -> Result<Vec<Finding>> {
    let nets = nets();
    if nets.is_empty() {
        bail!("this member stands on no private network: there is no house to survey");
    }
    // Asked on every subnet at once; each answer is held to the house below.
    let asks = nets.iter().map(|net| {
        let own = net.own;
        spawn_blocking(move || {
            (
                ssdp::search(own, LISTEN).unwrap_or_default(),
                mdns::ask(own, LISTEN).unwrap_or_default(),
            )
        })
    });
    let mut answered: Vec<(Ipv4Addr, Announce)> = Vec::new();
    let mut mdns: BTreeMap<Ipv4Addr, Announced> = BTreeMap::new();
    for ask in asks {
        let (s, m) = ask.await?;
        answered.extend(s);
        mdns.extend(m);
    }
    // Read after the questions, so a device that just answered is in it.
    let neighbours = neighbours::read().await?;

    // **A redirect is never followed**: a description is read from the
    // device's own address or not at all, never from wherever it points.
    let client = Client::builder()
        .timeout(DESCRIBE)
        .redirect(Policy::none())
        .build()?;
    let mut seen: Vec<(Ipv4Addr, String)> = Vec::new();
    let mut asked: Vec<(Ipv4Addr, Announce)> = Vec::new();
    for (from, announce) in answered {
        let first = !seen
            .iter()
            .any(|(f, l)| *f == from && *l == announce.location);
        if in_house(&nets, from) && first {
            seen.push((from, announce.location.clone()));
            asked.push((from, announce));
        }
    }
    // Every description asked for at once, so a survey takes as long as its
    // slowest device rather than all of them.
    let described = join_all(asked.iter().map(|(from, announce)| {
        let client = &client;
        async move {
            // Only from the device itself, on the house network.
            if ssdp::location_host(&announce.location) != Some(*from) {
                return None;
            }
            let reply = client.get(&announce.location).send().await.ok()?;
            if !reply.status().is_success() {
                return None;
            }
            reply.text().await.ok().map(|x| ssdp::parse_description(&x))
        }
    }))
    .await;
    let mut upnp: BTreeMap<Ipv4Addr, Vec<Upnp>> = BTreeMap::new();
    for ((from, announce), description) in asked.into_iter().zip(described) {
        upnp.entry(from).or_default().push(Upnp {
            announce,
            description,
        });
    }

    // Every neighbour's name asked for at once, for the same reason.
    let lookups: Vec<_> = neighbours
        .iter()
        .filter(|n| in_house(&nets, n.ip))
        .map(|n| {
            let ip = n.ip;
            spawn_blocking(move || (ip, lookup_addr(&IpAddr::V4(ip)).ok()))
        })
        .collect();
    let mut names = BTreeMap::new();
    for found in join_all(lookups).await {
        let (ip, name) = found?;
        if let Some(name) = name.filter(|name| name.parse::<IpAddr>().is_err()) {
            names.insert(ip, name);
        }
    }
    Ok(merge(&nets, &neighbours, &upnp, &mdns, &names))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> Ipv4Addr {
        s.parse().unwrap()
    }

    /// **One finding per device on the house network, named best first**: what
    /// it calls itself, then the router's name; its maker, model and services;
    /// the profile that reaches it. A neighbour off the house network is no
    /// finding.
    #[test]
    fn the_sources_merge_into_one_finding_per_device() {
        let nets = [LocalNet {
            own: ip("192.168.0.237"),
            mask: ip("255.255.255.0"),
        }];
        let neighbours = [
            Neighbour {
                ip: ip("192.168.0.89"),
                hardware: "f0:f6:c1:a0:79:b4".into(),
            },
            Neighbour {
                ip: ip("192.168.0.214"),
                hardware: "14:ae:85:71:eb:44".into(),
            },
            Neighbour {
                ip: ip("10.8.0.1"),
                hardware: "aa:aa:aa:aa:aa:aa".into(),
            },
        ];
        let upnp = BTreeMap::from([(
            ip("192.168.0.89"),
            vec![Upnp {
                announce: Announce::default(),
                description: Some(Description {
                    friendly: "192.168.0.89 - Sonos Arc".into(),
                    maker: "Sonos, Inc.".into(),
                    model: "Sonos Arc".into(),
                    device_type: "urn:schemas-upnp-org:device:ZonePlayer:1".into(),
                    services: vec![],
                }),
            }],
        )]);
        let mdns = BTreeMap::from([(
            ip("192.168.0.214"),
            Announced {
                services: vec!["_googlecast._tcp.local".into()],
                notes: BTreeMap::from([
                    ("fn".to_string(), "Family Room TV".to_string()),
                    ("md".to_string(), "Chromecast".to_string()),
                ]),
                ..Announced::default()
            },
        )]);
        let names = BTreeMap::from([(ip("192.168.0.89"), "SonosZP.modem".to_string())]);
        let found = merge(&nets, &neighbours, &upnp, &mdns, &names);
        assert_eq!(
            found,
            [
                Finding {
                    address: "192.168.0.89".into(),
                    hardware: "f0:f6:c1:a0:79:b4".into(),
                    names: vec!["192.168.0.89 - Sonos Arc".into(), "SonosZP".into()],
                    maker: "Sonos, Inc.".into(),
                    model: "Sonos Arc".into(),
                    services: vec!["urn:schemas-upnp-org:device:ZonePlayer:1".into()],
                    profile: "sonos".into(),
                },
                Finding {
                    address: "192.168.0.214".into(),
                    hardware: "14:ae:85:71:eb:44".into(),
                    names: vec!["Family Room TV".into()],
                    maker: String::new(),
                    model: "Chromecast".into(),
                    services: vec!["_googlecast._tcp.local".into()],
                    profile: "cast".into(),
                },
            ]
        );
    }
}
