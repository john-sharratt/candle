//! Which addresses a fetch may reach: the public internet, and nothing on
//! this host or the local network.

use std::net::{IpAddr, SocketAddr};

use super::error::FetchError;
use crate::grants::Grants;
use crate::net;

/// Whether `url_str` names this host or a private address literally, or is no
/// URL at all. A name is judged when it is resolved ([`public_addresses`]).
pub fn is_private_url(url_str: &str) -> bool {
    let Ok(parsed) = url::Url::parse(url_str) else {
        return true;
    };
    let host = parsed.host_str().unwrap_or("");
    if host == "localhost" || host == "127.0.0.1" || host == "::1" {
        return true;
    }
    if let Ok(ip) = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
    {
        return is_private_ip(ip);
    }
    false
}

/// Whether `ip` addresses this host, the local network, or anything else that
/// is not the public internet — the addresses a fetch may not reach.
pub fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                // 0.0.0.0/8 — "this network"; reaches the host itself.
                || a == 0
                // 100.64.0.0/10 — carrier-grade NAT, a private range.
                || (a == 100 && (64..128).contains(&b))
        }
        IpAddr::V6(v6) => {
            // An IPv4 address carried in IPv6 is judged as the IPv4 address it
            // is: `::ffff:127.0.0.1` is loopback, whatever family it arrives in.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_private_ip(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            v6.is_loopback()
                || v6.is_unspecified()
                // fc00::/7 — unique local.
                || (first & 0xfe00) == 0xfc00
                // fe80::/10 — link local.
                || (first & 0xffc0) == 0xfe80
        }
    }
}

/// The public address to fetch `host:port` from: resolved here, refused when
/// any address it resolves to is private.
///
/// Every address is checked, not the first, because the connection may use any
/// of them; and the caller pins the fetch to the address returned, so the host
/// cannot answer this check with a public address and the connection with a
/// private one.
pub(super) fn public_address(
    grants: Grants,
    host: &str,
    port: u16,
) -> Result<SocketAddr, FetchError> {
    let ips = public_addresses(grants, host)?;
    Ok(SocketAddr::new(ips[0], port))
}

/// Every address `host` resolves to — never empty — refused when any is
/// private.
pub(super) fn public_addresses(grants: Grants, host: &str) -> Result<Vec<IpAddr>, FetchError> {
    let ips = match host.parse::<IpAddr>() {
        Ok(ip) => vec![ip],
        Err(_) => net::lookup_host(grants, host)
            .map_err(|e| FetchError::FetchFailed(format!("{host}: {e}")))?,
    };
    if let Some(ip) = ips.iter().find(|ip| is_private_ip(**ip)) {
        return Err(FetchError::UrlBlocked(format!(
            "{host} resolves to {ip}, a private or local address"
        )));
    }
    if ips.is_empty() {
        return Err(FetchError::FetchFailed(format!(
            "{host} resolved to no address"
        )));
    }
    Ok(ips)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grants::Capability;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    /// Every spelling of "this host or the local network" is private,
    /// including an IPv4 address carried in IPv6.
    #[test]
    fn every_local_range_is_private() {
        for s in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.0.5",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "0.1.2.3",
            "255.255.255.255",
            "::1",
            "::",
            "::ffff:127.0.0.1",
            "::ffff:192.168.0.5",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
        ] {
            assert!(is_private_ip(ip(s)), "{s} was treated as public");
        }
        for s in ["8.8.8.8", "1.1.1.1", "2606:4700::1111", "100.128.0.1"] {
            assert!(!is_private_ip(ip(s)), "{s} was treated as private");
        }
    }

    /// **A name that resolves to a private address is refused**, not only a
    /// literal one — `localhost` is the name every resolver answers locally.
    #[test]
    fn a_name_resolving_to_a_private_address_is_refused() {
        let net = Grants::NONE.with(Capability::Network);
        assert!(matches!(
            public_addresses(net, "localhost"),
            Err(FetchError::UrlBlocked(_))
        ));
        assert!(matches!(
            public_addresses(net, "192.168.0.5"),
            Err(FetchError::UrlBlocked(_))
        ));
        assert_eq!(public_addresses(net, "8.8.8.8").unwrap(), [ip("8.8.8.8")]);
    }
}
