//! SSDP: what UPnP devices on the house network say they are.
//!
//! One `M-SEARCH` to the link-local multicast group, which no router forwards,
//! answered by each UPnP device with where its own description is; the
//! description — an XML file the device serves for exactly this — names its
//! maker, model and services.

use std::io::{self, ErrorKind};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

/// The SSDP multicast group and port.
const GROUP: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(239, 255, 255, 250), 1900);

/// The search: every device, answering within two seconds.
pub fn request() -> String {
    "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\nST: ssdp:all\r\n\r\n".to_string()
}

/// One device's answer to the search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Announce {
    /// Where its description is.
    pub location: String,
    /// What it says it runs on — `Linux UPnP/1.0 Sonos/80.1`.
    pub server: String,
    /// What this answer is for — a device or service type.
    pub st: String,
}

/// A header's value, matched case-blind.
fn header<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim().eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

/// A search answer, if `text` is one: an `HTTP/1.1 200` with a `LOCATION`.
pub fn parse_response(text: &str) -> Option<Announce> {
    if !text.starts_with("HTTP/1.1 200") {
        return None;
    }
    Some(Announce {
        location: header(text, "location")?.to_string(),
        server: header(text, "server").unwrap_or_default().to_string(),
        st: header(text, "st").unwrap_or_default().to_string(),
    })
}

/// What a device's description says it is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Description {
    pub friendly: String,
    pub maker: String,
    pub model: String,
    pub device_type: String,
    pub services: Vec<String>,
}

/// The text of the first `<tag>` in `xml`.
fn tag<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&format!("</{tag}>"))?;
    Some(xml[start..start + end].trim())
}

/// Every `<tag>` text in `xml`, in order.
fn tags<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(at) = rest.find(&open) {
        let after = &rest[at + open.len()..];
        let Some(end) = after.find(&close) else { break };
        out.push(after[..end].trim());
        rest = &after[end + close.len()..];
    }
    out
}

/// What a UPnP description says the device is — its first `<device>`.
pub fn parse_description(xml: &str) -> Description {
    let get = |t: &str| tag(xml, t).unwrap_or_default().to_string();
    let mut services: Vec<String> = tags(xml, "serviceType")
        .into_iter()
        .map(str::to_string)
        .collect();
    services.dedup();
    Description {
        friendly: get("friendlyName"),
        maker: get("manufacturer"),
        model: get("modelName"),
        device_type: get("deviceType"),
        services,
    }
}

/// The address and port a `LOCATION` URL names, if it is `http://a.b.c.d:port/`.
pub fn location_host(location: &str) -> Option<Ipv4Addr> {
    let rest = location.strip_prefix("http://")?;
    let host = rest.split(['/', ':']).next()?;
    host.parse().ok()
}

/// Search from `own`, the member's address on the house network, and gather
/// every answer that arrives within `wait`: each with the address it came from.
pub fn search(own: Ipv4Addr, wait: Duration) -> io::Result<Vec<(Ipv4Addr, Announce)>> {
    let socket = UdpSocket::bind(SocketAddrV4::new(own, 0))?;
    socket.set_multicast_ttl_v4(1)?;
    socket.send_to(request().as_bytes(), GROUP)?;
    let deadline = Instant::now() + wait;
    let mut answers = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        socket.set_read_timeout(Some(left))?;
        match socket.recv_from(&mut buf) {
            Ok((n, SocketAddr::V4(from))) => {
                if let Some(a) = parse_response(&String::from_utf8_lossy(&buf[..n])) {
                    answers.push((*from.ip(), a));
                }
            }
            Ok(_) => {}
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(answers)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A search answer names where the description is**; anything else is
    /// not an answer.
    #[test]
    fn an_answer_names_its_description() {
        let text = "HTTP/1.1 200 OK\r\nCACHE-CONTROL: max-age = 1800\r\nLOCATION: http://192.168.0.89:1400/xml/device_description.xml\r\nSERVER: Linux UPnP/1.0 Sonos/80.1-55014 (ZPS27)\r\nST: urn:schemas-upnp-org:device:ZonePlayer:1\r\n\r\n";
        assert_eq!(
            parse_response(text),
            Some(Announce {
                location: "http://192.168.0.89:1400/xml/device_description.xml".into(),
                server: "Linux UPnP/1.0 Sonos/80.1-55014 (ZPS27)".into(),
                st: "urn:schemas-upnp-org:device:ZonePlayer:1".into(),
            })
        );
        assert_eq!(
            parse_response(&request()),
            None,
            "a search is not an answer"
        );
        assert_eq!(
            location_host("http://192.168.0.89:1400/xml/device_description.xml"),
            Some(Ipv4Addr::new(192, 168, 0, 89))
        );
        assert_eq!(location_host("https://example.com/x.xml"), None);
    }

    /// **A description says the device's name, maker, model and services.**
    #[test]
    fn a_description_says_what_the_device_is() {
        let xml = r#"<?xml version="1.0"?><root><device><deviceType>urn:schemas-upnp-org:device:ZonePlayer:1</deviceType><friendlyName>192.168.0.89 - Sonos Arc</friendlyName><manufacturer>Sonos, Inc.</manufacturer><modelName>Sonos Arc</modelName><serviceList><service><serviceType>urn:schemas-upnp-org:service:AlarmClock:1</serviceType></service><service><serviceType>urn:schemas-upnp-org:service:AVTransport:1</serviceType></service></serviceList></device></root>"#;
        assert_eq!(
            parse_description(xml),
            Description {
                friendly: "192.168.0.89 - Sonos Arc".into(),
                maker: "Sonos, Inc.".into(),
                model: "Sonos Arc".into(),
                device_type: "urn:schemas-upnp-org:device:ZonePlayer:1".into(),
                services: vec![
                    "urn:schemas-upnp-org:service:AlarmClock:1".into(),
                    "urn:schemas-upnp-org:service:AVTransport:1".into()
                ],
            }
        );
        assert_eq!(parse_description("<root/>"), Description::default());
    }
}
