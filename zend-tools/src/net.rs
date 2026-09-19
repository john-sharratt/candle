//! Every outbound network primitive a tool may use, each refused without
//! [`Capability::Network`].
//!
//! Tools do not open sockets, resolve names or build HTTP clients themselves;
//! they call these. That makes the network permission a property of the
//! primitive rather than of each tool's declaration — see [`crate::grants`] —
//! and a test in this module holds tool code to it.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::io;
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::Duration;

use reqwest::blocking::{Client, ClientBuilder};

use crate::grants::{Capability, Grants, NotPermitted};

/// Why a network primitive did not produce a connection.
#[derive(Debug)]
pub enum NetError {
    /// The context holds no [`Capability::Network`]; nothing was attempted.
    NotPermitted(NotPermitted),
    /// A name resolved to no address.
    Unresolved(String),
    Io(io::Error),
}

impl Display for NetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NetError::NotPermitted(e) => e.fmt(f),
            NetError::Unresolved(addr) => write!(f, "{addr} resolved to no address"),
            NetError::Io(e) => e.fmt(f),
        }
    }
}

impl Error for NetError {}

impl From<NotPermitted> for NetError {
    fn from(e: NotPermitted) -> Self {
        NetError::NotPermitted(e)
    }
}

fn permit(grants: Grants) -> Result<(), NetError> {
    grants.require(Capability::Network).map_err(NetError::from)
}

/// Resolve `host:port` (or a literal socket address) to its first address.
pub fn resolve(grants: Grants, addr: &str) -> Result<SocketAddr, NetError> {
    permit(grants)?;
    if let Ok(sa) = addr.parse::<SocketAddr>() {
        return Ok(sa);
    }
    addr.to_socket_addrs()
        .map_err(NetError::Io)?
        .next()
        .ok_or_else(|| NetError::Unresolved(addr.to_string()))
}

/// Connect a TCP stream to `addr`, bounded by `timeout` when given.
pub fn tcp_connect(
    grants: Grants,
    addr: &SocketAddr,
    timeout: Option<Duration>,
) -> Result<TcpStream, NetError> {
    permit(grants)?;
    match timeout {
        Some(t) => TcpStream::connect_timeout(addr, t),
        None => TcpStream::connect(addr),
    }
    .map_err(NetError::Io)
}

/// Resolve `host:port` and connect a TCP stream to it.
pub fn tcp_connect_to(
    grants: Grants,
    addr: &str,
    timeout: Option<Duration>,
) -> Result<TcpStream, NetError> {
    let sa = resolve(grants, addr)?;
    tcp_connect(grants, &sa, timeout)
}

/// Bind a UDP socket at `addr`.
pub fn udp_bind(grants: Grants, addr: &str) -> Result<UdpSocket, NetError> {
    permit(grants)?;
    UdpSocket::bind(addr).map_err(NetError::Io)
}

/// Every address `host` resolves to.
pub fn lookup_host(grants: Grants, host: &str) -> Result<Vec<IpAddr>, NetError> {
    permit(grants)?;
    dns_lookup::lookup_host(host).map_err(NetError::Io)
}

/// The name `ip` resolves back to.
pub fn lookup_addr(grants: Grants, ip: &IpAddr) -> Result<String, NetError> {
    permit(grants)?;
    dns_lookup::lookup_addr(ip).map_err(NetError::Io)
}

/// A builder for an HTTP client of the tool's own configuration.
pub fn http_client_builder(grants: Grants) -> Result<ClientBuilder, NotPermitted> {
    grants.require(Capability::Network)?;
    Ok(Client::builder())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_scan::tool_sources_containing;

    /// **Without the capability nothing is attempted** — not even resolution,
    /// which is itself a request to the network.
    #[test]
    fn every_primitive_refuses_without_the_network_capability() {
        let none = Grants::NONE;
        let denied =
            |e: NetError| matches!(e, NetError::NotPermitted(NotPermitted(Capability::Network)));
        assert!(denied(resolve(none, "127.0.0.1:1").unwrap_err()));
        assert!(denied(
            tcp_connect(none, &"127.0.0.1:1".parse().unwrap(), None).unwrap_err()
        ));
        assert!(denied(
            tcp_connect_to(none, "localhost:1", None).unwrap_err()
        ));
        assert!(denied(udp_bind(none, "127.0.0.1:0").unwrap_err()));
        assert!(denied(lookup_host(none, "localhost").unwrap_err()));
        assert!(denied(
            lookup_addr(none, &"127.0.0.1".parse().unwrap()).unwrap_err()
        ));
        assert!(http_client_builder(none).is_err());
    }

    /// With it, the primitives work — a loopback bind needs no outside network.
    #[test]
    fn the_capability_lets_the_primitive_run() {
        let net = Grants::NONE.with(Capability::Network);
        assert!(udp_bind(net, "127.0.0.1:0").is_ok());
        assert_eq!(
            resolve(net, "127.0.0.1:80").unwrap(),
            "127.0.0.1:80".parse().unwrap()
        );
        assert!(http_client_builder(net).is_ok());
    }

    /// **Tool code reaches the network only through this module.** Scans every
    /// file under `src/tools` for the raw constructors and fails naming the
    /// file, so a new tool that opens a socket directly cannot slip past the
    /// capability check.
    #[test]
    fn tools_reach_the_network_only_through_this_module() {
        // A TLS or SSH session is layered on a stream from `tcp_connect`, so
        // the handshake itself is not a raw primitive — the stream under it is.
        const RAW: [&str; 9] = [
            "TcpStream::connect",
            "TcpListener::bind",
            "UdpSocket::bind",
            "to_socket_addrs",
            "dns_lookup::lookup",
            "Client::builder",
            "Client::new",
            "ClientBuilder::new",
            "reqwest::blocking::get",
        ];
        let offenders = tool_sources_containing(&RAW);
        assert!(
            offenders.is_empty(),
            "tool code opening the network directly: {offenders:#?}"
        );
    }
}
