//! One `--cluster`, parsed: a gateway and the URLs it is reached by, in the
//! order they are tried (`docs/replicated_substrate.md` §3.2, paths).

use std::net::IpAddr;

/// One URL of a cluster: where to dial, and the name its certificate is
/// verified against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    /// As given, for display.
    pub url: String,
    pub host: String,
    pub port: u16,
    /// The name its certificate is verified against; `None` for a plain
    /// loopback endpoint.
    pub tls_name: Option<String>,
}

/// A cluster: one gateway, by every endpoint it is reached through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// As given, for display: the endpoints' URLs, space-separated.
    pub url: String,
    /// Best first. Never empty.
    pub endpoints: Vec<Endpoint>,
}

fn is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

impl Endpoint {
    /// `url` is `https://<host>[:port]` or, to loopback, `http://…`; a path is
    /// ignored. `name` is `--cluster-name`.
    pub fn parse(url: &str, name: Option<&str>) -> anyhow::Result<Endpoint> {
        let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = url.strip_prefix("http://") {
            (false, r)
        } else {
            anyhow::bail!("--cluster {url}: expected an https:// URL");
        };
        let authority = rest.split('/').next().unwrap_or_default();
        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let (h, after) = v6
                .split_once(']')
                .ok_or_else(|| anyhow::anyhow!("--cluster {url}: an unclosed IPv6 address"))?;
            (h.to_string(), after.strip_prefix(':'))
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), Some(p)),
                None => (authority.to_string(), None),
            }
        };
        if host.is_empty() {
            anyhow::bail!("--cluster {url}: no host");
        }
        let port = match port {
            Some(p) => p
                .parse()
                .map_err(|_| anyhow::anyhow!("--cluster {url}: port `{p}` is not a port"))?,
            None if tls => 443,
            None => 80,
        };
        if !tls && !is_loopback(&host) {
            anyhow::bail!(
                "--cluster {url}: a cluster is dialled over https, verified; plain http is \
                 for loopback only"
            );
        }
        Ok(Endpoint {
            url: url.to_string(),
            tls_name: tls.then(|| name.map_or_else(|| host.clone(), str::to_string)),
            host,
            port,
        })
    }

    /// The socket for `app` at this endpoint.
    pub fn socket_url(&self, app: &str) -> String {
        let scheme = if self.tls_name.is_some() { "wss" } else { "ws" };
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("{scheme}://{host}:{}/v1/cluster/{app}", self.port)
    }
}

impl Target {
    /// `spec` is one or more endpoint URLs separated by whitespace, best
    /// first. `name` is `--cluster-name`, verifying every one of them.
    pub fn parse(spec: &str, name: Option<&str>) -> anyhow::Result<Target> {
        let endpoints = spec
            .split_whitespace()
            .map(|u| Endpoint::parse(u, name))
            .collect::<anyhow::Result<Vec<_>>>()?;
        if endpoints.is_empty() {
            anyhow::bail!("--cluster `{spec}`: no URL");
        }
        Ok(Target {
            url: endpoints
                .iter()
                .map(|e| e.url.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            endpoints,
        })
    }

    /// Whether every endpoint is verified TLS.
    pub fn secure(&self) -> bool {
        self.endpoints.iter().all(|e| e.tls_name.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_endpoints_verify_their_name_and_http_is_loopback_only() {
        let t = Endpoint::parse("https://code.tokera.com", None).unwrap();
        assert_eq!(
            (t.host.as_str(), t.port, t.tls_name.as_deref()),
            ("code.tokera.com", 443, Some("code.tokera.com"))
        );
        assert_eq!(
            t.socket_url("zend"),
            "wss://code.tokera.com:443/v1/cluster/zend"
        );

        // An address is verified by `--cluster-name`.
        let lan = Endpoint::parse("https://192.168.0.5:9443/", Some("code.tokera.com")).unwrap();
        assert_eq!(
            (lan.host.as_str(), lan.port, lan.tls_name.as_deref()),
            ("192.168.0.5", 9443, Some("code.tokera.com"))
        );
        assert_eq!(
            lan.socket_url("npcd"),
            "wss://192.168.0.5:9443/v1/cluster/npcd"
        );

        let local = Endpoint::parse("http://127.0.0.1:8080", None).unwrap();
        assert_eq!((local.port, local.tls_name.clone()), (8080, None));
        assert_eq!(
            local.socket_url("zend"),
            "ws://127.0.0.1:8080/v1/cluster/zend"
        );
        let v6 = Endpoint::parse("http://[::1]:8080", None).unwrap();
        assert_eq!(v6.socket_url("agent"), "ws://[::1]:8080/v1/cluster/agent");

        for bad in [
            "http://192.168.0.5:8080",
            "ws://127.0.0.1",
            "code.tokera.com",
            "https://:443",
            "https://host:port",
        ] {
            assert!(Endpoint::parse(bad, None).is_err(), "{bad}");
        }
    }

    /// A cluster is its URLs best first, each verified by `--cluster-name`;
    /// one URL is a cluster of one endpoint.
    #[test]
    fn a_cluster_is_its_endpoints_in_order() {
        let t = Target::parse(
            "https://gateway.tokera.local:9443  https://gateway.tokera.com:9443",
            Some("tokera.com"),
        )
        .unwrap();
        assert_eq!(
            t.url,
            "https://gateway.tokera.local:9443 https://gateway.tokera.com:9443"
        );
        let hosts: Vec<(&str, u16, Option<&str>)> = t
            .endpoints
            .iter()
            .map(|e| (e.host.as_str(), e.port, e.tls_name.as_deref()))
            .collect();
        assert_eq!(
            hosts,
            [
                ("gateway.tokera.local", 9443, Some("tokera.com")),
                ("gateway.tokera.com", 9443, Some("tokera.com")),
            ]
        );
        assert!(t.secure());

        let one = Target::parse("https://code.tokera.com", None).unwrap();
        assert_eq!(one.endpoints.len(), 1);
        assert_eq!(one.url, "https://code.tokera.com");

        let mixed = Target::parse("http://127.0.0.1:8080 https://code.tokera.com", None).unwrap();
        assert!(!mixed.secure());

        assert!(Target::parse("   ", None).is_err());
        // One bad URL refuses the cluster.
        assert!(Target::parse("https://code.tokera.com http://10.0.0.1", None).is_err());
    }
}
