//! Clients named by a Client ID Metadata Document (CIMD): the `client_id` is
//! an `https` URL, and the document fetched from it lists the redirects the
//! client may be sent to (`docs/claude_zend_gateway.md` §12).
//!
//! A redirect is allowed when the document lists it exactly, or — for a
//! loopback redirect, which Claude Code opens on a random port — when it
//! differs from a listed loopback redirect only in its port (RFC 8252 §7.3).
//!
//! **The gateway fetches a URL a caller names**, so the fetch is fenced: it is
//! made only for a signed-in user (`super::authorize`), only to a host every
//! address of which is public ([`public`]) — pinned to the address checked, so
//! the name cannot be re-pointed between the check and the fetch — and read to
//! at most [`MAX_DOCUMENT`] bytes as it arrives. Its failures say that the
//! document could not be read and nothing about what answered.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use reqwest::redirect::Policy;
use reqwest::{Client, Url};
use serde::Deserialize;
use tokio::net::lookup_host;

/// How long a client's document may take to answer.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
/// The largest document read.
const MAX_DOCUMENT: usize = 64 * 1024;

/// What a client's document says about it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ClientDoc {
    pub client_id: String,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub redirect_uris: Vec<String>,
}

impl ClientDoc {
    /// What the consent page calls the client.
    pub fn display_name(&self) -> String {
        self.client_name
            .clone()
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| self.client_id.clone())
    }
}

/// Refuse a `client_id` that cannot name a metadata document: it is an
/// `https` URL with a path, as CIMD requires.
pub fn check_id(client_id: &str) -> Result<Url, String> {
    let url =
        Url::parse(client_id).map_err(|_| format!("the client id `{client_id}` is not a URL"))?;
    if url.scheme() != "https" {
        return Err(format!("the client id `{client_id}` is not an https URL"));
    }
    if url.path().is_empty() || url.path() == "/" {
        return Err(format!("the client id `{client_id}` names no document"));
    }
    if url.fragment().is_some() || !url.username().is_empty() || url.password().is_some() {
        return Err(format!("the client id `{client_id}` is not a plain URL"));
    }
    Ok(url)
}

/// Whether `ip` is on the public internet: none of loopback, private,
/// link-local, shared (CGNAT), unique-local, multicast, unspecified,
/// documentation or broadcast space — nor an IPv4 address carried in IPv6.
pub fn public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast()
                || v4.is_documentation()
                || a == 0
                || (a == 100 && (64..128).contains(&b)))
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return public(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || first == 0x2001 && v6.segments()[1] == 0x0db8)
        }
    }
}

/// The one public address `url`'s host resolves to, refusing a host any of
/// whose addresses is not public.
async fn public_address(url: &Url) -> Result<SocketAddr, String> {
    let refused = || "the client's document is not on a public host".to_string();
    let host = url.host_str().ok_or_else(refused)?;
    let port = url.port_or_known_default().ok_or_else(refused)?;
    let addrs: Vec<SocketAddr> = lookup_host((host, port))
        .await
        .map_err(|_| refused())?
        .collect();
    if addrs.is_empty() || !addrs.iter().all(|a| public(a.ip())) {
        return Err(refused());
    }
    Ok(addrs[0])
}

/// Fetch and read the document `client_id` names.
pub async fn fetch(client_id: &str) -> Result<ClientDoc, String> {
    let unreadable = || "the client's document could not be read".to_string();
    let url = check_id(client_id)?;
    let addr = public_address(&url).await?;
    let host = url.host_str().ok_or_else(unreadable)?.to_string();
    let client = Client::builder()
        .timeout(FETCH_TIMEOUT)
        .redirect(Policy::none())
        // The address just checked, so the name cannot be re-pointed at a
        // private one between the check and the fetch.
        .resolve(&host, addr)
        .build()
        .map_err(|_| unreadable())?;
    let mut res = client
        .get(url)
        .header("accept", "application/json")
        .send()
        .await
        .map_err(|_| unreadable())?;
    if !res.status().is_success() {
        return Err(unreadable());
    }
    let mut body = Vec::new();
    while let Some(chunk) = res.chunk().await.map_err(|_| unreadable())? {
        if body.len() + chunk.len() > MAX_DOCUMENT {
            return Err("the client's document is too large".into());
        }
        body.extend_from_slice(&chunk);
    }
    read(client_id, &body)
}

/// Read a fetched document, refusing one that names another client.
pub fn read(client_id: &str, bytes: &[u8]) -> Result<ClientDoc, String> {
    let doc: ClientDoc = serde_json::from_slice(bytes)
        .map_err(|e| format!("the client's document is not one: {e}"))?;
    if doc.client_id != client_id {
        return Err(format!(
            "the client's document names `{}`, not `{client_id}`",
            doc.client_id
        ));
    }
    Ok(doc)
}

/// Whether `doc` lets its client be sent to `redirect`.
pub fn allows_redirect(doc: &ClientDoc, redirect: &str) -> bool {
    if doc.redirect_uris.iter().any(|r| r == redirect) {
        return true;
    }
    let Ok(asked) = Url::parse(redirect) else {
        return false;
    };
    if !is_loopback(&asked) {
        return false;
    }
    doc.redirect_uris.iter().any(|listed| {
        Url::parse(listed).is_ok_and(|listed| {
            is_loopback(&listed)
                && listed.scheme() == asked.scheme()
                && listed.host_str() == asked.host_str()
                && listed.path() == asked.path()
                && listed.query() == asked.query()
        })
    })
}

fn is_loopback(url: &Url) -> bool {
    url.scheme() == "http" && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(redirects: &[&str]) -> ClientDoc {
        ClientDoc {
            client_id: "https://claude.ai/oauth/mcp-oauth-client-metadata".into(),
            client_name: Some("Claude".into()),
            redirect_uris: redirects.iter().map(|r| r.to_string()).collect(),
        }
    }

    #[test]
    fn a_client_id_is_an_https_url_naming_a_document() {
        assert!(check_id("https://claude.ai/oauth/mcp-oauth-client-metadata").is_ok());
        for bad in [
            "claude",
            "http://claude.ai/doc",
            "https://claude.ai",
            "https://claude.ai/",
            "https://u:p@claude.ai/doc",
            "https://claude.ai/doc#x",
        ] {
            assert!(check_id(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_a_public_address_is_fetched_from() {
        for ok in ["160.79.104.10", "8.8.8.8", "2606:4700::6810:84e5"] {
            assert!(public(ok.parse().unwrap()), "{ok}");
        }
        for bad in [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "192.168.0.4",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "192.0.2.1",
            "::1",
            "::",
            "fc00::1",
            "fd12::1",
            "fe80::1",
            "::ffff:10.0.0.1",
            "2001:db8::1",
        ] {
            assert!(!public(bad.parse().unwrap()), "{bad}");
        }
    }

    #[tokio::test]
    async fn a_client_id_on_a_private_host_is_refused_before_any_fetch() {
        for id in [
            "https://127.0.0.1/doc",
            "https://192.168.0.4:8443/admin",
            "https://localhost/doc",
        ] {
            let err = fetch(id).await.unwrap_err();
            assert_eq!(err, "the client's document is not on a public host", "{id}");
        }
    }

    #[test]
    fn a_document_naming_another_client_is_refused() {
        let id = "https://claude.ai/oauth/mcp-oauth-client-metadata";
        let body = br#"{"client_id":"https://evil.example/doc","redirect_uris":[]}"#;
        assert!(read(id, body).is_err());
        let body = br#"{"client_id":"https://claude.ai/oauth/mcp-oauth-client-metadata","client_name":"Claude","redirect_uris":["https://claude.ai/api/mcp/auth_callback"]}"#;
        assert_eq!(
            read(id, body).unwrap(),
            doc(&["https://claude.ai/api/mcp/auth_callback"])
        );
    }

    #[test]
    fn a_listed_redirect_is_allowed_and_a_loopback_on_any_port() {
        let d = doc(&[
            "https://claude.ai/api/mcp/auth_callback",
            "http://localhost/callback",
        ]);
        assert!(allows_redirect(
            &d,
            "https://claude.ai/api/mcp/auth_callback"
        ));
        assert!(allows_redirect(&d, "http://localhost:53211/callback"));
        assert!(!allows_redirect(&d, "http://localhost:53211/other"));
        assert!(!allows_redirect(&d, "http://127.0.0.1:53211/callback"));
        assert!(!allows_redirect(
            &d,
            "https://claude.ai/api/mcp/auth_callback/x"
        ));
        assert!(!allows_redirect(&d, "https://evil.example/callback"));
    }
}
