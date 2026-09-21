//! `web_fetch` tool — fetch and clean a web page.

use std::future;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::grants::Grants;
use crate::net;
use crate::{RegisteredTool, Replay, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// URL to fetch; must start with http:// or https://. Private/localhost URLs are blocked.
    #[validate(length(min = 1))]
    pub url: String,
    /// Approximate max tokens of content to return (500-16000); content is truncated
    /// at roughly 4 chars per token. Default: 4000.
    #[validate(range(min = 500, max = 16000))]
    pub max_tokens: Option<u32>,
}

#[derive(Serialize)]
pub struct Response {
    pub url: String,
    pub final_url: String,
    pub title: String,
    pub content: String,
    pub truncated: bool,
}

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("URL blocked: {0}")]
    UrlBlocked(String),
    #[error("fetch failed: {0}")]
    FetchFailed(String),
    #[error("HTTP error {status}: {detail}")]
    HttpError { status: u16, detail: String },
}

impl ToolError for FetchError {
    fn code(&self) -> &'static str {
        match self {
            FetchError::UrlBlocked(_) => "url_blocked",
            FetchError::FetchFailed(_) => "fetch_failed",
            FetchError::HttpError { .. } => "http_error",
        }
    }

    fn detail(&self) -> String {
        match self {
            FetchError::HttpError { status, detail } => {
                // Include status in detail as JSON-embedded
                format!("HTTP {status}: {detail}")
            }
            _ => self.to_string(),
        }
    }
}

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
fn public_address(grants: Grants, host: &str, port: u16) -> Result<SocketAddr, FetchError> {
    let ips = public_addresses(grants, host)?;
    Ok(SocketAddr::new(ips[0], port))
}

/// Every address `host` resolves to — never empty — refused when any is
/// private.
fn public_addresses(grants: Grants, host: &str) -> Result<Vec<IpAddr>, FetchError> {
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

/// The fetch client's resolver: every name it looks up — the page's and each
/// redirect target's — goes through [`public_addresses`], and the connection
/// uses the addresses that check approved. A name that answers the check with
/// a public address and the connection with a private one (DNS rebinding) has
/// no second lookup to answer.
struct PublicOnly {
    grants: Grants,
}

impl Resolve for PublicOnly {
    fn resolve(&self, name: Name) -> Resolving {
        let resolved = public_addresses(self.grants, name.as_str())
            .map(|ips| -> Addrs { Box::new(ips.into_iter().map(|ip| SocketAddr::new(ip, 0))) })
            .map_err(Into::into);
        Box::pin(future::ready(resolved))
    }
}

/// Follow a redirect only to a public address — the same rule as the first
/// request, so a public page cannot bounce the fetch onto the local network.
/// A named target is checked again by [`PublicOnly`] when it is connected to;
/// an address literal is resolved by nothing, so this is its only check.
fn guard_redirect(grants: Grants, attempt: redirect::Attempt) -> redirect::Action {
    const MAX_REDIRECTS: usize = 5;
    if attempt.previous().len() >= MAX_REDIRECTS {
        return attempt.error("too many redirects");
    }
    let url = attempt.url();
    let (Some(host), Some(port)) = (url.host_str(), url.port_or_known_default()) else {
        return attempt.error("redirect to a URL with no host");
    };
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    match public_address(grants, &host, port) {
        Ok(_) => attempt.follow(),
        Err(e) => attempt.error(e.to_string()),
    }
}

fn html_to_text(html: &str) -> (String, String) {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    // Extract title
    let title_sel = Selector::parse("title").unwrap();
    let title = document
        .select(&title_sel)
        .next()
        .map(|e| e.text().collect::<String>().trim().to_string())
        .unwrap_or_default();

    // Build text from body
    let body_sel = Selector::parse("body").unwrap();
    let body = document.select(&body_sel).next();

    let mut text = String::new();
    if let Some(body) = body {
        for node in body.descendants() {
            if let Some(elem) = node.value().as_element() {
                match elem.name() {
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                        let level = elem
                            .name()
                            .chars()
                            .nth(1)
                            .unwrap_or('1')
                            .to_digit(10)
                            .unwrap_or(1) as usize;
                        let hashes = "#".repeat(level);
                        if let Some(t) = scraper::ElementRef::wrap(node) {
                            let content: String = t.text().collect();
                            text.push_str(&format!("\n{} {}\n", hashes, content.trim()));
                        }
                    }
                    "p" => {
                        if let Some(t) = scraper::ElementRef::wrap(node) {
                            let content: String = t.text().collect();
                            let trimmed = content.trim();
                            if !trimmed.is_empty() {
                                text.push('\n');
                                text.push_str(trimmed);
                                text.push('\n');
                            }
                        }
                    }
                    "script" | "style" | "nav" | "header" | "footer" => {}
                    _ => {}
                }
            } else if let Some(txt) = node.value().as_text() {
                let s = txt.trim();
                if !s.is_empty() && s.len() > 2 {
                    // avoid noise
                }
            }
        }
        // If text is sparse, fall back to simple text extraction
        if text.trim().len() < 100 {
            text = body.text().collect::<Vec<_>>().join(" ");
            // Collapse whitespace
            text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        }
    }

    (title, text)
}

pub struct WebFetchTool;

impl Tool for WebFetchTool {
    const NAME: &'static str = "web_fetch";
    const DESCRIPTION: &'static str =
        "Fetch a single public web page or document by URL and return its main content as \
         cleaned text. Use for: reading a specific article the user linked, retrieving \
         documentation pages, pulling content from a known URL, getting context about a page \
         the user mentioned. Triggered by \"read this page\", \"fetch the article at\", \"what \
         does this URL say\", \"open this link\", \"get me the content of\", \"summarise this \
         page\", or the user pasting a URL. Returns title, cleaned body text, final URL after \
         redirects, and a truncated flag. Use web_search when the URL is not yet known. Use \
         http_session_* for authenticated API calls or operations needing cookies/auth state.";

    type Request = Request;
    type Response = Response;
    type Error = FetchError;

    /// Retrieves a page by GET; nothing at the far end changes.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, FetchError> {
        if !req.url.starts_with("http://") && !req.url.starts_with("https://") {
            return Err(FetchError::UrlBlocked(
                "URL must start with http:// or https://".to_string(),
            ));
        }
        if is_private_url(&req.url) {
            return Err(FetchError::UrlBlocked(
                "private/localhost URLs are not allowed".to_string(),
            ));
        }
        let parsed = url::Url::parse(&req.url)
            .map_err(|e| FetchError::UrlBlocked(format!("not a URL: {e}")))?;
        let (Some(host), Some(port)) = (parsed.host_str(), parsed.port_or_known_default()) else {
            return Err(FetchError::UrlBlocked("the URL names no host".to_string()));
        };
        let host = host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_string();
        let grants = ctx.grants();
        let pinned = public_address(grants, &host, port)?;

        // A client of its own: pinned to the address just checked, resolving
        // every other name through `PublicOnly`, and following redirects only
        // to public addresses. The shared client does none of these.
        let client = net::http_client_builder(grants)
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?
            .timeout(Duration::from_secs(30))
            .dns_resolver(Arc::new(PublicOnly { grants }))
            .resolve(&host, pinned)
            .redirect(redirect::Policy::custom(move |a| guard_redirect(grants, a)))
            .build()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
        let resp = client
            .get(&req.url)
            .header("User-Agent", "Mozilla/5.0 (compatible; zend-tools/0.1)")
            .send()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;

        let status = resp.status();
        let final_url = resp.url().to_string();

        if !status.is_success() {
            return Err(FetchError::HttpError {
                status: status.as_u16(),
                detail: status.canonical_reason().unwrap_or("").to_string(),
            });
        }

        let html = resp
            .text()
            .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
        let (title, content) = html_to_text(&html);

        let max_chars = (req.max_tokens.unwrap_or(4000) * 4) as usize;
        let (content, truncated) = if content.len() > max_chars {
            // Cut on a character boundary: a byte index inside a multi-byte
            // character would panic the call.
            let mut end = max_chars;
            while !content.is_char_boundary(end) {
                end -= 1;
            }
            (content[..end].to_string(), true)
        } else {
            (content, false)
        };

        Ok(Response {
            url: req.url,
            final_url,
            title,
            content,
            truncated,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WebFetchTool>();

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::task::{Context, Poll, Waker};

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

    /// The resolver the fetch client connects through applies the same rule,
    /// so a redirect to a local name is refused where it is connected to.
    #[test]
    fn the_fetch_clients_resolver_refuses_private_names() {
        let resolver = PublicOnly {
            grants: Grants::NONE.with(Capability::Network),
        };
        let name: Name = "localhost".parse().unwrap();
        let resolved = poll_ready(resolver.resolve(name));
        assert!(resolved.is_err(), "localhost resolved for the fetch client");
    }

    /// The fetch itself refuses a local target before any request is sent,
    /// whether it is named or literal.
    #[test]
    fn a_fetch_of_a_local_target_is_blocked() {
        let ctx = ToolContext::new().granting(Grants::NONE.with(Capability::Network));
        for url in [
            "http://127.0.0.1:8081/v1/status",
            "http://localhost:8081/",
            "http://[::ffff:127.0.0.1]/",
            "http://192.168.0.5:8081/",
        ] {
            let err = WebFetchTool::run(
                &ctx,
                Request {
                    url: url.to_string(),
                    max_tokens: None,
                },
            )
            .err()
            .unwrap_or_else(|| panic!("{url} was fetched"));
            assert_eq!(err.code(), "url_blocked", "{url}: {err}");
        }
    }

    /// Drive a ready future: the resolver's lookup is synchronous, so its
    /// future is complete on the first poll.
    fn poll_ready(mut fut: Resolving) -> Result<Addrs, Box<dyn Error + Send + Sync>> {
        match fut.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(r) => r,
            Poll::Pending => panic!("the resolver's future is ready on the first poll"),
        }
    }
}
