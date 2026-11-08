//! An HTTP client for one URL: pinned to the public address its host was
//! checked at, resolving every other name through the same check, and
//! following redirects only to public addresses.

use std::future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect;

use super::address::{is_private_url, public_address, public_addresses};
use super::error::FetchError;
use crate::grants::Grants;
use crate::net;

/// The most redirects a fetch follows.
const MAX_REDIRECTS: usize = 5;

/// A client for one URL, and that URL.
pub struct Pinned {
    pub client: Client,
    pub url: String,
}

/// A client for `url`, which must be `http` or `https` and reach only public
/// addresses: refused before any request is sent when it names a local
/// target, literally or by a name that resolves to one.
pub fn public_client(grants: Grants, url: &str, timeout: Duration) -> Result<Pinned, FetchError> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(FetchError::UrlBlocked(
            "URL must start with http:// or https://".to_string(),
        ));
    }
    if is_private_url(url) {
        return Err(FetchError::UrlBlocked(
            "private/localhost URLs are not allowed".to_string(),
        ));
    }
    let parsed =
        url::Url::parse(url).map_err(|e| FetchError::UrlBlocked(format!("not a URL: {e}")))?;
    let (Some(host), Some(port)) = (parsed.host_str(), parsed.port_or_known_default()) else {
        return Err(FetchError::UrlBlocked("the URL names no host".to_string()));
    };
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    let pinned = public_address(grants, &host, port)?;
    let client = net::http_client_builder(grants)
        .map_err(|e| FetchError::FetchFailed(e.to_string()))?
        .timeout(timeout)
        .dns_resolver(Arc::new(PublicOnly { grants }))
        .resolve(&host, pinned)
        .redirect(redirect::Policy::custom(move |a| guard_redirect(grants, a)))
        .build()
        .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
    Ok(Pinned {
        client,
        url: url.to_string(),
    })
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

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::task::{Context, Poll, Waker};

    use super::*;
    use crate::grants::Capability;

    fn net() -> Grants {
        Grants::NONE.with(Capability::Network)
    }

    /// The resolver the fetch client connects through applies the same rule,
    /// so a redirect to a local name is refused where it is connected to.
    #[test]
    fn the_fetch_clients_resolver_refuses_private_names() {
        let resolver = PublicOnly { grants: net() };
        let name: Name = "localhost".parse().unwrap();
        let resolved = poll_ready(resolver.resolve(name));
        assert!(resolved.is_err(), "localhost resolved for the fetch client");
    }

    /// **A local target is refused before any request is sent**, named or
    /// literal, an IPv4 address carried in IPv6 among them; so is a scheme
    /// that is not HTTP.
    #[test]
    fn a_local_target_is_refused_before_anything_is_sent() {
        for url in [
            "http://127.0.0.1:8081/v1/status",
            "http://localhost:8081/",
            "http://[::ffff:127.0.0.1]/",
            "http://192.168.0.5:8081/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data",
            "file:///C:/Windows/win.ini",
        ] {
            let refused = public_client(net(), url, Duration::from_secs(1));
            assert!(
                matches!(refused, Err(FetchError::UrlBlocked(_))),
                "{url} was let through"
            );
        }
        assert!(matches!(
            public_client(Grants::NONE, "https://8.8.8.8/", Duration::from_secs(1)),
            Err(FetchError::FetchFailed(_))
        ));
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
