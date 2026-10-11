//! HTTP/2 and HTTP/3 requests, made to read the way an HTTP/1.1 one does.
//!
//! Both protocols name the host in the `:authority` pseudo-header, which arrives
//! as the URI's authority and never as a `Host` header. Everything downstream of
//! the entrance reads `Host`: the site table resolves a request by it, the proxy
//! forwards it as `X-Forwarded-Host`, and a daemon behind the proxy builds its
//! URLs from it. Without this every HTTP/2 and HTTP/3 request would land on the
//! default site whatever name it was made to.

use axum::http::{header, HeaderValue, Request, Uri};

/// Give the request a `Host` from its authority when it carries none, and put
/// its URI back in origin form — the path and query alone.
pub fn origin_form<B>(req: &mut Request<B>) {
    let Some(authority) = req.uri().authority().cloned() else {
        return;
    };
    if !req.headers().contains_key(header::HOST) {
        if let Ok(v) = HeaderValue::from_str(authority.as_str()) {
            req.headers_mut().insert(header::HOST, v);
        }
    }
    let pq = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/")
        .to_owned();
    if let Ok(uri) = pq.parse::<Uri>() {
        *req.uri_mut() = uri;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalised(uri: &str, host: Option<&str>) -> (String, Option<String>) {
        let mut b = Request::builder().uri(uri);
        if let Some(h) = host {
            b = b.header(header::HOST, h);
        }
        let mut req = b.body(()).unwrap();
        origin_form(&mut req);
        (
            req.uri().to_string(),
            req.headers()
                .get(header::HOST)
                .map(|v| v.to_str().unwrap().to_owned()),
        )
    }

    #[test]
    fn the_authority_becomes_the_host_and_the_uri_its_path() {
        assert_eq!(
            normalised("https://code.tokera.com/v1/status?x=1", None),
            ("/v1/status?x=1".into(), Some("code.tokera.com".into()))
        );
    }

    #[test]
    fn a_port_in_the_authority_is_kept() {
        assert_eq!(
            normalised("https://localhost:8443/", None),
            ("/".into(), Some("localhost:8443".into()))
        );
    }

    /// A client that sent `Host` as well keeps its own.
    #[test]
    fn a_host_header_the_client_sent_is_kept() {
        assert_eq!(
            normalised("https://a.test/x", Some("b.test")),
            ("/x".into(), Some("b.test".into()))
        );
    }

    /// An HTTP/1.1 request is already in origin form and is left alone.
    #[test]
    fn an_origin_form_request_is_untouched() {
        assert_eq!(
            normalised("/blog?page=2", Some("tokera.com")),
            ("/blog?page=2".into(), Some("tokera.com".into()))
        );
    }

    #[test]
    fn an_authority_with_no_path_is_the_root() {
        assert_eq!(
            normalised("https://tokera.com", None),
            ("/".into(), Some("tokera.com".into()))
        );
    }
}
