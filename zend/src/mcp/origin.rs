//! The public origin a request reached `/mcp` at: where the challenge sends a
//! client to sign in, and the origin a browser's `Origin` must match
//! (`docs/claude_zend_gateway.md` §12).
//!
//! Behind the gateway, the member sees the gateway's hop; the gateway states
//! the client's host on `X-Forwarded-Host` and its scheme, from its own
//! configuration, on `X-Forwarded-Proto` (`web::proxy`). Called directly, the
//! request's own `Host` is the origin, over plain HTTP.

use axum::http::{header, HeaderMap};

/// `https://code.tokera.com`, or `http://127.0.0.1:18081` when called directly.
pub fn of(headers: &HeaderMap) -> String {
    let read = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|v| !v.is_empty())
    };
    let host = read("x-forwarded-host")
        .or_else(|| read(header::HOST.as_str()))
        .unwrap_or("localhost");
    let scheme = match read("x-forwarded-proto") {
        Some("https") => "https",
        _ => "http",
    };
    format!("{scheme}://{host}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_static(v));
        }
        h
    }

    #[test]
    fn behind_the_gateway_the_clients_origin_is_used() {
        let h = headers(&[
            ("host", "192.168.0.237:18081"),
            ("x-forwarded-host", "code.tokera.com"),
            ("x-forwarded-proto", "https"),
        ]);
        assert_eq!(of(&h), "https://code.tokera.com");
    }

    #[test]
    fn called_directly_the_host_is_the_origin_over_http() {
        assert_eq!(
            of(&headers(&[("host", "127.0.0.1:18081")])),
            "http://127.0.0.1:18081"
        );
        assert_eq!(of(&HeaderMap::new()), "http://localhost");
    }
}
