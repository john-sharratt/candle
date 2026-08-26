//! Cookie shaping and reading.
//!
//! The one that matters is `Domain=.tokera.com`. That single attribute is what
//! makes signing in once reach `code.` and `bot.` — the browser presents the
//! cookie to every host under the domain, so the subsites need no handshake
//! with the parent and no cross-origin dance.

use axum::http::HeaderMap;

pub const SESSION: &str = "tokera_session";
pub const PENDING: &str = "tokera_oauth";

/// `Secure` is derived from the redirect URI's scheme rather than configured:
/// a deployment on https gets secure cookies without anyone remembering to ask
/// for them, and a local http provider still works because the same rule says
/// no.
pub fn set(name: &str, value: &str, domain: &str, max_age: i64, secure: bool) -> String {
    let mut c = format!("{name}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}");
    if !domain.is_empty() {
        c.push_str(&format!("; Domain={domain}"));
    }
    if secure {
        c.push_str("; Secure");
    }
    c
}

/// Expire a cookie. Must carry the same `Domain` it was set with, or the
/// browser treats it as a different cookie and the old one survives — which
/// looks exactly like a broken sign-out.
pub fn clear(name: &str, domain: &str, secure: bool) -> String {
    set(name, "", domain, 0, secure)
}

/// Read one cookie out of a request's headers.
pub fn get<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get_all(axum::http::header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|line| line.split(';'))
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| k.trim() == name)
        .map(|(_, v)| v.trim())
}

/// `SameSite=Lax` is what lets the cookie ride the provider's redirect back to
/// the callback. `Strict` would withhold it on that top-level navigation and
/// the sign-in would silently never complete.
pub fn same_site_note() -> &'static str {
    "Lax"
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn the_domain_attribute_is_what_reaches_the_subsites() {
        let c = set(SESSION, "abc", ".tokera.com", 3600, true);
        assert!(c.contains("Domain=.tokera.com"), "{c}");
        assert!(c.contains("HttpOnly"), "{c}");
        assert!(c.contains("Secure"), "{c}");
        assert!(c.contains("SameSite=Lax"), "{c}");
    }

    #[test]
    fn an_http_deployment_gets_no_secure_flag() {
        // Otherwise the cookie is silently dropped and sign-in never sticks.
        let c = set(SESSION, "abc", "", 3600, false);
        assert!(!c.contains("Secure"), "{c}");
        assert!(!c.contains("Domain="), "{c}");
    }

    #[test]
    fn clearing_keeps_the_domain_so_the_browser_matches_it() {
        let c = clear(SESSION, ".tokera.com", true);
        assert!(c.contains("Domain=.tokera.com"), "{c}");
        assert!(c.contains("Max-Age=0"), "{c}");
    }

    #[test]
    fn a_cookie_is_found_among_several() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static("theme=dark; tokera_session=abc.def; other=1"),
        );
        assert_eq!(get(&h, SESSION), Some("abc.def"));
        assert_eq!(get(&h, "theme"), Some("dark"));
        assert_eq!(get(&h, "missing"), None);
    }

    #[test]
    fn a_name_that_is_a_suffix_of_another_is_not_confused_for_it() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static("not_tokera_session=wrong; tokera_session=right"),
        );
        assert_eq!(get(&h, SESSION), Some("right"));
    }
}
