//! Clients registered in the gateway's config, `auth.mcp_clients`
//! (`docs/claude_zend_gateway.md` §12).
//!
//! A client that takes a fixed client id — Grok's connectors do — cannot name
//! a metadata document, so the gateway holds what one would say: the client's
//! name and the redirects it may be sent to. A registered client may also hold
//! a secret, which it presents at the token endpoint in the form
//! (`client_secret_post`) or a Basic header (`client_secret_basic`). The config
//! holds only the secret's SHA-256, so it carries no secret itself. Every
//! client, registered or not, still proves the code with PKCE.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use reqwest::Url;
use sha2::{Digest, Sha256};

use super::client::ClientDoc;
use crate::auth::session::constant_time_eq;
use crate::config::McpClient;

/// The registered client `id` names, if any.
pub fn find<'a>(clients: &'a [McpClient], id: &str) -> Option<&'a McpClient> {
    clients.iter().find(|c| c.id == id)
}

/// What a metadata document would say of `client`: its id, name and
/// redirects, so the redirect is matched as any client's is.
pub fn doc(client: &McpClient) -> ClientDoc {
    ClientDoc {
        client_id: client.id.clone(),
        client_name: Some(client.name.clone()),
        redirect_uris: client.redirect_uris.clone(),
    }
}

/// Refuse a client list the gateway could not honour, at load: an id that is
/// empty, padded with space, or holds a `:` — a URL always does, and would be
/// read as a metadata document, and a Basic header could not carry it — an id
/// listed twice, a client with no redirect or one that is neither `https` nor
/// loopback `http`, and a secret hash that is not 64 lowercase hex digits.
pub fn check(clients: &[McpClient]) -> Result<(), String> {
    for (i, c) in clients.iter().enumerate() {
        if c.id.is_empty() || c.id.trim() != c.id || c.id.contains(':') {
            return Err(format!(
                "mcp client `{}`: an id is a plain name — not a URL, no `:`, no surrounding space",
                c.id
            ));
        }
        if clients[..i].iter().any(|d| d.id == c.id) {
            return Err(format!("mcp client `{}` is listed twice", c.id));
        }
        if c.redirect_uris.is_empty() {
            return Err(format!("mcp client `{}` lists no redirect", c.id));
        }
        for r in &c.redirect_uris {
            let ok = Url::parse(r).is_ok_and(|u| {
                u.scheme() == "https"
                    || (u.scheme() == "http"
                        && matches!(u.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")))
            });
            if !ok {
                return Err(format!(
                    "mcp client `{}`: redirect `{r}` is neither https nor loopback http",
                    c.id
                ));
            }
        }
        if let Some(h) = &c.secret_sha256 {
            if h.len() != 64 || !h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
                return Err(format!(
                    "mcp client `{}`: secret_sha256 is not 64 lowercase hex digits",
                    c.id
                ));
            }
        }
    }
    Ok(())
}

/// The SHA-256 of `secret`, in lowercase hex — what `secret_sha256` holds.
pub fn secret_hash(secret: &str) -> String {
    Sha256::digest(secret.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Whether `presented` is `client`'s secret. A client with none needs none,
/// and one sent anyway is not refused; a client with one must present it.
/// Compared as hashes, in constant time.
pub fn authenticates(client: &McpClient, presented: Option<&str>) -> bool {
    let Some(want) = &client.secret_sha256 else {
        return true;
    };
    let Some(presented) = presented else {
        return false;
    };
    constant_time_eq(want.as_bytes(), secret_hash(presented).as_bytes())
}

/// The client id and secret an `Authorization` value carries: `None` when it
/// is not Basic, an error when it is Basic but unreadable — a client that
/// tried to authenticate and could not is refused, not judged on the form
/// instead.
///
/// Each half is form-urlencoded, then the two joined by `:` (RFC 6749
/// §2.3.1). A secret of URL-safe characters — what the gateway's own
/// generated secrets are — reads the same whether a client encodes it or not.
pub fn basic(value: &str) -> Option<Result<(String, String), String>> {
    let (scheme, encoded) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("basic") {
        return None;
    }
    let unreadable = || "the Authorization header's Basic credentials are unreadable".to_string();
    let read = || -> Option<(String, String)> {
        let joined = String::from_utf8(B64.decode(encoded.trim()).ok()?).ok()?;
        let (id, secret) = joined.split_once(':')?;
        Some((form_decode(id)?, form_decode(secret)?))
    };
    Some(read().ok_or_else(unreadable))
}

/// `s` with `application/x-www-form-urlencoded` escapes undone: `+` is a
/// space, `%XX` a byte. `None` for a broken escape, a NUL, or bytes that are
/// not UTF-8.
fn form_decode(s: &str) -> Option<String> {
    let mut out = Vec::with_capacity(s.len());
    let mut bytes = s.bytes();
    while let Some(b) = bytes.next() {
        match b {
            b'+' => out.push(b' '),
            b'%' => {
                let hi = (bytes.next()? as char).to_digit(16)?;
                let lo = (bytes.next()? as char).to_digit(16)?;
                out.push((hi * 16 + lo) as u8);
            }
            b => out.push(b),
        }
    }
    if out.contains(&0) {
        return None;
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
pub mod testing {
    use super::*;

    pub const GROK_ID: &str = "zen-grok";
    pub const GROK_SECRET: &str = "s3cret-value";
    pub const GROK_REDIRECT: &str = "https://grok.com/connectors-oauth-exchange-code/";

    pub fn grok() -> McpClient {
        McpClient {
            id: GROK_ID.into(),
            name: "Grok".into(),
            redirect_uris: vec![GROK_REDIRECT.into()],
            secret_sha256: Some(secret_hash(GROK_SECRET)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    /// The FIPS 180-2 vector, so the hash a deployment writes with any tool is
    /// the one compared here.
    #[test]
    fn the_secret_hash_is_sha256_in_lowercase_hex() {
        assert_eq!(
            secret_hash("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_client_with_a_secret_needs_that_secret() {
        let c = grok();
        assert!(authenticates(&c, Some(GROK_SECRET)));
        assert!(!authenticates(&c, Some("s3cret-valuf")));
        assert!(!authenticates(&c, None));
        let public = McpClient {
            secret_sha256: None,
            ..grok()
        };
        assert!(authenticates(&public, None));
        assert!(authenticates(&public, Some("anything")));
    }

    #[test]
    fn a_registered_client_reads_as_its_document() {
        let clients = vec![grok()];
        assert_eq!(find(&clients, GROK_ID), Some(&clients[0]));
        assert_eq!(find(&clients, "zen-other"), None);
        assert_eq!(
            doc(&clients[0]),
            ClientDoc {
                client_id: GROK_ID.into(),
                client_name: Some("Grok".into()),
                redirect_uris: vec![GROK_REDIRECT.into()],
            }
        );
    }

    #[test]
    fn a_client_list_the_gateway_cannot_honour_is_refused_at_load() {
        assert_eq!(check(&[grok()]), Ok(()));
        let bad = |edit: fn(&mut McpClient)| {
            let mut c = grok();
            edit(&mut c);
            check(&[c]).unwrap_err()
        };
        for id in [
            "https://grok.com/doc",
            "https:grok",
            "",
            " zen-grok",
            "zen-grok ",
        ] {
            let mut c = grok();
            c.id = id.into();
            let err = check(&[c]).unwrap_err();
            assert!(err.contains("plain name"), "`{id}`: {err}");
        }
        assert!(bad(|c| c.redirect_uris.clear()).contains("no redirect"));
        assert!(
            bad(|c| c.redirect_uris = vec!["http://grok.com/cb".into()]).contains("neither https")
        );
        assert!(bad(|c| c.secret_sha256 = Some("abc".into())).contains("64 lowercase hex"));
        assert!(
            bad(|c| c.secret_sha256 = c.secret_sha256.as_ref().map(|h| h.to_ascii_uppercase()))
                .contains("64 lowercase hex"),
            "the hash is compared as written, so it is written one way"
        );
        assert!(check(&[grok(), grok()]).unwrap_err().contains("twice"));
        let loopback = McpClient {
            redirect_uris: vec!["http://127.0.0.1:8123/cb".into()],
            ..grok()
        };
        assert_eq!(check(&[loopback]), Ok(()));
    }

    #[test]
    fn a_basic_header_carries_the_id_and_secret_form_encoded() {
        // base64("zen-grok:s3cret-value")
        assert_eq!(
            basic("Basic emVuLWdyb2s6czNjcmV0LXZhbHVl"),
            Some(Ok((GROK_ID.into(), GROK_SECRET.into())))
        );
        // base64("a%3Ab:c+d%25") — an escaped colon in the id, a space and a
        // percent sign in the secret.
        assert_eq!(
            basic("basic YSUzQWI6YytkJTI1"),
            Some(Ok(("a:b".into(), "c d%".into())))
        );
        assert_eq!(basic("Bearer emVuLWdyb2s6czNjcmV0LXZhbHVl"), None);
        // Basic, but unreadable: bad base64; base64("no-colon"); base64("a:%zz");
        // base64("a:%00").
        for unreadable in [
            "Basic !!!",
            "Basic bm8tY29sb24=",
            "Basic YToleno=",
            "Basic YTolMDA=",
        ] {
            assert!(matches!(basic(unreadable), Some(Err(_))), "{unreadable}");
        }
    }
}
