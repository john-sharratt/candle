//! Who may open a conversation `/mcp` started (`docs/claude_zend_gateway.md`
//! §3.2).
//!
//! A worker's conversation id names its owner — `mcp:<owner key>:<thread>/<name>`
//! ([`super::threads::conv_id`]) — so the check needs no lookup: an id that
//! names an owner is served only to that person or an Admin, on every route
//! that names a conversation. Ids that name no owner — the dialogues zend's own
//! UI makes — keep the access they have.
//!
//! A refusal is a `404`: a conversation someone may not open is, to them, not
//! there, and a `403` would confirm that it is.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use web::auth::forwarded::identify;
use web::auth::{Role, Roles};

use super::caller::Person;
use crate::api::CONVERSATION_HEADER;
use crate::session::ZendSession;

const CONVERSATIONS: &str = "/v1/conversations/";

/// The owner key `conv_id` names, if it is one `/mcp` started.
pub fn owner_of(conv_id: &str) -> Option<&str> {
    let rest = conv_id.strip_prefix("mcp:")?;
    rest.split_once(':').map(|(owner, _)| owner)
}

/// Whether the caller `headers` name may open `conv_id`.
pub fn may_open(conv_id: &str, headers: &HeaderMap, roles: &Roles) -> bool {
    let Some(owner) = owner_of(conv_id) else {
        return true;
    };
    let Ok(identity) = identify(headers) else {
        return false;
    };
    let person = Person::of(&identity);
    person.key() == owner || person.role(roles).at_least(Role::Admin)
}

/// `%XX` escapes in a path segment, decoded; a malformed escape is kept as
/// written.
///
/// Not the gateway's `content::percent_decode`, which refuses `%2F` because it
/// guards a file path: a worker's conversation id holds a `/`, written `%2F`
/// in a path segment, and must decode to it.
fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let hex = |b: u8| (b as char).to_digit(16);
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The conversation id a request's path names, if it names one.
pub fn path_conversation(path: &str) -> Option<String> {
    let rest = path.strip_prefix(CONVERSATIONS)?;
    let segment = rest.split('/').next().filter(|s| !s.is_empty())?;
    Some(percent_decode(segment))
}

/// The layer over zend's routes: a request naming — by its path or by the
/// conversation header — a conversation its caller may not open is answered
/// `404` before it reaches a handler. The chat route's body `conv_id` is
/// checked in the handler, which is where the body is read.
pub async fn guard(State(session): State<Arc<ZendSession>>, req: Request, next: Next) -> Response {
    let named = path_conversation(req.uri().path()).or_else(|| {
        req.headers()
            .get(CONVERSATION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    });
    if let Some(id) = named {
        if !may_open(&id, req.headers(), session.roles()) {
            return not_found();
        }
    }
    next.run(req).await
}

/// The refusal: the conversation is not there, to this caller.
pub fn not_found() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use web::auth::forwarded::{PROVIDER, USER};

    fn as_person(sub: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(USER, HeaderValue::from_str(sub).unwrap());
        h.insert(PROVIDER, HeaderValue::from_static("google"));
        h
    }

    fn key(sub: &str) -> String {
        Person {
            provider: "google".into(),
            sub: sub.into(),
            email: String::new(),
            name: String::new(),
        }
        .key()
    }

    #[test]
    fn only_an_mcp_id_names_an_owner() {
        assert_eq!(owner_of("mcp:abc:t/n"), Some("abc"));
        assert_eq!(owner_of("c-123"), None);
        assert_eq!(owner_of("mcp:"), None);
    }

    #[test]
    fn an_owned_conversation_opens_for_its_owner_and_an_admin_only() {
        let roles: Roles = serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap();
        let id = format!("mcp:{}:review/verify", key("alice"));
        assert!(may_open(&id, &as_person("alice"), &roles));
        assert!(may_open(&id, &as_person("boss"), &roles));
        assert!(!may_open(&id, &as_person("mallory"), &roles));
        assert!(!may_open(&id, &HeaderMap::new(), &roles));
        assert!(may_open("c-123", &HeaderMap::new(), &roles));
    }

    #[test]
    fn the_path_names_its_conversation_decoded() {
        assert_eq!(
            path_conversation("/v1/conversations/mcp%3Aabc%3Areview%2Fverify/journal"),
            Some("mcp:abc:review/verify".to_string())
        );
        assert_eq!(
            path_conversation("/v1/conversations/c-1"),
            Some("c-1".to_string())
        );
        assert_eq!(path_conversation("/v1/conversations"), None);
        assert_eq!(path_conversation("/v1/conversations/"), None);
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("%zz"), "%zz");
        assert_eq!(percent_decode("%é1"), "%é1");
        assert_eq!(percent_decode("%41%2f"), "A/");
    }
}
