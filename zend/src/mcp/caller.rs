//! Who is calling `/mcp`, and the key their records are kept under
//! (`docs/claude_zend_gateway.md` §3.2, §12).
//!
//! The identity is the gateway's: it signed the user in through Google, and on
//! `/mcp` it turns the client's token back into that identity and states it on
//! the `X-Tokera-*` headers (`web::auth::mcp`). The role is read from the
//! estate's table on every call, so a role changed in `roles.yaml` takes effect
//! on the next call rather than the next sign-in.

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use web::auth::forwarded::{identify, EMAIL, NAME, PROVIDER, USER};
use web::auth::{Identity, Role, Roles};

/// The person a token speaks for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Person {
    pub provider: String,
    pub sub: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub name: String,
}

impl Person {
    pub fn of(identity: &Identity) -> Self {
        Self {
            provider: identity.provider.clone(),
            sub: identity.sub.clone(),
            email: identity.email.clone(),
            name: identity.name.clone(),
        }
    }

    /// The key this person's records are kept under: the first 16 hex digits
    /// of `sha256(provider \0 sub)`, as npcd keys an account, so a subject id
    /// never appears in a key and two providers' subjects never meet.
    pub fn key(&self) -> String {
        let digest = Sha256::digest(format!("{}\0{}", self.provider, self.sub).as_bytes());
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// This person's role in the estate's table.
    pub fn role(&self, roles: &Roles) -> Role {
        let identity = Identity {
            provider: self.provider.clone(),
            sub: self.sub.clone(),
            email: self.email.clone(),
            name: self.name.clone(),
            picture: String::new(),
            exp: u64::MAX,
        };
        roles.of(Some(&identity))
    }
}

/// A call's caller: the person and their role now.
#[derive(Debug, Clone)]
pub struct Caller {
    pub person: Person,
    pub role: Role,
}

impl Caller {
    /// Whoever `headers` name — the identity the gateway states, which this
    /// member believes only from the gateways it trusts — and their role now.
    pub fn of(headers: &HeaderMap, roles: &Roles) -> Option<Self> {
        let identity = identify(headers).ok()?;
        let person = Person::of(&identity);
        Some(Self {
            role: person.role(roles),
            person,
        })
    }

    pub fn key(&self) -> String {
        self.person.key()
    }

    pub fn is_admin(&self) -> bool {
        self.role.at_least(Role::Admin)
    }

    /// The identity a request this member sends on the caller's behalf
    /// states — the headers the gateway believes from a member's socket — so
    /// every hop checks the caller's role, and the coordination log records
    /// who ran what.
    pub fn headers(&self) -> Vec<(HeaderName, HeaderValue)> {
        let p = &self.person;
        [
            (USER, p.sub.as_str()),
            (PROVIDER, p.provider.as_str()),
            (EMAIL, p.email.as_str()),
            (NAME, p.name.as_str()),
        ]
        .into_iter()
        .filter(|(_, v)| !v.is_empty())
        .filter_map(|(k, v)| Some((HeaderName::from_static(k), HeaderValue::from_str(v).ok()?)))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(provider: &str, sub: &str) -> Person {
        Person {
            provider: provider.into(),
            sub: sub.into(),
            email: String::new(),
            name: String::new(),
        }
    }

    #[test]
    fn a_key_is_sixteen_hex_digits_of_provider_and_subject() {
        let k = person("google", "1").key();
        assert_eq!(k.len(), 16);
        assert!(k.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(k, person("google", "1").key());
        assert_ne!(k, person("github", "1").key());
        assert_ne!(k, person("google", "2").key());
    }

    #[test]
    fn the_caller_is_whoever_the_gateway_names_and_nobody_otherwise() {
        use axum::http::HeaderValue;
        use web::auth::forwarded::{PROVIDER, USER};
        let roles: Roles = serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap();
        let mut h = HeaderMap::new();
        assert!(Caller::of(&h, &roles).is_none());
        h.insert(USER, HeaderValue::from_static("boss"));
        h.insert(PROVIDER, HeaderValue::from_static("google"));
        let c = Caller::of(&h, &roles).unwrap();
        assert_eq!(c.key(), person("google", "boss").key());
        assert!(c.is_admin());
    }

    /// What a caller states up a member's socket reads back as the same
    /// caller, and an empty field is not sent.
    #[test]
    fn the_headers_a_caller_states_identify_it_again() {
        let roles: Roles = serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap();
        let caller = Caller {
            person: Person {
                provider: "google".into(),
                sub: "boss".into(),
                email: "b@x.com".into(),
                name: String::new(),
            },
            role: Role::Admin,
        };
        let sent = caller.headers();
        assert_eq!(sent.len(), 3, "no empty name");
        let mut h = HeaderMap::new();
        for (k, v) in sent {
            h.insert(k, v);
        }
        let again = Caller::of(&h, &roles).unwrap();
        assert_eq!(again.person, caller.person);
        assert!(again.is_admin());
    }

    #[test]
    fn the_role_is_read_from_the_table() {
        let roles: Roles = serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap();
        assert_eq!(person("google", "boss").role(&roles), Role::Admin);
        assert_eq!(person("google", "someone").role(&roles), Role::User);
    }
}
