//! The ops key (`docs/replicated_substrate.md` §12): the admin an operator on
//! the gateway's box acts as without a browser session — for the emergency
//! that cannot wait for a sign-in. A request presenting it is the configured
//! admin on every route of the gateway's own (its page, its commands, every
//! upgrade command), and that admin is the identity the gateway names to the
//! member a command goes to.
//!
//! It is a file on the gateway's box, so the authority it carries is the
//! authority to read that file: the same an operator already has over the
//! binaries keep runs there.

use std::path::Path;

use axum::http::HeaderMap;

use super::bearer_key::BearerKey;
use crate::auth::Identity;

/// The provider a key-held identity names: no sign-in issued it, and the role
/// table matches it by the email it carries.
pub const OPS_PROVIDER: &str = "ops-key";

/// The loaded key and the admin it acts as.
pub struct OpsKey {
    key: BearerKey,
    admin: Identity,
}

impl OpsKey {
    /// The key in `file`, acting as the admin whose email is `admin`.
    pub fn load(file: &Path, admin: &str) -> anyhow::Result<OpsKey> {
        Ok(OpsKey::new(BearerKey::load(file)?, admin))
    }

    /// `key`, acting as the admin whose email is `admin`.
    pub fn new(key: BearerKey, admin: &str) -> OpsKey {
        OpsKey {
            key,
            admin: Identity {
                provider: OPS_PROVIDER.into(),
                sub: OPS_PROVIDER.into(),
                email: admin.trim().into(),
                name: "ops key".into(),
                picture: String::new(),
                exp: 0,
            },
        }
    }

    /// The admin, when `headers` present the key.
    pub fn caller(&self, headers: &HeaderMap) -> Option<Identity> {
        self.key.presented(headers).then(|| self.admin.clone())
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{header, HeaderValue};

    use super::*;
    use crate::auth::{Principal, Role, Roles};

    fn bearer(token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        h
    }

    /// **The key is the configured admin, and nothing else is.** A request
    /// presenting it is that admin to the role table — the check every one of
    /// the gateway's own routes makes — and one that does not, or presents a
    /// different key, is nobody.
    #[test]
    fn the_key_acts_as_its_admin_and_nothing_else_does() {
        let ops = OpsKey::new(
            BearerKey::from_hex(&"cd".repeat(32)).unwrap(),
            " op@tokera.com ",
        );
        let roles = Roles {
            admins: vec![Principal::Email("op@tokera.com".into())],
            creators: Vec::new(),
        };
        let id = ops.caller(&bearer(&"cd".repeat(32))).expect("presented");
        assert_eq!(
            (id.provider.as_str(), id.email.as_str()),
            (OPS_PROVIDER, "op@tokera.com")
        );
        assert_eq!(roles.of(Some(&id)), Role::Admin);
        assert_eq!(ops.caller(&bearer(&"ab".repeat(32))), None);
        assert_eq!(ops.caller(&HeaderMap::new()), None);
    }
}
