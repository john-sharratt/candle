//! The identity the gateway forwards, read back by a daemon behind it.
//!
//! The gateway resolves its session cookie on every request and forwards the
//! result on `X-Tokera-User` / `-Provider` / `-Email` / `-Name` / `-Picture`,
//! having first cleared whatever the client sent under those names
//! ([`crate::proxy`]). A daemon reads them here and believes them — sound for
//! exactly as long as the daemon is reachable only through the gateway, which
//! is a property of the deployment rather than of this code.
//!
//! One parser for every daemon, because the rules are the security boundary:
//! what counts as an account, and what counts as nobody. Two copies of those
//! rules would be two chances for one of them to admit a caller the other
//! refuses.

use axum::http::HeaderMap;

use super::session::Identity;

/// The provider's subject id — the account key.
pub const USER: &str = "x-tokera-user";
/// The issuer half of the account key.
pub const PROVIDER: &str = "x-tokera-provider";
pub const EMAIL: &str = "x-tokera-email";
pub const NAME: &str = "x-tokera-name";
pub const PICTURE: &str = "x-tokera-picture";

/// Why a request carries no identity: the gateway did not name the caller,
/// which means they are not signed in.
#[derive(Debug, PartialEq, Eq)]
pub struct NotSignedIn;

/// The identity of the caller, or [`NotSignedIn`].
///
/// `X-Tokera-User` is the provider's subject id and the only field that decides
/// *who* — it is the account key. The rest are descriptive: an email can be
/// reassigned and a display name can be anything, so neither is ever used to
/// look an account up.
pub fn identify(headers: &HeaderMap) -> Result<Identity, NotSignedIn> {
    let get = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned()
    };

    let sub = get(USER);
    let provider = get(PROVIDER);
    // The issuer half of the account key. Refused when absent rather than
    // assumed to be Google.
    //
    // Assuming would put every provider that ever forgets this header into one
    // namespace, which is the collision the field exists to prevent — and it
    // would do it silently, which is how the two accounts would stay merged.
    // A gateway that has not been updated fails sign-in visibly instead, and
    // is fixed by deploying it.
    if provider.is_empty() && !sub.is_empty() {
        tracing::warn!(
            "identity carried a subject but no provider — the gateway in front of this daemon \
             is older than the account key it is being asked for"
        );
        return Err(NotSignedIn);
    }
    // An empty subject is not an identity. The gateway omits the whole set for
    // an anonymous caller, but it also skips any single field whose value will
    // not fit in a header — so the absence of a *subject* is the only reliable
    // signal, and a blank one must not become an account.
    if sub.is_empty() {
        return Err(NotSignedIn);
    }

    Ok(Identity {
        provider,
        sub,
        email: get(EMAIL),
        name: get(NAME),
        picture: get(PICTURE),
        // The gateway owns expiry: it will not forward an identity it has
        // stopped honouring, so there is nothing here to expire. The field
        // exists because `Identity` is also the session cookie's payload.
        exp: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                v.parse().unwrap(),
            );
        }
        h
    }

    #[test]
    fn the_gateways_headers_name_the_caller() {
        let id = identify(&headers(&[
            (USER, "google-oauth2|1234"),
            (PROVIDER, "google"),
            (EMAIL, "wren@example.com"),
            (NAME, "Wren S"),
            (PICTURE, "https://example.com/a.png"),
        ]))
        .unwrap();

        assert_eq!(id.sub, "google-oauth2|1234");
        assert_eq!(id.provider, "google");
        assert_eq!(id.email, "wren@example.com");
        assert_eq!(id.name, "Wren S");
        assert_eq!(id.picture, "https://example.com/a.png");
    }

    #[test]
    fn no_headers_is_signed_out() {
        assert_eq!(identify(&HeaderMap::new()), Err(NotSignedIn));
    }

    /// The subject is what decides who. A caller the gateway did not name is
    /// anonymous however much else it sent.
    #[test]
    fn a_blank_subject_never_becomes_an_account() {
        assert_eq!(
            identify(&headers(&[(USER, ""), (EMAIL, "admin@tokera.com")])),
            Err(NotSignedIn)
        );
        assert_eq!(
            identify(&headers(&[(EMAIL, "admin@tokera.com"), (NAME, "Admin")])),
            Err(NotSignedIn)
        );
    }

    /// The descriptive fields are optional; the subject and its issuer are not.
    #[test]
    fn an_identity_survives_a_field_the_gateway_could_not_forward() {
        let id = identify(&headers(&[(USER, "google-1"), (PROVIDER, "google")])).unwrap();
        assert_eq!(id.sub, "google-1");
        assert_eq!(id.provider, "google");
        assert!(id.email.is_empty());
        assert!(id.picture.is_empty());
    }

    /// **The account key is issuer and subject.** A subject arriving without
    /// its issuer is refused rather than assumed to be Google.
    #[test]
    fn a_subject_without_its_issuer_is_not_an_identity() {
        assert_eq!(
            identify(&headers(&[(USER, "google-1"), (EMAIL, "a@b.c")])),
            Err(NotSignedIn)
        );
        assert_eq!(
            identify(&headers(&[(USER, "google-1"), (PROVIDER, "")])),
            Err(NotSignedIn)
        );
        // And an issuer with no subject is still just anonymous.
        assert_eq!(
            identify(&headers(&[(PROVIDER, "google")])),
            Err(NotSignedIn)
        );
    }
}
