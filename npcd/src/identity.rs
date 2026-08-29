//! Who is asking — read from the headers the gateway sets.
//!
//! Sign-in happens once, at the gateway, for the whole estate. It runs the OIDC
//! exchange and issues a session cookie on `.tokera.com`, which is what makes
//! one sign-in carry to tokera.com, code. and bot. without any of them taking
//! part. On every request it resolves that cookie and forwards the result on
//! `X-Tokera-User` / `-Email` / `-Name` / `-Picture`, having first cleared
//! whatever the client sent under those names.
//!
//! This daemon reads them and believes them. That is sound for exactly one
//! reason, and it is a property of the deployment rather than of this code:
//! `npcd` binds an address reachable only through the gateway, and says so by
//! calling [`web::Builder::behind_gateway`]. Without that call `web` clears
//! these headers on ingress, and nothing here would ever identify anybody — so
//! the trust is declared in one greppable place rather than assumed here.
//!
//! There is no key, no signature and no secret to distribute. An earlier design
//! verified `X-Tokera-Assertion` against the gateway's session key, which meant
//! copying that key to every daemon — and since the assertion *is* the session
//! cookie, it also handed each daemon the means to mint sessions for the whole
//! estate. A verifying credential that doubles as a minting credential is worse
//! than the header trust it replaced, so it is gone.

use axum::http::HeaderMap;
use web::auth::session::Identity;
use web::auth::{Role, Roles};

/// The headers the gateway speaks identity in.
const USER: &str = "x-tokera-user";
const PROVIDER: &str = "x-tokera-provider";
const EMAIL: &str = "x-tokera-email";
const NAME: &str = "x-tokera-name";
const PICTURE: &str = "x-tokera-picture";

/// Why a request carries no identity.
///
/// One variant, because there is now only one way to be anonymous: the gateway
/// did not name you, which means you are not signed in. The daemon has no
/// configuration that could be missing and no key that could be wrong.
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

/// The caller: who the gateway said they are, and what that lets them do.
///
/// Every route in this daemon starts by building one. Nothing reads the
/// headers directly and nothing decides access from anything else, so "what may
/// this request do" has exactly one answer computed in exactly one place.
pub struct Caller {
    identity: Option<Identity>,
    pub role: Role,
}

impl Caller {
    /// Read the gateway's headers and resolve the role from the config.
    ///
    /// Never fails: not being signed in is [`Role::Unauthenticated`], which is
    /// a role like any other and enough for every read route.
    pub fn read(headers: &HeaderMap, roles: &Roles) -> Self {
        let identity = identify(headers).ok();
        let role = roles.of(identity.as_ref());
        Self { identity, role }
    }

    /// The identity, consumed. `None` exactly when the role is
    /// [`Role::Unauthenticated`] — the two cannot disagree, because the role
    /// was derived from this field — so a caller that has already cleared a bar
    /// of [`Role::User`] can rely on it being `Some`.
    pub fn into_identity(self) -> Option<Identity> {
        self.identity
    }
}

/// Establish the caller, or produce the refusal that says why not.
///
/// The two refusals are different answers and are not interchangeable:
///
/// - **401** — nobody is signed in. Signing in would fix it.
/// - **403** — somebody is signed in and it is not enough. Signing in again
///   will not fix it, and telling them to try is how an operator wastes an
///   afternoon on a permissions problem that was never a session problem.
///
/// Boxed because a `Response` is large and this is the cold path.
pub fn require(
    headers: &HeaderMap,
    roles: &Roles,
    min: Role,
) -> Result<Caller, Box<axum::response::Response>> {
    let caller = Caller::read(headers, roles);
    if caller.role.at_least(min) {
        return Ok(caller);
    }
    Err(Box::new(refuse(caller.role, min)))
}

/// The body for a refused request.
///
/// It names the role required and the role held, because the alternative is an
/// operator guessing which of the two they got wrong. Neither is a secret: the
/// caller already knows who they signed in as, and the required level is in the
/// documented API surface.
fn refuse(held: Role, min: Role) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::Json;
    use serde_json::json;

    let status = if held == Role::Unauthenticated {
        StatusCode::UNAUTHORIZED
    } else {
        StatusCode::FORBIDDEN
    };
    let error = if status == StatusCode::UNAUTHORIZED {
        "unauthorized"
    } else {
        "forbidden"
    };
    (
        status,
        Json(json!({
            "error": error,
            "detail": format!("this needs the `{min}` role; you have `{held}`"),
            "required_role": min,
            "role": held,
        })),
    )
        .into_response()
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
    ///
    /// Assuming would put every provider that ever forgets this header into
    /// one namespace — the collision the field exists to prevent — and would do
    /// it silently. A gateway older than this daemon fails sign-in visibly
    /// instead, which is fixed by deploying it.
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
