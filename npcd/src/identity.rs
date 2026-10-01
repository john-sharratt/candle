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
use web::auth::forwarded::{EMAIL, NAME, PICTURE, PROVIDER, USER};
use web::auth::session::Identity;
use web::auth::{Role, Roles};

/// The parser for the gateway's headers is shared with every daemon behind it
/// (`web::auth::forwarded`), so the rules for who counts as an account are
/// written once.
pub use web::auth::forwarded::{identify, NotSignedIn};

/// Every header above, with the short label [`log_identity`] prints it under.
///
/// One list rather than a format string naming them again. A sixth header added
/// to [`identify`] but forgotten in the log would make that header invisible in
/// the one place built to make headers visible — the precise failure the log
/// exists to catch, reappearing inside it.
const IDENTITY_HEADERS: [(&str, &str); 5] = [
    ("user", USER),
    ("provider", PROVIDER),
    ("email", EMAIL),
    ("name", NAME),
    ("picture", PICTURE),
];

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

/// What a header carries, said without saying what it says.
///
/// Four answers, and the differences between them are the whole diagnostic
/// value — each points at a different fault, and collapsing any two sends the
/// reader after the wrong one:
///
/// - `-` — absent. The sender does not know about this header at all, which
///   means it predates it.
/// - `EMPTY` — present and blank. The sender knows the header but had nothing
///   to put in it; a session minted before the field existed reads exactly so.
/// - `BAD` — present and not UTF-8. Reported apart from absent because a
///   mangled value would otherwise read as a sender that never sent one.
/// - `set` — present and carries something.
///
/// The value itself is never returned. The subject is a durable account
/// identifier and the email address is personal data; both would be sitting in
/// a log file for a question that is about routing, not about the person.
fn header_state(raw: Option<&axum::http::HeaderValue>) -> &'static str {
    match raw {
        None => "-",
        Some(v) => match v.to_str() {
            Err(_) => "BAD",
            Ok(s) if s.trim().is_empty() => "EMPTY",
            Ok(_) => "set",
        },
    }
}

/// Log what the gateway put on this request, and the role it resolved to.
///
/// Sign-in crosses two processes on two machines, and when it fails each side
/// looks correct alone: the gateway holds a session, the daemon answers `401`,
/// and nothing states what actually arrived between them. A single missing
/// header — `x-tokera-provider`, say, which is half the account key — produces
/// exactly that, and is invisible from either end.
///
/// The role comes from [`Caller::read`], the same call the guard makes, so the
/// line cannot disagree with the verdict the request actually receives.
/// See [`header_state`] for what is reported and what is deliberately not.
pub fn log_identity(headers: &HeaderMap, roles: &Roles, method: &str, path: &str) {
    let caller = Caller::read(headers, roles);
    let mut fields = String::new();
    for (label, header) in IDENTITY_HEADERS {
        if !fields.is_empty() {
            fields.push(' ');
        }
        fields.push_str(label);
        fields.push('=');
        fields.push_str(header_state(headers.get(header)));
    }
    tracing::info!("{method} {path} → role={} | {fields}", caller.role);
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
mod header_state_tests {
    use super::*;
    use axum::http::HeaderValue;

    /// Absent and present-but-blank are different faults, and telling them
    /// apart is the whole reason this exists. A gateway too old to know the
    /// header sends nothing; a gateway that knows it but holds a session minted
    /// before the field existed sends it empty. Both refuse the request
    /// identically, so the log line is the only thing that separates them —
    /// which is exactly how a live sign-in failure was diagnosed.
    #[test]
    fn absent_and_empty_are_never_the_same_answer() {
        assert_eq!(header_state(None), "-");
        assert_eq!(header_state(Some(&HeaderValue::from_static(""))), "EMPTY");
        assert_eq!(
            header_state(Some(&HeaderValue::from_static("   "))),
            "EMPTY"
        );
        assert_eq!(
            header_state(Some(&HeaderValue::from_static("google"))),
            "set"
        );
    }

    /// A value that is not UTF-8 is its own answer. Folding it into `-` would
    /// report a mangled header as one that was never sent, pointing the reader
    /// at the sender when the fault is on the wire.
    #[test]
    fn a_non_utf8_value_is_not_reported_as_absent() {
        let bad = HeaderValue::from_bytes(&[0xff, 0xfe]).expect("valid header bytes");
        assert_eq!(header_state(Some(&bad)), "BAD");
        assert_ne!(header_state(Some(&bad)), header_state(None));
    }

    /// The log covers every header `identify` reads, and each is labelled once.
    ///
    /// A header this daemon acts on but does not print would be invisible in
    /// the one place built to make headers visible — which is how the missing
    /// `provider` stayed unexplained across two gateway redeploys.
    #[test]
    fn the_log_covers_every_header_identity_is_built_from() {
        let listed: Vec<&str> = IDENTITY_HEADERS.iter().map(|(_, h)| *h).collect();
        for header in [USER, PROVIDER, EMAIL, NAME, PICTURE] {
            assert!(
                listed.contains(&header),
                "{header} is read but never logged"
            );
        }
        let labels: Vec<&str> = IDENTITY_HEADERS.iter().map(|(l, _)| *l).collect();
        for (i, label) in labels.iter().enumerate() {
            assert!(
                !labels[i + 1..].contains(label),
                "two headers both log as `{label}`"
            );
        }
    }

    /// Whatever the state, the value never appears in it — the subject is an
    /// account key and the email is personal data.
    #[test]
    fn the_value_never_leaks_into_the_state() {
        let secret = "111930197703828817752";
        let state = header_state(Some(&HeaderValue::from_static(secret)));
        assert_eq!(state, "set");
        assert!(!state.contains(secret));
        // Every answer is one of four fixed words, so there is no path by which
        // a value could reach the log at all.
        assert!(["-", "EMPTY", "BAD", "set"].contains(&state));
    }
}
