//! `/oauth/authorize`: the browser leg of the sign-in
//! (`docs/claude_zend_gateway.md` §12).
//!
//! 1. `GET` with the client's request. The request is checked; then who the
//!    user is — the gateway's own session cookie. Nobody signed in is sent to
//!    `/auth/login`, which brings them back here.
//! 2. For a signed-in user only, the client's metadata document is fetched
//!    (`super::client`, fenced to public hosts) and the redirect matched
//!    against it — none of that redirects anywhere, since a redirect not yet
//!    checked is not one to send a browser to.
//! 3. **The user is asked.** A client document is anyone's to write, so a code
//!    is never issued unasked: a link to this endpoint naming an attacker's
//!    client would otherwise hand a signed-in user's code to the attacker. The
//!    consent page names the client and where it will be sent, and posts back
//!    the consent it showed — signed, and bound to the person it was shown to.
//! 4. `POST` with the answer: a code to the redirect, or `access_denied`.

use std::collections::HashMap;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use reqwest::Url;

use super::client::{self, ClientDoc};
use super::grants::{
    self, Code, Consent, Grant, Kind, CODE_TTL_SECS, CONSENT_TTL_SECS, REFRESH_TTL_SECS,
};
use super::page::respond;
use crate::auth::oidc::urlencode;
use crate::auth::session::{Identity, Key};
use crate::html::escape;

/// A client's authorization request, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub client_id: String,
    pub redirect_uri: String,
    pub challenge: String,
    pub state: Option<String>,
    pub resource: String,
}

/// Check `query` as a request for `resource` — the `/mcp` this origin serves.
pub fn parse(query: &HashMap<String, String>, resource: &str) -> Result<Request, String> {
    let get = |k: &str| query.get(k).map(String::as_str).filter(|v| !v.is_empty());
    if get("response_type") != Some("code") {
        return Err("`response_type` must be `code`".into());
    }
    let client_id = get("client_id").ok_or("`client_id` is missing")?;
    client::check_id(client_id)?;
    let redirect_uri = get("redirect_uri").ok_or("`redirect_uri` is missing")?;
    let challenge = get("code_challenge").ok_or("`code_challenge` is missing: PKCE is required")?;
    if get("code_challenge_method") != Some("S256") {
        return Err("`code_challenge_method` must be `S256`".into());
    }
    if let Some(asked) = get("resource") {
        if asked != resource {
            return Err(format!(
                "this server issues tokens for `{resource}`, not `{asked}`"
            ));
        }
    }
    Ok(Request {
        client_id: client_id.to_string(),
        redirect_uri: redirect_uri.to_string(),
        challenge: challenge.to_string(),
        state: get("state").map(str::to_string),
        resource: resource.to_string(),
    })
}

/// `url` with `pairs` added to its query.
pub fn with_query(url: &str, pairs: &[(&str, &str)]) -> String {
    let query: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
        .collect();
    let joiner = if url.contains('?') { '&' } else { '?' };
    format!("{url}{joiner}{}", query.join("&"))
}

/// Where the browser goes after the user answers: the code, or the refusal,
/// with the request's `state` and this server as the issuer (RFC 9207).
pub fn answer_redirect(
    redirect_uri: &str,
    code: Option<&str>,
    state: Option<&str>,
    issuer: &str,
) -> String {
    let mut pairs: Vec<(&str, &str)> = match code {
        Some(code) => vec![("code", code)],
        None => vec![("error", "access_denied")],
    };
    if let Some(state) = state {
        pairs.push(("state", state));
    }
    pairs.push(("iss", issuer));
    with_query(redirect_uri, &pairs)
}

/// The consent card: who asks, what it may do, as whom, and where it will be
/// sent. Nothing is focused, so no keystroke meant for another window can
/// press Allow.
pub fn consent_card(consent: &Consent, sealed: &str) -> String {
    let to = Url::parse(&consent.grant.redirect_uri)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_else(|| consent.grant.redirect_uri.clone());
    let who = if consent.grant.email.is_empty() {
        consent.grant.name.clone()
    } else {
        consent.grant.email.clone()
    };
    format!(
        r#"<div class="body">
<h1><span class="client">{name}</span> wants to use zend</h1>
<p class="lede">It will act as you, through the tools zend gives it.</p>
<div class="who"><span class="dot" aria-hidden="true"></span><span>Signed in as <b>{who}</b></span></div>
<ul class="can">
<li>Start and read your threads</li>
<li>Run tasks on zend's workers, and answer their questions</li>
<li>Read the journals and the evidence behind them</li>
</ul>
<p class="to">You will be sent back to <code>{to}</code>.</p>
<form method="post" action="/oauth/authorize" class="actions">
<input type="hidden" name="consent" value="{sealed}">
<button type="submit" name="decision" value="deny" class="deny">Deny</button>
<button type="submit" name="decision" value="allow" class="allow">Allow</button>
</form>
</div>
<div class="foot">Only allow a client you started yourself. Access lasts {days} days, and each use renews it.</div>"#,
        name = escape(&consent.client_name),
        who = escape(&who),
        to = escape(&to),
        sealed = escape(sealed),
        days = REFRESH_TTL_SECS / 86_400,
    )
}

/// A refusal, in the same look: what went wrong, and that the sign-in starts
/// again from the client.
fn page(status: StatusCode, message: &str) -> Response {
    let card = format!(
        r#"<div class="body refused">
<h1>Sign-in refused</h1>
<p class="lede">{}</p>
</div>
<div class="foot">Start the sign-in again from the client that sent you here.</div>"#,
        escape(message)
    );
    respond(status, "Sign-in refused", &card)
}

fn see_other(location: &str) -> Response {
    (
        StatusCode::SEE_OTHER,
        [(header::LOCATION, location.to_string())],
    )
        .into_response()
}

/// Steps 1–3: the request checked, then the sign-in, or the consent page for
/// `identity`.
pub async fn ask(
    key: &Key,
    identity: Option<Identity>,
    path_and_query: &str,
    query: &HashMap<String, String>,
    resource: &str,
    now: u64,
) -> Response {
    let req = match parse(query, resource) {
        Ok(req) => req,
        Err(why) => return page(StatusCode::BAD_REQUEST, &why),
    };
    // Signed in first: the client's document is a URL the request names, and
    // the gateway fetches it for a signed-in user only.
    let Some(identity) = identity else {
        return see_other(&format!("/auth/login?next={}", urlencode(path_and_query)));
    };
    let doc: ClientDoc = match client::fetch(&req.client_id).await {
        Ok(doc) => doc,
        Err(why) => return page(StatusCode::BAD_REQUEST, &why),
    };
    if !client::allows_redirect(&doc, &req.redirect_uri) {
        return page(
            StatusCode::BAD_REQUEST,
            &format!(
                "the client's document does not list `{}` as one of its redirects",
                req.redirect_uri
            ),
        );
    }
    let consent = Consent {
        grant: Grant {
            client_id: req.client_id,
            redirect_uri: req.redirect_uri,
            resource: req.resource,
            provider: identity.provider,
            sub: identity.sub,
            email: identity.email,
            name: identity.name,
        },
        challenge: req.challenge,
        state: req.state,
        client_name: doc.display_name(),
    };
    let sealed = grants::seal(key, Kind::McpConsent, now + CONSENT_TTL_SECS, &consent);
    respond(
        StatusCode::OK,
        "Allow access to zend",
        &consent_card(&consent, &sealed),
    )
}

/// Step 4: the answer `identity` gave.
pub fn decide(
    key: &Key,
    identity: Option<Identity>,
    form: &HashMap<String, String>,
    issuer: &str,
    now: u64,
) -> Response {
    let Some(sealed) = form.get("consent") else {
        return page(StatusCode::BAD_REQUEST, "the answer names no consent");
    };
    let Some(identity) = identity else {
        return page(StatusCode::UNAUTHORIZED, "you are not signed in");
    };
    let Some(consent) = grants::open::<Consent>(key, Kind::McpConsent, sealed, now) else {
        return page(
            StatusCode::BAD_REQUEST,
            "this consent has expired or is not one this server showed; start the sign-in again",
        );
    };
    if !consent.grant.is_for(&identity) {
        return page(
            StatusCode::FORBIDDEN,
            "this consent was shown to someone else",
        );
    }
    let state = consent.state.as_deref();
    if form.get("decision").map(String::as_str) != Some("allow") {
        return see_other(&answer_redirect(
            &consent.grant.redirect_uri,
            None,
            state,
            issuer,
        ));
    }
    let code = Code {
        grant: consent.grant.clone(),
        challenge: consent.challenge.clone(),
    };
    let sealed_code = grants::seal(key, Kind::McpCode, now + CODE_TTL_SECS, &code);
    see_other(&answer_redirect(
        &consent.grant.redirect_uri,
        Some(&sealed_code),
        state,
        issuer,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::mcp::grants::testing::{grant, key};
    use crate::auth::mcp::page::PAGE_POLICY;

    const RESOURCE: &str = "https://code.tokera.com/mcp";

    fn query(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn good() -> Vec<(&'static str, &'static str)> {
        vec![
            ("response_type", "code"),
            (
                "client_id",
                "https://claude.ai/oauth/mcp-oauth-client-metadata",
            ),
            ("redirect_uri", "https://claude.ai/api/mcp/auth_callback"),
            ("code_challenge", "abc"),
            ("code_challenge_method", "S256"),
            ("state", "xyz"),
            ("resource", RESOURCE),
        ]
    }

    #[test]
    fn a_good_request_reads_and_each_missing_piece_is_named() {
        let req = parse(&query(&good()), RESOURCE).unwrap();
        assert_eq!(req.state.as_deref(), Some("xyz"));
        for (drop, why) in [
            ("response_type", "response_type"),
            ("client_id", "client_id"),
            ("redirect_uri", "redirect_uri"),
            ("code_challenge", "PKCE"),
            ("code_challenge_method", "S256"),
        ] {
            let q: Vec<_> = good().into_iter().filter(|(k, _)| *k != drop).collect();
            let err = parse(&query(&q), RESOURCE).unwrap_err();
            assert!(err.contains(why), "{drop}: {err}");
        }
        let mut other = good();
        other.retain(|(k, _)| *k != "resource");
        other.push(("resource", "https://evil.example/mcp"));
        assert!(parse(&query(&other), RESOURCE).is_err());
    }

    #[test]
    fn the_answer_carries_the_code_or_the_refusal_with_state_and_issuer() {
        assert_eq!(
            answer_redirect(
                "https://claude.ai/cb",
                Some("c1"),
                Some("s 1"),
                "https://code.tokera.com"
            ),
            "https://claude.ai/cb?code=c1&state=s%201&iss=https%3A%2F%2Fcode.tokera.com"
        );
        assert_eq!(
            answer_redirect("http://localhost:5/cb?x=1", None, None, "https://i"),
            "http://localhost:5/cb?x=1&error=access_denied&iss=https%3A%2F%2Fi"
        );
    }

    fn consent() -> Consent {
        Consent {
            grant: grant(),
            challenge: "c".into(),
            state: Some("s".into()),
            client_name: "Claude".into(),
        }
    }

    #[test]
    fn the_consent_card_escapes_what_the_client_wrote_and_focuses_nothing() {
        let c = Consent {
            client_name: "<script>x</script>".into(),
            ..consent()
        };
        let html = consent_card(&c, "id\"1");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains(r#"value="id&quot;1""#));
        assert!(html.contains("<code>claude.ai</code>"));
        assert!(html.contains("boss@example.com"));
        assert!(!html.contains("autofocus"));
        assert!(html.contains("Access lasts 30 days"));
    }

    #[test]
    fn a_refusal_goes_out_unframeable_and_uncached() {
        let res = page(StatusCode::BAD_REQUEST, "<no>");
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        assert_eq!(res.headers()[header::CONTENT_SECURITY_POLICY], PAGE_POLICY);
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
    }

    fn location(res: &Response) -> String {
        res.headers()[header::LOCATION]
            .to_str()
            .unwrap()
            .to_string()
    }

    fn sealed(now: u64) -> String {
        grants::seal(&key(), Kind::McpConsent, now + CONSENT_TTL_SECS, &consent())
    }

    #[test]
    fn allowing_sends_a_code_that_opens_as_one() {
        let form = query(&[("consent", &sealed(100)), ("decision", "allow")]);
        let res = decide(&key(), Some(grant().identity(9)), &form, "https://i", 101);
        assert_eq!(res.status(), StatusCode::SEE_OTHER);
        let loc = location(&res);
        assert!(loc.starts_with("https://claude.ai/api/mcp/auth_callback?code="));
        let code = Url::parse(&loc)
            .unwrap()
            .query_pairs()
            .find(|(k, _)| k == "code")
            .unwrap()
            .1
            .to_string();
        let opened: Code = grants::open(&key(), Kind::McpCode, &code, 102).unwrap();
        assert_eq!(opened.grant, grant());
    }

    #[test]
    fn denying_another_person_and_nobody_are_each_answered() {
        let deny = query(&[("consent", &sealed(100)), ("decision", "deny")]);
        let res = decide(&key(), Some(grant().identity(9)), &deny, "https://i", 101);
        assert!(location(&res).contains("error=access_denied&state=s"));
        let allow = query(&[("consent", &sealed(100)), ("decision", "allow")]);
        let mut intruder = grant().identity(9);
        intruder.sub = "intruder".into();
        let res = decide(&key(), Some(intruder), &allow, "https://i", 101);
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let res = decide(&key(), None, &allow, "https://i", 101);
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let late = decide(
            &key(),
            Some(grant().identity(9)),
            &allow,
            "https://i",
            100 + CONSENT_TTL_SECS,
        );
        assert_eq!(late.status(), StatusCode::BAD_REQUEST);
    }
}
