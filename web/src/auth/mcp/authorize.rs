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
//! 3. **The user is asked** — if the role table names them: a guest is refused
//!    here, before any client document is fetched (`docs/guest_access.md`
//!    §6). A client document is anyone's to write, so a code is never issued
//!    unasked: a link to this endpoint naming an attacker's client would
//!    otherwise hand a signed-in user's code to the attacker. The consent page
//!    names the client and where it will be sent, and posts back the consent
//!    it showed — signed, and bound to the person it was shown to.
//! 4. `POST` with the answer: a code to the redirect, or `access_denied`. The
//!    role is read again, so a person taken off the table between the page
//!    and the answer is refused.
//!
//! Each step logs on `web::auth::mcp` — who, which client, where to, and the
//! outcome; never a code.

use std::collections::HashMap;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use reqwest::Url;

use super::client::{self, ClientDoc};
use super::grants::{
    self, Code, Consent, Grant, Kind, CODE_TTL_SECS, CONSENT_TTL_SECS, REFRESH_TTL_SECS,
};
use super::metadata::MCP_PATH;
use super::page::respond;
use crate::auth::oidc::urlencode;
use crate::auth::session::{Identity, Key};
use crate::auth::{Role, Roles};
use crate::config::covers;
use crate::html::escape;

/// What a guest is told at the consent page.
const NOT_ALLOWED: &str =
    "this account is not on the list of people who may use Zen; ask the estate's owner to add it";

/// Whether the role table lets `identity` connect a client: `User` and above.
pub fn may_connect(roles: &Roles, identity: &Identity) -> bool {
    roles.of(Some(identity)).at_least(Role::User)
}

/// Where a redirect sends a browser, by host — what the log and the consent
/// page name rather than the whole redirect, whose query a client writes.
fn host_of(redirect: &str) -> String {
    Url::parse(redirect)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_else(|| "(no readable host)".to_string())
}

/// A client's authorization request, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub client_id: String,
    pub redirect_uri: String,
    pub challenge: String,
    pub state: Option<String>,
    pub resource: String,
    /// The scopes asked for, of a site's `/api`; none for `/mcp`.
    pub scopes: Vec<String>,
}

/// A site's `/api` as a token may be asked for it: the resource, and the
/// scopes the site names (`docs/battlecities_app.md` §6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Api {
    pub resource: String,
    pub scopes: Vec<String>,
}

impl Api {
    /// Whether `query` asks for this API.
    pub fn asked_by(&self, query: &HashMap<String, String>) -> bool {
        query.get("resource") == Some(&self.resource)
    }

    /// The scopes `query` asks for, each one this API names: a request for
    /// none, or for one it does not name, is refused.
    pub fn scopes_asked(&self, query: &HashMap<String, String>) -> Result<Vec<String>, String> {
        let offered = || self.scopes.join(", ");
        let mut asked: Vec<String> = Vec::new();
        for scope in query
            .get("scope")
            .map_or("", String::as_str)
            .split_whitespace()
        {
            if !self.scopes.iter().any(|s| s == scope) {
                return Err(format!(
                    "`{scope}` is not a scope of `{}`: {}",
                    self.resource,
                    offered()
                ));
            }
            if !asked.iter().any(|s| s == scope) {
                asked.push(scope.to_string());
            }
        }
        if asked.is_empty() {
            return Err(format!("`scope` is missing: one or more of {}", offered()));
        }
        Ok(asked)
    }

    /// Check `query` as a request for this API, in the estate `estate`.
    pub fn parse(&self, query: &HashMap<String, String>, estate: &str) -> Result<Request, String> {
        let scopes = self.scopes_asked(query)?;
        Ok(Request {
            scopes,
            ..parse(query, &self.resource, estate)?
        })
    }
}

/// Whether `asked` is the `/mcp` of a site in the estate `estate` (a cookie
/// domain), on `resource`'s scheme — a resource this gateway serves and signs
/// tokens for, as it does this origin's own. A client may reach the sign-in on
/// one of the estate's hosts while asking for another's `/mcp`: claude.ai keeps
/// the authorization server it first discovered for a connector, so a
/// connector whose URL moved from `tokera.com/mcp` to `code.tokera.com/mcp`
/// signs in at the first and asks for the second. The token is bound to the
/// resource asked for, and is honoured on that origin's `/mcp` alone.
fn in_estate(asked: &str, resource: &str, estate: &str) -> bool {
    let (Ok(asked), Ok(ours)) = (Url::parse(asked), Url::parse(resource)) else {
        return false;
    };
    asked.scheme() == ours.scheme()
        && asked.path() == MCP_PATH
        && asked.query().is_none()
        && asked.fragment().is_none()
        && asked.username().is_empty()
        && asked.password().is_none()
        && asked.port().is_none()
        && asked.host_str().is_some_and(|host| covers(estate, host))
}

/// Check `query` as a request for `resource` — the `/mcp` this origin serves —
/// or for another site's `/mcp` in the estate `estate`.
pub fn parse(
    query: &HashMap<String, String>,
    resource: &str,
    estate: &str,
) -> Result<Request, String> {
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
    let resource = match get("resource") {
        None => resource,
        Some(asked) if asked == resource || in_estate(asked, resource, estate) => asked,
        Some(asked) => {
            return Err(format!(
                "this server issues tokens for `{resource}`, not `{asked}`"
            ));
        }
    };
    Ok(Request {
        client_id: client_id.to_string(),
        redirect_uri: redirect_uri.to_string(),
        challenge: challenge.to_string(),
        state: get("state").map(str::to_string),
        resource: resource.to_string(),
        scopes: Vec::new(),
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
    let to = host_of(&consent.grant.redirect_uri);
    let who = if consent.grant.email.is_empty() {
        consent.grant.name.clone()
    } else {
        consent.grant.email.clone()
    };
    format!(
        r#"<div class="body">
<p class="whisper">The Citadel stirs. Another mind asks to be joined.</p>
<h1><span class="client">{name}</span> wants to use Zen</h1>
<p class="lede">It will act as you, through the tools Zen gives it.</p>
<div class="who"><span class="dot" aria-hidden="true"></span><span>Signed in as <b>{who}</b></span></div>
<ul class="can">
<li>Start and read your threads</li>
<li>Run tasks on Zen's workers, and answer their questions</li>
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

/// The consent card for a site's `/api`: who asks, for which site, the scopes
/// it asks for by name, as whom, and where it will be sent.
pub fn api_card(consent: &Consent, sealed: &str) -> String {
    let to = host_of(&consent.grant.redirect_uri);
    let site = host_of(&consent.grant.resource);
    let who = if consent.grant.email.is_empty() {
        consent.grant.name.clone()
    } else {
        consent.grant.email.clone()
    };
    let scopes: String = consent
        .grant
        .scopes
        .iter()
        .map(|s| format!("<li><code>{}</code></li>\n", escape(s)))
        .collect();
    format!(
        r#"<div class="body">
<p class="whisper">Another program asks to act for you.</p>
<h1><span class="client">{name}</span> wants to use your account on {site}</h1>
<p class="lede">It will act as you there, within what it asks for:</p>
<div class="who"><span class="dot" aria-hidden="true"></span><span>Signed in as <b>{who}</b></span></div>
<ul class="can">
{scopes}</ul>
<p class="to">You will be sent back to <code>{to}</code>.</p>
<form method="post" action="/oauth/authorize" class="actions">
<input type="hidden" name="consent" value="{sealed}">
<button type="submit" name="decision" value="deny" class="deny">Deny</button>
<button type="submit" name="decision" value="allow" class="allow">Allow</button>
</form>
</div>
<div class="foot">Only allow a client you started yourself. Access lasts {days} days, and each use renews it.</div>"#,
        name = escape(&consent.client_name),
        site = escape(&site),
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
<p class="whisper">The Citadel's gate stays sealed.</p>
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
/// `identity`. `resource` is this origin's `/mcp`, `api` its `/api` when the
/// site names scopes for one, and `estate` the cookie domain it signs in on.
///
/// A request for the site's `/api` is anyone's who signs in: the site's own
/// accounts decide what a person may do there, so the role table, which says
/// who may use Zen, is not asked.
#[allow(clippy::too_many_arguments)]
pub async fn ask(
    key: &Key,
    roles: &Roles,
    identity: Option<Identity>,
    path_and_query: &str,
    query: &HashMap<String, String>,
    resource: &str,
    api: Option<&Api>,
    estate: &str,
    now: u64,
) -> Response {
    let for_api = api.filter(|api| api.asked_by(query));
    let parsed = match for_api {
        Some(api) => api.parse(query, estate),
        None => parse(query, resource, estate),
    };
    let req = match parsed {
        Ok(req) => req,
        Err(why) => return page(StatusCode::BAD_REQUEST, &why),
    };
    // Signed in first: the client's document is a URL the request names, and
    // the gateway fetches it for a signed-in user only.
    let Some(identity) = identity else {
        return see_other(&format!("/auth/login?next={}", urlencode(path_and_query)));
    };
    // And, for `/mcp`, named in the role table: a guest fetches nothing
    // either.
    if for_api.is_none() && !may_connect(roles, &identity) {
        tracing::info!(
            target: "web::auth::mcp",
            email = %identity.email,
            client = %req.client_id,
            to = %host_of(&req.redirect_uri),
            "consent refused: not on the role table"
        );
        return page(StatusCode::FORBIDDEN, NOT_ALLOWED);
    }
    let doc: ClientDoc = match client::fetch(&req.client_id).await {
        Ok(doc) => doc,
        Err(why) => {
            tracing::info!(
                target: "web::auth::mcp",
                email = %identity.email,
                client = %req.client_id,
                error = %why,
                "consent refused: the client's document could not be read"
            );
            return page(StatusCode::BAD_REQUEST, &why);
        }
    };
    if !client::allows_redirect(&doc, &req.redirect_uri) {
        tracing::info!(
            target: "web::auth::mcp",
            email = %identity.email,
            client = %req.client_id,
            to = %host_of(&req.redirect_uri),
            "consent refused: the redirect is not the client's"
        );
        return page(
            StatusCode::BAD_REQUEST,
            &format!(
                "the client's document does not list `{}` as one of its redirects",
                req.redirect_uri
            ),
        );
    }
    tracing::info!(
        target: "web::auth::mcp",
        email = %identity.email,
        client = %req.client_id,
        name = %doc.display_name(),
        to = %host_of(&req.redirect_uri),
        "consent shown"
    );
    let consent = Consent {
        grant: Grant {
            client_id: req.client_id,
            redirect_uri: req.redirect_uri,
            resource: req.resource,
            provider: identity.provider,
            sub: identity.sub,
            email: identity.email,
            name: identity.name,
            scopes: req.scopes,
        },
        challenge: req.challenge,
        state: req.state,
        client_name: doc.display_name(),
    };
    let sealed = grants::seal(key, Kind::McpConsent, now + CONSENT_TTL_SECS, &consent);
    match consent.grant.is_api() {
        true => respond(
            StatusCode::OK,
            "Allow access to your account",
            &api_card(&consent, &sealed),
        ),
        false => respond(
            StatusCode::OK,
            "Allow access to zend",
            &consent_card(&consent, &sealed),
        ),
    }
}

/// Step 4: the answer `identity` gave.
pub fn decide(
    key: &Key,
    roles: &Roles,
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
    if !consent.grant.is_api() && !may_connect(roles, &identity) {
        tracing::info!(
            target: "web::auth::mcp",
            email = %identity.email,
            client = %consent.grant.client_id,
            "consent refused: not on the role table"
        );
        return page(StatusCode::FORBIDDEN, NOT_ALLOWED);
    }
    let state = consent.state.as_deref();
    let allowed = form.get("decision").map(String::as_str) == Some("allow");
    tracing::info!(
        target: "web::auth::mcp",
        email = %identity.email,
        client = %consent.grant.client_id,
        to = %host_of(&consent.grant.redirect_uri),
        "consent {}",
        if allowed { "allowed" } else { "denied" }
    );
    if !allowed {
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
    use crate::auth::mcp::grants::testing::{grant, key, listed};
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

    const ESTATE: &str = ".tokera.com";

    fn asking_for(resource: &str) -> HashMap<String, String> {
        let mut q = good();
        q.retain(|(k, _)| *k != "resource");
        q.push(("resource", resource));
        query(&q)
    }

    /// **A sign-in on one of the estate's hosts may ask for another's `/mcp`**,
    /// as claude.ai does for a connector whose URL moved: the token is bound
    /// to the resource asked for. Nothing outside the estate, and nothing but
    /// `/mcp` exactly, is issued.
    #[test]
    fn a_sibling_sites_mcp_is_issued_for_and_nothing_else() {
        let apex = "https://tokera.com/mcp";
        let req = parse(&asking_for(RESOURCE), apex, ESTATE).unwrap();
        assert_eq!(req.resource, RESOURCE);
        let req = parse(&asking_for(apex), RESOURCE, ESTATE).unwrap();
        assert_eq!(req.resource, apex);
        for refused in [
            "https://evil.example/mcp",
            "https://tokera.com.evil.example/mcp",
            "https://eviltokera.com/mcp",
            "http://bot.tokera.com/mcp",
            "https://bot.tokera.com/mcp/x",
            "https://bot.tokera.com/other",
            "https://bot.tokera.com/mcp?x=1",
            "https://bot.tokera.com:8443/mcp",
            "https://u@bot.tokera.com/mcp",
            "not a url",
        ] {
            let err = parse(&asking_for(refused), apex, ESTATE).unwrap_err();
            assert!(err.contains(refused), "{refused}: {err}");
        }
        let req = parse(&asking_for("https://bot.tokera.com/mcp"), apex, ESTATE).unwrap();
        assert_eq!(req.resource, "https://bot.tokera.com/mcp");
        // Another estate's gateway origin asks within its own estate only.
        assert!(parse(
            &asking_for(RESOURCE),
            "https://yenbot.com.au/mcp",
            ".yenbot.com.au"
        )
        .is_err());
    }

    #[test]
    fn a_good_request_reads_and_each_missing_piece_is_named() {
        let req = parse(&query(&good()), RESOURCE, ESTATE).unwrap();
        assert_eq!(req.state.as_deref(), Some("xyz"));
        for (drop, why) in [
            ("response_type", "response_type"),
            ("client_id", "client_id"),
            ("redirect_uri", "redirect_uri"),
            ("code_challenge", "PKCE"),
            ("code_challenge_method", "S256"),
        ] {
            let q: Vec<_> = good().into_iter().filter(|(k, _)| *k != drop).collect();
            let err = parse(&query(&q), RESOURCE, ESTATE).unwrap_err();
            assert!(err.contains(why), "{drop}: {err}");
        }
        let mut other = good();
        other.retain(|(k, _)| *k != "resource");
        other.push(("resource", "https://evil.example/mcp"));
        assert!(parse(&query(&other), RESOURCE, ESTATE).is_err());
    }

    fn game() -> Api {
        Api {
            resource: "https://battlecities.net/api".into(),
            scopes: ["read", "chat", "play", "manage", "account", "admin"]
                .map(str::to_string)
                .to_vec(),
        }
    }

    fn asking_the_game(scope: Option<&str>) -> HashMap<String, String> {
        let mut q = asking_for("https://battlecities.net/api");
        if let Some(scope) = scope {
            q.insert("scope".into(), scope.into());
        }
        q
    }

    /// **A site's `/api` is asked for by its resource, with scopes it names**:
    /// each asked once, in the order asked; none, or one the site does not
    /// name, is refused, and `/mcp`'s request is not the API's.
    #[test]
    fn an_api_is_asked_for_with_scopes_it_names() {
        let api = game();
        assert!(api.asked_by(&asking_the_game(None)));
        assert!(!api.asked_by(&asking_for(RESOURCE)));
        assert!(!api.asked_by(&asking_for("https://battlecities.net/api/v1")));

        let estate = ".battlecities.net";
        let req = api
            .parse(&asking_the_game(Some("play read  play")), estate)
            .unwrap();
        assert_eq!(req.resource, "https://battlecities.net/api");
        assert_eq!(req.scopes, ["play", "read"]);
        assert_eq!(req.state.as_deref(), Some("xyz"));

        let none = api.parse(&asking_the_game(None), estate).unwrap_err();
        assert_eq!(
            none,
            "`scope` is missing: one or more of read, chat, play, manage, account, admin"
        );
        let other = api
            .parse(&asking_the_game(Some("read zend")), estate)
            .unwrap_err();
        assert!(other.starts_with("`zend` is not a scope of"), "{other}");
        // A request for `/mcp` names no scopes, whatever it says.
        let mcp = parse(&query(&good()), RESOURCE, ESTATE).unwrap();
        assert!(mcp.scopes.is_empty());
    }

    /// **Anyone who signs in may grant a client their account on a site's
    /// API**; `/mcp` stays the role table's. With nobody on the table, the
    /// API's consent is answered with a code and `/mcp`'s is refused.
    #[test]
    fn an_apis_consent_is_anyones_and_mcps_is_the_role_tables() {
        use crate::auth::mcp::grants::testing::api_grant;
        let nobody_listed = Roles::default();
        let answer = |grant: Grant| {
            let consent = Consent {
                grant: grant.clone(),
                challenge: "abc".into(),
                state: None,
                client_name: "A client".into(),
            };
            let sealed = grants::seal(&key(), Kind::McpConsent, 600, &consent);
            let form = query(&[("consent", sealed.as_str()), ("decision", "allow")]);
            decide(
                &key(),
                &nobody_listed,
                Some(grant.identity(900)),
                &form,
                "https://battlecities.net",
                100,
            )
        };
        let api = answer(api_grant());
        assert_eq!(api.status(), StatusCode::SEE_OTHER);
        let to = api.headers()[header::LOCATION]
            .to_str()
            .unwrap()
            .to_string();
        assert!(to.contains("code="), "{to}");
        assert_eq!(answer(grant()).status(), StatusCode::FORBIDDEN);
    }

    /// The API's consent names the site, the client and each scope asked.
    #[test]
    fn an_apis_consent_names_the_site_and_the_scopes() {
        use crate::auth::mcp::grants::testing::api_grant;
        let consent = Consent {
            grant: api_grant(),
            challenge: "abc".into(),
            state: None,
            client_name: "A <client>".into(),
        };
        let card = api_card(&consent, "sealed");
        assert!(
            card.contains("A &lt;client&gt;</span> wants to use your account on battlecities.net")
        );
        assert!(card.contains("<li><code>read</code></li>\n<li><code>play</code></li>"));
        assert!(!card.contains("Zen"));
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

    /// **A guest is refused at the consent page and at the answer**: signed in
    /// is not enough to connect a client. Refused before any client document
    /// is fetched.
    #[tokio::test]
    async fn a_guest_cannot_connect_a_client() {
        let guest = grant().identity(u64::MAX);
        let res = ask(
            &key(),
            &Roles::default(),
            Some(guest.clone()),
            "/oauth/authorize",
            &query(&good()),
            RESOURCE,
            ESTATE,
            100,
        )
        .await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);

        let allow = query(&[("consent", &sealed(100)), ("decision", "allow")]);
        let res = decide(
            &key(),
            &Roles::default(),
            Some(guest),
            &allow,
            "https://i",
            101,
        );
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        assert!(
            res.headers().get(header::LOCATION).is_none(),
            "no code went out"
        );
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
        let res = decide(
            &key(),
            &listed(),
            Some(grant().identity(9)),
            &form,
            "https://i",
            101,
        );
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
        let res = decide(
            &key(),
            &listed(),
            Some(grant().identity(9)),
            &deny,
            "https://i",
            101,
        );
        assert!(location(&res).contains("error=access_denied&state=s"));
        let allow = query(&[("consent", &sealed(100)), ("decision", "allow")]);
        let mut intruder = grant().identity(9);
        intruder.sub = "intruder".into();
        let res = decide(&key(), &listed(), Some(intruder), &allow, "https://i", 101);
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let res = decide(&key(), &listed(), None, &allow, "https://i", 101);
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let late = decide(
            &key(),
            &listed(),
            Some(grant().identity(9)),
            &allow,
            "https://i",
            100 + CONSENT_TTL_SECS,
        );
        assert_eq!(late.status(), StatusCode::BAD_REQUEST);
    }
}
