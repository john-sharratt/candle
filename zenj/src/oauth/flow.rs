//! Signing in to Spotify and TIDAL as a person (§4.1): authorization code
//! with PKCE through a loopback redirect on this machine, then refresh
//! tokens kept and used for as long as the service honours them.
//!
//! `zenj login <account>` runs [`login`] once per account at the console: it
//! prints the address to open, waits for the redirect, and keeps the token.
//! Every call after uses [`Account::access`], which refreshes on its own.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tokio::sync::Mutex;
use url::Url;

use super::pkce;
use super::token::{self, Answer, Token};

/// A service's sign-in endpoints and the app ZenJ signs in as.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Provider {
    pub authorize: String,
    pub token: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    /// `http://127.0.0.1:<port>/callback`: a loopback address, which both
    /// services take without a secret.
    pub redirect: String,
}

/// The address a person opens to sign in.
pub fn authorize_url(p: &Provider, challenge: &str, state: &str) -> Result<Url, String> {
    if p.client_id.trim().is_empty() {
        return Err(format!(
            "no app to sign in as at {}: register one in the service's developer dashboard, \
             with the redirect {}, and put its Client ID in zenj.yaml",
            p.authorize, p.redirect
        ));
    }
    let mut u = Url::parse(&p.authorize).map_err(|e| format!("{}: {e}", p.authorize))?;
    u.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &p.client_id)
        .append_pair("redirect_uri", &p.redirect)
        .append_pair("scope", &p.scopes.join(" "))
        .append_pair("code_challenge_method", "S256")
        .append_pair("code_challenge", challenge)
        .append_pair("state", state);
    Ok(u)
}

/// The code a redirect's request line carries, when its state is `state`.
pub fn code_of(request_line: &str, state: &str) -> Result<String, String> {
    let target = request_line
        .split_whitespace()
        .nth(1)
        .ok_or("the redirect carried no request")?;
    let u = Url::parse(&format!("http://127.0.0.1{target}")).map_err(|e| e.to_string())?;
    let mut code = None;
    let mut said_state = None;
    let mut error = None;
    for (k, v) in u.query_pairs() {
        match k.as_ref() {
            "code" => code = Some(v.into_owned()),
            "state" => said_state = Some(v.into_owned()),
            "error" => error = Some(v.into_owned()),
            _ => {}
        }
    }
    if let Some(e) = error {
        return Err(format!("the sign-in was refused: {e}"));
    }
    if said_state.as_deref() != Some(state) {
        return Err("the redirect's state is not this sign-in's".into());
    }
    code.ok_or_else(|| "the redirect carried no code".into())
}

/// Wait at `redirect`'s loopback port for the one redirect, and answer the
/// browser.
fn await_code(redirect: &str, state: &str) -> Result<String, String> {
    let u = Url::parse(redirect).map_err(|e| format!("{redirect}: {e}"))?;
    let port = u.port().ok_or("the redirect names no port")?;
    let listener =
        TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("127.0.0.1:{port}: {e}"))?;
    loop {
        let (mut stream, _) = listener.accept().map_err(|e| e.to_string())?;
        let mut line = String::new();
        BufReader::new(&stream)
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if !line.contains(u.path()) {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            continue;
        }
        let got = code_of(&line, state);
        let body = match &got {
            Ok(_) => "Signed in. ZenJ has it from here; this tab can close.",
            Err(_) => "The sign-in did not complete; see the console.",
        };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        return got;
    }
}

async fn post_form(p: &Provider, form: &[(&str, &str)]) -> Result<Answer, String> {
    let res = reqwest::Client::new()
        .post(&p.token)
        .form(form)
        .send()
        .await
        .map_err(|e| format!("{}: {e}", p.token))?;
    let status = res.status();
    let text = res.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("{}: {status}: {text}", p.token));
    }
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}: {text}", p.token))
}

/// Sign `name` in to `p` at the console, and keep the token under `data`.
pub async fn login(p: &Provider, data: &Path, name: &str) -> Result<Token, String> {
    let verifier = pkce::verifier();
    let state = pkce::state();
    let url = authorize_url(p, &pkce::challenge(&verifier), &state)?;
    println!("Open this to sign in as {name}:\n\n  {url}\n");
    let redirect = p.redirect.clone();
    let code = tokio::task::spawn_blocking(move || await_code(&redirect, &state))
        .await
        .map_err(|e| e.to_string())??;
    let answer = post_form(
        p,
        &[
            ("grant_type", "authorization_code"),
            ("code", &code),
            ("redirect_uri", &p.redirect),
            ("client_id", &p.client_id),
            ("code_verifier", &verifier),
        ],
    )
    .await?;
    let t = Token::from_answer(answer, token::now(), None)?;
    token::save(data, name, &t)?;
    Ok(t)
}

/// A signed-in account: its token, refreshed when it is stale.
pub struct Account {
    pub name: String,
    provider: Provider,
    data: PathBuf,
    token: Mutex<Token>,
}

impl Account {
    /// The account `name`, signed in before (`login`).
    pub fn open(provider: Provider, data: PathBuf, name: &str) -> Result<Account, String> {
        let t = token::load(&data, name)?;
        Ok(Account {
            name: name.to_string(),
            provider,
            data,
            token: Mutex::new(t),
        })
    }

    /// An access token good for the next couple of minutes.
    pub async fn access(&self) -> Result<String, String> {
        let mut t = self.token.lock().await;
        if t.stale(token::now()) {
            let answer = post_form(
                &self.provider,
                &[
                    ("grant_type", "refresh_token"),
                    ("refresh_token", &t.refresh),
                    ("client_id", &self.provider.client_id),
                ],
            )
            .await?;
            let fresh = Token::from_answer(answer, token::now(), Some(&t))?;
            token::save(&self.data, &self.name, &fresh)?;
            *t = fresh;
        }
        Ok(t.access.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spotify() -> Provider {
        Provider {
            authorize: "https://accounts.spotify.com/authorize".into(),
            token: "https://accounts.spotify.com/api/token".into(),
            client_id: "abc123".into(),
            scopes: vec!["playlist-read-private".into(), "user-library-read".into()],
            redirect: "http://127.0.0.1:43821/callback".into(),
        }
    }

    #[test]
    fn no_app_is_no_sign_in() {
        let p = Provider {
            client_id: " ".into(),
            ..spotify()
        };
        assert!(authorize_url(&p, "C", "S").unwrap_err().starts_with("no app to sign in as"));
    }

    #[test]
    fn the_sign_in_address_carries_the_challenge() {
        let u = authorize_url(&spotify(), "CH", "ST").unwrap();
        assert_eq!(
            u.as_str(),
            "https://accounts.spotify.com/authorize?response_type=code&client_id=abc123\
             &redirect_uri=http%3A%2F%2F127.0.0.1%3A43821%2Fcallback\
             &scope=playlist-read-private+user-library-read\
             &code_challenge_method=S256&code_challenge=CH&state=ST"
        );
    }

    /// **The redirect's code is taken only with this sign-in's state**; a
    /// refusal says why.
    #[test]
    fn the_code_is_taken_only_with_its_state() {
        assert_eq!(
            code_of("GET /callback?code=C0DE&state=ST HTTP/1.1", "ST"),
            Ok("C0DE".into())
        );
        assert_eq!(
            code_of("GET /callback?code=C0DE&state=XX HTTP/1.1", "ST"),
            Err("the redirect's state is not this sign-in's".into())
        );
        assert_eq!(
            code_of("GET /callback?error=access_denied&state=ST HTTP/1.1", "ST"),
            Err("the sign-in was refused: access_denied".into())
        );
        assert_eq!(
            code_of("GET /callback?state=ST HTTP/1.1", "ST"),
            Err("the redirect carried no code".into())
        );
    }
}
