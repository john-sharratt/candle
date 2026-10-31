//! The Discord REST calls a bot's connect makes.
//!
//! Three, each authorised as the bot (`Authorization: Bot <token>`): who the
//! bot is (`GET /users/@me`), which application it belongs to
//! (`GET /applications/@me`), and its name and avatar (`PATCH /users/@me`).
//! The API's base is a field, Discord's own in production ([`PRODUCTION`]),
//! so a test points it at a fake server and asserts the exact requests.
//!
//! A refusal is answered once, as a [`DiscordError`]; nothing here retries.

use std::fmt::{self, Debug, Formatter};

use reqwest::header::{HeaderMap, AUTHORIZATION, RETRY_AFTER, USER_AGENT};
use reqwest::{Client as Http, Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::discord::error::DiscordError;

/// Discord's API, at the version this client speaks.
pub const PRODUCTION: &str = "https://discord.com/api/v10";

/// The `User-Agent` Discord asks a bot to send: `DiscordBot (<url>, <version>)`.
const AGENT: &str = concat!(
    "DiscordBot (https://tokera.com, ",
    env!("CARGO_PKG_VERSION"),
    ")"
);

/// How long a `429` without a `retry_after` is taken to ask for, in seconds.
const RETRY_UNSTATED_S: f64 = 1.0;

/// A bot's own user, as `GET /users/@me` answers it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BotUser {
    pub id: String,
    pub username: String,
    /// Absent on a user's account, which is the same as `false`.
    #[serde(default)]
    pub bot: bool,
}

/// A bot's application, as `GET /applications/@me` answers it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Application {
    pub id: String,
    pub name: String,
}

/// Discord's REST API, from one base.
#[derive(Clone)]
pub struct Client {
    http: Http,
    base: String,
}

impl Debug for Client {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").field("base", &self.base).finish()
    }
}

impl Client {
    /// Discord's own API.
    pub fn production() -> Self {
        Self::at(PRODUCTION)
    }

    /// The API at `base` — Discord's, or a test's fake of it.
    pub fn at(base: &str) -> Self {
        Self {
            http: Http::new(),
            base: base.trim_end_matches('/').to_string(),
        }
    }

    /// The bot `token` is, refusing a token whose user is not a bot.
    pub async fn me(&self, token: &str) -> Result<BotUser, DiscordError> {
        let user: BotUser = self.call(Method::GET, "/users/@me", token, None).await?;
        if !user.bot {
            return Err(DiscordError::NotABot);
        }
        Ok(user)
    }

    /// The application `token`'s bot belongs to.
    pub async fn application(&self, token: &str) -> Result<Application, DiscordError> {
        self.call(Method::GET, "/applications/@me", token, None)
            .await
    }

    /// Name the bot `username` and picture it with `avatar` (a data URI from
    /// [`crate::discord::profile::avatar`]), each only when given; the bot as
    /// it now is.
    pub async fn set_profile(
        &self,
        token: &str,
        username: Option<&str>,
        avatar: Option<&str>,
    ) -> Result<BotUser, DiscordError> {
        let mut body = Map::new();
        if let Some(username) = username {
            body.insert("username".into(), Value::from(username));
        }
        if let Some(avatar) = avatar {
            body.insert("avatar".into(), Value::from(avatar));
        }
        self.call(
            Method::PATCH,
            "/users/@me",
            token,
            Some(Value::Object(body)),
        )
        .await
    }

    /// One request, as the bot, and its answer read as `T`.
    async fn call<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        token: &str,
        body: Option<Value>,
    ) -> Result<T, DiscordError> {
        let mut request = self
            .http
            .request(method, format!("{}{path}", self.base))
            .header(AUTHORIZATION, format!("Bot {token}"))
            .header(USER_AGENT, AGENT);
        if let Some(body) = body {
            request = request.json(&body);
        }
        // Without the URL: reqwest names it in an error, and it is no secret,
        // but nothing about a failed call should be more than its kind.
        let response = request
            .send()
            .await
            .map_err(|e| DiscordError::Transport(e.without_url().to_string()))?;
        let status = response.status();
        let headers = response.headers().clone();
        let text = response
            .text()
            .await
            .map_err(|e| DiscordError::Transport(e.without_url().to_string()))?;
        if !status.is_success() {
            return Err(refusal(status, &headers, &text));
        }
        serde_json::from_str(&text)
            .map_err(|e| DiscordError::Transport(format!("an answer that is not Discord's: {e}")))
    }
}

/// What a failed status means: a rejected token, a rate limit and how long
/// it asks for, or a failure with what Discord said.
fn refusal(status: StatusCode, headers: &HeaderMap, text: &str) -> DiscordError {
    let body: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    match status {
        StatusCode::UNAUTHORIZED => DiscordError::TokenRejected,
        StatusCode::TOO_MANY_REQUESTS => DiscordError::RateLimited {
            retry_after_s: body["retry_after"]
                .as_f64()
                .or_else(|| {
                    headers
                        .get(RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(RETRY_UNSTATED_S),
        },
        _ => DiscordError::Status {
            status: status.as_u16(),
            detail: body["message"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| text.chars().take(200).collect()),
        },
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::discord::fake::{self, Mode, Seen};

    const TOKEN: &str = "MTEx.fake.token";

    fn seen(method: &str, path: &str, body: Value) -> Seen {
        Seen {
            method: method.into(),
            path: path.into(),
            authorization: format!("Bot {TOKEN}"),
            user_agent: AGENT.into(),
            body,
        }
    }

    /// **Each call is the request Discord documents**: its method, its path
    /// under the base, the bot's authorization, and — for the `PATCH` —
    /// exactly the fields given.
    #[tokio::test]
    async fn each_call_is_the_documented_request_as_the_bot() {
        let discord = fake::start(Mode::Ok).await;
        let client = Client::at(&format!("{}/", discord.url));

        let me = client.me(TOKEN).await.unwrap();
        assert_eq!(
            me,
            BotUser {
                id: fake::USER_ID.into(),
                username: fake::OLD_NAME.into(),
                bot: true
            }
        );
        let app = client.application(TOKEN).await.unwrap();
        assert_eq!(
            app,
            Application {
                id: fake::APPLICATION_ID.into(),
                name: "Zen".into()
            }
        );
        let renamed = client
            .set_profile(TOKEN, Some("Zen"), Some("data:image/png;base64,iVBORw=="))
            .await
            .unwrap();
        assert_eq!(renamed.username, "Zen");
        client.set_profile(TOKEN, None, None).await.unwrap();

        assert_eq!(
            discord.seen(),
            [
                seen("GET", "/users/@me", Value::Null),
                seen("GET", "/applications/@me", Value::Null),
                seen(
                    "PATCH",
                    "/users/@me",
                    json!({ "username": "Zen", "avatar": "data:image/png;base64,iVBORw==" })
                ),
                seen("PATCH", "/users/@me", json!({})),
            ]
        );
    }

    #[tokio::test]
    async fn a_401_is_a_rejected_token() {
        let discord = fake::start(Mode::Unauthorized).await;
        let client = Client::at(&discord.url);
        assert_eq!(client.me(TOKEN).await, Err(DiscordError::TokenRejected));
        assert_eq!(
            client.application(TOKEN).await,
            Err(DiscordError::TokenRejected)
        );
        // Answered once each: no retry.
        assert_eq!(discord.seen().len(), 2);
    }

    #[tokio::test]
    async fn a_429_says_how_long_to_wait() {
        let discord = fake::start(Mode::RateLimited).await;
        let client = Client::at(&discord.url);
        assert_eq!(
            client.me(TOKEN).await,
            Err(DiscordError::RateLimited { retry_after_s: 2.5 })
        );
        assert_eq!(
            DiscordError::RateLimited { retry_after_s: 2.5 }.to_string(),
            "Discord is rate limiting; retry after 2.5 s"
        );
        assert_eq!(discord.seen().len(), 1);
    }

    #[tokio::test]
    async fn a_user_token_is_not_a_bot() {
        let discord = fake::start(Mode::NotABot).await;
        assert_eq!(
            Client::at(&discord.url).me(TOKEN).await,
            Err(DiscordError::NotABot)
        );
    }

    #[tokio::test]
    async fn any_other_failure_carries_its_status_and_what_discord_said() {
        let discord = fake::start(Mode::Failing).await;
        assert_eq!(
            Client::at(&discord.url).me(TOKEN).await,
            Err(DiscordError::Status {
                status: 502,
                detail: "upstream is down".into()
            })
        );
    }

    #[test]
    fn a_refusal_reads_its_wait_from_the_body_then_the_header() {
        let mut headers = HeaderMap::new();
        assert_eq!(
            refusal(
                StatusCode::TOO_MANY_REQUESTS,
                &headers,
                r#"{"retry_after": 0.75}"#
            ),
            DiscordError::RateLimited {
                retry_after_s: 0.75
            }
        );
        headers.insert(RETRY_AFTER, "4".parse().unwrap());
        assert_eq!(
            refusal(StatusCode::TOO_MANY_REQUESTS, &headers, ""),
            DiscordError::RateLimited { retry_after_s: 4.0 }
        );
        assert_eq!(
            refusal(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new(), ""),
            DiscordError::RateLimited { retry_after_s: 1.0 }
        );
        assert_eq!(
            refusal(
                StatusCode::FORBIDDEN,
                &HeaderMap::new(),
                r#"{"message": "Missing Access", "code": 50001}"#
            ),
            DiscordError::Status {
                status: 403,
                detail: "Missing Access".into()
            }
        );
    }
}
