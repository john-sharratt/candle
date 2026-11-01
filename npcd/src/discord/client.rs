//! The Discord REST calls a bot makes.
//!
//! Each is authorised as the bot (`Authorization: Bot <token>`). A connect
//! asks who the bot is (`GET /users/@me`), which application it belongs to
//! (`GET /applications/@me`), and sets its name and avatar
//! (`PATCH /users/@me`). A DM posts a message (`POST /channels/{id}/messages`),
//! opens a DM channel with a person (`POST /users/@me/channels`), and reads
//! what was written in one since a message (`GET /channels/{id}/messages`).
//! The API's base is a field, Discord's own in production ([`PRODUCTION`]),
//! so a test points it at a fake server and asserts the exact requests.
//!
//! A refusal is answered once, as a [`DiscordError`]; nothing here retries.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::Instant;

use reqwest::header::{HeaderMap, AUTHORIZATION, RETRY_AFTER, USER_AGENT};
use reqwest::{Client as Http, Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::discord::error::DiscordError;
use crate::discord::limit::InvalidRequests;
use crate::discord::message::{snowflake, DiscordMessage};

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

/// Anything Discord answers with only its id read: a posted message, an
/// opened channel.
#[derive(Deserialize)]
struct Made {
    #[serde(deserialize_with = "snowflake")]
    id: u64,
}

/// A server, as `GET /guilds/{id}` answers it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GuildInfo {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    pub name: String,
}

/// One of a server's channels, as `GET /guilds/{id}/channels` answers it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GuildChannel {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    /// `0` a text channel, `4` a category, `5` an announcement channel.
    #[serde(rename = "type")]
    pub kind: u8,
    /// Absent on a channel of a kind that has none.
    #[serde(default)]
    pub name: String,
}

/// One permission overwrite on a channel: whose it is — a role's, the
/// server's own id being `@everyone`, or a member's — and what it allows and
/// denies, as Discord's permission bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overwrite {
    pub id: u64,
    /// `0` a role's, `1` a member's.
    pub kind: u8,
    pub allow: u64,
    pub deny: u64,
}

impl Overwrite {
    /// As Discord takes and gives it: ids and bits as strings.
    fn wire(&self) -> Value {
        json!({
            "id": self.id.to_string(),
            "type": self.kind,
            "allow": self.allow.to_string(),
            "deny": self.deny.to_string(),
        })
    }
}

/// An overwrite as Discord answers it.
#[derive(Deserialize)]
struct OverwriteWire {
    #[serde(deserialize_with = "snowflake")]
    id: u64,
    #[serde(rename = "type")]
    kind: u8,
    allow: String,
    deny: String,
}

/// A channel's overwrites, as `GET /channels/{id}` answers them.
#[derive(Deserialize)]
struct ChannelWire {
    #[serde(default)]
    permission_overwrites: Vec<OverwriteWire>,
}

/// A channel's webhook, as `POST /channels/{id}/webhooks` answers it: what
/// posting through it needs.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Webhook {
    #[serde(deserialize_with = "snowflake")]
    pub id: u64,
    pub token: String,
}

/// The most messages one read of a channel's history answers, as Discord
/// caps it.
pub const HISTORY_PAGE: u32 = 100;

/// Discord's REST API, from one base. Its clones share one count of the
/// host's invalid requests ([`InvalidRequests`]).
#[derive(Clone)]
pub struct Client {
    http: Http,
    base: String,
    invalid: Arc<InvalidRequests>,
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
        Self::limited(base, InvalidRequests::default())
    }

    /// The API at `base`, sending nothing past `invalid`'s ceiling.
    pub fn limited(base: &str, invalid: InvalidRequests) -> Self {
        Self {
            http: Http::new(),
            base: base.trim_end_matches('/').to_string(),
            invalid: Arc::new(invalid),
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

    /// Post `content` to channel `channel_id` as the bot; the message's id.
    ///
    /// Mentions in it ping nobody: what a character says is never an
    /// `@everyone`.
    pub async fn create_message(
        &self,
        token: &str,
        channel_id: u64,
        content: &str,
    ) -> Result<u64, DiscordError> {
        let made: Made = self
            .call(
                Method::POST,
                &format!("/channels/{channel_id}/messages"),
                token,
                Some(json!({ "content": content, "allowed_mentions": { "parse": [] } })),
            )
            .await?;
        Ok(made.id)
    }

    /// The bot's DM channel with Discord user `user_id`, opened if it was not.
    pub async fn open_dm(&self, token: &str, user_id: u64) -> Result<u64, DiscordError> {
        let made: Made = self
            .call(
                Method::POST,
                "/users/@me/channels",
                token,
                Some(json!({ "recipient_id": user_id.to_string() })),
            )
            .await?;
        Ok(made.id)
    }

    /// Server `guild_id`, as the bot sees it: refused when the bot is not in
    /// it.
    pub async fn guild(&self, token: &str, guild_id: u64) -> Result<GuildInfo, DiscordError> {
        self.call(Method::GET, &format!("/guilds/{guild_id}"), token, None)
            .await
    }

    /// Every channel of server `guild_id`.
    pub async fn guild_channels(
        &self,
        token: &str,
        guild_id: u64,
    ) -> Result<Vec<GuildChannel>, DiscordError> {
        self.call(
            Method::GET,
            &format!("/guilds/{guild_id}/channels"),
            token,
            None,
        )
        .await
    }

    /// Make a text channel called `name` in server `guild_id`, with
    /// `overwrites` when there are any; its id.
    pub async fn create_channel(
        &self,
        token: &str,
        guild_id: u64,
        name: &str,
        overwrites: &[Overwrite],
    ) -> Result<u64, DiscordError> {
        let mut body = json!({ "name": name, "type": 0 });
        if !overwrites.is_empty() {
            body["permission_overwrites"] = overwrites.iter().map(Overwrite::wire).collect();
        }
        let made: Made = self
            .call(
                Method::POST,
                &format!("/guilds/{guild_id}/channels"),
                token,
                Some(body),
            )
            .await?;
        Ok(made.id)
    }

    /// Channel `channel_id`'s permission overwrites.
    pub async fn overwrites(
        &self,
        token: &str,
        channel_id: u64,
    ) -> Result<Vec<Overwrite>, DiscordError> {
        let channel: ChannelWire = self
            .call(Method::GET, &format!("/channels/{channel_id}"), token, None)
            .await?;
        Ok(channel
            .permission_overwrites
            .into_iter()
            .map(|o| Overwrite {
                id: o.id,
                kind: o.kind,
                allow: o.allow.parse().unwrap_or(0),
                deny: o.deny.parse().unwrap_or(0),
            })
            .collect())
    }

    /// Set `overwrite` on channel `channel_id`, in place of any it had for
    /// the same id.
    pub async fn put_overwrite(
        &self,
        token: &str,
        channel_id: u64,
        overwrite: &Overwrite,
    ) -> Result<(), DiscordError> {
        let mut body = overwrite.wire();
        if let Some(fields) = body.as_object_mut() {
            fields.remove("id");
        }
        self.call_empty(
            Method::PUT,
            &format!("/channels/{channel_id}/permissions/{}", overwrite.id),
            token,
            Some(body),
        )
        .await
    }

    /// Take channel `channel_id`'s overwrite for `target` away.
    pub async fn delete_overwrite(
        &self,
        token: &str,
        channel_id: u64,
        target: u64,
    ) -> Result<(), DiscordError> {
        self.call_empty(
            Method::DELETE,
            &format!("/channels/{channel_id}/permissions/{target}"),
            token,
            None,
        )
        .await
    }

    /// Make a private thread called `name` under channel `channel_id`,
    /// that only who is added to it sees and nobody on it may invite to; its
    /// id.
    pub async fn create_private_thread(
        &self,
        token: &str,
        channel_id: u64,
        name: &str,
    ) -> Result<u64, DiscordError> {
        let made: Made = self
            .call(
                Method::POST,
                &format!("/channels/{channel_id}/threads"),
                token,
                Some(json!({ "name": name, "type": 12, "invitable": false })),
            )
            .await?;
        Ok(made.id)
    }

    /// Show the bot typing in channel `channel_id`, for the next ten seconds
    /// or until it posts there.
    pub async fn trigger_typing(&self, token: &str, channel_id: u64) -> Result<(), DiscordError> {
        self.call_empty(
            Method::POST,
            &format!("/channels/{channel_id}/typing"),
            token,
            None,
        )
        .await
    }

    /// Add Discord user `user_id` to thread `thread_id`.
    pub async fn add_thread_member(
        &self,
        token: &str,
        thread_id: u64,
        user_id: u64,
    ) -> Result<(), DiscordError> {
        self.call_empty(
            Method::PUT,
            &format!("/channels/{thread_id}/thread-members/{user_id}"),
            token,
            None,
        )
        .await
    }

    /// Make a webhook called `name` on channel `channel_id`, for the
    /// characters without a bot of their own to speak through.
    pub async fn create_webhook(
        &self,
        token: &str,
        channel_id: u64,
        name: &str,
    ) -> Result<Webhook, DiscordError> {
        self.call(
            Method::POST,
            &format!("/channels/{channel_id}/webhooks"),
            token,
            Some(json!({ "name": name })),
        )
        .await
    }

    /// Post `content` through webhook `hook` under the name `username`, with
    /// the picture at `avatar_url` when there is one, into `thread` of the
    /// webhook's channel when one is named; the message's id. Asked to
    /// `wait`, so Discord answers with the message it made. Authorised by the
    /// webhook's own token in the path, not a bot's.
    pub async fn execute_webhook(
        &self,
        hook: &Webhook,
        username: &str,
        avatar_url: Option<&str>,
        thread: Option<u64>,
        content: &str,
    ) -> Result<u64, DiscordError> {
        let mut body = json!({
            "content": content,
            "username": username,
            "allowed_mentions": { "parse": [] },
        });
        if let Some(url) = avatar_url {
            body["avatar_url"] = Value::from(url);
        }
        let into = thread.map_or_else(String::new, |t| format!("&thread_id={t}"));
        let made: Made = self
            .call_unauthorised(
                Method::POST,
                &format!("/webhooks/{}/{}?wait=true{into}", hook.id, hook.token),
                body,
            )
            .await?;
        Ok(made.id)
    }

    /// What was written in channel `channel_id` after message `after`, up to
    /// [`HISTORY_PAGE`] of it, oldest first.
    pub async fn messages_after(
        &self,
        token: &str,
        channel_id: u64,
        after: u64,
    ) -> Result<Vec<DiscordMessage>, DiscordError> {
        let mut messages: Vec<DiscordMessage> = self
            .call(
                Method::GET,
                &format!("/channels/{channel_id}/messages?after={after}&limit={HISTORY_PAGE}"),
                token,
                None,
            )
            .await?;
        // Discord answers newest first.
        messages.sort_by_key(|m| m.id);
        Ok(messages)
    }

    /// One request, as the bot, and its answer read as `T`.
    async fn call<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        token: &str,
        body: Option<Value>,
    ) -> Result<T, DiscordError> {
        self.send(method, path, Some(token), body).await
    }

    /// One request, as the bot, that Discord answers with no body (`204`).
    async fn call_empty(
        &self,
        method: Method,
        path: &str,
        token: &str,
        body: Option<Value>,
    ) -> Result<(), DiscordError> {
        self.send::<Value>(method, path, Some(token), body)
            .await
            .map(|_| ())
    }

    /// One request authorised by what its path carries — a webhook's token —
    /// and its answer read as `T`.
    async fn call_unauthorised<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Value,
    ) -> Result<T, DiscordError> {
        self.send(method, path, None, Some(body)).await
    }

    /// One request, as the bot `token` names when there is one, and its
    /// answer read as `T`.
    async fn send<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        token: Option<&str>,
        body: Option<Value>,
    ) -> Result<T, DiscordError> {
        if let Err(retry_after_s) = self.invalid.check(Instant::now()) {
            return Err(DiscordError::Paused { retry_after_s });
        }
        let mut request = self
            .http
            .request(method, format!("{}{path}", self.base))
            .header(USER_AGENT, AGENT);
        if let Some(token) = token {
            request = request.header(AUTHORIZATION, format!("Bot {token}"));
        }
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
        self.invalid.answered(status.as_u16(), Instant::now());
        let headers = response.headers().clone();
        let text = response
            .text()
            .await
            .map_err(|e| DiscordError::Transport(e.without_url().to_string()))?;
        if !status.is_success() {
            return Err(refusal(status, &headers, &text));
        }
        // A `204` has no body: read as `null`, which a call wanting nothing
        // takes.
        let text = match text.trim().is_empty() {
            true => "null",
            false => text.as_str(),
        };
        serde_json::from_str(text)
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
            code: body["code"].as_u64().unwrap_or(0),
            detail: body["message"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| text.chars().take(200).collect()),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

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

    /// **A DM's calls are the requests Discord documents**: a post with the
    /// text and no mentions, a channel opened by the recipient's id as a
    /// string, and a history read after a message, answered oldest first.
    #[tokio::test]
    async fn the_dm_calls_are_the_documented_requests() {
        let discord = fake::start(Mode::Ok).await;
        let client = Client::at(&discord.url);
        discord.history(
            31,
            vec![
                json!({"id": "903", "channel_id": "31", "author": {"id": "7", "username": "wren"}, "content": "b"}),
                json!({"id": "902", "channel_id": "31", "author": {"id": "7", "username": "wren"}, "content": "a"}),
            ],
        );

        assert_eq!(
            client.create_message(TOKEN, 31, "hello").await,
            Ok(fake::FIRST_POSTED)
        );
        assert_eq!(client.open_dm(TOKEN, 77).await, Ok(fake::DM_CHANNEL));
        let read = client.messages_after(TOKEN, 31, 901).await.unwrap();
        assert_eq!(
            read.iter()
                .map(|m| (m.id, m.content.as_str()))
                .collect::<Vec<_>>(),
            [(902, "a"), (903, "b")]
        );

        assert_eq!(
            discord.seen(),
            [
                seen(
                    "POST",
                    "/channels/31/messages",
                    json!({ "content": "hello", "allowed_mentions": { "parse": [] } })
                ),
                seen(
                    "POST",
                    "/users/@me/channels",
                    json!({ "recipient_id": "77" })
                ),
                seen(
                    "GET",
                    "/channels/31/messages?after=901&limit=100",
                    Value::Null
                ),
            ]
        );
    }

    /// **A world's server's calls are the requests Discord documents**: the
    /// server and its channels read as the bot, a text channel made by name,
    /// a webhook made on it, and a post through the webhook authorised by its
    /// own token in the path — no bot's — under the speaker's name, waited
    /// for, pinging nobody.
    #[tokio::test]
    async fn the_world_server_calls_are_the_documented_requests() {
        let discord = fake::start(Mode::Ok).await;
        let client = Client::at(&discord.url);
        discord.channels(&[(610, 0, "general")]);

        assert_eq!(
            client.guild(TOKEN, 600).await,
            Ok(GuildInfo {
                id: 600,
                name: fake::GUILD_NAME.into()
            })
        );
        assert_eq!(
            client.guild_channels(TOKEN, 600).await,
            Ok(vec![GuildChannel {
                id: 610,
                kind: 0,
                name: "general".into()
            }])
        );
        assert_eq!(
            client.create_channel(TOKEN, 600, "the-channel", &[]).await,
            Ok(fake::FIRST_CHANNEL)
        );
        let hook = client
            .create_webhook(TOKEN, fake::FIRST_CHANNEL, "Life")
            .await
            .unwrap();
        assert_eq!(
            hook,
            Webhook {
                id: fake::FIRST_CHANNEL + fake::HOOK_OFFSET,
                token: fake::HOOK_TOKEN.into()
            }
        );
        assert_eq!(
            client
                .execute_webhook(&hook, "Wick", Some("https://a/p.png"), None, "hello")
                .await,
            Ok(fake::FIRST_POSTED)
        );
        client
            .execute_webhook(&hook, "Wren S", None, None, "hi")
            .await
            .unwrap();

        let hook_path = format!("/webhooks/{}/{}?wait=true", hook.id, fake::HOOK_TOKEN);
        let unauthorised = |body: Value| Seen {
            method: "POST".into(),
            path: hook_path.clone(),
            authorization: String::new(),
            user_agent: AGENT.into(),
            body,
        };
        assert_eq!(
            discord.seen(),
            [
                seen("GET", "/guilds/600", Value::Null),
                seen("GET", "/guilds/600/channels", Value::Null),
                seen(
                    "POST",
                    "/guilds/600/channels",
                    json!({ "name": "the-channel", "type": 0 })
                ),
                seen(
                    "POST",
                    &format!("/channels/{}/webhooks", fake::FIRST_CHANNEL),
                    json!({ "name": "Life" })
                ),
                unauthorised(json!({
                    "content": "hello",
                    "username": "Wick",
                    "avatar_url": "https://a/p.png",
                    "allowed_mentions": { "parse": [] },
                })),
                unauthorised(json!({
                    "content": "hi",
                    "username": "Wren S",
                    "allowed_mentions": { "parse": [] },
                })),
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

    /// **Past the host's ceiling of invalid answers nothing more is sent**,
    /// for any token.
    #[tokio::test]
    async fn past_the_hosts_ceiling_nothing_is_sent() {
        let discord = fake::start(Mode::Unauthorized).await;
        let client = Client::limited(
            &discord.url,
            InvalidRequests::new(2, Duration::from_secs(600)),
        );
        assert_eq!(client.me(TOKEN).await, Err(DiscordError::TokenRejected));
        assert_eq!(
            client.clone().me("another.token").await,
            Err(DiscordError::TokenRejected)
        );
        assert!(matches!(
            client.me(TOKEN).await,
            Err(DiscordError::Paused { retry_after_s }) if retry_after_s > 599.0
        ));
        assert_eq!(discord.seen().len(), 2, "a request went past the ceiling");
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
                code: 0,
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
                code: 50001,
                detail: "Missing Access".into()
            }
        );
        let no_server = refusal(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            r#"{"message": "Cannot send messages to this user", "code": 50278}"#,
        );
        assert!(no_server.no_shared_server());
        assert!(!DiscordError::TokenRejected.no_shared_server());
    }
}
