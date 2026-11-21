//! Running an effector-device call in-process and recording what the world said.
//!
//! The fast path drives the device's router directly rather than over a socket,
//! so an act on a device is the same service call the external mount would
//! answer, with no network hop. A refusal is an answer and not an error: the
//! character reads the world's own sentence, and the feed shows why a refused act
//! was refused. The world lock is never held across the call, because the handler
//! behind the router takes it, and re-entering it would deadlock silently.

use super::*;

impl Runtime {
    /// Retain the `local` effector router so the fast path can drive it. Called
    /// once at startup, after the router is built.
    pub fn set_effector_router(&self, router: Router) {
        *self.effector.write().unwrap() = Some(router);
    }

    /// Act at an effector address in-process — the `invoke` fast path (effector
    /// design §8.1).
    ///
    /// It skips the socket, not the API: it builds a real `http::Request`, stamps
    /// the character's as-npc bearer token ([`Tokens::ensure`], **never** an
    /// `x-tokera-*`, or the operator middleware would read the character as a
    /// human) and drives the retained `local` router through `oneshot`, the
    /// identical service the external mount answers on — a `POST` carrying a JSON
    /// body. It returns the status alongside the value rather than erroring on a
    /// non-2xx, because a refusal (`409`, §12) is a legitimate answer to an act:
    /// the world would not have it, and the caller reads the status and the body
    /// the way an external client would.
    ///
    /// **Never call this while holding the world lock** — the handler behind the
    /// router takes that lock itself, so re-entering from inside a
    /// `hosted.with(...)`/`read(...)` closure deadlocks the non-reentrant world
    /// mutex silently (effector design §8.1). Acquire, act, release, then invoke.
    pub async fn effector_invoke(
        &self,
        npc_id: u64,
        path: &str,
        body: Value,
        pinned: Option<&npc_map::world::Where>,
    ) -> anyhow::Result<(StatusCode, Value)> {
        let router = self
            .effector
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the effector router is not installed"))?;
        let tokens = self
            .tokens()
            .ok_or_else(|| anyhow::anyhow!("the token store is not installed"))?;

        // The fast path always acts as the character's own body, so it takes the
        // as-npc token specifically (`as_npc`, not `ensure`) — a stored `Direct`
        // scope must never become the in-fiction identity (§8.3).
        let token = tokens.as_npc(npc_id)?;
        let mut builder = HttpRequest::builder()
            .method("POST")
            .uri(path)
            .header(AUTHORIZATION, format!("Bearer {token}"));
        // Pin the reach check to where the body stood when this turn's grammar
        // was built, if a standpoint was supplied. A typed extension only the
        // in-process path can insert, so it never reaches — and could never be
        // forged by — an external caller (see [`PinnedStandpoint`]).
        if let Some(at) = pinned {
            builder = builder.extension(PinnedStandpoint(at.clone()));
        }
        let request = builder
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(to_vec(&body)?))?;

        // The router's error type is `Infallible`, so this cannot fail; the
        // `expect` documents that rather than papering over a real error.
        let response = router
            .oneshot(request)
            .await
            .expect("the effector router is infallible");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await?;
        let value: Value = from_slice(&bytes).unwrap_or(Value::Null);
        Ok((status, value))
    }

    /// Enact one effector-device call — `invoke` — on the async fast path and
    /// build the same [`Recorded`] a body act produces through
    /// [`Self::record_act`], so the character reads a device call's outcome back
    /// exactly as it reads any other act's (effector design §5, §6, §12).
    ///
    /// **What comes back.** `invoke` acts and its `answer` is the world's own
    /// line: the receipt the world hands back *this* turn (§10), a `200`
    /// acknowledgement or a `409` refusal in the world's own words (§12). The
    /// lasting effect arrives later through perception, not through this return
    /// value.
    ///
    /// **The url becomes a router path** via [`device_path`]: a leading
    /// `http://local` — the only host the device can reach in-fiction (§7) — is
    /// stripped, so `http://local/lift/command-shaft/call` and the bare
    /// `/lift/command-shaft/call` address the same route.
    ///
    /// **The `invoke` body is a JSON object** (§11), and the grammar writes it as
    /// one, typed by the address it was sent to ([`crate::engine::invoke_body`]).
    /// A caller that is not the grammar may still write it out as a string: an
    /// absent or blank one defaults to `{}`, and a string that will not parse is
    /// answered with a prescriptive `{error:"bad_json",…}` rather than panicking,
    /// so the caller can correct it.
    ///
    /// A cooldown is charged only when the call landed, mirroring `record_act`.
    ///
    /// **Never holds the world lock across the await.** It calls
    /// [`Self::effector_invoke`], whose handler takes the non-reentrant world lock
    /// itself; this method holds nothing, so the lock is never re-entered (§8.1).
    pub(crate) async fn enact_device(
        &self,
        npc_id: u64,
        act: &Act,
        pinned: Option<&npc_map::world::Where>,
    ) -> Recorded {
        let url = act
            .args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim()
            .to_string();
        let path = device_path(&url);
        // A desk's `compose`, reached through the device, is the same sitting
        // down to write as the act itself — at an address this character is
        // offered `compose` at, not any url that happens to end in the word.
        let composes_here = path.rsplit('/').next().is_some_and(body::composes)
            && self
                .within(npc_id)
                .invokable
                .iter()
                .any(|i| device_path(&i.url) == path && body::composes(i.act));
        if composes_here {
            return self.compose(npc_id, act).await;
        }

        match act.tool {
            "invoke" => {
                let body = match act.args.get("body") {
                    None => json!({}),
                    Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                    Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                        Ok(value) => value,
                        Err(e) => {
                            // Prescriptive, not a panic: the field that was wrong,
                            // named, so the next attempt can fix it (§8.4, §12).
                            let err = json!({
                                "error": "bad_json",
                                "detail": e.to_string(),
                                "field": "body",
                            });
                            let answer = to_string(&err).unwrap_or_else(|_| err.to_string());
                            return Recorded {
                                feed: format!("invoke {url} {REFUSED} bad_json"),
                                answer,
                                landed: false,
                                departing: false,
                            };
                        }
                    },
                    // The grammar writes `body` as an object — taken as-is.
                    Some(other) => other.clone(),
                };
                match self.effector_invoke(npc_id, &path, body, pinned).await {
                    Ok((status, value)) => {
                        let landed = status.is_success();
                        // **The character reads the world's own line, not the HTTP
                        // envelope.** The route answers `{ok, detail}` on success
                        // and `{error, detail}` on a refusal (`enact_response`);
                        // either way `detail` is the sentence the world would have
                        // handed a body that acted any other way — "You take it up.
                        // What has been asked of you: …", or "You are already
                        // carrying a mission." So that is what rides back as the
                        // answer. The status is on the feed line for the operator,
                        // not wrapped around the prose the character has to read.
                        let answer = value
                            .get("detail")
                            .and_then(|d| d.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| {
                                to_string(&value).unwrap_or_else(|_| value.to_string())
                            });
                        if landed {
                            self.cooldowns.took_act(npc_id, act);
                        }
                        // **A refusal reads its reason on the feed, like a compiled
                        // act does.** A landed act is self-evident, so its feed line
                        // carries the status (`→ 200 OK`); a refused one carries the
                        // world's own line — the same `detail` the character reads as
                        // its `<tool_response>` — so an operator watching the pulse
                        // sees *why* it was refused, not a bare `✗ 409 Conflict`.
                        // This is the device-act counterpart of `record_act` putting
                        // a refusal's words on the feed (see `Recorded`).
                        let feed = match landed {
                            true => format!("invoke {url} {LANDED} {status}"),
                            false => format!("invoke {url} {REFUSED} {answer}"),
                        };
                        Recorded {
                            feed,
                            answer,
                            landed,
                            departing: false,
                        }
                    }
                    Err(why) => Recorded {
                        feed: format!("invoke {url} {REFUSED} {why}"),
                        answer: format!("The device could not reach {url}: {why}"),
                        landed: false,
                        departing: false,
                    },
                }
            }
            // Only `invoke` is a device act (`body::is_device`), and the loop
            // routes nothing else here — this arm is the honest answer if it ever
            // does, rather than a silent success.
            other => Recorded {
                feed: other.to_string(),
                answer: format!("Nothing on your device answers `{other}`."),
                landed: false,
                departing: false,
            },
        }
    }
}
