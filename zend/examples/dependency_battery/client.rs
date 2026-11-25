//! The battery's calls to a running zend, over its HTTP API as any client of
//! it makes them: a question as one chat turn on a conversation of its own,
//! that turn's reasoning length, tool calls and projection placements read
//! back, and the conversation deleted once read.
//!
//! The battery speaks as the service identity `service:battery`
//! (`host/roles.yaml`), stated on the forwarded-identity headers a member
//! believes from its own address, and names its conversations
//! `mcp:<its key>:<thread>/<name>` so they open to it alone (`ownership`).
//!
//! **Every answer is one member's.** A joined member forwards each request up
//! to a gateway, which may place it on any member; the battery measures one
//! card, so each request names that member's node on `x-tokera-node`. The
//! member keeps the header on a request from a peer it trusts — its own
//! address among them (`host::cluster::listener::scrub`) — and the gateway
//! honours it on a request a member sent up (`web::cluster::route::addressed`).
//! Run the battery on the member it asks: from anywhere else the header is
//! scrubbed and the request is placed as any client's is.

use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use reqwest::Client;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::time::sleep;

use cluster::gateway::NODE_HEADER;

use crate::arms::Arm;
use crate::placement::{answer_token, Point};
use crate::view::governing;

pub const PROVIDER: &str = "service";
pub const SUBJECT: &str = "service:battery";

/// What every request states: who asks, and the member — by its node, 16 hex
/// digits — that answers it.
pub fn headers(node: &str) -> [(&'static str, String); 3] {
    [
        ("x-tokera-user", SUBJECT.to_string()),
        ("x-tokera-provider", PROVIDER.to_string()),
        (NODE_HEADER, node.to_string()),
    ]
}

/// `node` as a member's node id: 16 hex digits, lowercased.
pub fn node_id(node: &str) -> anyhow::Result<String> {
    let n = node.trim().to_ascii_lowercase();
    if n.len() != 16 || !n.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("`{node}` is not a node id: 16 hex digits, as `cluster_status` lists them");
    }
    Ok(n)
}

/// The thinking effort every question is asked at: the chat tools' default.
pub const EFFORT: u8 = 2;

/// The longest one answer may take.
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(1_200);

/// The key the battery's records are kept under: as `Person::key` derives it,
/// the first 16 hex digits of `sha256(provider \0 subject)`.
pub fn owner_key() -> String {
    let digest = Sha256::digest(format!("{PROVIDER}\0{SUBJECT}").as_bytes());
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The conversation one answer is asked on.
pub fn conv_id(thread: &str, arm: Arm, run: u32, qid: &str) -> String {
    format!("mcp:{}:{thread}/{}-r{run}-{qid}", owner_key(), arm.id())
}

/// The chat request asking `prompt` on `conv_id` under `arm`'s dials and
/// tools mode, thinking at [`EFFORT`].
pub fn request(conv_id: &str, prompt: &str, arm: Arm) -> Value {
    let mut body = json!({
        "model": "zen-code",
        "stream": false,
        "conv_id": conv_id,
        "messages": [{"role": "user", "content": prompt}],
        "effort": EFFORT,
        "think": true,
        "tools": arm.tools(),
    });
    if let (Some(b), Value::Object(dials)) = (body.as_object_mut(), arm.dials()) {
        b.extend(dials);
    }
    body
}

/// One answer as the daemon gave it.
pub struct Answered {
    /// The whole decode, its think block included.
    pub text: String,
    pub completion_tokens: u64,
    /// The reasoning's length in tokens, when the turn reasoned.
    pub think: Option<u32>,
    /// The turn's projection points.
    pub points: Vec<Point>,
    /// The point governing the answer's first token in full, its injected
    /// turns' bodies beside it; read for an arm that projects.
    pub first_answer: Option<Value>,
    /// The tool calls the daemon ran during the turn.
    pub tool_calls: u32,
    pub secs: f64,
}

pub struct Zend {
    base: String,
    node: String,
    http: Client,
}

impl Zend {
    /// The daemon at `base`, every request answered by member `node`.
    pub fn new(base: &str, node: &str) -> anyhow::Result<Self> {
        Ok(Self {
            base: base.trim_end_matches('/').to_string(),
            node: node_id(node)?,
            http: Client::builder().timeout(ANSWER_TIMEOUT).build()?,
        })
    }

    fn with_identity(&self, r: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        headers(&self.node)
            .into_iter()
            .fold(r, |r, (k, v)| r.header(k, v))
    }

    fn path(&self, conv_id: &str, rest: &str) -> String {
        format!(
            "{}/v1/conversations/{}{rest}",
            self.base,
            urlencode(conv_id)
        )
    }

    async fn get_json(&self, url: &str) -> anyhow::Result<Value> {
        let r = self.with_identity(self.http.get(url)).send().await?;
        let status = r.status();
        if !status.is_success() {
            bail!("GET {url}: {status}");
        }
        Ok(r.json().await?)
    }

    /// Ask `prompt` on `conv_id` under `arm`, and read back what the turn
    /// reasoned and placed.
    pub async fn ask(&self, conv_id: &str, prompt: &str, arm: Arm) -> anyhow::Result<Answered> {
        let started = Instant::now();
        let r = self
            .with_identity(self.http.post(format!("{}/v1/chat/completions", self.base)))
            .json(&request(conv_id, prompt, arm))
            .send()
            .await
            .context("the chat request")?;
        let status = r.status();
        let body: Value = r.json().await.context("the chat answer")?;
        if !status.is_success() {
            bail!("the turn failed ({status}): {body}");
        }
        let secs = started.elapsed().as_secs_f64();
        let text = body
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let completion_tokens = body
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if answered_nothing(&text, completion_tokens) {
            bail!("the turn answered nothing: no text and no tokens decoded");
        }
        let history = self.get_json(&self.path(conv_id, "")).await?;
        let think = thinking_of(&history);
        let placements = self.get_json(&self.path(conv_id, "/placements")).await?;
        let points = last_turn_points(&placements)?;
        let starts: Vec<u32> = points.iter().map(|p| p.start_token).collect();
        let first_answer = match (arm.projects(), governing(&starts, answer_token(think))) {
            (true, Some(event)) => {
                let turn = placements
                    .as_array()
                    .map_or(0, |t| t.len().saturating_sub(1));
                Some(
                    self.get_json(&self.path(conv_id, &format!("/projections/{turn}/{event}")))
                        .await
                        .context("the point governing the answer's first token")?,
                )
            }
            _ => None,
        };
        Ok(Answered {
            text,
            completion_tokens,
            think,
            points,
            first_answer,
            tool_calls: tool_calls_of(&history),
            secs,
        })
    }

    /// Delete `conv_id`, as the GUI's delete does.
    pub async fn delete(&self, conv_id: &str) -> anyhow::Result<()> {
        let url = self.path(conv_id, "");
        let r = self.with_identity(self.http.delete(&url)).send().await?;
        if !r.status().is_success() {
            bail!("DELETE {url}: {}", r.status());
        }
        Ok(())
    }

    /// `GET path` on the daemon, for the telemetry sampler.
    pub async fn sample(&self, path: &str) -> anyhow::Result<Value> {
        self.get_json(&format!("{}{path}", self.base)).await
    }

    /// Whether the member reports itself ready to serve.
    async fn ready(&self) -> bool {
        self.sample("/v1/status")
            .await
            .is_ok_and(|s| status_is_ready(&s))
    }

    /// After a failed answer: whether the member was down, waiting until it
    /// serves again and its memory has settled, for at most [`READY_WAIT`]. A
    /// failure while the member restarts is the deploy's, not the answer's, so
    /// it costs the answer no try; a member that never comes back within the
    /// wait answers `false`, and the try counts.
    pub async fn was_down(&self) -> bool {
        if self.ready().await {
            return false;
        }
        let since = Instant::now();
        while since.elapsed() < READY_WAIT {
            sleep(READY_POLL).await;
            if self.ready().await {
                // A restart re-learns its score backgrounds: what it answers
                // before they settle is no measurement of the memory.
                return self.until_settled().await.is_ok();
            }
        }
        false
    }

    /// Wait until the member's memory is settled, saying each minute what it
    /// waits on. An error when its zend does not report settling.
    pub async fn until_settled(&self) -> anyhow::Result<()> {
        loop {
            let status = self.sample("/v1/status").await?;
            match settled_of(&status) {
                Some((true, _)) => return Ok(()),
                Some((false, waiting_on)) => {
                    println!(
                        "waiting for the memory to settle: {}",
                        waiting_on.join("; ")
                    );
                    sleep(SETTLED_POLL).await;
                }
                None => bail!("the member's zend does not report whether its memory is settled"),
            }
        }
    }
}

/// How often a down member is asked whether it serves again.
pub const READY_POLL: Duration = Duration::from_secs(15);

/// The longest the battery waits for a member to come back.
pub const READY_WAIT: Duration = Duration::from_secs(3_600);

/// Whether a `/v1/status` answer says the member serves.
pub fn status_is_ready(status: &Value) -> bool {
    status.get("state").and_then(Value::as_str) == Some("ready")
}

/// What a `/v1/status` answer says of zend's memory: whether it is settled —
/// read, calibrated and normalized — and what it waits on (`zend::settled`).
/// `None` from a member whose zend does not report it.
pub fn settled_of(status: &Value) -> Option<(bool, Vec<String>)> {
    let s = status.pointer("/tenants/zend/settled")?;
    let settled = s.get("settled")?.as_bool()?;
    let waiting_on = s
        .get("waiting_on")
        .and_then(Value::as_array)
        .map(|w| {
            w.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Some((settled, waiting_on))
}

/// How often an unsettled member is asked again.
pub const SETTLED_POLL: Duration = Duration::from_secs(60);

/// Whether a reply is no answer at all: nothing said and nothing decoded — a
/// turn the daemon declined without saying why, which is a failure to retry,
/// never a score of zero.
pub fn answered_nothing(text: &str, completion_tokens: u64) -> bool {
    text.trim().is_empty() && completion_tokens == 0
}

/// The reasoning length of a history's last assistant bubble that carries one.
pub fn thinking_of(history: &Value) -> Option<u32> {
    history
        .get("messages")?
        .as_array()?
        .iter()
        .rev()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .find_map(|m| m.pointer("/thinking/tokens")?.as_u64())
        .map(|t| t as u32)
}

/// The tool calls a history's assistant bubbles carry: each bubble's parts
/// (`zend::parts::Part`), of which a call is `{"kind": "call", ..}`. A
/// battery conversation holds one question, so every call is its turn's.
pub fn tool_calls_of(history: &Value) -> u32 {
    history
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|m| m.get("parts")?.as_array())
        .flatten()
        .filter(|p| p.get("kind").and_then(Value::as_str) == Some("call"))
        .count() as u32
}

/// The last assistant turn's points, from the placements route's answer.
pub fn last_turn_points(placements: &Value) -> anyhow::Result<Vec<Point>> {
    let turns: Vec<Vec<Point>> =
        serde_json::from_value(placements.clone()).context("the placements")?;
    Ok(turns.into_iter().last().unwrap_or_default())
}

/// `s` percent-encoded for one path segment.
pub fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_battery_owns_its_conversations_by_its_key() {
        let key = owner_key();
        assert_eq!(key.len(), 16);
        assert_eq!(
            conv_id("battery-128", Arm::C1, 2, "t07"),
            format!("mcp:{key}:battery-128/c1-r2-t07")
        );
    }

    /// **A question goes out with tools off and the arm's dials**, and
    /// nothing the request must not carry.
    #[test]
    fn a_request_carries_the_arms_dials() {
        assert_eq!(
            request("c", "Where?", Arm::B0),
            json!({
                "model": "zen-code", "stream": false, "conv_id": "c",
                "messages": [{"role": "user", "content": "Where?"}],
                "effort": 2, "think": true, "tools": "none",
                "seed": false, "projection": "full", "reprojection": "held",
            })
        );
        assert_eq!(
            request("c", "Where?", Arm::T1),
            json!({
                "model": "zen-code", "stream": false, "conv_id": "c",
                "messages": [{"role": "user", "content": "Where?"}],
                "effort": 2, "think": true, "tools": "restricted",
                "seed": false, "projection": "minimal",
            })
        );
    }

    /// **Every call of every assistant bubble is counted**, and nothing else.
    #[test]
    fn the_history_gives_the_tool_calls() {
        let h = json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "parts": [
                {"kind": "think", "id": "t0", "text": "x"},
                {"kind": "call", "id": "c0", "args": [], "ok": true},
                {"kind": "call", "id": "c1", "args": [], "ok": false},
            ]},
            {"role": "user", "content": "<tool_response>r</tool_response>",
             "parts": [{"kind": "call", "id": "u", "args": [], "ok": true}]},
            {"role": "assistant", "content": "b", "parts": [
                {"kind": "call", "id": "c2", "args": [], "ok": true},
                {"kind": "prose", "text": "done"},
            ]},
        ]});
        assert_eq!(tool_calls_of(&h), 3);
        assert_eq!(tool_calls_of(&json!({"messages": []})), 0);
        assert_eq!(tool_calls_of(&json!({})), 0);
    }

    #[test]
    fn an_empty_reply_is_no_answer() {
        assert!(answered_nothing("", 0));
        assert!(answered_nothing("  \n", 0));
        assert!(!answered_nothing("", 12), "decoded tokens are an answer");
        assert!(!answered_nothing("I do not remember.", 5));
    }

    /// Only a member that says `ready` serves; loading, draining or a body
    /// with no state is down.
    #[test]
    fn a_member_serves_only_when_it_says_ready() {
        assert!(status_is_ready(&json!({"state": "ready"})));
        assert!(!status_is_ready(&json!({"state": "loading"})));
        assert!(!status_is_ready(&json!({})));
    }

    /// **The member's zend says whether its memory is settled**, and what it
    /// waits on; a zend that does not say is told apart from one that is not
    /// settled.
    #[test]
    fn the_status_says_whether_the_memory_is_settled() {
        let waiting = json!({"tenants": {"zend": {"settled": {
            "settled": false,
            "waiting_on": ["calibration: 2 owed", "warming: 3 of 16 files"],
        }}}});
        assert_eq!(
            settled_of(&waiting),
            Some((
                false,
                vec![
                    "calibration: 2 owed".to_string(),
                    "warming: 3 of 16 files".to_string()
                ]
            ))
        );
        let done = json!({"tenants": {"zend": {"settled": {"settled": true, "waiting_on": []}}}});
        assert_eq!(settled_of(&done), Some((true, vec![])));
        assert_eq!(settled_of(&json!({"tenants": {"zend": {}}})), None);
    }

    #[test]
    fn the_history_gives_the_reasoning_length() {
        let h = json!({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "thinking": {"tokens": 812, "exact": true}},
        ]});
        assert_eq!(thinking_of(&h), Some(812));
        assert_eq!(thinking_of(&json!({"messages": []})), None);
    }

    #[test]
    fn the_last_turns_points_are_read() {
        let p = json!([
            [{"start_token": 0, "placed": []}],
            [{"start_token": 0, "placed": [{"kind": "Reading", "path": "candle/a.rs", "tokens": 9}]},
             {"start_token": 128, "placed": []}],
        ]);
        let points = last_turn_points(&p).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].placed[0].path.as_deref(), Some("candle/a.rs"));
        assert!(last_turn_points(&json!([])).unwrap().is_empty());
    }

    /// **Every request names the member that answers it**, on the header the
    /// gateway routes by, and a node that is no node id is refused.
    #[test]
    fn every_request_names_its_member() {
        let h = headers("223ebb209828d84c");
        assert_eq!(
            h,
            [
                ("x-tokera-user", "service:battery".to_string()),
                ("x-tokera-provider", "service".to_string()),
                ("x-tokera-node", "223ebb209828d84c".to_string()),
            ]
        );
        assert_eq!(h[2].0, NODE_HEADER);
        assert_eq!(node_id(" 223EBB209828D84C ").unwrap(), "223ebb209828d84c");
        assert!(node_id("BlackWorld").is_err());
        assert!(node_id("223ebb209828d84").is_err());
    }

    #[test]
    fn a_conversation_id_is_encoded_as_one_segment() {
        assert_eq!(urlencode("mcp:ab:t/n-1"), "mcp%3Aab%3At%2Fn-1");
    }
}
