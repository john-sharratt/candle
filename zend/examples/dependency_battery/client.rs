//! The battery's calls to a running zend, over its HTTP API as any client of
//! it makes them: a question as one chat turn on a conversation of its own,
//! that turn's reasoning length and projection placements read back, and the
//! conversation deleted once read.
//!
//! The battery speaks as the service identity `service:battery`
//! (`host/roles.yaml`), stated on the forwarded-identity headers a member
//! believes from its own address, and names its conversations
//! `mcp:<its key>:<thread>/<name>` so they open to it alone (`ownership`).

use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use reqwest::Client;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::arms::Arm;
use crate::placement::Point;

pub const PROVIDER: &str = "service";
pub const SUBJECT: &str = "service:battery";

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

/// The chat request asking `prompt` on `conv_id` under `arm`'s dials, tools
/// off, thinking at [`EFFORT`].
pub fn request(conv_id: &str, prompt: &str, arm: Arm) -> Value {
    let mut body = json!({
        "model": "zen-code",
        "stream": false,
        "conv_id": conv_id,
        "messages": [{"role": "user", "content": prompt}],
        "effort": EFFORT,
        "think": true,
        "tools": "none",
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
    pub secs: f64,
}

pub struct Zend {
    base: String,
    http: Client,
}

impl Zend {
    pub fn new(base: &str) -> anyhow::Result<Self> {
        Ok(Self {
            base: base.trim_end_matches('/').to_string(),
            http: Client::builder().timeout(ANSWER_TIMEOUT).build()?,
        })
    }

    fn with_identity(&self, r: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        r.header("x-tokera-user", SUBJECT)
            .header("x-tokera-provider", PROVIDER)
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
        let history = self.get_json(&self.path(conv_id, "")).await?;
        let think = thinking_of(&history);
        let placements = self.get_json(&self.path(conv_id, "/placements")).await?;
        Ok(Answered {
            text,
            completion_tokens,
            think,
            points: last_turn_points(&placements)?,
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

    #[test]
    fn a_conversation_id_is_encoded_as_one_segment() {
        assert_eq!(urlencode("mcp:ab:t/n-1"), "mcp%3Aab%3At%2Fn-1");
    }
}
