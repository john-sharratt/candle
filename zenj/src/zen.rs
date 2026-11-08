//! Asking Zen (§5.2, §9): one chat completion through the gateway, as the
//! service `zenj` (a service key, as yenbot acts as itself), on the night's
//! own conversation — so every choice and its reason stays in Zen's memory of
//! the night. No tools and no thinking: Zen chooses from what it is given,
//! inside a deadline, and a call that misses it is not waited for.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

/// What ZenJ needs to reach Zen.
#[derive(Debug, Clone, Deserialize)]
pub struct ZenCfg {
    /// The gateway, `http://127.0.0.1`.
    pub gateway: String,
    /// The site zend is served at, named in `Host`.
    pub host: String,
    /// The service key's file (`web/secrets/zenj.key`).
    pub key_file: PathBuf,
    /// The conversation the night is kept on.
    pub conversation: String,
    /// How long a choice may take.
    #[serde(default = "default_deadline")]
    pub deadline_s: u64,
}

fn default_deadline() -> u64 {
    20
}

pub struct Zen {
    cfg: ZenCfg,
    key: String,
    http: reqwest::Client,
}

/// The request's body.
pub fn body(conversation: &str, system: &str, user: &str) -> Value {
    json!({
        "model": "zen-code",
        "conv_id": conversation,
        "stream": false,
        "think": false,
        "tools": "none",
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    })
}

/// The reply's text: the first choice's message, with any thinking taken
/// out.
pub fn answer(text: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("{e}: {text}"))?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| format!("no answer in {text}"))?;
    let content = match content.find("</think>") {
        Some(at) => &content[at + "</think>".len()..],
        None => content,
    };
    Ok(content.trim().to_string())
}

impl Zen {
    pub fn new(cfg: ZenCfg, base: &Path) -> Result<Zen, String> {
        let path = base.join(&cfg.key_file);
        let key = fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .trim()
            .to_string();
        Ok(Zen {
            cfg,
            key,
            http: reqwest::Client::new(),
        })
    }

    pub fn deadline(&self) -> Duration {
        Duration::from_secs(self.cfg.deadline_s)
    }

    /// Zen's answer to `user` under `system`, or why there is none in time.
    pub async fn ask(&self, system: &str, user: &str) -> Result<String, String> {
        let url = format!("{}/v1/chat/completions", self.cfg.gateway.trim_end_matches('/'));
        let send = self
            .http
            .post(&url)
            .header("host", &self.cfg.host)
            .bearer_auth(&self.key)
            .json(&body(&self.cfg.conversation, system, user))
            .send();
        let res = tokio::time::timeout(self.deadline(), send)
            .await
            .map_err(|_| format!("Zen did not answer within {} s", self.cfg.deadline_s))?
            .map_err(|e| format!("{url}: {e}"))?;
        let status = res.status();
        let text = res.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("{url}: {status}: {text}"));
        }
        answer(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_request_is_one_turn_on_the_nights_conversation() {
        assert_eq!(
            body("zenj-2026-12-31", "S", "U").to_string(),
            r#"{"model":"zen-code","conv_id":"zenj-2026-12-31","stream":false,"think":false,"tools":"none","max_tokens":200,"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}"#
        );
    }

    #[test]
    fn the_answer_is_the_first_choice_without_its_thinking() {
        let a = r#"{"choices": [{"message": {"role": "assistant", "content": "<think>hm</think>\n {\"pick\": 2} "}}]}"#;
        assert_eq!(answer(a).unwrap(), r#"{"pick": 2}"#);
        assert!(answer(r#"{"error": "x"}"#).unwrap_err().starts_with("no answer"));
    }
}
