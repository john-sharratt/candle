//! One turn sent through the gateway and read as it streams: every event's
//! arrival time is taken as it is parsed, so the measurement is what a client
//! on the gateway's side sees, gateway hop included.

use std::time::Duration;

use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;
use tokio::time::Instant;

use crate::sample::Sample;
use crate::sse::{Item, Parser};

/// Where requests go.
pub struct Target {
    pub http: Client,
    /// The gateway's chat endpoint.
    pub chat_url: String,
    /// The site the gateway routes by.
    pub host: String,
}

/// The header naming the member that served a request.
const SERVED_BY: &str = "x-served-by";

/// Send `body` and read its stream to the end, or for at most `limit`.
/// Returns the sample and the reply's content.
pub async fn turn(
    target: &Target,
    scenario: &str,
    body: &Value,
    limit: Duration,
) -> (Sample, String) {
    let sent = Instant::now();
    let res = target
        .http
        .post(&target.chat_url)
        .header("host", &target.host)
        .json(body)
        .send()
        .await;
    let res = match res {
        Ok(r) => r,
        Err(e) => {
            return (
                Sample::refused(scenario, "-", e.to_string(), sent.elapsed()),
                String::new(),
            )
        }
    };
    let member = res
        .headers()
        .get(SERVED_BY)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_whitespace().next())
        .unwrap_or("-")
        .to_string();
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        let why = format!("{status}: {}", text.chars().take(300).collect::<String>());
        return (
            Sample::refused(scenario, &member, why, sent.elapsed()),
            String::new(),
        );
    }
    let mut parser = Parser::default();
    let mut events: Vec<(Duration, Item)> = Vec::new();
    let mut content = String::new();
    let mut stream = res.bytes_stream();
    let deadline = sent + limit;
    let end = loop {
        let bytes = match tokio::time::timeout_at(deadline, stream.next()).await {
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(e))) => break format!("transport: {e}"),
            Ok(None) => break "closed".to_string(),
            Err(_) => break "deadline".to_string(),
        };
        let at = sent.elapsed();
        for item in parser.push(&bytes) {
            if let Item::Text { content: c, .. } = &item {
                content.push_str(c);
            }
            let done = item == Item::Done;
            events.push((at, item));
            if done {
                return (
                    Sample::of(scenario, &member, &events, sent.elapsed(), "done"),
                    content,
                );
            }
        }
    };
    (
        Sample::of(scenario, &member, &events, sent.elapsed(), &end),
        content,
    )
}
