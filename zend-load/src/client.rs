//! One turn sent through the gateway and read as it streams: every event's
//! arrival time is taken as it is parsed, so the measurement is what a client
//! on the gateway's side sees, gateway hop included.

use std::time::Duration;

use futures::StreamExt;
use reqwest::{Client, Method, RequestBuilder};
use serde_json::Value;
use tokio::time::Instant;

use crate::sample::Sample;
use crate::sse::{Item, Parser};

/// Where requests go, and as whom.
pub struct Target {
    pub http: Client,
    /// The gateway's chat endpoint.
    pub chat_url: String,
    /// The site the gateway routes by.
    pub host: String,
    /// The person the run speaks as.
    pub caller: Caller,
}

/// A signed-in person, as a gateway forwards them to the member behind it.
///
/// A member reads who a caller is only from these headers, and clears them on
/// every request whose peer is not a trusted gateway. So a run carries them
/// when it is sent from a gateway's own address straight to a member's
/// listener; through the gateway itself they are cleared and replaced by
/// whoever the gateway has signed in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caller {
    /// The account's subject, as `<provider>:<id>`.
    pub user: String,
    /// The identity provider that signed it in.
    pub provider: String,
    pub email: String,
}

/// The forwarded-identity headers a member reads (`web::auth::forwarded`).
const USER: &str = "x-tokera-user";
const PROVIDER: &str = "x-tokera-provider";
const EMAIL: &str = "x-tokera-email";

impl Target {
    /// A request to `url`, addressed to the site and carrying the caller.
    pub fn request(&self, method: Method, url: &str) -> RequestBuilder {
        self.http
            .request(method, url)
            .header("host", &self.host)
            .header(USER, &self.caller.user)
            .header(PROVIDER, &self.caller.provider)
            .header(EMAIL, &self.caller.email)
    }
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
    // The limit covers the whole turn, the wait for the response's headers
    // included: a request the gateway or a member holds without answering is a
    // cut turn, not a client that never returns.
    let deadline = sent + limit;
    let res = tokio::time::timeout_at(
        deadline,
        target
            .request(Method::POST, &target.chat_url)
            .json(body)
            .send(),
    )
    .await;
    let res = match res {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return (
                Sample::refused(scenario, "-", e.to_string(), sent.elapsed()),
                String::new(),
            )
        }
        Err(_) => {
            return (
                Sample::of(
                    scenario,
                    "-",
                    &[],
                    sent.elapsed(),
                    "deadline before headers",
                ),
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

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use serde_json::json;

    use super::*;
    use crate::sample::Outcome;

    fn caller() -> Caller {
        Caller {
            user: "google:someone@example.com".to_string(),
            provider: "google".to_string(),
            email: "someone@example.com".to_string(),
        }
    }

    /// **Every request carries the site and the caller**, as the gateway's
    /// forwarded headers, byte for byte.
    #[test]
    fn a_request_carries_the_site_and_the_caller() {
        let target = Target {
            http: Client::new(),
            chat_url: "http://127.0.0.1:1/v1/chat/completions".to_string(),
            host: "code.example.com".to_string(),
            caller: caller(),
        };
        let req = target
            .request(Method::DELETE, "http://127.0.0.1:1/v1/conversations/c")
            .build()
            .unwrap();
        let h = req.headers();
        assert_eq!(req.method(), Method::DELETE);
        assert_eq!(h["host"], "code.example.com");
        assert_eq!(h["x-tokera-user"], "google:someone@example.com");
        assert_eq!(h["x-tokera-provider"], "google");
        assert_eq!(h["x-tokera-email"], "someone@example.com");
    }

    /// **A request held without headers is cut at the turn's limit.** The
    /// server accepts and never answers; the turn ends as a cut, not a client
    /// that waits for ever.
    #[tokio::test]
    async fn a_request_held_without_headers_is_cut_at_the_limit() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Accepted and held open, never answered, until the test ends.
        let held = std::thread::spawn(move || listener.accept().map(|(s, _)| s));
        let target = Target {
            http: Client::new(),
            chat_url: format!("http://{addr}/v1/chat/completions"),
            host: "zend.localhost".to_string(),
            caller: caller(),
        };
        let limit = Duration::from_millis(300);
        let (sample, content) = turn(&target, "held", &json!({}), limit).await;
        assert_eq!(
            sample.outcome,
            Outcome::Cut("deadline before headers".to_string())
        );
        assert!(content.is_empty());
        assert!(
            sample.total < Duration::from_secs(5),
            "ended at the limit, not long after: {:?}",
            sample.total
        );
        drop(held.join().unwrap());
    }
}
