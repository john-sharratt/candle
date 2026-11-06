//! The sites whose logs the connector reads (`docs/mcp_file_tools.md` §6.1): a
//! site marked `logs: true` answers `GET /v1/logs` (the members' query —
//! `grep`, `level`, `since`, `limit` — plus `source`) and
//! `GET /v1/logs/sources`, and the gateway reaches them for a member:
//!
//! ```text
//! GET /gateway/logs                         → {"sites": ["yenbot"]}
//! GET /gateway/logs/<site>/v1/logs?…        → the site's answer
//! GET /gateway/logs/<site>/v1/logs/sources  → the site's answer
//! ```
//!
//! Only a member's stream is served: the identity it states — the person whose
//! connector asks — goes on to the site, which decides what that person may
//! read. A request a member's listener forwarded for a client carries no
//! identity (the listener strips what a client sends), so the site sees
//! nobody. Anything off a member's stream is refused.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Extension, Json, Router};
use serde_json::json;

use super::MemberOrigin;
use crate::config::Config;
use crate::proxy::IDENTITY_HEADERS;

/// How long a site has to answer a log search.
const WITHIN: Duration = Duration::from_secs(20);

/// The two paths a log source answers.
const PATHS: [&str; 2] = ["v1/logs", "v1/logs/sources"];

#[derive(Clone)]
struct LogSites {
    /// Each site marked `logs: true`, and the upstream its `/v1/logs` goes to.
    sites: Arc<Vec<(String, String)>>,
    http: reqwest::Client,
}

/// The log sources among `cfg`'s sites: name, and the upstream URL serving
/// its `/v1/logs`.
pub fn sources(cfg: &Config) -> Vec<(String, String)> {
    cfg.sites
        .iter()
        .filter(|s| s.logs)
        .filter_map(|s| {
            let url = s.route_for("/v1/logs")?.upstream.url()?;
            Some((s.name.clone(), url.trim_end_matches('/').to_string()))
        })
        .collect()
}

pub fn router(cfg: &Config) -> Router {
    Router::new()
        .route("/gateway/logs", get(list))
        .route("/gateway/logs/:site/*rest", get(read))
        .with_state(LogSites {
            sites: Arc::new(sources(cfg)),
            http: reqwest::Client::builder()
                .timeout(WITHIN)
                .build()
                .expect("a plain HTTP client builds"),
        })
}

fn refused(status: StatusCode, detail: String) -> Response {
    (status, Json(json!({"error": "refused", "detail": detail}))).into_response()
}

async fn list(State(s): State<LogSites>, origin: Option<Extension<MemberOrigin>>) -> Response {
    if origin.is_none() {
        return refused(
            StatusCode::UNAUTHORIZED,
            "the site logs are a member's to read".into(),
        );
    }
    let names: Vec<&str> = s.sites.iter().map(|(n, _)| n.as_str()).collect();
    Json(json!({"sites": names})).into_response()
}

async fn read(
    State(s): State<LogSites>,
    origin: Option<Extension<MemberOrigin>>,
    Path((site, rest)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
) -> Response {
    if origin.is_none() {
        return refused(
            StatusCode::UNAUTHORIZED,
            "the site logs are a member's to read".into(),
        );
    }
    if !PATHS.contains(&rest.as_str()) {
        return refused(
            StatusCode::NOT_FOUND,
            format!("a log source answers {PATHS:?}, not {rest}"),
        );
    }
    let Some((_, upstream)) = s.sites.iter().find(|(n, _)| *n == site) else {
        return refused(
            StatusCode::NOT_FOUND,
            format!("no site `{site}` is a log source"),
        );
    };
    let url = match query {
        Some(q) => format!("{upstream}/{rest}?{q}"),
        None => format!("{upstream}/{rest}"),
    };
    let mut req = s.http.get(&url);
    for name in IDENTITY_HEADERS {
        if let Some(v) = headers.get(name) {
            req = req.header(name, v);
        }
    }
    match req.send().await {
        Ok(res) => {
            let status =
                StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match res.bytes().await {
                Ok(body) => (status, [("content-type", "application/json")], body).into_response(),
                Err(e) => refused(StatusCode::BAD_GATEWAY, format!("{site}: {e}")),
            }
        }
        Err(e) => refused(
            StatusCode::BAD_GATEWAY,
            format!("{site} did not answer: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::path::Path as FsPath;

    use axum::body::{to_bytes, Body};
    use axum::extract::Query;
    use axum::http::Request;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    /// A site upstream that echoes the path, query and caller it was asked
    /// with.
    async fn witness() -> SocketAddr {
        let app = Router::new().fallback(
            |uri: axum::http::Uri,
             Query(q): Query<std::collections::BTreeMap<String, String>>,
             headers: HeaderMap| async move {
                Json(json!({
                    "path": uri.path(),
                    "query": q,
                    "email": headers.get("x-tokera-email").and_then(|v| v.to_str().ok()),
                }))
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    }

    fn config(upstream: &str) -> Config {
        let yaml = format!(
            "sites:\n\
             \x20 - name: tokera\n    default: true\n    redirect: \"https://tokera.com\"\n\
             \x20 - name: yenbot\n    hosts: [yenbot.com.au]\n    logs: true\n    api:\n\
             \x20     - {{prefix: /, upstream: \"http://{upstream}\"}}\n"
        );
        Config::from_yaml(&yaml, FsPath::new(".")).unwrap()
    }

    async fn ask(router: Router, path: &str, member: bool) -> (StatusCode, Value) {
        let mut req = Request::get(path)
            .header("x-tokera-email", "yen@example.com")
            .body(Body::empty())
            .unwrap();
        if member {
            req.extensions_mut().insert(MemberOrigin {
                app: "zend".into(),
                node: 1,
            });
        }
        let res = router.oneshot(req).await.unwrap();
        let status = res.status();
        let body = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    /// **Only sites marked `logs: true` are sources**, with the upstream
    /// their `/v1/logs` goes to.
    #[test]
    fn the_sources_are_the_sites_marked_logs() {
        assert_eq!(
            sources(&config("127.0.0.1:9")),
            vec![("yenbot".to_string(), "http://127.0.0.1:9".to_string())]
        );
    }

    /// **A member's read reaches the site with the caller's identity**, the
    /// query as asked; anything else, and any other path, is refused.
    #[tokio::test]
    async fn a_members_read_reaches_the_site_as_its_caller() {
        let addr = witness().await;
        let router = router(&config(&addr.to_string()));
        let (status, listed) = ask(router.clone(), "/gateway/logs", true).await;
        assert_eq!(
            (status, listed),
            (StatusCode::OK, json!({"sites": ["yenbot"]}))
        );

        let (status, seen) = ask(
            router.clone(),
            "/gateway/logs/yenbot/v1/logs?grep=error&source=yenbot%2Frecipes",
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            seen,
            json!({"path": "/v1/logs", "query": {"grep": "error", "source": "yenbot/recipes"},
                   "email": "yen@example.com"})
        );
        let (status, _) = ask(router.clone(), "/gateway/logs/yenbot/v1/logs/sources", true).await;
        assert_eq!(status, StatusCode::OK);

        assert_eq!(
            ask(router.clone(), "/gateway/logs/yenbot/v1/logs", false)
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ask(router.clone(), "/gateway/logs/yenbot/v1/data", true)
                .await
                .0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ask(router, "/gateway/logs/tokera/v1/logs", true).await.0,
            StatusCode::NOT_FOUND
        );
    }
}
