//! Error responses, in the shape the caller is asking for.
//!
//! A browser navigating to a page whose backend is down should get a page it
//! can read. A script calling `/v1/status` should get the same `{error, detail,
//! field}` object every other API error uses, so it needs no special case for
//! "the proxy could not reach the daemon". The decision is made from the path
//! and the Accept header, never guessed.

use std::time::Duration;

use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

pub struct Problem {
    pub status: StatusCode,
    pub code: &'static str,
    pub title: String,
    pub detail: String,
    pub retry_after: Option<Duration>,
}

impl Problem {
    pub fn upstream_down(detail: impl Into<String>, retry_after: Option<Duration>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            code: "upstream_unavailable",
            title: "That service is not answering".into(),
            detail: detail.into(),
            retry_after,
        }
    }

    pub fn backing_off(detail: impl Into<String>, retry_after: Duration) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "upstream_backoff",
            title: "Reconnecting".into(),
            detail: detail.into(),
            retry_after: Some(retry_after),
        }
    }

    pub fn not_found(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found",
            title: "Not found".into(),
            detail: detail.into(),
            retry_after: None,
        }
    }

    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            title: "Bad request".into(),
            detail: detail.into(),
            retry_after: None,
        }
    }
}

/// Does this caller want a page, or an error object?
///
/// Two signals, both from the request, neither guessed:
///
///   * **The path.** A path naming a non-HTML file never gets a page. A browser
///     fetching `dom.js` sends `Accept: */*`, and answering that with styled
///     markup is how a missing module becomes `Unexpected token '<'` in the
///     console instead of a 404 the network tab shows plainly.
///   * **`Accept`, explicitly.** Only a caller that names `text/html` gets one.
///     Address-bar navigation always does; `fetch`, `XHR`, and every CLI send
///     `*/*` or `application/json` and get the error object.
///
/// Deliberately *not* a signal: whether the path is under an API prefix. A site
/// that proxies everything — as `zend` does until its console is split out —
/// serves its navigation through the same route as its API, so a rule keyed on
/// the prefix would hand a person raw JSON for the front page. What the caller
/// asked for settles it, at either kind of path.
pub fn wants_html(path: &str, headers: &HeaderMap) -> bool {
    if let Some(seg) = path.rsplit('/').next() {
        if let Some((_, ext)) = seg.rsplit_once('.') {
            if !ext.eq_ignore_ascii_case("html") && !ext.eq_ignore_ascii_case("htm") {
                return false;
            }
        }
    }
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|a| a.contains("text/html"))
        .unwrap_or(false)
}

pub fn respond(p: Problem, html: bool) -> Response {
    let mut res = if html { html_page(&p) } else { json_body(&p) };
    if let Some(ra) = p.retry_after {
        let secs = ra.as_secs().max(1);
        if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
            res.headers_mut().insert(header::RETRY_AFTER, v);
        }
    }
    res
}

fn json_body(p: &Problem) -> Response {
    (
        p.status,
        axum::Json(json!({
            "error": p.code,
            "detail": p.detail,
            "field": null,
            "retry_after_secs": p.retry_after.map(|d| d.as_secs().max(1)),
        })),
    )
        .into_response()
}

/// A self-contained page — no stylesheet, no script, no external anything, so
/// it renders correctly even when the thing that was meant to serve the CSS is
/// the thing that is down. It reloads itself once the backoff window is up,
/// which is what makes recovery look automatic from the browser's side.
fn html_page(p: &Problem) -> Response {
    let retry = p.retry_after.map(|d| d.as_secs().max(1));
    let meta = retry
        .map(|s| format!(r#"<meta http-equiv="refresh" content="{s}">"#))
        .unwrap_or_default();
    let note = match retry {
        Some(s) => format!(
            r#"<p class="r">Retrying automatically in <b>{s}s</b>. \
               You do not need to do anything.</p>"#
        ),
        None => String::new(),
    };
    let body = format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>{meta}
<style>
:root{{color-scheme:dark light}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#1c1917;color:#ede9e5;
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:24px}}
.card{{max-width:560px;background:#221f1d;border-left:3px solid #c98a3e;border-radius:12px;padding:28px 30px}}
h1{{margin:0 0 10px;font-size:1.35rem;letter-spacing:-.3px}}
p{{margin:0 0 12px;color:#948c84}}
.r{{color:#c98a3e}}
code{{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:.82rem;color:#cdc6be;
  background:#151312;border-radius:6px;padding:2px 6px;overflow-wrap:anywhere}}
.s{{margin-top:18px;font-size:.72rem;color:#5b5450;font-family:ui-monospace,monospace}}
</style></head><body><div class="card">
<h1>{title}</h1>
<p>{detail}</p>
{note}
<div class="s">{code} · HTTP {status}</div>
</div></body></html>"#,
        title = esc(&p.title),
        detail = esc(&p.detail),
        code = p.code,
        status = p.status.as_u16(),
        meta = meta,
        note = note,
    );
    (
        p.status,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accept(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::ACCEPT, HeaderValue::from_str(v).unwrap());
        h
    }

    #[test]
    fn a_navigating_browser_gets_html() {
        assert!(wants_html(
            "/npc/1",
            &accept("text/html,application/xhtml+xml,*/*;q=0.8")
        ));
        // Including at an API path — a fully-proxied site navigates through one.
        assert!(wants_html("/v1/status", &accept("text/html")));
    }

    #[test]
    fn a_json_client_gets_json() {
        assert!(!wants_html("/npc/1", &accept("application/json")));
    }

    #[test]
    fn a_wildcard_accept_is_not_a_request_for_a_page() {
        // What fetch(), XHR, and curl send. Only an explicit text/html counts.
        assert!(!wants_html("/npc/1", &accept("*/*")));
    }

    #[test]
    fn no_accept_header_gets_json() {
        assert!(!wants_html("/x", &HeaderMap::new()));
    }

    #[test]
    fn a_missing_asset_never_gets_a_page() {
        // Even from a browser, which sends the navigation Accept for a
        // stylesheet preload.
        assert!(!wants_html("/lib/dom.js", &accept("text/html")));
        assert!(!wants_html("/app.css", &accept("text/html")));
        assert!(wants_html("/index.html", &accept("text/html")));
    }
}
