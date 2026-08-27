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
    /// The deployment's backoff ceiling, when the caller knows it.
    ///
    /// The page's own retry loop climbs to this rather than to a number baked
    /// into the script: a config with `max_ms: 400` had the browser waiting ten
    /// seconds between probes while the gateway was ready to dial every 0.4,
    /// and the doc comment claiming the two matched was true only for the
    /// default. `None` falls back to the same default the config uses.
    pub retry_cap: Option<Duration>,
}

impl Problem {
    /// An upstream did not produce a response.
    ///
    /// The status follows whether it is worth coming back, not which call site
    /// produced it. A daemon that is merely down is `503` with a `Retry-After`
    /// — the same answer as [`backing_off`](Self::backing_off), because to a
    /// reader it is the same condition seen a moment later. A panic, a bad
    /// upstream URI or a missing local API is `502`: nothing about waiting will
    /// help, and saying "retrying" would be a lie.
    ///
    /// They used to differ. Every second refresh of the reconnecting page
    /// landed on the health probe, which answered `502 Bad Gateway` under a
    /// different heading — so a service being off looked like two alternating
    /// faults, and the 502 read as the site itself being broken.
    pub fn upstream_down(detail: impl Into<String>, retry_after: Option<Duration>) -> Self {
        Self {
            status: if retry_after.is_some() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            },
            code: "upstream_unavailable",
            title: if retry_after.is_some() {
                "Reconnecting"
            } else {
                "That service is not answering"
            }
            .into(),
            detail: detail.into(),
            retry_after,
            retry_cap: None,
        }
    }

    pub fn backing_off(detail: impl Into<String>, retry_after: Duration) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "upstream_backoff",
            title: "Reconnecting".into(),
            detail: detail.into(),
            retry_after: Some(retry_after),
            retry_cap: None,
        }
    }

    /// Tell the page how long the deployment is willing to wait between
    /// attempts, so its retry loop climbs to the same ceiling this gateway
    /// uses rather than to a compiled-in guess.
    pub fn with_cap(mut self, cap: Duration) -> Self {
        self.retry_cap = Some(cap);
        self
    }

    pub fn not_found(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found",
            title: "Not found".into(),
            detail: detail.into(),
            retry_after: None,
            retry_cap: None,
        }
    }

    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            title: "Bad request".into(),
            detail: detail.into(),
            retry_after: None,
            retry_cap: None,
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

/// The header marking a response as the gateway's own failure rather than an
/// upstream's answer.
///
/// The retry loop needs to know whether the service came back, and a status
/// code cannot tell it: a daemon that has recovered may perfectly well answer
/// `401` because the session expired during the outage, or `404` because the
/// probe URL does not exist there. Treating anything but `2xx` as "still down"
/// leaves the page reconnecting forever against a service a manual refresh
/// would reach. What actually distinguishes the two is *who* answered, so the
/// gateway says so.
pub const GATEWAY_ERROR: &str = "x-tokera-gateway-error";

pub fn respond(p: Problem, html: bool) -> Response {
    let mut res = if html { html_page(&p) } else { json_body(&p) };
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(p.code) {
        h.insert(header::HeaderName::from_static(GATEWAY_ERROR), v);
    }
    if let Some(ra) = p.retry_after {
        let secs = ra.as_secs().max(1);
        if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
            h.insert(header::RETRY_AFTER, v);
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

/// A self-contained page — no stylesheet, no script file, no external anything,
/// so it renders correctly even when the thing that was meant to serve the CSS
/// is the thing that is down.
///
/// When the problem is retryable the page keeps trying by itself, and reloads
/// the moment the service answers. That is the whole design goal: a daemon
/// restart should look like a page that came back on its own, not like
/// something the reader had to nurse.
fn html_page(p: &Problem) -> Response {
    let retryable = p.retry_after.is_some();
    // Without scripting there is nothing to run the backoff, so fall back to
    // the flat meta refresh — inside `<noscript>` so the two never both fire.
    let fallback = p
        .retry_after
        .map(|d| {
            format!(
                r#"<noscript><meta http-equiv="refresh" content="{}"></noscript>"#,
                d.as_secs().max(1)
            )
        })
        .unwrap_or_default();
    // The whole retry affordance, or none of it. The progress track used to sit
    // outside this, so a 404 rendered a bar with nothing to drive it.
    //
    // Only the text inside `#msg` is rewritten as the countdown runs — the dot
    // is a sibling, because replacing it restarts its keyframe from frame zero
    // every 250ms and the "live" pulse never actually pulses.
    let live = if retryable {
        r#"<p class="r"><span class="dot"></span><span id="msg">Reconnecting&hellip;</span></p>
<div class="bar"><i id="bar"></i></div>"#
            .to_string()
    } else {
        String::new()
    };

    // The server's own numbers, so the page and the gateway agree. `first` is
    // what the health gate is currently waiting; `cap` is the deployment's
    // ceiling.
    let first_ms = p
        .retry_after
        .map(|d| d.as_millis().max(250) as u64)
        .unwrap_or(1000);
    let cap_ms = p
        .retry_cap
        .map(|d| d.as_millis().max(first_ms as u128) as u64)
        .unwrap_or(10_000);
    let script = if retryable {
        format!("{RETRY_SCRIPT_HEAD}var FIRST={first_ms},CAP={cap_ms},MARK='{GATEWAY_ERROR}';{RETRY_SCRIPT_BODY}")
    } else {
        String::new()
    };

    let body = format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>{fallback}
<style>
:root{{color-scheme:dark light}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;
  background:radial-gradient(1200px 600px at 50% -10%,#2a2522 0%,#1c1917 60%);color:#ede9e5;
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.card{{max-width:580px;width:100%;background:#221f1d;border:1px solid #332c27;
  border-left:3px solid #c98a3e;border-radius:14px;padding:30px 32px;
  box-shadow:0 24px 60px rgba(0,0,0,.42)}}
h1{{margin:0 0 10px;font-size:1.4rem;letter-spacing:-.3px}}
p{{margin:0 0 12px;color:#948c84}}
.r{{color:#c98a3e;display:flex;align-items:center;gap:9px;margin-top:16px}}
.dot{{width:8px;height:8px;border-radius:50%;background:#c98a3e;flex:none;
  animation:pulse 1.4s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.35;transform:scale(.72)}}}}
.bar{{height:3px;border-radius:2px;background:#332c27;overflow:hidden;margin-top:4px}}
.bar i{{display:block;height:100%;width:0;background:#c98a3e;border-radius:2px}}
code{{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:.82rem;color:#cdc6be;
  background:#151312;border-radius:6px;padding:2px 6px;overflow-wrap:anywhere}}
.s{{margin-top:20px;font-size:.72rem;color:#5b5450;font-family:ui-monospace,monospace}}
@media (prefers-reduced-motion:reduce){{.dot{{animation:none}}.bar i{{transition:none}}}}
</style></head><body><div class="card">
<h1>{title}</h1>
<p>{detail}</p>
{live}
<div class="s">{code} &middot; HTTP {status}</div>
</div>{script}</body></html>"#,
        title = esc(&p.title),
        detail = esc(&p.detail),
        code = p.code,
        status = p.status.as_u16(),
        fallback = fallback,
        live = live,
        script = script,
    );
    (
        p.status,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

/// Client-side retry: exponential backoff, doubling from the gateway's current
/// window up to the deployment's ceiling.
///
/// Both numbers come from the server rather than from this file, because the
/// two disagreeing is how a browser ends up polling a gateway faster than it is
/// willing to probe — every one of those requests answered out of the backoff
/// window, learning nothing.
///
/// The cap is what keeps it a live page. Without one an unattended tab drifts
/// to retrying once an hour; with a ceiling it stays responsive to a restart
/// while costing at most one request per tab per interval.
///
/// Emitted **after** the elements it touches. It used to sit above them, so
/// `getElementById('bar')` captured `null` for the lifetime of the page and
/// every line that drew the progress bar was dead.
///
/// Split in two so the numbers can be interpolated between the halves without
/// doubling every brace in the body — JavaScript escaped for a format string is
/// JavaScript nobody wants to edit.
const RETRY_SCRIPT_HEAD: &str = "<script>\n(function(){";
const RETRY_SCRIPT_BODY: &str = r#"
  var n=0, msg=document.getElementById('msg'), bar=document.getElementById('bar');
  // Inline styles beat a stylesheet rule, so the reduced-motion opt-out has to
  // be honoured here too — the CSS `transition:none` cannot override this.
  var still = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function say(t){ if(msg) msg.innerHTML=t; }
  function sweep(ms){
    if(!bar) return;
    if(still){ bar.style.width='100%'; return; }
    bar.style.transition='none'; bar.style.width='0';
    // Force a reflow, or the browser coalesces the reset and the animation into
    // one frame and the bar jumps instead of travelling.
    void bar.offsetWidth;
    bar.style.transition='width '+ms+'ms linear'; bar.style.width='100%';
  }
  function probe(){
    n++;
    say('Checking&hellip; &middot; attempt '+n);
    if(bar){ bar.style.transition='none'; bar.style.width='100%'; }
    // A unique query and `no-store`, or a cached copy of this very page answers
    // and we reload straight back into it.
    fetch(location.pathname+(location.search?location.search+'&':'?')+'_probe='+Date.now(),
          {cache:'no-store',redirect:'follow'})
      .then(function(r){
        // Whether the service is back is a question of WHO answered, not of the
        // status: a recovered daemon may legitimately reply 401 because the
        // session expired during the outage, or 404 because this path does not
        // exist there. Only the gateway sets this header, and only on its own
        // failures — so its absence means something behind the gateway replied.
        if(!r.headers.get(MARK)){ say('Back &mdash; reloading&hellip;'); location.reload(); }
        else next();
      })
      .catch(next);
  }
  function next(){
    var ms=Math.min(CAP, FIRST*Math.pow(2,n)), end=Date.now()+ms;
    sweep(ms);
    (function tick(){
      var left=Math.max(0, end-Date.now());
      say('Retrying in <b>'+Math.ceil(left/1000)+'s</b> &middot; attempt '+(n+1));
      if(left>0) setTimeout(tick, left>1000?250:left);
    })();
    setTimeout(probe, ms);
  }
  next();
})();
</script>"#;

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
