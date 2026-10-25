//! The authorization server's pages — the consent and the refusal — in
//! zend's own look: the warm stone ground and amber accent of
//! `content/common/base.css`, dark by default and warm paper for a light
//! preference.
//!
//! Self-contained on purpose: every style is inline, and nothing is fetched,
//! so the page renders the same on every host the gateway answers and leaks
//! no request while someone is deciding whether to grant access. The colours
//! are base.css's own, and a test holds them to it.
//!
//! Every page goes out through [`respond`], with [`PAGE_POLICY`] and
//! `no-store`, so no page of the authorization server can be framed or kept.

use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};

use crate::html::escape;

/// What a page may do: show its inline style and nothing else — never be
/// framed, so no other page can lay itself over the Allow button and borrow
/// the click.
///
/// No `form-action`: the consent form's answer is a redirect to the client,
/// which is off this origin, and browsers apply `form-action` to the redirects
/// a submission follows — `'self'` would stop the code reaching the client.
pub const PAGE_POLICY: &str =
    "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'";

const STYLE: &str = r#"
:root {
  --bg: #1c1917; --bg-deep: #151312; --panel: #221f1d; --panel-2: #272320;
  --line: #2b2724; --line-2: #38332d;
  --ink: #ede9e5; --ink-mid: #b3aba3; --ink-soft: #948c84; --ink-faint: #7a7169;
  --accent: #c98a3e; --accent-ink: #241a0c; --accent-soft: rgba(201,138,62,.15);
  --accent-line: rgba(201,138,62,.40); --crit: #cf6f5e;
  --shadow: 0 20px 50px rgba(0,0,0,.5);
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #e1d9c9; --bg-deep: #d6ccb9; --panel: #eae4d6; --panel-2: #e3dbcb;
    --line: #d8cfbc; --line-2: #c6baa3;
    --ink: #2a2521; --ink-mid: #554c41; --ink-soft: #6d6357; --ink-faint: #857a6c;
    --accent: #9a6420; --accent-ink: #fdf9f2; --accent-soft: rgba(154,100,32,.12);
    --accent-line: rgba(154,100,32,.36); --crit: #a8462f;
    --shadow: 0 18px 44px rgba(90,72,45,.20);
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; }
body {
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(1100px 600px at 50% -10%, var(--accent-soft), transparent 70%),
    linear-gradient(180deg, var(--bg), var(--bg-deep));
  display: flex; align-items: center; justify-content: center;
  padding: 32px 16px;
}
.card {
  width: 100%; max-width: 440px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 13px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.head {
  display: flex; align-items: center; gap: 12px;
  padding: 18px 22px;
  border-bottom: 1px solid var(--line);
  background: var(--panel-2);
}
.mark {
  width: 34px; height: 34px; border-radius: 9px;
  display: grid; place-items: center;
  background: var(--accent); color: var(--accent-ink);
  font-weight: 700; font-size: 18px; letter-spacing: -.02em;
}
.brand { font-weight: 650; letter-spacing: .01em; }
.brand small { display: block; font-weight: 400; color: var(--ink-faint); font-size: 12px; }
.body { padding: 22px; }
h1 { font-size: 19px; line-height: 1.35; margin: 0 0 6px; font-weight: 650; }
h1 .client { color: var(--accent); }
.lede { margin: 0 0 18px; color: var(--ink-mid); }
.who {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px; margin: 0 0 16px;
  border: 1px solid var(--line); border-radius: 9px;
  color: var(--ink-mid); font-size: 13.5px;
}
.who b { color: var(--ink); font-weight: 600; word-break: break-all; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); flex: none; }
ul.can { list-style: none; margin: 0 0 16px; padding: 0; }
ul.can li {
  position: relative; padding: 7px 0 7px 26px;
  border-top: 1px solid var(--line); color: var(--ink);
}
ul.can li:first-child { border-top: 0; }
ul.can li::before {
  content: ""; position: absolute; left: 4px; top: 14px;
  width: 9px; height: 5px;
  border-left: 2px solid var(--accent); border-bottom: 2px solid var(--accent);
  transform: rotate(-45deg);
}
.to { font-size: 13px; color: var(--ink-soft); margin: 0 0 20px; }
code {
  font: 12.5px/1.4 ui-monospace, "SF Mono", Consolas, monospace;
  background: var(--accent-soft); color: var(--ink);
  border: 1px solid var(--accent-line); border-radius: 6px;
  padding: 1px 6px; word-break: break-all;
}
.actions { display: flex; gap: 10px; }
button {
  flex: 1; font: inherit; font-weight: 600; cursor: pointer;
  padding: 11px 16px; border-radius: 9px; border: 1px solid transparent;
  transition: filter .15s, background .15s;
}
button.allow { background: var(--accent); color: var(--accent-ink); }
button.allow:hover { filter: brightness(1.08); }
button.deny { background: transparent; color: var(--ink-mid); border-color: var(--line-2); }
button.deny:hover { background: var(--panel-2); color: var(--ink); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.foot { padding: 14px 22px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--ink-faint); }
.refused h1 { color: var(--crit); }
"#;

/// A full page titled `title` around `card` — the card's inner HTML, already
/// escaped where it holds what anyone else wrote.
pub fn shell(title: &str, card: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{title}</title>
<style>{STYLE}</style>
</head><body>
<main class="card">
<div class="head"><div class="mark" aria-hidden="true">z</div><div class="brand">zend<small>Tokera</small></div></div>
{card}
</main>
</body></html>
"#,
        title = escape(title),
    )
}

/// The page titled `title` around `card`, answered with `status` under
/// [`PAGE_POLICY`] and never cached.
pub fn respond(status: StatusCode, title: &str, card: &str) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::CONTENT_SECURITY_POLICY, PAGE_POLICY),
        ],
        Html(shell(title, card)),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shell_fetches_nothing_and_escapes_its_title() {
        let html = shell("<t>", "<p>card</p>");
        assert!(html.contains("<title>&lt;t&gt;</title>"));
        assert!(html.contains("<p>card</p>"));
        for fetch in ["<link", "<script", "src=", "url("] {
            assert!(!html.contains(fetch), "{fetch}");
        }
    }

    #[test]
    fn every_page_is_unframeable_uncached_and_free_to_redirect_its_form() {
        let res = respond(StatusCode::OK, "t", "c");
        assert_eq!(res.headers()[header::CONTENT_SECURITY_POLICY], PAGE_POLICY);
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
        assert!(PAGE_POLICY.contains("frame-ancestors 'none'"));
        assert!(!PAGE_POLICY.contains("form-action"));
    }

    /// The colours are base.css's: each token the page sets, dark and light,
    /// is the value the design system gives it, so a palette change there
    /// fails here rather than leaving the sign-in pages in the old colours.
    #[test]
    fn the_palette_is_the_design_systems() {
        let base = include_str!("../../../content/common/base.css");
        let block = |start: &str, end: &str| -> String {
            let from = base.find(start).expect("base.css block") + start.len();
            let to = from + base[from..].find(end).expect("block end");
            base[from..to].to_string()
        };
        let dark = block("[data-theme=\"dark\"] {", "}");
        let light = block("[data-theme=\"light\"] {", "}");
        let (page_dark, page_light) = STYLE.split_once("@media").unwrap();
        let tokens = [
            "--bg",
            "--bg-deep",
            "--panel",
            "--panel-2",
            "--line",
            "--line-2",
            "--ink",
            "--ink-mid",
            "--ink-soft",
            "--ink-faint",
            "--accent",
            "--accent-ink",
            "--accent-soft",
            "--accent-line",
            "--crit",
        ];
        let value = |css: &str, token: &str| -> Option<String> {
            let at = css.find(&format!("{token}:"))?;
            let rest = &css[at + token.len() + 1..];
            Some(rest[..rest.find(';')?].trim().to_string())
        };
        for token in tokens {
            assert_eq!(value(page_dark, token), value(&dark, token), "dark {token}");
            assert_eq!(
                value(page_light, token),
                value(&light, token),
                "light {token}"
            );
        }
    }
}
