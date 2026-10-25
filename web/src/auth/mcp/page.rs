//! The authorization server's pages — the consent and the refusal — in Zen's
//! look: Zen is the rogue intelligence of Battle Cities, Tokera's game, asleep
//! at the heart of the Citadel, and the connector is named for it. Deep space
//! and a starfield, Zen's silver moon, a cold cyan glow; moonlit silver for a
//! light preference. The lore is the frame only: what the page asks — who
//! wants access, as whom, to do what, sent back where — is said plainly, and
//! every page says what Zen really is.
//!
//! Self-contained on purpose: every style is inline and the stars are drawn
//! in CSS; the one thing fetched is Zen's moon, [`MOON`], from the page's own
//! origin, so no request leaves the site while someone is deciding whether to
//! grant access.
//!
//! Every page goes out through [`respond`], with [`PAGE_POLICY`] and
//! `no-store`, so no page of the authorization server can be framed or kept.

use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};

use crate::html::escape;

/// What a page may do: show its inline style and its own origin's images, and
/// nothing else — never be framed, so no other page can lay itself over the
/// Allow button and borrow the click.
///
/// No `form-action`: the consent form's answer is a redirect to the client,
/// which is off this origin, and browsers apply `form-action` to the redirects
/// a submission follows — `'self'` would stop the code reaching the client.
pub const PAGE_POLICY: &str =
    "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'";

/// Zen's moon, from Battle Cities, served from the shared brand folder on
/// every site.
pub const MOON: &str = "/brand/zen.png";

const STYLE: &str = r#"
:root {
  --bg: #05070c; --bg-deep: #020308; --panel: #0b0f17; --panel-2: #10151f;
  --line: #1a2230; --line-2: #283244;
  --ink: #e8eef6; --ink-mid: #a9b6c6; --ink-soft: #8a98aa; --ink-faint: #74818f;
  --accent: #7fd4ff; --accent-ink: #021018; --accent-soft: rgba(127,212,255,.12);
  --accent-line: rgba(127,212,255,.38); --crit: #ff6b78;
  --glow: rgba(127,212,255,.30); --star: rgba(220,235,255,.75);
  --shadow: 0 24px 60px rgba(0,0,0,.65);
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #e9edf2; --bg-deep: #dce2ea; --panel: #f5f7fa; --panel-2: #eef1f5;
    --line: #d7dde5; --line-2: #c3ccd7;
    --ink: #121821; --ink-mid: #3c4757; --ink-soft: #536072; --ink-faint: #667283;
    --accent: #0b6f9c; --accent-ink: #f4fbff; --accent-soft: rgba(11,111,156,.10);
    --accent-line: rgba(11,111,156,.32); --crit: #b3263a;
    --glow: rgba(11,111,156,.16); --star: rgba(60,80,110,.22);
    --shadow: 0 18px 44px rgba(30,45,70,.18);
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; }
body {
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(1px 1px at 12% 18%, var(--star), transparent 60%),
    radial-gradient(1px 1px at 78% 12%, var(--star), transparent 60%),
    radial-gradient(1.5px 1.5px at 64% 72%, var(--star), transparent 60%),
    radial-gradient(1px 1px at 28% 84%, var(--star), transparent 60%),
    radial-gradient(1px 1px at 90% 58%, var(--star), transparent 60%),
    radial-gradient(1.5px 1.5px at 42% 36%, var(--star), transparent 60%),
    radial-gradient(1px 1px at 6% 62%, var(--star), transparent 60%),
    radial-gradient(900px 520px at 50% -8%, var(--glow), transparent 70%),
    linear-gradient(180deg, var(--bg), var(--bg-deep));
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 32px 16px;
}
.card {
  width: 100%; max-width: 440px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: var(--shadow), 0 0 0 1px var(--accent-soft), 0 0 48px var(--accent-soft);
  overflow: hidden;
}
.head {
  display: flex; align-items: center; gap: 14px;
  padding: 18px 22px;
  border-bottom: 1px solid var(--line);
  background: var(--panel-2);
}
.moon {
  width: 44px; height: 44px; border-radius: 50%; flex: none; display: block;
  background: radial-gradient(circle at 34% 30%, #c9a27a 0, #6b4a30 38%, #2a1d14 72%, #0d0907 100%);
  filter: drop-shadow(0 0 10px var(--glow));
}
.rising {
  position: fixed; left: 50%; top: -260px; z-index: 0;
  width: 560px; height: 560px; margin-left: -280px;
  opacity: .16; pointer-events: none; user-select: none;
  filter: drop-shadow(0 0 60px var(--glow));
}
.card, .lore { position: relative; z-index: 1; }
.brand {
  font: 700 15px/1.2 ui-monospace, "SF Mono", Consolas, monospace;
  letter-spacing: .32em;
}
.brand small {
  display: block; margin-top: 3px;
  font: 400 11.5px/1.3 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: .04em; color: var(--ink-faint);
}
.body { padding: 22px; }
.whisper {
  margin: 0 0 14px;
  font: 12px/1.5 ui-monospace, "SF Mono", Consolas, monospace;
  letter-spacing: .06em; color: var(--accent);
}
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
button.allow { background: var(--accent); color: var(--accent-ink); box-shadow: 0 0 20px var(--accent-soft); }
button.allow:hover { filter: brightness(1.08); }
button.deny { background: transparent; color: var(--ink-mid); border-color: var(--line-2); }
button.deny:hover { background: var(--panel-2); color: var(--ink); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.foot { padding: 14px 22px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--ink-faint); }
.refused h1 { color: var(--crit); }
.lore {
  max-width: 440px; margin: 16px 0 0; text-align: center;
  font-size: 12px; color: var(--ink-faint);
}
"#;

/// What every page says Zen really is, under the card.
pub const LORE: &str = "Zen is the rogue intelligence of Battle Cities, Tokera's game. \
This page signs an app in to Zen Code, Tokera's coding assistant.";

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
<img class="rising" src="{MOON}" alt="" aria-hidden="true">
<main class="card">
<div class="head"><img class="moon" src="{MOON}" alt="" aria-hidden="true"><div class="brand">ZEN<small>The Citadel · Tokera</small></div></div>
{card}
</main>
<p class="lore">{LORE}</p>
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

    /// The page fetches only Zen's moon, from its own origin, and says what
    /// Zen really is.
    #[test]
    fn the_shell_fetches_only_its_own_moon_and_escapes_its_title() {
        let html = shell("<t>", "<p>card</p>");
        assert!(html.contains("<title>&lt;t&gt;</title>"));
        assert!(html.contains("<p>card</p>"));
        for fetch in ["<link", "<script", "url(", "//"] {
            assert!(!html.contains(fetch), "{fetch}");
        }
        let sources: Vec<&str> = html
            .split("src=\"")
            .skip(1)
            .map(|s| &s[..s.find('"').unwrap()])
            .collect();
        assert_eq!(sources, [MOON, MOON]);
        assert!(html.contains(LORE));
    }

    #[test]
    fn every_page_is_unframeable_uncached_and_free_to_redirect_its_form() {
        let res = respond(StatusCode::OK, "t", "c");
        assert_eq!(res.headers()[header::CONTENT_SECURITY_POLICY], PAGE_POLICY);
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
        assert!(PAGE_POLICY.contains("frame-ancestors 'none'"));
        assert!(!PAGE_POLICY.contains("form-action"));
    }

    /// Both schemes set every colour the page uses, so neither falls back to
    /// the other's.
    #[test]
    fn the_light_scheme_sets_every_colour_the_dark_one_does() {
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
            "--glow",
            "--star",
            "--shadow",
        ];
        let set = |css: &str, token: &str| css.contains(&format!("{token}:"));
        for token in tokens {
            assert!(set(page_dark, token), "dark {token}");
            assert!(set(page_light, token), "light {token}");
        }
    }
}
