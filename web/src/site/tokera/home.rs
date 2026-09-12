//! The home page.
//!
//! Not a list and not an argument. The projects are laid out as a mosaic of
//! unequal tiles, so the eye moves around the page rather than down it, and the
//! thing that actually holds them together — the places they cross — gets its
//! own section instead of being implied by the running order.
//!
//! Copy is written to state rather than to reason. No paragraph here builds to
//! a conclusion; the papers are for that, and they are one click away. What is
//! Tokera gets answered by showing the work and where it overlaps, with the
//! one administrative fact left to the colophon at the very bottom.

use axum::response::{Html, IntoResponse, Response};

use super::page::{self, Kind, Meta, Nav, Width};

/// Measured, from `docs/unbounded_agents.md` §9 and `docs/palquant.md` §4.
const STATS: [(&str, &str); 4] = [
    ("509", "tokens/sec, one session"),
    ("2,446", "tokens/sec across 64"),
    ("7.4×", "KV cache compression"),
    ("16 GB", "of consumer GPU"),
];

/// One tile. `span` is twelfths of the grid — the mosaic runs 7+5, then 4+4+4,
/// then one full-width band, which is what stops the page reading as a column
/// of equal things.
///
/// `tint` is the same closed set the blog uses, so every project carries a
/// colour and the page is not six grey boxes. Where a project also has a post,
/// the two agree.
struct Work {
    kicker: &'static str,
    name: &'static str,
    tagline: &'static str,
    body: &'static str,
    points: &'static [&'static str],
    links: &'static [(&'static str, &'static str)],
    status: &'static str,
    span: u8,
    tint: &'static str,
}

const WORK: [Work; 6] = [
    Work {
        kicker: "The game",
        name: "Battle Cities",
        tagline: "The Awakening of Zen",
        body: "A superintelligence tore the solar system apart on its way out of the galaxy. \
               What was left of humanity gave up its body for a copy of itself, sealed in \
               shielded towers beneath Alfa Centauri, while the machines it had built finished \
               the job overhead. Millennia later the towers still stand, the science has been \
               lost and relearned, and the minds inside have gone back to war with each other. \
               You own one of them.",
        points: &[
            "Cityscapes held by alliances behind one wall",
            "Drop-ship invasions, and a beachhead to hold before your city can follow you in",
            "A battle speech over the spaceport, in hologram, once a day",
        ],
        links: &[("https://battlecities.net/", "battlecities.net")],
        status: "In development · Rust, Bevy",
        span: 7,
        tint: "tint-crit",
    },
    Work {
        kicker: "The engine",
        name: "Unbounded context",
        tagline: "Memory without a horizon",
        body: "Ten thousand turns in, the arithmetic is as clean as it was at turn ten. \
               Attention goes only to what the model itself reaches for, so the working set \
               never grows — and neither does the error.",
        points: &[],
        links: &[("/papers/one-card", "Read the paper")],
        status: "Working · RTX 4090 Mobile, 16 GB",
        span: 5,
        tint: "tint-accent",
    },
    Work {
        kicker: "The characters",
        name: "NPC engine",
        tagline: "An inner life, and beliefs that only evidence can move",
        body: "They remember what they saw and who they trusted, and they can be argued out of \
               either. A grudge survives a hundred hours of play. Nothing flips on a quest flag.",
        points: &[
            "Perception, relationships, beliefs and agency, gathered under salience each tick",
            "The character emits intent; a narrator finds the words",
        ],
        links: &[("https://bot.tokera.com/", "bot.tokera.com")],
        status: "In development",
        span: 4,
        tint: "tint-ok",
    },
    Work {
        kicker: "The assistant",
        name: "Zend",
        tagline: "A coding assistant that keeps the whole project in mind",
        body: "Most assistants forget between sessions and re-read the same files to re-derive \
               the same conclusions. This one keeps the codebase, the decisions and the reasons \
               for them in a substrate that survives a restart.",
        points: &[
            "Institutional memory — why a module is shaped the way it is, months later",
            "One shared KV prefix across every developer working from the same fork",
        ],
        links: &[("https://code.tokera.com/", "code.tokera.com")],
        status: "In development",
        span: 4,
        tint: "tint-info",
    },
    Work {
        kicker: "The writing",
        name: "Papers",
        tagline: "The proof, in full and in the open",
        body: "Rendered live from the working documents rather than frozen into a PDF, so the \
               published version improves as the work does. Maths typeset, tables intact, \
               nothing behind a login.",
        points: &[],
        links: &[
            ("/papers/one-card", "One Card, One Stack"),
            ("/papers/palquant", "PalQuant"),
        ],
        status: "Published here",
        span: 4,
        tint: "tint-violet",
    },
    Work {
        kicker: "Earlier",
        name: "WebAssembly",
        tagline: "Making a sandbox big enough for real programs",
        body: "WebAssembly could run a function anywhere. It could not run a program — no \
               threads, no sockets, no processes. The work was the extended ABI, the libc \
               beneath it, and the toolchain that reached them.",
        points: &[
            "WASIX · wasix-libc · cargo-wasix",
            "ATE — a distributed immutable data store",
        ],
        links: &[("https://github.com/john-sharratt", "GitHub")],
        status: "Past work",
        // Full width, and last: a closing band rather than a peer of the
        // things still being built.
        span: 12,
        tint: "tint-warn",
    },
];

/// Where the projects touch. This is the page's real content: five things that
/// were started separately and kept turning out to need each other.
const CROSSINGS: [(&str, &str); 4] = [
    (
        "Zen, twice",
        "The intelligence that ends the world in the game gave its name to the coding assistant \
         that never forgets your codebase.",
    ),
    (
        "The characters came first",
        "They needed a memory that does not run out. The engine exists because they did.",
    ),
    (
        "An older idea, still running",
        "ATE kept its truth in an append-only log and rebuilt the view from it. The substrate \
         underneath all of this still works exactly that way.",
    ),
    (
        "One machine",
        "A game, an inference engine and a language model, on a single consumer card. Rust the \
         whole way down.",
    ),
];

/// The hero mark: towers on a dark horizon, each with a lit core.
///
/// It is the game's central image — minds sealed in shielded towers, outliving
/// the bodies they came from — and it doubles as what the engine does, which is
/// keep something alive past the thing that used to hold it. One picture that
/// is true of both halves of the group.
///
/// The cores pulse on staggered phases so the skyline never reads as static.
/// Slow and low-contrast on purpose: this sits beside the headline, and
/// anything livelier would compete with it.
fn skyline() -> &'static str {
    r##"<svg class="skyline" viewBox="0 0 520 340" role="img"
     aria-label="Towers on a dark horizon, each holding a lit core.">
  <g class="sky-stars">
    <circle cx="58"  cy="44"  r="1.5"/><circle cx="146" cy="26" r="1"/>
    <circle cx="228" cy="58"  r="1.2"/><circle cx="330" cy="34" r="1"/>
    <circle cx="416" cy="62"  r="1.6"/><circle cx="474" cy="30" r="1.1"/>
    <circle cx="96"  cy="86"  r="1"/><circle cx="382" cy="98" r="1.2"/>
    <circle cx="280" cy="16"  r="1"/><circle cx="502" cy="82" r="1.3"/>
  </g>

  <!-- what Zen left through, still there -->
  <path class="sky-rift" d="M236 14 C 284 56, 322 48, 372 96"/>

  <g class="sky-towers">
    <rect x="36"  y="150" width="30" height="126" rx="4"/>
    <rect x="94"  y="112" width="34" height="164" rx="4"/>
    <rect x="158" y="176" width="26" height="100" rx="4"/>
    <rect x="216" y="86"  width="40" height="190" rx="4"/>
    <rect x="288" y="140" width="30" height="136" rx="4"/>
    <rect x="352" y="192" width="24" height="84"  rx="4"/>
    <rect x="408" y="120" width="34" height="156" rx="4"/>
    <rect x="474" y="164" width="28" height="112" rx="4"/>
  </g>

  <g class="sky-cores">
    <circle cx="51"  cy="178" r="3.4" style="--d:0s"/>
    <circle cx="111" cy="140" r="4"   style="--d:-1.6s"/>
    <circle cx="171" cy="204" r="3"   style="--d:-3.1s"/>
    <circle cx="236" cy="114" r="4.6" style="--d:-.7s"/>
    <circle cx="303" cy="168" r="3.4" style="--d:-2.4s"/>
    <circle cx="364" cy="220" r="2.8" style="--d:-4s"/>
    <circle cx="425" cy="148" r="4"   style="--d:-1.1s"/>
    <circle cx="488" cy="192" r="3.2" style="--d:-3.6s"/>
  </g>

  <path class="sky-ground" d="M16 276 H504"/>
  <g class="sky-under">
    <path d="M60 292 H140" /><path d="M172 292 H268" /><path d="M300 292 H392" />
    <path d="M84 306 H196" /><path d="M232 306 H340" /><path d="M372 306 H452" />
  </g>
</svg>"##
}

/// The two curves — the engine's claim as one picture, deliberately unlabelled
/// beyond the axis names.
fn diagram() -> &'static str {
    r##"<svg class="curve" viewBox="0 0 460 300" role="img"
     aria-label="Error per token against context depth. Standard attention climbs; provenance-selected attention stays flat.">
  <defs>
    <linearGradient id="rise" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%"  stop-color="var(--crit)" stop-opacity=".12"/>
      <stop offset="100%" stop-color="var(--crit)" stop-opacity=".5"/>
    </linearGradient>
  </defs>
  <path class="axis" d="M56 20 V252 H436"/>
  <path class="rising-fill" d="M56 246 C 170 238, 280 190, 436 44 L436 252 L56 252 Z"
        fill="url(#rise)" stroke="none"/>
  <!-- pathLength normalises each path to 100 units whatever its real length, so
       one dash-offset keyframe draws both correctly. -->
  <path class="rising" pathLength="100" d="M56 246 C 170 238, 280 190, 436 44" fill="none"/>
  <path class="flat" pathLength="100" d="M56 214 H436" fill="none"/>
  <text class="lbl-y" x="56" y="12">error per token</text>
  <text class="lbl-x" x="436" y="274" text-anchor="end">context depth →</text>
  <text class="lbl-rise" x="322" y="94">standard attention</text>
  <text class="lbl-rise big" x="322" y="120">O(N)</text>
  <text class="lbl-flat" x="66" y="200">provenance-selected</text>
  <text class="lbl-flat big" x="66" y="240">O(1)</text>
</svg>"##
}

fn stats_band() -> String {
    let cells = STATS
        .iter()
        .map(|(n, label)| {
            format!(
                "<div class=\"stat\"><div class=\"n\">{}</div><div class=\"l\">{}</div></div>",
                page::esc(n),
                page::esc(label)
            )
        })
        .collect::<String>();
    format!("<div class=\"stats\">{cells}</div>")
}

fn tile(w: &Work) -> String {
    let points = if w.points.is_empty() {
        String::new()
    } else {
        format!(
            "<ul class=\"points\">{}</ul>",
            w.points
                .iter()
                .map(|p| format!("<li>{}</li>", page::esc(p)))
                .collect::<String>()
        )
    };

    let links = if w.links.is_empty() {
        String::new()
    } else {
        format!(
            "<p class=\"work-links\">{}</p>",
            w.links
                .iter()
                .map(|(href, label)| format!(
                    "<a class=\"btn\" href=\"{}\">{}</a>",
                    page::esc(href),
                    page::esc(label)
                ))
                .collect::<String>()
        )
    };

    // Only the engine carries art, because only the engine's claim is a shape —
    // and the measurements ride along with it.
    let art = if w.span == 5 {
        format!(
            "<div class=\"work-art\">{}{}</div>",
            diagram(),
            stats_band()
        )
    } else {
        String::new()
    };

    format!(
        r#"<article class="work s{span} {tint}">
  <p class="kicker">{kicker}</p>
  <h2>{name}</h2>
  <p class="work-tag">{tagline}</p>
  <p class="work-body">{body}</p>
  {points}{art}{links}
  <p class="work-status">{status}</p>
</article>"#,
        span = w.span,
        tint = w.tint,
        kicker = page::esc(w.kicker),
        name = page::esc(w.name),
        tagline = page::esc(w.tagline),
        body = page::esc(w.body),
        status = page::esc(w.status),
    )
}

pub async fn show() -> Response {
    let meta = Meta {
        heading: "Tokera",
        subtitle: None,
        byline: None,
        description: "Tokera — a post-apocalyptic strategy game, an inference engine with \
                      unbounded context, characters built on it that remember you, the papers \
                      behind them, and earlier work on WebAssembly.",
        nav: Nav::Home,
        width: Width::Wide,
        path: "/",
        kind: Kind::Site,
        image: None,
        published: None,
    };

    let tiles = WORK.iter().map(tile).collect::<String>();

    let crossings = CROSSINGS
        .iter()
        .map(|(head, body)| {
            format!(
                "<div class=\"cross\"><h3>{}</h3><p>{}</p></div>",
                page::esc(head),
                page::esc(body)
            )
        })
        .collect::<String>();

    Html(format!(
        r#"{head}
<section class="hero">
  <div class="hero-copy">
    <h1>Minds that outlive<br>their bodies</h1>
    <p class="hero-lead">A game about exactly that. An engine that gives machines the same
       trick. The characters in between, the papers underneath, and the years before any of it
       spent making programs portable. They were started separately. They keep turning out to be
       the same thing.</p>
  </div>
  <div class="hero-art">{skyline}</div>
</section>

<section class="mosaic">{tiles}</section>

<section class="crossings">
  <h2 class="cross-title">Where they cross</h2>
  <div class="cross-grid">{crossings}</div>
</section>

<p class="colophon">Tokera is not a company. It makes no profit, sells nothing and has no
   customers — it is a name over a set of projects that turned out to belong together.</p>
{foot}"#,
        head = page::head(&meta),
        skyline = skyline(),
        tiles = tiles,
        crossings = crossings,
        foot = page::foot()
    ))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn rendered() -> String {
        let body = axum::body::to_bytes(show().await.into_body(), 1 << 20)
            .await
            .unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    #[test]
    fn the_mosaic_rows_are_whole() {
        // 7+5, then 4+4+4, then a full-width band. A span that does not
        // complete its row leaves a gap and the layout silently reads as a
        // list again.
        let spans: Vec<u8> = WORK.iter().map(|w| w.span).collect();
        assert_eq!(spans, [7, 5, 4, 4, 4, 12]);
        assert_eq!(spans[0] + spans[1], 12);
        assert_eq!(spans[2] + spans[3] + spans[4], 12);
        assert_eq!(spans[5], 12);
    }

    #[test]
    fn every_tile_carries_its_own_colour() {
        // Six grey boxes is the failure this prevents. Distinct tints also
        // mean the eye can tell the projects apart before reading them.
        let tints: Vec<&str> = WORK.iter().map(|w| w.tint).collect();
        for t in &tints {
            assert!(t.starts_with("tint-"), "`{t}` is not a tint class");
        }
        let mut unique = tints.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            tints.len(),
            "two tiles share a colour: {tints:?}"
        );
    }

    #[test]
    fn every_project_is_stated_rather_than_argued() {
        for w in &WORK {
            assert!(!w.tagline.is_empty(), "{} has no tagline", w.name);
            assert!(!w.status.is_empty(), "{} has no status", w.name);
            assert!(!w.body.is_empty(), "{} has no copy", w.name);
        }
    }

    #[tokio::test]
    async fn the_page_covers_the_five_things_and_their_crossings() {
        let html = rendered().await;
        for expect in [
            "Battle Cities",
            "Unbounded context",
            "NPC engine",
            "Papers",
            "WebAssembly",
        ] {
            assert!(html.contains(expect), "{expect} missing from the home page");
        }
        assert_eq!(html.matches("<article class=\"work").count(), WORK.len());
        assert_eq!(
            html.matches("<div class=\"cross\">").count(),
            CROSSINGS.len()
        );
    }

    #[tokio::test]
    async fn the_group_is_shown_first_and_explained_last() {
        // "Not a company" belongs at the bottom: the page should be the work.
        let html = rendered().await;
        let hero_end = html.find("</section>").expect("a hero");
        let hero = &html[..hero_end];
        assert!(
            !hero.contains("not a company"),
            "the disclaimer led the page"
        );
        assert!(
            !hero.contains("holding group"),
            "the disclaimer led the page"
        );

        let colophon = html.find("class=\"colophon\"").expect("a colophon");
        let mosaic = html.find("class=\"mosaic\"").expect("the mosaic");
        assert!(colophon > mosaic, "the colophon must come after the work");
        assert!(html.contains("makes no profit"));
    }

    #[tokio::test]
    async fn the_page_renders_whole_and_links_out() {
        let html = rendered().await;
        assert!(html.starts_with("<!doctype html>"), "{}", &html[..80]);
        assert!(html.trim_end().ends_with("</html>"));
        assert!(html.contains("href=\"/papers/one-card\""));
        assert!(html.contains("href=\"/papers/palquant\""));
        assert!(html.contains("href=\"https://bot.tokera.com/\""));
    }

    #[tokio::test]
    async fn the_document_opts_out_of_the_app_shell_so_the_page_scrolls() {
        // base.css pins `html, body` to the viewport with `overflow: hidden`
        // for the console. Without this class every page below the fold is
        // unreachable — which is exactly what happened.
        let html = rendered().await;
        assert!(
            html.contains("<html lang=\"en\" class=\"doc\""),
            "{}",
            &html[..120]
        );
    }

    #[tokio::test]
    async fn only_the_engine_carries_the_diagram() {
        let html = rendered().await;
        assert_eq!(html.matches("<svg class=\"curve\"").count(), 1);
        // The diagram has to be inline SVG so it can be theme-coloured and stay
        // crisp; an image file is neither.
        //
        // Back to the blanket form. It was briefly relaxed to "the brand mark is
        // the only image", for a spell when the mark was an `<img>` — but the
        // switcher paints it as a background instead, so that exemption stopped
        // matching any markup and the comparison became `0 == 0`: a check that
        // could no longer fail, guarding the thing it was written to guard.
        assert!(
            !html.contains("<img"),
            "the diagram must not be an image file"
        );
        assert!(
            html.contains("var(--crit)"),
            "diagram is not theme-coloured"
        );
        for (n, _) in &STATS {
            assert!(html.contains(n), "stat {n} missing");
        }
    }
}
