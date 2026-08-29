//! The page shell every tokera.com page is poured into.
//!
//! Server-rendered rather than an app: this is a site people read, and a paper
//! that needs JavaScript before it shows a word is worse in every way that
//! matters here — first paint, deep links, printing, search indexing, and
//! reading it on something old. The only script on the page is the few lines
//! that swap "Sign in" for your name, and the page is complete without it.

pub struct Meta<'a> {
    /// The `<h1>`.
    ///
    /// There is deliberately no separate browser title beside it. Every page
    /// here is titled `Tokera` and stays that way while you read — a tab that
    /// renames itself as you click is restless, and the page already says what
    /// it is in letters an inch tall.
    pub heading: &'a str,
    pub subtitle: Option<&'a str>,
    /// Small text under the heading — a date, authors.
    pub byline: Option<&'a str>,
    pub description: &'a str,
    /// Which nav item is the current page.
    pub nav: Nav,
    /// Extra stylesheet beyond the shared base.
    pub width: Width,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Nav {
    Home,
    Blog,
    Papers,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Width {
    /// Prose column — posts and other single documents.
    Reading,
    /// Full bleed — the home page lays out its own sections.
    Wide,
    /// Two panes: an index on the left, the document on the right, each
    /// scrolling on its own. See [`split`].
    Split,
}

impl Width {
    fn class(self) -> &'static str {
        match self {
            Width::Reading => "reading",
            Width::Wide => "wide",
            Width::Split => "split",
        }
    }

    /// A split page pins itself to the viewport so the right pane can scroll
    /// independently; everything else scrolls as an ordinary document. The
    /// class goes on `<html>` because both it and `<body>` have to be released
    /// or held together — releasing one alone leaves the other clipping it.
    fn root_class(self) -> &'static str {
        match self {
            Width::Split => "doc pinned",
            _ => "doc",
        }
    }
}

/// Pages on this site.
pub const LINKS: [(&str, &str, Nav); 3] = [
    ("/", "Home", Nav::Home),
    ("/blog", "Blog", Nav::Blog),
    ("/papers", "Papers", Nav::Papers),
];

/// Every site in the estate, in the order they are offered.
///
/// `(id, name, url, icon)`. The id is what marks the current one; the icon is
/// served by whichever site is showing the menu, from `content/common/brand/`.
///
/// It used to be each site's own favicon by absolute URL, which failed twice
/// over: zend's icon came from `code.tokera.com`, so it vanished exactly when
/// zend was down and you were most likely looking for a way elsewhere; and the
/// colour painted behind it as a fallback sat *through* Tokera's transparent
/// mark, leaving a red triskelion on a red square.
///
/// **This list exists twice** — here, and in `content/common/lib/estate.js` for
/// every page that renders the menu in JavaScript. zend has a whole copy of
/// `content/common` under `zend/web/common/`, because it embeds its assets with
/// `include_dir!` and cannot reach this crate's content directory; that copy is
/// pinned wholesale by [`tests::zend_carries_the_shared_framework_unchanged`].
/// Two copies of one list is exactly the thing that drifts, so
/// [`tests::the_estate_list_matches_the_shared_module`] compares this against
/// the JavaScript rather than trusting them to stay level.
pub const ESTATE: [(&str, &str, &str, &str); 4] = [
    (
        "tokera",
        "Tokera",
        "https://tokera.com/",
        "/brand/tokera.png",
    ),
    (
        "zend",
        "Zend",
        "https://code.tokera.com/",
        "/brand/zend.svg",
    ),
    ("npcd", "NPCs", "https://bot.tokera.com/", "/brand/npcd.svg"),
    (
        "battlecities",
        "Battle Cities",
        "https://battlecities.net/",
        "/brand/battlecities.png",
    ),
];

/// Which entry of [`ESTATE`] this site is.
const ME: &str = "tokera";

/// Everything from the doctype to the end of the nav bar.
fn doc_open(m: &Meta) -> String {
    format!(
        r#"<!doctype html>
<html lang="en" class="{root}" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>Tokera</title>
<meta name="description" content="{description}">
<link rel="icon" type="image/png" href="/favicon.png">
<link rel="stylesheet" href="/base.css">
<link rel="stylesheet" href="/lib/estate.css">
<link rel="stylesheet" href="/site.css">
</head>
<body class="tokera">
{nav}
"#,
        root = m.width.root_class(),
        description = esc(m.description),
        nav = nav_bar(m.nav),
    )
}

pub fn head(m: &Meta) -> String {
    format!("{}<main class=\"{}\">\n", doc_open(m), m.width.class())
}

pub fn nav_bar(current: Nav) -> String {
    let links = LINKS
        .iter()
        .map(|(href, label, which)| {
            let aria = if *which == current {
                " aria-current=\"page\""
            } else {
                ""
            };
            format!("<a href=\"{href}\"{aria}>{label}</a>")
        })
        .collect::<Vec<_>>()
        .join("");

    // The other sites, as a row of icons-with-labels. Rendered here as well as
    // inside the switcher because the two answer different questions: this row
    // is "what else exists", visible without a click, and the switcher is
    // "where am I and how do I move".
    let elsewhere = ESTATE
        .iter()
        .filter(|(id, ..)| *id != ME)
        .map(|(_, name, url, icon)| {
            format!(
                "<a href=\"{url}\"><span class=\"estate-chip\" \
                 style=\"background-image:url('{icon}')\"></span>{name}</a>"
            )
        })
        .collect::<Vec<_>>()
        .join("");

    // The brand, opened out into every site. `<details>` rather than a scripted
    // popover: open, close, keyboard and Escape are the element's own
    // behaviour, so the control still works with scripts off — which matters on
    // a documents site whose whole point is that it renders without them.
    let rows = ESTATE
        .iter()
        .map(|(id, name, url, icon)| {
            let here = *id == ME;
            // The site you are on links to its own root: the switcher is also
            // the way home, which is what the brand did before it grew a menu.
            let href = if here { "/" } else { url };
            format!(
                "<a href=\"{href}\" class=\"estate-row{cls}\"{aria}>\
                 <span class=\"estate-chip\" style=\"background-image:url('{icon}')\"></span>\
                 <span class=\"estate-name\">{name}</span>{tag}</a>",
                cls = if here { " is-current" } else { "" },
                aria = if here { " aria-current=\"true\"" } else { "" },
                tag = if here {
                    "<span class=\"estate-here\">you are here</span>"
                } else {
                    ""
                },
            )
        })
        .collect::<Vec<_>>()
        .join("");

    // `hidden` until the script decides which of the two to show, so a reader
    // with JavaScript off sees neither a wrong state nor a flash of both.
    format!(
        r#"<header class="site-top">
  <details class="estate">
    <summary class="estate-current" aria-label="Switch site">
      <span class="estate-chip" style="background-image:url('/brand/tokera.png')"></span>
      <span class="estate-name">Tokera</span>
      <span class="estate-caret" aria-hidden="true">&#9662;</span>
    </summary>
    <nav class="estate-menu" aria-label="Sites">{rows}</nav>
  </details>
  <nav class="site-nav">{links}</nav>
  <nav class="site-away" aria-label="Other sites">{elsewhere}</nav>
  <div class="site-auth" id="site-auth" hidden></div>
</header>
<script type="module" src="/lib/auth.js"></script>
{DISMISS}
"#
    )
}

/// The one thing `<details>` will not do for itself: close when you click past
/// it. Without this the menu stays open behind whatever you clicked next, which
/// reads as stuck rather than as a menu.
///
/// Held out of the `format!` above because its braces would have to be doubled
/// there, and JavaScript that has been escaped for a format string is
/// JavaScript nobody wants to edit.
const DISMISS: &str = r#"<script type="module">
  document.addEventListener('click', (e) => {
    for (const d of document.querySelectorAll('details.estate[open]')) {
      if (!d.contains(e.target)) d.open = false;
    }
  });
</script>"#;

pub fn title_block(m: &Meta) -> String {
    let mut s = format!("<header class=\"doc-head\"><h1>{}</h1>", esc(m.heading));
    if let Some(sub) = m.subtitle {
        s.push_str(&format!("<p class=\"doc-sub\">{}</p>", esc(sub)));
    }
    if let Some(by) = m.byline {
        s.push_str(&format!("<p class=\"doc-by\">{}</p>", esc(by)));
    }
    s.push_str("</header>");
    s
}

fn footer() -> String {
    // Built from the same two lists the bar uses. The other-sites group is
    // hidden from the bar on a narrow screen, so the footer is where those
    // links have to survive — generating both from one source is what stops
    // that fallback quietly losing an entry.
    let pages = LINKS
        .iter()
        .map(|(href, label, _)| format!("<a href=\"{href}\">{label}</a>"))
        .collect::<String>();
    let away = ESTATE
        .iter()
        .filter(|(id, ..)| *id != ME)
        .map(|(_, name, url, ..)| format!("<a href=\"{url}\">{name}</a>"))
        .collect::<String>();

    format!(
        "<footer class=\"site-foot\">\
           <div>© Tokera</div>\
           <nav>{pages}<span class=\"foot-sep\" aria-hidden=\"true\"></span>{away}</nav>\
         </footer>"
    )
}

pub fn foot() -> String {
    format!("</main>\n{}\n</body>\n</html>\n", footer())
}

/// A two-pane page: an index that stays put, and a document that scrolls on its
/// own.
///
/// The footer goes **inside** the scrolling pane rather than under both. A
/// footer pinned below a pane that scrolls is a strip of dead space stealing
/// height from the thing you are reading; put it at the end of the document and
/// it arrives when the document does.
pub fn split(m: &Meta, index: &str, view: &str) -> String {
    format!(
        "{open}<main class=\"split\">\n\
         <aside class=\"pane-index\">{index}</aside>\n\
         <section class=\"pane-view\"><div class=\"pane-inner\">{view}\n{footer}</div></section>\n\
         </main>\n</body>\n</html>\n",
        open = doc_open(m),
        index = index,
        view = view,
        footer = footer(),
    )
}

/// Escape for HTML text and for a double-quoted attribute value.
///
/// `crate::markdown::render` carries a private copy of this for its own output,
/// because the markdown layer must not depend on the page template. The two are
/// deliberate duplicates and must escape the same set — `& < > " '`. Anything
/// added here belongs there too.
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Every file under `dir`, depth first. Sorted, so a failure names the same
    /// file on every machine rather than whichever the filesystem yielded first.
    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    }

    fn meta() -> Meta<'static> {
        Meta {
            heading: "H",
            subtitle: None,
            byline: None,
            description: "D",
            nav: Nav::Blog,
            width: Width::Reading,
        }
    }

    #[test]
    fn the_current_page_is_marked_for_screen_readers_and_css() {
        let bar = nav_bar(Nav::Papers);
        assert!(
            bar.contains("href=\"/papers\" aria-current=\"page\""),
            "{bar}"
        );
        // `page` specifically: the switcher marks the current *site* with
        // `aria-current="true"`, which is a different claim about a different
        // thing, and counting both together would make this assertion drift
        // every time either nav changed.
        assert_eq!(bar.matches("aria-current=\"page\"").count(), 1);
    }

    #[test]
    fn the_other_sites_are_a_separate_group_from_this_site_s_pages() {
        let bar = nav_bar(Nav::Home);
        assert!(bar.contains("class=\"site-nav\""));
        assert!(bar.contains("class=\"site-away\""));
        for (id, name, url, ..) in ESTATE {
            if id == ME {
                continue;
            }
            assert!(bar.contains(url), "{name} missing from the bar");
            // Every one is an absolute URL to a different host — a relative
            // link here would silently resolve against this site.
            assert!(url.starts_with("https://"), "{url} is not absolute");
        }
    }

    /// The switcher names every site, marks exactly one as current, and sends
    /// that one home rather than to itself.
    #[test]
    fn the_switcher_lists_the_whole_estate_and_says_which_one_this_is() {
        let bar = nav_bar(Nav::Blog);

        assert!(bar.contains("<details class=\"estate\">"), "{bar}");
        for (_, name, ..) in ESTATE {
            assert!(bar.contains(name), "{name} is not in the switcher");
        }

        assert_eq!(
            bar.matches("estate-row").count(),
            ESTATE.len(),
            "the switcher does not have one row per site"
        );
        assert_eq!(bar.matches("is-current").count(), 1);
        assert_eq!(bar.matches("you are here").count(), 1);

        // The current site's row is the way home. Every other row leaves.
        assert!(
            bar.contains("<a href=\"/\" class=\"estate-row is-current\""),
            "the current site does not link home: {bar}"
        );
        assert!(
            !bar.contains("https://tokera.com/\" class=\"estate-row"),
            "the current site links to itself by absolute URL"
        );
    }

    /// The estate list exists three times and must say the same thing in all
    /// three.
    ///
    /// Rust renders it here, `content/common/lib/estate.js` renders it for the
    /// npcd console, and `zend/web/lib/estate.js` is a copy because zend embeds
    /// its own assets. Nothing forces them to agree, and a switcher that offers
    /// a different set of sites depending on which site you opened it from is
    /// worse than no switcher — so the copies are compared rather than trusted.
    #[test]
    fn the_estate_list_matches_the_shared_module() {
        let shared = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("content")
            .join("common")
            .join("lib")
            .join("estate.js");
        let js = std::fs::read_to_string(&shared)
            .unwrap_or_else(|e| panic!("reading {}: {e}", shared.display()));

        for (id, name, url, icon) in ESTATE {
            for needle in [id, name, url, icon] {
                assert!(
                    js.contains(needle),
                    "`{needle}` is in the Rust list but not in {}",
                    shared.display()
                );
            }
        }

        // And nothing extra on the other side: count the entries rather than
        // only checking that ours are present, or a site added to the module
        // alone would never show up here.
        assert_eq!(
            js.matches("url: 'https://").count(),
            ESTATE.len(),
            "the shared module lists a different number of sites"
        );
    }

    /// Every switcher icon is the mark that site actually serves.
    ///
    /// They are copies, because the switcher is rendered by all four sites and
    /// a site's own favicon is unreachable exactly when that site is down —
    /// zend's row lost its mark the moment zend stopped, which is the one time
    /// somebody is looking for the way somewhere else. A copy is a second file
    /// to keep current, so it is compared against the original rather than
    /// trusted to have been updated alongside it.
    #[test]
    fn the_switcher_icons_are_the_marks_their_sites_serve() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let content = root.join("content");
        for (copy, original) in [
            ("tokera.png", content.join("tokera").join("favicon.png")),
            ("npcd.svg", content.join("npcd").join("favicon.svg")),
            (
                "battlecities.png",
                content.join("battlecities").join("favicon-32x32.png"),
            ),
            (
                "zend.svg",
                root.join("..").join("zend").join("web").join("favicon.svg"),
            ),
        ] {
            let mine = content.join("common").join("brand").join(copy);
            let a =
                std::fs::read(&mine).unwrap_or_else(|e| panic!("reading {}: {e}", mine.display()));
            let b = std::fs::read(&original)
                .unwrap_or_else(|e| panic!("reading {}: {e}", original.display()));
            assert_eq!(a, b, "brand/{copy} has drifted from {}", original.display());
        }

        // And every icon the list names actually exists to be served.
        for (_, name, _, icon) in ESTATE {
            let file = content.join("common").join(
                icon.strip_prefix('/')
                    .expect("switcher icons are root-relative"),
            );
            assert!(file.is_file(), "{name} has no icon at {}", file.display());
        }
    }

    /// zend carries the whole shared framework, byte for byte.
    ///
    /// It is a copy because zend embeds `zend/web` with `include_dir!` and
    /// cannot reach this crate's content directory — a deployment fact rather
    /// than a choice. What is left is to make divergence impossible to land, so
    /// this compares the *directory* rather than a hand-written list of files:
    /// a list is one more thing to remember, and the failure it misses is a new
    /// module that zend silently does not have.
    ///
    /// `/brand` sits at zend's web root rather than under `common/`, because
    /// the switcher addresses icons as `/brand/<id>` on every host and that
    /// path has to resolve the same way here.
    /// Bytes with `\r\n` collapsed to `\n`, so a comparison sees content rather
    /// than the line endings a checkout happened to produce.
    fn nolf(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bytes.len());
        for &b in bytes {
            if b != b'\r' {
                out.push(b);
            }
        }
        out
    }

    #[test]
    fn zend_carries_the_shared_framework_unchanged() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let shared = root.join("content").join("common");
        let zend_web = root.join("..").join("zend").join("web");

        let mut checked = 0;
        for entry in walk(&shared) {
            let rel = entry.strip_prefix(&shared).expect("under the shared root");
            // `brand/` is the one part that lives at zend's root instead.
            let mine = match rel.starts_with("brand") {
                true => zend_web.join(rel),
                false => zend_web.join("common").join(rel),
            };
            let a = std::fs::read(&entry).unwrap();
            let b = std::fs::read(&mine).unwrap_or_else(|e| {
                panic!(
                    "zend is missing {} ({e}) — copy `web/content/common` into `zend/web/common`",
                    rel.display()
                )
            });
            // Compared with line endings normalised, because a `\r\n` here is
            // not drift — it is `core.autocrlf`. Git stores these files with
            // LF and checks them out with CRLF on Windows, so whether two
            // copies agree byte-for-byte in the working tree depends on the
            // developer's git config and on whether each file happens to be
            // tracked yet. That is a property of the checkout, not of the code,
            // and it made this test fail for a reason it does not exist to
            // catch. Any real difference in content still fails.
            assert_eq!(
                nolf(&a),
                nolf(&b),
                "zend's copy of {} has drifted",
                rel.display()
            );
            checked += 1;
        }
        assert!(
            checked > 8,
            "only {checked} files compared — walk is broken"
        );
    }

    /// Every asset zend's pages ask for exists in the directory it embeds.
    ///
    /// zend cannot be compiled on every machine that touches this repo — it
    /// pulls in `candle-kernels`, which needs a CUDA toolchain — so its pages
    /// cannot be opened as part of an ordinary check. A mistyped import is then
    /// invisible: the module 404s, the page renders as a blank shell, and
    /// nothing says why. This is the cheapest thing that catches it.
    #[test]
    fn zends_pages_only_reference_files_it_ships() {
        let web = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("zend")
            .join("web");

        let mut pages = 0;
        for page in walk(&web) {
            if page.extension().is_some_and(|e| e == "html") {
                let html = std::fs::read_to_string(&page).unwrap();
                let name = page.file_name().unwrap().to_string_lossy().into_owned();
                for reference in asset_refs(&html) {
                    // Relative references only. A root-relative path is a route
                    // the daemon serves — `/substrate` is a page, not a file —
                    // and an absolute URL belongs to somebody else entirely.
                    if reference.starts_with("http")
                        || reference.starts_with("//")
                        || reference.starts_with('/')
                    {
                        continue;
                    }
                    let target = web.join(reference.trim_start_matches("./"));
                    assert!(
                        target.is_file(),
                        "{name} references `{reference}`, which is not in zend/web"
                    );
                }
                pages += 1;
            }
        }
        assert!(pages >= 3, "only {pages} pages scanned — the walk is wrong");
    }

    /// The `src`, `href` and `import … from` targets in a page.
    fn asset_refs(html: &str) -> Vec<String> {
        let mut out = Vec::new();
        for (marker, close) in [("src=\"", '"'), ("href=\"", '"'), ("from '", '\'')] {
            let mut rest = html;
            while let Some(i) = rest.find(marker) {
                rest = &rest[i + marker.len()..];
                if let Some(end) = rest.find(close) {
                    let v = &rest[..end];
                    // A bare `/` is the site root, and `#…` is in-page.
                    if !v.is_empty() && v != "/" && !v.starts_with('#') {
                        out.push(v.to_owned());
                    }
                }
            }
        }
        out
    }

    /// Sign-in sends where to come back to as a whole URL, host included.
    ///
    /// The provider's registered redirect URI names one host for the estate, so
    /// the browser always returns there. A relative `next` resolves against that
    /// host rather than the one it was sent from — sign in from the npcd console
    /// and you land on tokera.com's home page, having asked to come back to the
    /// console. It reads as sign-in "moving" you and nothing errors.
    ///
    /// `safe_next` accepts an absolute URL under the cookie domain and refuses
    /// everything else, so this cannot become an open redirect.
    #[test]
    fn sign_in_comes_back_to_the_host_it_left() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for rel in [
            &["content", "npcd", "app.js"][..],
            &["content", "common", "lib", "auth.js"][..],
        ] {
            let path = rel.iter().fold(root.clone(), |p, s| p.join(s));
            let js = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));

            // Stated as "never assemble one without the host" rather than by
            // matching the call site, because either file may build it inline
            // or through a helper and both are fine. `location.pathname` is the
            // only way to get a host-less one, so its absence is the property.
            assert!(
                !js.contains("location.pathname"),
                "{} builds a return address without its host",
                path.display()
            );
            assert!(
                js.contains("location.href"),
                "{} has no whole-URL return address at all",
                path.display()
            );
            assert!(
                js.contains("/auth/login?next="),
                "{} does not send anyone to sign in",
                path.display()
            );
        }
    }

    /// Every shell that shows the switcher must also load its stylesheet.
    ///
    /// The failure this catches is silent rather than loud: the markup renders,
    /// the menu still opens, and it appears as an unstyled list of links piled
    /// on top of the page.
    #[test]
    fn every_shell_that_shows_the_switcher_loads_its_stylesheet() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for (path, href) in [
            (
                root.join("content").join("npcd").join("index.html"),
                "/lib/estate.css",
            ),
            (
                root.join("..").join("zend").join("web").join("index.html"),
                "lib/estate.css",
            ),
        ] {
            let html = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            assert!(
                html.contains(href),
                "{} does not link {href}",
                path.display()
            );
        }
        // And the server-rendered one.
        assert!(head(&meta()).contains(r#"<link rel="stylesheet" href="/lib/estate.css">"#));
    }

    /// It must work with scripts off, because this site's whole claim is that
    /// it renders without them. `<details>` is what buys that; a scripted
    /// popover would not.
    #[test]
    fn the_switcher_needs_no_script_to_open() {
        let bar = nav_bar(Nav::Home);
        let menu = bar.split("<details").nth(1).expect("a switcher");
        let menu = menu.split("</details>").next().unwrap();
        assert!(
            !menu.contains("<script") && !menu.contains("onclick"),
            "the switcher's markup depends on script: {menu}"
        );
    }

    #[test]
    fn the_footer_carries_every_link_the_narrow_bar_drops() {
        // Under 860px the other-sites group is hidden from the bar, so the
        // footer is the only route to those three. Both are generated from the
        // same constants; this asserts the fallback actually holds.
        let f = footer();
        for (id, name, url, ..) in ESTATE {
            if id == ME {
                continue;
            }
            assert!(f.contains(url), "{name} missing from the footer");
        }
        for (href, label, _) in LINKS {
            assert!(f.contains(href), "{label} missing from the footer");
        }
    }

    #[test]
    fn the_head_carries_a_title_and_description() {
        let h = head(&meta());
        assert!(h.contains("<title>Tokera</title>"), "{h}");
        assert!(h.contains("name=\"description\" content=\"D\""), "{h}");
    }

    /// The tab says `Tokera` on every page and keeps saying it.
    ///
    /// A title that renames itself as you click is restless, and the page
    /// already says what it is in letters an inch tall. The description still
    /// varies per page — that is what search results and link previews read,
    /// and it is not the thing sitting in front of you while you read.
    #[test]
    fn the_tab_title_does_not_follow_the_page() {
        let mut seen = std::collections::BTreeSet::new();
        for nav in [Nav::Home, Nav::Blog, Nav::Papers] {
            for width in [Width::Reading, Width::Wide, Width::Split] {
                let m = Meta {
                    heading: "something else entirely",
                    nav,
                    width,
                    ..meta()
                };
                let h = head(&m);
                let t = h
                    .split("<title>")
                    .nth(1)
                    .and_then(|s| s.split("</title>").next())
                    .expect("a title");
                seen.insert(t.to_owned());
            }
        }
        assert_eq!(
            seen,
            ["Tokera".to_owned()].into_iter().collect(),
            "the tab title changes with the page"
        );
    }

    #[test]
    fn text_from_a_document_cannot_close_a_tag() {
        let m = Meta {
            heading: "<script>alert(1)</script>",
            description: "a\"b",
            ..meta()
        };
        let html = format!("{}{}", head(&m), title_block(&m));
        assert!(!html.contains("<script>alert"), "{html}");
        assert!(html.contains("&lt;script&gt;"), "{html}");
        // A quote in a description would otherwise close the `content="` it
        // sits in and let the rest of the string become attributes.
        assert!(html.contains("a&quot;b"), "{html}");
    }
}
