//! The page shell every tokera.com page is poured into.
//!
//! Server-rendered rather than an app: this is a site people read, and a paper
//! that needs JavaScript before it shows a word is worse in every way that
//! matters here — first paint, deep links, printing, search indexing, and
//! reading it on something old. The only script on the page is the few lines
//! that swap "Sign in" for your name, and the page is complete without it.

pub struct Meta<'a> {
    /// Browser title, before the site suffix.
    pub title: &'a str,
    /// The `<h1>`. Usually the same as `title`, but a paper's h1 is its full
    /// name while the tab wants something shorter.
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

/// The other sites in the estate — separate hosts, so they are rendered as a
/// distinct group rather than mixed in with the pages above. A nav that makes
/// "somewhere else on this site" look identical to "a different site" is a nav
/// that lies about where a click goes.
pub const ELSEWHERE: [(&str, &str); 3] = [
    ("https://code.tokera.com/", "Zen Code"),
    ("https://bot.tokera.com/", "NPCs"),
    ("https://battlecities.net/", "Battle Cities"),
];

/// Everything from the doctype to the end of the nav bar.
fn doc_open(m: &Meta) -> String {
    format!(
        r#"<!doctype html>
<html lang="en" class="{root}" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>{title} · Tokera</title>
<meta name="description" content="{description}">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="stylesheet" href="/base.css">
<link rel="stylesheet" href="/site.css">
</head>
<body class="tokera">
{nav}
"#,
        root = m.width.root_class(),
        title = esc(m.title),
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

    let elsewhere = ELSEWHERE
        .iter()
        .map(|(href, label)| format!("<a href=\"{href}\">{label}</a>"))
        .collect::<Vec<_>>()
        .join("");

    // The same mark as favicon.svg — a T that is also a figure with its arms
    // out — inline so the counter-form takes the page's own background instead
    // of a baked colour that goes wrong the moment someone switches to light.
    //
    // The tile and the head are brand colours rather than theme tokens: a mark
    // that changes because somebody retuned `--info` is not a mark.
    let mark = r##"<svg class="mark" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
    <defs><clipPath id="tkmark"><rect width="32" height="32" rx="7.5"/></clipPath></defs>
    <g clip-path="url(#tkmark)">
      <rect width="32" height="32" fill="var(--brand)"/>
      <circle cx="16" cy="6.6" r="3.5" fill="var(--brand-head)"/>
      <rect x="-2" y="12" width="36" height="5.4" fill="var(--bg)"/>
      <rect x="13.3" y="12" width="5.4" height="26" fill="var(--bg)"/>
    </g>
  </svg>"##;

    // `hidden` until the script decides which of the two to show, so a reader
    // with JavaScript off sees neither a wrong state nor a flash of both.
    format!(
        r#"<header class="site-top">
  <a class="site-brand" href="/">{mark}Tokera</a>
  <nav class="site-nav">{links}</nav>
  <nav class="site-away" aria-label="Other sites">{elsewhere}</nav>
  <div class="site-auth" id="site-auth" hidden></div>
</header>
<script type="module" src="/lib/auth.js"></script>
"#
    )
}

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
    let away = ELSEWHERE
        .iter()
        .map(|(href, label)| format!("<a href=\"{href}\">{label}</a>"))
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

    fn meta() -> Meta<'static> {
        Meta {
            title: "T",
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
        assert_eq!(bar.matches("aria-current").count(), 1);
    }

    #[test]
    fn the_other_sites_are_a_separate_group_from_this_site_s_pages() {
        let bar = nav_bar(Nav::Home);
        assert!(bar.contains("class=\"site-nav\""));
        assert!(bar.contains("class=\"site-away\""));
        for (href, label) in ELSEWHERE {
            assert!(bar.contains(href), "{label} missing from the bar");
        }
        // Every one of them is an absolute URL to a different host — a
        // relative link here would silently resolve against this site.
        for (href, _) in ELSEWHERE {
            assert!(href.starts_with("https://"), "{href} is not absolute");
        }
    }

    #[test]
    fn the_footer_carries_every_link_the_narrow_bar_drops() {
        // Under 860px the other-sites group is hidden from the bar, so the
        // footer is the only route to those three. Both are generated from the
        // same constants; this asserts the fallback actually holds.
        let f = footer();
        for (href, label) in ELSEWHERE {
            assert!(f.contains(href), "{label} missing from the footer");
        }
        for (href, label, _) in LINKS {
            assert!(f.contains(href), "{label} missing from the footer");
        }
    }

    #[test]
    fn the_head_carries_a_title_and_description() {
        let h = head(&meta());
        assert!(h.contains("<title>T · Tokera</title>"), "{h}");
        assert!(h.contains("name=\"description\" content=\"D\""), "{h}");
    }

    #[test]
    fn text_from_a_document_cannot_close_a_tag() {
        let m = Meta {
            title: "a\"b",
            heading: "<script>alert(1)</script>",
            ..meta()
        };
        let html = format!("{}{}", head(&m), title_block(&m));
        assert!(!html.contains("<script>alert"), "{html}");
        assert!(html.contains("&lt;script&gt;"), "{html}");
        assert!(html.contains("content=\"D\""));
        assert!(html.contains("a&quot;b"), "{html}");
    }
}
