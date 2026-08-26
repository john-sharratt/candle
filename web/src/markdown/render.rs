//! Markdown → HTML, through the pulldown-cmark event stream.
//!
//! A document is linear and is rendered linearly. The output is one stream of
//! ordinary HTML — headings, paragraphs, tables, code, maths — and everything
//! about how it *looks* is CSS. There is no contents apparatus, no navigation
//! furniture and nothing injected into the prose, because a paper does not need
//! any of that to be read from the top.
//!
//! Only four transforms happen on the way past, which is the whole reason this
//! is an event walk rather than a call to `push_html`:
//!
//!   * **Headings get ids.** Invisible, and the reason `#9-4-throughput` keeps
//!     working after a re-render — a link someone shared must not break.
//!   * **Maths becomes MathML** ([`super::math`]). The parser hands over
//!     `InlineMath`/`DisplayMath` events, which is what makes this safe: a `$`
//!     inside a code span or a fenced block never reaches the converter,
//!     because it never becomes a maths event in the first place.
//!   * **The leading `# Title` is lifted out** into [`Document::title`], since
//!     the page shell renders it and printing it twice looks like a bug.
//!   * **Links that leave the site open in a new tab**, with the opener guard
//!     and a hidden note saying so. Done here rather than by rewriting the
//!     output because the parser already knows which destinations are absolute,
//!     and re-deriving that from finished HTML is a worse version of the same
//!     question.
//!
//! Tables, footnotes, strikethrough and task lists are on; smart punctuation is
//! deliberately off, because these documents contain code and file paths where
//! turning `--` into an en dash silently corrupts the text.

use pulldown_cmark::{CowStr, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use super::math;
use super::slug::Slugger;

/// A rendered document, plus the title a page needs to frame it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// Text of the leading `# H1`, if the document opened with one. Lifted out
    /// of the body.
    pub title: Option<String>,
    pub html: String,
}

pub fn options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_MATH
        | Options::ENABLE_HEADING_ATTRIBUTES
}

pub fn render(markdown: &str) -> Document {
    let events: Vec<Event> = Parser::new_ext(markdown, options()).collect();
    let mut slugger = Slugger::default();
    let mut out: Vec<Event> = Vec::with_capacity(events.len() + 16);
    let mut title = None;
    // CommonMark forbids nested links, so one flag is the whole state machine —
    // it only has to tell the closing tag which opening tag it belongs to.
    let mut external_link = false;

    let mut i = 0;
    while i < events.len() {
        match &events[i] {
            Event::Start(Tag::Heading {
                level,
                id,
                classes,
                attrs,
            }) => {
                let end = find_heading_end(&events, i);
                let text = plain_text(&events[i + 1..end]);

                // The first h1, and only if nothing has been emitted yet: a
                // later h1 is a real section of a document that happens to use
                // h1s throughout, and stealing it would lose it.
                if *level == HeadingLevel::H1 && out.is_empty() && title.is_none() {
                    title = Some(text);
                    i = end + 1;
                    continue;
                }

                let anchor = match id {
                    Some(existing) => existing.to_string(),
                    None => slugger.slug(&text),
                };
                out.push(Event::Start(Tag::Heading {
                    level: *level,
                    id: Some(CowStr::from(anchor)),
                    classes: classes.clone(),
                    attrs: attrs.clone(),
                }));
                out.extend(events[i + 1..=end].iter().cloned());
                i = end + 1;
            }
            Event::InlineMath(latex) => {
                out.push(Event::InlineHtml(CowStr::from(math::inline(latex))));
                i += 1;
            }
            Event::DisplayMath(latex) => {
                out.push(Event::Html(CowStr::from(math::display(latex))));
                i += 1;
            }
            // A link that leaves the site opens in its own tab. These documents
            // are long and are read in one sitting; following a citation should
            // not cost the reader their place in a 60 KB paper, and the browser
            // back button does not restore a scroll position inside a pane that
            // scrolls independently of the window.
            Event::Start(Tag::Link {
                dest_url, title, ..
            }) if is_external(dest_url) => {
                let t = if title.is_empty() {
                    String::new()
                } else {
                    format!(" title=\"{}\"", attr(title))
                };
                out.push(Event::InlineHtml(CowStr::from(format!(
                    "<a href=\"{}\"{t} target=\"_blank\" rel=\"noopener noreferrer\">",
                    attr(dest_url)
                ))));
                external_link = true;
                i += 1;
            }
            Event::End(TagEnd::Link) if external_link => {
                // A sighted reader sees the tab appear. Somebody on a screen
                // reader gets no signal at all unless the link says so, so the
                // link says so — visually hidden, inside the anchor, where it
                // is read as part of the link text rather than stranded next
                // to it.
                out.push(Event::InlineHtml(CowStr::from(
                    "<span class=\"vh\"> (opens in a new tab)</span></a>",
                )));
                external_link = false;
                i += 1;
            }
            other => {
                out.push(other.clone());
                i += 1;
            }
        }
    }

    let mut html = String::with_capacity(markdown.len() * 3 / 2);
    pulldown_cmark::html::push_html(&mut html, out.into_iter());
    Document { title, html }
}

/// Does this link leave the site? Scheme-relative `//host/path` counts: it is
/// absolute in every way that matters to a reader.
///
/// The scheme test is case-insensitive because URL schemes are — `HTTPS://` is
/// legal and turns up in autolinks. Matching case-sensitively would drop those
/// back to same-tab navigation silently, for a subset of links, which is the
/// worst shape a bug of this kind can take.
fn is_external(dest: &str) -> bool {
    let d = dest.trim();
    if d.starts_with("//") {
        return true;
    }
    let lower = d.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Escape for an HTML attribute value.
///
/// Deliberately a local copy of [`super::super::site::tokera::page::esc`]'s
/// behaviour rather than a call to it: markdown rendering has no business
/// depending on the page template, and this crate's two escapers sit on
/// opposite sides of that line. They must not drift — both escape
/// `& < > " '`, and anything added to one belongs in the other.
fn attr(s: &str) -> String {
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

fn find_heading_end(events: &[Event], start: usize) -> usize {
    events[start + 1..]
        .iter()
        .position(|e| matches!(e, Event::End(TagEnd::Heading(_))))
        .map(|p| start + 1 + p)
        // An unterminated heading cannot come out of the parser; if it somehow
        // did, treating the rest of the document as the heading is the one
        // outcome that must not happen.
        .unwrap_or(start + 1)
}

/// The readable text of a run of events — for slugs and the table of contents,
/// where markup would be noise.
fn plain_text(events: &[Event]) -> String {
    let mut s = String::new();
    for e in events {
        match e {
            Event::Text(t) | Event::Code(t) => s.push_str(t),
            Event::InlineMath(t) | Event::DisplayMath(t) => s.push_str(t),
            Event::SoftBreak | Event::HardBreak => s.push(' '),
            _ => {}
        }
    }
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_leading_h1_becomes_the_title_and_leaves_the_body() {
        let d = render("# One Card, One Stack\n\nBody text.\n");
        assert_eq!(d.title.as_deref(), Some("One Card, One Stack"));
        assert!(!d.html.contains("<h1"), "{}", d.html);
        assert!(d.html.contains("Body text."));
    }

    #[test]
    fn a_later_h1_is_kept_because_it_is_content() {
        let d = render("intro\n\n# Part One\n\nbody\n");
        assert!(d.title.is_none());
        assert!(d.html.contains("<h1"), "{}", d.html);
    }

    #[test]
    fn headings_get_ids_but_nothing_is_injected_into_the_prose() {
        let d = render("## §9.4 Throughput\n");
        assert!(d.html.contains("id=\"9-4-throughput\""), "{}", d.html);
        // The id makes a shared link work. Nothing visible is added to the
        // heading — the document renders exactly as it reads.
        assert!(!d.html.contains("<a"), "markup was injected: {}", d.html);
        assert!(
            d.html.trim().ends_with("</h2>"),
            "trailing markup: {}",
            d.html
        );
    }

    #[test]
    fn maths_renders_but_dollars_in_code_are_left_alone() {
        let d = render("Let $\\alpha = 2$.\n\n```sh\necho $PATH $HOME\n```\n");
        assert!(d.html.contains("<math"), "{}", d.html);
        // The shell line must survive verbatim — this is the failure mode of
        // every regex-based maths substitution.
        assert!(d.html.contains("echo $PATH $HOME"), "{}", d.html);
    }

    #[test]
    fn display_maths_becomes_a_block() {
        let d = render("$$S = \\frac{1}{3}$$\n");
        assert!(d.html.contains("class=\"math-block\""), "{}", d.html);
    }

    #[test]
    fn tables_and_footnotes_are_on() {
        let d = render("| a | b |\n|---|---|\n| 1 | 2 |\n");
        assert!(d.html.contains("<table>"), "{}", d.html);

        let d = render("text[^1]\n\n[^1]: the note\n");
        assert!(d.html.contains("footnote"), "{}", d.html);
    }

    /// A link that leaves the site opens in its own tab; one that does not, does
    /// not. Both halves matter — sending internal navigation to a new tab would
    /// litter the reader with tabs, which is the same bug wearing a hat.
    #[test]
    fn only_links_that_leave_the_site_open_a_new_tab() {
        for external in [
            "https://example.com/a",
            "http://example.com/a",
            // Schemes are case-insensitive and autolinks carry them verbatim.
            "HTTPS://example.com/a",
            "HtTp://example.com/a",
            // Scheme-relative is absolute in every way a reader cares about.
            "//example.com/a",
        ] {
            let d = render(&format!("see [it]({external})\n"));
            assert!(
                d.html.contains("target=\"_blank\""),
                "{external} did not open a new tab: {}",
                d.html
            );
            assert!(
                d.html.contains("rel=\"noopener noreferrer\""),
                "{external} has no opener guard: {}",
                d.html
            );
        }

        for internal in ["/blog/a-post", "/papers/one-card", "#a-heading", "a.md"] {
            let d = render(&format!("see [it]({internal})\n"));
            assert!(
                !d.html.contains("target="),
                "{internal} was treated as external: {}",
                d.html
            );
        }
    }

    /// The anchor is assembled by hand rather than by `push_html`, so the
    /// escaping is ours and has to be tested as ours.
    #[test]
    fn an_external_href_cannot_break_out_of_its_attribute() {
        let d = render("[x](https://e.com/\"><script>alert(1)</script>)\n");
        assert!(!d.html.contains("<script>alert"), "{}", d.html);
        assert!(
            d.html.contains("&quot;"),
            "the quote was not escaped: {}",
            d.html
        );
    }

    /// The tab change is announced to anyone who cannot see it happen.
    #[test]
    fn an_external_link_says_that_it_opens_a_new_tab() {
        let d = render("see [it](https://example.com)\n");
        assert!(
            d.html
                .contains("<span class=\"vh\"> (opens in a new tab)</span></a>"),
            "{}",
            d.html
        );
    }

    #[test]
    fn smart_punctuation_stays_off_so_paths_and_flags_survive() {
        let d = render("Run `web --check`, then -- really -- read it.\n");
        assert!(d.html.contains("--check"), "{}", d.html);
        assert!(!d.html.contains('\u{2013}'), "en dash appeared: {}", d.html);
    }

    #[test]
    fn html_in_the_source_is_not_a_way_to_inject_script() {
        // These documents are ours, so raw HTML is allowed through — but a
        // fenced block must never become live markup.
        let d = render("```html\n<script>alert(1)</script>\n```\n");
        assert!(d.html.contains("&lt;script&gt;"), "{}", d.html);
    }
}
