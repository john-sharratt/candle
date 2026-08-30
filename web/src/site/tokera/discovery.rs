//! What a crawler and a reader ask for before they ask for a page:
//! `/robots.txt`, `/sitemap.xml`, and the blog's feed.
//!
//! # Generated, never a file
//!
//! All three are built from the same walk that renders the index — the posts on
//! disk and the papers named in `papers.yaml`. A checked-in `sitemap.xml` is a
//! list that is correct on the day it is written and wrong the first time
//! somebody adds a post without remembering it, and the failure is silent in
//! both directions: a new page nothing points at, or a dead URL a crawler keeps
//! returning to.
//!
//! Nothing here is expensive. The blog walk is a directory read of fifteen
//! files and the manifest is one YAML document, which is what the index page
//! already does on every request.
//!
//! # Drafts stay out
//!
//! A post marked `draft: true` is readable at its own address — that is how you
//! show somebody an unfinished piece — and must not appear in any of these. A
//! sitemap is a statement that a page is ready to be indexed, and a draft is
//! the definition of one that is not.

use std::sync::Arc;

use axum::http::header;
use axum::response::{IntoResponse, Response};

use super::page::{esc, ORIGIN};
use super::{blog, papers, State};

/// One entry of the sitemap.
struct Url {
    path: String,
    /// `YYYY-MM-DD`, when the page itself carries a date.
    date: Option<String>,
    /// A hint, not a promise: how often this address is worth re-reading.
    changes: &'static str,
    /// Relative to the other pages on *this* site only, which is all this
    /// number has ever meant.
    priority: &'static str,
}

/// `/robots.txt` — open to crawlers, and pointing at the map.
///
/// Deliberately permissive. Everything this site serves is meant to be read and
/// found; there is no members' area to keep out of an index, and the one thing
/// worth excluding is `/_/`, which does not exist yet but would be the place a
/// machine-only route went.
///
/// `Sitemap:` is absolute, because the line is read outside the context of the
/// host that served it.
pub async fn robots() -> Response {
    let body = format!(
        "# Tokera — everything here is meant to be read.\n\
         User-agent: *\n\
         Allow: /\n\
         \n\
         Sitemap: {ORIGIN}/sitemap.xml\n"
    );
    text(body, "text/plain; charset=utf-8")
}

/// `/sitemap.xml` — every page worth indexing, with the date it last said
/// something new.
pub async fn sitemap(state: Arc<State>) -> Response {
    let mut urls = vec![
        Url {
            path: "/".into(),
            date: None,
            changes: "weekly",
            priority: "1.0",
        },
        Url {
            path: "/blog".into(),
            date: None,
            changes: "weekly",
            priority: "0.8",
        },
        // `/papers` is deliberately absent. It redirects to the newest paper
        // rather than rendering an index, and a sitemap is a list of addresses
        // that *are* pages — putting a redirect in one asks a crawler to
        // discover the destination twice and to wonder which is canonical.
    ];

    for (slug, post) in blog::published(&state).await {
        urls.push(Url {
            path: format!("/blog/{slug}"),
            date: Some(post.date.clone()),
            // A post is finished when it is published. Saying `weekly` here
            // would ask a crawler to keep coming back to something that does
            // not change, and spend the site's crawl budget on it.
            changes: "yearly",
            priority: "0.7",
        });
    }

    for paper in papers::manifest(&state).await.papers {
        urls.push(Url {
            path: format!("/papers/{}", paper.slug),
            date: (!paper.date.is_empty()).then(|| paper.date.clone()),
            changes: "monthly",
            priority: "0.9",
        });
    }

    let mut body = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n",
    );
    for u in &urls {
        body.push_str("  <url>\n");
        body.push_str(&format!("    <loc>{ORIGIN}{}</loc>\n", esc(&u.path)));
        if let Some(d) = &u.date {
            body.push_str(&format!("    <lastmod>{}</lastmod>\n", esc(d)));
        }
        body.push_str(&format!("    <changefreq>{}</changefreq>\n", u.changes));
        body.push_str(&format!("    <priority>{}</priority>\n", u.priority));
        body.push_str("  </url>\n");
    }
    body.push_str("</urlset>\n");

    text(body, "application/xml; charset=utf-8")
}

/// `/blog/feed.xml` — the posts, for anyone who would rather be told.
///
/// RSS 2.0 rather than Atom for no better reason than that more readers take it
/// without argument, and the two carry the same information here.
///
/// The dates go out in RFC 822, which is what the format specifies. A post's
/// stored date is `YYYY-MM-DD` with no time — so it becomes midnight UTC, which
/// is a statement about the date and honest about knowing nothing finer.
pub async fn feed(state: Arc<State>) -> Response {
    let posts = blog::published(&state).await;

    let mut body = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <rss version=\"2.0\" xmlns:atom=\"http://www.w3.org/2005/Atom\">\n<channel>\n",
    );
    body.push_str("<title>Tokera — the blog</title>\n");
    body.push_str(&format!("<link>{ORIGIN}/blog</link>\n"));
    body.push_str(&format!(
        "<atom:link href=\"{ORIGIN}/blog/feed.xml\" rel=\"self\" type=\"application/rss+xml\"/>\n"
    ));
    body.push_str(
        "<description>Writing from Tokera on inference, memory and agent architecture.\
         </description>\n<language>en</language>\n",
    );

    for (slug, post) in &posts {
        body.push_str("<item>\n");
        body.push_str(&format!("  <title>{}</title>\n", esc(&post.title)));
        body.push_str(&format!("  <link>{ORIGIN}/blog/{}</link>\n", esc(slug)));
        // The permalink is the address, which is what makes it stable across a
        // retitle — a reader that keyed on the title would show the post again.
        body.push_str(&format!(
            "  <guid isPermaLink=\"true\">{ORIGIN}/blog/{}</guid>\n",
            esc(slug)
        ));
        if let Some(d) = rfc822(&post.date) {
            body.push_str(&format!("  <pubDate>{d}</pubDate>\n"));
        }
        body.push_str(&format!(
            "  <description>{}</description>\n",
            esc(&post.summary)
        ));
        body.push_str("</item>\n");
    }
    body.push_str("</channel>\n</rss>\n");

    text(body, "application/rss+xml; charset=utf-8")
}

/// `YYYY-MM-DD` as RFC 822, or `None` if it is not that.
///
/// The weekday is required by the format and is computed rather than guessed —
/// a wrong one is the kind of thing a strict reader rejects the whole item over.
fn rfc822(date: &str) -> Option<String> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];

    let mut parts = date.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }

    // Days since the epoch, by the civil-date algorithm. The weekday follows
    // from it, and 1970-01-01 was a Thursday — which is why `DAYS` starts there.
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let weekday = DAYS[days.rem_euclid(7) as usize];

    Some(format!(
        "{weekday}, {d:02} {} {y} 00:00:00 +0000",
        MONTHS[(m - 1) as usize]
    ))
}

/// One of these documents, with the type that makes a reader parse it.
///
/// A short age rather than the site's default: these are indexes of everything
/// else, so a stale one is a crawler being told about a post that no longer
/// exists or missing one that does.
fn text(body: String, mime: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, mime),
            (header::CACHE_CONTROL, "public, max-age=300"),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weekday has to be right, or a strict reader drops the item.
    #[test]
    fn dates_become_rfc_822_with_the_correct_weekday() {
        // 1970-01-01 was a Thursday; the algorithm is anchored there.
        assert_eq!(
            rfc822("1970-01-01").unwrap(),
            "Thu, 01 Jan 1970 00:00:00 +0000"
        );
        // 2026-08-11 is a Tuesday — the date on a real post in this blog.
        assert_eq!(
            rfc822("2026-08-11").unwrap(),
            "Tue, 11 Aug 2026 00:00:00 +0000"
        );
        // A leap day, which is where a hand-rolled calendar goes wrong.
        assert_eq!(
            rfc822("2024-02-29").unwrap(),
            "Thu, 29 Feb 2024 00:00:00 +0000"
        );
        assert_eq!(
            rfc822("2000-03-01").unwrap(),
            "Wed, 01 Mar 2000 00:00:00 +0000"
        );
    }

    /// Anything that is not a date is left out rather than guessed at.
    #[test]
    fn a_date_that_is_not_one_produces_nothing() {
        for bad in ["", "soon", "2026", "2026-13-01", "2026-01-32", "x-y-z"] {
            assert!(rfc822(bad).is_none(), "accepted `{bad}`");
        }
    }

    /// `robots.txt` has to point at the sitemap absolutely — the line is read
    /// away from the host that served it.
    #[tokio::test]
    async fn robots_names_the_sitemap_in_full() {
        let res = robots().await;
        assert_eq!(res.status(), 200);
        let body = axum::body::to_bytes(res.into_body(), 1 << 16)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("Sitemap: https://tokera.com/sitemap.xml"), "{body}");
        assert!(body.contains("Allow: /"), "{body}");
    }
}
