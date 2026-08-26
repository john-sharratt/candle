//! tokera.com — the home page, the blog, and the papers.
//!
//! This site has no daemon behind it. `web` is its backend, which is why the
//! router lives here rather than in a separate service: the whole of it is
//! reading markdown off disk and wrapping it in a shell, and a process to do
//! that would be a process to deploy, monitor and restart for no gain.
//!
//! Everything is server-rendered. Papers are long documents people deep-link
//! into, quote, print and read on bad connections; a page that needs a bundle
//! to execute before it shows a sentence fails all four.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State as AxState};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;

use crate::content::Roots;
use crate::markdown::Cache;

pub mod blog;
pub mod home;
pub mod page;
pub mod papers;

use page::{Meta, Nav, Width};

pub struct State {
    pub roots: Roots,
    /// Where the manifested papers are read from. `None` when the site has no
    /// `papers:` directory configured — the section then says so rather than
    /// showing an empty list that looks like a content problem.
    pub papers_dir: Option<PathBuf>,
    pub cache: Cache,
}

/// The site's routes. Mounted at the prefixes the site table gives them, so
/// this is the whole of tokera.com apart from its static assets.
pub fn router(roots: Roots, papers_dir: Option<PathBuf>) -> Router {
    let state = Arc::new(State {
        roots,
        papers_dir,
        cache: Cache::new(),
    });

    Router::new()
        .route("/", get(|| async { home::show().await }))
        .route(
            "/blog",
            get(|AxState(s): AxState<Arc<State>>| blog::index(s)),
        )
        .route(
            "/blog/:slug",
            get(
                |AxState(s): AxState<Arc<State>>, Path(slug): Path<String>| async move {
                    blog::show(s, &slug).await
                },
            ),
        )
        .route(
            "/papers",
            get(|AxState(s): AxState<Arc<State>>| papers::index(s)),
        )
        .route(
            "/papers/:slug",
            get(
                |AxState(s): AxState<Arc<State>>, Path(slug): Path<String>| async move {
                    papers::show(s, &slug).await
                },
            ),
        )
        .fallback(|| async { not_found(Nav::Home, "There is nothing at that address.") })
        .with_state(state)
}

/// A 404 that is a page, because everything this site serves is a page.
pub fn not_found(nav: Nav, detail: &str) -> Response {
    (StatusCode::NOT_FOUND, message(nav, "Not found", detail)).into_response()
}

/// Something is wrong with the content rather than with the request — the
/// distinction matters, because one of these is the author's problem to fix.
pub fn oops(nav: Nav, detail: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        message(nav, "That did not work", detail),
    )
        .into_response()
}

fn message(nav: Nav, heading: &str, detail: &str) -> Html<String> {
    let meta = Meta {
        heading,
        subtitle: None,
        byline: None,
        description: detail,
        nav,
        width: Width::Reading,
    };
    Html(format!(
        "{}{}<p class=\"empty\">{}</p>{}",
        page::head(&meta),
        page::title_block(&meta),
        page::esc(detail),
        page::foot()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn content_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("content")
    }

    fn site() -> Router {
        let roots = Roots::disk(&[content_root().join("tokera"), content_root().join("common")]);
        let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("docs")
            .canonicalize()
            .expect("the repo's docs/ directory");
        router(roots, Some(docs))
    }

    async fn get(path: &str) -> (StatusCode, String) {
        let res = site()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 64 << 20)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// `GET`, following one redirect. `/papers` opens the top paper.
    async fn follow(path: &str) -> (StatusCode, String) {
        let (status, body) = get(path).await;
        if status != StatusCode::TEMPORARY_REDIRECT && status != StatusCode::FOUND {
            return (status, body);
        }
        let res = site()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let to = res
            .headers()
            .get(axum::http::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .expect("a redirect names a location")
            .to_string();
        get(&to).await
    }

    #[tokio::test]
    async fn the_three_sections_render() {
        for path in ["/", "/blog", "/papers"] {
            let (status, html) = follow(path).await;
            assert_eq!(status, 200, "{path}");
            assert!(html.contains("<!doctype html>"), "{path}");
            assert!(html.contains("site-nav"), "{path} has no nav");
        }
    }

    #[tokio::test]
    async fn papers_opens_the_top_of_the_manifest_rather_than_a_chooser() {
        let manifest = papers::Manifest::parse(
            &std::fs::read_to_string(content_root().join("tokera").join("papers.yaml")).unwrap(),
        )
        .unwrap();
        let top = &manifest.papers[0];

        let res = site()
            .oneshot(
                Request::builder()
                    .uri("/papers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Temporary, never permanent: the target moves when a paper is added to
        // the top of the manifest, and a 301 would be cached against that.
        assert_eq!(res.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            res.headers()
                .get(axum::http::header::LOCATION)
                .and_then(|v| v.to_str().ok()),
            Some(format!("/papers/{}", top.slug).as_str())
        );

        // And what you land on is that paper, already open.
        let (status, html) = follow("/papers").await;
        assert_eq!(status, 200);
        assert!(
            html.contains("<article class=\"prose paper\">"),
            "no document"
        );
        assert!(html.contains(&page::esc(&top.title)));
    }

    /// The index leads with the featured posts, in the order they name, and
    /// only then falls back to newest-first.
    ///
    /// This is an editorial decision rather than a mechanical one — which is
    /// exactly why it needs pinning. A mistyped `feature:` silently reorders the
    /// front of the site, and nothing about the page looks broken afterwards.
    #[tokio::test]
    async fn the_blog_leads_with_the_featured_posts_in_order() {
        let (status, html) = get("/blog").await;
        assert_eq!(status, 200);

        // Slugs in the order the page lists them, first appearance only.
        let mut seen: Vec<String> = Vec::new();
        for chunk in html.split("href=\"/blog/").skip(1) {
            if let Some(slug) = chunk.split('"').next() {
                if !seen.iter().any(|s| s == slug) {
                    seen.push(slug.to_string());
                }
            }
        }

        let expected = [
            "waves-and-the-pcie-bottleneck",
            "palquant-per-block",
            "battle-cities-coming-soon",
            "one-card-unbounded-context",
        ];
        assert!(
            seen.len() > expected.len(),
            "the index listed almost nothing: {seen:?}"
        );
        assert_eq!(
            &seen[..expected.len()],
            &expected,
            "featured posts are not leading the index in order"
        );

        // And the tail is still date-descending, so opting out of `feature`
        // keeps the behaviour every other post relies on.
        let manifest_dates: Vec<String> = seen[expected.len()..]
            .iter()
            .map(|slug| {
                let path = content_root()
                    .join("tokera")
                    .join(blog::DIR)
                    .join(format!("{slug}.md"));
                let src = std::fs::read_to_string(&path).expect("a listed post exists on disk");
                let (post, _) =
                    crate::markdown::frontmatter::post(&src).expect("a listed post parses");
                post.date
            })
            .collect();
        let mut sorted = manifest_dates.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(
            manifest_dates, sorted,
            "the unfeatured tail is not newest-first"
        );
    }

    /// A paper carries the apparatus that makes it citable: the archived date,
    /// the DOI, and a citation a reader can lift.
    ///
    /// The date is the point of the test. It establishes priority, and it is
    /// only worth anything if it matches the deposit — a page that disagrees
    /// with the archive it is citing is worse than a page with no date at all.
    #[tokio::test]
    async fn a_paper_states_its_archived_date_and_doi() {
        let manifest = papers::Manifest::parse(
            &std::fs::read_to_string(content_root().join("tokera").join("papers.yaml")).unwrap(),
        )
        .expect("papers.yaml parses");
        for p in &manifest.papers {
            if p.doi.is_empty() {
                continue;
            }
            let (status, html) = get(&format!("/papers/{}", p.slug)).await;
            assert_eq!(status, 200, "{}", p.slug);

            assert!(
                html.contains(&format!("<time datetime=\"{}\">", p.published)),
                "{}: no machine-readable archived date",
                p.slug
            );
            assert!(
                html.contains(&format!("https://doi.org/{}", p.doi)),
                "{}: the DOI is not resolvable from the page",
                p.slug
            );
            // The citation has to carry the deposit year, not this year.
            // `get` rather than a slice: a paper with a DOI but a malformed
            // date should fail this assertion with something a reader can act
            // on, not panic inside the test harness.
            let year = p
                .published
                .get(..4)
                .unwrap_or_else(|| panic!("{}: `published` is not an ISO date", p.slug));
            assert!(
                html.contains(&format!("({year})")),
                "{}: citation is missing the deposit year {year}",
                p.slug
            );
            assert!(
                html.contains("data-act=\"print\"") && html.contains("data-act=\"copy\""),
                "{}: missing the print/copy controls",
                p.slug
            );
            if !p.pdf.is_empty() {
                assert!(
                    html.contains(&page::esc(&p.pdf)),
                    "{}: no download link to the deposited PDF",
                    p.slug
                );
            }
        }
    }

    /// Every paper the manifest names, rendered from the working document in
    /// `docs/`. Driven off the manifest rather than a hard-coded list, so
    /// publishing a paper cannot skip this check.
    #[tokio::test]
    async fn every_published_paper_renders_from_its_working_document() {
        let manifest = papers::Manifest::parse(
            &std::fs::read_to_string(content_root().join("tokera").join("papers.yaml")).unwrap(),
        )
        .expect("papers.yaml parses");
        assert!(!manifest.papers.is_empty(), "nothing is published");

        for paper in &manifest.papers {
            let (status, html) = get(&format!("/papers/{}", paper.slug)).await;
            assert_eq!(status, 200, "{} did not render", paper.slug);
            assert!(
                html.contains(&page::esc(&paper.title)),
                "{}: no title",
                paper.slug
            );
            assert!(html.contains("<table"), "{}: no tables", paper.slug);
            assert!(html.contains("<math"), "{}: no maths", paper.slug);
            // Linear: the document's own headings and prose, with no contents
            // apparatus and nothing injected between them.
            assert!(html.contains("<h2 id="), "{}: no headings", paper.slug);
            assert!(
                !html.contains("class=\"toc\"") && !html.contains("class=\"anchor\""),
                "{}: navigation furniture is back in the prose",
                paper.slug
            );
            // The marker `latex2mathml` leaves in its output when it meets a
            // construct it cannot convert. Publishing that would be visible to
            // every reader.
            assert!(
                !html.contains("PARSE ERROR"),
                "{}: an expression leaked a converter error",
                paper.slug
            );
            // Every expression in a published paper must actually typeset. The
            // fallback exists so one unconvertible expression cannot take the
            // page down — not so a paper's central definition can quietly
            // render as raw LaTeX, which is what `\mathcal` and `\big[` each
            // did until they were handled.
            let fell_back = html.matches("math-raw").count();
            assert_eq!(
                fell_back, 0,
                "{}: {fell_back} expressions rendered as literal LaTeX",
                paper.slug
            );
        }
    }

    /// The nav mark and the tab icon are one file, not two drawings.
    ///
    /// They used to be two — an inline SVG and a `favicon.svg` holding the same
    /// coordinates — and this test compared the geometry because two copies of
    /// one drawing is exactly the thing that drifts. Pointing both at the same
    /// file removes the failure rather than detecting it, so what is left to
    /// check is that the file is really there and that nothing still reaches
    /// for the drawing it replaced.
    #[tokio::test]
    async fn the_nav_mark_and_the_favicon_are_one_file() {
        let (_, page) = get("/").await;

        assert!(
            page.contains(r#"<link rel="icon" type="image/png" href="/favicon.png">"#),
            "the tab icon is not the brand mark"
        );
        // The nav mark is now the switcher's own chip — same file, and
        // relative, so it is this site's mark whichever host is serving it.
        let summary = page
            .split("<summary")
            .nth(1)
            .and_then(|s| s.split("</summary>").next())
            .expect("the switcher's summary");
        assert!(
            summary.contains("url('/favicon.png')"),
            "the nav mark is not the brand mark: {summary}"
        );
        // Root-relative specifically. Other sites in the switcher legitimately
        // have `.svg` favicons, and their absolute URLs are not this site's
        // superseded drawing.
        for stale in [r#"href="/favicon.svg""#, "url('/favicon.svg')"] {
            assert!(
                !page.contains(stale),
                "something still points at the superseded drawing: {stale}"
            );
        }

        let mark = content_root().join("tokera").join("favicon.png");
        let bytes =
            std::fs::read(&mark).unwrap_or_else(|e| panic!("reading {}: {e}", mark.display()));
        assert!(
            bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "the brand mark is not a PNG"
        );
    }

    #[tokio::test]
    async fn the_index_pane_lists_what_the_manifest_names() {
        let (_, html) = follow("/papers").await;
        assert!(html.contains("href=\"/papers/one-card\""), "{html}");
        assert!(html.contains("href=\"/papers/palquant\""), "{html}");
    }

    /// The two-pane reader: an index that stays put and a document that scrolls
    /// on its own. Every piece of this is load-bearing and none of it is
    /// visible in a unit test except as markup, so assert on the markup.
    #[tokio::test]
    async fn the_papers_section_is_a_two_pane_reader() {
        let manifest = papers::Manifest::parse(
            &std::fs::read_to_string(content_root().join("tokera").join("papers.yaml")).unwrap(),
        )
        .unwrap();

        for path in ["/papers", "/papers/one-card", "/papers/palquant"] {
            let (status, html) = follow(path).await;
            assert_eq!(status, 200, "{path}");

            // Pinned to the viewport, or the right pane cannot scroll alone.
            assert!(
                html.contains("class=\"doc pinned\""),
                "{path}: not pinned, so both panes would scroll together"
            );
            assert!(html.contains("<main class=\"split\">"), "{path}: no split");
            assert!(
                html.contains("<aside class=\"pane-index\">"),
                "{path}: no index pane"
            );
            assert!(
                html.contains("<section class=\"pane-view\">"),
                "{path}: no view pane"
            );

            // The index survives navigation — that is the point of two panes.
            for p in &manifest.papers {
                assert!(
                    html.contains(&format!("href=\"/papers/{}\"", p.slug)),
                    "{path}: {} missing from the index pane",
                    p.slug
                );
            }

            // The footer belongs inside the scrolling pane, not under both.
            let view_at = html.find("pane-view").unwrap();
            let foot_at = html.find("site-foot").unwrap();
            assert!(foot_at > view_at, "{path}: the footer escaped the pane");
        }
    }

    #[tokio::test]
    async fn the_open_paper_is_marked_in_the_index() {
        // Deliberately the paper that is NOT the default, so this cannot pass
        // by accident on whatever `/papers` happens to open.
        let (_, html) = get("/papers/one-card").await;
        assert!(
            html.contains("href=\"/papers/one-card\" aria-current=\"page\""),
            "the open paper is not marked"
        );
        assert_eq!(
            html.matches("aria-current=\"page\"").count(),
            2,
            "exactly one paper and one nav item should be current"
        );
    }

    /// Every post on the index, opened and checked. Driven off the index rather
    /// than a hard-coded list, so adding a post cannot skip this.
    #[tokio::test]
    async fn every_post_renders_with_its_figures_and_callouts() {
        let (_, index) = get("/blog").await;
        let slugs: Vec<String> = index
            .split("href=\"/blog/")
            .skip(1)
            .filter_map(|s| s.split('"').next())
            .map(str::to_owned)
            .collect();
        assert!(slugs.len() >= 5, "only {} posts on the index", slugs.len());

        for slug in &slugs {
            let (status, html) = get(&format!("/blog/{slug}")).await;
            assert_eq!(status, 200, "{slug}");
            assert!(
                html.contains("<article class=\"prose\">"),
                "{slug}: no body"
            );
            // The front matter must never reach the page — the symptom is a
            // rule and a paragraph of YAML above the first sentence.
            assert!(!html.contains("summary:"), "{slug}: front matter leaked");
            // Each post carries a tint class, so it is coloured rather than grey.
            assert!(
                html.contains("class=\"tint-"),
                "{slug}: no tint, so the post renders in the default accent"
            );
        }
    }

    /// Posts embed diagrams as raw inline SVG. Markdown passes raw HTML through,
    /// but a fenced block must not — this checks both halves of that.
    #[tokio::test]
    async fn inline_svg_survives_markdown_but_fenced_code_does_not() {
        let (_, html) = get("/blog/one-card-unbounded-context").await;
        assert!(html.contains("<svg viewBox="), "the figure did not render");
        assert!(html.contains("<figcaption>"), "the caption did not render");
        assert!(html.contains("class=\"key\""), "the callout did not render");
        // The animation depends on this: without it the dash offset is measured
        // in user units against a path of unknown length.
        assert!(html.contains("pathLength=\"100\""), "curves will not draw");
    }

    /// **A blank line inside a raw HTML block ends the block** in CommonMark.
    /// Everything after it is re-parsed as markdown, and any line indented four
    /// spaces — which is most of an SVG — becomes an indented code block and
    /// renders as escaped text. It happened to two of these posts, it is
    /// invisible in the source, so the check has to be on the output.
    #[tokio::test]
    async fn no_post_leaks_escaped_markup_from_a_broken_html_block() {
        let (_, index) = get("/blog").await;
        let slugs: Vec<String> = index
            .split("href=\"/blog/")
            .skip(1)
            .filter_map(|s| s.split('"').next())
            .map(str::to_owned)
            .collect();
        assert!(!slugs.is_empty(), "no posts to check");

        for slug in &slugs {
            let (_, html) = get(&format!("/blog/{slug}")).await;
            for tag in [
                "&lt;svg",
                "&lt;rect",
                "&lt;path",
                "&lt;text",
                "&lt;circle",
                "&lt;g ",
                "&lt;div",
            ] {
                assert!(
                    !html.contains(tag),
                    "{slug}: `{tag}` was escaped — a blank line inside a raw HTML block \
                     closed it early, and the rest became an indented code block"
                );
            }

            // The quieter half of the same bug. When the re-parsed remainder is
            // indented fewer than four spaces it is a paragraph of inline HTML,
            // not a code block: nothing is escaped, so the check above passes,
            // and a `<p>` lands inside the `<svg>` where no `<p>` may go. The
            // browser hoists it out and takes the rest of the figure with it.
            for figure in html.split("<svg").skip(1) {
                let body = figure.split("</svg>").next().unwrap_or_default();
                assert!(
                    !body.contains("<p>"),
                    "{slug}: a `<p>` is inside an `<svg>` — a blank line inside the \
                     figure closed its HTML block and the remainder was re-parsed \
                     as markdown paragraphs"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_post_renders_and_the_index_links_to_it() {
        let (_, index) = get("/blog").await;
        assert!(
            index.contains("href=\"/blog/"),
            "the index lists no posts:\n{index}"
        );

        let slug = index
            .split("href=\"/blog/")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("a post link")
            .to_string();
        let (status, html) = get(&format!("/blog/{slug}")).await;
        assert_eq!(status, 200);
        assert!(html.contains("<article class=\"prose\">"), "{html}");
    }

    #[tokio::test]
    async fn unknown_slugs_are_404_pages_rather_than_500s() {
        for path in ["/papers/nope", "/blog/nope", "/nothing-here"] {
            let (status, html) = get(path).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert!(html.contains("<!doctype html>"), "{path} was not a page");
        }
    }

    #[tokio::test]
    async fn a_slug_cannot_reach_outside_the_content() {
        // `..%2f` decodes to `../` before routing, so this is the real shape of
        // the attack rather than a literal `..` that the router would reject.
        for path in ["/blog/..%2f..%2fCargo.toml", "/papers/..%2f..%2fCargo.toml"] {
            let (status, body) = get(path).await;
            assert_ne!(status, StatusCode::OK, "{path}");
            assert!(!body.contains("[package]"), "{path} escaped");
        }
    }
}
