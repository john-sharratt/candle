//! The blog.
//!
//! A post is one markdown file in `content/tokera/blog/`, with front matter for
//! its title and date. The index is the directory: there is no manifest to keep
//! in step, so publishing is `git add` and nothing else, and a post cannot be
//! written and then forgotten because someone missed the second edit.
//!
//! Ordering is by the `date` in the front matter, newest first, not by mtime —
//! a typo fix should not move a two-year-old post to the top. A post may also
//! carry `feature: N`, which lifts it out of the date order and into an
//! editorial one: the featured posts lead the index in ascending `N`, and
//! everything without a `feature` follows by date as before.

use std::sync::Arc;

use axum::response::{Html, IntoResponse, Response};

use crate::markdown::{frontmatter, Post};

use super::page::{self, Meta, Nav, Width};
use super::State;

pub const DIR: &str = "blog";

struct Listed {
    slug: String,
    meta: Post,
}

async fn listing(state: &State) -> Vec<Listed> {
    let mut out = Vec::new();
    for name in state.roots.list(DIR).await {
        let Some(slug) = name.strip_suffix(".md") else {
            continue;
        };
        let Some(bytes) = state.roots.read(&format!("{DIR}/{name}")).await else {
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        match frontmatter::post(&text) {
            Ok((meta, _)) if !meta.draft => out.push(Listed {
                slug: slug.to_string(),
                meta,
            }),
            Ok(_) => {}
            // A post with broken front matter is a mistake to fix, not a reason
            // for the index to fail — the rest of the blog still lists.
            Err(e) => tracing::warn!(post = %name, error = %e, "skipping post"),
        }
    }
    // Featured posts lead, in the order they name; the rest follow by date,
    // newest first. `None` sorts after every `Some`, so a post opts in to the
    // front of the list and opts out by saying nothing.
    out.sort_by(|a, b| {
        match (a.meta.feature, b.meta.feature) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then_with(|| b.meta.date.cmp(&a.meta.date))
        .then_with(|| a.slug.cmp(&b.slug))
    });
    out
}

/// The left pane: every post, newest first, with the one being read marked.
///
/// Rendered on both routes so it survives navigation — same reader as the
/// papers section. Each row carries its post's tint, so the list is the site's
/// most colourful surface rather than a column of grey links.
fn pane_index(posts: &[Listed], current: Option<&str>) -> String {
    if posts.is_empty() {
        return "<p class=\"empty\">Nothing published yet.</p>".to_string();
    }
    let items = posts
        .iter()
        .map(|p| {
            let here = current == Some(p.slug.as_str());
            format!(
                "<li class=\"{tint}\"><a href=\"/blog/{slug}\"{aria} class=\"{cls}\">\
                   <span class=\"t\">{title}</span>\
                   <span class=\"m\">{date}</span>\
                 </a></li>",
                tint = p.meta.tint.class(),
                slug = page::esc(&p.slug),
                aria = if here { " aria-current=\"page\"" } else { "" },
                cls = if here { "is-current" } else { "" },
                title = page::esc(&p.meta.title),
                date = page::esc(&p.meta.date),
            )
        })
        .collect::<String>();

    format!("<p class=\"pane-title\">Writing</p><ul class=\"pane-list\">{items}</ul>")
}

pub async fn index(state: Arc<State>) -> Response {
    let posts = listing(&state).await;
    let mut body = String::from(
        "<header class=\"doc-head\"><h1>Blog</h1>\
         <p class=\"doc-sub\">Notes on building the engine, and on what it turns out to \
         imply.</p></header>",
    );
    if posts.is_empty() {
        body.push_str("<p class=\"empty\">Nothing published yet.</p>");
    }
    for p in &posts {
        let tags = p
            .meta
            .tags
            .iter()
            .map(|t| format!("<span class=\"chip\">{}</span>", page::esc(t)))
            .collect::<String>();
        body.push_str(&format!(
            r#"<article class="entry {tint}">
  <h2><a href="/blog/{slug}">{title}</a></h2>
  <p class="entry-meta"><time datetime="{date}">{date}</time>{tags}</p>
  <p class="entry-sum">{summary}</p>
</article>"#,
            tint = p.meta.tint.class(),
            slug = page::esc(&p.slug),
            title = page::esc(&p.meta.title),
            date = page::esc(&p.meta.date),
            tags = tags,
            summary = page::esc(&p.meta.summary),
        ));
    }

    let meta = Meta {
        title: "Blog",
        heading: "Blog",
        subtitle: None,
        byline: None,
        description: "Writing from Tokera on inference, memory and agent architecture.",
        nav: Nav::Blog,
        width: Width::Split,
    };
    // Unlike the papers section, `/blog` renders an index rather than
    // redirecting to the newest post: with summaries to read, choosing what to
    // read next is a real decision rather than a page in the way.
    Html(page::split(
        &meta,
        &pane_index(&posts, None),
        &format!("<div class=\"entries\">{body}</div>"),
    ))
    .into_response()
}

pub async fn show(state: Arc<State>, slug: &str) -> Response {
    // A slug is one path segment and nothing else — it is about to become a
    // file name.
    if slug.is_empty() || slug.contains(['/', '\\', '.']) {
        return super::not_found(Nav::Blog, "No post by that name.");
    }
    let rel = format!("{DIR}/{slug}.md");
    let Some(bytes) = state.roots.read(&rel).await else {
        return super::not_found(Nav::Blog, "No post by that name.");
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return super::oops(Nav::Blog, "That post is not valid UTF-8.");
    };
    let Ok((post, _)) = frontmatter::post(&text) else {
        return super::oops(
            Nav::Blog,
            "That post is missing its front matter, so it has no title or date.",
        );
    };

    // Rendered through the same cache as the papers, keyed by the file on disk
    // — the front matter is stripped before rendering so it does not become a
    // stray `<hr>` and a paragraph of YAML at the top of the post.
    let doc = match state.roots.disk_path(&rel) {
        Some(path) => match state
            .cache
            .get(&path, |s| frontmatter::split(s).1.to_string())
            .await
        {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(error = %e, "post unreadable after listing");
                return super::oops(Nav::Blog, "That post could not be read.");
            }
        },
        // Embedded content has no path to key a cache on, and no mtime to
        // invalidate by — it cannot change without a rebuild, so rendering it
        // each time is the honest option rather than caching it forever.
        None => Arc::new(crate::markdown::render(frontmatter::split(&text).1)),
    };

    let meta = Meta {
        title: &post.title,
        heading: &post.title,
        subtitle: None,
        byline: Some(&post.date),
        description: &post.summary,
        nav: Nav::Blog,
        width: Width::Split,
    };
    // The tint goes on a wrapper around the whole document, so the heading
    // rules, links and callouts inside it all resolve `--post` to the same
    // colour without any of them naming it.
    Html(page::split(
        &meta,
        &pane_index(&listing(&state).await, Some(slug)),
        &format!(
            "<div class=\"{}\">{}<article class=\"prose\">{}</article></div>",
            post.tint.class(),
            page::title_block(&meta),
            doc.html
        ),
    ))
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::markdown::frontmatter;

    #[test]
    fn a_posts_front_matter_never_reaches_the_rendered_body() {
        // The failure this guards is unmistakable on the page: a horizontal
        // rule and a paragraph of YAML above the first sentence.
        let src = "---\ntitle: T\ndate: 2026-08-25\n---\n# Heading\n\nbody\n";
        let doc = crate::markdown::render(frontmatter::split(src).1);
        assert!(!doc.html.contains("date:"), "{}", doc.html);
        assert!(doc.html.contains("body"), "{}", doc.html);
    }
}
