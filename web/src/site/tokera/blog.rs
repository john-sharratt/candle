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

use axum::http::HeaderMap;
use axum::response::Response;

use crate::markdown::{frontmatter, Post};

use super::page::{self, Kind, Meta, Nav, Width};
use super::{page_answer, State};

pub const DIR: &str = "blog";

/// What a post's front matter is kept as.
const KIND: &str = "post";

struct Listed {
    slug: String,
    meta: Arc<Post>,
}

/// Every published post — slug, title, date and summary.
///
/// `pub(super)` so the sitemap and the feed are built from the same walk the
/// index is. A second enumeration would be a second definition of "published",
/// free to disagree about a draft.
pub(super) async fn published(state: &State) -> Vec<(String, Arc<Post>)> {
    listing(state)
        .await
        .into_iter()
        .map(|l| (l.slug, l.meta))
        .collect()
}

/// The front matter of the post at `rel`, when it is published: parsed once
/// per version of its file and kept, a draft or a broken one as `None`.
async fn post_at(state: &State, rel: &str) -> Option<Arc<Post>> {
    let resolved = state.roots.resolve(rel).await?;
    let key = resolved.key(KIND);
    if let Some(kept) = state.posts.get(&key, resolved.stamp()) {
        return kept;
    }
    let text = String::from_utf8(resolved.bytes().await?).ok()?;
    let post = match frontmatter::post(&text) {
        Ok((meta, _)) if !meta.draft => Some(Arc::new(meta)),
        Ok(_) => None,
        // A post with broken front matter is a mistake to fix, not a reason for
        // the index to fail — the rest of the blog still lists.
        Err(e) => {
            tracing::warn!(post = %rel, error = %e, "skipping post");
            None
        }
    };
    if resolved.unchanged().await {
        state
            .posts
            .put(key, resolved.stamp(), post.clone(), text.len());
    }
    post
}

async fn listing(state: &State) -> Vec<Listed> {
    let mut out = Vec::new();
    for name in state.roots.list(DIR).await {
        let Some(slug) = name.strip_suffix(".md") else {
            continue;
        };
        if let Some(meta) = post_at(state, &format!("{DIR}/{name}")).await {
            out.push(Listed {
                slug: slug.to_string(),
                meta,
            });
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

pub async fn index(state: Arc<State>, req: &HeaderMap) -> Response {
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
        heading: "Blog",
        subtitle: None,
        byline: None,
        description: "Writing from Tokera on inference, memory and agent architecture.",
        nav: Nav::Blog,
        width: Width::Split,
        path: "/blog",
        kind: Kind::Site,
        image: None,
        published: None,
    };
    // Unlike the papers section, `/blog` renders an index rather than
    // redirecting to the newest post: with summaries to read, choosing what to
    // read next is a real decision rather than a page in the way.
    let html = page::split(
        &meta,
        &pane_index(&posts, None),
        &format!("<div class=\"entries\">{body}</div>"),
    );
    page_answer(&state, html, req)
}

pub async fn show(state: Arc<State>, slug: &str, req: &HeaderMap) -> Response {
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
        // The file went between the read above and this lookup: render what
        // was read, uncached, rather than key a cache on a file that is gone.
        None => Arc::new(crate::markdown::render(frontmatter::split(&text).1)),
    };

    // The social card is the post's own opening illustration, pulled out of the
    // rendered body rather than named a second time in the frontmatter.
    let path = format!("/blog/{slug}");
    let meta = Meta {
        heading: &post.title,
        subtitle: None,
        byline: Some(&post.date),
        description: &post.summary,
        nav: Nav::Blog,
        width: Width::Split,
        path: &path,
        kind: Kind::Article,
        image: page::first_image(&doc.html),
        published: Some(&post.date),
    };
    // The tint goes on a wrapper around the whole document, so the heading
    // rules, links and callouts inside it all resolve `--post` to the same
    // colour without any of them naming it.
    let html = page::split(
        &meta,
        &pane_index(&listing(&state).await, Some(slug)),
        &format!(
            "<div class=\"{}\">{}<article class=\"prose\">{}</article></div>",
            post.tint.class(),
            page::title_block(&meta),
            doc.html
        ),
    );
    page_answer(&state, html, req)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::time::{Duration, SystemTime};

    use filetime::{set_file_mtime, FileTime};
    use tempfile::TempDir;

    use super::*;
    use crate::config::Cache as Policy;
    use crate::content::{Roots, Stamped};
    use crate::markdown::{frontmatter, Cache};

    fn write_post(dir: &Path, slug: &str, title: &str, secs: u64) {
        let path = dir.join(DIR).join(format!("{slug}.md"));
        fs::write(
            &path,
            format!("---\ntitle: {title}\ndate: 2026-08-25\nsummary: s\n---\nbody\n"),
        )
        .unwrap();
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        set_file_mtime(&path, FileTime::from_system_time(at)).unwrap();
    }

    /// **A post is parsed once per version of its file**: two listings share
    /// each post, and an edited post's alone is parsed again.
    #[tokio::test]
    async fn a_listing_parses_only_what_changed() {
        let dir = TempDir::new().unwrap();
        fs::create_dir(dir.path().join(DIR)).unwrap();
        write_post(dir.path(), "a", "A", 1_000);
        write_post(dir.path(), "b", "B", 1_000);
        let state = State {
            roots: Roots::disk(&[dir.path().to_path_buf()]),
            papers_dir: None,
            cache: Cache::new(),
            policy: Policy {
                short_secs: 60,
                long_secs: 900,
            },
            posts: Stamped::default(),
            manifest: Stamped::default(),
        };
        let first = listing(&state).await;
        let second = listing(&state).await;
        assert_eq!(first.len(), 2);
        for (x, y) in first.iter().zip(&second) {
            assert!(Arc::ptr_eq(&x.meta, &y.meta), "{}", x.slug);
        }

        write_post(dir.path(), "b", "B, revised", 2_000);
        let third = listing(&state).await;
        let by =
            |l: &[Listed], slug: &str| Arc::clone(&l.iter().find(|p| p.slug == slug).unwrap().meta);
        assert!(Arc::ptr_eq(&by(&first, "a"), &by(&third, "a")));
        assert!(!Arc::ptr_eq(&by(&first, "b"), &by(&third, "b")));
        assert_eq!(by(&third, "b").title, "B, revised");
    }

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
