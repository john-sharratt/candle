//! The papers section.
//!
//! Papers are **not copied into the site**. Each one is a working document in
//! `docs/`, rendered live at read time, so the published version is never a
//! stale duplicate of the real one and publishing is not a step anybody has to
//! remember.
//!
//! What is published is decided by `papers.yaml` in the site's content, not by
//! what happens to be in the directory. That manifest is the difference between
//! "the two papers are online" and "the entire design directory is online",
//! and it is also where the title, authors and blurb live — metadata that has
//! no business inside a document that is still being edited.

use std::path::PathBuf;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use serde::Deserialize;

use super::page::{self, Meta, Nav, Width};
use super::State;

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Paper {
    /// URL segment: `/papers/<slug>`. Fixed by hand rather than derived from
    /// the title, because a title can be revised and a shared link cannot.
    pub slug: String,
    pub title: String,
    #[serde(default)]
    pub subtitle: String,
    #[serde(default)]
    pub authors: String,
    #[serde(default)]
    pub date: String,
    /// ISO `YYYY-MM-DD` of the archived record, which is the date that
    /// establishes priority. It must match the deposit exactly — the whole
    /// value of a DOI is that a third party timestamped it, and a page that
    /// disagrees with the archive undermines the thing it is citing.
    #[serde(default)]
    pub published: String,
    /// Bare DOI, no `https://doi.org/` prefix — the prefix is presentation and
    /// belongs in the template, the identifier belongs here.
    #[serde(default)]
    pub doi: String,
    /// The archived record's landing page.
    #[serde(default)]
    pub zenodo: String,
    /// Direct link to the deposited PDF. That file is the citable artefact and
    /// this site's rendering is not — the document here tracks the working copy
    /// and will drift from the deposit by design.
    #[serde(default)]
    pub pdf: String,
    #[serde(default)]
    pub summary: String,
    /// File name inside the site's `papers:` directory. A bare name — no
    /// separators — so a manifest entry cannot reach out of it.
    pub source: String,
    #[serde(default)]
    pub status: String,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(default)]
    pub papers: Vec<Paper>,
}

impl Manifest {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let m: Manifest = serde_yaml::from_str(text)?;
        for p in &m.papers {
            if p.source.contains('/') || p.source.contains('\\') || p.source.contains("..") {
                anyhow::bail!(
                    "paper `{}`: source `{}` must be a plain file name in the papers directory",
                    p.slug,
                    p.source
                );
            }
        }
        Ok(m)
    }
}

pub async fn manifest(state: &State) -> Manifest {
    let Some(bytes) = state.roots.read("papers.yaml").await else {
        return Manifest::default();
    };
    match std::str::from_utf8(&bytes)
        .map_err(anyhow::Error::from)
        .and_then(Manifest::parse)
    {
        Ok(m) => m,
        Err(e) => {
            // Loud, because the symptom otherwise is an empty papers page with
            // no indication that the manifest was the problem.
            tracing::error!(error = %e, "papers.yaml is not usable — the section will be empty");
            Manifest::default()
        }
    }
}

/// The left pane: every published paper, with the one being read marked.
///
/// Rendered on both routes so it survives navigation — that is the whole point
/// of the two panes. `current` is `None` on the index.
fn pane_index(m: &Manifest, current: Option<&str>) -> String {
    if m.papers.is_empty() {
        return "<p class=\"empty\">No papers are published yet.</p>".to_string();
    }
    let items = m
        .papers
        .iter()
        .map(|p| {
            let here = current == Some(p.slug.as_str());
            format!(
                "<li><a href=\"/papers/{slug}\"{aria} class=\"{cls}\">\
                   <span class=\"t\">{title}</span>\
                   <span class=\"m\">{meta}</span>\
                 </a></li>",
                slug = page::esc(&p.slug),
                aria = if here { " aria-current=\"page\"" } else { "" },
                cls = if here { "is-current" } else { "" },
                title = page::esc(&p.title),
                meta = page::esc(
                    &[p.date.as_str(), p.status.as_str()]
                        .iter()
                        .filter(|s| !s.is_empty())
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" · ")
                ),
            )
        })
        .collect::<String>();

    format!("<p class=\"pane-title\">Papers</p><ul class=\"pane-list\">{items}</ul>")
}

/// `/papers` opens the paper at the top of the manifest.
///
/// A selection screen is a page whose only content is a decision, and the
/// decision is nearly always "the newest one". Redirecting rather than
/// rendering the paper here keeps one canonical URL per paper: the same
/// document does not exist at two addresses, and the index marks the right
/// entry without special-casing.
///
/// **302, never 301.** The target changes the moment a paper is added to the
/// top of the manifest, and a permanent redirect would be cached against that.
pub async fn index(state: Arc<State>) -> Response {
    let m = manifest(&state).await;
    match m.papers.first() {
        Some(top) => Redirect::temporary(&format!("/papers/{}", top.slug)).into_response(),
        None => super::not_found(Nav::Papers, "No papers are published yet."),
    }
}

pub async fn show(state: Arc<State>, slug: &str) -> Response {
    let m = manifest(&state).await;
    let Some(paper) = m.papers.iter().find(|p| p.slug == slug) else {
        return super::not_found(Nav::Papers, "No paper by that name.");
    };
    let Some(dir) = state.papers_dir.clone() else {
        return super::oops(
            Nav::Papers,
            "This site has no papers directory configured, so its papers cannot be read.",
        );
    };

    let path: PathBuf = dir.join(&paper.source);
    let doc = match state.cache.get(&path, |s| s.to_string()).await {
        Ok(d) => d,
        Err(e) => {
            tracing::error!(error = %e, path = %path.display(), "paper source unreadable");
            return super::oops(
                Nav::Papers,
                "That paper is listed but its source document could not be read.",
            );
        }
    };

    // The manifest's title wins over the document's own `# H1`: the document is
    // still a working file and its heading can be revised without anyone
    // meaning to rename a published paper.
    let byline = [paper.authors.as_str(), paper.date.as_str()]
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" · ");
    let meta = Meta {
        title: &paper.title,
        heading: &paper.title,
        subtitle: (!paper.subtitle.is_empty()).then_some(paper.subtitle.as_str()),
        byline: (!byline.is_empty()).then_some(byline.as_str()),
        description: &paper.summary,
        nav: Nav::Papers,
        width: Width::Split,
    };

    Html(page::split(
        &meta,
        &pane_index(&m, Some(slug)),
        &format!(
            "{title}{sheet}<article class=\"prose paper\">{body}</article>{cite}\
             <script type=\"module\" src=\"/lib/paper.js\"></script>",
            title = page::title_block(&meta),
            sheet = toolbar(paper),
            body = doc.html,
            cite = cite_block(paper),
        ),
    ))
    .into_response()
}

/// Plain-text citation, which is what a reader pastes. Deliberately not BibTeX:
/// the button has to produce something useful in an email and a slide as well
/// as a `.bib`, and a DOI carries enough for a manager to recover the rest.
fn citation(p: &Paper) -> String {
    let year = p.published.get(..4).unwrap_or("");
    let title = if p.subtitle.is_empty() {
        p.title.clone()
    } else {
        format!("{}: {}", p.title, p.subtitle)
    };
    let mut s = String::new();
    if !p.authors.is_empty() {
        s.push_str(&p.authors);
        s.push_str(". ");
    }
    if !year.is_empty() {
        s.push_str(&format!("({year}). "));
    }
    s.push_str(&title);
    s.push_str(". Zenodo.");
    if !p.doi.is_empty() {
        s.push_str(&format!(" https://doi.org/{}", p.doi));
    }
    s
}

/// Read / print / download, above the document.
///
/// `print` and `copy` need script; `download` is a plain link, so the one
/// control that matters most to somebody archiving the paper still works with
/// script disabled.
fn toolbar(p: &Paper) -> String {
    let mut s = String::from("<div class=\"paper-tools\" data-citation=\"");
    s.push_str(&page::esc(&citation(p)));
    s.push_str("\">");
    s.push_str("<button type=\"button\" class=\"pt\" data-act=\"copy\">Copy citation</button>");
    s.push_str("<button type=\"button\" class=\"pt\" data-act=\"print\">Print</button>");
    if !p.pdf.is_empty() {
        s.push_str(&format!(
            "<a class=\"pt\" href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">Download PDF</a>",
            page::esc(&p.pdf)
        ));
    }
    if !p.doi.is_empty() {
        s.push_str(&format!(
            "<span class=\"pt-doi\">DOI <a href=\"https://doi.org/{doi}\" target=\"_blank\" rel=\"noopener noreferrer\">{doi}</a></span>",
            doi = page::esc(&p.doi)
        ));
    }
    s.push_str("</div>");
    s
}

/// The citation panel under the document.
///
/// It names the archived deposit rather than this page, and says so. The
/// rendering here follows the working document and will drift; the DOI resolves
/// to a fixed file with a third-party timestamp, and that is the thing worth
/// citing.
fn cite_block(p: &Paper) -> String {
    if p.doi.is_empty() && p.zenodo.is_empty() {
        return String::new();
    }
    let mut s = String::from("<section class=\"cite\"><h2>Cite this paper</h2>");
    s.push_str(&format!(
        "<p class=\"cite-text\">{}</p>",
        page::esc(&citation(p))
    ));
    s.push_str("<p class=\"cite-note\">");
    if !p.published.is_empty() {
        s.push_str(&format!(
            "Archived <time datetime=\"{iso}\">{shown}</time>. ",
            iso = page::esc(&p.published),
            shown = page::esc(if p.date.is_empty() {
                &p.published
            } else {
                &p.date
            }),
        ));
    }
    s.push_str(
        "The deposited PDF is the citable version of record. This page renders \
         the working document, which continues to move.",
    );
    if !p.zenodo.is_empty() {
        s.push_str(&format!(
            " <a href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">View the record</a>.",
            page::esc(&p.zenodo)
        ));
    }
    s.push_str("</p></section>");
    s
}

pub fn bad_request(detail: &str) -> Response {
    (StatusCode::BAD_REQUEST, detail.to_string()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_parses() {
        let m = Manifest::parse(
            "papers:\n  - slug: one-card\n    title: One Card, One Stack\n    source: unbounded_agents.md\n",
        )
        .unwrap();
        assert_eq!(m.papers.len(), 1);
        assert_eq!(m.papers[0].slug, "one-card");
        assert_eq!(m.papers[0].source, "unbounded_agents.md");
    }

    #[test]
    fn a_source_cannot_escape_the_papers_directory() {
        // The manifest is the only thing standing between `papers: ../docs` and
        // the whole filesystem, so this is the check that matters most here.
        for bad in ["../secrets.md", "sub/dir.md", "..\\win.md"] {
            let y = format!("papers:\n  - slug: s\n    title: t\n    source: {bad}\n");
            assert!(Manifest::parse(&y).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn an_unknown_key_is_an_error_rather_than_a_silent_default() {
        assert!(Manifest::parse(
            "papers:\n  - slug: s\n    title: t\n    source: a.md\n    athors: x\n"
        )
        .is_err());
    }

    #[test]
    fn an_empty_manifest_is_valid() {
        assert_eq!(Manifest::parse("papers: []\n").unwrap().papers.len(), 0);
    }
}
