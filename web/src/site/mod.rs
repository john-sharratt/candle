//! Sites whose backend is `web` itself.
//!
//! Most sites here are a front for a daemon: the files are served and `/v1` is
//! forwarded. tokera.com is the exception — it is a site made of documents, and
//! standing up a service to read markdown off a disk would be a process to
//! deploy and monitor in exchange for nothing.
//!
//! The distinction to keep: a site belongs here only while it has no state and
//! no engine. The moment one needs either, it becomes a daemon and this crate
//! goes back to forwarding — which is a config edit, not a rewrite.

use axum::Router;
use std::path::PathBuf;

use crate::config::Site;
use crate::content::Roots;

pub mod tokera;

/// The in-process router for a site this crate serves itself, if it has one.
///
/// The single place that maps a site name to its own backend, so `--check` and
/// the running server can never disagree about which sites have one.
pub fn for_site(site: &Site, roots: Roots) -> Option<Router> {
    match site.name.as_str() {
        "tokera" => Some(tokera::router(roots, site.papers_abs.clone())),
        _ => None,
    }
}

/// Whether a site is served from this crate — for the startup summary, without
/// building the router.
pub fn built_in(name: &str) -> bool {
    matches!(name, "tokera")
}

/// The papers directory a site would use, for reporting.
pub fn papers_dir(site: &Site) -> Option<&PathBuf> {
    site.papers_abs.as_ref()
}
