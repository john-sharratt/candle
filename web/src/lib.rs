//! `web` — the front door, in two roles from one crate.
//!
//! * **Authoritative.** Embedded in a daemon: it serves that daemon's console
//!   from disk or from assets compiled into the binary, and answers `/v1` from
//!   an `axum::Router` merged into the same process. One binary, testable on
//!   its own, no second service to run.
//!
//! * **Proxy.** Standalone on a DMZ box: it owns whatever content it is given,
//!   and forwards the rest by hostname to whichever machine runs that daemon —
//!   with exponential backoff, automatic recovery, and an error page a person
//!   can read when a backend is down.
//!
//! The difference between the two is one line of YAML per route
//! (`upstream: local` versus a URL), so promoting a site from local to remote
//! is an edit rather than a migration.
//!
//! Either role can also hold the TLS entrance ([`tls`]): HTTP/3 over QUIC, with
//! HTTP/2 and HTTP/1.1 over TCP for a client that cannot reach UDP, presenting
//! certificates the gateway obtains and renews from Let's Encrypt ([`acme`]).
//!
//! There is a third way to run it, which is the first two combined: `web
//! --authoritative` forces every route local and answers from [`mock`], so the
//! whole console works over real sockets with no daemon anywhere.
//!
//! ```no_run
//! # use std::path::Path;
//! // Inside a daemon: serve the embedded console and answer /v1 in-process.
//! static SITE:   include_dir::Dir<'_> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/../web/content/npcd");
//! static COMMON: include_dir::Dir<'_> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/../web/content/common");
//!
//! # async fn run(api: axum::Router) -> anyhow::Result<()> {
//! let cfg = web::Config::from_yaml(include_str!("../web.yaml"), Path::new("."))?;
//! web::Builder::new(cfg)
//!     .content("npcd", web::Roots::embedded(&[&SITE, &COMMON]))
//!     .local_api("npcd", api)
//!     .serve()
//!     .await
//! # }
//! ```

pub mod acme;
pub mod asset;
pub mod auth;
pub mod compress;
pub mod config;
pub mod content;
pub mod errors;
pub mod health;
pub mod markdown;
pub mod mock;
pub mod proxy;
pub mod server;
pub mod site;
pub mod tls;
pub mod upstream_tls;

pub use config::{Backoff, Config, Route, Site, Tls, Upstream};
pub use content::{Roots, Source};
pub use health::Health;
pub use server::Builder;
