//! Host → site → (local API | proxy | files), and the builder that lets a
//! daemon embed all of it.
//!
//! The same code serves both roles. What differs is one line of config:
//!
//! ```yaml
//! api:
//!   - {prefix: /v1, upstream: local}                 # authoritative
//!   - {prefix: /v1, upstream: "http://10.0.0.9:8081"} # proxy
//! ```
//!
//! `local` dispatches into an `axum::Router` the host handed us at build time —
//! which is how `npcd` and `zend` become single testable binaries that serve
//! their own console *and* their own API, while the DMZ box runs the identical
//! crate with URLs instead.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tower::Service;

use crate::auth::Auth;
use crate::config::{Config, Site, Upstream};
use crate::content::{self, Roots};
use crate::errors::{self, Problem};
use crate::health::Health;
use crate::proxy::{self, HttpClient};

#[derive(Clone)]
struct App {
    cfg: Arc<Config>,
    client: HttpClient,
    health: Health,
    /// Sign-in, if this deployment has any. Shared by every site: one account
    /// across the estate is the whole point of the gateway owning it.
    auth: Option<Arc<Auth>>,
    /// Per-site content roots, keyed by site name.
    roots: Arc<HashMap<String, Roots>>,
    /// Per-site in-process APIs, for routes whose upstream is `local`.
    ///
    /// Keyed by site rather than global because one process can host several:
    /// `web --authoritative` serves the npcd console's API and would serve
    /// zend's beside it, and a single shared router would answer whichever
    /// site's path it happened to recognise.
    local: Arc<HashMap<String, Router>>,
}

/// Builds a server that is either authoritative, a proxy, or both.
pub struct Builder {
    cfg: Config,
    roots: HashMap<String, Roots>,
    local: HashMap<String, Router>,
    auth: Option<Arc<Auth>>,
}

impl Builder {
    pub fn new(cfg: Config) -> Self {
        // Default every site to its configured disk roots. `content()` replaces
        // one with embedded assets when a daemon ships them inside the binary.
        let roots = cfg
            .sites
            .iter()
            .map(|s| (s.name.clone(), Roots::disk(&s.roots_abs)))
            .collect();
        let mut b = Self {
            cfg,
            roots,
            local: HashMap::new(),
            auth: None,
        };

        // Sites this crate is itself the backend for — tokera.com is documents,
        // not a daemon. Wired here rather than by every caller, so `web`, a
        // test and an embedded daemon all get the same site.
        let built_in: Vec<(String, Router)> = b
            .cfg
            .sites
            .iter()
            .filter_map(|s| {
                let roots = b.roots.get(&s.name)?.clone();
                crate::site::for_site(s, roots).map(|r| (s.name.clone(), r))
            })
            .collect();
        for (name, router) in built_in {
            b.local.insert(name, router);
        }
        b
    }

    /// Enable sign-in from the config's `auth:` block.
    ///
    /// Separate from [`new`](Self::new) because it reads secrets off disk and
    /// can fail, and a builder that panics on a missing key file would be a
    /// poor way to find out.
    pub fn with_auth(mut self) -> anyhow::Result<Self> {
        if let Some(cfg) = self.cfg.auth.clone() {
            self.auth = Some(Arc::new(Auth::new(cfg)?));
        }
        Ok(self)
    }

    /// Serve this site's files from embedded directories instead of disk.
    /// Order matters: `[site, common]` searches the site first.
    pub fn content(mut self, site: &str, roots: Roots) -> Self {
        self.roots.insert(site.to_string(), roots);
        self
    }

    /// The configured site names, in order — so a caller can ask what to
    /// supply before it has given anything up to the builder.
    pub fn sites(&self) -> impl Iterator<Item = &str> {
        self.cfg.sites.iter().map(|s| s.name.as_str())
    }

    /// The API answering this site's routes whose upstream is `local`.
    pub fn local_api(mut self, site: &str, router: Router) -> Self {
        self.local.insert(site.to_string(), router);
        self
    }

    /// Answer every API route from this process instead of forwarding.
    ///
    /// Rewrites each `upstream: <url>` to `local`, which is what
    /// `--authoritative` does. A site with no [`local_api`](Self::local_api)
    /// says so plainly per request rather than silently serving nothing.
    pub fn authoritative(mut self) -> Self {
        for site in &mut self.cfg.sites {
            for route in &mut site.api {
                route.upstream = Upstream::Local;
            }
        }
        self
    }

    pub fn router(self) -> Router {
        let health = Health::new(self.cfg.server.backoff);
        let client = proxy::client(Duration::from_millis(self.cfg.server.connect_timeout_ms));
        // Ahead of the fallback, so `/auth/*` is reached on every hostname —
        // including a site that proxies `/` and would otherwise swallow it.
        // Sign-in belongs to the estate, not to whichever daemon a name points
        // at, and a subsite must be able to ask who you are.
        let auth_routes = match &self.auth {
            Some(a) => a.clone().router(),
            None => crate::auth::unconfigured_router(),
        };
        let app = App {
            cfg: Arc::new(self.cfg),
            client,
            health,
            auth: self.auth,
            roots: Arc::new(self.roots),
            local: Arc::new(self.local),
        };
        // `/auth/*` are explicit routes and the site handler is the fallback,
        // so the merge order does not decide which wins — axum prefers the
        // explicit match either way.
        Router::new()
            .fallback(handle)
            .with_state(app)
            .merge(auth_routes)
    }

    pub async fn serve(self) -> anyhow::Result<()> {
        let addr = self.cfg.server.bind;
        announce(&self.cfg, &self.roots, &self.local);
        let router = self.router();
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("web listening on http://{addr}");
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
        Ok(())
    }
}

fn announce(cfg: &Config, roots: &HashMap<String, Roots>, local: &HashMap<String, Router>) {
    for site in &cfg.sites {
        let hosts = if site.hosts.is_empty() {
            "*".to_string()
        } else {
            site.hosts.join(", ")
        };
        let r = roots.get(&site.name);
        let where_ = match r {
            Some(rs) if !rs.is_empty() => format!("{:?}", rs.0),
            _ => "(no content)".into(),
        };
        tracing::info!(
            "site {:<14} {:<40} {}{}",
            site.name,
            hosts,
            where_,
            if site.default { "  [default]" } else { "" }
        );
        let has_local = local.contains_key(&site.name);
        for route in &site.api {
            match &route.upstream {
                Upstream::Local if has_local => {
                    tracing::info!("      {:<10} → local API", route.prefix)
                }
                Upstream::Local => tracing::warn!(
                    "      {:<10} → local, but site `{}` has no in-process API — these will 502",
                    route.prefix,
                    site.name
                ),
                Upstream::Url(u) => tracing::info!("      {:<10} → {u}", route.prefix),
            }
        }
    }
}

async fn handle(
    State(app): State<App>,
    peer: Option<ConnectInfo<SocketAddr>>,
    mut req: Request,
) -> Response {
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let site = app.cfg.site_for(host.as_deref());
    let path = req.uri().path().to_owned();

    // **Strip inbound identity headers here, before anything is dispatched.**
    //
    // `x-tokera-*` is how this gateway tells a daemon who the caller is, so a
    // client that sets one is asserting an identity. `proxy::forward` clears
    // them before every proxied request — but a route with `upstream: local`
    // never reaches it, and hands the daemon's own router the client's headers
    // untouched. `zend` and `npcd` embed their APIs exactly that way, so a
    // handler written against the documented header contract accepts
    // `x-tokera-email: admin@tokera.com` from anyone when run in-process, while
    // the identical code is safe behind the proxy.
    //
    // Doing it on ingress rather than in the local arm is what stops the two
    // dispatch paths from disagreeing again the next time one is added. The
    // strip in `proxy::forward` stays: it is on the other side of a `.await`
    // from here, and cheap.
    for name in proxy::IDENTITY_HEADERS {
        req.headers_mut().remove(name);
    }

    // Resolve who this is once, before anything downstream can see the request.
    // A local router reads it from the extensions; a proxied daemon reads it
    // from the headers the proxy sets. Both get the same answer from the same
    // check — the session cookie, which the strip above does not touch.
    let identity = app.auth.as_ref().and_then(|a| a.identity(req.headers()));
    let assertion = app
        .auth
        .as_ref()
        .and_then(|a| a.raw_token(req.headers()))
        .map(str::to_owned);
    if let Some(id) = identity.clone() {
        req.extensions_mut().insert(id);
    }

    if let Some(route) = site.route_for(&path) {
        let want_html = errors::wants_html(&path, req.headers());
        return match &route.upstream {
            Upstream::Local => match app.local.get(&site.name).cloned() {
                Some(mut r) => match r.call(req).await {
                    Ok(res) => res.into_response(),
                    // The router's error type is Infallible, so this arm is
                    // unreachable in practice; it exists so the match is total.
                    Err(_) => errors::respond(
                        Problem::upstream_down("the local API panicked", None),
                        want_html,
                    ),
                },
                None => errors::respond(
                    Problem::upstream_down(
                        format!(
                            "this route is `upstream: local`, but site `{}` has no API in this \
                             process — point the route at a URL, or run a build that carries \
                             one for this site",
                            site.name
                        ),
                        None,
                    ),
                    want_html,
                ),
            },
            Upstream::Url(url) => {
                proxy::forward(
                    proxy::Forward {
                        client: &app.client,
                        health: &app.health,
                        upstream: url,
                        rewrite_host: route.rewrite_host,
                        peer: peer.map(|ConnectInfo(a)| a.ip()),
                        want_html,
                        identity: identity.as_ref(),
                        assertion: assertion.as_deref(),
                        secure: app.auth.as_ref().is_some_and(|a| a.is_secure()),
                    },
                    req,
                )
                .await
            }
        };
    }

    serve_files(&app, site, &path, req.headers()).await
}

/// Resolution order for `/some/path`:
///   1. the file
///   2. `path.html` — pretty URLs for free
///   3. `path/index.html`
///   4. the site fallback, but ONLY for an extensionless path.
///
/// Rule 4 lets a client-side router own navigation while a deep link survives a
/// refresh. It deliberately excludes anything with an extension: a missing
/// `.js` must 404 loudly rather than quietly returning HTML, which otherwise
/// surfaces as a baffling `Unexpected token '<'`.
async fn serve_files(app: &App, site: &Site, path: &str, headers: &header::HeaderMap) -> Response {
    let want_html = errors::wants_html(path, headers);
    let Some(roots) = app.roots.get(&site.name) else {
        return errors::respond(
            Problem::not_found(format!("site `{}` has no content", site.name)),
            want_html,
        );
    };
    if roots.is_empty() {
        return errors::respond(
            Problem::not_found(format!(
                "site `{}` is a gateway and serves no files",
                site.name
            )),
            want_html,
        );
    }

    let Some(rel) = content::safe_rel(path) else {
        return errors::respond(
            Problem::bad_request("path escapes the site root"),
            want_html,
        );
    };

    let ext = content::has_extension(&rel);
    let mut tried: Vec<String> = Vec::with_capacity(4);
    if rel.is_empty() {
        tried.push("index.html".into());
    } else {
        tried.push(rel.clone());
        if !ext {
            tried.push(format!("{rel}.html"));
            tried.push(format!("{}/index.html", rel.trim_end_matches('/')));
        }
    }
    if !ext {
        tried.push(site.fallback.clone());
    }

    for candidate in &tried {
        if let Some(bytes) = roots.read(candidate).await {
            return file(candidate, bytes, &app.cfg.server.cache_control);
        }
    }

    errors::respond(
        Problem::not_found(format!("no such file in site `{}`", site.name)),
        want_html,
    )
}

fn file(name: &str, bytes: Vec<u8>, cache_control: &str) -> Response {
    let mime = mime_guess::from_path(name).first_or_octet_stream();
    let mut res = Response::new(Body::from(bytes));
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(mime.as_ref()) {
        h.insert(header::CONTENT_TYPE, v);
    }
    if let Ok(v) = HeaderValue::from_str(cache_control) {
        h.insert(header::CACHE_CONTROL, v);
    }
    *res.status_mut() = StatusCode::OK;
    res
}
