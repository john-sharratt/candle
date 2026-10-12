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
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tower::Service;

use axum::middleware;
use cluster::gateway::Class;

use crate::acme::names::groups;
use crate::acme::peers::AcmePeers;
use crate::acme::{Challenges, Manager};
use crate::asset;
use crate::auth::forwarded::identify;
use crate::auth::{Auth, Gateways};
use crate::cluster::peers::dial_all;
use crate::cluster::route::{self, Routed};
use crate::cluster::{endpoint, page, ClusterPeer, Gateway, MemberOrigin};
use crate::compress;
use crate::config::{Config, Site, Upstream};
use crate::content::{self, Roots};
use crate::errors::{self, Problem};
use crate::health::Health;
use crate::proxy::{self, To};
use crate::tls::{Entrance, Resolver, SecureEntrance};
use crate::upstream_tls::Clients;

#[derive(Clone)]
struct App {
    cfg: Arc<Config>,
    clients: Clients,
    health: Health,
    /// Sign-in, if this deployment has any. Shared by every site: one account
    /// across the estate is the whole point of the gateway owning it.
    auth: Option<Arc<Auth>>,
    /// The peers whose inbound `x-tokera-*` headers are believed.
    ///
    /// Set by [`Builder::trust`], and the difference between reading those
    /// headers and clearing them. `None` clears them from everyone. Declared
    /// rather than inferred: "has no `auth:` block" would look like the same
    /// thing and is not — an authoritative instance without sign-in is still an
    /// entrance a client can reach, and would then believe whatever it was told.
    trusted: Option<Gateways>,
    /// Per-site content roots, keyed by site name.
    roots: Arc<HashMap<String, Roots>>,
    /// Per-site in-process APIs, for routes whose upstream is `local`.
    ///
    /// Keyed by site rather than global because one process can host several:
    /// `web --authoritative` serves the npcd console's API and would serve
    /// zend's beside it, and a single shared router would answer whichever
    /// site's path it happened to recognise.
    local: Arc<HashMap<String, Router>>,
    /// The cluster this gateway runs, when its config has a `cluster:` block.
    cluster: Option<Arc<Gateway>>,
}

/// Builds a server that is either authoritative, a proxy, or both.
pub struct Builder {
    cfg: Config,
    roots: HashMap<String, Roots>,
    local: HashMap<String, Router>,
    auth: Option<Arc<Auth>>,
    trusted: Option<Gateways>,
    /// HTTP-01 answers, set by the ACME manager and served on every hostname.
    /// Empty — every token a 404 — where there is no TLS entrance.
    challenges: Challenges,
    cluster: Option<Arc<Gateway>>,
    /// Applied to the finished router, outermost: a daemon's own listener
    /// rules (`host::cluster::listener`).
    wrap: Option<Wrap>,
}

/// A layer a daemon puts around the whole router.
pub type Wrap = Box<dyn FnOnce(Router) -> Router + Send>;

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
            trusted: None,
            challenges: Challenges::default(),
            cluster: None,
            wrap: None,
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
            self.auth = Some(Arc::new(Auth::new(cfg, self.cfg.roles.clone())?));
        }
        Ok(self)
    }

    /// Run the cluster the config's `cluster:` block names: read its group key
    /// and take member sockets at `/v1/cluster/:app`
    /// (`docs/replicated_substrate.md` §3). Fallible for the same reason as
    /// [`with_auth`](Self::with_auth): a key file that does not load stops the
    /// start rather than leaving a gateway no member can join.
    pub fn with_cluster(mut self) -> anyhow::Result<Self> {
        if self.cfg.cluster.is_some() {
            self.cluster = Some(Gateway::new(&self.cfg)?);
        }
        Ok(self)
    }

    /// Run `gateway` as this server's cluster.
    pub fn cluster(mut self, gateway: Arc<Gateway>) -> Self {
        self.cluster = Some(gateway);
        self
    }

    /// Put `wrap` around the finished router, outside everything else.
    pub fn wrap(mut self, wrap: Wrap) -> Self {
        self.wrap = Some(wrap);
        self
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

    /// Believe the identity `gateways` forward on `x-tokera-*`, and clear it
    /// from every other peer.
    ///
    /// This is how a daemon behind the gateway learns who is calling. Identity
    /// crosses a process boundary as headers and an in-process boundary as a
    /// request extension; a daemon on another box only ever gets the former, so
    /// a server that clears them can never identify anybody. Without this call
    /// they are cleared on ingress from everyone, because a client that sets one
    /// is claiming to be someone.
    ///
    /// The check is the socket peer's address, so a machine on the LAN that
    /// connects directly is anonymous whatever it sends. Loopback is always
    /// among the trusted ([`Gateways`]), so anything on this box that relays
    /// remote traffic onto loopback — a tunnel, a proxy pointed at
    /// `127.0.0.1` — is believed as this machine would be, and must clear the
    /// headers itself.
    pub fn trust(mut self, gateways: Gateways) -> Self {
        self.trusted = Some(gateways);
        self
    }

    /// The HTTP-01 answers this server publishes — the map the ACME manager
    /// writes and the challenge route reads.
    pub fn challenges(&self) -> Challenges {
        self.challenges.clone()
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
        let clients = Clients::new(Duration::from_millis(self.cfg.server.connect_timeout_ms));
        // Ahead of the fallback, so `/auth/*` is reached on every hostname —
        // including a site that proxies `/` and would otherwise swallow it.
        // Sign-in belongs to the estate, not to whichever daemon a name points
        // at, and a subsite must be able to ask who you are.
        let auth_routes = match &self.auth {
            Some(a) => a.clone().router(),
            None => crate::auth::unconfigured_router(),
        };
        // The same reasoning for the CA's challenges: asked for on every name,
        // including one that redirects or proxies `/`, and answered by the
        // gateway rather than by whichever site the name belongs to.
        let challenge_routes = self.challenges.router();
        // The cluster's socket and status page, on every host, like `/auth/*`.
        let cluster_routes = match &self.cluster {
            Some(gw) => endpoint::router(gw.clone()).merge(page::router(
                gw.clone(),
                self.auth.clone(),
                self.cfg.roles.clone(),
            )),
            None => Router::new(),
        };
        let cluster = self.cluster.clone();
        let wrap = self.wrap;
        let app = App {
            cfg: Arc::new(self.cfg),
            clients,
            health,
            auth: self.auth,
            trusted: self.trusted,
            roots: Arc::new(self.roots),
            local: Arc::new(self.local),
            cluster: self.cluster,
        };
        // `/auth/*` are explicit routes and the site handler is the fallback,
        // so the merge order does not decide which wins — axum prefers the
        // explicit match either way.
        let router = Router::new()
            .fallback(handle)
            .with_state(app)
            .merge(auth_routes)
            .merge(challenge_routes)
            .merge(cluster_routes)
            // Outermost, so it sees every answer — files, proxied and local
            // APIs, sign-in, error pages — and compresses whatever arrived
            // uncompressed. See `compress`.
            .layer(middleware::from_fn(compress::layer));
        let router = match wrap {
            Some(wrap) => wrap(router),
            None => router,
        };
        // Requests members forward up are served by this same router.
        if let Some(gw) = cluster {
            gw.set_router(router.clone());
        }
        router
    }

    /// Serve until Ctrl-C, with the TLS entrance the config's `tls:` asks for —
    /// certificates from Let's Encrypt. The gateway's way to run.
    pub async fn serve(self) -> anyhow::Result<()> {
        let tls = self.cfg.server.tls.clone();
        let groups = groups(&self.cfg);
        let challenges = self.challenges();
        let peers = self
            .cfg
            .cluster
            .as_ref()
            .map(|c| c.peers.clone())
            .unwrap_or_default();
        let cluster = self.cluster.clone();
        // The plain entrance first: it answers the CA's HTTP-01 challenges, so
        // it is listening before the manager can ask for a certificate.
        let bound = self.bind().await?;
        // Bound before anything is served, so a store that does not load or a
        // port that is taken stops the start rather than leaving the plain
        // entrance up without it. The stored certificates are installed first,
        // so a restart serves what it has without waiting on the CA.
        let entrance = match tls {
            Some(tls) => {
                let resolver = Resolver::new();
                let mut manager =
                    Manager::new(tls.acme.clone(), groups, challenges, resolver.clone());
                manager.install_stored()?;
                // With peers, the ACME they share is ready before any link is
                // taken or dialled (§13.4).
                if let Some((gw, peering)) = cluster
                    .as_ref()
                    .and_then(|gw| gw.peering.clone().map(|p| (gw, p)))
                {
                    let acme = AcmePeers::new(peering.clone(), manager.certs());
                    gw.set_acme(acme.clone());
                    manager = manager.with_peers(acme);
                    dial_all(&peering, &peers)?;
                }
                let entrance = Entrance::bind(tls.bind, resolver).await?;
                tokio::spawn(manager.run());
                Some(entrance)
            }
            None => None,
        };
        bound
            .serve_until(entrance, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
    }

    /// Bind the plain entrance on `server.bind`. Nothing is served until
    /// [`Bound::serve_until`].
    pub async fn bind(self) -> anyhow::Result<Bound> {
        let addr = self.cfg.server.bind;
        announce(&self.cfg, &self.roots, &self.local);
        let cluster = self.cluster.clone();
        let router = self.router();
        let listener = TcpListener::bind(addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        tracing::info!("web listening on http://{addr}");
        Ok(Bound {
            listener,
            router,
            cluster,
        })
    }
}

/// A server whose plain entrance is bound and not yet serving.
pub struct Bound {
    listener: TcpListener,
    router: Router,
    /// The cluster this gateway runs, whose members are told when it stops.
    cluster: Option<Arc<Gateway>>,
}

impl Bound {
    /// Serve the plain entrance, and `entrance` beside it when there is one,
    /// until `shutdown` resolves; then stop taking connections and return once
    /// every request in flight on either entrance has been answered.
    ///
    /// The entrance is bound by the caller, so a daemon that presents its own
    /// certificate ([`crate::tls::self_signed`]) and the gateway that presents
    /// Let's Encrypt's serve through the same code.
    pub async fn serve_until(
        self,
        entrance: Option<Entrance>,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> anyhow::Result<()> {
        let Bound {
            listener,
            router,
            cluster,
        } = self;
        // One signal for both entrances: the plain one drains on it, and the
        // TLS one is told through this channel.
        let (stopping_tx, mut stopping) = watch::channel(false);
        let tls_task = match entrance {
            Some(entrance) => {
                tracing::info!(
                    "web listening on https://{} — HTTP/3 over udp, HTTP/2 and HTTP/1.1 over tcp",
                    entrance.local_addr()?
                );
                Some(tokio::spawn(entrance.serve(router.clone(), async move {
                    let _ = stopping.changed().await;
                })))
            }
            None => None,
        };
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        // **No Nagle on the client side.** A response body leaves in many small
        // writes; with Nagle on, each one waits for the ACK of the last, and the
        // client's delayed ACK holds that for up to ~200 ms. Over Cloudflare's
        // round trip that capped a 150 KB response at ~20 KB/s — the GUI's page
        // and tool catalog took 7–8 s to load while the LAN path took 2 ms.
        .tcp_nodelay(true)
        .with_graceful_shutdown(async move {
            shutdown.await;
            tracing::info!("shutting down");
            // A member's socket is an upgraded connection the drain does not
            // wait for: each is told, so it carries on through its other
            // clusters rather than finding the socket gone.
            if let Some(gw) = &cluster {
                gw.hub.close_all("the gateway is shutting down");
                if let Some(p) = &gw.peering {
                    p.close_all("the gateway is shutting down");
                }
            }
            let _ = stopping_tx.send(true);
        })
        .await?;
        // The TLS entrance drains on the same signal; its requests in flight
        // are answered before this returns.
        if let Some(task) = tls_task {
            task.await??;
        }
        Ok(())
    }
}

/// One permanent redirect, carrying the path and query across.
///
/// `301`, because these are alternate names for one brand and that is the
/// status that tells a search engine to move the address rather than note a
/// detour. It is worth knowing that browsers cache a `301` hard and for a long
/// time — a host redirected by mistake stays redirected in every browser that
/// saw it, so this is a decision to make deliberately and rarely.
///
/// The tail is taken from the request line, which is a client's to choose, so
/// the `Location` is built through `HeaderValue::from_str` — it rejects the
/// control characters that would otherwise let a crafted path inject a second
/// header. A tail that will not go in a header is dropped and the redirect
/// lands on the target's root, which is the safe direction to be wrong in.
fn redirect(to: &str, req: &Request) -> Response {
    let tail = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let location = format!("{}{tail}", to.trim_end_matches('/'));
    let mut res = Response::new(Body::empty());
    *res.status_mut() = StatusCode::MOVED_PERMANENTLY;
    let value = HeaderValue::from_str(&location)
        .or_else(|_| HeaderValue::from_str(to.trim_end_matches('/')))
        .ok();
    if let Some(v) = value {
        res.headers_mut().insert(header::LOCATION, v);
    }
    // A redirect that never changes may be kept, but not for ever: these are
    // the one thing here somebody might need to undo, and a browser holding a
    // permanent redirect it can no longer re-check is how that becomes hard.
    res.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=900"),
    );
    res
}

fn announce(cfg: &Config, roots: &HashMap<String, Roots>, local: &HashMap<String, Router>) {
    for site in &cfg.sites {
        let hosts = if site.hosts.is_empty() {
            "*".to_string()
        } else {
            site.hosts.join(", ")
        };
        let r = roots.get(&site.name);
        let where_ = match (&site.redirect, r) {
            // Said plainly, because a redirected host silently serving nothing
            // looks identical to a misconfigured one in this table.
            (Some(to), _) => format!("→ {to} (301)"),
            (None, Some(rs)) if !rs.is_empty() => format!("{:?}", rs.0),
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
    // A request a member forwarded up is its app's, whatever `Host` it was
    // sent to on the member's own listener (§5.8).
    let origin = req.extensions().get::<MemberOrigin>().cloned();
    let site = origin
        .as_ref()
        .and_then(|o| app.cfg.site_for_app(&o.app))
        .unwrap_or_else(|| app.cfg.site_for(host.as_deref()));
    let path = req.uri().path().to_owned();

    /* A redirected host is answered before anything else happens to the
     * request.
     *
     * Before identity, before sign-in, before routing — none of it applies to a
     * name that serves nothing. It also means an alternate domain costs one
     * response and never touches a content root or an upstream, so adding one
     * cannot affect what the real site does. */
    if let Some(to) = &site.redirect {
        return redirect(to, &req);
    }

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
    //
    // The exception is a request from a peer the server declared it trusts
    // ([`Builder::trust`]): the gateway, whose headers are the only identity a
    // daemon behind it will ever receive, because identity crossed a network
    // hop to reach it. Stripping those would make the header contract
    // unimplementable rather than safe. Declared, never inferred — "has no
    // `auth:` block" is not evidence of it, and an authoritative instance
    // without sign-in is still an entrance.
    //
    // A request on a stream a member opened is believed too: the member
    // proved the group key, and vouches for whoever reached its own listener
    // by the same rules it would have served them by (§13.2).
    let believed = origin.is_some()
        || req.extensions().get::<ClusterPeer>().is_some()
        || match (&app.trusted, &peer) {
            (Some(gateways), Some(ConnectInfo(addr))) => gateways.trusts(addr.ip()),
            _ => false,
        };
    if !believed {
        for name in proxy::IDENTITY_HEADERS {
            req.headers_mut().remove(name);
        }
    }

    // Resolve who this is once, before anything downstream can see the request.
    // A local router reads it from the extensions; a proxied daemon reads it
    // from the headers the proxy sets. Both get the same answer from the same
    // check — the session cookie, which the strip above does not touch — or,
    // for a request a member forwarded up, the identity it carries.
    let identity = app
        .auth
        .as_ref()
        .and_then(|a| a.identity(req.headers()))
        .or_else(|| origin.as_ref().and_then(|_| identify(req.headers()).ok()));
    let assertion = app
        .auth
        .as_ref()
        .and_then(|a| a.raw_token(req.headers()))
        .map(str::to_owned);
    if let Some(id) = identity.clone() {
        req.extensions_mut().insert(id);
    }

    // A request over the TLS entrance is https by construction; anything else
    // is https only where the deployment says its public entrance is.
    let secure = req.extensions().get::<SecureEntrance>().is_some()
        || app.auth.as_ref().is_some_and(|a| a.is_secure());

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
                let peer_ip = peer.as_ref().map(|ConnectInfo(a)| a.ip());
                if let (Some(cluster_app), Some(gw)) = (&route.cluster, &app.cluster) {
                    let ctx = || route::Context {
                        want_html,
                        peer: peer_ip,
                        identity: identity.as_ref(),
                        secure,
                        cache: app.cfg.server.cache,
                        roles: &app.cfg.roles,
                        from_member: origin.as_ref().map(|o| o.node),
                    };
                    if gw.hub.clustered(cluster_app) {
                        match route::route(gw, cluster_app, ctx(), req).await {
                            Routed::Answered(res) => return res,
                            Routed::Dumb(back) => req = back,
                        }
                    } else if app.health.is_down(url) {
                        // Dumb, and the configured upstream is not answering: a
                        // member still loading serves its own loading page in
                        // place of the outage page (§5.7).
                        if let Some(node) = gw.hub.standing_by(cluster_app) {
                            return route::forward(gw, cluster_app, node, &ctx(), req).await;
                        }
                    }
                    gw.hub.traffic.request(Class::Configured);
                }
                proxy::forward(
                    proxy::Forward {
                        to: To::Url {
                            client: app.clients.for_route(route.self_signed),
                            health: &app.health,
                            url,
                        },
                        rewrite_host: route.rewrite_host,
                        peer: peer_ip,
                        want_html,
                        identity: identity.as_ref(),
                        assertion: assertion.as_deref(),
                        secure,
                        // Applied only where the upstream said nothing — a
                        // daemon that states its own policy keeps it.
                        cache: app.cfg.server.cache,
                        from_member: None,
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
            // The request's own headers decide the answer: what it already
            // holds, and what encodings it will take. See [`crate::asset`].
            return asset::respond(candidate, bytes, app.cfg.server.cache, headers);
        }
    }

    errors::respond(
        Problem::not_found(format!("no such file in site `{}`", site.name)),
        want_html,
    )
}

#[cfg(test)]
mod challenge_routing_tests {
    use super::*;
    use http_body_util::BodyExt;

    const YAML: &str = r#"
sites:
  - name: home
    default: true
    roots: ["."]
  - name: alt
    hosts: ["www.example.net"]
    redirect: "https://example.net"
  - name: proxied
    hosts: ["code.example.net"]
    api:
      - prefix: /
        upstream: "http://127.0.0.1:9"
"#;

    async fn ask(router: &Router, host: &str, path: &str) -> (StatusCode, String) {
        let req = Request::get(path)
            .header(header::HOST, host)
            .body(Body::empty())
            .unwrap();
        let res = router.clone().call(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    /// **The CA's challenge is the gateway's to answer, on every name.** A
    /// redirect site would otherwise answer `301` and a site proxying `/`
    /// would hand the path to a daemon that has never heard of the token.
    #[tokio::test]
    async fn the_challenge_is_answered_ahead_of_redirects_and_proxies() {
        let b = Builder::new(Config::from_yaml(YAML, std::path::Path::new(".")).unwrap());
        b.challenges().set("tok", "tok.thumb");
        let router = b.router();
        for host in ["example.net", "www.example.net", "code.example.net"] {
            assert_eq!(
                ask(&router, host, "/.well-known/acme-challenge/tok").await,
                (StatusCode::OK, "tok.thumb".to_string()),
                "{host}"
            );
        }
        // Everything else on the redirect site still redirects.
        assert_eq!(
            ask(&router, "www.example.net", "/other").await.0,
            StatusCode::MOVED_PERMANENTLY
        );
    }
}

#[cfg(test)]
mod redirect_tests {
    use super::*;

    fn to(uri: &str) -> String {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        redirect("https://tokera.com", &req)
            .headers()
            .get(header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string()
    }

    /// **A redirect keeps the address somebody typed.**
    ///
    /// Dropping the path sends every deep link to the front page, which for a
    /// consolidated domain means every inbound link to a paper or a post
    /// arrives nowhere in particular — and a search engine following one learns
    /// only that the old address is gone.
    #[test]
    fn the_path_and_query_come_across() {
        assert_eq!(to("/"), "https://tokera.com/");
        assert_eq!(to("/papers/o1"), "https://tokera.com/papers/o1");
        assert_eq!(
            to("/blog/x?utm=1&b=2"),
            "https://tokera.com/blog/x?utm=1&b=2"
        );
        assert_eq!(to("/a/b/c/"), "https://tokera.com/a/b/c/");
    }

    /// A target written with a trailing slash must not produce `//`.
    #[test]
    fn the_target_is_joined_without_doubling_the_slash() {
        let req = Request::builder().uri("/x").body(Body::empty()).unwrap();
        let res = redirect("https://tokera.com/", &req);
        assert_eq!(res.headers()[header::LOCATION], "https://tokera.com/x");
    }

    #[test]
    fn it_is_permanent() {
        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        assert_eq!(
            redirect("https://tokera.com", &req).status(),
            StatusCode::MOVED_PERMANENTLY
        );
    }

    /// **The tail is the client's to choose, so it must not be able to write a
    /// second header.**
    ///
    /// A `Location` carrying a carriage return is header injection. The URI
    /// parser rejects most of it before this is reached, and `from_str` is the
    /// backstop — anything it will not take falls back to the target's root,
    /// which is the safe direction to be wrong in.
    #[test]
    fn a_crafted_path_cannot_inject_a_header() {
        for raw in [
            "/x%0d%0aSet-Cookie:%20a=b",
            "/x%0aLocation:%20https://evil.example",
            "/x%00y",
        ] {
            let Ok(req) = Request::builder().uri(raw).body(Body::empty()) else {
                continue; // refused before it got here, which is also correct
            };
            let res = redirect("https://tokera.com", &req);
            let loc = res.headers()[header::LOCATION].to_str().unwrap();
            assert!(
                !loc.contains('\r') && !loc.contains('\n') && !loc.contains('\0'),
                "`{raw}` produced `{loc}`"
            );
            assert!(
                loc.starts_with("https://tokera.com"),
                "`{raw}` escaped the target"
            );
        }
    }
}
