//! One HTTP server for every tenant (`docs/shared_model_host.md` §3.4).
//!
//! One `web::Builder` serves everything, resolved in this order:
//!
//! 1. the host's shared endpoints and every tenant's own API — one router, the
//!    same on every host name, answering `/v1`, `/ws`, `/common` and `/brand`;
//! 2. the web UIs, picked by `Host`: each tenant's UI is a site with its own
//!    names, and the default site answers a name no tenant claims — a bare IP.
//!
//! The router is served on a plain entrance and, beside it, a TLS one with a
//! certificate made at startup, for the hop from the gateway.

use std::env::current_dir;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

use axum::middleware;
use axum::Router;
use cluster::upgrade::switch::PROOF;
use serde_json::{json, Value};
use web::auth::Gateways;
use web::tls::{self_signed, Entrance};
use web::{Builder, Config, Roots};

use crate::access::LocalSignin;
use crate::cluster::listener::{self, Listener};
use crate::cluster::Cluster;
use crate::tenant::TenantUi;

/// The path prefixes the one router answers on every site: the API, the
/// sockets, and the shared front-end.
const API_PREFIXES: [&str; 4] = ["/v1", "/ws", "/common", "/brand"];

/// Where the host listens, and whom it believes.
#[derive(Debug, Clone)]
pub struct Listen {
    pub bind: IpAddr,
    pub port: u16,
    /// Also serve HTTPS — HTTP/3 over UDP, HTTP/2 and HTTP/1.1 over TCP — on
    /// this port of the same address, with a self-signed certificate.
    pub tls_port: Option<u16>,
    /// Peers whose `x-tokera-*` identity is believed, beside loopback and
    /// `bind`.
    pub gateways: Vec<IpAddr>,
}

/// The site table for `uis`: one site per tenant UI, every one answering the
/// API prefixes from this process. With one tenant its UI is the default
/// whatever it says; with several, exactly one must say so. The table names no
/// roots: each site's pages are compiled in, and [`builder`] hands them over.
pub fn site_table(
    uis: &[(&str, TenantUi)],
    bind: SocketAddr,
    base: &Path,
) -> anyhow::Result<Config> {
    let defaults = uis.iter().filter(|(_, ui)| ui.default).count();
    if uis.len() > 1 && defaults != 1 {
        anyhow::bail!(
            "{defaults} tenant UIs claim to be the default site; exactly one must, so a request \
             whose Host names no tenant has somewhere to go"
        );
    }
    let api: Vec<Value> = API_PREFIXES
        .iter()
        .map(|p| json!({"prefix": p, "upstream": "local"}))
        .collect();
    let sites: Vec<Value> = uis
        .iter()
        .map(|(name, ui)| {
            json!({
                "name": name,
                "hosts": ui.hosts,
                "default": uis.len() == 1 || ui.default,
                "fallback": "index.html",
                "api": api,
            })
        })
        .collect();
    let doc = json!({
        "server": {"bind": bind.to_string()},
        "sites": sites,
    });
    Config::from_yaml(&serde_yaml::to_string(&doc)?, base)
}

/// The server for `uis` answering `router`, as `listen` describes: the site
/// table, the trusted peers, and `--local-signin`. Paths resolve against the
/// working directory.
///
/// A member of `cluster` puts its own listener's rules around the whole
/// router (`cluster::listener`): forward up while joined, serve while cut off.
/// That router is also what serves the requests its gateways send.
pub fn builder(
    listen: &Listen,
    uis: &[(&str, TenantUi)],
    router: Router,
    local_signin: LocalSignin,
    cluster: Option<Arc<Cluster>>,
) -> anyhow::Result<Builder> {
    let addr = SocketAddr::new(listen.bind, listen.port);
    let cfg = site_table(uis, addr, &current_dir()?)?;
    let gateways = Gateways::new(listen.bind, &listen.gateways);
    tracing::info!(
        trusted = %gateways,
        "identity headers are believed only from these peers; every other caller is anonymous",
    );
    // `--local-signin` runs after the server has cleared untrusted identity
    // headers, so it writes onto a request that carries only what is believed.
    let signin = local_signin.clone();
    let router = router.layer(middleware::from_fn(move |req, next| {
        signin.clone().layer(req, next)
    }));
    let mut builder = Builder::new(cfg).trust(gateways.clone());
    for (name, ui) in uis {
        builder = builder
            .content(name, Roots::embedded(&[ui.content]))
            .local_api(name, router.clone());
    }
    if let Some(cluster) = cluster {
        let listener = Arc::new(Listener {
            cluster: cluster.clone(),
            sites: Arc::new(site_table(uis, addr, &current_dir()?)?),
            gateways,
            local_signin,
        });
        builder = builder.wrap(Box::new(move |router: Router| {
            let router = router.layer(middleware::from_fn_with_state(listener, listener::layer));
            cluster.set_router(router.clone());
            router
        }));
    }
    Ok(builder)
}

/// Serve `router` for `uis` until `shutdown` resolves, then return once every
/// request in flight is answered. `proof` prints the serving line on stderr
/// once the entrances are bound. The boot server passes `false`: it is up
/// while the model loads, and that is not serving.
pub async fn serve(
    listen: &Listen,
    uis: &[(&str, TenantUi)],
    router: Router,
    local_signin: LocalSignin,
    cluster: Option<Arc<Cluster>>,
    shutdown: impl Future<Output = ()> + Send + 'static,
    proof: bool,
) -> anyhow::Result<()> {
    if !listen.bind.is_loopback() {
        tracing::warn!(
            bind = %listen.bind,
            "binding a non-loopback address — reachable from the network; identity is still \
             believed only from the trusted peers",
        );
    }
    let bound = builder(listen, uis, router, local_signin, cluster)?
        .bind()
        .await?;
    let entrance = match listen.tls_port {
        Some(port) => {
            let tls_addr = SocketAddr::new(listen.bind, port);
            let names = vec![
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                listen.bind.to_string(),
            ];
            Some(Entrance::bind(tls_addr, self_signed(&names)?).await?)
        }
        None => None,
    };
    if proof {
        eprintln!("{PROOF}");
    }
    bound.serve_until(entrance, shutdown).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{to_bytes, Body};
    use axum::extract::ConnectInfo;
    use axum::http::{HeaderMap, Request};
    use axum::routing::get;
    use include_dir::{include_dir, Dir};
    use tower::ServiceExt;
    use web::auth::forwarded::{identify, EMAIL, PROVIDER, USER};

    static ZEND: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/tests/ui/zend");
    static NPCD: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/tests/ui/npcd");

    fn uis() -> Vec<(&'static str, TenantUi)> {
        let ui = |content: &'static Dir<'static>, hosts: &[&str], default: bool| TenantUi {
            hosts: hosts.iter().map(|h| h.to_string()).collect(),
            content,
            default,
        };
        vec![
            (
                "zend",
                ui(&ZEND, &["code.tokera.com", "zend.localhost"], true),
            ),
            (
                "npcd",
                ui(&NPCD, &["bot.tokera.com", "npcd.localhost"], false),
            ),
        ]
    }

    fn base() -> &'static Path {
        Path::new(".")
    }

    fn app(uis: &[(&str, TenantUi)]) -> Router {
        let cfg = site_table(uis, "127.0.0.1:0".parse().unwrap(), base()).unwrap();
        let api = Router::new().route("/v1/status", get(|| async { "status" }));
        let mut b = Builder::new(cfg);
        for (name, ui) in uis {
            b = b
                .content(name, Roots::embedded(&[ui.content]))
                .local_api(name, api.clone());
        }
        b.router()
    }

    async fn ask(app: &Router, host: &str, path: &str) -> (u16, String) {
        let res = app
            .clone()
            .oneshot(
                Request::get(path)
                    .header("host", host)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status().as_u16();
        let body = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    /// `/` is zend's UI for `code.tokera.com` and a bare IP, npcd's for
    /// `bot.tokera.com` — and so is every same-named file; the API is the same
    /// on every name.
    #[tokio::test]
    async fn the_host_name_picks_the_ui_and_the_api_is_everywhere() {
        let app = app(&uis());
        for host in ["code.tokera.com", "zend.localhost", "192.168.0.5:8081"] {
            assert_eq!(
                ask(&app, host, "/").await,
                (200, "<p>zend</p>".into()),
                "{host}"
            );
            assert_eq!(
                ask(&app, host, "/favicon.svg").await.1,
                "<svg>zend</svg>",
                "{host}"
            );
        }
        for host in ["bot.tokera.com", "npcd.localhost:8081"] {
            assert_eq!(
                ask(&app, host, "/").await,
                (200, "<p>npcd</p>".into()),
                "{host}"
            );
            assert_eq!(
                ask(&app, host, "/favicon.svg").await.1,
                "<svg>npcd</svg>",
                "{host}"
            );
        }
        for host in ["code.tokera.com", "bot.tokera.com", "10.0.0.1"] {
            assert_eq!(
                ask(&app, host, "/v1/status").await,
                (200, "status".into()),
                "{host}"
            );
        }
    }

    /// A deep link no file answers is the UI's `index.html` — the compiled-in
    /// pages take the site's fallback as the disk ones did.
    #[tokio::test]
    async fn a_deep_link_falls_back_to_the_ui() {
        let app = app(&uis());
        assert_eq!(
            ask(&app, "bot.tokera.com", "/npc/42").await,
            (200, "<p>npcd</p>".into())
        );
    }

    /// The server [`builder`] assembles, asked who `peer` is when it sends
    /// `forged` as its identity (or nothing).
    async fn who(app: &Router, peer: &str, forged: Option<&str>) -> String {
        let mut req = Request::get("/v1/who").header("host", "code.tokera.com");
        if let Some(email) = forged {
            req = req
                .header(USER, "g-1")
                .header(PROVIDER, "google")
                .header(EMAIL, email);
        }
        let mut req = req.body(Body::empty()).unwrap();
        let addr: SocketAddr = peer.parse().unwrap();
        req.extensions_mut().insert(ConnectInfo(addr));
        let res = app.clone().oneshot(req).await.unwrap();
        let body = to_bytes(res.into_body(), 1 << 16).await.unwrap();
        String::from_utf8_lossy(&body).into_owned()
    }

    fn assembled(
        u: &[(&'static str, TenantUi)],
        gateways: Vec<IpAddr>,
        local_signin: Option<&str>,
    ) -> Router {
        let listen = Listen {
            bind: IpAddr::from([192, 168, 0, 5]),
            port: 0,
            tls_port: None,
            gateways,
        };
        let api = Router::new().route(
            "/v1/who",
            get(|headers: HeaderMap| async move {
                identify(&headers).map_or_else(|_| "anonymous".to_string(), |id| id.email)
            }),
        );
        builder(
            &listen,
            u,
            api,
            LocalSignin::new(local_signin).unwrap(),
            None,
        )
        .unwrap()
        .router()
    }

    /// **The assembled server believes an identity only from its trusted
    /// peers**, and `--local-signin` names loopback alone: a LAN peer forging
    /// headers is anonymous, the named gateway and the bind address are
    /// believed, and a stand-in identity never reaches past loopback.
    #[tokio::test]
    async fn the_assembled_server_believes_only_its_trusted_peers() {
        let u = uis();
        let gated = assembled(&u, vec![IpAddr::from([10, 0, 0, 9])], None);
        assert_eq!(
            who(&gated, "10.0.0.7:1", Some("forged@example.com")).await,
            "anonymous"
        );
        assert_eq!(
            who(&gated, "10.0.0.9:1", Some("real@example.com")).await,
            "real@example.com"
        );
        assert_eq!(
            who(&gated, "192.168.0.5:1", Some("real@example.com")).await,
            "real@example.com"
        );
        assert_eq!(who(&gated, "127.0.0.1:1", None).await, "anonymous");

        let local = assembled(&u, Vec::new(), Some("me@example.com"));
        assert_eq!(who(&local, "127.0.0.1:1", None).await, "me@example.com");
        assert_eq!(
            who(&local, "127.0.0.1:1", Some("other@example.com")).await,
            "other@example.com"
        );
        assert_eq!(who(&local, "10.0.0.7:1", None).await, "anonymous");
        assert_eq!(
            who(&local, "10.0.0.7:1", Some("forged@example.com")).await,
            "anonymous"
        );
    }

    /// One tenant is the default whatever it says; two must name exactly one.
    #[test]
    fn exactly_one_default_site() {
        let u = uis();
        let mut alone = vec![u[1].clone()];
        assert!(site_table(&alone, "127.0.0.1:0".parse().unwrap(), base()).is_ok());
        alone[0].1.default = false;
        let cfg = site_table(&alone, "127.0.0.1:0".parse().unwrap(), base()).unwrap();
        assert_eq!(cfg.site_for(Some("anything")).name, "npcd");

        let mut both = u.clone();
        both[1].1.default = true;
        let err = site_table(&both, "127.0.0.1:0".parse().unwrap(), base())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("2 tenant UIs claim to be the default"),
            "{err}"
        );
    }
}
