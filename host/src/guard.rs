//! Route registration that cannot forget a role — for the host's endpoints and
//! every tenant's.
//!
//! # Why the check is not in the handler
//!
//! A route that carries its own check in its own handler —
//!
//! ```ignore
//! .route("/v1/world/:wid", put(put_world))   // and, inside put_world:
//! if let Err(r) = admin(&s, &headers) { return *r; }
//! ```
//!
//! — is guarded exactly as long as everybody remembers, and one registered
//! without the check answers 200 to a request with no headers at all while
//! looking completely ordinary beside the routes that have one. Nothing fails
//! and nothing warns.
//!
//! # How it is done here
//!
//! [`Api`] is a router that has no way to add an unguarded route. Its only
//! registration methods take a [`Role`], and the check is applied here rather
//! than in the handler — so a new route is *unable* to be silently open. A
//! genuinely public route is spelled [`Role::Unauthenticated`], which reads as
//! the deliberate choice it is and shows up in the table as one.
//!
//! # The table is the host's too
//!
//! [`Api::declared`] reports every route and its role. The host reads it from
//! every tenant and from its own endpoints, and refuses to start when two of
//! them register the same path ([`Routes::merge`]) — an overlap is a shared
//! endpoint the host should serve once (`docs/shared_model_host.md` §3.4).
//!
//! # Every route says how a cluster dispatches it
//!
//! A registration also names the route's [`Dispatch`]
//! (`docs/replicated_substrate.md` §5.2) and is built from this module's
//! [`get`], [`post`], [`put`], [`delete`] and [`patch`], which record the
//! methods it answers. The table a member sends its gateways ([`Guarded::route`])
//! is the declared table with both, so a read and a write of one path can be
//! dispatched differently, each by its own registration.

use std::fmt;

use axum::extract::Request;
use axum::handler::Handler;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{self, MethodRouter};
use axum::{Json, Router};
use cluster::dispatch::{Dispatch, Mirror, Route, RouteTable};
use serde_json::json;
use web::auth::{Role, Roles};

use crate::access::role;

/// One registered route, for the startup log, the route-table tests and the
/// dispatch table a member sends its gateways.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guarded {
    pub path: &'static str,
    pub min: Role,
    /// A mounted router ([`Api::nest_self_authenticating`]): it answers every
    /// path under `path`, not `path` alone.
    pub nested: bool,
    /// The methods it answers, upper case; empty for a mount, which answers
    /// every method.
    pub methods: Vec<&'static str>,
    pub dispatch: Dispatch,
}

impl fmt::Display for Guarded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let under = if self.nested { "/…" } else { "" };
        write!(f, "{:<16} {}{under}", self.min.as_str(), self.path)
    }
}

/// A route's handlers, with the methods they answer recorded.
pub struct Methods<S> {
    router: MethodRouter<S>,
    names: Vec<&'static str>,
}

macro_rules! method {
    ($name:ident, $upper:literal $(, $also:literal)*) => {
        /// Answer
        #[doc = $upper]
        /// with `handler`.
        pub fn $name<H, T, S>(handler: H) -> Methods<S>
        where
            H: Handler<T, S>,
            T: 'static,
            S: Clone + Send + Sync + 'static,
        {
            Methods {
                router: routing::$name(handler),
                names: vec![$upper $(, $also)*],
            }
        }

        impl<S> Methods<S>
        where
            S: Clone + Send + Sync + 'static,
        {
            /// Also answer
            #[doc = $upper]
            /// with `handler`.
            pub fn $name<H, T>(mut self, handler: H) -> Self
            where
                H: Handler<T, S>,
                T: 'static,
            {
                self.router = self.router.$name(handler);
                self.names.extend([$upper $(, $also)*]);
                self
            }
        }
    };
}

// axum answers `HEAD` from a `GET` handler, so a `GET` route answers both.
method!(get, "GET", "HEAD");
method!(post, "POST");
method!(put, "PUT");
method!(delete, "DELETE");
method!(patch, "PATCH");

impl Guarded {
    /// This route as a member's dispatch table lists it (§5.2): one entry, or
    /// for a mount its own path and everything beneath.
    pub fn routes(&self) -> Vec<Route> {
        let route = |path: String| Route {
            methods: self.methods.iter().map(|m| m.to_string()).collect(),
            path,
            dispatch: self.dispatch.clone(),
            mirror: Mirror::None,
        };
        if self.nested {
            vec![
                route(self.path.to_string()),
                route(format!("{}/*rest", self.path.trim_end_matches('/'))),
            ]
        } else {
            vec![route(self.path.to_string())]
        }
    }

    /// The prefix this route answers everything under, when it answers more
    /// than its own path: a mount, or a catch-all (`/common/*path` answers
    /// everything under `/common`).
    fn prefix(&self) -> Option<&'static str> {
        if self.nested {
            return Some(self.path);
        }
        self.path.find("/*").map(|at| &self.path[..at])
    }

    /// Whether a request could reach both `self` and `other` — the same path
    /// whatever its parameters are named, or one under the other's prefix.
    fn overlaps(&self, other: &Guarded) -> bool {
        let shape = |p: &str| -> Vec<String> {
            p.split('/')
                .map(|s| {
                    if s.starts_with(':') {
                        ":".into()
                    } else {
                        s.to_string()
                    }
                })
                .collect()
        };
        let under = |prefix: &str, path: &str| {
            path == prefix
                || path
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('/'))
        };
        shape(self.path) == shape(other.path)
            || self.prefix().is_some_and(|p| under(p, other.path))
            || other.prefix().is_some_and(|p| under(p, self.path))
    }
}

/// Establish that the caller holds `min`, or produce the refusal that says why
/// not.
///
/// The two refusals are different answers and are not interchangeable:
///
/// - **401** — nobody is signed in. Signing in would fix it.
/// - **403** — somebody is signed in and it is not enough. Signing in again
///   will not fix it, and telling them to try is how an operator wastes an
///   afternoon on a permissions problem that was never a session problem.
///
/// Boxed because a `Response` is large and this is the cold path.
pub fn require(headers: &HeaderMap, roles: &Roles, min: Role) -> Result<Role, Box<Response>> {
    let held = role(headers, roles);
    if held.at_least(min) {
        return Ok(held);
    }
    Err(Box::new(refuse(held, min)))
}

/// The body for a refused request. It names the role required and the role
/// held, because the alternative is an operator guessing which of the two they
/// got wrong. Neither is a secret.
fn refuse(held: Role, min: Role) -> Response {
    let (status, error) = if held == Role::Unauthenticated {
        (StatusCode::UNAUTHORIZED, "unauthorized")
    } else {
        (StatusCode::FORBIDDEN, "forbidden")
    };
    (
        status,
        Json(json!({
            "error": error,
            "detail": format!("this needs the `{min}` role; you have `{held}`"),
            "required_role": min,
            "role": held,
        })),
    )
        .into_response()
}

/// A router whose every route names the role it needs.
pub struct Api<S> {
    router: Router<S>,
    roles: Roles,
    declared: Vec<Guarded>,
}

impl<S> Api<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub fn new(roles: Roles) -> Self {
        Self {
            router: Router::new(),
            roles,
            declared: Vec::new(),
        }
    }

    /// Register a route behind `min`, dispatched by a cluster as `dispatch`.
    ///
    /// A path whose methods need *different* roles, or a different dispatch,
    /// is registered twice — once per role or dispatch — which axum merges.
    /// `/v1/world/:wid` is the case: reading a world is open and writing one is
    /// an admin's, and splitting the registration is what lets each say so on
    /// its own line.
    pub fn route(
        mut self,
        path: &'static str,
        min: Role,
        dispatch: Dispatch,
        methods: Methods<S>,
    ) -> Self {
        let roles = self.roles.clone();
        let check = middleware::from_fn(move |req: Request, next: Next| {
            let roles = roles.clone();
            async move {
                match require(req.headers(), &roles, min) {
                    Ok(_) => next.run(req).await,
                    Err(refusal) => *refusal,
                }
            }
        });
        let guarded = methods.router.route_layer(check);
        self.declared.push(Guarded {
            path,
            min,
            nested: false,
            methods: methods.names,
            dispatch,
        });
        self.router = self.router.route(path, guarded);
        self
    }

    /// Mount `router` under `path`, unguarded by role: for a surface that
    /// authenticates its callers itself (npcd's effector API, which reads a
    /// device token and never `x-tokera-*`). Recorded in the table at
    /// [`Role::Unauthenticated`], which is what the host's role check grants it,
    /// and dispatched as `dispatch` whatever the path beneath.
    pub fn nest_self_authenticating(
        mut self,
        path: &'static str,
        dispatch: Dispatch,
        router: Router<S>,
    ) -> Self {
        self.declared.push(Guarded {
            path,
            min: Role::Unauthenticated,
            nested: true,
            methods: Vec::new(),
            dispatch,
        });
        self.router = self.router.nest(path, router);
        self
    }

    /// The finished routes.
    pub fn into_routes(self, state: S) -> Routes {
        Routes {
            router: self.router.with_state(state),
            declared: self.declared,
        }
    }
}

impl<S> Api<S> {
    /// Every route registered, in declaration order.
    pub fn declared(&self) -> &[Guarded] {
        &self.declared
    }
}

/// A finished router and the table it was built from.
pub struct Routes {
    pub router: Router,
    pub declared: Vec<Guarded>,
}

/// The dispatch table a member of an app sends its gateways (§5.2): every
/// route in `declared`, with its methods and dispatch.
pub fn dispatch_table(declared: &[Guarded]) -> anyhow::Result<RouteTable> {
    RouteTable::new(declared.iter().flat_map(Guarded::routes).collect())
        .map_err(|e| anyhow::anyhow!("the dispatch table: {e}"))
}

impl Routes {
    /// No routes.
    pub fn empty() -> Self {
        Self {
            router: Router::new(),
            declared: Vec::new(),
        }
    }

    /// A table with no handlers behind it — what a [`Self::merge`] needs to
    /// check two tables for overlap without the state their handlers run on.
    pub fn table(declared: &[Guarded]) -> Self {
        Self {
            router: Router::new(),
            declared: declared.to_vec(),
        }
    }

    /// These routes and `other`'s, refused when a request could reach a route
    /// of each — the same path, or one under the other's mount or catch-all
    /// ([`Guarded::overlaps`]). `owner` and `other_owner` name the two in the
    /// refusal.
    pub fn merge(self, owner: &str, other: Routes, other_owner: &str) -> anyhow::Result<Routes> {
        for a in &self.declared {
            if let Some(b) = other.declared.iter().find(|b| a.overlaps(b)) {
                anyhow::bail!(
                    "{owner} serves {a} and {other_owner} serves {b}, and a request can reach \
                     both — a path two tenants share is a shared endpoint, and the host serves it \
                     once (docs/shared_model_host.md §3.5)"
                );
            }
        }
        let mut declared = self.declared;
        declared.extend(other.declared);
        Ok(Routes {
            router: self.router.merge(other.router),
            declared,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use cluster::dispatch::Merge;
    use tower::ServiceExt;

    fn roles() -> Roles {
        serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap()
    }

    fn app(min: Role) -> Router {
        Api::<()>::new(roles())
            .route("/x", min, Dispatch::Any, get(|| async { "ok" }))
            .into_routes(())
            .router
    }

    async fn call(min: Role, sub: Option<&str>) -> StatusCode {
        let mut b = HttpRequest::builder().uri("/x");
        if let Some(s) = sub {
            b = b
                .header("x-tokera-user", s)
                .header("x-tokera-provider", "google");
        }
        app(min)
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    /// The bar is enforced by the registration, with no cooperation from the
    /// handler — the handler here does nothing but return `"ok"`.
    #[tokio::test]
    async fn the_role_is_enforced_without_the_handler_helping() {
        assert_eq!(call(Role::Unauthenticated, None).await, StatusCode::OK);
        assert_eq!(
            call(Role::Unauthenticated, Some("nobody")).await,
            StatusCode::OK
        );

        assert_eq!(call(Role::User, None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(call(Role::User, Some("nobody")).await, StatusCode::OK);
        assert_eq!(call(Role::User, Some("boss")).await, StatusCode::OK);

        assert_eq!(call(Role::Admin, None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(
            call(Role::Admin, Some("nobody")).await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(call(Role::Admin, Some("boss")).await, StatusCode::OK);
    }

    /// What the startup log prints and the route-table tests assert against.
    #[test]
    fn every_registration_is_recorded_with_its_role() {
        let api = Api::<()>::new(roles())
            .route(
                "/a",
                Role::Unauthenticated,
                Dispatch::Any,
                get(|| async { "" }),
            )
            .route(
                "/b",
                Role::Admin,
                Dispatch::sticky_path("id"),
                post(|| async { "" }).delete(|| async { "" }),
            )
            .nest_self_authenticating("/c", Dispatch::Best, Router::new());
        assert_eq!(
            api.declared(),
            [
                Guarded {
                    path: "/a",
                    min: Role::Unauthenticated,
                    nested: false,
                    methods: vec!["GET", "HEAD"],
                    dispatch: Dispatch::Any,
                },
                Guarded {
                    path: "/b",
                    min: Role::Admin,
                    nested: false,
                    methods: vec!["POST", "DELETE"],
                    dispatch: Dispatch::sticky_path("id"),
                },
                Guarded {
                    path: "/c",
                    min: Role::Unauthenticated,
                    nested: true,
                    methods: vec![],
                    dispatch: Dispatch::Best,
                },
            ]
        );
    }

    /// **The dispatch table is the declared one** (§5.2): a read and a write
    /// of one path dispatch as each registration says, and a mount dispatches
    /// everything beneath it; a path nothing declares is a UI file.
    #[test]
    fn the_dispatch_table_is_the_declared_one() {
        let api = Api::<()>::new(roles())
            .route(
                "/v1/world/:wid",
                Role::Unauthenticated,
                Dispatch::Any,
                get(|| async { "" }),
            )
            .route(
                "/v1/world/:wid",
                Role::Admin,
                Dispatch::Best,
                put(|| async { "" }).delete(|| async { "" }),
            )
            .route(
                "/v1/status",
                Role::Unauthenticated,
                Dispatch::Aggregate(Merge::Envelope),
                get(|| async { "" }),
            )
            .nest_self_authenticating("/v1/local", Dispatch::Best, Router::new());
        let table = dispatch_table(api.declared()).unwrap();
        assert_eq!(table.dispatch("GET", "/v1/world/w1"), Dispatch::Any);
        assert_eq!(table.dispatch("HEAD", "/v1/world/w1"), Dispatch::Any);
        assert_eq!(table.dispatch("PUT", "/v1/world/w1"), Dispatch::Best);
        assert_eq!(table.dispatch("DELETE", "/v1/world/w1"), Dispatch::Best);
        assert_eq!(
            table.dispatch("GET", "/v1/status"),
            Dispatch::Aggregate(Merge::Envelope)
        );
        assert_eq!(table.dispatch("POST", "/v1/local/tick/now"), Dispatch::Best);
        assert_eq!(table.dispatch("GET", "/v1/local"), Dispatch::Best);
        assert_eq!(table.dispatch("GET", "/index.html"), Dispatch::Ui);
    }

    fn g(path: &'static str, nested: bool) -> Guarded {
        Guarded {
            path,
            min: Role::Unauthenticated,
            nested,
            methods: vec![],
            dispatch: Dispatch::Any,
        }
    }

    /// A request can reach both routes when they are one path whatever their
    /// parameters are named, or one sits under the other's mount or
    /// catch-all; neighbours that merely share a prefix string do not.
    #[test]
    fn overlap_is_reachability() {
        assert!(g("/v1/npc/:id", false).overlaps(&g("/v1/npc/:nid", false)));
        assert!(g("/v1/local", true).overlaps(&g("/v1/local/tick", false)));
        assert!(g("/v1/local/tick", false).overlaps(&g("/v1/local", true)));
        assert!(g("/common/*path", false).overlaps(&g("/common/lib/a.js", false)));
        assert!(g("/common/*path", false).overlaps(&g("/common", false)));

        assert!(!g("/v1/local", true).overlaps(&g("/v1/localities", false)));
        assert!(!g("/v1/npc/:id", false).overlaps(&g("/v1/npc/:id/turns", false)));
        assert!(!g("/v1/status", false).overlaps(&g("/v1/substrate", false)));
    }

    /// **A path two tenants register refuses the start**, naming the path and
    /// both owners; disjoint tables merge.
    #[test]
    fn a_shared_path_is_refused_and_disjoint_ones_merge() {
        let routes = |paths: &[&'static str]| {
            let mut api = Api::<()>::new(roles());
            for p in paths {
                api = api.route(
                    p,
                    Role::Unauthenticated,
                    Dispatch::Any,
                    get(|| async { "" }),
                );
            }
            api.into_routes(())
        };
        let err = routes(&["/v1/status", "/v1/a"])
            .merge("zend", routes(&["/v1/b", "/v1/status"]), "npcd")
            .err()
            .expect("an overlap refuses")
            .to_string();
        assert!(err.starts_with("zend serves"), "{err}");
        assert!(err.contains("/v1/status and npcd serves"), "{err}");

        let mounted = Api::<()>::new(roles())
            .nest_self_authenticating("/v1/local", Dispatch::Best, Router::new())
            .into_routes(());
        assert!(routes(&["/v1/local/tick"])
            .merge("zend", mounted, "npcd")
            .is_err());

        let merged = routes(&["/v1/a"])
            .merge("zend", routes(&["/v1/b"]), "npcd")
            .unwrap();
        let paths: Vec<&str> = merged.declared.iter().map(|g| g.path).collect();
        assert_eq!(paths, ["/v1/a", "/v1/b"]);
    }
}
