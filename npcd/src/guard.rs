//! Route registration that cannot forget a role.
//!
//! # The weakness this removes
//!
//! Every route used to carry its own check, in its own handler:
//!
//! ```ignore
//! .route("/v1/world/:wid", put(put_world))   // and, inside put_world:
//! if let Err(r) = admin(&s, &headers) { return *r; }
//! ```
//!
//! That works exactly as long as everybody remembers. It is also how the worst
//! finding in this daemon's security review happened: four routes that wrote
//! and deleted authored files were registered without a check, answered 200 to
//! a request with no headers at all, and looked completely ordinary next to the
//! routes that did have one. Nothing failed. Nothing warned. The only way to
//! find it was to read every handler and notice an absence — and an absence is
//! the hardest thing to see in a diff.
//!
//! # What replaces it
//!
//! [`Api`] is a router that has no way to add an unguarded route. Its only
//! registration method takes a [`Role`], the type system requires it, and the
//! check is applied here rather than in the handler — so a new route is
//! *unable* to be silently open. Forgetting is not a mistake you can make; the
//! most you can do is name the wrong role, which is a decision visible on the
//! line rather than an omission visible nowhere.
//!
//! There is no escape hatch on purpose. `Api` does not expose its inner
//! `Router`, and there is no `route_unguarded`, because a hatch is the thing
//! that gets used at 2am and never revisited. A genuinely public route is
//! spelled [`Role::Unauthenticated`], which reads as the deliberate choice it
//! is and shows up in the table below as one.
//!
//! # And the table is pinned by a test
//!
//! [`Api::declared`] reports every route and its role, and
//! `api::tests::the_route_table_is_what_we_think_it_is` asserts the whole set.
//! Adding a route fails that test until somebody writes the new line down,
//! which is the second half of "no silent weakness": the compiler stops you
//! omitting a role, and the test stops you adding one nobody looked at.

use std::fmt;

use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::routing::MethodRouter;
use axum::Router;
use web::auth::{Role, Roles};

use crate::identity::require;

/// One registered route, for the startup log and the route-table test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guarded {
    pub path: &'static str,
    pub min: Role,
}

impl fmt::Display for Guarded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<16} {}", self.min.as_str(), self.path)
    }
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

    /// Register a route behind `min`.
    ///
    /// A path whose methods need *different* roles is registered twice — once
    /// per role — which axum merges. `/v1/world/:wid` is the case: reading a
    /// world is open and writing one is an admin's, and splitting the
    /// registration is what lets each say so on its own line.
    pub fn route(mut self, path: &'static str, min: Role, methods: MethodRouter<S>) -> Self {
        let roles = self.roles.clone();
        let guarded = methods.route_layer(middleware::from_fn(move |req: Request, next: Next| {
            let roles = roles.clone();
            async move {
                match require(req.headers(), &roles, min) {
                    Ok(_) => next.run(req).await,
                    // The refusal names the role required and the role
                    // held — see `identity::require`.
                    Err(refusal) => *refusal,
                }
            }
        }));
        self.declared.push(Guarded { path, min });
        self.router = self.router.route(path, guarded);
        self
    }

    /// The finished router.
    pub fn into_router(self, state: S) -> Router {
        self.router.with_state(state)
    }
}

impl<S> Api<S> {
    /// Every route registered, in declaration order.
    ///
    /// Outside the bounded `impl` so a caller can read the table without
    /// naming the state type's bounds — reporting what was registered has
    /// nothing to do with what the router needs to serve it.
    pub fn declared(&self) -> &[Guarded] {
        &self.declared
    }
}

/// Put a role in front of a whole service — the fallback, in practice.
///
/// The fallback is the quietest surface in the daemon: it answers every path
/// the real routes did not claim, so a route deleted from [`Api`] does not stop
/// being served, it starts being served by whatever is behind it. Naming a role
/// for it is the same discipline applied to the one route nobody registers.
pub fn behind<S>(roles: Roles, min: Role, service: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    service.layer(middleware::from_fn(move |req: Request, next: Next| {
        let roles = roles.clone();
        async move {
            match require(req.headers(), &roles, min) {
                Ok(_) => next.run(req).await,
                Err(refusal) => *refusal,
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::routing::get;
    use tower::ServiceExt;

    fn roles() -> Roles {
        serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap()
    }

    fn app(min: Role) -> Router {
        Api::<()>::new(roles())
            .route("/x", min, get(|| async { "ok" }))
            .into_router(())
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

    /// What the startup log prints and the route-table test asserts against.
    #[test]
    fn every_registration_is_recorded_with_its_role() {
        let api = Api::<()>::new(roles())
            .route("/a", Role::Unauthenticated, get(|| async { "" }))
            .route("/b", Role::Admin, get(|| async { "" }));
        assert_eq!(
            api.declared(),
            [
                Guarded {
                    path: "/a",
                    min: Role::Unauthenticated
                },
                Guarded {
                    path: "/b",
                    min: Role::Admin
                },
            ]
        );
    }

    /// The fallback is a route too, and the one nobody thinks of.
    #[tokio::test]
    async fn a_guarded_fallback_refuses_what_it_should() {
        let inner: Router<()> = Router::new().route("/anything", get(|| async { "ok" }));
        let app = behind(roles(), Role::Admin, inner).with_state(());
        let res = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/anything")
                    .header("x-tokera-user", "nobody")
                    .header("x-tokera-provider", "google")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
    }
}
