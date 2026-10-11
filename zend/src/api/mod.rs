//! zend's own HTTP routes.
//!
//! [`routes`] wires the OpenAI-compatible `/v1/chat/completions` endpoint plus
//! zend's own REST surface to per-resource submodules — `chat`,
//! `conversations`, `projections` (one projection point in full, for the
//! projection panel), `files` (conversation-attached uploads), `models`,
//! `repo_map`, and `substrate` (zend's layer/timeline inspection and projection
//! debug, under `/v1/substrate/zend/`).
//!
//! What both daemons serve — status, identity, telemetry, memory, the log
//! stream, the tool catalog, the substrate overview — is the host's, served
//! once for every tenant (`host::endpoints`), and the UI is served from disk by
//! the host (`zend/web/`). Nothing here overlaps either.

use std::sync::Arc;

use axum::routing::{get, post};
use host::guard::{Api, Guarded, Routes};
use web::auth::{Role, Roles};

use crate::session::ZendSession;

pub mod chat;
mod chat_frames;
mod compressed;
pub mod conversations;
pub mod files;
pub mod maintenance;
pub mod models;
pub mod projections;
mod repo_map;
pub mod substrate;

/// zend's routes, graded by `roles`, on `session`.
pub fn routes(session: Arc<ZendSession>, roles: Roles) -> Routes {
    table(roles).into_routes(session)
}

/// Every route zend serves, and the role each needs — the table the host
/// merges with its own and the other tenants' at start, read without a
/// session.
pub fn declared(roles: Roles) -> Vec<Guarded> {
    table(roles).declared().to_vec()
}

/// zend's route table.
///
/// The chat and conversation routes are open: an OpenAI client (Continue)
/// sends no identity, and the tools dial resolves the caller's role itself
/// ([`crate::access`]). The substrate views are an admin's: they read every
/// conversation and name paths on this host.
fn table(roles: Roles) -> Api<Arc<ZendSession>> {
    Api::new(roles)
        .route(
            "/v1/chat/completions",
            Role::Unauthenticated,
            post(chat::completions),
        )
        .route("/v1/models", Role::Unauthenticated, get(models::list))
        .route(
            "/v1/repo_map",
            Role::Unauthenticated,
            get(repo_map::completeness),
        )
        .route(
            "/v1/substrate/zend/system-prompt",
            Role::Admin,
            get(substrate::system_prompt).delete(substrate::retire_system_prompt),
        )
        .route(
            "/v1/substrate/zend/layer/:name",
            Role::Admin,
            get(substrate::layer),
        )
        .route(
            "/v1/substrate/zend/layer/:name/toggle",
            Role::Admin,
            post(substrate::toggle_layer),
        )
        .route(
            "/v1/substrate/zend/timeline/:tl",
            Role::Admin,
            get(substrate::timeline).delete(substrate::delete_timeline),
        )
        .route(
            "/v1/substrate/zend/timeline/:tl/selection",
            Role::Admin,
            get(substrate::selection),
        )
        .route(
            "/v1/substrate/zend/project",
            Role::Admin,
            post(substrate::project),
        )
        .route(
            "/v1/substrate/zend/maintenance",
            Role::Admin,
            post(maintenance::force_maintenance),
        )
        .route(
            "/v1/conversations",
            Role::Unauthenticated,
            get(conversations::list),
        )
        .route(
            "/v1/conversations/:id",
            Role::Unauthenticated,
            get(conversations::get).delete(conversations::delete),
        )
        .route(
            "/v1/conversations/:id/files",
            Role::Unauthenticated,
            get(files::list).post(files::upload),
        )
        .route(
            "/v1/conversations/:id/files/:file_id",
            Role::Unauthenticated,
            get(files::content).delete(files::delete),
        )
        .route(
            "/v1/conversations/:id/archive",
            Role::Unauthenticated,
            post(conversations::archive),
        )
        .route(
            "/v1/conversations/:id/live",
            Role::Unauthenticated,
            get(conversations::live),
        )
        .route(
            "/v1/conversations/:id/projections/:turn/:event",
            Role::Unauthenticated,
            get(projections::get),
        )
        .route(
            "/v1/conversations/:id/projection-context",
            Role::Unauthenticated,
            post(projections::context),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The route table is what we think it is.** Each path and the role it
    /// needs, in order: a route that drifts open or shut fails here, by name.
    #[test]
    fn zends_route_table_is_what_we_think_it_is() {
        let got: Vec<(&str, &str)> = table(Roles::default())
            .declared()
            .iter()
            .map(|g| (g.path, g.min.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("/v1/chat/completions", "unauthenticated"),
                ("/v1/models", "unauthenticated"),
                ("/v1/repo_map", "unauthenticated"),
                ("/v1/substrate/zend/system-prompt", "admin"),
                ("/v1/substrate/zend/layer/:name", "admin"),
                ("/v1/substrate/zend/layer/:name/toggle", "admin"),
                ("/v1/substrate/zend/timeline/:tl", "admin"),
                ("/v1/substrate/zend/timeline/:tl/selection", "admin"),
                ("/v1/substrate/zend/project", "admin"),
                ("/v1/substrate/zend/maintenance", "admin"),
                ("/v1/conversations", "unauthenticated"),
                ("/v1/conversations/:id", "unauthenticated"),
                ("/v1/conversations/:id/files", "unauthenticated"),
                ("/v1/conversations/:id/files/:file_id", "unauthenticated"),
                ("/v1/conversations/:id/archive", "unauthenticated"),
                ("/v1/conversations/:id/live", "unauthenticated"),
                (
                    "/v1/conversations/:id/projections/:turn/:event",
                    "unauthenticated"
                ),
                (
                    "/v1/conversations/:id/projection-context",
                    "unauthenticated"
                ),
            ]
        );
    }
}
