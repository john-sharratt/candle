//! zend's own HTTP routes.
//!
//! [`routes`] wires the OpenAI-compatible `/v1/chat/completions` endpoint plus
//! zend's own REST surface to per-resource submodules — `chat`,
//! `conversations`, `projections` (one projection point in full, for the
//! projection panel), `files` (conversation-attached uploads), `models`,
//! `repo_map`, `substrate` (zend's layer/timeline inspection and projection
//! debug, under `/v1/substrate/zend/`), and `agents` (the machines its
//! commands run on, watched and switched by an admin).
//!
//! What both daemons serve — status, identity, telemetry, memory, the log
//! stream, the tool catalog, the substrate overview — is the host's, served
//! once for every tenant (`host::endpoints`), and the UI is served from disk by
//! the host (`zend/web/`). Nothing here overlaps either.

use std::sync::Arc;

use cluster::dispatch::{Derived, Dispatch, KeySource, Merge, Mirror};
use host::guard::{delete, get, post, put, Api, Guarded, Routes};
use web::auth::{Role, Roles};

use crate::session::ZendSession;

pub mod agents;
pub mod chat;
mod chat_frames;
mod compressed;
pub mod conversations;
pub mod files;
pub mod journal;
pub mod models;
pub mod projections;
pub mod prompt_tokens;
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

/// The header the zend GUI names its conversation in, so a gateway routes a
/// chat without buffering its body (`docs/replicated_substrate.md` §5.2).
pub const CONVERSATION_HEADER: &str = "x-tokera-conversation";

/// zend's route table.
///
/// The chat and conversation routes are open: an OpenAI client (Continue)
/// sends no identity, and the tools dial resolves the caller's role itself
/// ([`crate::access`]). The substrate views are an admin's: they read every
/// conversation and name paths on this host. So are the agents: they show
/// every conversation's commands, and switch the machines that run them.
///
/// Under a cluster a conversation is served by the member holding its claim;
/// the substrate's replicated state is read from any member, and what lives in
/// one member's memory — a layer's kill switch, a projection's selection — is
/// asked of each. A maintenance pass is the host's `maintain` command.
fn table(roles: Roles) -> Api<Arc<ZendSession>> {
    let conversation = || Dispatch::sticky_path("id");
    let each = || Dispatch::Aggregate(Merge::Envelope);
    Api::new(roles)
        .route(
            "/v1/chat/completions",
            Role::Unauthenticated,
            Dispatch::Sticky(vec![
                KeySource::Header(CONVERSATION_HEADER.into()),
                KeySource::Body("conv_id".into()),
                KeySource::Derived(Derived::Passthrough),
            ]),
            post(chat::completions),
        )
        .route(
            "/v1/models",
            Role::Unauthenticated,
            Dispatch::Any,
            get(models::list),
        )
        .route(
            "/v1/repo_map",
            Role::Unauthenticated,
            Dispatch::Any,
            get(repo_map::completeness),
        )
        // By shape: each member maps its own checkouts (§3.8).
        .mirror(Mirror::Shape)
        .route(
            "/v1/substrate/zend/system-prompt",
            Role::Admin,
            Dispatch::Any,
            get(substrate::system_prompt),
        )
        .route(
            "/v1/substrate/zend/system-prompt",
            Role::Admin,
            Dispatch::Best,
            delete(substrate::retire_system_prompt),
        )
        .route(
            "/v1/substrate/zend/layer/:name",
            Role::Admin,
            Dispatch::Any,
            get(substrate::layer),
        )
        .route(
            "/v1/substrate/zend/layer/:name/toggle",
            Role::Admin,
            each(),
            post(substrate::toggle_layer),
        )
        .route(
            "/v1/substrate/zend/timeline/:tl",
            Role::Admin,
            Dispatch::Any,
            get(substrate::timeline),
        )
        .route(
            "/v1/substrate/zend/timeline/:tl",
            Role::Admin,
            Dispatch::Best,
            delete(substrate::delete_timeline),
        )
        .route(
            "/v1/substrate/zend/timeline/:tl/selection",
            Role::Admin,
            Dispatch::Aggregate(Merge::FirstFound),
            get(substrate::selection),
        )
        .route(
            "/v1/substrate/zend/project",
            Role::Admin,
            Dispatch::Best,
            post(substrate::project),
        )
        // Every member reaches every agent through its gateway, so any one
        // answers for them.
        .route("/v1/agents", Role::Admin, Dispatch::Any, get(agents::list))
        .route(
            "/v1/agents/:node",
            Role::Admin,
            Dispatch::Any,
            put(agents::switch),
        )
        .route(
            "/v1/agents/:node/jobs/:id",
            Role::Admin,
            Dispatch::Any,
            delete(agents::cancel),
        )
        .route(
            "/v1/agents/:node/jobs/:id/stream",
            Role::Admin,
            Dispatch::Any,
            get(agents::stream),
        )
        .route(
            "/v1/conversations",
            Role::Unauthenticated,
            // Every member's list, merged by id: a member rejoining lists the
            // conversations it wrote cut off, which no other member holds yet.
            Dispatch::Aggregate(Merge::Union),
            get(conversations::list),
        )
        // Exactly: the list is replicated state, the same on every member —
        // but for a turn landing between the two calls.
        .mirror(Mirror::Exact(vec![
            "updated_ms".into(),
            "turn_count".into(),
        ]))
        .route(
            "/v1/conversations/:id",
            Role::Unauthenticated,
            conversation(),
            get(conversations::get).delete(conversations::delete),
        )
        .route(
            "/v1/conversations/:id/files",
            Role::Unauthenticated,
            conversation(),
            get(files::list).post(files::upload),
        )
        .route(
            "/v1/conversations/:id/files/:file_id",
            Role::Unauthenticated,
            conversation(),
            get(files::content).delete(files::delete),
        )
        .route(
            "/v1/conversations/:id/archive",
            Role::Unauthenticated,
            conversation(),
            post(conversations::archive),
        )
        .route(
            "/v1/conversations/:id/live",
            Role::Unauthenticated,
            conversation(),
            get(conversations::live),
        )
        .route(
            "/v1/conversations/:id/projections/:turn/:event",
            Role::Unauthenticated,
            conversation(),
            get(projections::get),
        )
        .route(
            "/v1/conversations/:id/projection-context",
            Role::Unauthenticated,
            conversation(),
            post(projections::context),
        )
        .route(
            "/v1/conversations/:id/prompt-tokens",
            Role::Unauthenticated,
            conversation(),
            get(prompt_tokens::get),
        )
        .route(
            "/v1/conversations/:id/journal",
            Role::Unauthenticated,
            conversation(),
            get(journal::get),
        )
        .route(
            "/v1/conversations/:id/journal/draft",
            Role::Unauthenticated,
            conversation(),
            post(journal::draft),
        )
        .route(
            "/v1/conversations/:id/journal/:eid",
            Role::Unauthenticated,
            conversation(),
            delete(journal::forget),
        )
        .route(
            "/v1/conversations/:id/journal/:eid/turns",
            Role::Unauthenticated,
            conversation(),
            get(journal::turns),
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
                ("/v1/substrate/zend/system-prompt", "admin"),
                ("/v1/substrate/zend/layer/:name", "admin"),
                ("/v1/substrate/zend/layer/:name/toggle", "admin"),
                ("/v1/substrate/zend/timeline/:tl", "admin"),
                ("/v1/substrate/zend/timeline/:tl", "admin"),
                ("/v1/substrate/zend/timeline/:tl/selection", "admin"),
                ("/v1/substrate/zend/project", "admin"),
                ("/v1/agents", "admin"),
                ("/v1/agents/:node", "admin"),
                ("/v1/agents/:node/jobs/:id", "admin"),
                ("/v1/agents/:node/jobs/:id/stream", "admin"),
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
                ("/v1/conversations/:id/prompt-tokens", "unauthenticated"),
                ("/v1/conversations/:id/journal", "unauthenticated"),
                ("/v1/conversations/:id/journal/draft", "unauthenticated"),
                ("/v1/conversations/:id/journal/:eid", "unauthenticated"),
                (
                    "/v1/conversations/:id/journal/:eid/turns",
                    "unauthenticated"
                ),
            ]
        );
    }

    /// **Every route dispatches as §5.2 says**, method by method: a
    /// conversation by its claim, a chat by its header ahead of its body, the
    /// substrate's reads anywhere and its writes to one member.
    #[test]
    fn zends_dispatch_table_follows_the_design() {
        let table = host::guard::dispatch_table(table(Roles::default()).declared()).unwrap();
        let conversation = Dispatch::sticky_path("id");
        assert_eq!(
            table.dispatch("POST", "/v1/chat/completions"),
            Dispatch::Sticky(vec![
                KeySource::Header("x-tokera-conversation".into()),
                KeySource::Body("conv_id".into()),
                KeySource::Derived(Derived::Passthrough),
            ])
        );
        for (method, path) in [
            ("GET", "/v1/conversations/c-1"),
            ("DELETE", "/v1/conversations/c-1"),
            ("POST", "/v1/conversations/c-1/files"),
            ("GET", "/v1/conversations/c-1/files/f-2"),
            ("POST", "/v1/conversations/c-1/archive"),
            ("GET", "/v1/conversations/c-1/live"),
            ("GET", "/v1/conversations/c-1/projections/3/4"),
            ("POST", "/v1/conversations/c-1/projection-context"),
            ("GET", "/v1/conversations/c-1/prompt-tokens"),
            ("GET", "/v1/conversations/c-1/journal"),
            ("POST", "/v1/conversations/c-1/journal/draft"),
            ("DELETE", "/v1/conversations/c-1/journal/5"),
            ("GET", "/v1/conversations/c-1/journal/5/turns"),
        ] {
            assert_eq!(
                table.dispatch(method, path),
                conversation,
                "{method} {path}"
            );
        }
        for (method, path, want) in [
            (
                "GET",
                "/v1/conversations",
                Dispatch::Aggregate(Merge::Union),
            ),
            ("GET", "/v1/models", Dispatch::Any),
            ("GET", "/v1/repo_map", Dispatch::Any),
            ("GET", "/v1/agents", Dispatch::Any),
            ("PUT", "/v1/agents/00000000000000a1", Dispatch::Any),
            (
                "DELETE",
                "/v1/agents/00000000000000a1/jobs/j",
                Dispatch::Any,
            ),
            (
                "GET",
                "/v1/agents/00000000000000a1/jobs/j/stream",
                Dispatch::Any,
            ),
            ("GET", "/v1/substrate/zend/timeline/7", Dispatch::Any),
            ("DELETE", "/v1/substrate/zend/timeline/7", Dispatch::Best),
            ("GET", "/v1/substrate/zend/system-prompt", Dispatch::Any),
            ("DELETE", "/v1/substrate/zend/system-prompt", Dispatch::Best),
            (
                "GET",
                "/v1/substrate/zend/timeline/7/selection",
                Dispatch::Aggregate(Merge::FirstFound),
            ),
            (
                "POST",
                "/v1/substrate/zend/layer/repo_map/toggle",
                Dispatch::Aggregate(Merge::Envelope),
            ),
        ] {
            assert_eq!(table.dispatch(method, path), want, "{method} {path}");
        }
    }
}
