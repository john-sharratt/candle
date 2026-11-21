//! The route table, where each route's role bar is written down.
//!
//! Every line names the role its route needs. The router has no method for
//! adding a route without one, so the check happens once in the guard and not
//! in each handler. Paths whose methods differ in role are registered twice,
//! and axum merges them. Reads of authored content are open to a signed-in
//! user and writes are admin-only. A character's life plan and the guest jobs
//! that stop the world are admin-only, including their reads, because they
//! cost the whole estate.

use super::*;

/// The routes this daemon owns, to be layered over the mock.
///
/// Every line names the role it needs, because [`Api`] has no method that adds
/// a route without one — see `host::guard`. The check runs there, not in the
/// handlers, so there is exactly one place per route where the answer lives.
///
/// Paths whose methods differ in role are registered twice, which axum merges:
/// reading a world is open and writing one is an admin's, and one line each is
/// what makes that visible.
///
/// Under a cluster (`docs/replicated_substrate.md` §5.2) authored content is
/// read from any member and written on the best one; a character's routes go to
/// the member holding its lease, and a life's to the member that claimed it, so
/// a generation and the job that reports on it are on one machine.
pub fn api(roles: Roles) -> Api<Arc<Authored>> {
    let character = || Dispatch::owner_path("nid");
    let life = || Dispatch::sticky_path("who");
    Api::new(roles)
        // Authored content. Anyone may read it; only an admin may change it, as
        // a draft of their own until they publish it (§10.3). Its reads are
        // mirrored by shape when a new build certifies: each member reads its
        // own checkout of the mind, which may stand at another commit.
        .route(
            "/v1/world",
            Role::Unauthenticated,
            Dispatch::Any,
            get(list_worlds),
        )
        .mirror(Mirror::Shape)
        .route(
            "/v1/world/:wid",
            Role::Unauthenticated,
            Dispatch::Any,
            get(get_world),
        )
        .mirror(Mirror::Shape)
        // The response and mood libraries, read from the mind. This used to
        // fall through to the fixture, which answered with six invented
        // templates where the mind holds 596 — see [`crate::collections`].
        .route(
            "/v1/world/:wid/collections",
            Role::Unauthenticated,
            Dispatch::Any,
            get(world_collections),
        )
        .route(
            "/v1/world/:wid",
            Role::Admin,
            Dispatch::Best,
            put(put_world).delete(delete_world),
        )
        // The narrative clock. Admin for the same reason the document is: every
        // character in the world dates what they remember by it.
        .route(
            "/v1/world/:wid/time",
            Role::Admin,
            Dispatch::Best,
            put(put_world_time),
        )
        .route(
            "/v1/personality",
            Role::Unauthenticated,
            Dispatch::Any,
            get(list_personalities),
        )
        .mirror(Mirror::Shape)
        .route(
            "/v1/personality/:aid",
            Role::Unauthenticated,
            Dispatch::Any,
            get(get_personality),
        )
        .mirror(Mirror::Shape)
        .route(
            "/v1/personality/:aid",
            Role::Admin,
            Dispatch::Best,
            put(put_personality).delete(delete_personality),
        )
        // The anchor, the facets and the doctrine, from the document itself.
        .route(
            "/v1/personality/:aid/collections",
            Role::Unauthenticated,
            Dispatch::Any,
            get(personality_collections),
        )
        // A character's life: the plan, the strata, and generating them.
        //
        // **Admin throughout, including the reads.** Every other authored
        // document is readable by a signed-in user, but a life plan carries the
        // seed — the arc, the shape, the world events a life is hung on — which
        // is the authoring intent behind a character rather than anything the
        // character is. And every write here puts prose into the substrate that
        // a character will think it remembers, which is a larger act than
        // editing a page of canon.
        // **Co-resident models, and why they are `Admin`.**
        //
        // A guest job stops the world: the scheduler evicts the engine's KV
        // working set to make room, and every character in every world stops
        // thinking until it finishes. That is a legitimate thing for an operator
        // to ask for and not a thing to expose to a player — the ask is cheap to
        // make and expensive to serve, which is the shape of request that needs
        // a hand on it rather than a rate limit.
        // Separating a picture from its background.
        //
        // **`user`, because the split is by audience and not by cost.** Every
        // console image route is `user` — `/v1/image/generate` included, and
        // that one stops the world for ten to twenty-five seconds — while the
        // raw `/v1/guest/*` operator routes are `admin`. This serves the Images
        // page, which is itself a `user` page, so `admin` here would be the one
        // console image route that refused the people the page is for.
        //
        // What bounds the exposure is not the role: a matte is one forward at a
        // fixed size, `GuestRequest::check` caps the picture, and the queue
        // serialises drains. It is the cheapest world-stopping thing this
        // daemon offers, by an order of magnitude.
        .route(
            "/v1/image/cutout",
            Role::User,
            Dispatch::Best,
            post(guest_routes::post_cutout),
        )
        // A prompt cleared and signed (`docs/guest_draw_orchestration.md` §1):
        // for members — another member drawing — and admins, gated in the
        // handler, which reads the member stamp the route table cannot.
        .route(
            authorize::peer::PATH,
            Role::Unauthenticated,
            Dispatch::Best,
            post(authorize::route::post_authorize),
        )
        // The guests each member is running.
        .route(
            "/v1/guest",
            Role::Admin,
            Dispatch::Aggregate(Merge::Envelope),
            get(guest_routes::get_guests),
        )
        // Drawn on a member whose card reaches its area (`guest::reach`).
        .route(
            "/v1/guest/image",
            Role::Admin,
            Dispatch::Best,
            post(guest_routes::post_image),
        )
        .reach(Reach::Draw)
        .route(
            "/v1/generate/prose",
            Role::Admin,
            Dispatch::Best,
            post(prose::post_prose),
        )
        // One stratum of a life, rewritten in the narrator's voice. The ladder
        // (`/generate`) is the right shape for a five-hundred-node fan-out; this
        // is the other case — one node, on demand.
        .route(
            "/v1/life/:who/node/:key/narrate",
            Role::Admin,
            life(),
            post(lifegen_narrate::post_narrate),
        )
        .route(
            "/v1/life/catalog",
            Role::Admin,
            Dispatch::Any,
            get(lifegen::get_catalog),
        )
        .route("/v1/life/:who", Role::Admin, life(), get(lifegen::get_life))
        .route(
            "/v1/life/:who/seed",
            Role::Admin,
            life(),
            put(lifegen::put_seed),
        )
        .route(
            "/v1/life/:who/node/:key",
            Role::Admin,
            life(),
            put(lifegen::put_node),
        )
        .route(
            "/v1/life/:who/day",
            Role::Admin,
            life(),
            post(lifegen::post_day),
        )
        .route(
            "/v1/life/:who/day/:date",
            Role::Admin,
            life(),
            delete(lifegen::delete_day),
        )
        .route(
            "/v1/life/:who/day/:date/consequences",
            Role::Admin,
            life(),
            put(lifegen::put_consequences),
        )
        .route(
            "/v1/life/:who/generate",
            Role::Admin,
            life(),
            post(lifegen::post_generate),
        )
        .route(
            "/v1/life/:who/job",
            Role::Admin,
            life(),
            get(lifegen::get_job),
        )
        .route(
            "/v1/life/:who/cancel",
            Role::Admin,
            life(),
            post(lifegen::post_cancel),
        )
        // The authored corpus — canon, craft, characters, settings. Addressed
        // by what things ARE (`canon/ammo/bolt`), never by where they are
        // stored; see [`crate::mind::address`].
        //
        // Reading is `User` rather than open, unlike the documents above. The
        // difference is enumeration: `/v1/personality/cindy-tan` answers
        // somebody who already knows the id, while listing hands out the whole
        // corpus a level at a time, which is exactly the browsing the `hidden`
        // flag exists to prevent. The console is a signed-in tool, so this
        // costs nothing that was available anyway.
        .route("/v1/mind/list", Role::User, Dispatch::Any, get(mind_list))
        .route("/v1/mind/entry", Role::User, Dispatch::Any, get(mind_entry))
        .route(
            "/v1/mind/entry",
            Role::Admin,
            Dispatch::Best,
            put(put_mind_entry).delete(delete_mind_entry),
        )
        // The same entry as fields rather than as text, so it can be edited by
        // somebody who does not know YAML. A save patches the values into the
        // document that is there, keeping the authoring comments above every
        // key — see [`crate::mind::section`].
        .route(
            "/v1/mind/fields",
            Role::User,
            Dispatch::Any,
            get(mind_fields),
        )
        .route(
            "/v1/mind/fields",
            Role::Admin,
            Dispatch::Best,
            put(put_mind_fields),
        )
        // An admin's drafts of the mind, and landing them on the record
        // (§10.3). An author's store replicates, so any member reads it; the
        // git work runs on the best one.
        .route(
            "/v1/mind/drafts",
            Role::Admin,
            Dispatch::Any,
            get(drafts::get_drafts),
        )
        .route(
            "/v1/mind/drafts",
            Role::Admin,
            Dispatch::Best,
            delete(drafts::delete_drafts),
        )
        .route(
            "/v1/mind/publish",
            Role::Admin,
            Dispatch::Best,
            post(drafts::post_publish),
        )
        .route(
            "/v1/mind/sync",
            Role::Admin,
            Dispatch::Best,
            post(drafts::post_sync),
        )
        // The nine layers are the host's `/v1/schema/layers`, which reads them
        // from npcd through [`schema_layers`].
        //
        // The cast. These replace the console's fixture: a character here is a
        // record in the substrate owned by the signed-in account. `User` is the
        // bar; *ownership* is the rest of the answer and is checked per-record
        // in the handler, because a role cannot express "yours" (§8.2).
        .route("/v1/npc", Role::User, Dispatch::Any, get(list_npcs))
        .route("/v1/npc", Role::User, Dispatch::Best, post(create_npc))
        // Battle Cities' MCP server (`crate::mcp`). Open, because the server
        // names its own caller and refuses nobody with the challenge that
        // starts a sign-in. A tool call goes to the member holding the
        // character or world it acts on, as the route for the same act would.
        // Under npcd's own prefix: zend serves `/v1/mcp`, and one host may
        // carry both tenants, whose routes must not meet. A call that draws
        // goes to a member whose card reaches its area.
        .route(
            "/v1/npc/mcp",
            Role::Unauthenticated,
            Dispatch::Owner(KeySource::Derived(Derived::McpSubject)),
            post(mcp_http::post::<Authored>)
                .get(mcp_http::get)
                .delete(mcp_http::delete::<Authored>),
        )
        .reach(Reach::Draw)
        // Backup and restore the whole cast, for a substrate wipe-and-rebuild.
        // Admin-only: export carries every character's record regardless of
        // owner, and import writes them verbatim into a fresh substrate.
        .route(
            "/v1/npcs/export",
            Role::Admin,
            Dispatch::Any,
            get(export_npcs),
        )
        // Every kept custom object, out as text and back, for a substrate moved
        // aside under a new model key (`docs/hit_driven_projection.md` §13.4).
        // Admin-only for the reason the cast's backup is.
        .route(
            "/v1/substrate/npcd/objects/export",
            Role::Admin,
            Dispatch::Any,
            get(export_objects),
        )
        .route(
            "/v1/substrate/npcd/objects/import",
            Role::Admin,
            Dispatch::Best,
            post(import_objects).body_limit(IMPORT_BODY_LIMIT),
        )
        .route(
            "/v1/npcs/import",
            Role::Admin,
            Dispatch::Best,
            post(import_npcs),
        )
        // Who owns what, and handing a character to another account. Admin:
        // the listing reads every owner's cast, and the reassignment is the one
        // write that crosses ownership.
        .route(
            "/v1/npcs/owners",
            Role::Admin,
            Dispatch::Any,
            get(get_owners),
        )
        .route(
            "/v1/npc/:nid/owner",
            Role::Admin,
            Dispatch::Best,
            put(put_owner),
        )
        .route(
            "/v1/npc/:nid",
            Role::User,
            character(),
            get(get_npc).patch(patch_npc).delete(delete_npc),
        )
        // Two fields with routes of their own, because the console edits them
        // from controls that are nowhere near the rest of the form: a tag chip
        // and a checkbox, each saving on the spot. They are `PATCH` underneath
        // — same validation, same record, same supersession — and exist so the
        // console does not have to send a whole character to add one tag.
        .route(
            "/v1/npc/:nid/tags",
            Role::User,
            character(),
            put(put_npc_tags),
        )
        .route(
            "/v1/npc/:nid/hidden",
            Role::User,
            character(),
            put(put_npc_hidden),
        )
        // Suspending: a suspended character takes no turn of any kind until it
        // is resumed (`crate::suspend`). One character goes to the member
        // running it, so its own scheduler holds it at once; the whole cast
        // goes to any member, and the others follow the records it writes.
        // `User` plus the ownership check inside, which an admin passes for
        // any.
        .route(
            "/v1/npc/:nid/suspend",
            Role::User,
            character(),
            post(suspend::post_suspend),
        )
        .route(
            "/v1/npcs/suspend",
            Role::User,
            Dispatch::Best,
            post(suspend::post_suspend_all),
        )
        // The authoring plane (§16). `User` is the bar; ownership is the rest
        // of the answer and is checked per record.
        .route(
            "/v1/npc/:nid/beliefs",
            Role::User,
            character(),
            get(get_beliefs),
        )
        .route(
            "/v1/npc/:nid/beliefs/:bid",
            Role::User,
            character(),
            put(put_belief).delete(delete_belief),
        )
        .route(
            "/v1/npc/:nid/relationships",
            Role::User,
            character(),
            get(get_relationships),
        )
        .route(
            "/v1/npc/:nid/relationships/:eid",
            Role::User,
            character(),
            put(put_relationship),
        )
        .route(
            "/v1/npc/:nid/agency",
            Role::User,
            character(),
            get(get_agency),
        )
        .route(
            "/v1/npc/:nid/agency/:sid",
            Role::User,
            character(),
            put(put_strategy),
        )
        // Stops the character, runs the reflection's decodes, throws the
        // conversation away.
        // `User` for the same reason the portrait route is: it is *your own*
        // character, and the handler refuses one that is not.
        .route(
            "/v1/npc/:nid/reflect",
            Role::User,
            character(),
            post(post_reflect),
        )
        .route(
            "/v1/npc/:nid/dream",
            Role::User,
            character(),
            post(post_dream),
        )
        .route(
            "/v1/npc/:nid/modulation",
            Role::User,
            character(),
            get(get_modulation).put(put_modulation),
        )
        // A portrait is a file, and needs no engine — see [`crate::images`].
        .route(
            "/v1/npc/:nid/portrait",
            Role::User,
            character(),
            put(put_portrait),
        )
        // Drawing one does need an engine, and stops every character thinking
        // while it runs. `User` rather than `Admin` all the same: it is a
        // portrait of *your own* character, the route refuses anything you do
        // not own, and making it admin-only would mean nobody could ever use
        // the button on the character they just created.
        .route(
            "/v1/npc/:nid/portrait/generate",
            Role::User,
            character(),
            post(portrait::post_generate),
        )
        // The character's own Discord bot (`docs/npcd_discord.md` §4.3): its
        // owner pastes the token, and it is checked, named, sealed and kept.
        // `User` and the ownership check inside, like the portrait.
        .route(
            "/v1/npc/:nid/discord",
            Role::User,
            character(),
            get(discord::get_discord)
                .put(discord::put_discord)
                .delete(discord::delete_discord),
        )
        // What the bot has done on Discord (`docs/npcd_discord.md` §5.5): its
        // records, its shard and its last events, for any character — so
        // `Admin`, the bar of a read that crosses ownership.
        .route(
            "/v1/npc/:nid/discord/trace",
            Role::Admin,
            character(),
            get(discord_trace::get_trace),
        )
        // `/v1/image/models` and `/v1/generate/description` are **not** here.
        // They already exist in `engine::routes` as the placeholders the console
        // was written against, and that is where their real implementations now
        // live — a second registration is not an override, it is an
        // "overlapping method route" panic at startup.
        .route("/v1/image/:iid", Role::User, Dispatch::Best, get(get_image))
        // The same bytes through a link that carries its own permission — a
        // signature over the id and an expiry ([`crate::image_links`]). Open,
        // because the link is the permission; the handler refuses any other.
        .route(
            "/v1/image/link/:iid",
            Role::Unauthenticated,
            Dispatch::Best,
            get(get_linked_image),
        )
        // A picture handed in to be edited, through an upload link
        // (`image_upload`): open for the same reason, the link being the
        // permission and the handler refusing anything else before it reads a
        // byte of the body.
        .route(
            "/v1/image/upload",
            Role::Unauthenticated,
            Dispatch::Best,
            put(put_upload).body_limit(UPLOAD_BODY),
        )
        // An account and its profile are the caller's own, so `User` and then
        // the record is keyed by their subject — there is no id in these paths
        // to belong to somebody else. `/v1/me` itself is the host's, which
        // reads the account from npcd through [`account`].
        .route(
            "/v1/me/profile",
            Role::Guest,
            Dispatch::Any,
            get(get_profile),
        )
        .route(
            "/v1/me/profile",
            Role::Guest,
            Dispatch::Best,
            put(put_profile),
        )
        .route(
            "/v1/me/profile/history",
            Role::Guest,
            Dispatch::Any,
            get(get_profile_history),
        )
        .route(
            "/v1/me/profile/history/:rev",
            Role::Guest,
            Dispatch::Any,
            get(get_profile_revision),
        )
        .route(
            "/v1/me/profile/restore/:rev",
            Role::Guest,
            Dispatch::Best,
            post(restore_profile),
        )
        .route(
            "/v1/me/unique-name",
            Role::Guest,
            Dispatch::Best,
            put(put_unique_name),
        )
        // The caller's own Discord account, linked by a code sent to a
        // character's bot (`docs/npcd_discord.md` §3.2). The caller's own
        // like the profile; the read anywhere, the writes on the best member.
        .route(
            "/v1/me/discord",
            Role::Guest,
            Dispatch::Any,
            get(discord_link::get_link),
        )
        .route(
            "/v1/me/discord",
            Role::Guest,
            Dispatch::Best,
            delete(discord_link::delete_link),
        )
        .route(
            "/v1/me/discord/link",
            Role::Guest,
            Dispatch::Best,
            post(discord_link::post_link),
        )
}

/// The finished router, for tests. The tenant builds it from [`api`] so the
/// host can read the route table on the way past.
#[cfg(test)]
pub fn router(state: Arc<Authored>) -> Router {
    api(state.roles.clone()).into_routes(state).router
}
