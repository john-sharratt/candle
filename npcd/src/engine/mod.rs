//! The engine: the cast's loop, the act vocabulary, and the routes that let people
//! watch and steer the characters.
//!
//! Characters wake at startup, tick on their own salience-driven heartbeats,
//! drain their inboxes as prose and decode through the mind. Where a route needs
//! machinery that does not yet exist, it answers honestly: empty where empty is
//! the measurement, null where nothing has been measured, and 503 for work nothing
//! can do yet. Nothing is invented. Under a cluster each route goes to the member
//! that holds the character, interaction or world it names.
//!
//! - `character_state`: a character's layers, memory, projections and monitor.
//! - `console_stream`: the act stream a console attaches to, and where it resumes.
//! - `generation`: the world-building generation routes.
//! - `interactions`: the sessions a person has with a character.
//! - `messages`: messages to a character and the world's channel.
//! - `voices`: how a person is named on a thread.
//! - Tests: `tests_console_stream`, `tests_voices`.

pub mod act;
pub mod acts;
pub mod agenda;
pub mod archive;
pub mod ask_api;
pub mod at_work;
pub mod authoring;
pub mod authorize;
pub mod bearings;
pub mod bench;
pub mod beyond;
pub mod bodies;
pub mod body;
pub mod build;
pub mod calibrate;
pub mod channels;
pub mod check;
pub mod chronology;
pub mod compose;
pub mod cooldown;
pub mod dreams;
pub mod driver;
pub mod enact;
pub mod environment;
pub mod event;
pub mod experience;
pub mod guardian;
pub mod guardian_api;
pub mod held;
pub mod identity;
pub mod ingest;
pub mod interaction;
pub mod invoke_body;
pub mod journal;
pub mod layers;
pub mod life;
pub mod loading;
pub mod loopguard;
pub mod mind;
pub mod mind_record;
pub mod mission;
pub mod mission_acts;
pub mod mission_api;
pub mod mission_gen;
pub mod narration;
pub mod narrator;
pub mod on_you;
pub mod operations_api;
pub mod order_table;
pub mod passage;
pub mod people;
pub mod perceived;
pub mod persona;
pub mod prompt;
pub mod prose;
pub mod pulse;
pub mod reach;
pub mod reflect;
pub mod rendered;
pub mod reported;
pub mod retention;
pub mod rooms;
pub mod runtime;
pub mod schema;
pub mod simulate;
pub mod slash;
pub mod sleep;
pub mod start_over;
pub mod station;
pub mod stir;
pub mod substrate_views;
pub mod survey;
pub mod talk;
pub mod terminal;
pub mod tick;
pub mod tools;
pub mod trial;
pub mod trial_api;
pub mod watch;
pub mod watcher;
pub mod whereabouts;
pub mod whole;
pub mod window;
pub mod witnessed;
pub mod work;
pub mod workflow;
pub mod zen_task;

#[cfg(test)]
mod tests_slot_sizes;
#[cfg(test)]
mod tests_slots;

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::dispatch::{Dispatch, Merge, Reach};
use host::guard::{delete, get, patch, post};
use serde_json::{json, Value};
use web::auth::{Identity, Role, Roles};

use crate::api::{err, owner_of, Authored};
use crate::authority::ops::{OpReply, WorldOp};
use crate::discord::world::routes as world_discord;
use crate::engine::bodies::{Channel, ChannelSent, MessageSent, Messages, Said, Say};
use crate::engine::interaction::Interlocutor;
use crate::engine::mind::Minds;
use crate::npcs;
use crate::projection;
use crate::sim::phone::{Who, CHANNEL};
use host::guard::Api;

/// The one refusal this module makes, worded once.
///
/// `503`, not `501`: the route exists and is correct, and the thing it needs is
/// expected to arrive. A `501` would say the daemon does not implement it,
/// which is the wrong thing to tell somebody whose console is otherwise working.
pub fn no_engine(what: &str) -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "no_engine",
        &format!("{what} needs an inference engine, and this daemon is not running one"),
    )
}

/// A write to a world that runs on another member did not reach it
/// (`docs/replicated_substrate.md` §9.3): `502`, naming why, since the request
/// was right and the world will be back.
pub fn world_unreached(e: &anyhow::Error) -> Response {
    err(
        StatusCode::BAD_GATEWAY,
        "world_unreached",
        &format!("the world this character stands in did not answer: {e:#}"),
    )
}

/// The engine's routes. Under a cluster (`docs/replicated_substrate.md` §5.2)
/// a character's go to the member holding its lease, an interaction's to the
/// member holding its character (the id is `<nid>-<n>`), a world's live state
/// to its authority, the cast-wide instruments to every member, and generation
/// to the best one.
pub fn api(roles: Roles) -> Api<Arc<Authored>> {
    let character = || Dispatch::owner_path("nid");
    let interaction_owner = || Dispatch::owner_path("ix");
    let world = || Dispatch::owner_path("wid");
    let each = || Dispatch::Aggregate(Merge::Envelope);
    Api::new(roles)
        // ── the substrate, as it actually is ────────────────────────────────
        .route(
            "/v1/npc/:nid/substrate",
            Role::User,
            character(),
            get(substrate),
        )
        .route(
            "/v1/npc/:nid/substrate/layer/:layer",
            Role::User,
            character(),
            get(layer),
        )
        .route(
            "/v1/npc/:nid/substrate/turn/:layer/:turn",
            Role::User,
            character(),
            get(turn),
        )
        .route("/v1/npc/:nid/memory", Role::User, character(), get(memory))
        // ── the substrate inspector ─────────────────────────────────────────
        //
        // Admin, like the host's `/v1/substrate` they hang under: they read
        // every character's conversations, whoever owns them. Any member
        // answers a read — the substrate is replicated — and every member
        // flips a toggle, which is in-memory and its own.
        .route(
            "/v1/substrate/npcd/system-prompt",
            Role::Admin,
            Dispatch::Any,
            get(substrate_views::system_prompt),
        )
        .route(
            "/v1/substrate/npcd/layer/:name",
            Role::Admin,
            Dispatch::Any,
            get(substrate_views::layer),
        )
        .route(
            "/v1/substrate/npcd/layer/:name/toggle",
            Role::Admin,
            each(),
            post(substrate_views::toggle_layer),
        )
        .route(
            "/v1/substrate/npcd/timeline/:tl",
            Role::Admin,
            Dispatch::Any,
            get(substrate_views::timeline),
        )
        // ── the scenario harness ────────────────────────────────────────────
        //
        // Put a situation to a character that does not exist and report
        // everything about what came back. Admin, because it spends the one
        // card's time and reports the daemon's own prompt.
        .route(
            "/v1/simulate",
            Role::Admin,
            Dispatch::Best,
            post(simulate::run),
        )
        // ── instruments ─────────────────────────────────────────────────────
        .route(
            "/v1/npc/:nid/projection",
            Role::User,
            character(),
            get(projection_now),
        )
        .route(
            "/v1/npc/:nid/projection/:tick",
            Role::User,
            character(),
            get(projection_at),
        )
        .route(
            "/v1/npc/:nid/system-prompt",
            Role::User,
            character(),
            get(pulse::system_prompt),
        )
        .route(
            "/v1/npc/:nid/prompt-tokens",
            Role::User,
            character(),
            get(pulse::prompt_tokens),
        )
        .route(
            "/v1/npc/:nid/monitor",
            Role::User,
            character(),
            get(monitor),
        )
        .route("/v1/npc/:nid/project", Role::User, character(), post(probe))
        .route(
            "/v1/npc/:nid/perceive",
            Role::User,
            character(),
            post(perceive),
        )
        // ── interactions ────────────────────────────────────────────────────
        .route(
            "/v1/npc/:nid/interaction",
            Role::User,
            character(),
            get(list_interactions).post(open_interaction),
        )
        .route(
            "/v1/interaction/:ix",
            Role::User,
            interaction_owner(),
            get(interaction).delete(end_interaction),
        )
        .route(
            "/v1/interaction/:ix/inject",
            Role::User,
            interaction_owner(),
            post(inject),
        )
        .route(
            "/v1/interaction/:ix/stream",
            Role::User,
            interaction_owner(),
            get(stream),
        )
        // ── messaging a character ───────────────────────────────────────────
        //
        // **Real, and the same handset the characters use.** Not a private
        // channel between a console and a mind: a person messaging a character
        // is one more party on a thread, and the character answers with the
        // same `message` act it would use to answer anybody. That is what makes
        // the reply worth having — it is the character speaking to you, from
        // inside the world, rather than a chat window bolted to the side of it.
        .route(
            "/v1/npc/:nid/message",
            Role::User,
            character(),
            get(get_messages).post(post_message),
        )
        // ── missions ────────────────────────────────────────────────────────
        //
        // Lodge a mission for a character, and read how one turned out. `User`
        // plus the ownership check inside, like everything else that writes to
        // one owned character — a mission is a thing asked of somebody's cast.
        .route(
            "/v1/npc/:nid/mission",
            Role::User,
            character(),
            get(mission_api::status).post(mission_api::lodge),
        )
        // Call off one owned character's mission.
        .route(
            "/v1/npc/:nid/mission/cancel",
            Role::User,
            character(),
            post(mission_api::cancel),
        )
        // Tick off or add a step on one owned character's open mission.
        .route(
            "/v1/npc/:nid/mission/step",
            Role::User,
            character(),
            post(mission_api::step),
        )
        // ── looking inside a character, and steering it ─────────────────────
        //
        // Ask it a question and force an answer; put a thought in its head. Both
        // write to one owned character, so `User` plus the ownership check.
        .route(
            "/v1/npc/:nid/ask",
            Role::User,
            character(),
            post(ask_api::ask),
        )
        .route(
            "/v1/npc/:nid/mind_control",
            Role::User,
            character(),
            post(ask_api::mind_control),
        )
        // A trial turn on a suspended character: one turn answered as one
        // call, nothing of it kept (`docs/npc_trials.md`). One owned character,
        // so `User` plus the ownership check.
        .route(
            "/v1/npc/:nid/trial",
            Role::User,
            character(),
            post(trial_api::trial),
        )
        // ── the world's open channel ────────────────────────────────────────
        //
        // The standing group every character joins on arrival — see
        // `sim::phone::CHANNEL`. Registered twice because the two methods need
        // different roles, which is what one line each is for: reading it is a
        // `User`'s, and speaking on it is a `Creator`'s, because it reaches
        // every character in a world at once and ownership cannot express that.
        // See [`post_channel`].
        .route(
            "/v1/world/:wid/channel",
            Role::User,
            world(),
            get(get_channel),
        )
        .route(
            "/v1/world/:wid/channel",
            Role::Creator,
            world(),
            post(post_channel),
        )
        // Words left on something in a world, for whoever comes to it — the
        // world's half of `post_notice`. Creator for the same reason the
        // channel's write side is: it writes into a world rather than into a
        // character somebody owns.
        .route(
            "/v1/world/:wid/posting",
            Role::Creator,
            world(),
            post(post_posting),
        )
        // The world's own Discord server (`docs/npcd_discord.md` §4.4): set,
        // read and taken away by a Creator, for the same reason the channel's
        // write side is — it mirrors the world as a whole.
        .route(
            "/v1/world/:wid/discord",
            Role::Creator,
            world(),
            get(world_discord::get_world_discord)
                .put(world_discord::put_world_discord)
                .delete(world_discord::delete_world_discord),
        )
        // ── the act vocabulary ──────────────────────────────────────────────
        //
        // Real: the catalog and the `/` command list are compiled in, and both
        // are served rather than duplicated in the console so the two cannot
        // drift. The catalog is the host's `/v1/tools`, which reads it from
        // npcd through [`pulse::catalog`].
        .route(
            "/v1/commands",
            Role::User,
            Dispatch::Any,
            get(pulse::commands),
        )
        // ── Pulse: the cast's loop, as an instrument ────────────────────────
        .route("/v1/pulse", Role::User, each(), get(pulse::feed))
        .route("/v1/pulse/census", Role::User, each(), get(pulse::census))
        // Where everybody is standing. The one question the feed and the window
        // between them cannot answer — see [`pulse::world`].
        //
        // Under `/v1/pulse`, not `/v1/world`: that one is the authored world
        // registry, which is a different thing entirely — the documents an
        // author wrote, not the simulation running from them.
        .route("/v1/pulse/world", Role::User, each(), get(pulse::world))
        .route(
            "/v1/npc/:nid/pulse",
            Role::User,
            character(),
            post(pulse::inject),
        )
        // A line spoken straight into a character's world, from any in-world
        // voice at a chosen loudness — the influence primitive under the
        // console's `/` notation. `User` plus the ownership check inside, like
        // `pulse`: it reaches only a character the caller owns.
        .route(
            "/v1/npc/:nid/direct",
            Role::User,
            character(),
            post(pulse::direct),
        )
        .route(
            "/v1/npc/:nid/window",
            Role::User,
            character(),
            get(pulse::window),
        )
        // What the character's `scan` tells it, as prose and as the data behind it.
        .route(
            "/v1/npc/:nid/scan",
            Role::User,
            character(),
            get(pulse::scan),
        )
        // The character's journal and what its drafts have been doing, and a way
        // to draft one now. `User` plus the ownership check inside, like `window`.
        .route(
            "/v1/npc/:nid/journal",
            Role::User,
            character(),
            get(pulse::journal),
        )
        .route(
            "/v1/npc/:nid/journal/draft",
            Role::User,
            character(),
            post(pulse::journal_draft),
        )
        // Forget one entry: out of the character's memory and its prompt.
        .route(
            "/v1/npc/:nid/journal/:eid",
            Role::User,
            character(),
            delete(pulse::journal_forget),
        )
        // Admin: it reaches characters the caller does not own, which every
        // other route on this daemon refuses to do — on every member, each to
        // the characters it runs.
        .route(
            "/v1/pulse/broadcast",
            Role::Admin,
            each(),
            post(pulse::broadcast),
        )
        // A plain announcement to the whole world at URGENT — the announcement-
        // shaped sibling of `broadcast`. Admin for the same reason: it reaches
        // every character, including ones the caller does not own.
        .route(
            "/v1/pulse/announce",
            Role::Admin,
            each(),
            post(pulse::announce),
        )
        // The command table: open it to call the cast to take up missions, shut
        // it to stand them down. Admin — it reaches the whole cast, and drives
        // every character, including ones the caller does not own.
        .route(
            "/v1/pulse/command-table",
            Role::Admin,
            each(),
            post(mission_api::command_table),
        )
        // The operator's house bridge (`docs/zen_moon_mind.md` §15): work a
        // device directly through an in-network agent, no character needed.
        // One member takes it and submits the job; admin, like the table.
        .route(
            "/v1/pulse/bridge",
            Role::Admin,
            Dispatch::Any,
            post(mission_api::bridge),
        )
        // Set a device's notes and credentials on the world part standing for
        // it (`docs/zen_moon_mind.md` §15) — the onboarding metadata the world
        // keeps and folds into every effector command. Admin.
        .route(
            "/v1/pulse/device",
            Role::Admin,
            Dispatch::Any,
            post(mission_api::device),
        )
        // Pair with a device through an in-network agent and keep the
        // credentials on its world part (`docs/zen_moon_mind.md` §15). Admin.
        .route(
            "/v1/pulse/pair",
            Role::Admin,
            Dispatch::Any,
            post(mission_api::pair),
        )
        // What the guardian has found and done. Admin: it names every
        // character, including ones the caller does not own.
        .route(
            "/v1/pulse/guardian",
            Role::Admin,
            each(),
            get(guardian_api::records),
        )
        // The mission board and the operations on it are a world's, and only
        // its authority changes them (§9.2): a route naming one world goes to
        // its authority, and one naming none goes to every member, each acting
        // on the worlds it runs. Reading them is any member's: every member
        // holds every world's state.
        //
        // Clear the whole board's missions at once.
        .route(
            "/v1/pulse/missions/cancel",
            Role::Admin,
            each(),
            post(mission_api::cancel_all),
        )
        // Start named characters over: their missions called off and their
        // conversations retired. Admin — it reaches characters the caller does
        // not own, and cannot be undone.
        .route(
            "/v1/pulse/start-over",
            Role::Admin,
            each(),
            post(start_over::start_over),
        )
        // The table's generator: write one mission now, and read or clear what
        // it has put on the table and what it has found. Admin — the work it
        // sets is taken up by the whole cast.
        .route(
            "/v1/pulse/missions/generate",
            Role::Admin,
            each(),
            post(mission_api::generate),
        )
        .route(
            "/v1/pulse/missions/pool",
            Role::Admin,
            Dispatch::Any,
            get(mission_api::pool),
        )
        .route(
            "/v1/pulse/missions/pool",
            Role::Admin,
            each(),
            delete(mission_api::discard_pool),
        )
        // Operations: the objectives the table holds, each a draft and a review
        // by two Makers. Read every world's, put a document through review,
        // edit one, call one off. Admin, like the rest of the table.
        .route(
            "/v1/pulse/operations",
            Role::Admin,
            Dispatch::Any,
            get(operations_api::list),
        )
        .route(
            "/v1/pulse/operations",
            Role::Admin,
            each(),
            post(operations_api::review),
        )
        .route(
            "/v1/pulse/operations/:wid/:oid",
            Role::Admin,
            world(),
            patch(operations_api::edit),
        )
        .route(
            "/v1/pulse/operations/:wid/:oid/cancel",
            Role::Admin,
            world(),
            post(operations_api::cancel),
        )
        .route(
            "/v1/pulse/operations/:wid/:oid/step",
            Role::Admin,
            world(),
            post(operations_api::step),
        )
        .route(
            "/v1/pulse/operations/:wid/:oid/retry",
            Role::Admin,
            world(),
            post(operations_api::retry),
        )
        .route(
            "/v1/pulse/operations/:wid/:oid/document",
            Role::Admin,
            world(),
            get(operations_api::document),
        )
        // ── generation ──────────────────────────────────────────────────────
        .route(
            "/v1/generate/description",
            Role::User,
            Dispatch::Best,
            post(gen_description),
        )
        // The same generation, arriving a token at a time. Both, because they
        // are different operations to a caller — see [`describe::post_describe_stream`].
        .route(
            "/v1/generate/description/stream",
            Role::User,
            Dispatch::Best,
            post(gen_description_stream),
        )
        // Named before described: the create form fills the name field the
        // moment it opens, and the description is then written about that
        // person rather than inventing a second one.
        .route(
            "/v1/generate/name",
            Role::User,
            Dispatch::Best,
            post(gen_name),
        )
        .route(
            "/v1/generate/attributes",
            Role::User,
            Dispatch::Best,
            post(gen_attributes),
        )
        // Drawn on a member whose card reaches its area (`guest::reach`).
        .route(
            "/v1/image/generate",
            Role::User,
            Dispatch::Best,
            post(gen_image),
        )
        .reach(Reach::Draw)
        .route(
            "/v1/image/models",
            Role::User,
            Dispatch::Best,
            get(image_models),
        )
        .route(
            "/v1/image/queue",
            Role::User,
            Dispatch::Best,
            get(image_queue),
        )
        // ── the push stream ─────────────────────────────────────────────────
        .route("/ws/events", Role::User, Dispatch::Any, get(events))
}

mod character_state;
mod console_stream;
mod generation;
mod interactions;
mod messages;
#[cfg(test)]
mod tests_console_stream;
#[cfg(test)]
mod tests_voices;
mod voices;

pub(crate) use character_state::*;
use console_stream::*;
use generation::*;
use interactions::*;
pub(crate) use messages::*;
pub(crate) use voices::*;
