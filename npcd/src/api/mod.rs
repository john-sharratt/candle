//! The HTTP routes npcd answers itself, laid over the mock that answers the rest.
//!
//! Worlds and personalities are authored files in the mind. A save is an admin's
//! draft of one over a mind under git, or the file itself over a mind folder
//! that is not under git. An id in a path is a key in a map read from the
//! folder and never a path, so the read path makes no filesystem call. The one
//! place an id becomes a file name is a save, and it is checked first. Any route
//! not yet answered here falls through to the mock, so each surface moves over
//! one route at a time.
//!
//! - `account_routes`: the caller's own account, profile and unique name.
//! - `errors`: failure responses that keep server detail off the wire.
//! - `image_routes`: portrait upload, serving and signed image links.
//! - `mind_routes`: the authored corpus, by address, for reading and editing.
//! - `npc_routes`: the cast's listing, creation, edits and ownership.
//! - `npc_sections`: a character's beliefs, relationships and agency.
//! - `record_routes`: worlds and personalities, as the caller may see them.
//! - `route_table`: every route, with the role it needs.
//! - `worlds`: world and personality listings, the narrative clock and documents.
//! - Tests: `tests_accounts`, `tests_characters`, `tests_images`, `tests_mind`, `tests_routes`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, RwLock as SyncRwLock};

use axum::body::{to_bytes, Body};
use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use cluster::dispatch::{Derived, Dispatch, KeySource, Merge, Mirror, Reach};
use host::guard::{delete, get, post, put};
use mcp_server::http as mcp_http;
use mcp_server::origin;
use mcp_server::SealedStore;
// Only the test-only `router` builds one directly; the tenant goes through `api`.
#[cfg(test)]
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use tokio::task::spawn_blocking;
use web::auth::session::Identity;

use candle_conversation::persistence::cast;
use candle_conversation::persistence::kept_objects::IMPORT_BODY_LIMIT;
use web::auth::{Role, Roles};

use crate::accounts::{self, account_of, Accounts, NameError, PatchError};
use crate::authorize;
use crate::authorize::{Cache as AuthorizationCache, Signer};
use crate::clock::{self, Clock};
use crate::collections::{self, Libraries};
use crate::discord::bots::Bots;
use crate::discord::client::Client as DiscordClient;
use crate::discord::link::routes as discord_link;
use crate::discord::routes as discord;
use crate::discord::seal::Sealer;
use crate::discord::trace::route as discord_trace;
use crate::drafts::routes as drafts;
use crate::drafts::{caller, records, Author};
use crate::engine::journal::text as journal_text;
use crate::engine::runtime::{OwnedPersona, Runtime};
use crate::guest_routes;
use crate::identity::require;
use crate::image_links::{ImageLinks, Refused};
use crate::image_seed;
use crate::images::{ImageError, Images, Kept};
use crate::lifegen::job::Jobs;
use crate::lifegen::narrate as lifegen_narrate;
use crate::lifegen::routes as lifegen;
use crate::mind::catalog::CatalogError;
use crate::mind::doc::{DocError, Wrote};
use crate::mind::{
    catalog as mind_catalog, parts as mind_parts, section as mind_section, Address, Mind, MindPath,
    Scope,
};
use crate::npcs::{self, Filter, NpcError, Npcs};
use crate::personality_portrait;
use crate::portrait;
use crate::projection::BUNDLED;
use crate::prose;
use crate::refimage;
use crate::registry::authored::{PERSONALITIES, WORLDS};
use crate::registry::{self, PutError, Record, Registry};
use crate::sim::shelf::{Published, Shelf};
use crate::suspend;
use crate::visibility;
use crate::world_clocks::WorldClocks;
use crate::world_state::renamed;
use host::guard::Api;

/// Both authored collections, shared with the HTTP layer.
///
/// A write lock is held only for the duration of a save. Reads are the common
/// case by orders of magnitude and take the read lock, so a GUI listing worlds
/// never waits on another GUI editing one.
pub struct Authored {
    pub worlds: RwLock<Registry>,
    /// Every world's clock, as `worlds` last held it, read without a lock —
    /// see [`crate::world_clocks`]. Stored whenever `worlds` is read again.
    pub clocks: Arc<WorldClocks>,
    pub personalities: RwLock<Registry>,
    pub accounts: RwLock<Accounts>,
    /// The cast, in memory, backed by the substrate — see [`crate::npcs`]. No
    /// lock: a read loads the published cast ([`Npcs::view`]) and a write runs
    /// on the persistence owner's thread (`docs/npcd_cast_lock_free.md`).
    ///
    /// Held behind an `Arc` so the one cast — and with it the process's one
    /// writable substrate handle ([`crate::npcs::Npcs::substrate`]) — can be
    /// shared with the engine: [`crate::engine::runtime::Runtime::set_npcs`]
    /// adopts this same handle, so the effector device reads a character's own
    /// projected state and writes its agency plane through the identical `Npcs`
    /// there is no second of.
    pub npcs: Arc<Npcs>,
    /// Who is an admin, from the config. Not behind a lock: it is decided at
    /// startup and there is deliberately no way to change it while running —
    /// see [`web::auth::role`].
    pub roles: Roles,
    /// The response and mood libraries, read from the mind at start — and once
    /// more when a mind pending at start arrives, which is the only write.
    /// Behind an `Arc` so a reader clones the handle out and lets the lock go.
    libraries: SyncRwLock<Arc<Libraries>>,
    /// Uploaded portraits, on disk. No lock: it holds a path, and the writes
    /// are content-addressed and atomic — two uploads of the same image are the
    /// same file, and of different ones are different names.
    pub images: Images,
    /// Signs and checks the links that open a stored image without signing in
    /// ([`crate::image_links`]). This process's own key until the tenant,
    /// joining a cluster, sets the one every member derives.
    pub image_links: ImageLinks,
    /// Signs and checks prompt authorizations ([`crate::authorize`]). This
    /// process's own key until the tenant, joining a cluster, sets the one
    /// every member derives.
    pub authorizations: Signer,
    /// Seals and opens the characters' Discord bot tokens
    /// ([`crate::discord::seal`]). This process's own key until the tenant,
    /// joining a cluster, sets the one every member derives.
    pub discord_tokens: Sealer,
    /// Discord's REST API, which a bot's connect checks its token with
    /// ([`crate::discord::client`]).
    pub discord: DiscordClient,
    /// The characters' bots this member runs, on the authority of each
    /// one's world ([`crate::discord::bots`]).
    pub bots: Bots,
    /// The authorizations this member has, for their lifetime.
    pub authorized: AuthorizationCache,
    /// `personality_id -> image_id` for the portraits personalities were
    /// authored with — see [`crate::personality_portrait`].
    ///
    /// Derived again each time the personality registry is read again. It is
    /// *derived* from the registry rather than stored in it: the registry holds
    /// the document as authored, and an id minted by this daemon's image store
    /// is not something the author wrote. Keeping it out here is what stops a
    /// save round-tripping a local id back into the mind.
    pub personality_portraits: SyncRwLock<BTreeMap<String, String>>,
    /// The mind directory, for the file editor.
    pub mind: Mind,
    /// The folder the worlds and personalities are kept in, and an author's
    /// writes go to when the mind has no folder of its own: the data
    /// directory of a daemon with none.
    authored_dir: PathBuf,
    /// The engine.
    ///
    /// Always present, and always *there before the model is*. The daemon binds
    /// its port and serves the console while the weights are still loading, so
    /// what varies is not whether this exists but what it can answer — see
    /// [`Runtime::is_ready`].
    ///
    /// `Option` only so the tests can stand this state up without an engine.
    pub runtime: Option<Arc<Runtime>>,
    /// Life generations, running and finished — see [`crate::lifegen`].
    ///
    /// Not behind a lock and not built at startup from anything: it is a
    /// registry that starts empty and fills as an operator generates lives, and
    /// it holds its own mutex over the one map it owns. Kept here rather than on
    /// the runtime because a generation is an *authoring* action against the
    /// mind directory — the engine is something it borrows, not something it
    /// belongs to.
    pub lifegen: Jobs,
    /// The MCP server's sessions ([`crate::mcp`]): carried in their ids, since
    /// its tools hold nothing on a session and a call may reach any member.
    pub mcp_sessions: Arc<SealedStore>,
}

impl Authored {
    /// Nine arguments, and they stay nine.
    ///
    /// Every one is a distinct thing the daemon read at startup from a different
    /// place — two registries, the account store, the cast, the role table, the
    /// libraries, the mind, the portrait store and the portraits personalities
    /// were authored with. Bundling them into a struct to satisfy an arity
    /// threshold would be the same nine fields written twice, named once for the
    /// builder and once here, with a second place for them to disagree. Called
    /// exactly once, from the tenant's start.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        worlds: Registry,
        personalities: Registry,
        accounts: Accounts,
        npcs: Npcs,
        roles: Roles,
        libraries: Libraries,
        mind: Mind,
        images: Images,
        personality_portraits: BTreeMap<String, String>,
    ) -> Self {
        // Both registries are read from their folders of one place, which is
        // where an author's writes go when the mind has no folder.
        let authored_dir = worlds
            .dir()
            .parent()
            .map(FsPath::to_path_buf)
            .unwrap_or_default();
        Self {
            clocks: Arc::new(WorldClocks::of(&worlds)),
            worlds: RwLock::new(worlds),
            personalities: RwLock::new(personalities),
            accounts: RwLock::new(accounts),
            npcs: Arc::new(npcs),
            roles,
            libraries: SyncRwLock::new(Arc::new(libraries)),
            images,
            image_links: ImageLinks::minted(),
            authorizations: Signer::minted(),
            discord_tokens: Sealer::minted(),
            discord: DiscordClient::production(),
            bots: Bots::production(),
            authorized: AuthorizationCache::default(),
            personality_portraits: SyncRwLock::new(personality_portraits),
            mind,
            authored_dir,
            runtime: None,
            lifegen: Jobs::new(),
            mcp_sessions: Arc::new(SealedStore),
        }
    }

    /// Account `account` as an author: its view of the authored folder, and
    /// where its writes go — its drafts, over a mind under git (§10.3).
    pub fn author(&self, account: &str) -> Author {
        let stores = self.runtime.as_ref().map(|rt| &rt.ops);
        Author::open(&self.mind, &self.authored_dir, stores, account)
    }

    /// What follows a write through `author`: over a mind that keeps no
    /// drafts, the write is what stands, so what is read from the folder is
    /// read again.
    pub async fn wrote(&self, author: &Author) {
        if !author.keeps_drafts() {
            self.reload_authored().await;
        }
    }

    /// The response and mood libraries as they stand.
    pub fn libraries(&self) -> Arc<Libraries> {
        Arc::clone(&self.libraries.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Take `libraries` in place of the ones read at start: the mind they
    /// were read from had not arrived then.
    pub fn set_libraries(&self, libraries: Libraries) {
        *self.libraries.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(libraries);
    }

    /// Tell `rt` the ids the worlds had before they were renamed, as their files
    /// name them now, and move each former id's characters to the world that
    /// names it (`docs/world_rename.md` §3).
    pub async fn renamings(&self, rt: &Runtime) {
        let formers = renamed::formers(&*self.worlds.read().await);
        for (new, olds) in &formers {
            for old in olds {
                match self.npcs.rehome(old, new, now_ms()) {
                    Ok(0) => {}
                    Ok(n) => tracing::info!("world `{new}`: {n} characters moved from `{old}`"),
                    Err(e) => tracing::error!(
                        "world `{new}`: the characters of `{old}` were not moved: {e:?}"
                    ),
                }
            }
        }
        rt.standing.set_formers(formers);
    }

    /// Read the worlds and personalities again from the folder, as it stands
    /// now: after the mind folder moves to a newer record, or a write to a
    /// mind that keeps no drafts. A world that is gone stops running.
    pub async fn reload_authored(&self) {
        let gone = {
            let mut worlds = self.worlds.write().await;
            let gone = match worlds.reload() {
                Ok(gone) => gone,
                Err(e) => {
                    tracing::warn!("worlds could not be read again: {e:#}");
                    Vec::new()
                }
            };
            // The clocks as the registry stands now, whether it read again or
            // not, stored while it is still held.
            self.clocks.store(&worlds);
            gone
        };
        if let Err(e) = self.personalities.write().await.reload() {
            tracing::warn!("personalities could not be read again: {e:#}");
        }
        if let Some(rt) = self.runtime.as_ref() {
            self.renamings(rt).await;
            for wid in &gone {
                // **A former id is renamed, not removed** (`docs/world_rename.md`
                // §3): it stops running here, and its records stay for the
                // world that names it to take in.
                match rt.standing.is_former(wid) {
                    true => {
                        rt.unhost(wid);
                        tracing::info!("world `{wid}`: renamed — stopped, its records kept");
                    }
                    false => {
                        rt.remove_world(wid);
                    }
                }
            }
            // A personality's `takes_orders` reaches characters already standing
            // in their worlds.
            rt.mark_orders().await;
        }
        let portraits = {
            let personalities = self.personalities.read().await;
            personality_portrait::ingest(&personalities, personalities.dir(), &self.images)
        };
        *self
            .personality_portraits
            .write()
            .unwrap_or_else(|e| e.into_inner()) = portraits;
    }

    /// The same state with an engine behind it.
    ///
    /// The runtime is built first — it needs only the mind directory — and the
    /// two things it needs *from* this state, the world clock and the persona
    /// source, are installed afterwards through
    /// [`Runtime::set_clock`]/[`Runtime::set_persona_source`]. That ordering is
    /// what breaks the cycle: `Authored` needs the runtime to answer routes, and
    /// the runtime needs `Authored` to know what time it is for a character.
    ///
    /// It was a rebuild-by-unwrap once, which panicked in exactly the case it
    /// was written for: the clock closure captures a clone of this state, so by
    /// the time the runtime existed the `Arc` already had two holders and
    /// `try_unwrap` could never succeed.
    pub fn with_runtime(mut self, runtime: Arc<Runtime>) -> Arc<Self> {
        self.runtime = Some(runtime);
        Arc::new(self)
    }

    /// Let image `id`'s key go at once, at this member's own records as they
    /// stand — the image's blob among them. An image is held by nobody; the
    /// release is what tells the gateway which members hold it, so an edit
    /// naming it is placed among them, never on one that would have to wait,
    /// with this member the fallback (`docs/cooperative_admission.md` §2.2). A
    /// daemon with no cluster has nobody to tell.
    pub fn hand_over_image(&self, id: &str) {
        let Some(member) = self.runtime.as_ref().and_then(|rt| rt.standing.member()) else {
            return;
        };
        let Some(through) = member.replicator().map(|r| r.own_through()) else {
            return;
        };
        member.hold(id.as_bytes());
        member.let_go(id.as_bytes(), through);
    }

    /// Every image a portrait names — a character's or a personality's: what
    /// no retirement of a kept picture may take.
    pub async fn portrait_image_ids(&self) -> HashSet<String> {
        let mut named = self.npcs.view().portrait_ids();
        named.extend(
            self.personality_portraits
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .values()
                .cloned(),
        );
        named
    }

    /// What time it is in a character's world: a lookup in [`Self::clocks`],
    /// taking no lock, so a handler and the tick driver read it alike.
    pub fn world_ms(&self, npc_id: u64) -> u64 {
        self.clocks
            .world_ms(&self.npcs.view(), npc_id, now_ms() as i64)
    }
}

/// Wall-clock milliseconds, for stamping a record. A clock before the epoch is
/// not a reason to refuse a write, so it reads as zero.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

mod account_routes;
mod errors;
mod image_routes;
mod mind_routes;
mod npc_routes;
mod npc_sections;
mod record_routes;
mod route_table;
#[cfg(test)]
mod tests_accounts;
mod tests_cast_backup;
#[cfg(test)]
mod tests_characters;
#[cfg(test)]
mod tests_images;
#[cfg(test)]
mod tests_mind;
#[cfg(test)]
mod tests_routes;
mod worlds;

pub use account_routes::*;
pub use errors::*;
pub use image_routes::*;
pub use mind_routes::*;
pub(crate) use npc_routes::*;
use npc_sections::*;
pub(crate) use record_routes::*;
pub use route_table::*;
use worlds::*;
