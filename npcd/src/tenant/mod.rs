//! npcd, the character engine and its authored world state, run as one tenant of
//! the host's shared runtime.
//!
//! `NpcdArgs` holds npcd's settings, from its own flags and the `npcd:` section of
//! `motherd.yaml`, and `NpcdArgs::resolve` turns them into an `NpcdTenant` while
//! refusing a mistyped mind before anything is opened. The tenant then stands the
//! daemon up on the host's runtime: the authored registries, the cast on its
//! substrate, the runtime and its sinks, the hosted worlds and the routes. It
//! starts the engine load on the host's engine slot, so the server binds before
//! the model finishes loading.
//!
//! - `routes_and_sinks`: the route tables and the runtime resolvers and sinks.
//! - `shutdown_upkeep`: ordered shutdown and the periodic picture and content-check upkeep.
//! - `tenant_start`: the start sequence that reads the mind, builds the engine and reloads on mind moves.
//! - `worlds_and_identities`: hosting each world and the identity collections.
//!
//! Tests:
//!
//! - `tests_flags`: settings defaults, flag and YAML agreement, and the console's visibility rule.
//! - `tests_resolve`: resolution of the mind and data directory, and late mind arrival.

use std::cell::Cell;
use std::env::current_dir;
use std::net::SocketAddr;
use std::path::{absolute, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::extract::{ConnectInfo, Request};
use axum::middleware::{self, Next};
use axum::Router;
use candle_conversation::engine_slot::CoreRequests;
use candle_conversation::models::{DialectType, Model};
use candle_conversation::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;
use candle_conversation::persistence::{OpenStage, SUBSTRATE_DIR};
use candle_conversation::Capabilities;
use clap::{ArgAction, Args};
use cluster::dispatch::Dispatch;
use futures::future::BoxFuture;
use host::guard::{Api, Guarded, Routes};
use host::{OneShot, Running, Tenant, TenantContext, TenantUi};
use include_dir::{include_dir, Dir};
use journals::slug::Slug;
use serde::Deserialize;
use serde_json::Value;
use tokio::runtime::Handle;
use tokio::task::{spawn_blocking, JoinHandle};
use tokio::time::interval;
use web::auth::Roles;

use crate::accounts::Accounts;
use crate::api::{self, Authored};
use crate::authority;
use crate::authorize::{self, Signer};
use crate::collections::Libraries;
use crate::compliance;
use crate::discord::bots;
use crate::discord::client::{self as discord_client, Client as DiscordClient};
use crate::discord::limit::{self, InvalidRequests};
use crate::discord::seal::{self, Sealer};
use crate::effector::router::{router as effector_router, Local};
use crate::effector::token::Tokens;
use crate::engine;
use crate::engine::beyond;
use crate::engine::guardian::{GuardianBuilder, GuardianConfig, RuntimeProbe};
use crate::engine::identity::Authored as Identities;
use crate::engine::runtime::{engine_asks, start as start_engine, Arrived, LoadPlan, Runtime};
use crate::identity::log_identity;
use crate::image_links::{self, ImageLinks};
use crate::images::Images;
use crate::mind::files::MindFiles;
use crate::mind::{Mind, MindRepo};
use crate::mind_store::{self, MindSource};
use crate::model::MODEL;
use crate::npcs::{Casting, Npcs};
use crate::personality_portrait;
use crate::projection;
use crate::registry::authored::{PERSONALITIES, WORLDS};
use crate::registry::Registry;
use crate::reports::NpcdReports;

/// The tenant's name: its log attribution, status key and route owner.
pub const NAME: &str = "npcd";

/// What npcd needs of a model: `<tool_call>`, `<think>` and `</think>` as one
/// token each — its act stencils and parsers key on them — and the Qwen3.5
/// dialect its prompts are written in.
pub fn requires() -> Capabilities {
    Capabilities {
        tokens: &["<tool_call>", "<think>", "</think>"],
        dialect: DialectType::Qwen35,
    }
}

/// npcd's console, `web/content/npcd`, compiled in.
pub static CONSOLE: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../web/content/npcd");

/// The public host the console is served on: the first of the hosts the
/// gateway routes to npcd (the tenant's `ui`), and where a person a
/// character's bot does not know yet is sent to link their Discord account.
pub const HOST: &str = "bot.tokera.com";

/// npcd's flags.
#[derive(Args, Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct NpcdArgs {
    /// Where the engine's own state lives — `substrate/`, `accounts/` and the
    /// mind's store, the things the daemon writes rather than a person. The
    /// `npcd` directory in the source tree when absent.
    #[arg(long)]
    pub data: Option<PathBuf>,

    /// The mind: a repository an agent holds, by its name there — the
    /// cluster's agents, or with no cluster the one in this process
    /// (`docs/substrate_git.md` §8, §9). Its packs and branches are records in
    /// the substrate and its files are read at a commit, so this machine needs
    /// neither the repository nor a way to its origin. With none, the daemon
    /// runs on the bundled schema with no authored content.
    #[arg(long, value_name = "NAME")]
    pub mind_repo: Option<String>,

    /// The branch of `--mind-repo` that is the record.
    #[arg(long, value_name = "BRANCH", default_value = DEFAULT_MIND_BRANCH)]
    pub mind_branch: String,

    /// npcd's checkout of the journals repository, `journals-npcd`: where a
    /// world that names a journals group writes its bots' experience
    /// (`docs/npc_conversations.md` §5.0). Its own clone, never zend's — a
    /// store commits on a private ref, so two never share a checkout. With
    /// none, no world keeps journals on this member.
    #[arg(long, value_name = "DIR")]
    pub journals_dir: Option<PathBuf>,

    /// Log the identity each request to npcd's own routes arrived with, the
    /// peer it came from, and the role it resolved to. Header **values are
    /// never logged** — each header reports only `set`, `EMPTY` or absent. Off
    /// by default and noisy when on: every such request, polls included.
    #[arg(long)]
    pub log_identity: bool,

    /// Retire every character's conversation at startup, so each one opens a
    /// fresh conversation instead of rejoining where it stopped. Each
    /// character's memory, beliefs, relationships and place in the world are
    /// untouched.
    #[arg(long)]
    #[serde(skip)]
    pub forget_conversations: bool,

    /// Retire every dream every character has kept, at startup.
    #[arg(long)]
    #[serde(skip)]
    pub forget_dreams: bool,

    /// Empty the journal every character has kept, at startup.
    #[arg(long)]
    #[serde(skip)]
    pub forget_journal: bool,

    /// Retire every conversation in the substrate at startup, and re-ingest the
    /// mind. The cast is kept: who exists, who owns them and where they stand
    /// are untouched.
    #[arg(long)]
    #[serde(skip)]
    pub wipe_conversations: bool,

    /// How many days the substrate keeps a retired conversation's tombstone
    /// before a compaction may shrink it to a 24-byte entry in the deleted set
    /// (`docs/replicated_substrate.md` §6.8). Only when the bytes are reclaimed
    /// depends on it: the conversation stays retired either way.
    #[arg(long, value_name = "DAYS", default_value_t = DEFAULT_TOMBSTONE_RETENTION_DAYS)]
    pub tombstone_retention: u64,

    /// Whether this member runs characters. `false` — `--no-characters`, or
    /// `characters: false` in `motherd.yaml` — keeps it out of every world and
    /// character lease (`docs/replicated_substrate.md` §9.5): it replicates
    /// the substrate, serves the routes and draws, and no character or Maker
    /// runs here, even with every other member away.
    #[arg(long = "no-characters", action = ArgAction::SetFalse)]
    pub characters: bool,

    /// Whether this member, running no characters, still hosts worlds —
    /// `--host-worlds`, or `host_worlds: true` in `motherd.yaml`. It takes no
    /// character lease, and is a world's authority while no member that runs
    /// characters is ready to be (`authority::world_lease::asker`): the
    /// characters run elsewhere and reach the world over the cluster. Without
    /// it, a member running no characters hosts no world.
    #[arg(long)]
    pub host_worlds: bool,

    /// Whether this member writes the mind into the substrate at its start —
    /// the layers' documents, the lives, and the retirement of what left the
    /// record. `false` — `--no-ingest`, or `ingest: false` in `motherd.yaml` —
    /// leaves that to another member of the cluster: this one reads the mind
    /// they wrote, as it replicates.
    #[arg(long = "no-ingest", action = ArgAction::SetFalse)]
    pub ingest: bool,

    /// From how many invalid answers (`401`/`403`/`429`) from Discord in
    /// the last ten minutes every request to it is spaced out
    /// (`discord::limit`). Discord bans the public address past 10,000, for
    /// every member behind it, so this is the address's line: each npcd
    /// member of the group holds an even share of it.
    #[arg(long, value_name = "COUNT", default_value_t = limit::THROTTLE)]
    pub discord_invalid_throttle: u32,

    /// At how many invalid answers in the last ten minutes nothing more is
    /// sent to Discord until the window rolls on: the address's line, shared
    /// evenly among the group's npcd members.
    #[arg(long, value_name = "COUNT", default_value_t = limit::CEILING)]
    pub discord_invalid_ceiling: u32,
}

/// `--mind-branch`'s default.
const DEFAULT_MIND_BRANCH: &str = "main";

/// `--tombstone-retention`'s default, in days: the substrate's own.
const DEFAULT_TOMBSTONE_RETENTION_DAYS: u64 =
    DEFAULT_TOMBSTONE_RETENTION.as_secs() / SECONDS_PER_DAY;

const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

impl NpcdArgs {
    /// The flags that act once, at the start they are given to. A self-heal
    /// relaunch drops them (`host::OneShot`), and `motherd.yaml` cannot set
    /// them — a config file is read at every start, so one there would forget
    /// at every start. Run the `npcd` binary with the flag once instead.
    pub const ONE_SHOT: &'static [OneShot] = &[
        OneShot {
            flag: "--forget-conversations",
            takes_value: false,
        },
        OneShot {
            flag: "--forget-dreams",
            takes_value: false,
        },
        OneShot {
            flag: "--forget-journal",
            takes_value: false,
        },
        OneShot {
            flag: "--wipe-conversations",
            takes_value: false,
        },
    ];

    /// What npcd asks of the engine it shares, without loading npcd: what a
    /// host running zend alone seats in npcd's place, so zend's engine is the
    /// one built beside npcd. Its guests stay out — co-resident models only a
    /// running npcd calls.
    pub fn engine_requests(&self) -> CoreRequests {
        engine_asks(MODEL.builder()).take_core_requests()
    }
}

impl Default for NpcdArgs {
    fn default() -> Self {
        Self {
            data: None,
            mind_repo: None,
            mind_branch: DEFAULT_MIND_BRANCH.to_string(),
            journals_dir: None,
            log_identity: false,
            forget_conversations: false,
            forget_dreams: false,
            forget_journal: false,
            wipe_conversations: false,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION_DAYS,
            characters: true,
            host_worlds: false,
            ingest: true,
            discord_invalid_throttle: limit::THROTTLE,
            discord_invalid_ceiling: limit::CEILING,
        }
    }
}

impl NpcdArgs {
    /// npcd, as a tenant: the schema resolved and the data directory chosen.
    /// Runs before logging is up.
    ///
    /// The schema first, because a mistyped `--mind` should be refused on the
    /// spot, not after the substrate has been opened and half the daemon stood
    /// up. Fatal rather than a fallback: a daemon that quietly reverted to the
    /// bundled placeholder would run with none of the named mind's content.
    pub fn resolve(self) -> anyhow::Result<NpcdTenant> {
        // Absolute from here on: the substrate's volume is measured from its
        // path, and a mirror's repository is opened by its path, so neither
        // may depend on where it is read from.
        let explicit = self
            .data
            .as_deref()
            .map(absolute)
            .transpose()
            .context("--data")?;
        // A mind repository is read from the substrate (`docs/substrate_git.md`
        // §8): its schema once the substrate is open, at start; until then the
        // bundled one stands in.
        let repo = match &self.mind_repo {
            Some(name) => {
                if explicit.is_none() {
                    anyhow::bail!("--mind-repo keeps its store under --data: name one");
                }
                let bad =
                    name.is_empty() || name.contains(['/', '\\', ':']) || name.starts_with('.');
                if bad {
                    anyhow::bail!("--mind-repo `{name}` is not a repository's name");
                }
                Some(MindSource {
                    name: name.clone(),
                    branch: self.mind_branch.clone(),
                })
            }
            None => None,
        };
        let schema = projection::Source::bundled();
        // **The compiled-in default is a path on the machine that built
        // this.** Used when it is really there; the working directory stands in
        // when it is not, and the choice is logged at start.
        let (data, fell_back) = match explicit {
            Some(dir) => (absolute(&dir).context("the data directory")?, false),
            None => {
                let built_at = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                if built_at.is_dir() {
                    (built_at, false)
                } else {
                    (current_dir()?, true)
                }
            }
        };
        Ok(NpcdTenant {
            args: self,
            schema,
            data,
            fell_back,
            repo,
        })
    }
}

/// npcd, configured and not yet started.
pub struct NpcdTenant {
    args: NpcdArgs,
    schema: projection::Source,
    data: PathBuf,
    /// The compiled-in data directory was not on this machine, and the working
    /// directory stands in.
    fell_back: bool,
    /// The mind repository the agents hold, with `--mind-repo`.
    repo: Option<MindSource>,
}

impl Tenant for NpcdTenant {
    fn name(&self) -> &'static str {
        NAME
    }

    fn model(&self) -> Model {
        MODEL
    }

    fn requires(&self) -> Capabilities {
        requires()
    }

    fn substrate_dir(&self) -> PathBuf {
        self.data.join(SUBSTRATE_DIR)
    }

    fn ui(&self) -> TenantUi {
        TenantUi {
            hosts: vec![HOST.into(), "npcd.localhost".into()],
            content: &CONSOLE,
            default: false,
        }
    }

    fn start(self: Box<Self>, ctx: TenantContext) -> BoxFuture<'static, anyhow::Result<Running>> {
        Box::pin(start(*self, ctx))
    }
}

mod routes_and_sinks;
mod shutdown_upkeep;
mod tenant_start;
#[cfg(test)]
mod tests_flags;
#[cfg(test)]
mod tests_resolve;
mod worlds_and_identities;

pub use routes_and_sinks::*;
use shutdown_upkeep::*;
use tenant_start::*;
use worlds_and_identities::*;
