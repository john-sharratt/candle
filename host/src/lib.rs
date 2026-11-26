//! `host` — the process a daemon runs in.
//!
//! One HTTP server, one engine per model the tenants name, and the daemons
//! beside each other as tenants (`docs/shared_model_host.md`,
//! `docs/parasite_model.md`). `zend`, `npcd` and `motherd` are each
//! `host::run([tenant, …])`: the host owns what is the process's — the runtime,
//! logging, the panic hook, signals, the engines, the shared endpoints, the web
//! UIs, shutdown — and each tenant keeps its own work and names its own model.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`run`] | the one startup path, and the exit |
//! | [`slots`] | one engine slot per model the tenants name |
//! | [`tenant`] | the contract a daemon implements |
//! | [`server`] | one `web::Builder`: the shared and tenant APIs, the UIs by host name |
//! | [`cluster`] | with `--cluster`, each tenant's app as a cluster member |
//! | [`endpoints`] | `/v1/status`, `/v1/me`, `/v1/telemetry`, `/ws/logs`, … served once |
//! | [`commands`] | the maintenance commands and their jobs |
//! | [`stopping`] | how the process comes to stop: Ctrl-C, `shutdown`, `restart` |
//! | [`guard`] | routes that cannot forget their role, and the path-overlap refusal |
//! | [`access`] | the estate's role table and `--local-signin` |
//! | [`ui`] | UIs from disk, `/common/`, the on-disk build id |
//! | [`substrate_views`] | the substrate inspector's views, served by each tenant over its own substrate |
//! | [`model`] | a model as the host reports it |
//! | [`expert_split`] | the CPU expert split's settings, applied before the engine loads |
//! | [`build_info`] | `--build-info`, and what every binary does first |
//! | [`calibration`] | the load phase that writes each tool's exemplars, run by every tenant that offers tools |
//! | [`logs`], [`telemetry`], [`loading`] | what the process says about itself |
//! | [`memory_census`] | what the process holds, read once a minute, judged for growth |
//! | [`process`], [`ctrl_c`], [`self_heal`] | process-wide concerns (the log file is `web::log_file`, which the gateway writes too) |

pub mod access;
pub mod agent_hold;
pub mod blocking_canary;
pub mod boot;
pub mod build_info;
pub mod calibration;
pub mod cli;
pub mod cluster;
pub mod commands;
pub mod ctrl_c;
pub mod download;
pub mod endpoints;
pub mod expert_split;
pub mod guard;
pub mod keep_stop;
pub mod loading;
pub mod logs;
pub mod machines;
pub mod memory_census;
pub mod model;
pub mod process;
pub mod ram_census;
pub mod run;
pub mod self_heal;
pub mod server;
pub mod slots;
pub mod starting;
pub mod stopping;
pub mod substrate_views;
pub mod syncing;
pub mod telemetry;
pub mod tenant;
pub mod ui;

pub use ::cluster::launch::OneShot;
pub use cli::HostArgs;
pub use run::{run, run_seated, HostConfig, Seat};
pub use server::Listen;
pub use tenant::{
    CheckProgress, Maintenance, MeShare, Reports, Running, Stop, Tenant, TenantContext,
    TenantStatus, TenantUi,
};
