//! `host` — the process a daemon runs in.
//!
//! One model, one engine, one HTTP server, and the daemons beside each other as
//! tenants (`docs/shared_model_host.md`). `zend`, `npcd` and `tokerad` are each
//! `host::run([tenant, …])`: the host owns what is the process's — the runtime,
//! logging, the panic hook, signals, the prescribed model, the shared
//! endpoints, the web UIs, shutdown — and each tenant keeps its own work.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`run`] | the one startup path, and the exit |
//! | [`tenant`] | the contract a daemon implements |
//! | [`server`] | one `web::Builder`: the shared and tenant APIs, the UIs by host name |
//! | [`cluster`] | with `--cluster`, each tenant's app as a cluster member |
//! | [`endpoints`] | `/v1/status`, `/v1/me`, `/v1/telemetry`, `/ws/logs`, … served once |
//! | [`commands`] | the maintenance commands and their jobs |
//! | [`stopping`] | how the process comes to stop: Ctrl-C, `shutdown`, `restart` |
//! | [`guard`] | routes that cannot forget their role, and the path-overlap refusal |
//! | [`access`] | the estate's role table and `--local-signin` |
//! | [`ui`] | UIs from disk, `/common/`, the on-disk build id |
//! | [`model`] | the one model every tenant runs |
//! | [`logs`], [`telemetry`], [`loading`] | what the process says about itself |
//! | [`process`], [`ctrl_c`], [`self_heal`], [`log_file`] | process-wide concerns |

pub mod access;
pub mod cli;
pub mod cluster;
pub mod commands;
pub mod ctrl_c;
pub mod download;
pub mod endpoints;
pub mod guard;
pub mod loading;
pub mod log_file;
pub mod logs;
pub mod model;
pub mod process;
pub mod run;
pub mod self_heal;
pub mod server;
pub mod stopping;
pub mod telemetry;
pub mod tenant;
pub mod ui;

pub use cli::HostArgs;
pub use model::MODEL;
pub use run::{run, HostConfig};
pub use self_heal::OneShot;
pub use server::Listen;
pub use tenant::{
    CheckProgress, Maintenance, MeShare, Reports, Running, Stop, Tenant, TenantContext,
    TenantStatus, TenantUi,
};
