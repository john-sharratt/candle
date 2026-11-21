//! The `cluster:` block of the configuration: the keys, services and pushes that
//! make this gateway the front of a replicated cluster.
//!
//! Every secret here is a file, resolved against the config, so the table can be
//! read and committed without one. The group key is what members prove to join.
//! The status key lets a caller read the status page without a session, and the
//! ops key lets a caller act as admin without one. Service keys let a service on
//! this machine act as itself on the routes and topics it names. Certification's
//! grace period and live share are set here. The machine table every daemon
//! compiles in is read here too, so the gateway places work by the same table the
//! members size themselves by. A table that is missing or does not parse is fatal,
//! since a gateway without one gives every machine no work.

use super::*;

/// The `cluster:` block's `ops_key`.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpsKeyConfig {
    /// The key's file, resolved against this config.
    pub file: PathBuf,
    /// The admin it acts as, by the email the role table names them by: what
    /// this gateway's role check sees and what it names to the member a
    /// command goes to, so both must list them as an admin.
    pub admin: String,
}

/// One of the `cluster:` block's `service_keys` (`docs/mcp_file_tools.md` §4).
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ServiceKeyConfig {
    /// The service: its identity's subject is `service:<name>`.
    pub name: String,
    /// The key's file, resolved against this config.
    pub key_file: PathBuf,
    /// The paths it may call, each with everything under it.
    pub routes: Vec<String>,
    /// The bus topics it may use.
    #[serde(default)]
    pub topics: ServiceTopics,
}

/// A service key's bus topics; a trailing `*` admits everything under it.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ServiceTopics {
    #[serde(default)]
    pub subscribe: Vec<String>,
    #[serde(default)]
    pub publish: Vec<String>,
}

/// `cluster.deploy_on_push`: the repositories whose `main` is deployed as it
/// moves, told by GitHub's webhook and by a watch of origin
/// (`docs/battlecities_app.md` §4).
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeployOnPush {
    /// The repositories, by the name the workspace knows them: `battle-cities`.
    pub repos: Vec<String>,
    /// The file that holds the webhook's secret, resolved against this
    /// config. Absent, or naming a file that is not there, no delivery is
    /// believed and the watch alone tells of a push.
    #[serde(default)]
    pub github_hook_key_file: Option<PathBuf>,
}

/// One entry of `cluster.udp_forward`: the ports whose datagrams go to the
/// members of `app`.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UdpForward {
    pub app: String,
    pub ports: Vec<UdpPort>,
}

/// One forwarded port: where the gateway listens, and the port each member
/// listens on. They differ where a member runs on the gateway's own machine,
/// which cannot bind one port twice.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UdpPort {
    pub listen: u16,
    pub member: u16,
}

/// The `cluster:` block.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Cluster {
    /// The group key's file (§13.1), resolved against this config. A file,
    /// never an inline value, like every other secret here.
    pub group_key_file: PathBuf,
    /// The status key's file (§12), resolved against this config: a caller
    /// presenting it reads `/gateway/status.json` without a session. Absent,
    /// only an admin's session reads it. `web --new-status-key` writes one.
    #[serde(default)]
    pub status_key_file: Option<PathBuf>,
    /// The ops key (§12): a caller presenting it acts as `admin` on every
    /// route of the gateway's own — its page and commands, and every upgrade
    /// command, an emergency `rollout` back included — without a session.
    /// Absent, those take an admin's session. `web --new-ops-key` writes one.
    #[serde(default)]
    pub ops_key: Option<OpsKeyConfig>,
    /// The services on this gateway's machine that act as themselves
    /// (`docs/mcp_file_tools.md` §4), each on the routes and topics it names.
    #[serde(default)]
    pub service_keys: Vec<ServiceKeyConfig>,
    /// The pushes that are deployed at once (`docs/battlecities_app.md` §4):
    /// the repositories, and the secret GitHub signs its webhook with.
    #[serde(default)]
    pub deploy_on_push: Option<DeployOnPush>,
    /// The UDP ports forwarded to an app's members (`docs/battlecities_app.md`
    /// §5).
    #[serde(default)]
    pub udp_forward: Vec<UdpForward>,
    /// This gateway's name in the coordination log; default its bind address.
    #[serde(default)]
    pub name: Option<String>,
    /// How long, in minutes, a newer build is certified before the floor may
    /// rise to it (`docs/replicated_substrate.md` §3.5).
    #[serde(default = "floor_grace_minutes")]
    pub floor_grace_minutes: u64,
    /// The candidate's share of live traffic while it certifies, in percent
    /// (§3.8).
    #[serde(default = "canary_share")]
    pub canary_share: u64,
    /// The other gateways answering the same public names, each dialled and
    /// kept linked (§13.4).
    #[serde(default)]
    pub peers: Vec<Peer>,
    /// The machines whose agents stand by (§11.3): given a job only when no
    /// other agent can run it, or when the job names them.
    #[serde(default)]
    pub standby_agents: Vec<String>,
    /// The machine table, `host/machines.yaml` — the one file every daemon
    /// compiles in (`host::machines::MACHINES`), so the gateway places work by
    /// the same table the members size themselves by
    /// (`docs/static_placement.md` §2). Resolved against this config and read
    /// into [`machines`](Self::machines); named but missing or unparseable is
    /// fatal, since a gateway with no table gives every machine no work.
    pub machines_file: PathBuf,
    /// What each machine takes, from [`machines_file`](Self::machines_file).
    #[serde(skip)]
    pub machines: MachineTable,
}

/// One peer gateway in `cluster.peers`.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Peer {
    /// Its TLS entrance, `https://<host>[:port]`.
    pub url: String,
    /// The name its certificate is verified against, when `url` names an
    /// address rather than a name the certificate covers.
    #[serde(default)]
    pub name: Option<String>,
}

pub(super) fn floor_grace_minutes() -> u64 {
    GRACE_DEFAULT.as_secs() / 60
}

pub(super) fn canary_share() -> u64 {
    CANARY_DEFAULT
}
