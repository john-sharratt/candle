//! The process's own settings — where it listens and whom it believes — as
//! command-line flags (`zend`) and as the `host:` section of `motherd.yaml`. One
//! struct for both, so a setting means the same thing whichever binary reads
//! it.

use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgAction, Args};
use cluster::upgrade::kept::KEEP_DEFAULT;
use serde::Deserialize;

use crate::cluster::ClusterSettings;
use crate::expert_split::ExpertSplitArgs;
use crate::server::Listen;

/// The process's flags.
#[derive(Args, Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct HostArgs {
    /// Address to bind the HTTP server to. Loopback by default — reachable from
    /// this machine alone. Pass a LAN address or `0.0.0.0` to serve the
    /// network; identity is still believed only from loopback, this address
    /// and each `--gateway`.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Also serve HTTPS — HTTP/3 over UDP, HTTP/2 and HTTP/1.1 over TCP — on
    /// this port of the same address, with a self-signed certificate made at
    /// startup. For the hop from the `web` gateway (`self_signed: true` on its
    /// route); a browser will not trust the certificate.
    #[arg(long)]
    pub tls_port: Option<u16>,

    /// Address of a gateway whose `x-tokera-*` identity headers are believed,
    /// in addition to loopback and the `--host` address (repeatable). Identity
    /// from any other peer is cleared, and the caller is anonymous.
    #[arg(long = "gateway", value_name = "IP")]
    pub gateways: Vec<IpAddr>,

    /// Recognize a loopback caller that sends no `x-tokera-*` headers as this
    /// email, resolved against the estate's role table — for running a daemon
    /// directly, with no gateway in front of it, and still being seen as an
    /// admin or creator on this machine. Never applies to a request that
    /// carries a forwarded identity, or to a peer that is not genuine loopback.
    /// Refused together with `--gateway`: a daemon with a gateway in front of
    /// it has one to sign people in. Off by default: any other local process or
    /// account on this machine — and anything that relays remote traffic onto
    /// loopback, such as a tunnel or a proxy on this box — is also recognized as
    /// this email while it is set.
    #[arg(long, value_name = "EMAIL")]
    pub local_signin: Option<String>,

    /// A gateway to join as a cluster member (`docs/replicated_substrate.md`
    /// §3.2): `https://code.tokera.com`, or `https://192.168.0.5:9443` with
    /// `--cluster-name` — or several URLs for the one gateway, space-separated
    /// and best first, which each socket fails over between:
    /// `"https://gateway.tokera.local:9443 https://gateway.tokera.com:9443"`.
    /// Repeatable, one per gateway. Each tenant dials each one, and while
    /// joined its requests are routed by the cluster. Absent, the daemon runs
    /// standalone and opens no socket.
    #[arg(long = "cluster", value_name = "URL")]
    pub clusters: Vec<String>,

    /// The name a gateway's certificate is verified against when a `--cluster`
    /// names an address; by default the URL's host.
    #[arg(long, value_name = "HOST")]
    pub cluster_name: Option<String>,

    /// The group key file (64 hex digits) a member proves it holds. Required
    /// with `--cluster`.
    #[arg(long, value_name = "FILE")]
    pub group_key: Option<PathBuf>,

    /// How long, in minutes, a conversation's claim lives on this member
    /// without serving a request before it is released (§5.4).
    #[arg(long, value_name = "MINUTES", default_value_t = 10)]
    pub cluster_affinity_idle: u64,

    /// Write the group's `max_members` as `N` at startup, lowering the base of
    /// its quorum (§5.9): for retiring machines for good, or knowingly running
    /// with less safety.
    #[arg(long, value_name = "N")]
    pub cluster_quorum_max: Option<u32>,

    /// Also join the cluster's agents, running build and git jobs on the
    /// checkouts this agent file names (§11.5) — and, cut off from every
    /// cluster, running this process's own. Needs `--cluster`.
    #[arg(long, value_name = "FILE")]
    pub agent: Option<PathBuf>,

    /// How many previous builds to keep beside the executable, in `builds/`,
    /// for rollback (§3.9).
    #[arg(long, value_name = "N", default_value_t = KEEP_DEFAULT)]
    pub keep_builds: usize,

    /// The build an install restored this process from, `count-hash`: it is
    /// never tried again (§3.9). `keep` sets it when it restores this
    /// process; a relaunch drops it, and the config file cannot set it.
    #[arg(long, value_name = "COUNT-HASH")]
    #[serde(skip)]
    pub upgrade_failed: Option<String>,

    /// Wait until this pid has exited before loading. The predecessor sets it
    /// when it starts the successor (§3.9). A relaunch drops it, and the
    /// config file cannot set it.
    #[arg(long, value_name = "PID")]
    #[serde(skip)]
    pub after_pid: Option<u32>,

    /// The CPU expert split (`expert_split:` in the file).
    #[command(flatten)]
    pub expert_split: ExpertSplitArgs,

    /// Increase log verbosity. -v = DEBUG, -vv = TRACE.
    #[arg(short, long, action = ArgAction::Count)]
    pub verbose: u8,
}

impl Default for HostArgs {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            tls_port: None,
            gateways: Vec::new(),
            local_signin: None,
            clusters: Vec::new(),
            cluster_name: None,
            group_key: None,
            cluster_affinity_idle: 10,
            cluster_quorum_max: None,
            agent: None,
            keep_builds: KEEP_DEFAULT,
            upgrade_failed: None,
            after_pid: None,
            expert_split: ExpertSplitArgs::default(),
            verbose: 0,
        }
    }
}

impl HostArgs {
    /// Where to listen, the address parsed. `--local-signin` beside a
    /// `--gateway` is refused.
    pub fn listen(&self) -> anyhow::Result<Listen> {
        if self.local_signin.is_some() && !self.gateways.is_empty() {
            anyhow::bail!(
                "--local-signin stands in for a gateway that is not there, and --gateway names \
                 one that is: a caller the gateway forwards without an identity would be signed \
                 in as the local email. Use one or the other"
            );
        }
        let bind: IpAddr = match self.host.as_str() {
            "localhost" => IpAddr::from([127, 0, 0, 1]),
            h => h.parse().map_err(|_| {
                anyhow::anyhow!(
                    "invalid --host {h:?}: expected an IP address, e.g. 127.0.0.1 or 0.0.0.0"
                )
            })?,
        };
        Ok(Listen {
            bind,
            port: self.port,
            tls_port: self.tls_port,
            gateways: self.gateways.clone(),
        })
    }

    /// The cluster this process joins, if any `--cluster` is given; its
    /// machine name is `host`.
    pub fn cluster(&self, host: String) -> anyhow::Result<Option<ClusterSettings>> {
        ClusterSettings::parse(
            &self.clusters,
            self.cluster_name.as_deref(),
            self.group_key.as_deref(),
            Duration::from_secs(self.cluster_affinity_idle * 60),
            self.cluster_quorum_max,
            self.agent.as_deref(),
            host,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        host: HostArgs,
    }

    /// The flags and the YAML read the same settings, with the same defaults.
    #[test]
    fn flags_and_yaml_agree() {
        let flags = Cli::try_parse_from([
            "x",
            "--host",
            "192.168.0.5",
            "--port",
            "8081",
            "--tls-port",
            "8444",
            "--gateway",
            "10.0.0.9",
            "-v",
        ])
        .unwrap()
        .host;
        let yaml: HostArgs = serde_yaml::from_str(
            "host: 192.168.0.5\nport: 8081\ntls_port: 8444\ngateways: [10.0.0.9]\nverbose: 1\n",
        )
        .unwrap();
        assert_eq!(flags, yaml);
        assert_eq!(
            Cli::try_parse_from(["x"]).unwrap().host,
            HostArgs::default()
        );
        assert_eq!(
            serde_yaml::from_str::<HostArgs>("{}").unwrap(),
            HostArgs::default()
        );
        assert!(serde_yaml::from_str::<HostArgs>("bind: 1.2.3.4").is_err());
    }

    /// The split's settings are a section of their own under `host:`, and the
    /// same flags beside the process's.
    #[test]
    fn the_expert_split_is_a_section() {
        let yaml: HostArgs =
            serde_yaml::from_str("port: 8081\nexpert_split:\n  enabled: true\n  cpu_scale: 1.5\n")
                .unwrap();
        let flags = Cli::try_parse_from([
            "x",
            "--port",
            "8081",
            "--expert-split",
            "--expert-split-cpu-scale",
            "1.5",
        ])
        .unwrap()
        .host;
        assert_eq!(flags, yaml);
        assert!(yaml.expert_split.enabled);
        assert!(serde_yaml::from_str::<HostArgs>("expert_split:\n  bogus: 1\n").is_err());
    }

    #[test]
    fn localhost_is_loopback_and_a_name_is_refused() {
        let named = |host: &str| HostArgs {
            host: host.into(),
            ..HostArgs::default()
        };
        assert_eq!(
            named("localhost").listen().unwrap().bind,
            IpAddr::from([127, 0, 0, 1])
        );
        assert!(named("zend.local").listen().is_err());
    }

    /// The cluster flags read the same from the command line and from YAML; a
    /// cluster needs its key and a key needs a cluster; each `--cluster`
    /// becomes a target verified by `--cluster-name`.
    #[test]
    fn cluster_flags_parse_and_need_their_key() {
        let tmp = tempfile::TempDir::new().unwrap();
        let key = tmp.path().join("group.key");
        std::fs::write(&key, "ab".repeat(32)).unwrap();
        let key_arg = key.to_string_lossy().to_string();
        let flags = Cli::try_parse_from([
            "x",
            "--cluster",
            "https://code.tokera.com",
            "--cluster",
            "https://192.168.0.5:9443",
            "--cluster-name",
            "code.tokera.com",
            "--group-key",
            &key_arg,
            "--cluster-affinity-idle",
            "5",
            "--cluster-quorum-max",
            "2",
        ])
        .unwrap()
        .host;
        let yaml: HostArgs = serde_yaml::from_str(&format!(
            "clusters: [\"https://code.tokera.com\", \"https://192.168.0.5:9443\"]\n\
             cluster_name: code.tokera.com\ngroup_key: '{key_arg}'\ncluster_affinity_idle: 5\n\
             cluster_quorum_max: 2\n"
        ))
        .unwrap();
        assert_eq!(flags, yaml);
        let settings = flags.cluster("m5".into()).unwrap().expect("a cluster");
        assert_eq!(settings.targets.len(), 2);
        assert_eq!(
            settings.targets[1].endpoints[0].tls_name.as_deref(),
            Some("code.tokera.com")
        );
        assert_eq!(settings.affinity_idle, Duration::from_secs(300));
        assert_eq!(settings.quorum_max, Some(2));
        assert_eq!(settings.host, "m5");

        assert!(HostArgs::default().cluster("m".into()).unwrap().is_none());
        let keyless = HostArgs {
            clusters: vec!["https://code.tokera.com".into()],
            ..HostArgs::default()
        };
        let err = keyless.cluster("m".into()).unwrap_err().to_string();
        assert!(err.contains("--group-key"), "{err}");
        let clusterless = HostArgs {
            group_key: Some(key),
            ..HostArgs::default()
        };
        assert!(clusterless.cluster("m".into()).is_err());
    }

    /// One `clusters:` entry naming several URLs is one gateway reached by
    /// each, best first — not one cluster per URL.
    #[test]
    fn a_cluster_of_several_urls_is_one_gateway() {
        let tmp = tempfile::TempDir::new().unwrap();
        let key = tmp.path().join("group.key");
        std::fs::write(&key, "ab".repeat(32)).unwrap();
        let yaml: HostArgs = serde_yaml::from_str(&format!(
            "clusters: [\"https://gateway.tokera.local:9443 https://gateway.tokera.com:9443\"]\n\
             cluster_name: tokera.com\ngroup_key: '{}'\n",
            key.to_string_lossy()
        ))
        .unwrap();
        let settings = yaml.cluster("m".into()).unwrap().expect("a cluster");
        assert_eq!(settings.targets.len(), 1);
        let hosts: Vec<&str> = settings.targets[0]
            .endpoints
            .iter()
            .map(|e| e.host.as_str())
            .collect();
        assert_eq!(hosts, ["gateway.tokera.local", "gateway.tokera.com"]);
    }

    /// `--keep-builds` reads the same from the flag and the YAML, and
    /// defaults to 5; `--upgrade-failed` is a flag only — a file read at every
    /// start cannot name a build that failed once.
    #[test]
    fn the_upgrade_flags_parse() {
        let flags = Cli::try_parse_from([
            "x",
            "--keep-builds",
            "3",
            "--upgrade-failed",
            "4815-9f8e7d6",
        ])
        .unwrap()
        .host;
        assert_eq!(flags.keep_builds, 3);
        assert_eq!(flags.upgrade_failed.as_deref(), Some("4815-9f8e7d6"));
        let yaml: HostArgs = serde_yaml::from_str("keep_builds: 3\n").unwrap();
        assert_eq!(yaml.keep_builds, 3);
        assert_eq!(HostArgs::default().keep_builds, 5);
        assert!(serde_yaml::from_str::<HostArgs>("upgrade_failed: 1-a\n").is_err());
    }

    /// `--local-signin` is for a daemon with no gateway; beside one it is
    /// refused, and alone it is accepted.
    #[test]
    fn local_signin_with_a_gateway_is_refused() {
        let args = |gateways: Vec<IpAddr>| HostArgs {
            gateways,
            local_signin: Some("me@example.com".into()),
            ..HostArgs::default()
        };
        assert!(args(Vec::new()).listen().is_ok());
        let err = args(vec![IpAddr::from([192, 168, 0, 5])])
            .listen()
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("--local-signin stands in"), "{err}");
    }
}
