//! The aggregator's configuration (`docs/mcp_app.md` §5): one file per
//! machine — `mcp/mcp.yaml` on BlackWorld, `mcp/mcp.whiteworld.yaml` on
//! WhiteWorld — each naming the one `served` file every machine serves the
//! same card and backends from.

use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::Context;
use mcp_server::Identity;
use serde::de::DeserializeOwned;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Where it listens: the address the gateway's route names as its
    /// upstream, which serves while no member is joined.
    pub bind: String,
    /// The peers on that listener whose `X-Tokera-*` identity is believed.
    pub trusted_gateways: Vec<IpAddr>,
    /// The gateway entrance every backend is reached through: `http://<host>`.
    pub gateway: String,
    /// The cluster it joins as the app `mcp`.
    pub cluster: ClusterJoin,
    /// Where its node is kept, so it is the same member across restarts.
    pub state: PathBuf,
    /// Who the aggregated server says it is.
    pub server: Card,
    /// The MCP servers it serves the tools of, in the order their tools are
    /// listed.
    pub backends: Vec<BackendConfig>,
}

/// What one machine's file says.
#[derive(Debug, Deserialize)]
struct Machine {
    bind: String,
    trusted_gateways: Vec<IpAddr>,
    gateway: String,
    cluster: ClusterJoin,
    state: PathBuf,
    /// The card and backends, beside this file.
    served: PathBuf,
}

/// What every machine serves alike.
#[derive(Debug, Deserialize)]
struct Served {
    server: Card,
    backends: Vec<BackendConfig>,
}

/// The gateway it joins, as motherd names one.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ClusterJoin {
    /// The gateway's socket URLs, space-separated, best first.
    pub urls: String,
    /// The name its certificate is verified against.
    pub name: String,
    /// The group key every member proves.
    pub group_key: PathBuf,
    /// This machine's name on the status page.
    pub host: String,
}

/// The server card a client shows.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Card {
    pub name: String,
    pub title: String,
    pub description: String,
    pub instructions: String,
    /// Where its icon is served, under the origin a request reached it at.
    #[serde(default)]
    pub icon: Option<String>,
}

/// One MCP server whose tools are served: the site it is reached at through
/// the gateway, named in `Host`, and its path there — each app's own, since
/// one host may carry several apps whose routes must not meet.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BackendConfig {
    pub name: String,
    pub host: String,
    pub path: String,
}

fn read<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

impl Config {
    /// The machine's file at `path`, and the `served` file it names, read
    /// from beside it.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let machine: Machine = read(path)?;
        if !machine.gateway.starts_with("http://") && !machine.gateway.starts_with("https://") {
            anyhow::bail!("`gateway` is an http(s) URL: {}", machine.gateway);
        }
        let beside = path.parent().unwrap_or(Path::new(""));
        let served: Served = read(&beside.join(&machine.served))?;
        if served.backends.is_empty() {
            anyhow::bail!("`backends` names no server to serve the tools of");
        }
        Ok(Self {
            bind: machine.bind,
            trusted_gateways: machine.trusted_gateways,
            gateway: machine.gateway,
            cluster: machine.cluster,
            state: machine.state,
            server: served.server,
            backends: served.backends,
        })
    }

    /// The card, as the server says it.
    pub fn identity(&self) -> Identity {
        Identity {
            name: self.server.name.clone(),
            title: self.server.title.clone(),
            description: self.server.description.clone(),
            instructions: self.server.instructions.clone(),
            icon: self.server.icon.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped(file: &str) -> Config {
        Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(file)).unwrap()
    }

    fn served_alike(c: &Config) {
        assert_eq!(
            c.backends,
            [
                BackendConfig {
                    name: "coding".into(),
                    host: "code.tokera.com".into(),
                    path: "/v1/mcp".into()
                },
                BackendConfig {
                    name: "bot".into(),
                    host: "bot.tokera.com".into(),
                    path: "/v1/npc/mcp".into()
                }
            ]
        );
        let card = c.identity();
        assert_eq!((card.name.as_str(), card.title.as_str()), ("zen", "Zen"));
        assert_eq!(card.icon.as_deref(), Some("/brand/zen.png"));
    }

    /// BlackWorld's aggregator sits beside the gateway: it listens and calls
    /// back on loopback, and joins as `blackworld`.
    #[test]
    fn blackworlds_config_reads() {
        let c = shipped("mcp.yaml");
        assert_eq!(c.bind, "127.0.0.1:8095");
        assert_eq!(c.trusted_gateways, ["127.0.0.1".parse::<IpAddr>().unwrap()]);
        assert_eq!(c.gateway, "http://127.0.0.1");
        assert_eq!(
            c.cluster,
            ClusterJoin {
                urls: "https://gateway.tokera.local:9443 https://gateway.tokera.com:9443".into(),
                name: "tokera.com".into(),
                group_key: "web/secrets/group.key".into(),
                host: "blackworld".into(),
            }
        );
        assert_eq!(c.state, Path::new("mcp/state"));
        served_alike(&c);
    }

    /// WhiteWorld's calls the gateway across the LAN, and joins as
    /// `whiteworld`; it serves the same card and backends.
    #[test]
    fn whiteworlds_config_reads() {
        let c = shipped("mcp.whiteworld.yaml");
        assert_eq!(c.gateway, "http://192.168.0.5");
        assert_eq!(c.cluster.host, "whiteworld");
        assert_eq!(c.cluster.name, "tokera.com");
        served_alike(&c);
    }

    #[test]
    fn a_gateway_that_is_not_a_url_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("served.yaml"),
            "server: {name: n, title: t, description: d, instructions: i}\n\
             backends: [{name: b, host: h, path: /p}]\n",
        )
        .unwrap();
        let machine = |gateway: &str| {
            format!(
                "bind: 127.0.0.1:1\ntrusted_gateways: []\ngateway: \"{gateway}\"\n\
                 cluster: {{urls: \"https://g:1\", name: g, group_key: k, host: h}}\n\
                 state: s\nserved: served.yaml\n"
            )
        };
        let path = dir.path().join("m.yaml");
        fs::write(&path, machine("192.168.0.5")).unwrap();
        let err = Config::load(&path).unwrap_err().to_string();
        assert_eq!(err, "`gateway` is an http(s) URL: 192.168.0.5");
        fs::write(&path, machine("http://g")).unwrap();
        assert_eq!(Config::load(&path).unwrap().backends.len(), 1);
    }
}
