//! The aggregator's configuration, `mcp/mcp.yaml` (`docs/mcp_app.md` §5).

use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::Context;
use mcp_server::Identity;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Config {
    /// Where it listens.
    pub bind: String,
    /// The peers whose `X-Tokera-*` identity is believed.
    pub trusted_gateways: Vec<IpAddr>,
    /// The gateway entrance every backend is reached through: `http://<host>`.
    pub gateway: String,
    /// Where MCP sessions are kept.
    pub state: PathBuf,
    /// Who the aggregated server says it is.
    pub server: Card,
    /// The MCP servers it serves the tools of, in the order their tools are
    /// listed.
    pub backends: Vec<BackendConfig>,
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

/// One MCP server whose tools are served: the site its `/v1/mcp` is reached
/// at through the gateway, named in `Host`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BackendConfig {
    pub name: String,
    pub host: String,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: Self =
            serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if !config.gateway.starts_with("http://") && !config.gateway.starts_with("https://") {
            anyhow::bail!("`gateway` is an http(s) URL: {}", config.gateway);
        }
        if config.backends.is_empty() {
            anyhow::bail!("`backends` names no server to serve the tools of");
        }
        Ok(config)
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

    #[test]
    fn the_shipped_config_reads() {
        let c = Config::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/mcp.yaml"))).unwrap();
        assert_eq!(c.bind, "0.0.0.0:8095");
        assert_eq!(
            c.trusted_gateways,
            [
                "192.168.0.5".parse::<IpAddr>().unwrap(),
                "192.168.0.4".parse().unwrap()
            ]
        );
        assert_eq!(
            c.backends,
            [
                BackendConfig {
                    name: "zend".into(),
                    host: "code.tokera.com".into()
                },
                BackendConfig {
                    name: "npcd".into(),
                    host: "bot.tokera.com".into()
                }
            ]
        );
        let card = c.identity();
        assert_eq!((card.name.as_str(), card.title.as_str()), ("zen", "Zen"));
        assert_eq!(card.icon.as_deref(), Some("/brand/zen.png"));
    }
}
