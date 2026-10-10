//! `zend` — Zen Code daemon.
//!
//! Run from a workspace folder — one holding a `workspace.yaml` that lists the
//! repositories (folders beside it) in scope:
//!
//! ```text
//! zend                        # workspace = cwd, port 8080
//! zend /path/to/workspace     # explicit workspace path
//! zend --port 9090            # custom port
//! ```
//!
//! Continue configuration:
//! ```json
//! { "provider": "openai", "apiBase": "http://localhost:8080", "model": "zen-code" }
//! ```
//!
//! The binary parses zend's flags and the process's, and runs zend as the one
//! tenant of a host process (`host::run`). `tokerad` is the same call with npcd
//! beside it.

use clap::Parser;
use host::{run, HostArgs, HostConfig, Tenant};
use zend::cli::ZendArgs;

#[derive(Parser, Debug)]
#[command(
    name = "zend",
    about = "Zen Code daemon — persistent AI coding assistant",
    long_about = None,
)]
struct Cli {
    #[command(flatten)]
    host: HostArgs,
    #[command(flatten)]
    zend: ZendArgs,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let tenant: Box<dyn Tenant> = Box::new(cli.zend.resolve()?);
    run(
        HostConfig {
            binary: "zend",
            version: env!("CARGO_PKG_VERSION"),
            args: cli.host,
            one_shot: ZendArgs::ONE_SHOT,
        },
        vec![tenant],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    /// zend's flags and the process's live side by side without clashing, and
    /// the positional workspace still comes last.
    #[test]
    fn the_two_flag_sets_parse_together() {
        let cli = Cli::try_parse_from([
            "zend",
            "--host",
            "192.168.0.5",
            "--port",
            "8081",
            "--tls-port",
            "8444",
            "--local-signin",
            "me@example.com",
            "--disable-layer",
            "repo_map",
            "../mind",
        ])
        .unwrap();
        assert_eq!(cli.host.port, 8081);
        assert_eq!(cli.host.tls_port, Some(8444));
        assert_eq!(cli.zend.disable_layer, ["repo_map"]);
        assert_eq!(cli.zend.workspace, PathBuf::from("../mind"));
    }

    /// There is no model flag: zend runs the host's prescribed model.
    #[test]
    fn there_is_no_model_flag() {
        assert!(Cli::try_parse_from(["zend", "--model", "Qwen35_0_8B_Q8"]).is_err());
        assert!(Cli::try_parse_from(["zend", "--download-deepseek"]).is_err());
    }
}
