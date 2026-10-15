//! `npcd` binary — the NPC engine daemon.
//!
//! It parses npcd's flags and the process's, and runs npcd as the one tenant of
//! a host process (`host::run`): the host serves the console from disk and
//! answers `/v1` — npcd's own routes beside the shared endpoints — from the
//! same process. `motherd` is the same call with zend beside it.
//!
//! The process's flags are the host's (`host::HostArgs`), the same in every
//! binary. npcd is reached from other machines on the estate, so a launch
//! names its address: `--host 0.0.0.0 --port 8081`.

use clap::Parser;
use cluster::version::info::NPCD;
use host::build_info::first;
use host::{run, HostArgs, HostConfig, Tenant};
use npcd::tenant::NpcdArgs;

#[derive(Parser, Debug)]
#[command(
    name = "npcd",
    about = "NPC engine daemon — authored content, the cast and the console"
)]
struct Cli {
    #[command(flatten)]
    host: HostArgs,
    #[command(flatten)]
    npcd: NpcdArgs,
}

fn main() -> anyhow::Result<()> {
    first(NPCD);
    let cli = Cli::parse();
    let tenant: Box<dyn Tenant> = Box::new(cli.npcd.resolve()?);
    run(
        HostConfig {
            binary: "npcd",
            version: env!("CARGO_PKG_VERSION"),
            args: cli.host,
            one_shot: NpcdArgs::ONE_SHOT,
        },
        vec![tenant],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::net::IpAddr;

    /// The process's flags and npcd's own live side by side without clashing.
    #[test]
    fn the_two_flag_sets_parse_together() {
        let cli = Cli::try_parse_from([
            "npcd",
            "--host",
            "0.0.0.0",
            "--port",
            "8081",
            "--gateway",
            "192.168.0.5",
            "--mind-repo",
            "battle-mind",
        ])
        .unwrap();
        assert_eq!((cli.host.host.as_str(), cli.host.port), ("0.0.0.0", 8081));
        assert_eq!(
            cli.host.gateways,
            ["192.168.0.5".parse::<IpAddr>().unwrap()]
        );
        assert_eq!(cli.npcd.mind_repo.as_deref(), Some("battle-mind"));
        assert!(Cli::try_parse_from(["npcd", "--bind", "0.0.0.0:8081"]).is_err());
    }
}
