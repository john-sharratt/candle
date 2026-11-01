//! A command-line driver for the effector bridges, for trying a real device on
//! the local network without the cluster — the agent runs exactly this `work`
//! call in production.
//!
//! ```text
//! cargo run -p effector-bridge --example effector -- <profile> <address> <action> [key=value ...]
//! # e.g.
//! cargo run -p effector-bridge --example effector -- cast 192.168.0.214 launch app=netflix
//! cargo run -p effector-bridge --example effector -- cast 192.168.0.214 status
//! cargo run -p effector-bridge --example effector -- sonos 192.168.0.89 volume volume=20
//! cargo run -p effector-bridge --example effector -- receiver 192.168.0.61 status
//! cargo run -p effector-bridge --example effector -- wake "the pc" --mac=AA:BB:CC:DD:EE:FF
//! ```
//!
//! A `key=value` whose value parses as a whole number is passed as a number;
//! everything else is a string. `wake` is a special case: pass `--mac=<addr>`.

use anyhow::{bail, Result};
use effector_bridge::{nets, wake, work};
use serde_json::{Map, Value};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: <profile> <address> <action> [key=value ...]   (profile `wake`: <wake> <name> --mac=<addr>)");
    }

    if args[0] == "wake" {
        let name = &args[1];
        let mac = args
            .iter()
            .find_map(|a| a.strip_prefix("--mac="))
            .unwrap_or_else(|| panic!("wake needs --mac=<hardware address>"));
        println!("{}", wake(name, mac, &nets())?);
        return Ok(());
    }

    let [profile, address, action] = match &args[..3] {
        [p, a, c] => [p.clone(), a.clone(), c.clone()],
        _ => bail!("usage: <profile> <address> <action> [key=value ...]"),
    };

    let mut map = Map::new();
    for kv in &args[3..] {
        let Some((k, v)) = kv.split_once('=') else {
            bail!("arguments after the action are key=value, got `{kv}`");
        };
        let value = v
            .parse::<u64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::from(v));
        map.insert(k.to_string(), value);
    }

    let line = work(&profile, "device", &address, &action, &map, &nets()).await?;
    println!("{line}");
    Ok(())
}
