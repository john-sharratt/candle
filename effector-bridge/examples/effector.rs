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

use anyhow::{anyhow, bail, Result};
use effector_bridge::profile::androidtv;
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

    // Android TV: pairing holds one TLS session open while a human reads the
    // on-screen code (fed through a file here, since this has no stdin), and
    // control reads the saved cert/key back in.
    if profile == "androidtv" {
        let path = |k: &str, d: &str| map.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
        let certfile = path("certfile", "androidtv-cert.pem");
        let keyfile = path("keyfile", "androidtv-key.pem");
        if action == "pair" {
            let pinfile = path("pinfile", "androidtv.pin");
            let pairing = androidtv::pair_start(&address, &nets(), "effector-bridge").await?;
            let _ = std::fs::remove_file(&pinfile);
            println!("PIN is on the TV. Write its 6 hex digits to {pinfile}");
            let mut code = String::new();
            for _ in 0..150 {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if let Ok(s) = std::fs::read_to_string(&pinfile) {
                    let s = s.trim().to_string();
                    if s.len() == 6 {
                        code = s;
                        break;
                    }
                }
            }
            if code.is_empty() {
                bail!("no code was provided in time");
            }
            let id = pairing.finish(&code).await?;
            std::fs::write(&certfile, &id.cert_pem)?;
            std::fs::write(&keyfile, &id.key_pem)?;
            println!("Paired. Saved {certfile} and {keyfile}.");
            return Ok(());
        }
        if action != "on" {
            let cert = std::fs::read_to_string(&certfile)
                .map_err(|_| anyhow!("not paired yet: {certfile} missing"))?;
            let key = std::fs::read_to_string(&keyfile)
                .map_err(|_| anyhow!("not paired yet: {keyfile} missing"))?;
            map.insert("cert".into(), Value::from(cert));
            map.insert("key".into(), Value::from(key));
        }
    }

    let line = work(&profile, "device", &address, &action, &map, &nets()).await?;
    println!("{line}");
    Ok(())
}
