//! The WiZ bulb profile over its local UDP control ([`crate::proto::wiz`]):
//! on and off, a dimming, a colour, a white, and what it shows now. A bulb
//! off the house network is refused.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::proto::wiz::{ask, named, read, took, Pilot, GET_PILOT};
use crate::scope::{in_house, LocalNet};

/// How long a bulb is waited for.
const WAIT: Duration = Duration::from_secs(1);

fn at(address: &str, nets: &[LocalNet]) -> Result<Ipv4Addr> {
    let ip: Ipv4Addr = address.parse()?;
    if !in_house(nets, ip) {
        bail!("{address} is not on the house network");
    }
    Ok(ip)
}

fn level(args: &Map<String, Value>, key: &str, default: u8) -> u8 {
    args.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |n| n.min(100) as u8)
}

/// The colour `args` asks for: a name, or `r`, `g` and `b`.
fn colour_of(args: &Map<String, Value>) -> Result<(u8, u8, u8)> {
    if let Some(name) = args.get("colour").and_then(Value::as_str) {
        return named(name).ok_or_else(|| {
            anyhow!(
                "no colour called {name}: red, orange, amber, yellow, green, cyan, blue, purple, \
                 magenta, pink or white"
            )
        });
    }
    let part = |k: &str| {
        args.get(k)
            .and_then(Value::as_u64)
            .map(|n| n.min(255) as u8)
            .ok_or_else(|| anyhow!("Say the colour, by its name or as r, g and b."))
    };
    Ok((part("r")?, part("g")?, part("b")?))
}

/// The pilot `action` with `args` asks for — `None` for `status`, which sets
/// nothing.
pub fn pilot_of(action: &str, args: &Map<String, Value>) -> Result<Option<Pilot>> {
    Ok(Some(match action {
        "status" => return Ok(None),
        "off" => Pilot::Off,
        "on" => Pilot::White {
            kelvin: 2700,
            dimming: level(args, "level", 100),
        },
        "dim" => Pilot::White {
            kelvin: args
                .get("kelvin")
                .and_then(Value::as_u64)
                .map_or(2700, |k| k.min(u16::MAX as u64) as u16),
            dimming: level(args, "level", 50),
        },
        "white" => Pilot::White {
            kelvin: args
                .get("kelvin")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the white, in kelvin: 2200 warm to 6500 cool."))?
                .min(u16::MAX as u64) as u16,
            dimming: level(args, "level", 100),
        },
        "colour" => {
            let (r, g, b) = colour_of(args)?;
            Pilot::Colour {
                r,
                g,
                b,
                dimming: level(args, "level", 100),
            }
        }
        other => bail!("a WiZ bulb has no `{other}`"),
    }))
}

/// What a set comes to, said.
fn said(name: &str, pilot: &Pilot) -> String {
    match *pilot {
        Pilot::Off => format!("The {name} is off."),
        Pilot::Colour { r, g, b, dimming } => {
            format!("The {name} is coloured {r},{g},{b} at {dimming}%.")
        }
        Pilot::White { kelvin, dimming } => {
            format!("The {name} is white at {kelvin} K, {dimming}%.")
        }
        Pilot::Scene { id, .. } => format!("The {name} is showing its scene {id}."),
    }
}

/// Work `action` on the WiZ bulb `name` at `address`.
pub async fn work(
    name: &str,
    address: &str,
    action: &str,
    args: &Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    let ip = at(address, nets)?;
    let Some(pilot) = pilot_of(action, args)? else {
        let answer = ask(ip, GET_PILOT, WAIT).await?;
        return Ok(read(&answer)?.line(name));
    };
    let answer = ask(ip, &pilot.message(), WAIT).await?;
    if !took(&answer) {
        bail!("the {name} answered and did not take it");
    }
    Ok(said(name, &pilot))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    /// **Each act is the pilot it names**, by a colour's name or its parts,
    /// with a dimming or the act's own default.
    #[test]
    fn each_act_is_the_pilot_it_names() {
        use serde_json::json;
        assert_eq!(pilot_of("status", &Map::new()).unwrap(), None);
        assert_eq!(pilot_of("off", &Map::new()).unwrap(), Some(Pilot::Off));
        assert_eq!(
            pilot_of("on", &Map::new()).unwrap(),
            Some(Pilot::White {
                kelvin: 2700,
                dimming: 100
            })
        );
        assert_eq!(
            pilot_of("dim", &args(json!({"level": 30}))).unwrap(),
            Some(Pilot::White {
                kelvin: 2700,
                dimming: 30
            })
        );
        assert_eq!(
            pilot_of("colour", &args(json!({"colour": "blue", "level": 80}))).unwrap(),
            Some(Pilot::Colour {
                r: 0,
                g: 0,
                b: 255,
                dimming: 80
            })
        );
        assert_eq!(
            pilot_of("colour", &args(json!({"r": 10, "g": 20, "b": 300}))).unwrap(),
            Some(Pilot::Colour {
                r: 10,
                g: 20,
                b: 255,
                dimming: 100
            })
        );
        assert!(pilot_of("colour", &args(json!({"colour": "chartreuse"}))).is_err());
        assert!(pilot_of("white", &Map::new()).is_err());
        assert!(pilot_of("strobe", &Map::new()).is_err());
    }

    /// **A bulb off the house network is refused** before anything is sent.
    #[tokio::test]
    async fn a_bulb_off_the_house_network_is_refused() {
        let err = work("bulb", "8.8.8.8", "on", &Map::new(), &[])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("not on the house network"),
            "{err}"
        );
    }
}
