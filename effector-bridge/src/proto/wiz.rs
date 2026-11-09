//! WiZ's local control (`docs/zenj.md` §9): JSON over UDP on port 38899, to
//! one bulb by its address. `setPilot` sets what the bulb shows and
//! `getPilot` reads it back. There is no authentication — the bulb's own
//! "Allow local communication" switch is the only gate — and the answer to a
//! `setPilot` is a courtesy, not a timing a sender can lean on: a lighting
//! renderer sends and goes on.
//!
//! The messages are written by hand rather than through a serializer, so each
//! one is exactly the bytes its tests name.

use std::net::{Ipv4Addr, SocketAddrV4};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokio::net::UdpSocket;
use tokio::time::timeout;

/// The port a bulb listens on.
pub const PORT: u16 = 38899;

/// The dimmest a bulb goes while on: below it, it is off.
pub const DIM_FLOOR: u8 = 10;

/// The white a bulb can make, in kelvin.
pub const KELVIN: std::ops::RangeInclusive<u16> = 2200..=6500;

/// The speeds a bulb's own scenes run at.
pub const SPEED: std::ops::RangeInclusive<u8> = 10..=200;

/// What a bulb is told to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pilot {
    Off,
    /// A colour at a dimming.
    Colour {
        r: u8,
        g: u8,
        b: u8,
        dimming: u8,
    },
    /// White at a temperature, at a dimming.
    White {
        kelvin: u16,
        dimming: u8,
    },
    /// One of the bulb's own scenes, by its id, at a speed and a dimming.
    Scene {
        id: u8,
        speed: u8,
        dimming: u8,
    },
}

/// `dimming` as a bulb takes it: [`DIM_FLOOR`] to 100.
pub fn dimming(level: u8) -> u8 {
    level.clamp(DIM_FLOOR, 100)
}

impl Pilot {
    /// The `setPilot` that shows it, as the bytes sent.
    pub fn message(&self) -> String {
        let params = match *self {
            Pilot::Off => r#""state":false"#.to_string(),
            Pilot::Colour {
                r,
                g,
                b,
                dimming: d,
            } => {
                format!(r#""r":{r},"g":{g},"b":{b},"dimming":{}"#, dimming(d))
            }
            Pilot::White { kelvin, dimming: d } => format!(
                r#""temp":{},"dimming":{}"#,
                kelvin.clamp(*KELVIN.start(), *KELVIN.end()),
                dimming(d)
            ),
            Pilot::Scene {
                id,
                speed,
                dimming: d,
            } => format!(
                r#""sceneId":{id},"speed":{},"dimming":{}"#,
                speed.clamp(*SPEED.start(), *SPEED.end()),
                dimming(d)
            ),
        };
        format!(r#"{{"method":"setPilot","params":{{{params}}}}}"#)
    }
}

/// The message that asks a bulb what it shows.
pub const GET_PILOT: &str = r#"{"method":"getPilot","params":{}}"#;

/// What a bulb said it shows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reading {
    pub mac: String,
    pub on: bool,
    pub dimming: Option<u8>,
    pub colour: Option<(u8, u8, u8)>,
    pub kelvin: Option<u16>,
    pub scene: Option<u8>,
    /// Its signal, in dBm.
    pub rssi: Option<i32>,
}

impl Reading {
    /// It, in a line: `The bulb is on, warm white at 2700 K, 60%.`
    pub fn line(&self, name: &str) -> String {
        if !self.on {
            return format!("The {name} is off.");
        }
        let what = match (self.scene, self.kelvin, self.colour) {
            (Some(id), _, _) if id != 0 => format!("showing its scene {id}"),
            (_, Some(k), _) => format!("white at {k} K"),
            (_, _, Some((r, g, b))) => format!("coloured {r},{g},{b}"),
            _ => "lit".to_string(),
        };
        match self.dimming {
            Some(d) => format!("The {name} is on, {what}, at {d}%."),
            None => format!("The {name} is on, {what}."),
        }
    }
}

fn small<T: TryFrom<u64>>(v: &Value) -> Option<T> {
    v.as_u64().and_then(|n| T::try_from(n).ok())
}

/// A bulb's answer to [`GET_PILOT`].
pub fn read(answer: &[u8]) -> Result<Reading> {
    let v: Value = serde_json::from_slice(answer)
        .map_err(|e| anyhow!("the bulb's answer is not JSON: {e}"))?;
    if let Some(e) = v.get("error") {
        bail!("the bulb refused: {e}");
    }
    let r = v
        .get("result")
        .ok_or_else(|| anyhow!("the bulb's answer has no result"))?;
    let rgb = match (small(&r["r"]), small(&r["g"]), small(&r["b"])) {
        (Some(r), Some(g), Some(b)) => Some((r, g, b)),
        _ => None,
    };
    Ok(Reading {
        mac: r["mac"].as_str().unwrap_or_default().to_string(),
        on: r["state"].as_bool().unwrap_or(false),
        dimming: small(&r["dimming"]),
        colour: rgb,
        kelvin: small(&r["temp"]),
        scene: small(&r["sceneId"]),
        rssi: r["rssi"].as_i64().and_then(|n| i32::try_from(n).ok()),
    })
}

/// Whether a bulb's answer to a `setPilot` says it took it.
pub fn took(answer: &[u8]) -> bool {
    serde_json::from_slice::<Value>(answer)
        .ok()
        .and_then(|v| v["result"]["success"].as_bool())
        .unwrap_or(false)
}

/// Send `message` to the bulb at `address` and wait at most `wait` for its
/// answer.
pub async fn ask(address: Ipv4Addr, message: &str, wait: Duration) -> Result<Vec<u8>> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).await?;
    socket
        .send_to(message.as_bytes(), SocketAddrV4::new(address, PORT))
        .await?;
    let mut buf = vec![0u8; 2048];
    let n = timeout(wait, socket.recv(&mut buf))
        .await
        .map_err(|_| anyhow!("the bulb at {address} did not answer"))??;
    buf.truncate(n);
    Ok(buf)
}

/// A named colour's red, green and blue.
pub fn named(colour: &str) -> Option<(u8, u8, u8)> {
    Some(match colour.trim().to_ascii_lowercase().as_str() {
        "red" => (255, 0, 0),
        "orange" => (255, 96, 0),
        "amber" => (255, 150, 0),
        "yellow" => (255, 220, 0),
        "green" => (0, 255, 0),
        "cyan" => (0, 255, 255),
        "blue" => (0, 0, 255),
        "purple" => (128, 0, 255),
        "magenta" => (255, 0, 255),
        "pink" => (255, 40, 120),
        "white" => (255, 255, 255),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each pilot is its own `setPilot`, byte for byte**, its dimming held
    /// to what a bulb takes and its temperature to the white it can make.
    #[test]
    fn each_pilot_is_its_message_to_the_byte() {
        assert_eq!(
            Pilot::Off.message(),
            r#"{"method":"setPilot","params":{"state":false}}"#
        );
        assert_eq!(
            Pilot::Colour {
                r: 255,
                g: 0,
                b: 64,
                dimming: 100
            }
            .message(),
            r#"{"method":"setPilot","params":{"r":255,"g":0,"b":64,"dimming":100}}"#
        );
        assert_eq!(
            Pilot::White {
                kelvin: 9000,
                dimming: 3
            }
            .message(),
            r#"{"method":"setPilot","params":{"temp":6500,"dimming":10}}"#
        );
        assert_eq!(
            Pilot::Scene {
                id: 4,
                speed: 255,
                dimming: 60
            }
            .message(),
            r#"{"method":"setPilot","params":{"sceneId":4,"speed":200,"dimming":60}}"#
        );
        assert_eq!(GET_PILOT, r#"{"method":"getPilot","params":{}}"#);
    }

    /// **A bulb's reading is what it says**: on or off, its colour or white,
    /// its dimming, and the line Zen reads.
    #[test]
    fn a_reading_is_what_the_bulb_says() {
        let answer = br#"{"method":"getPilot","env":"pro","result":{"mac":"a8bb50112233","rssi":-58,"state":true,"sceneId":0,"temp":2700,"dimming":60}}"#;
        let r = read(answer).unwrap();
        assert_eq!(
            r,
            Reading {
                mac: "a8bb50112233".into(),
                on: true,
                dimming: Some(60),
                colour: None,
                kelvin: Some(2700),
                scene: Some(0),
                rssi: Some(-58),
            }
        );
        assert_eq!(r.line("bulb"), "The bulb is on, white at 2700 K, at 60%.");
        let red = read(br#"{"result":{"state":true,"r":255,"g":0,"b":0,"dimming":100}}"#).unwrap();
        assert_eq!(
            red.line("bulb"),
            "The bulb is on, coloured 255,0,0, at 100%."
        );
        let off = read(br#"{"result":{"state":false}}"#).unwrap();
        assert_eq!(off.line("bulb"), "The bulb is off.");
        assert!(read(br#"{"error":{"code":-32600}}"#).is_err());
        assert!(read(b"not json").is_err());
    }

    #[test]
    fn a_set_is_taken_only_when_the_bulb_says_so() {
        assert!(took(
            br#"{"method":"setPilot","env":"pro","result":{"success":true}}"#
        ));
        assert!(!took(br#"{"result":{"success":false}}"#));
        assert!(!took(b""));
    }

    #[test]
    fn colours_are_named() {
        assert_eq!(named(" Red "), Some((255, 0, 0)));
        assert_eq!(named("chartreuse"), None);
    }

    /// **A bulb that is not there is said not to answer**, inside the wait.
    #[tokio::test]
    async fn a_bulb_that_is_not_there_does_not_answer() {
        let err = ask(
            "127.0.0.1".parse().unwrap(),
            GET_PILOT,
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("did not answer") || err.to_string().contains("refused"),
            "{err}"
        );
    }
}
