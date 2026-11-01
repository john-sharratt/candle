//! The AV-receiver profile: power, mute and volume on a Pioneer or Onkyo
//! receiver, over its text control protocol ([`crate::proto::textline`]) on
//! port 8102. Its short status lines (`PWR0`, `VOL120`, `MUT1`) parse here; the
//! exchange is the protocol library's. A receiver off the house network is
//! refused.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::proto::textline;
use crate::scope::{in_house, LocalNet};

const PORT: u16 = 8102;
const ANSWER: Duration = Duration::from_secs(4);

/// The loudest the receiver's own scale goes: `VOL185` is its maximum.
const MAX_NATIVE: u16 = 185;

/// The receiver's own volume step for a percentage `0..=100`.
pub fn native_from_pct(pct: u8) -> u16 {
    let pct = pct.min(100) as u32;
    ((pct * MAX_NATIVE as u32 + 50) / 100) as u16
}

/// A percentage `0..=100` for the receiver's own step `0..=185`.
pub fn pct_from_native(native: u16) -> u8 {
    let native = native.min(MAX_NATIVE) as u32;
    ((native * 100 + MAX_NATIVE as u32 / 2) / MAX_NATIVE as u32) as u8
}

/// The command that sets the volume to `pct` percent: three digits and `VL`.
pub fn set_volume_cmd(pct: u8) -> String {
    format!("{:03}VL", native_from_pct(pct))
}

/// Whether a status line says the receiver is on: `PWR0` on, `PWR1`/`PWR2`
/// standby — the Pioneer convention, where zero is on.
pub fn parse_power(line: &str) -> Option<bool> {
    match line.trim() {
        "PWR0" => Some(true),
        "PWR1" | "PWR2" => Some(false),
        _ => None,
    }
}

/// Whether a status line says the sound is muted: `MUT0` muted, `MUT1` not.
pub fn parse_mute(line: &str) -> Option<bool> {
    match line.trim() {
        "MUT0" => Some(true),
        "MUT1" => Some(false),
        _ => None,
    }
}

/// The volume a `VOL***` status line reports, as a percentage.
pub fn parse_volume(line: &str) -> Option<u8> {
    let digits = line.trim().strip_prefix("VOL")?;
    (digits.len() == 3)
        .then(|| digits.parse::<u16>().ok())
        .flatten()
        .map(pct_from_native)
}

/// What the receiver is doing. A field is `None` when it did not say.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub on: Option<bool>,
    pub volume_pct: Option<u8>,
    pub muted: Option<bool>,
}

impl Status {
    fn take(&mut self, line: &str) {
        if let Some(on) = parse_power(line) {
            self.on = Some(on);
        }
        if let Some(v) = parse_volume(line) {
            self.volume_pct = Some(v);
        }
        if let Some(m) = parse_mute(line) {
            self.muted = Some(m);
        }
    }

    fn of(lines: &[String]) -> Status {
        let mut s = Status::default();
        for l in lines {
            s.take(l);
        }
        s
    }

    /// What the receiver `name` is doing, as a line its body reads.
    pub fn line(&self, name: &str) -> String {
        let power = match self.on {
            Some(true) => "on",
            Some(false) => "in standby",
            None => "— its power unknown",
        };
        let mut parts = vec![format!("The {name} is {power}")];
        if let Some(v) = self.volume_pct {
            parts.push(format!("at volume {v} of 100"));
        }
        if self.muted == Some(true) {
            parts.push("muted".to_string());
        }
        format!("{}.", parts.join(", "))
    }
}

/// An AV receiver at its address on the house network.
pub struct Receiver {
    address: Ipv4Addr,
}

impl Receiver {
    pub fn at(address: &str, nets: &[LocalNet]) -> Result<Receiver> {
        let address: Ipv4Addr = address.parse()?;
        if !in_house(nets, address) {
            bail!("{address} is not on the house network");
        }
        Ok(Receiver { address })
    }

    async fn exchange(&self, cmds: &[&str], want: &str) -> Result<Status> {
        let lines = textline::exchange(self.address, PORT, cmds, want, ANSWER).await?;
        Ok(Status::of(&lines))
    }

    pub async fn status(&self) -> Result<Status> {
        self.exchange(&["?P", "?V", "?M"], "MUT").await
    }
    pub async fn power_on(&self) -> Result<Status> {
        self.exchange(&["PO", "?P"], "PWR").await
    }
    pub async fn power_off(&self) -> Result<Status> {
        self.exchange(&["PF", "?P"], "PWR").await
    }
    pub async fn mute(&self) -> Result<Status> {
        self.exchange(&["MO", "?M"], "MUT").await
    }
    pub async fn unmute(&self) -> Result<Status> {
        self.exchange(&["MF", "?M"], "MUT").await
    }
    pub async fn set_volume(&self, pct: u8) -> Result<Status> {
        self.exchange(&[&set_volume_cmd(pct), "?V"], "VOL").await
    }
}

/// Work `action` on the receiver `name` at `address`, returning the line its
/// body reads.
pub async fn work(
    name: &str,
    address: &str,
    action: &str,
    args: &Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    let receiver = Receiver::at(address, nets)?;
    let status = match action {
        "status" => receiver.status().await?,
        "on" => receiver.power_on().await?,
        "off" => receiver.power_off().await?,
        "mute" => receiver.mute().await?,
        "unmute" => receiver.unmute().await?,
        "volume" => {
            let v = args
                .get("volume")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the volume, 0 to 100."))?;
            receiver.set_volume(v.min(100) as u8).await?
        }
        other => bail!("an AV receiver has no `{other}`"),
    };
    Ok(status.line(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_maps_between_percent_and_the_receivers_scale() {
        assert_eq!(native_from_pct(0), 0);
        assert_eq!(native_from_pct(100), 185);
        assert_eq!(native_from_pct(50), 93);
        assert_eq!(pct_from_native(93), 50);
        assert_eq!(set_volume_cmd(40), "074VL");
    }

    #[test]
    fn status_lines_read_as_what_the_receiver_is_doing() {
        assert_eq!(parse_power("PWR0"), Some(true));
        assert_eq!(parse_power("PWR1"), Some(false));
        assert_eq!(parse_mute("MUT0"), Some(true));
        assert_eq!(parse_volume("VOL120"), Some(65));
        assert_eq!(parse_volume("VOL18"), None);
        let s = Status::of(&["PWR0".into(), "VOL093".into(), "MUT1".into()]);
        assert_eq!(
            s,
            Status {
                on: Some(true),
                volume_pct: Some(50),
                muted: Some(false)
            }
        );
        assert_eq!(s.line("stereo"), "The stereo is on, at volume 50 of 100.");
        let m = Status::of(&["PWR0".into(), "MUT0".into()]);
        assert_eq!(m.line("stereo"), "The stereo is on, muted.");
    }

    #[test]
    fn a_receiver_off_the_house_network_is_refused() {
        let home = [LocalNet {
            own: "192.168.0.237".parse().unwrap(),
            mask: "255.255.255.0".parse().unwrap(),
        }];
        assert!(Receiver::at("192.168.0.61", &home).is_ok());
        assert!(Receiver::at("8.8.8.8", &home).is_err());
    }
}
