//! The AV-receiver profile: an Onkyo, Integra or Pioneer network receiver over
//! the eISCP control protocol ([`crate::proto::eiscp`]) on port 60128.
//!
//! It drives most of what the receiver can do — power, master volume (set and
//! step), mute, the input selector, the listening mode, the display dimmer,
//! network/USB transport and now-playing, and Zone 2 — each a short three-letter
//! command (`PWR`, `MVL`, `SLI`, `LMD`, …) and a parameter. The command codes
//! are the Integra Serial Control Protocol's; the friendly names map to them
//! here. Anything not named is still reachable as a raw command. A receiver off
//! the house network is refused.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::proto::eiscp;
use crate::scope::{in_house, LocalNet};

const PORT: u16 = 60128;
/// How long to wait for a status the receiver confirms (power, volume, input).
const ANSWER: Duration = Duration::from_secs(4);
/// How long to gather the chatter after a command the receiver does not confirm
/// with a known reply (transport, dimmer, a raw command).
const GRACE: Duration = Duration::from_millis(900);

/// Input selector: a friendly name to its `SLI` code. Several names may share a
/// code (a receiver's one physical input has many labels across models).
const INPUTS: &[(&str, &str)] = &[
    ("dvd", "10"),
    ("bd", "10"),
    ("blu-ray", "10"),
    ("bluray", "10"),
    ("cbl", "01"),
    ("sat", "01"),
    ("cable", "01"),
    ("satellite", "01"),
    ("stb", "00"),
    ("dvr", "00"),
    ("vcr", "00"),
    ("streambox", "11"),
    ("strm-box", "11"),
    ("stream-box", "11"),
    ("stream", "11"),
    ("game", "02"),
    ("aux", "03"),
    ("pc", "05"),
    ("tv", "12"),
    ("cd", "23"),
    ("phono", "22"),
    ("tuner", "26"),
    ("fm", "24"),
    ("am", "25"),
    ("dab", "33"),
    ("network", "2B"),
    ("net", "2B"),
    ("usb", "29"),
    ("internet-radio", "28"),
    ("iradio", "28"),
    ("music-server", "27"),
    ("server", "27"),
    ("dlna", "27"),
    ("bluetooth", "2E"),
    ("bt", "2E"),
    ("multi-ch", "30"),
];

/// Input selector: an `SLI` code to the one name shown for it.
const INPUT_NAMES: &[(&str, &str)] = &[
    ("00", "STB/DVR"),
    ("01", "Cable/Sat"),
    ("02", "Game"),
    ("03", "Aux"),
    ("05", "PC"),
    ("10", "BD/DVD"),
    ("11", "Stream Box"),
    ("12", "TV"),
    ("22", "Phono"),
    ("23", "CD"),
    ("24", "FM"),
    ("25", "AM"),
    ("26", "Tuner"),
    ("27", "Music Server"),
    ("28", "Internet Radio"),
    ("29", "USB"),
    ("2B", "Network"),
    ("2E", "Bluetooth"),
    ("30", "Multi-ch"),
    ("33", "DAB"),
];

/// Listening mode: a friendly name to its `LMD` code.
const MODES: &[(&str, &str)] = &[
    ("stereo", "00"),
    ("direct", "01"),
    ("surround", "02"),
    ("thx", "04"),
    ("all-ch-stereo", "0C"),
    ("all-stereo", "0C"),
    ("mono", "0F"),
    ("pure", "11"),
    ("pure-audio", "11"),
    ("full-mono", "13"),
    ("straight", "40"),
    ("straight-decode", "40"),
    ("dolby", "80"),
    ("dolby-surround", "80"),
    ("atmos", "80"),
    ("dts-x", "82"),
    ("neural-x", "82"),
    ("auto", "FF"),
    ("auto-surround", "FF"),
];

/// Listening mode: an `LMD` code to the one name shown for it.
const MODE_NAMES: &[(&str, &str)] = &[
    ("00", "Stereo"),
    ("01", "Direct"),
    ("02", "Surround"),
    ("04", "THX"),
    ("0C", "All-Ch Stereo"),
    ("0F", "Mono"),
    ("11", "Pure Audio"),
    ("13", "Full Mono"),
    ("40", "Straight Decode"),
    ("80", "Dolby Surround"),
    ("82", "DTS:X / Neural:X"),
    ("FF", "Auto Surround"),
];

/// Display dimmer: a friendly name to its `DIM` code.
const DIMS: &[(&str, &str)] = &[
    ("bright", "00"),
    ("dim", "01"),
    ("dark", "02"),
    ("off", "03"),
    ("shut-off", "03"),
    ("bright-led-off", "08"),
];

/// Network/USB transport: a friendly action to the `NTC` key it sends.
const TRANSPORT: &[(&str, &str)] = &[
    ("play", "PLAY"),
    ("pause", "PAUSE"),
    ("stop", "STOP"),
    ("next", "TRUP"),
    ("previous", "TRDN"),
    ("prev", "TRDN"),
];

/// The code a friendly `name` maps to in `table`, matched case-insensitively.
fn code_for(table: &[(&'static str, &'static str)], name: &str) -> Option<&'static str> {
    let n = name.trim().to_ascii_lowercase();
    table.iter().find(|(k, _)| *k == n).map(|(_, v)| *v)
}

/// The name shown for `code` in `table`, matched case-insensitively.
fn name_for(table: &[(&'static str, &'static str)], code: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(c, _)| c.eq_ignore_ascii_case(code))
        .map(|(_, n)| *n)
}

/// Whether `s` is two hexadecimal digits — a raw selector/mode code a caller may
/// give directly when no friendly name covers it.
fn is_hex2(s: &str) -> bool {
    s.len() == 2 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Resolve a caller's `value` against `table` to a protocol code: a friendly
/// name if it is one, otherwise the value itself if it is a raw two-hex code.
fn resolve(table: &[(&'static str, &'static str)], value: &str) -> Result<String> {
    let v = value.trim();
    if let Some(code) = code_for(table, v) {
        return Ok(code.to_string());
    }
    if is_hex2(v) {
        return Ok(v.to_ascii_uppercase());
    }
    bail!("there is no `{v}` to set here")
}

/// The command that sets the master volume to `pct` percent: `MVL` and the
/// level as two hex digits, on the receiver's absolute `0..=100` master scale.
pub fn set_volume_cmd(pct: u8) -> String {
    format!("MVL{:02X}", pct.min(100))
}

/// Whether a message says the receiver is on: `PWR01` on, `PWR00` standby — the
/// Onkyo convention, where one is on.
pub fn parse_power(msg: &str) -> Option<bool> {
    match msg.trim() {
        "PWR01" => Some(true),
        "PWR00" => Some(false),
        _ => None,
    }
}

/// Whether a message says the sound is muted: `AMT01` muted, `AMT00` not.
pub fn parse_mute(msg: &str) -> Option<bool> {
    match msg.trim() {
        "AMT01" => Some(true),
        "AMT00" => Some(false),
        _ => None,
    }
}

/// The volume an `MVL**` message reports, as a percentage — the two hex digits
/// are the level on the receiver's `0..=100` master scale.
pub fn parse_volume(msg: &str) -> Option<u8> {
    let hex = msg.trim().strip_prefix("MVL")?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(hex, 16).ok())
        .flatten()
        .map(|v| v.min(100))
}

/// The input an `SLI**` message reports, shown by name when it is one the world
/// knows, else by its raw code.
pub fn parse_input(msg: &str) -> Option<String> {
    let code = msg.trim().strip_prefix("SLI")?;
    Some(
        name_for(INPUT_NAMES, code)
            .map(str::to_string)
            .unwrap_or_else(|| format!("input {code}")),
    )
}

/// The listening mode an `LMD**` message reports, by name when known, else raw.
pub fn parse_mode(msg: &str) -> Option<String> {
    let code = msg.trim().strip_prefix("LMD")?;
    Some(
        name_for(MODE_NAMES, code)
            .map(str::to_string)
            .unwrap_or_else(|| format!("mode {code}")),
    )
}

/// What the receiver is doing. A field is `None` when it did not say.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub on: Option<bool>,
    pub volume_pct: Option<u8>,
    pub muted: Option<bool>,
    pub input: Option<String>,
    pub mode: Option<String>,
}

impl Status {
    fn take(&mut self, msg: &str) {
        if let Some(on) = parse_power(msg) {
            self.on = Some(on);
        }
        if let Some(v) = parse_volume(msg) {
            self.volume_pct = Some(v);
        }
        if let Some(m) = parse_mute(msg) {
            self.muted = Some(m);
        }
        if let Some(i) = parse_input(msg) {
            self.input = Some(i);
        }
        if let Some(m) = parse_mode(msg) {
            self.mode = Some(m);
        }
    }

    fn of(msgs: &[String]) -> Status {
        let mut s = Status::default();
        for m in msgs {
            s.take(m);
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
        if let Some(i) = &self.input {
            parts.push(format!("its source {i}"));
        }
        if let Some(m) = &self.mode {
            parts.push(format!("in {m}"));
        }
        if let Some(v) = self.volume_pct {
            parts.push(format!("at volume {v} of 100"));
        }
        if self.muted == Some(true) {
            parts.push("muted".to_string());
        }
        format!("{}.", parts.join(", "))
    }
}

/// What a network source is playing.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct NowPlaying {
    pub title: String,
    pub artist: String,
    pub album: String,
    /// `Play`, `Paused`, `Stopped`, … from the first letter of an `NST` status.
    pub state: String,
}

impl NowPlaying {
    fn take(&mut self, msg: &str) {
        let m = msg.trim();
        if let Some(t) = m.strip_prefix("NTI") {
            self.title = t.to_string();
        } else if let Some(a) = m.strip_prefix("NAT") {
            self.artist = a.to_string();
        } else if let Some(a) = m.strip_prefix("NAL") {
            self.album = a.to_string();
        } else if let Some(s) = m.strip_prefix("NST") {
            self.state = match s.chars().next() {
                Some('P') => "Playing",
                Some('p') => "Paused",
                Some('S') => "Stopped",
                Some('F') => "Fast-forwarding",
                Some('R') => "Rewinding",
                _ => "",
            }
            .to_string();
        }
    }

    fn of(msgs: &[String]) -> NowPlaying {
        let mut n = NowPlaying::default();
        for m in msgs {
            n.take(m);
        }
        n
    }

    fn line(&self, name: &str) -> String {
        if self.title.is_empty() && self.artist.is_empty() {
            return format!("The {name} is not playing anything from a network source.");
        }
        let what = match (self.title.is_empty(), self.artist.is_empty()) {
            (false, false) => format!("\"{}\" by {}", self.title, self.artist),
            (false, true) => format!("\"{}\"", self.title),
            (true, _) => "something it does not name".to_string(),
        };
        let lead = match self.state.as_str() {
            "Playing" => "is playing".to_string(),
            "Paused" => "is paused on".to_string(),
            "Stopped" => "is stopped, last on".to_string(),
            "" => "has".to_string(),
            other => format!("is {}", other.to_ascii_lowercase()),
        };
        format!("The {name} {lead} {what}.")
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

    /// Send `cmds` and read confirmations until one begins with `want`.
    async fn exchange(&self, cmds: &[&str], want: &str) -> Result<Status> {
        let msgs = eiscp::exchange(self.address, PORT, cmds, want, ANSWER).await?;
        Ok(Status::of(&msgs))
    }

    /// Send `cmds` and gather whatever comes back over the grace — for commands
    /// the receiver does not confirm with a known reply.
    async fn tell(&self, cmds: &[&str]) -> Result<Vec<String>> {
        eiscp::tell(self.address, PORT, cmds, GRACE).await
    }

    pub async fn status(&self) -> Result<Status> {
        self.exchange(
            &["PWRQSTN", "MVLQSTN", "AMTQSTN", "SLIQSTN", "LMDQSTN"],
            "LMD",
        )
        .await
    }
    pub async fn power_on(&self) -> Result<Status> {
        self.exchange(&["PWR01", "PWRQSTN"], "PWR").await
    }
    pub async fn power_off(&self) -> Result<Status> {
        self.exchange(&["PWR00", "PWRQSTN"], "PWR").await
    }
    pub async fn mute(&self) -> Result<Status> {
        self.exchange(&["AMT01", "AMTQSTN"], "AMT").await
    }
    pub async fn unmute(&self) -> Result<Status> {
        self.exchange(&["AMT00", "AMTQSTN"], "AMT").await
    }
    pub async fn set_volume(&self, pct: u8) -> Result<Status> {
        self.exchange(&[&set_volume_cmd(pct), "MVLQSTN"], "MVL")
            .await
    }
    pub async fn louder(&self) -> Result<Status> {
        self.exchange(&["MVLUP", "MVLQSTN"], "MVL").await
    }
    pub async fn quieter(&self) -> Result<Status> {
        self.exchange(&["MVLDOWN", "MVLQSTN"], "MVL").await
    }
    pub async fn set_input(&self, source: &str) -> Result<Status> {
        let code = resolve(INPUTS, source)?;
        self.exchange(&[&format!("SLI{code}"), "SLIQSTN"], "SLI")
            .await
    }
    pub async fn set_mode(&self, mode: &str) -> Result<Status> {
        let code = resolve(MODES, mode)?;
        self.exchange(&[&format!("LMD{code}"), "LMDQSTN"], "LMD")
            .await
    }
    pub async fn cycle_mode(&self) -> Result<Status> {
        self.exchange(&["LMDUP", "LMDQSTN"], "LMD").await
    }
    pub async fn dim(&self, level: &str) -> Result<Status> {
        let code = code_for(DIMS, level).ok_or_else(|| {
            anyhow!("the display can be bright, dim, dark, off or bright-led-off")
        })?;
        let msgs = self.tell(&[&format!("DIM{code}"), "DIMQSTN"]).await?;
        Ok(Status::of(&msgs))
    }
    pub async fn transport(&self, action: &str) -> Result<()> {
        let key = code_for(TRANSPORT, action)
            .ok_or_else(|| anyhow!("a network source can play, pause, stop, next or previous"))?;
        self.tell(&[&format!("NTC{key}")]).await.map(drop)
    }
    pub async fn now_playing(&self) -> Result<NowPlaying> {
        let msgs = self
            .tell(&["NTIQSTN", "NATQSTN", "NALQSTN", "NSTQSTN"])
            .await?;
        Ok(NowPlaying::of(&msgs))
    }
    pub async fn zone2_power(&self, on: bool) -> Result<Vec<String>> {
        let set = if on { "ZPW01" } else { "ZPW00" };
        self.tell(&[set, "ZPWQSTN"]).await
    }
    pub async fn zone2_mute(&self, on: bool) -> Result<Vec<String>> {
        let set = if on { "ZMT01" } else { "ZMT00" };
        self.tell(&[set, "ZMTQSTN"]).await
    }
    pub async fn zone2_volume(&self, pct: u8) -> Result<Vec<String>> {
        self.tell(&[&format!("ZVL{:02X}", pct.min(100)), "ZVLQSTN"])
            .await
    }
    pub async fn zone2_input(&self, source: &str) -> Result<Vec<String>> {
        let code = resolve(INPUTS, source)?;
        self.tell(&[&format!("SLZ{code}"), "SLZQSTN"]).await
    }
    /// Send a raw ISCP command (e.g. `SLI2B`, `NTCPLAY`, `TUN00879`) and read
    /// back whatever the receiver says — the escape hatch for anything the
    /// named acts do not cover.
    pub async fn raw(&self, command: &str) -> Result<Vec<String>> {
        self.tell(&[command.trim()]).await
    }
}

fn arg<'a>(args: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
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
    let r = Receiver::at(address, nets)?;
    let line = match action {
        "status" => r.status().await?.line(name),
        "on" => r.power_on().await?.line(name),
        "off" => r.power_off().await?.line(name),
        "mute" => r.mute().await?.line(name),
        "unmute" => r.unmute().await?.line(name),
        "louder" => r.louder().await?.line(name),
        "quieter" => r.quieter().await?.line(name),
        "volume" => {
            let v = args
                .get("volume")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the volume, 0 to 100."))?;
            r.set_volume(v.min(100) as u8).await?.line(name)
        }
        "input" => {
            let src =
                arg(args, "source").ok_or_else(|| anyhow!("Say which input to switch to."))?;
            r.set_input(src).await?.line(name)
        }
        "mode" => {
            let m = arg(args, "mode").ok_or_else(|| anyhow!("Say which listening mode."))?;
            r.set_mode(m).await?.line(name)
        }
        "next_mode" => r.cycle_mode().await?.line(name),
        "dim" => {
            let level =
                arg(args, "level").ok_or_else(|| anyhow!("Say bright, dim, dark or off."))?;
            r.dim(level).await?.line(name)
        }
        "play" | "pause" | "stop" | "next" | "previous" | "prev" => {
            r.transport(action).await?;
            let doing = match action {
                "play" => "playing",
                "pause" => "paused",
                "stop" => "stopped",
                "next" => "on the next track",
                _ => "on the previous track",
            };
            format!("The {name} is {doing}.")
        }
        "playing" => r.now_playing().await?.line(name),
        "zone2_on" => {
            r.zone2_power(true).await?;
            format!("The {name}'s second zone is on.")
        }
        "zone2_off" => {
            r.zone2_power(false).await?;
            format!("The {name}'s second zone is off.")
        }
        "zone2_mute" => {
            r.zone2_mute(true).await?;
            format!("The {name}'s second zone is muted.")
        }
        "zone2_unmute" => {
            r.zone2_mute(false).await?;
            format!("The {name}'s second zone is unmuted.")
        }
        "zone2_volume" => {
            let v = args
                .get("volume")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the volume, 0 to 100."))?;
            r.zone2_volume(v.min(100) as u8).await?;
            format!(
                "The {name}'s second zone is at volume {} of 100.",
                v.min(100)
            )
        }
        "zone2_input" => {
            let src = arg(args, "source").ok_or_else(|| anyhow!("Say which input."))?;
            r.zone2_input(src).await?;
            format!("The {name}'s second zone is set to {src}.")
        }
        "command" => {
            let cmd = arg(args, "command").ok_or_else(|| anyhow!("Say the raw ISCP command."))?;
            let replies = r.raw(cmd).await?;
            if replies.is_empty() {
                format!("The {name} took `{cmd}`.")
            } else {
                format!("The {name} answered `{cmd}` with: {}.", replies.join(", "))
            }
        }
        other => bail!("an AV receiver has no `{other}`"),
    };
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_is_the_percentage_as_two_hex_digits() {
        assert_eq!(set_volume_cmd(0), "MVL00");
        assert_eq!(set_volume_cmd(40), "MVL28");
        assert_eq!(set_volume_cmd(100), "MVL64");
        assert_eq!(set_volume_cmd(200), "MVL64"); // clamped
        assert_eq!(parse_volume("MVL28"), Some(40));
        assert_eq!(parse_volume("MVL4D"), Some(77)); // the real reading from .61
        assert_eq!(parse_volume("MVL4"), None);
    }

    #[test]
    fn power_and_mute_read_the_onkyo_way() {
        assert_eq!(parse_power("PWR01"), Some(true));
        assert_eq!(parse_power("PWR00"), Some(false));
        assert_eq!(parse_mute("AMT01"), Some(true));
        assert_eq!(parse_mute("AMT00"), Some(false));
    }

    #[test]
    fn inputs_and_modes_resolve_both_ways() {
        // Friendly name -> code, case-insensitive, with aliases.
        assert_eq!(code_for(INPUTS, "CD"), Some("23"));
        assert_eq!(code_for(INPUTS, "blu-ray"), Some("10"));
        assert_eq!(code_for(INPUTS, "net"), Some("2B"));
        assert_eq!(code_for(MODES, "Pure"), Some("11"));
        // A raw two-hex code is accepted when no name covers it.
        assert_eq!(resolve(INPUTS, "cd").unwrap(), "23");
        assert_eq!(resolve(INPUTS, "7f").unwrap(), "7F");
        assert!(resolve(INPUTS, "nonsense").is_err());
        // Reply code -> shown name, unknown code shown raw.
        assert_eq!(parse_input("SLI23").as_deref(), Some("CD"));
        assert_eq!(parse_input("SLI2B").as_deref(), Some("Network"));
        assert_eq!(parse_input("SLI99").as_deref(), Some("input 99"));
        assert_eq!(parse_mode("LMD00").as_deref(), Some("Stereo"));
        assert_eq!(parse_mode("LMD11").as_deref(), Some("Pure Audio"));
        assert_eq!(parse_mode("LMDZZ").as_deref(), Some("mode ZZ"));
    }

    #[test]
    fn transport_and_dim_map_to_protocol_keys() {
        assert_eq!(code_for(TRANSPORT, "next"), Some("TRUP"));
        assert_eq!(code_for(TRANSPORT, "previous"), Some("TRDN"));
        assert_eq!(code_for(TRANSPORT, "play"), Some("PLAY"));
        assert_eq!(code_for(DIMS, "dark"), Some("02"));
        assert_eq!(code_for(DIMS, "off"), Some("03"));
        assert_eq!(code_for(DIMS, "nope"), None);
    }

    #[test]
    fn status_reads_as_what_the_receiver_is_doing() {
        let s = Status::of(&[
            "PWR01".into(),
            "MVL28".into(),
            "AMT00".into(),
            "SLI23".into(),
            "LMD00".into(),
        ]);
        assert_eq!(
            s.line("stereo"),
            "The stereo is on, its source CD, in Stereo, at volume 40 of 100."
        );
        // The real standby reading from the Onkyo at .61, with input and mode.
        let standby = Status::of(&["PWR00".into(), "MVL4D".into(), "AMT00".into()]);
        assert_eq!(
            standby.line("stereo"),
            "The stereo is in standby, at volume 77 of 100."
        );
        let muted = Status::of(&["PWR01".into(), "AMT01".into()]);
        assert_eq!(muted.line("stereo"), "The stereo is on, muted.");
    }

    #[test]
    fn now_playing_reads_the_network_track() {
        let n = NowPlaying::of(&[
            "NTIHarbour Lights".into(),
            "NATThe Waves".into(),
            "NALSingles".into(),
            "NSTPrs".into(),
        ]);
        assert_eq!(
            n.line("stereo"),
            "The stereo is playing \"Harbour Lights\" by The Waves."
        );
        let idle = NowPlaying::of(&["NSTSrs".into()]);
        assert_eq!(
            idle.line("stereo"),
            "The stereo is not playing anything from a network source."
        );
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
