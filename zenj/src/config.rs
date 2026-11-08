//! ZenJ's configuration: one file for the machine the decks are cabled to
//! (`zenj/zenj.yaml`, BlackWorld's). Paths in it are read from the
//! repository's root, where `zenj` is started.

use std::collections::BTreeMap;
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, TimeZone};
use serde::Deserialize;

use crate::brain::curve::PointCfg;
use crate::hands::Combos;
use crate::oauth::Provider;
use crate::zen::ZenCfg;

/// The gateway it joins, as the other members name one.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ClusterJoin {
    pub urls: String,
    pub name: String,
    pub group_key: PathBuf,
    pub host: String,
}

/// One person the night is for.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PersonCfg {
    /// The name ZenJ knows them by: `john`, `yen`.
    pub name: String,
    /// The email the gateway signs them in with: who may use ZenJ's tools.
    #[serde(default)]
    pub email: Option<String>,
    /// Their Spotify user id, which names who added a request.
    #[serde(default)]
    pub spotify_user: Option<String>,
    /// Their unpacked Spotify privacy export.
    #[serde(default)]
    pub export: Option<PathBuf>,
    /// Their share of the night's music, `0..=1`.
    pub share: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpotifyCfg {
    #[serde(flatten)]
    pub provider: Provider,
    /// The shared *ZenJ Requests* playlist the phones add to.
    pub requests_playlist: String,
    /// How often it is looked at, in seconds.
    #[serde(default = "default_poll")]
    pub poll_s: u64,
}

fn default_poll() -> u64 {
    5
}

#[derive(Debug, Clone, Deserialize)]
pub struct TidalCfg {
    #[serde(flatten)]
    pub provider: Provider,
    /// The catalogue's country: `AU`.
    pub country: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RekordboxCfg {
    /// The release whose offsets the reader uses (`offsets/rekordbox.txt`).
    pub version: String,
    pub decks: usize,
    /// `%APPDATA%/Pioneer/rekordbox`: `master.db` and `share/` are in it.
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MidiCfg {
    /// The FLX4's messages (`maps/ddj-flx4.yaml`).
    pub flx4: PathBuf,
    /// The virtual controller's (`maps/rekordbox-7.yaml`).
    pub hands: PathBuf,
    /// How far ahead of the beat a move is sent, for the MIDI path.
    #[serde(default = "default_lead")]
    pub lead_ms: f64,
    #[serde(default)]
    pub combos: Combos,
}

fn default_lead() -> f64 {
    8.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScreenCfg {
    /// Which monitor, in the order Windows lists them.
    pub monitor: usize,
    /// Milliseconds the output shows a frame late (the TV's), which its
    /// scenes lead the beat by.
    #[serde(default)]
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScreensCfg {
    /// The MilkDrop presets the scenes are drawn from.
    pub presets: PathBuf,
    pub outputs: Vec<ScreenCfg>,
}

/// What the file says.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Where it listens: the address the gateway's route names as its
    /// upstream, and keep's guard reads.
    pub bind: String,
    pub trusted_gateways: Vec<IpAddr>,
    pub cluster: ClusterJoin,
    /// Its data folder: the library, tokens, the nights' records, its node.
    pub state: PathBuf,
    pub people: Vec<PersonCfg>,
    /// The requests' share of the night.
    pub request_share: f32,
    /// The night: the date it starts on.
    pub night: NaiveDate,
    /// When midnight is, `2027-01-01T00:00:00`.
    pub midnight: NaiveDateTime,
    pub curve: Vec<PointCfg>,
    pub spotify: SpotifyCfg,
    pub tidal: TidalCfg,
    pub rekordbox: RekordboxCfg,
    pub midi: MidiCfg,
    /// The output device the FLX4 plays the master through.
    pub audio_device: String,
    pub zen: ZenCfg,
    pub screens: ScreensCfg,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let c: Config = serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let total: f32 = c.people.iter().map(|p| p.share).sum::<f32>() + c.request_share;
        if (total - 1.0).abs() > 0.01 {
            return Err(format!(
                "{}: the people's shares and the requests' add to {total:.2}, not 1",
                path.display()
            ));
        }
        Ok(c)
    }

    pub fn midnight(&self) -> Result<DateTime<Local>, String> {
        Local
            .from_local_datetime(&self.midnight)
            .earliest()
            .ok_or_else(|| format!("{} does not exist here", self.midnight))
    }

    /// The emails that may use ZenJ's tools.
    pub fn emails(&self) -> Vec<String> {
        self.people
            .iter()
            .filter_map(|p| p.email.clone())
            .map(|e| e.to_ascii_lowercase())
            .collect()
    }

    /// Spotify user id → person.
    pub fn spotify_people(&self) -> BTreeMap<String, String> {
        self.people
            .iter()
            .filter_map(|p| Some((p.spotify_user.clone()?, p.name.clone())))
            .collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.people.iter().map(|p| p.name.clone()).collect()
    }

    /// The token name of `person`'s Spotify account.
    pub fn spotify_account(person: &str) -> String {
        format!("spotify-{person}")
    }

    pub const TIDAL_ACCOUNT: &'static str = "tidal";

    pub fn library_path(&self) -> PathBuf {
        self.state.join("library.json")
    }

    pub fn record_path(&self) -> PathBuf {
        self.state.join("nights").join(format!("{}.jsonl", self.night))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **BlackWorld's file reads**: its shares add up, midnight is the new
    /// year's, its maps and offsets are the shipped ones.
    #[test]
    fn blackworlds_config_reads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("zenj.yaml");
        let c = Config::load(&path).unwrap();
        assert_eq!(c.bind, "127.0.0.1:8097");
        assert_eq!(c.cluster.host, "blackworld");
        assert_eq!(c.names(), ["john", "yen"]);
        assert_eq!(c.emails(), ["johnathan.sharratt@gmail.com"]);
        assert_eq!(c.midnight().unwrap().format("%Y-%m-%d %H:%M").to_string(), "2027-01-01 00:00");
        assert_eq!(c.rekordbox.version, "7.2.19");
        assert_eq!(c.spotify.provider.redirect, "http://127.0.0.1:43821/callback");
        assert_eq!(c.record_path(), c.state.join("nights").join("2026-12-31.jsonl"));
        assert!(!c.midi.combos.zenj.is_empty() && !c.midi.combos.assist.is_empty());
    }

    #[test]
    fn shares_that_do_not_add_up_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let text = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("zenj.yaml"))
            .unwrap()
            .replace("request_share: 0.2", "request_share: 0.5");
        let path = dir.path().join("z.yaml");
        fs::write(&path, text).unwrap();
        assert!(Config::load(&path).unwrap_err().ends_with("add to 1.30, not 1"));
    }
}
