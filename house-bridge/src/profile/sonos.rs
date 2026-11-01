//! The Sonos profile: what a player is doing, and playing, pausing and setting
//! its volume — over UPnP SOAP ([`crate::proto::soap`]) on port 1400, the
//! interface the Sonos apps use on a home network. A player off the member's
//! own subnets is refused.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::proto::soap::{self, Service};
use crate::scope::{in_house, LocalNet};

const PORT: u16 = 1400;
const ANSWER: Duration = Duration::from_secs(4);

const TRANSPORT: Service = Service {
    urn: "urn:schemas-upnp-org:service:AVTransport:1",
    path: "/MediaRenderer/AVTransport/Control",
};
const RENDERING: Service = Service {
    urn: "urn:schemas-upnp-org:service:RenderingControl:1",
    path: "/MediaRenderer/RenderingControl/Control",
};

/// What a player is doing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    /// `PLAYING`, `PAUSED_PLAYBACK`, `STOPPED`, `TRANSITIONING`.
    pub state: String,
    pub title: String,
    pub artist: String,
    pub volume: Option<u8>,
}

impl Status {
    /// What it is doing, in a sentence.
    pub fn line(&self, name: &str) -> String {
        let doing = match self.state.as_str() {
            "PLAYING" => "is playing",
            "PAUSED_PLAYBACK" => "is paused on",
            "STOPPED" => "is stopped, last on",
            "TRANSITIONING" => "is starting",
            _ => "is",
        };
        let what = match (self.title.is_empty(), self.artist.is_empty()) {
            (true, _) => "nothing it names".to_string(),
            (false, true) => format!("\"{}\"", self.title),
            (false, false) => format!("\"{}\" by {}", self.title, self.artist),
        };
        let volume = self
            .volume
            .map(|v| format!(", at volume {v} of 100"))
            .unwrap_or_default();
        format!("The {name} {doing} {what}{volume}.")
    }
}

/// The state from a `GetTransportInfo` answer.
pub fn parse_transport(xml: &str) -> Option<String> {
    soap::tag(xml, "CurrentTransportState")
}

/// The title and artist from a `GetPositionInfo` answer, out of its
/// `TrackMetaData` — DIDL-Lite, escaped inside the answer.
pub fn parse_track(xml: &str) -> (String, String) {
    let Some(meta) = soap::tag(xml, "TrackMetaData") else {
        return Default::default();
    };
    let title = soap::tag(&meta, "dc:title").unwrap_or_default();
    let artist = soap::tag(&meta, "dc:creator")
        .or_else(|| soap::tag(&meta, "upnp:artist"))
        .unwrap_or_default();
    (title, artist)
}

/// The volume from a `GetVolume` answer.
pub fn parse_volume(xml: &str) -> Option<u8> {
    soap::tag(xml, "CurrentVolume")?.parse().ok()
}

/// One player, at its address.
pub struct Player {
    address: Ipv4Addr,
    client: reqwest::Client,
}

impl Player {
    /// The player at `address`, which must be on one of `nets` — the house
    /// network this member stands on.
    pub fn at(address: &str, nets: &[LocalNet]) -> Result<Player> {
        let address: Ipv4Addr = address.parse()?;
        if !in_house(nets, address) {
            bail!("{address} is not on the house network");
        }
        Ok(Player {
            address,
            client: soap::client(ANSWER)?,
        })
    }

    async fn call(&self, service: Service, action: &str, args: &[(&str, &str)]) -> Result<String> {
        soap::call(&self.client, self.address, PORT, service, action, args).await
    }

    pub async fn status(&self) -> Result<Status> {
        let transport = self
            .call(TRANSPORT, "GetTransportInfo", &[("InstanceID", "0")])
            .await?;
        let position = self
            .call(TRANSPORT, "GetPositionInfo", &[("InstanceID", "0")])
            .await?;
        let volume = self
            .call(
                RENDERING,
                "GetVolume",
                &[("InstanceID", "0"), ("Channel", "Master")],
            )
            .await?;
        let (title, artist) = parse_track(&position);
        Ok(Status {
            state: parse_transport(&transport).unwrap_or_default(),
            title,
            artist,
            volume: parse_volume(&volume),
        })
    }

    pub async fn play(&self) -> Result<()> {
        self.call(TRANSPORT, "Play", &[("InstanceID", "0"), ("Speed", "1")])
            .await
            .map(drop)
    }

    pub async fn pause(&self) -> Result<()> {
        self.call(TRANSPORT, "Pause", &[("InstanceID", "0")])
            .await
            .map(drop)
    }

    pub async fn set_volume(&self, volume: u8) -> Result<()> {
        let v = volume.min(100).to_string();
        self.call(
            RENDERING,
            "SetVolume",
            &[
                ("InstanceID", "0"),
                ("Channel", "Master"),
                ("DesiredVolume", &v),
            ],
        )
        .await
        .map(drop)
    }
}

/// Work `action` on the Sonos player `name` at `address`, returning the line
/// its body reads.
pub async fn work(
    name: &str,
    address: &str,
    action: &str,
    args: &Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    let player = Player::at(address, nets)?;
    let line = match action {
        "status" => player.status().await?.line(name),
        "play" => {
            player.play().await?;
            format!("The {name} is playing.")
        }
        "pause" => {
            player.pause().await?;
            format!("The {name} is paused.")
        }
        "volume" => {
            let v = args
                .get("volume")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the volume, 0 to 100."))?;
            let v = v.min(100) as u8;
            player.set_volume(v).await?;
            format!("The {name} is at volume {v} of 100.")
        }
        other => bail!("a Sonos player has no `{other}`"),
    };
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answers_read_as_what_the_player_is_doing() {
        let transport = "<u:R><CurrentTransportState>PLAYING</CurrentTransportState></u:R>";
        assert_eq!(parse_transport(transport).as_deref(), Some("PLAYING"));
        let position = "<u:R><TrackMetaData>&lt;DIDL&gt;&lt;dc:title&gt;Harbour Lights&lt;/dc:title&gt;&lt;dc:creator&gt;The Waves &amp;amp; Co&lt;/dc:creator&gt;&lt;/DIDL&gt;</TrackMetaData></u:R>";
        assert_eq!(
            parse_track(position),
            ("Harbour Lights".to_string(), "The Waves & Co".to_string())
        );
        assert_eq!(
            parse_volume("<x><CurrentVolume>23</CurrentVolume></x>"),
            Some(23)
        );
        let s = Status {
            state: "PLAYING".into(),
            title: "Harbour Lights".into(),
            artist: "The Waves & Co".into(),
            volume: Some(23),
        };
        assert_eq!(
            s.line("Sonos speaker"),
            "The Sonos speaker is playing \"Harbour Lights\" by The Waves & Co, at volume 23 of 100."
        );
    }

    #[test]
    fn a_player_off_the_house_network_is_refused() {
        let home = [LocalNet {
            own: "192.168.0.237".parse().unwrap(),
            mask: "255.255.255.0".parse().unwrap(),
        }];
        assert!(Player::at("192.168.0.89", &home).is_ok());
        assert!(Player::at("8.8.8.8", &home).is_err());
    }
}
