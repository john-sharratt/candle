//! A device a survey found (`docs/zen_moon_mind.md` §13): what it said about
//! itself, and the bridge profile that can reach it.
//!
//! A **profile** is the kind of thing a device is, by what it announces, and so
//! how the world works it (`docs/zen_moon_mind.md` §15). The survey matches one
//! to every device from its mDNS services, its UPnP device and service types,
//! and its maker; each profile a bridge module knows how to reach, or — for
//! [`DEVICE`] — a thing the world can see on the network but has no way to work.

use serde::{Deserialize, Serialize};

/// A Sonos player: UPnP on port 1400 ([`crate::profile::sonos`]).
pub const SONOS: &str = "sonos";
/// A Google Cast device — a Chromecast, a Cast-built-in TV or speaker.
pub const CAST: &str = "cast";
/// A UPnP MediaRenderer: a TV or player with AVTransport and RenderingControl.
pub const RENDERER: &str = "renderer";
/// A Pioneer or Onkyo AV receiver ([`crate::profile::receiver`]).
pub const RECEIVER: &str = "receiver";
/// An AirPlay receiver — an Apple TV, a HomePod, an AirPlay speaker.
pub const AIRPLAY: &str = "airplay";
/// A Spotify Connect endpoint.
pub const SPOTIFY: &str = "spotify";
/// The home's router: a UPnP InternetGatewayDevice.
pub const GATEWAY: &str = "gateway";
/// A network printer.
pub const PRINTER: &str = "printer";
/// A device the world can see but has no profile to work: a phone, a laptop, a
/// thing that announces nothing the world can drive.
pub const DEVICE: &str = "device";
/// What the id of a part onboarded from a finding begins with.
pub const ONBOARDED: &str = "device-";

/// One device on the house network.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    /// Its address on the network.
    pub address: String,
    /// Its hardware address — what makes it the same device across a change of
    /// address, and the finding's id.
    pub hardware: String,
    /// Every name it goes by, best first: what it calls itself, then the
    /// router's name for it.
    pub names: Vec<String>,
    pub maker: String,
    pub model: String,
    /// What it announces: UPnP device and service types, mDNS services.
    pub services: Vec<String>,
    /// The bridge profile that reaches it — one of the constants in this module.
    pub profile: String,
}

impl Finding {
    /// How it is named to whoever reads the survey and onboards from it: its
    /// best name, and where it is — `SonosZP at 192.168.0.89`.
    pub fn label(&self) -> String {
        let name = self
            .names
            .first()
            .cloned()
            .unwrap_or_else(|| self.hardware.clone());
        format!("{name} at {}", self.address)
    }

    /// Whether `label` names it — its [`Self::label`], whatever the case and
    /// the spaces around it.
    pub fn answers_to(&self, label: &str) -> bool {
        self.label().eq_ignore_ascii_case(label.trim())
    }

    /// The id a part onboarded from it is given: [`ONBOARDED`] and its hardware
    /// address — `device-f0f6c1a079b4`.
    pub fn part_id(&self) -> String {
        format!("{ONBOARDED}{}", self.hardware.replace(':', ""))
    }

    /// One line of what it is, for the survey's answer.
    pub fn line(&self) -> String {
        let mut what = Vec::new();
        if !self.maker.is_empty() || !self.model.is_empty() {
            what.push(format!("{} {}", self.maker, self.model).trim().to_string());
        }
        if self.names.len() > 1 {
            what.push(format!("also called {}", self.names[1..].join(", ")));
        }
        if !self.services.is_empty() {
            what.push(format!("announces {}", self.services.join(", ")));
        }
        what.push(reach(&self.profile).to_string());
        format!("{} — {}", self.label(), what.join("; "))
    }
}

/// What the world can do with a device of `profile`, as the survey's answer
/// puts it — and whether it can do anything at all ([`workable`]).
pub fn reach(profile: &str) -> &'static str {
    match profile {
        SONOS => "the world can work it (Sonos)",
        RECEIVER => "the world can work it (an AV receiver)",
        CAST => "the world can see it (a Cast device)",
        RENDERER => "the world can see it (a media player)",
        GATEWAY => "the world can see it (the router)",
        AIRPLAY => "the world can see it (AirPlay)",
        SPOTIFY => "the world can see it (Spotify)",
        PRINTER => "the world can see it (a printer)",
        _ => "the world can see it, not work it",
    }
}

/// Whether a part linked with `profile` has acts the world can take at it —
/// what makes it a terminal rather than a thing merely seen.
pub fn workable(profile: &str) -> bool {
    matches!(profile, SONOS | RECEIVER)
}

/// The profile that reaches a device saying what `f` says.
///
/// **Most specific first.** A Sonos and many TVs are also MediaRenderers; a
/// Cast device and some receivers also do AirPlay — so the narrower profile is
/// matched before the broader one it would also answer to.
pub fn profile_of(f: &Finding) -> &'static str {
    let said = |needles: &[&str]| {
        f.services
            .iter()
            .chain(std::iter::once(&f.maker))
            .chain(std::iter::once(&f.model))
            .any(|x| {
                let x = x.to_ascii_lowercase();
                needles.iter().any(|n| x.contains(n))
            })
    };
    if said(&["sonos"]) {
        SONOS
    } else if said(&["pioneer", "onkyo"]) {
        RECEIVER
    } else if said(&["googlecast", "google cast"]) {
        CAST
    } else if said(&["internetgatewaydevice", "wandevice", "wanconnectiondevice"]) {
        GATEWAY
    } else if said(&["mediarenderer", "avtransport"]) {
        RENDERER
    } else if said(&["airplay", "_raop", "appletv", "audio-accessory"]) {
        AIRPLAY
    } else if said(&["spotify-connect"]) {
        SPOTIFY
    } else if said(&["_ipp", "_printer", "pdl-datastream", "_scanner"]) {
        PRINTER
    } else {
        DEVICE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(services: &[&str], maker: &str, model: &str) -> Finding {
        Finding {
            services: services.iter().map(|s| s.to_string()).collect(),
            maker: maker.into(),
            model: model.into(),
            ..Finding::default()
        }
    }

    #[test]
    fn a_finding_is_named_by_its_best_name_and_address() {
        let f = Finding {
            address: "192.168.0.89".into(),
            hardware: "f0:f6:c1:a0:79:b4".into(),
            names: vec!["Sonos Arc".into(), "SonosZP".into()],
            maker: "Sonos, Inc.".into(),
            model: "Sonos Arc".into(),
            services: vec!["_sonos._tcp.local".into()],
            profile: SONOS.into(),
        };
        assert_eq!(f.label(), "Sonos Arc at 192.168.0.89");
        assert_eq!(f.part_id(), "device-f0f6c1a079b4");
        assert!(f.answers_to(" sonos arc at 192.168.0.89 "));
    }

    /// **Every kind of thing a home network holds is matched to its profile**,
    /// the narrower before the broader one it would also answer to.
    #[test]
    fn each_kind_of_device_is_matched_to_its_profile() {
        assert_eq!(
            profile_of(&with(&["_sonos._tcp"], "Sonos, Inc.", "")),
            SONOS
        );
        assert_eq!(
            profile_of(&with(
                &["_sonos._tcp", "urn:schemas-upnp-org:device:MediaRenderer:1"],
                "Sonos, Inc.",
                ""
            )),
            SONOS
        );
        assert_eq!(
            profile_of(&with(
                &["_googlecast._tcp.local"],
                "Google Inc.",
                "Chromecast"
            )),
            CAST
        );
        assert_eq!(profile_of(&with(&[], "Pioneer", "VSX-935")), RECEIVER);
        assert_eq!(
            profile_of(&with(
                &["urn:schemas-upnp-org:device:MediaRenderer:1"],
                "Samsung",
                "QN90"
            )),
            RENDERER
        );
        assert_eq!(
            profile_of(&with(
                &["urn:schemas-upnp-org:device:InternetGatewayDevice:1"],
                "mymodem",
                ""
            )),
            GATEWAY
        );
        assert_eq!(
            profile_of(&with(&["_airplay._tcp.local"], "Apple", "Apple TV")),
            AIRPLAY
        );
        assert_eq!(
            profile_of(&with(&["_spotify-connect._tcp.local"], "", "")),
            SPOTIFY
        );
        assert_eq!(
            profile_of(&with(&["_ipp._tcp.local"], "Brother", "")),
            PRINTER
        );
        assert_eq!(
            profile_of(&with(&[], "Samsung", "Galaxy S24 Ultra")),
            DEVICE
        );
    }

    #[test]
    fn only_a_workable_profile_is_a_terminal() {
        for p in [SONOS, RECEIVER] {
            assert!(workable(p), "{p} should be workable");
        }
        for p in [CAST, RENDERER, GATEWAY, AIRPLAY, SPOTIFY, PRINTER, DEVICE] {
            assert!(!workable(p), "{p} should be see-only");
        }
    }
}
