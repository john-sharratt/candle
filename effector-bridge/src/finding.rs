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
/// An LG webOS TV ([`crate::profile::webos`]).
pub const WEBOS: &str = "webos";
/// A Google TV / Android TV ([`crate::profile::androidtv`]).
pub const ANDROIDTV: &str = "androidtv";
/// A UPnP MediaRenderer: a TV or player with AVTransport and RenderingControl.
pub const RENDERER: &str = "renderer";
/// A Pioneer or Onkyo AV receiver ([`crate::profile::receiver`]).
pub const RECEIVER: &str = "receiver";
/// A WiZ bulb ([`crate::profile::wiz`]).
pub const WIZ: &str = "wiz";
/// The hardware-address prefixes WiZ's bulbs are made with. A bulb announces
/// nothing a survey hears — no mDNS, no UPnP — so it is known by these, or by
/// the `wiz_` name it gives the router.
pub const WIZ_PREFIXES: &[&str] = &["a8:bb:50", "44:4f:8e", "6c:29:90", "d8:a0:11"];
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
        CAST => "the world can work it (a Cast device)",
        WEBOS => "the world can work it (an LG webOS TV)",
        ANDROIDTV => "the world can work it (a Google TV)",
        WIZ => "the world can work it (a WiZ bulb)",
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
    matches!(profile, SONOS | RECEIVER | CAST | WEBOS | ANDROIDTV | WIZ)
}

/// Whether `f` is a WiZ bulb: by its hardware address's maker prefix, or by
/// the `wiz_…` name it gives the router.
fn is_wiz(f: &Finding) -> bool {
    let hw = f.hardware.to_ascii_lowercase().replace('-', ":");
    WIZ_PREFIXES.iter().any(|p| hw.starts_with(p))
        || f.names
            .iter()
            .any(|n| n.to_ascii_lowercase().starts_with("wiz_"))
}

/// Whether a device of `profile` earns its credentials by pairing before the
/// world can work it — an LG webOS TV (accept on the screen, one client-key)
/// and a Google TV (a six-digit code, an RSA client certificate). The others
/// need no secret.
pub fn pairs(profile: &str) -> bool {
    matches!(profile, WEBOS | ANDROIDTV)
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
    } else if is_wiz(f) {
        WIZ
    } else if said(&["pioneer", "onkyo"]) {
        RECEIVER
    } else if said(&["androidtvremote", "android tv", "google tv"]) {
        // A Google TV also announces Cast; the remote protocol drives it more
        // fully, so it is matched first.
        ANDROIDTV
    } else if said(&["googlecast", "google cast"]) {
        CAST
    } else if said(&["webos", "lge-com", "lg smart tv"]) {
        WEBOS
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
                &["urn:lge-com:service:webos-second-screen:1"],
                "LG Electronics",
                "LG Smart TV"
            )),
            WEBOS
        );
        // A Google TV announces both the remote protocol and Cast; the remote wins.
        assert_eq!(
            profile_of(&with(
                &["_androidtvremote2._tcp.local", "_googlecast._tcp.local"],
                "Google Inc.",
                "Chromecast"
            )),
            ANDROIDTV
        );
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
        // A WiZ bulb announces nothing: it is known by its maker's prefix, or
        // by the name it gives the router.
        let by_prefix = Finding {
            hardware: "A8:BB:50:12:34:56".into(),
            ..Finding::default()
        };
        assert_eq!(profile_of(&by_prefix), WIZ);
        let by_name = Finding {
            hardware: "02:00:00:00:00:01".into(),
            names: vec!["wiz_6e1f2a".into()],
            ..Finding::default()
        };
        assert_eq!(profile_of(&by_name), WIZ);
    }

    #[test]
    fn only_a_workable_profile_is_a_terminal() {
        for p in [SONOS, RECEIVER, CAST, WEBOS, ANDROIDTV, WIZ] {
            assert!(workable(p), "{p} should be workable");
        }
        for p in [RENDERER, GATEWAY, AIRPLAY, SPOTIFY, PRINTER, DEVICE] {
            assert!(!workable(p), "{p} should be see-only");
        }
    }

    #[test]
    fn only_the_two_tvs_pair() {
        for p in [WEBOS, ANDROIDTV] {
            assert!(pairs(p), "{p} earns a credential by pairing");
        }
        for p in [SONOS, RECEIVER, CAST, RENDERER, GATEWAY, PRINTER, DEVICE] {
            assert!(!pairs(p), "{p} needs no pairing");
        }
    }
}
