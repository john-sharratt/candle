//! A device a survey found (`docs/zen_moon_mind.md` §13): what it said about
//! itself, and the bridge profile that can reach it.

use serde::{Deserialize, Serialize};

/// The profile of a device the world can work: a Sonos player.
pub const SONOS: &str = "sonos";
/// The profile of a device the world can see but not work.
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
    /// The bridge profile that reaches it — [`SONOS`], or [`DEVICE`].
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

    /// The id a part onboarded from it is given: [`ONBOARDED`] and its
    /// hardware address — `device-f0f6c1a079b4`.
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
        let reach = match self.profile.as_str() {
            SONOS => "the world can work it (Sonos)",
            _ => "the world can see it, not work it",
        };
        what.push(reach.to_string());
        format!("{} — {}", self.label(), what.join("; "))
    }
}

/// The profile that reaches a device saying what `f` says.
pub fn profile_of(f: &Finding) -> &'static str {
    let said = |s: &str| {
        f.services
            .iter()
            .chain(std::iter::once(&f.maker))
            .any(|x| x.to_ascii_lowercase().contains(s))
    };
    match said("sonos") {
        true => SONOS,
        false => DEVICE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sonos() -> Finding {
        Finding {
            address: "192.168.0.89".into(),
            hardware: "f0:f6:c1:a0:79:b4".into(),
            names: vec!["Sonos Arc".into(), "SonosZP".into()],
            maker: "Sonos, Inc.".into(),
            model: "Sonos Arc".into(),
            services: vec!["_sonos._tcp.local".into()],
            profile: SONOS.into(),
        }
    }

    #[test]
    fn a_finding_is_named_by_its_best_name_and_address() {
        let f = sonos();
        assert_eq!(f.label(), "Sonos Arc at 192.168.0.89");
        assert_eq!(f.part_id(), "device-f0f6c1a079b4");
        assert_eq!(
            f.line(),
            "Sonos Arc at 192.168.0.89 — Sonos, Inc. Sonos Arc; also called SonosZP; announces \
             _sonos._tcp.local; the world can work it (Sonos)"
        );
        let bare = Finding {
            address: "192.168.0.214".into(),
            hardware: "14:ae:85:71:eb:44".into(),
            profile: DEVICE.into(),
            ..Finding::default()
        };
        assert_eq!(bare.label(), "14:ae:85:71:eb:44 at 192.168.0.214");
        assert_eq!(
            bare.line(),
            "14:ae:85:71:eb:44 at 192.168.0.214 — the world can see it, not work it"
        );
    }

    /// **A Sonos player is reached through the Sonos profile**, whatever says
    /// so — its service or its maker; anything else is a device.
    #[test]
    fn a_sonos_player_is_reached_through_its_profile() {
        assert_eq!(profile_of(&sonos()), SONOS);
        let by_maker = Finding {
            maker: "Sonos, Inc.".into(),
            ..Finding::default()
        };
        assert_eq!(profile_of(&by_maker), SONOS);
        let cast = Finding {
            services: vec!["_googlecast._tcp.local".into()],
            ..Finding::default()
        };
        assert_eq!(profile_of(&cast), DEVICE);
    }
}
