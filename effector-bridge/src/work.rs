//! The one abstraction a caller reaches a device through (`docs/zen_moon_mind.md`
//! §15): given a device's profile and an action, [`work`] does it through that
//! profile's library. An NPC's effector command resolves to this, and it is
//! this that runs on an in-network agent — the agent calls [`work`] (or
//! [`wake`]) to do the real network commands locally, the way it runs the git
//! commands for a member.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::finding::{CAST, RECEIVER, SONOS};
use crate::profile::{cast, receiver, sonos};
use crate::proto::udp::{self, WAKE_PORT};
use crate::scope::LocalNet;

/// If `tool` is `<profile>_<action>` for a profile the world drives, the
/// profile and the action — `sonos_play` → (`sonos`, `play`). `None` for any
/// other tool, including the cross-device `wake_device` and the survey.
pub fn profile_act(tool: &str) -> Option<(&'static str, &str)> {
    for p in [SONOS, RECEIVER, CAST] {
        if let Some(action) = tool.strip_prefix(&format!("{p}_")) {
            return Some((p, action));
        }
    }
    None
}

/// Work `action` on the device `name` of `profile` at `address`, with `args`,
/// on the house network `nets` — the line the caller reads, or why it was
/// refused.
pub async fn work(
    profile: &str,
    name: &str,
    address: &str,
    action: &str,
    args: &Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    match profile {
        SONOS => sonos::work(name, address, action, args, nets).await,
        RECEIVER => receiver::work(name, address, action, args, nets).await,
        CAST => cast::work(name, address, action, args, nets).await,
        other => bail!("the world has no way to work a {other}"),
    }
}

/// Wake the device `name` with hardware address `mac` by broadcasting a magic
/// packet on the house's own subnets — the one thing the world can do to a
/// machine that is asleep, needing no account and nothing switched on at it.
pub fn wake(name: &str, mac: &str, nets: &[LocalNet]) -> Result<String> {
    let Some(packet) = udp::magic(mac) else {
        bail!("{mac} is not a hardware address to wake");
    };
    udp::broadcast(&packet, nets, WAKE_PORT)?;
    Ok(format!(
        "A wake-up was sent to the {name}. If it was asleep, it should come back on the network \
         in a moment."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_act_splits_into_profile_and_action() {
        assert_eq!(profile_act("sonos_play"), Some(("sonos", "play")));
        assert_eq!(profile_act("receiver_volume"), Some(("receiver", "volume")));
        assert_eq!(profile_act("wake_device"), None);
        assert_eq!(profile_act("survey_network"), None);
        assert_eq!(profile_act("tell"), None);
    }

    #[tokio::test]
    async fn an_unknown_profile_is_refused() {
        let out = work("toaster", "x", "192.168.0.5", "status", &Map::new(), &[]).await;
        assert!(out
            .unwrap_err()
            .to_string()
            .contains("no way to work a toaster"));
    }

    #[test]
    fn a_wake_with_no_house_network_is_refused() {
        assert!(wake("pc", "01:02:03:04:05:06", &[]).is_err());
        assert!(wake("pc", "not a mac", &[]).is_err());
    }
}
