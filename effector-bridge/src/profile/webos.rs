//! The LG webOS TV profile over SSAP ([`crate::proto::webos`]) on port 3000.
//!
//! It wakes the TV (Wake-on-LAN), turns it off, sets volume and mute, launches
//! apps and switches inputs, drives playback and channels, shows a toast, and
//! reads what is on. Control needs the `client-key` the TV grants on pairing;
//! the `pair` action obtains it. A TV off the house network is refused.

use std::net::Ipv4Addr;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Map, Value};

use crate::proto::udp::{broadcast, magic, WAKE_PORT};
use crate::proto::webos::Remote;
use crate::scope::{in_house, LocalNet};

/// App launcher ids by friendly name; anything else is taken as a raw id.
const APPS: &[(&str, &str)] = &[
    ("netflix", "netflix"),
    ("youtube", "youtube.leanback.v4"),
    ("prime", "amazon"),
    ("amazon", "amazon"),
    ("disney", "com.disney.disneyplus-prod"),
    ("spotify", "spotify-beehive"),
    ("appletv", "com.apple.appletv"),
    ("plex", "cdp-30"),
    ("browser", "com.webos.app.browser"),
    ("livetv", "com.webos.app.livetv"),
    ("tv", "com.webos.app.livetv"),
    ("hdmi1", "com.webos.app.hdmi1"),
    ("hdmi2", "com.webos.app.hdmi2"),
    ("hdmi3", "com.webos.app.hdmi3"),
    ("hdmi4", "com.webos.app.hdmi4"),
];

/// The friendly name shown for a launcher id, else the id itself.
fn app_name(id: &str) -> String {
    APPS.iter()
        .find(|(_, v)| v.eq_ignore_ascii_case(id))
        .map(|(n, _)| n.to_string())
        .unwrap_or_else(|| id.to_string())
}

fn app_id(app: &str) -> &str {
    APPS.iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(app.trim()))
        .map(|(_, v)| *v)
        .unwrap_or(app.trim())
}

/// A webOS TV at its address on the house network.
pub struct WebOs {
    address: String,
}

impl WebOs {
    pub fn at(address: &str, nets: &[LocalNet]) -> Result<WebOs> {
        let ip: Ipv4Addr = address.parse()?;
        if !in_house(nets, ip) {
            bail!("{address} is not on the house network");
        }
        Ok(WebOs {
            address: address.to_string(),
        })
    }

    async fn remote(&self, key: Option<&str>) -> Result<Remote> {
        Remote::open(&self.address, key).await
    }

    /// Volume level and mute, tolerant of the two shapes webOS reports.
    async fn status(&self, name: &str, key: Option<&str>) -> Result<String> {
        let mut r = self.remote(key).await?;
        let vol = r.request("ssap://audio/getVolume", json!({})).await?;
        let (level, muted) = read_volume(&vol);
        let app = r
            .request(
                "ssap://com.webos.applicationManager/getForegroundAppInfo",
                json!({}),
            )
            .await
            .ok()
            .and_then(|p| p["appId"].as_str().map(app_name));
        let mut parts = vec![format!("The {name} is on")];
        if let Some(a) = app.filter(|a| !a.is_empty()) {
            parts.push(a);
        }
        if let Some(v) = level {
            parts.push(format!("at volume {v} of 100"));
        }
        if muted == Some(true) {
            parts.push("muted".to_string());
        }
        Ok(format!("{}.", parts.join(", ")))
    }

    /// Press one remote button over the pointer-input socket.
    async fn button(&self, key: Option<&str>, name: &str) -> Result<()> {
        let mut r = self.remote(key).await?;
        let socket = r.pointer_socket().await?;
        crate::proto::webos::send_button(&socket, name).await
    }

    /// The installed apps, by their shown titles.
    async fn apps(&self, name: &str, key: Option<&str>) -> Result<String> {
        let mut r = self.remote(key).await?;
        let p = r
            .request(
                "ssap://com.webos.applicationManager/listLaunchPoints",
                json!({}),
            )
            .await?;
        let titles: Vec<String> = p["launchPoints"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x["title"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        if titles.is_empty() {
            Ok(format!("The {name} listed no apps."))
        } else {
            Ok(format!("The {name} has: {}.", titles.join(", ")))
        }
    }

    /// The external inputs, by their labels.
    async fn inputs(&self, name: &str, key: Option<&str>) -> Result<String> {
        let mut r = self.remote(key).await?;
        let p = r
            .request("ssap://tv/getExternalInputList", json!({}))
            .await?;
        let labels: Vec<String> = p["devices"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| {
                        x["label"]
                            .as_str()
                            .or_else(|| x["id"].as_str())
                            .map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default();
        if labels.is_empty() {
            Ok(format!("The {name} listed no inputs."))
        } else {
            Ok(format!("The {name}'s inputs: {}.", labels.join(", ")))
        }
    }
}

/// The webOS input id for a friendly name (`hdmi2` → `HDMI_2`), else the value
/// itself so a raw id or a device-specific label passes through.
fn input_id(input: &str) -> String {
    let t = input
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-', '_'], "");
    match t.as_str() {
        "hdmi" | "hdmi1" => "HDMI_1",
        "hdmi2" => "HDMI_2",
        "hdmi3" => "HDMI_3",
        "hdmi4" => "HDMI_4",
        "av" | "av1" => "AV_1",
        "comp" | "component" | "comp1" => "COMP_1",
        _ => return input.trim().to_string(),
    }
    .to_string()
}

/// The webOS sound-output id for a friendly name, else the value itself.
fn sound_output(output: &str) -> String {
    match output.trim().to_ascii_lowercase().as_str() {
        "tv" | "tv speaker" | "internal" | "speaker" => "tv_speaker",
        "arc" | "earc" | "receiver" | "hdmi" | "amp" => "external_arc",
        "optical" => "external_optical",
        "bluetooth" | "bt" | "soundbar" => "bt_soundbar",
        "headphones" | "wired" | "lineout" => "tv_external_speaker",
        other => return other.to_string(),
    }
    .to_string()
}

/// Read `{volume, muted}` or `{volumeStatus:{volume, muted}}`.
fn read_volume(p: &Value) -> (Option<u8>, Option<bool>) {
    let v = p["volume"]
        .as_u64()
        .or_else(|| p["volumeStatus"]["volume"].as_u64());
    let m = p["muted"]
        .as_bool()
        .or_else(|| p["volumeStatus"]["muted"].as_bool());
    (v.map(|x| x.min(100) as u8), m)
}

fn arg(args: &Map<String, Value>, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Work `action` on the webOS TV `name` at `address`. Control actions read the
/// `key` argument (the stored client-key); `pair` obtains one, `on` wakes by
/// `mac`.
pub async fn work(
    name: &str,
    address: &str,
    action: &str,
    args: &Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    let tv = WebOs::at(address, nets)?;
    let key = arg(args, "key");
    let key = key.as_deref();

    // Power-on is Wake-on-LAN — the TV's SSAP socket is down when it is off.
    if action == "on" {
        let mac = arg(args, "mac").ok_or_else(|| anyhow!("Say the TV's hardware address."))?;
        let packet = magic(&mac).ok_or_else(|| anyhow!("{mac} is not a hardware address"))?;
        broadcast(&packet, nets, WAKE_PORT)?;
        return Ok(format!("A wake-up was sent to the {name}."));
    }

    // Pairing returns the client-key to keep for every later call.
    if action == "pair" {
        let r = tv.remote(key).await?;
        return match r.client_key() {
            Some(k) => Ok(format!("Paired with the {name}. Its key is {k}")),
            None => bail!("the {name} did not grant a key"),
        };
    }

    if action == "status" {
        return tv.status(name, key).await;
    }
    if action == "apps" {
        return tv.apps(name, key).await;
    }
    if action == "inputs" {
        return tv.inputs(name, key).await;
    }

    // Remote-button actions go over the pointer-input socket, not SSAP.
    let named_button = match action {
        "home" => Some("HOME"),
        "back" => Some("BACK"),
        "up" => Some("UP"),
        "down" => Some("DOWN"),
        "left" => Some("LEFT"),
        "right" => Some("RIGHT"),
        "ok" | "enter" | "select" => Some("ENTER"),
        "exit" => Some("EXIT"),
        "menu" => Some("MENU"),
        "info" => Some("INFO"),
        "guide" | "dash" => Some("DASH"),
        _ => None,
    };
    if let Some(b) = named_button {
        tv.button(key, b).await?;
        return Ok(format!("Pressed {} on the {name}.", b.to_ascii_lowercase()));
    }
    if action == "button" {
        let b = arg(args, "name").ok_or_else(|| anyhow!("Say which button."))?;
        tv.button(key, &b.to_ascii_uppercase()).await?;
        return Ok(format!("Pressed {} on the {name}.", b.to_ascii_lowercase()));
    }

    let mut r = tv.remote(key).await?;
    let line = match action {
        "off" => {
            r.tell("ssap://system/turnOff", json!({})).await?;
            format!("The {name} is turning off.")
        }
        "volume" => {
            let v = args
                .get("volume")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the volume, 0 to 100."))?
                .min(100);
            r.tell("ssap://audio/setVolume", json!({ "volume": v }))
                .await?;
            format!("The {name} is at volume {v} of 100.")
        }
        "louder" => {
            r.tell("ssap://audio/volumeUp", json!({})).await?;
            format!("The {name} is a little louder.")
        }
        "quieter" => {
            r.tell("ssap://audio/volumeDown", json!({})).await?;
            format!("The {name} is a little quieter.")
        }
        "mute" => {
            r.tell("ssap://audio/setMute", json!({ "mute": true }))
                .await?;
            format!("The {name} is muted.")
        }
        "unmute" => {
            r.tell("ssap://audio/setMute", json!({ "mute": false }))
                .await?;
            format!("The {name} is unmuted.")
        }
        "launch" => {
            let app = arg(args, "app").ok_or_else(|| anyhow!("Say which app to open."))?;
            r.tell(
                "ssap://system.launcher/launch",
                json!({ "id": app_id(&app) }),
            )
            .await?;
            format!("The {name} opened {app}.")
        }
        "input" => {
            let input = arg(args, "input").ok_or_else(|| anyhow!("Say which input."))?;
            let id = input_id(&input);
            r.tell("ssap://tv/switchInput", json!({ "inputId": id }))
                .await?;
            format!("The {name} switched to {input}.")
        }
        "play" => {
            r.tell("ssap://media.controls/play", json!({})).await?;
            format!("The {name} is playing.")
        }
        "pause" => {
            r.tell("ssap://media.controls/pause", json!({})).await?;
            format!("The {name} is paused.")
        }
        "stop" => {
            r.tell("ssap://media.controls/stop", json!({})).await?;
            format!("The {name} is stopped.")
        }
        "forward" | "fast_forward" => {
            r.tell("ssap://media.controls/fastForward", json!({}))
                .await?;
            format!("The {name} is fast-forwarding.")
        }
        "rewind" => {
            r.tell("ssap://media.controls/rewind", json!({})).await?;
            format!("The {name} is rewinding.")
        }
        "close" => {
            let app = arg(args, "app").ok_or_else(|| anyhow!("Say which app to close."))?;
            r.tell(
                "ssap://system.launcher/close",
                json!({ "id": app_id(&app) }),
            )
            .await?;
            format!("The {name} closed {app}.")
        }
        "open" => {
            let url = arg(args, "url").ok_or_else(|| anyhow!("Say the URL to open."))?;
            r.tell("ssap://system.launcher/open", json!({ "target": url }))
                .await?;
            format!("The {name} opened {url}.")
        }
        "channel" => {
            let ch = arg(args, "channel").ok_or_else(|| anyhow!("Say the channel number."))?;
            r.tell("ssap://tv/openChannel", json!({ "channelNumber": ch }))
                .await?;
            format!("The {name} went to channel {ch}.")
        }
        "sound" => {
            let out = arg(args, "output").ok_or_else(|| anyhow!("Say the sound output."))?;
            r.tell(
                "ssap://com.webos.service.apiadapter/audio/changeSoundOutput",
                json!({ "output": sound_output(&out) }),
            )
            .await?;
            format!("The {name}'s sound is on {out}.")
        }
        "screen_off" => {
            r.tell(
                "ssap://com.webos.service.tvpower/power/turnOffScreen",
                json!({}),
            )
            .await?;
            format!("The {name}'s screen is off.")
        }
        "screen_on" => {
            r.tell(
                "ssap://com.webos.service.tvpower/power/turnOnScreen",
                json!({}),
            )
            .await?;
            format!("The {name}'s screen is on.")
        }
        "channel_up" => {
            r.tell("ssap://tv/channelUp", json!({})).await?;
            format!("The {name} went up a channel.")
        }
        "channel_down" => {
            r.tell("ssap://tv/channelDown", json!({})).await?;
            format!("The {name} went down a channel.")
        }
        "toast" => {
            let msg = arg(args, "message").ok_or_else(|| anyhow!("Say the message to show."))?;
            r.tell(
                "ssap://system.notifications/createToast",
                json!({ "message": msg }),
            )
            .await?;
            format!("The {name} is showing \"{msg}\".")
        }
        other => bail!("a webOS TV has no `{other}`"),
    };
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apps_resolve_both_ways() {
        assert_eq!(app_id("netflix"), "netflix");
        assert_eq!(app_id("youtube"), "youtube.leanback.v4");
        assert_eq!(app_id("hdmi2"), "com.webos.app.hdmi2");
        assert_eq!(app_id("com.some.raw.id"), "com.some.raw.id");
        assert_eq!(app_name("youtube.leanback.v4"), "youtube");
        assert_eq!(app_name("com.webos.app.livetv"), "livetv");
        assert_eq!(app_name("com.unknown"), "com.unknown");
    }

    #[test]
    fn inputs_and_sound_outputs_resolve() {
        assert_eq!(input_id("hdmi2"), "HDMI_2");
        assert_eq!(input_id("HDMI 3"), "HDMI_3");
        assert_eq!(input_id("HDMI_1"), "HDMI_1");
        assert_eq!(input_id("AV"), "AV_1");
        assert_eq!(input_id("SOME_CUSTOM"), "SOME_CUSTOM");
        assert_eq!(sound_output("arc"), "external_arc");
        assert_eq!(sound_output("tv"), "tv_speaker");
        assert_eq!(sound_output("bt_soundbar"), "bt_soundbar");
    }

    #[test]
    fn volume_is_read_from_either_shape() {
        assert_eq!(
            read_volume(&json!({"volume": 12, "muted": false})),
            (Some(12), Some(false))
        );
        assert_eq!(
            read_volume(&json!({"volumeStatus": {"volume": 30, "muted": true}})),
            (Some(30), Some(true))
        );
    }

    #[test]
    fn a_tv_off_the_house_network_is_refused() {
        let home = [LocalNet {
            own: "192.168.0.237".parse().unwrap(),
            mask: "255.255.255.0".parse().unwrap(),
        }];
        assert!(WebOs::at("192.168.0.48", &home).is_ok());
        assert!(WebOs::at("8.8.8.8", &home).is_err());
    }
}
