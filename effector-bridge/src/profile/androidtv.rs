//! The Android TV / Google TV profile over the Android TV Remote v2 protocol
//! ([`crate::proto::androidtv`]).
//!
//! Control needs the paired identity — the `cert` and `key` PEM the TV trusts,
//! obtained once by pairing (the world holds that; the agent is handed the
//! credential). It presses remote keys, launches apps and deep-links (a YouTube
//! video opens YouTube on it), and wakes the TV by Wake-on-LAN. A TV off the
//! house network is refused.

use std::net::Ipv4Addr;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

use crate::proto::androidtv::{generate_identity, Identity2, Pairing};
use crate::proto::androidtv::{key, Remote};
use crate::proto::udp::{broadcast, magic, WAKE_PORT};
use crate::scope::{in_house, LocalNet};

/// App package ids by friendly name; anything else is taken as a raw package.
const APPS: &[(&str, &str)] = &[
    ("youtube", "com.google.android.youtube.tv"),
    ("netflix", "com.netflix.ninja"),
    ("prime", "com.amazon.amazonvideo.livingroom"),
    ("amazon", "com.amazon.amazonvideo.livingroom"),
    ("disney", "com.disney.disneyplus"),
    ("spotify", "com.spotify.tv.android"),
    ("plex", "com.plexapp.android"),
    ("hbo", "com.hbo.hbonow"),
    ("youtube-music", "com.google.android.youtube.tvmusic"),
];

/// The key code for a friendly button name.
fn key_code(name: &str) -> Option<i32> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "home" => key::HOME,
        "back" => key::BACK,
        "up" => key::UP,
        "down" => key::DOWN,
        "left" => key::LEFT,
        "right" => key::RIGHT,
        "ok" | "enter" | "select" | "center" => key::CENTER,
        "menu" => key::MENU,
        "info" => key::INFO,
        "louder" | "volume_up" => key::VOLUME_UP,
        "quieter" | "volume_down" => key::VOLUME_DOWN,
        "mute" => key::MUTE,
        "power" => key::POWER,
        "play" | "pause" | "play_pause" => key::PLAY_PAUSE,
        "stop" => key::STOP,
        "next" => key::NEXT,
        "previous" | "prev" => key::PREVIOUS,
        "rewind" => key::REWIND,
        "forward" | "fast_forward" => key::FAST_FORWARD,
        _ => return None,
    })
}

/// The app-link for a launch target: a URL or market scheme as given, a known
/// app name as its package, else a bare package id — matching how the protocol
/// launches apps (a scheme-less target becomes `market://launch?id=…`).
fn app_link(target: &str) -> String {
    let t = target.trim();
    if t.contains("://") {
        return t.to_string();
    }
    let pkg = APPS
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(t))
        .map(|(_, p)| *p)
        .unwrap_or(t);
    format!("market://launch?id={pkg}")
}

/// The YouTube video id from a bare id, a `watch?v=` URL, or a `youtu.be` link.
fn youtube_id(v: &str) -> String {
    let v = v.trim();
    if let Some(i) = v.find("v=") {
        return v[i + 2..]
            .split(['&', '#'])
            .next()
            .unwrap_or("")
            .to_string();
    }
    if let Some(i) = v.rfind("youtu.be/") {
        return v[i + 9..]
            .split(['?', '&', '#'])
            .next()
            .unwrap_or("")
            .to_string();
    }
    v.to_string()
}

fn checked(address: &str, nets: &[LocalNet]) -> Result<()> {
    let ip: Ipv4Addr = address.parse()?;
    if !in_house(nets, ip) {
        bail!("{address} is not on the house network");
    }
    Ok(())
}

fn arg(args: &Map<String, Value>, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Begin pairing: the TV shows a six-hex code. The returned session is finished
/// with [`Pairing::finish`] once the code is known. The agent does not drive
/// this — the world holds the open session while a human reads the code.
pub async fn pair_start(address: &str, nets: &[LocalNet], client_name: &str) -> Result<Pairing> {
    checked(address, nets)?;
    Pairing::start(address, generate_identity(client_name)?, client_name).await
}

/// Work `action` on the Android TV `name` at `address`. Control reads the paired
/// `cert`/`key` PEM from `args`; `on` wakes by `mac`.
pub async fn work(
    name: &str,
    address: &str,
    action: &str,
    args: &Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    checked(address, nets)?;

    // Power-on is Wake-on-LAN — the remote protocol needs the TV already up.
    if action == "on" {
        let mac = arg(args, "mac").ok_or_else(|| anyhow!("Say the TV's hardware address."))?;
        let packet = magic(&mac).ok_or_else(|| anyhow!("{mac} is not a hardware address"))?;
        broadcast(&packet, nets, WAKE_PORT)?;
        return Ok(format!("A wake-up was sent to the {name}."));
    }

    let cert = arg(args, "cert").ok_or_else(|| anyhow!("the {name} is not paired yet"))?;
    let key = arg(args, "key").ok_or_else(|| anyhow!("the {name} is not paired yet"))?;
    let identity = Identity2 {
        cert_pem: cert,
        key_pem: key,
        modulus: Vec::new(),
        exponent: Vec::new(),
    };
    let mut r = Remote::open(address, &identity).await?;

    // A named button, pressed over the remote.
    if let Some(code) = key_code(action) {
        r.key(code).await?;
        return Ok(format!("Pressed {action} on the {name}."));
    }

    let line = match action {
        "off" => {
            r.key(key::POWER).await?;
            format!("The {name} was told to turn off.")
        }
        "key" => {
            let k = arg(args, "key_name").or_else(|| arg(args, "name"));
            let code = match k {
                Some(n) => key_code(&n).ok_or_else(|| anyhow!("no button called {n}"))?,
                None => args
                    .get("code")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| anyhow!("Say which button."))? as i32,
            };
            r.key(code).await?;
            format!("The {name} got a key.")
        }
        "launch" => {
            let app = arg(args, "app").ok_or_else(|| anyhow!("Say which app to open."))?;
            r.app_link(&app_link(&app)).await?;
            format!("The {name} opened {app}.")
        }
        "open" => {
            let url = arg(args, "url").ok_or_else(|| anyhow!("Say the URL to open."))?;
            r.app_link(&url).await?;
            format!("The {name} opened {url}.")
        }
        "youtube" => {
            let video =
                arg(args, "video").ok_or_else(|| anyhow!("Say the YouTube video id or link."))?;
            let id = youtube_id(&video);
            r.app_link(&format!("https://www.youtube.com/watch?v={id}"))
                .await?;
            format!("The {name} is playing YouTube {id}.")
        }
        other => bail!("an Android TV has no `{other}`"),
    };
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_resolve_by_name() {
        assert_eq!(key_code("home"), Some(key::HOME));
        assert_eq!(key_code("OK"), Some(key::CENTER));
        assert_eq!(key_code("volume_up"), Some(key::VOLUME_UP));
        assert_eq!(key_code("nonsense"), None);
    }

    #[test]
    fn app_links_resolve() {
        assert_eq!(app_link("netflix"), "market://launch?id=com.netflix.ninja");
        assert_eq!(app_link("com.some.pkg"), "market://launch?id=com.some.pkg");
        assert_eq!(
            app_link("https://www.youtube.com/watch?v=x"),
            "https://www.youtube.com/watch?v=x"
        );
    }

    #[test]
    fn youtube_ids_are_pulled() {
        assert_eq!(youtube_id("aqz-KE-bpKQ"), "aqz-KE-bpKQ");
        assert_eq!(
            youtube_id("https://www.youtube.com/watch?v=aqz-KE-bpKQ&t=1s"),
            "aqz-KE-bpKQ"
        );
    }
}
