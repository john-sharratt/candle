//! The Google Cast profile: a Chromecast, a Cast-built-in TV or a Cast speaker,
//! over CASTv2 ([`crate::proto::cast`]) on port 8009.
//!
//! It reads what the device is doing and drives it: volume and mute on the
//! receiver, play/pause/stop/skip on whatever is casting, casting a media URL
//! to it from nothing, and launching or quitting an app. A Cast reply is JSON;
//! the receiver namespace carries volume and the running app, the media
//! namespace what is playing. A device off the house network is refused.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::proto::cast::{self, Conn, NS_MEDIA, NS_RECEIVER, RECEIVER};
use crate::scope::{in_house, LocalNet};

const ANSWER: Duration = Duration::from_secs(5);
/// Google's Default Media Receiver — the app that plays an arbitrary media URL.
const DEFAULT_MEDIA_RECEIVER: &str = "CC1AD845";

/// Apps that can be launched by a friendly name; anything else is taken as a
/// raw Cast application id.
const APPS: &[(&str, &str)] = &[
    ("youtube", "233637DE"),
    ("netflix", "CA5E8412"),
    ("spotify", "CC32E753"),
    ("plex", "9AC194DC"),
    ("youtube-music", "2DAD5533"),
    ("default", DEFAULT_MEDIA_RECEIVER),
];

/// The media content type to declare for a URL, guessed from its extension.
fn content_type(url: &str) -> &'static str {
    let u = url.to_ascii_lowercase();
    let ext = u.rsplit('.').next().unwrap_or("");
    match ext {
        "mp3" => "audio/mpeg",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "m3u8" => "application/vnd.apple.mpegurl",
        "mpd" => "application/dash+xml",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        _ => "video/mp4",
    }
}

/// The running app on a receiver, from a `RECEIVER_STATUS`.
#[derive(Debug, Default, Clone)]
struct App {
    app_id: String,
    display_name: String,
    session_id: String,
    transport_id: String,
    status_text: String,
}

/// What the receiver itself is doing.
#[derive(Debug, Default)]
struct Receiver {
    volume_pct: Option<u8>,
    muted: Option<bool>,
    app: Option<App>,
}

fn parse_receiver(v: &Value) -> Receiver {
    let st = &v["status"];
    let volume_pct = st["volume"]["level"]
        .as_f64()
        .map(|x| (x * 100.0).round().clamp(0.0, 100.0) as u8);
    let app = st["applications"]
        .as_array()
        .and_then(|a| a.first())
        .map(|a| App {
            app_id: a["appId"].as_str().unwrap_or_default().to_string(),
            display_name: a["displayName"].as_str().unwrap_or_default().to_string(),
            session_id: a["sessionId"].as_str().unwrap_or_default().to_string(),
            transport_id: a["transportId"].as_str().unwrap_or_default().to_string(),
            status_text: a["statusText"].as_str().unwrap_or_default().to_string(),
        });
    Receiver {
        volume_pct,
        muted: st["volume"]["muted"].as_bool(),
        app,
    }
}

/// What a media session is playing. Times are whole seconds, kept `Option` so
/// a source that does not report them is simply silent about position.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct NowPlaying {
    pub session: u64,
    pub state: String,
    pub title: String,
    pub artist: String,
    pub current_time: Option<u64>,
    pub duration: Option<u64>,
    pub idle_reason: String,
}

/// `m:ss`, or `h:mm:ss` past an hour.
fn mmss(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// The media session in a `MEDIA_STATUS`, if any.
pub fn parse_media(v: &Value) -> Option<NowPlaying> {
    let s = v["status"].as_array().and_then(|a| a.first())?;
    let md = &s["media"]["metadata"];
    let secs = |x: &Value| x.as_f64().filter(|n| *n >= 0.0).map(|n| n as u64);
    Some(NowPlaying {
        session: s["mediaSessionId"].as_u64().unwrap_or(0),
        state: s["playerState"].as_str().unwrap_or_default().to_string(),
        title: md["title"].as_str().unwrap_or_default().to_string(),
        artist: md["artist"]
            .as_str()
            .or_else(|| md["subtitle"].as_str())
            .or_else(|| md["albumName"].as_str())
            .unwrap_or_default()
            .to_string(),
        current_time: secs(&s["currentTime"]),
        duration: secs(&s["media"]["duration"]),
        idle_reason: s["idleReason"].as_str().unwrap_or_default().to_string(),
    })
}

fn playing_line(np: &NowPlaying) -> String {
    let what = match (np.title.is_empty(), np.artist.is_empty()) {
        (false, false) => format!("\"{}\" by {}", np.title, np.artist),
        (false, true) => format!("\"{}\"", np.title),
        (true, _) => "something it does not name".to_string(),
    };
    let lead = match np.state.as_str() {
        "PLAYING" => "is playing",
        "PAUSED" => "is paused on",
        "BUFFERING" => "is buffering",
        "IDLE" => "is idle, last on",
        _ => "has",
    };
    let pos = match (np.current_time, np.duration) {
        (Some(c), Some(d)) if d > 0 => format!(" ({} of {})", mmss(c), mmss(d)),
        (Some(c), _) if c > 0 => format!(" ({} in)", mmss(c)),
        _ => String::new(),
    };
    format!("{lead} {what}{pos}")
}

/// A Google Cast device at its address on the house network.
pub struct Cast {
    address: Ipv4Addr,
}

impl Cast {
    pub fn at(address: &str, nets: &[LocalNet]) -> Result<Cast> {
        let address: Ipv4Addr = address.parse()?;
        if !in_house(nets, address) {
            bail!("{address} is not on the house network");
        }
        Ok(Cast { address })
    }

    async fn conn(&self) -> Result<Conn> {
        Conn::open(self.address, ANSWER).await
    }

    async fn receiver_status(&self, c: &mut Conn, id: u64) -> Result<Receiver> {
        c.connect_to(RECEIVER).await?;
        let v = c
            .request(
                RECEIVER,
                NS_RECEIVER,
                &json!({"type": "GET_STATUS", "requestId": id}).to_string(),
                id,
            )
            .await?;
        Ok(parse_receiver(&v))
    }

    /// The app running on the receiver, or a refusal that nothing is casting.
    async fn running_app(&self, c: &mut Conn, id: u64) -> Result<App> {
        self.receiver_status(c, id)
            .await?
            .app
            .filter(|a| !a.transport_id.is_empty())
            .ok_or_else(|| anyhow!("nothing is casting to it right now"))
    }

    async fn media_status(
        &self,
        c: &mut Conn,
        transport: &str,
        id: u64,
    ) -> Result<Option<NowPlaying>> {
        c.connect_to(transport).await?;
        let v = c
            .request(
                transport,
                NS_MEDIA,
                &json!({"type": "GET_STATUS", "requestId": id}).to_string(),
                id,
            )
            .await?;
        Ok(parse_media(&v))
    }

    /// The line [`work`] returns for `status`: volume, what app, what playing.
    pub async fn status(&self, name: &str) -> Result<String> {
        let mut c = self.conn().await?;
        let r = self.receiver_status(&mut c, 1).await?;
        let mut parts = Vec::new();
        match &r.app {
            Some(a) if !a.display_name.is_empty() => {
                parts.push(format!("is on {}", a.display_name))
            }
            _ => parts.push("is idle".to_string()),
        }
        if let Some(a) = &r.app {
            if !a.transport_id.is_empty() {
                if let Some(np) = self.media_status(&mut c, &a.transport_id, 2).await? {
                    if !np.title.is_empty() {
                        parts.push(playing_line(&np));
                    } else if !a.status_text.is_empty() {
                        parts.push(a.status_text.clone());
                    }
                }
            }
        }
        if let Some(v) = r.volume_pct {
            parts.push(format!("at volume {v} of 100"));
        }
        if r.muted == Some(true) {
            parts.push("muted".to_string());
        }
        Ok(format!("The {name} {}.", parts.join(", ")))
    }

    async fn set_volume(&self, pct: u8) -> Result<u8> {
        let level = (pct.min(100) as f64) / 100.0;
        let mut c = self.conn().await?;
        c.connect_to(RECEIVER).await?;
        let v = c
            .request(
                RECEIVER,
                NS_RECEIVER,
                &json!({"type": "SET_VOLUME", "volume": {"level": level}, "requestId": 1})
                    .to_string(),
                1,
            )
            .await?;
        Ok(parse_receiver(&v).volume_pct.unwrap_or(pct.min(100)))
    }

    async fn step_volume(&self, up: bool) -> Result<u8> {
        let mut c = self.conn().await?;
        let now = self
            .receiver_status(&mut c, 1)
            .await?
            .volume_pct
            .unwrap_or(30);
        let next = if up {
            now.saturating_add(5).min(100)
        } else {
            now.saturating_sub(5)
        };
        self.set_volume(next).await
    }

    async fn set_mute(&self, muted: bool) -> Result<()> {
        let mut c = self.conn().await?;
        c.connect_to(RECEIVER).await?;
        c.request(
            RECEIVER,
            NS_RECEIVER,
            &json!({"type": "SET_VOLUME", "volume": {"muted": muted}, "requestId": 1}).to_string(),
            1,
        )
        .await?;
        Ok(())
    }

    /// Play/pause/stop/skip on whatever is casting.
    async fn media_command(&self, kind: &str) -> Result<()> {
        let mut c = self.conn().await?;
        let app = self.running_app(&mut c, 1).await?;
        let np = self
            .media_status(&mut c, &app.transport_id, 2)
            .await?
            .ok_or_else(|| anyhow!("nothing is playing on it to control"))?;
        c.tell(
            &app.transport_id,
            NS_MEDIA,
            &json!({"type": kind, "mediaSessionId": np.session, "requestId": 3}).to_string(),
        )
        .await
    }

    /// The transport of the Default Media Receiver if it is up and ready.
    async fn media_transport(&self, c: &mut Conn) -> Result<Option<String>> {
        Ok(self
            .receiver_status(c, 1)
            .await?
            .app
            .filter(|a| {
                a.app_id.eq_ignore_ascii_case(DEFAULT_MEDIA_RECEIVER) && !a.transport_id.is_empty()
            })
            .map(|a| a.transport_id))
    }

    /// Launch the Default Media Receiver (reusing it if already up) and return
    /// its transport id, waiting for it to register one.
    async fn launch_media(&self, c: &mut Conn) -> Result<String> {
        if let Some(t) = self.media_transport(c).await? {
            return Ok(t);
        }
        let v = c
            .request(
                RECEIVER,
                NS_RECEIVER,
                &json!({"type": "LAUNCH", "appId": DEFAULT_MEDIA_RECEIVER, "requestId": 2})
                    .to_string(),
                2,
            )
            .await?;
        cast::not_an_error(&v)?;
        // The app reports its transport a moment after the LAUNCH is acked.
        for _ in 0..12 {
            if let Some(t) = self.media_transport(c).await? {
                return Ok(t);
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        bail!("the device would not start its media player")
    }

    /// Cast a media URL to the device and start it, optionally naming it.
    async fn cast_url(&self, url: &str, title: Option<&str>) -> Result<()> {
        let mut c = self.conn().await?;
        c.connect_to(RECEIVER).await?;
        let transport = self.launch_media(&mut c).await?;
        c.connect_to(&transport).await?;
        let mut media = json!({
            "contentId": url,
            "streamType": "BUFFERED",
            "contentType": content_type(url),
        });
        if let Some(t) = title {
            media["metadata"] = json!({"metadataType": 0, "title": t});
        }
        let mut load =
            json!({ "type": "LOAD", "autoplay": true, "currentTime": 0, "media": media });
        // The media channel can need a moment after launch; try, wait, retry.
        for (attempt, id) in [3u64, 4].into_iter().enumerate() {
            load["requestId"] = json!(id);
            let v = c
                .request(&transport, NS_MEDIA, &load.to_string(), id)
                .await?;
            match cast::not_an_error(&v) {
                Ok(()) => return Ok(()),
                Err(e) if attempt == 1 => return Err(e),
                Err(_) => tokio::time::sleep(Duration::from_millis(800)).await,
            }
        }
        Ok(())
    }

    /// Jump to `seconds` into whatever is playing.
    async fn seek(&self, seconds: u64) -> Result<()> {
        let mut c = self.conn().await?;
        let app = self.running_app(&mut c, 1).await?;
        let np = self
            .media_status(&mut c, &app.transport_id, 2)
            .await?
            .ok_or_else(|| anyhow!("nothing is playing to seek"))?;
        c.tell(
            &app.transport_id,
            NS_MEDIA,
            &json!({"type": "SEEK", "mediaSessionId": np.session, "currentTime": seconds, "requestId": 3})
                .to_string(),
        )
        .await
    }

    /// Whether `app_id` is installed and launchable on the device.
    async fn app_available(&self, c: &mut Conn, app_id: &str) -> Result<bool> {
        let v = c
            .request(
                RECEIVER,
                NS_RECEIVER,
                &json!({"type": "GET_APP_AVAILABILITY", "appId": [app_id], "requestId": 1})
                    .to_string(),
                1,
            )
            .await?;
        Ok(v["availability"][app_id].as_str() == Some("APP_AVAILABLE"))
    }

    /// Launch an app by friendly name or raw application id, returning the name
    /// the device shows for what actually came up.
    async fn launch(&self, app: &str) -> Result<String> {
        let id = APPS
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(app.trim()))
            .map(|(_, id)| *id)
            .unwrap_or(app.trim());
        let mut c = self.conn().await?;
        c.connect_to(RECEIVER).await?;
        // Ask first, so an app the device does not have is a clear answer
        // rather than a raw launch error.
        if !self.app_available(&mut c, id).await? {
            bail!("{} is not installed on it", app.trim());
        }
        let v = c
            .request(
                RECEIVER,
                NS_RECEIVER,
                &json!({"type": "LAUNCH", "appId": id, "requestId": 2}).to_string(),
                2,
            )
            .await?;
        cast::not_an_error(&v)?;
        // The app may still be starting when the LAUNCH is acked, so read the
        // receiver again for what is actually running. A pure cast receiver
        // with no sender holding it open can idle straight back, so `launch` is
        // best-effort — `cast_url` persists because media is playing.
        let running = self.receiver_status(&mut c, 3).await?.app;
        match running {
            Some(a) if a.app_id.eq_ignore_ascii_case(id) => Ok(if a.display_name.is_empty() {
                app.trim().to_string()
            } else {
                a.display_name
            }),
            Some(a) if !a.display_name.is_empty() => Ok(a.display_name),
            _ => bail!("the device did not keep it open — it may not be installed, or needs its own remote to open"),
        }
    }

    /// Quit whatever app is running.
    async fn quit(&self) -> Result<()> {
        let mut c = self.conn().await?;
        let app = self.running_app(&mut c, 1).await?;
        c.tell(
            RECEIVER,
            NS_RECEIVER,
            &json!({"type": "STOP", "sessionId": app.session_id, "requestId": 2}).to_string(),
        )
        .await
    }

    async fn now_playing(&self, name: &str) -> Result<String> {
        let mut c = self.conn().await?;
        let app = match self.receiver_status(&mut c, 1).await?.app {
            Some(a) if !a.transport_id.is_empty() => a,
            _ => return Ok(format!("The {name} is not casting anything right now.")),
        };
        match self.media_status(&mut c, &app.transport_id, 2).await? {
            Some(np) if !np.title.is_empty() => Ok(format!("The {name} {}.", playing_line(&np))),
            _ => Ok(format!(
                "The {name} is on {} with nothing it names.",
                app.display_name
            )),
        }
    }
}

/// Work `action` on the Cast device `name` at `address`, returning the line its
/// body reads.
pub async fn work(
    name: &str,
    address: &str,
    action: &str,
    args: &serde_json::Map<String, Value>,
    nets: &[LocalNet],
) -> Result<String> {
    let cast = Cast::at(address, nets)?;
    let arg = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let line = match action {
        "status" => cast.status(name).await?,
        "playing" => cast.now_playing(name).await?,
        "volume" => {
            let v = args
                .get("volume")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the volume, 0 to 100."))?;
            let set = cast.set_volume(v.min(100) as u8).await?;
            format!("The {name} is at volume {set} of 100.")
        }
        "louder" => {
            let v = cast.step_volume(true).await?;
            format!("The {name} is at volume {v} of 100.")
        }
        "quieter" => {
            let v = cast.step_volume(false).await?;
            format!("The {name} is at volume {v} of 100.")
        }
        "mute" => {
            cast.set_mute(true).await?;
            format!("The {name} is muted.")
        }
        "unmute" => {
            cast.set_mute(false).await?;
            format!("The {name} is unmuted.")
        }
        "play" => {
            cast.media_command("PLAY").await?;
            format!("The {name} is playing.")
        }
        "pause" => {
            cast.media_command("PAUSE").await?;
            format!("The {name} is paused.")
        }
        "stop" => {
            cast.media_command("STOP").await?;
            format!("The {name} is stopped.")
        }
        "next" => {
            cast.media_command("QUEUE_NEXT").await?;
            format!("The {name} skipped to the next.")
        }
        "previous" | "prev" => {
            cast.media_command("QUEUE_PREV").await?;
            format!("The {name} went to the previous.")
        }
        "cast" => {
            let url = arg("url").ok_or_else(|| anyhow!("Say the media URL to cast."))?;
            let title = arg("title");
            cast.cast_url(&url, title.as_deref()).await?;
            match title {
                Some(t) => format!("The {name} is casting \"{t}\"."),
                None => format!("The {name} is casting {url}."),
            }
        }
        "seek" => {
            let s = args
                .get("seconds")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("Say the position in seconds."))?;
            cast.seek(s).await?;
            format!("The {name} jumped to {}.", mmss(s))
        }
        "launch" => {
            let app = arg("app").ok_or_else(|| anyhow!("Say which app to launch."))?;
            let got = cast.launch(&app).await?;
            format!("The {name} is on {got}.")
        }
        "quit" => {
            cast.quit().await?;
            format!("The {name}'s app was closed.")
        }
        other => bail!("a Cast device has no `{other}`"),
    };
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_types_are_guessed_from_the_url() {
        assert_eq!(content_type("http://h/clip.mp4"), "video/mp4");
        assert_eq!(content_type("http://h/song.mp3"), "audio/mpeg");
        assert_eq!(
            content_type("http://h/live.m3u8"),
            "application/vnd.apple.mpegurl"
        );
        assert_eq!(content_type("http://h/stream"), "video/mp4");
    }

    #[test]
    fn a_receiver_status_is_read() {
        let v = json!({"status": {
            "volume": {"level": 0.4, "muted": false},
            "applications": [{
                "displayName": "Default Media Receiver",
                "sessionId": "S1",
                "transportId": "T1",
                "statusText": "Casting"
            }]
        }});
        let r = parse_receiver(&v);
        assert_eq!(r.volume_pct, Some(40));
        assert_eq!(r.muted, Some(false));
        let app = r.app.unwrap();
        assert_eq!(app.transport_id, "T1");
        assert_eq!(app.display_name, "Default Media Receiver");
    }

    #[test]
    fn a_media_status_reads_as_what_is_playing() {
        let v = json!({"status": [{
            "mediaSessionId": 7,
            "playerState": "PLAYING",
            "media": {"metadata": {"title": "Harbour Lights", "artist": "The Waves"}}
        }]});
        let np = parse_media(&v).unwrap();
        assert_eq!(np.session, 7);
        assert_eq!(
            playing_line(&np),
            "is playing \"Harbour Lights\" by The Waves"
        );
        // No media session at all.
        assert_eq!(parse_media(&json!({"status": []})), None);
    }

    #[test]
    fn a_device_off_the_house_network_is_refused() {
        let home = [LocalNet {
            own: "192.168.0.237".parse().unwrap(),
            mask: "255.255.255.0".parse().unwrap(),
        }];
        assert!(Cast::at("192.168.0.214", &home).is_ok());
        assert!(Cast::at("8.8.8.8", &home).is_err());
    }
}
