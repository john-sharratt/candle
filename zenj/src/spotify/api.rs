//! The Spotify Web API as ZenJ reads it, in development mode (§4.1): the
//! signed-in person's playlists and liked tracks, a playlist's items when its
//! `snapshot_id` moved, and tracks by id for their ISRC. Responses page at 50
//! (`next`).

use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::oauth::Account;

pub const API: &str = "https://api.spotify.com/v1";

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Artist {
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ExternalIds {
    #[serde(default)]
    pub isrc: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Album {
    #[serde(default)]
    pub release_date: Option<String>,
}

/// A track object.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Track {
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub artists: Vec<Artist>,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub external_ids: ExternalIds,
    #[serde(default)]
    pub album: Album,
}

impl Track {
    /// The year its album came out.
    pub fn year(&self) -> Option<u16> {
        self.album.release_date.as_deref()?.get(..4)?.parse().ok()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct User {
    pub id: String,
    #[serde(default)]
    pub display_name: Option<String>,
}

/// One playlist item: the track, who added it and when.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Item {
    #[serde(default, alias = "item")]
    pub track: Option<Track>,
    #[serde(default)]
    pub added_at: Option<String>,
    #[serde(default)]
    pub added_by: Option<User>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Playlist {
    pub id: String,
    pub name: String,
    pub snapshot_id: String,
}

/// A saved track.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Saved {
    pub track: Track,
}

#[derive(Debug, Deserialize)]
struct Page<T> {
    items: Vec<T>,
    #[serde(default)]
    next: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Tracks {
    tracks: Vec<Option<Track>>,
}

#[derive(Debug, Deserialize)]
struct Snapshot {
    snapshot_id: String,
}

/// One person's Spotify.
pub struct Spotify {
    pub account: Account,
    http: reqwest::Client,
}

impl Spotify {
    pub fn new(account: Account) -> Spotify {
        Spotify {
            account,
            http: reqwest::Client::new(),
        }
    }

    async fn get<T: DeserializeOwned>(&self, url: &str) -> Result<T, String> {
        let token = self.account.access().await?;
        let res = self
            .http
            .get(url)
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| format!("{url}: {e}"))?;
        let status = res.status();
        let text = res.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("{url}: {status}: {text}"));
        }
        serde_json::from_str(&text).map_err(|e| format!("{url}: {e}"))
    }

    async fn all<T: DeserializeOwned>(&self, first: String) -> Result<Vec<T>, String> {
        let mut out = Vec::new();
        let mut next = Some(first);
        while let Some(url) = next {
            let page: Page<T> = self.get(&url).await?;
            out.extend(page.items);
            next = page.next;
        }
        Ok(out)
    }

    pub async fn me(&self) -> Result<User, String> {
        self.get(&format!("{API}/me")).await
    }

    pub async fn playlists(&self) -> Result<Vec<Playlist>, String> {
        self.all(format!("{API}/me/playlists?limit=50")).await
    }

    pub async fn liked(&self) -> Result<Vec<Saved>, String> {
        self.all(format!("{API}/me/tracks?limit=50")).await
    }

    /// The playlist's snapshot: it moves whenever the playlist changes.
    pub async fn snapshot(&self, playlist: &str) -> Result<String, String> {
        let s: Snapshot = self
            .get(&format!("{API}/playlists/{playlist}?fields=snapshot_id"))
            .await?;
        Ok(s.snapshot_id)
    }

    pub async fn items(&self, playlist: &str) -> Result<Vec<Item>, String> {
        self.all(format!("{API}/playlists/{playlist}/items?limit=50"))
            .await
    }

    /// The tracks with `ids`, up to 50 at a time; an id Spotify has no
    /// track for is left out.
    pub async fn tracks(&self, ids: &[String]) -> Result<Vec<Track>, String> {
        let mut out = Vec::new();
        for chunk in ids.chunks(50) {
            let t: Tracks = self
                .get(&format!("{API}/tracks?ids={}", chunk.join(",")))
                .await?;
            out.extend(t.tracks.into_iter().flatten());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A playlist item reads under either name** the API has given its
    /// track — `track`, or `item` on the `/items` path — and a removed one
    /// as none.
    #[test]
    fn items_read_their_track() {
        let page: Page<Item> = serde_json::from_str(
            r#"{"items": [
                {"added_at": "2026-12-31T22:01:00Z", "added_by": {"id": "yen"},
                 "item": {"id": "T1", "name": "Dancing Queen", "artists": [{"name": "ABBA"}],
                          "duration_ms": 230000, "external_ids": {"isrc": "SEAYD7601020"},
                          "album": {"release_date": "1976-08-16"}}},
                {"added_at": "2026-12-31T22:02:00Z", "track": {"id": "T2", "name": "x"}},
                {"added_at": "2026-12-31T22:03:00Z", "track": null}
            ], "next": null}"#,
        )
        .unwrap();
        let t = page.items[0].track.as_ref().unwrap();
        assert_eq!(t.external_ids.isrc.as_deref(), Some("SEAYD7601020"));
        assert_eq!(t.year(), Some(1976));
        assert_eq!(page.items[0].added_by.as_ref().unwrap().id, "yen");
        assert_eq!(page.items[1].track.as_ref().unwrap().id.as_deref(), Some("T2"));
        assert_eq!(page.items[2].track, None);
        assert_eq!(page.next, None);
    }
}
