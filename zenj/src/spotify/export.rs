//! Spotify's privacy export ("Download your data"): every play in the
//! extended streaming history, and the account data's liked tracks and
//! playlists — a person's taste as Spotify recorded it, read into plays and
//! likes per track (§4.1).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

/// One record of `Streaming_History_Audio_*.json`.
#[derive(Debug, Deserialize)]
struct Play {
    ms_played: u64,
    #[serde(default)]
    spotify_track_uri: Option<String>,
    #[serde(default)]
    master_metadata_track_name: Option<String>,
    #[serde(default)]
    master_metadata_album_artist_name: Option<String>,
    #[serde(default)]
    skipped: Option<bool>,
}

/// `YourLibrary.json`.
#[derive(Debug, Deserialize)]
struct Library {
    #[serde(default)]
    tracks: Vec<LibraryTrack>,
}

#[derive(Debug, Deserialize)]
struct LibraryTrack {
    #[serde(default)]
    artist: String,
    #[serde(default)]
    track: String,
    uri: String,
}

/// `Playlist*.json`.
#[derive(Debug, Deserialize)]
struct Playlists {
    playlists: Vec<Playlist>,
}

#[derive(Debug, Deserialize)]
struct Playlist {
    name: String,
    #[serde(default)]
    items: Vec<PlaylistItem>,
}

#[derive(Debug, Deserialize)]
struct PlaylistItem {
    #[serde(default)]
    track: Option<PlaylistTrack>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlaylistTrack {
    #[serde(default)]
    track_name: String,
    #[serde(default)]
    artist_name: String,
    track_uri: String,
}

/// A play under this long is not a listen.
pub const LISTEN_MS: u64 = 30_000;

/// What one person's export says of one track.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Heard {
    pub title: String,
    pub artist: String,
    /// Plays of 30 s or more.
    pub listens: u32,
    pub skips: u32,
    pub liked: bool,
    /// The playlists it is in, by name.
    pub playlists: Vec<String>,
}

/// One person's taste, by Spotify track id.
pub type Taste = BTreeMap<String, Heard>;

/// The track id of a `spotify:track:<id>` URI.
pub fn track_id(uri: &str) -> Option<&str> {
    uri.strip_prefix("spotify:track:")
        .filter(|id| !id.is_empty())
}

fn entry<'a>(taste: &'a mut Taste, uri: &str, title: &str, artist: &str) -> Option<&'a mut Heard> {
    let id = track_id(uri)?;
    let h = taste.entry(id.to_string()).or_default();
    if h.title.is_empty() {
        h.title = title.to_string();
        h.artist = artist.to_string();
    }
    Some(h)
}

/// Fold one streaming history file's `text` into `taste`. A record with no
/// track — a podcast, a play with the URI withheld — is skipped.
pub fn add_history(taste: &mut Taste, text: &str) -> Result<usize, String> {
    let plays: Vec<Play> = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut n = 0;
    for p in plays {
        let Some(uri) = p.spotify_track_uri.as_deref() else {
            continue;
        };
        let title = p.master_metadata_track_name.as_deref().unwrap_or("");
        let artist = p.master_metadata_album_artist_name.as_deref().unwrap_or("");
        let Some(h) = entry(taste, uri, title, artist) else {
            continue;
        };
        if p.ms_played >= LISTEN_MS {
            h.listens += 1;
        }
        if p.skipped == Some(true) {
            h.skips += 1;
        }
        n += 1;
    }
    Ok(n)
}

/// Fold `YourLibrary.json`'s `text` into `taste`: every track in it liked.
pub fn add_library(taste: &mut Taste, text: &str) -> Result<usize, String> {
    let lib: Library = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut n = 0;
    for t in lib.tracks {
        if let Some(h) = entry(taste, &t.uri, &t.track, &t.artist) {
            h.liked = true;
            n += 1;
        }
    }
    Ok(n)
}

/// Fold a `Playlist*.json`'s `text` into `taste`.
pub fn add_playlists(taste: &mut Taste, text: &str) -> Result<usize, String> {
    let pls: Playlists = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut n = 0;
    for pl in pls.playlists {
        for item in pl.items {
            let Some(t) = item.track else { continue };
            if let Some(h) = entry(taste, &t.track_uri, &t.track_name, &t.artist_name) {
                if !h.playlists.contains(&pl.name) {
                    h.playlists.push(pl.name.clone());
                }
                n += 1;
            }
        }
    }
    Ok(n)
}

/// One person's taste from their unpacked export at `dir`: every history,
/// library and playlist file under it, at any depth.
pub fn read_dir(dir: &Path) -> Result<Taste, String> {
    let mut taste = Taste::new();
    let mut stack = vec![dir.to_path_buf()];
    let mut files = Vec::new();
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    for p in files {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let add: fn(&mut Taste, &str) -> Result<usize, String> =
            if name.starts_with("Streaming_History_Audio") && name.ends_with(".json") {
                add_history
            } else if name == "YourLibrary.json" {
                add_library
            } else if name.starts_with("Playlist") && name.ends_with(".json") {
                add_playlists
            } else {
                continue;
            };
        let text = fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        add(&mut taste, &text).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(taste)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HISTORY: &str = r#"[
      {"ts": "2024-05-01T20:00:00Z", "ms_played": 210000, "spotify_track_uri": "spotify:track:AAA",
       "master_metadata_track_name": "Song A", "master_metadata_album_artist_name": "Artist A", "skipped": false},
      {"ts": "2024-05-02T20:00:00Z", "ms_played": 12000, "spotify_track_uri": "spotify:track:AAA",
       "master_metadata_track_name": "Song A", "master_metadata_album_artist_name": "Artist A", "skipped": true},
      {"ts": "2024-05-03T20:00:00Z", "ms_played": 900000, "spotify_track_uri": null,
       "episode_name": "A podcast"},
      {"ts": "2024-05-04T20:00:00Z", "ms_played": 40000, "spotify_track_uri": "spotify:track:BBB",
       "master_metadata_track_name": "Song B", "master_metadata_album_artist_name": "Artist B", "skipped": null}
    ]"#;

    /// **Plays fold into listens and skips per track**; a record with no
    /// track URI is skipped, as the export allows.
    #[test]
    fn the_history_folds_into_listens_and_skips() {
        let mut t = Taste::new();
        assert_eq!(add_history(&mut t, HISTORY).unwrap(), 3);
        assert_eq!(
            t["AAA"],
            Heard {
                title: "Song A".into(),
                artist: "Artist A".into(),
                listens: 1,
                skips: 1,
                liked: false,
                playlists: vec![],
            }
        );
        assert_eq!((t["BBB"].listens, t["BBB"].skips), (1, 0));
    }

    #[test]
    fn likes_and_playlists_fold_in_beside_the_plays() {
        let mut t = Taste::new();
        add_history(&mut t, HISTORY).unwrap();
        add_library(
            &mut t,
            r#"{"tracks": [{"artist": "Artist C", "album": "X", "track": "Song C", "uri": "spotify:track:CCC"},
                           {"artist": "Artist A", "album": "X", "track": "Song A", "uri": "spotify:track:AAA"}],
                "albums": []}"#,
        )
        .unwrap();
        add_playlists(
            &mut t,
            r#"{"playlists": [{"name": "Party", "lastModifiedDate": "2024-01-01",
                "items": [{"track": {"trackName": "Song B", "artistName": "Artist B",
                           "albumName": "Y", "trackUri": "spotify:track:BBB"}, "episode": null},
                          {"track": null, "episode": {"episodeName": "x"}}]}]}"#,
        )
        .unwrap();
        assert!(t["AAA"].liked && t["CCC"].liked && !t["BBB"].liked);
        assert_eq!(t["CCC"].title, "Song C");
        assert_eq!(t["BBB"].playlists, vec!["Party".to_string()]);
    }

    #[test]
    fn an_unpacked_export_is_read_whole() {
        let dir = tempfile::tempdir().unwrap();
        let deep = dir.path().join("Spotify Extended Streaming History");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("Streaming_History_Audio_2024_0.json"), HISTORY).unwrap();
        fs::write(deep.join("ReadMeFirst.pdf"), b"%PDF").unwrap();
        fs::write(
            dir.path().join("YourLibrary.json"),
            r#"{"tracks": [{"artist": "A", "track": "T", "uri": "spotify:track:DDD"}]}"#,
        )
        .unwrap();
        let t = read_dir(dir.path()).unwrap();
        assert_eq!(t.len(), 3);
        assert!(t["DDD"].liked);
    }
}
