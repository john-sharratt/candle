//! TIDAL's API v2 (`openapi.tidal.com/v2`, JSON:API) as ZenJ drives it
//! (§4.1): tracks by ISRC and by search, with their artists, tempo and key;
//! the playlists ZenJ owns — its crates and *NYE Next* — made, read, added
//! to, taken from and reordered. Responses page at 20 items (`links.next`).
//!
//! Request bodies and response reading are pure functions, so both are
//! tested against exact JSON; [`Tidal`] only sends them.

use reqwest::Method;
use serde::Deserialize;
use serde_json::{json, Value};
use url::form_urlencoded::byte_serialize;

use crate::music::Camelot;
use crate::oauth::Account;

pub const API: &str = "https://openapi.tidal.com/v2";
const JSON_API: &str = "application/vnd.api+json";

/// One TIDAL track.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub id: String,
    pub title: String,
    pub artists: Vec<String>,
    pub isrc: Option<String>,
    pub duration_ms: u64,
    pub bpm: Option<f32>,
    pub key: Option<Camelot>,
}

/// One item of a playlist: the track, and the item's own id, which removing
/// and moving it name.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub track: String,
    pub item: String,
}

/// An ISO 8601 duration — `PT3M20S`, `PT1H2M3.5S` — in milliseconds.
pub fn duration_ms(iso: &str) -> Option<u64> {
    let rest = iso.strip_prefix("PT")?;
    let mut total = 0f64;
    let mut num = String::new();
    for c in rest.chars() {
        match c {
            '0'..='9' | '.' => num.push(c),
            'H' | 'M' | 'S' => {
                let v: f64 = num.parse().ok()?;
                num.clear();
                total += v * match c {
                    'H' => 3_600_000.0,
                    'M' => 60_000.0,
                    _ => 1_000.0,
                };
            }
            _ => return None,
        }
    }
    num.is_empty().then_some(total.round() as u64)
}

#[derive(Debug, Deserialize)]
struct Resource {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    attributes: Value,
    #[serde(default)]
    relationships: Value,
    #[serde(default)]
    meta: Value,
}

#[derive(Debug, Deserialize)]
struct Document {
    #[serde(default)]
    data: Value,
    #[serde(default)]
    included: Vec<Resource>,
    #[serde(default)]
    links: Value,
}

fn artist_names(r: &Resource, included: &[Resource]) -> Vec<String> {
    r.relationships["artists"]["data"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x["id"].as_str())
                .filter_map(|id| {
                    included
                        .iter()
                        .find(|i| i.kind == "artists" && i.id == id)
                        .and_then(|i| i.attributes["name"].as_str())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn track_of(r: &Resource, included: &[Resource]) -> Track {
    let a = &r.attributes;
    let key = match (a["key"].as_str(), a["keyScale"].as_str()) {
        (Some(k), Some(s)) => Camelot::of_tidal(k, s),
        _ => None,
    };
    Track {
        id: r.id.clone(),
        title: a["title"].as_str().unwrap_or_default().to_string(),
        artists: artist_names(r, included),
        isrc: a["isrc"].as_str().map(str::to_string),
        duration_ms: a["duration"].as_str().and_then(duration_ms).unwrap_or(0),
        bpm: a["bpm"].as_f64().map(|b| b as f32).filter(|b| *b > 0.0),
        key,
    }
}

/// The tracks a document holds — as its data, or included beside it — and
/// the next page's address.
pub fn read_tracks(text: &str) -> Result<(Vec<Track>, Option<String>), String> {
    let doc: Document = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let data: Vec<Resource> = match &doc.data {
        Value::Array(_) => serde_json::from_value(doc.data.clone()).map_err(|e| e.to_string())?,
        Value::Object(_) => {
            vec![serde_json::from_value(doc.data.clone()).map_err(|e| e.to_string())?]
        }
        _ => Vec::new(),
    };
    let mut tracks: Vec<Track> = data
        .iter()
        .filter(|r| r.kind == "tracks" && !r.attributes.is_null())
        .map(|r| track_of(r, &doc.included))
        .collect();
    // A relationship document names its tracks and includes them whole.
    for r in data
        .iter()
        .filter(|r| r.kind == "tracks" && r.attributes.is_null())
    {
        if let Some(full) = doc
            .included
            .iter()
            .find(|i| i.kind == "tracks" && i.id == r.id)
        {
            tracks.push(track_of(full, &doc.included));
        }
    }
    let next = doc.links["next"].as_str().map(str::to_string);
    Ok((tracks, next))
}

/// A playlist's items, and the next page's address.
pub fn read_items(text: &str) -> Result<(Vec<Item>, Option<String>), String> {
    let doc: Document = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let data: Vec<Resource> = serde_json::from_value(doc.data).map_err(|e| e.to_string())?;
    let items = data
        .into_iter()
        .filter(|r| r.kind == "tracks")
        .map(|r| Item {
            item: r.meta["itemId"].as_str().unwrap_or_default().to_string(),
            track: r.id,
        })
        .collect();
    Ok((items, doc.links["next"].as_str().map(str::to_string)))
}

/// The id of the resource a document is.
pub fn read_id(text: &str) -> Result<String, String> {
    let doc: Document = serde_json::from_str(text).map_err(|e| e.to_string())?;
    doc.data["id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("no id in {text}"))
}

/// A new playlist's body.
pub fn new_playlist(name: &str, description: &str) -> Value {
    json!({"data": {"type": "playlists", "attributes": {
        "name": name, "description": description, "accessType": "UNLISTED"}}})
}

/// The body adding `tracks`, before item `before` when given (else at the
/// end).
pub fn add_items(tracks: &[String], before: Option<&str>) -> Value {
    let data: Vec<Value> = tracks
        .iter()
        .map(|t| json!({"id": t, "type": "tracks"}))
        .collect();
    match before {
        Some(b) => json!({"data": data, "meta": {"positionBefore": b}}),
        None => json!({"data": data}),
    }
}

/// The body naming `items` — to remove, or to move before `before`.
pub fn name_items(items: &[Item], before: Option<&str>) -> Value {
    let data: Vec<Value> = items
        .iter()
        .map(|i| json!({"id": i.track, "type": "tracks", "meta": {"itemId": i.item}}))
        .collect();
    match before {
        Some(b) => json!({"data": data, "meta": {"positionBefore": b}}),
        None => json!({"data": data}),
    }
}

/// John's TIDAL.
pub struct Tidal {
    pub account: Account,
    /// The catalogue's country, `AU`.
    pub country: String,
    http: reqwest::Client,
}

impl Tidal {
    pub fn new(account: Account, country: &str) -> Tidal {
        Tidal {
            account,
            country: country.to_string(),
            http: reqwest::Client::new(),
        }
    }

    fn url(&self, path: &str) -> String {
        let sep = if path.contains('?') { '&' } else { '?' };
        if path.starts_with("http") {
            path.to_string()
        } else {
            format!("{API}{path}{sep}countryCode={}", self.country)
        }
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<String, String> {
        let url = self.url(path);
        let token = self.account.access().await?;
        let mut req = self
            .http
            .request(method.clone(), &url)
            .bearer_auth(token)
            .header("accept", JSON_API);
        if let Some(b) = body {
            req = req.header("content-type", JSON_API).body(b.to_string());
        }
        let res = req
            .send()
            .await
            .map_err(|e| format!("{method} {url}: {e}"))?;
        let status = res.status();
        let text = res.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("{method} {url}: {status}: {text}"));
        }
        Ok(text)
    }

    /// The tracks with `isrc`.
    pub async fn by_isrc(&self, isrc: &str) -> Result<Vec<Track>, String> {
        let text = self
            .send(
                Method::GET,
                &format!("/tracks?filter%5Bisrc%5D={isrc}&include=artists"),
                None,
            )
            .await?;
        Ok(read_tracks(&text)?.0)
    }

    /// The tracks a search for `query` finds.
    pub async fn search(&self, query: &str) -> Result<Vec<Track>, String> {
        let q: String = byte_serialize(query.as_bytes()).collect();
        let text = self
            .send(
                Method::GET,
                &format!("/searchResults/{q}/relationships/tracks?include=tracks"),
                None,
            )
            .await?;
        let (mut found, _) = read_tracks(&text)?;
        // A search names tracks without their artists: fill them by ISRC.
        for t in found.iter_mut().filter(|t| t.artists.is_empty()) {
            if let Some(isrc) = t.isrc.clone() {
                if let Some(full) = self
                    .by_isrc(&isrc)
                    .await?
                    .into_iter()
                    .find(|f| f.id == t.id)
                {
                    *t = full;
                }
            }
        }
        Ok(found)
    }

    pub async fn create_playlist(&self, name: &str, description: &str) -> Result<String, String> {
        let text = self
            .send(
                Method::POST,
                "/playlists",
                Some(new_playlist(name, description)),
            )
            .await?;
        read_id(&text)
    }

    /// Every item of `playlist`, in order.
    pub async fn items(&self, playlist: &str) -> Result<Vec<Item>, String> {
        let mut out = Vec::new();
        let mut next = Some(format!("/playlists/{playlist}/relationships/items"));
        while let Some(path) = next {
            let text = self.send(Method::GET, &path, None).await?;
            let (items, n) = read_items(&text)?;
            out.extend(items);
            next = n.map(|n| {
                if n.starts_with('/') {
                    n.trim_start_matches("/v2").to_string()
                } else {
                    n
                }
            });
        }
        Ok(out)
    }

    pub async fn add(
        &self,
        playlist: &str,
        tracks: &[String],
        before: Option<&str>,
    ) -> Result<(), String> {
        for chunk in tracks.chunks(20) {
            self.send(
                Method::POST,
                &format!("/playlists/{playlist}/relationships/items"),
                Some(add_items(chunk, before)),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn remove(&self, playlist: &str, items: &[Item]) -> Result<(), String> {
        for chunk in items.chunks(20) {
            self.send(
                Method::DELETE,
                &format!("/playlists/{playlist}/relationships/items"),
                Some(name_items(chunk, None)),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn move_before(
        &self,
        playlist: &str,
        items: &[Item],
        before: &str,
    ) -> Result<(), String> {
        self.send(
            Method::PATCH,
            &format!("/playlists/{playlist}/relationships/items"),
            Some(name_items(items, Some(before))),
        )
        .await
        .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_iso_duration_is_read_to_milliseconds() {
        assert_eq!(duration_ms("PT3M20S"), Some(200_000));
        assert_eq!(duration_ms("PT1H2M3.5S"), Some(3_723_500));
        assert_eq!(duration_ms("PT45S"), Some(45_000));
        assert_eq!(duration_ms("3:20"), None);
        assert_eq!(duration_ms("PT3X"), None);
        assert_eq!(duration_ms("PT3"), None);
    }

    /// **A track reads with its artists from what is included**, its tempo,
    /// and its key on the wheel.
    #[test]
    fn a_track_document_reads_with_its_artists_and_key() {
        let (t, next) = read_tracks(
            r#"{"data": [{"id": "77646168", "type": "tracks",
                 "attributes": {"title": "One More Time", "isrc": "GBDUW0000059",
                   "duration": "PT5M20S", "bpm": 122.0, "key": "FSharp", "keyScale": "MINOR"},
                 "relationships": {"artists": {"data": [{"id": "8847", "type": "artists"}]}}}],
               "included": [{"id": "8847", "type": "artists", "attributes": {"name": "Daft Punk"}}],
               "links": {"self": "/tracks?filter[isrc]=GBDUW0000059", "next": "/tracks?page[cursor]=x"}}"#,
        )
        .unwrap();
        assert_eq!(
            t,
            vec![Track {
                id: "77646168".into(),
                title: "One More Time".into(),
                artists: vec!["Daft Punk".into()],
                isrc: Some("GBDUW0000059".into()),
                duration_ms: 320_000,
                bpm: Some(122.0),
                key: Camelot::parse("11A"),
            }]
        );
        assert_eq!(next.as_deref(), Some("/tracks?page[cursor]=x"));
    }

    #[test]
    fn a_relationship_document_reads_its_included_tracks() {
        let (t, _) = read_tracks(
            r#"{"data": [{"id": "5", "type": "tracks"}],
               "included": [{"id": "5", "type": "tracks", "attributes": {"title": "S", "duration": "PT1M"}}]}"#,
        )
        .unwrap();
        assert_eq!(
            (t[0].id.as_str(), t[0].title.as_str(), t[0].duration_ms),
            ("5", "S", 60_000)
        );
        assert_eq!(t[0].bpm, None);
    }

    #[test]
    fn items_read_with_their_item_ids() {
        let (items, next) = read_items(
            r#"{"data": [{"id": "11", "type": "tracks", "meta": {"itemId": "a-1", "addedAt": "x"}},
                         {"id": "12", "type": "tracks", "meta": {"itemId": "a-2"}}],
               "links": {}}"#,
        )
        .unwrap();
        assert_eq!(
            items,
            vec![
                Item {
                    track: "11".into(),
                    item: "a-1".into()
                },
                Item {
                    track: "12".into(),
                    item: "a-2".into()
                }
            ]
        );
        assert_eq!(next, None);
    }

    /// **Every body is exact.**
    #[test]
    fn the_bodies_are_exact() {
        assert_eq!(
            new_playlist("NYE Next", "ZenJ's next tracks").to_string(),
            r#"{"data":{"type":"playlists","attributes":{"name":"NYE Next","description":"ZenJ's next tracks","accessType":"UNLISTED"}}}"#
        );
        assert_eq!(
            add_items(&["11".into(), "12".into()], Some("a-1")).to_string(),
            r#"{"data":[{"id":"11","type":"tracks"},{"id":"12","type":"tracks"}],"meta":{"positionBefore":"a-1"}}"#
        );
        assert_eq!(
            name_items(
                &[Item {
                    track: "11".into(),
                    item: "a-1".into()
                }],
                None
            )
            .to_string(),
            r#"{"data":[{"id":"11","type":"tracks","meta":{"itemId":"a-1"}}]}"#
        );
        assert_eq!(
            read_id(r#"{"data": {"id": "pl-9", "type": "playlists"}}"#).unwrap(),
            "pl-9"
        );
    }
}
