//! rekordbox's library (`master.db`), read-only: each track's title,
//! artists, tempo, key, length, ISRC, where its audio is and where its
//! analysis is. The file is decrypted (`sqlcipher`) to a copy in ZenJ's data
//! folder and the copy opened, so rekordbox's own file is never touched.

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags};

use super::sqlcipher::{decrypt, REKORDBOX_KEY, ROUNDS};

/// One track as rekordbox lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct Content {
    pub id: String,
    pub title: String,
    pub artist: String,
    /// Tempo as analysed, from rekordbox's hundredths.
    pub bpm: f32,
    pub key: Option<String>,
    pub length_s: u32,
    pub isrc: Option<String>,
    pub folder_path: String,
    /// The analysis file, as `/PIONEER/USBANLZ/…/ANLZ0000.DAT`.
    pub anlz: Option<String>,
    pub year: Option<u16>,
}

impl Content {
    /// The TIDAL track id, when the audio is a TIDAL stream: the last run of
    /// digits in a path that names TIDAL.
    pub fn tidal_id(&self) -> Option<String> {
        let p = self.folder_path.to_ascii_lowercase();
        if !p.contains("tidal") {
            return None;
        }
        let digits: String = p
            .chars()
            .rev()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        (!digits.is_empty()).then(|| digits.chars().rev().collect())
    }
}

/// Where rekordbox keeps its library on Windows: `%APPDATA%/Pioneer/rekordbox`.
pub fn library_dir(appdata: &Path) -> PathBuf {
    appdata.join("Pioneer").join("rekordbox")
}

/// `master.db` decrypted to `copy`, which is replaced.
pub fn decrypt_to(master: &Path, copy: &Path) -> Result<(), String> {
    let enc = fs::read(master).map_err(|e| format!("{}: {e}", master.display()))?;
    let plain = decrypt(&enc, REKORDBOX_KEY.as_bytes(), ROUNDS)?;
    let tmp = copy.with_extension("tmp");
    fs::write(&tmp, plain).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, copy).map_err(|e| format!("{}: {e}", copy.display()))
}

const QUERY: &str =
    "SELECT c.ID, COALESCE(c.Title, ''), COALESCE(a.Name, ''), COALESCE(c.BPM, 0), \
    k.ScaleName, COALESCE(c.Length, 0), c.ISRC, COALESCE(c.FolderPath, ''), c.AnalysisDataPath, \
    c.ReleaseYear \
    FROM djmdContent c \
    LEFT JOIN djmdArtist a ON a.ID = c.ArtistID \
    LEFT JOIN djmdKey k ON k.ID = c.KeyID \
    WHERE COALESCE(c.rb_local_deleted, 0) = 0 \
    ORDER BY c.ID";

fn blank(s: Option<String>) -> Option<String> {
    s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Every track in the plain library at `db`.
pub fn contents(db: &Path) -> Result<Vec<Content>, String> {
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("{}: {e}", db.display()))?;
    let mut stmt = conn.prepare(QUERY).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let year: Option<i64> = r.get(9)?;
            Ok(Content {
                id: r.get::<_, String>(0)?,
                title: r.get(1)?,
                artist: r.get(2)?,
                bpm: r.get::<_, i64>(3)? as f32 / 100.0,
                key: blank(r.get(4)?),
                length_s: r.get::<_, i64>(5)?.max(0) as u32,
                isrc: blank(r.get(6)?),
                folder_path: r.get(7)?,
                anlz: blank(r.get(8)?),
                year: year.and_then(|y| u16::try_from(y).ok()).filter(|y| *y > 0),
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

#[cfg(test)]
pub mod fixture {
    use std::path::Path;

    use rusqlite::Connection;

    /// A plain library with rekordbox's tables, and the rows given as
    /// `(id, title, artist, bpm×100, key, length, isrc, path, anlz)`.
    #[allow(clippy::type_complexity)]
    pub fn library(at: &Path, rows: &[(&str, &str, &str, i64, &str, i64, &str, &str, &str)]) {
        let c = Connection::open(at).unwrap();
        c.execute_batch(
            "CREATE TABLE djmdArtist (ID VARCHAR(255) PRIMARY KEY, Name VARCHAR(255));
             CREATE TABLE djmdKey (ID VARCHAR(255) PRIMARY KEY, ScaleName VARCHAR(255));
             CREATE TABLE djmdContent (ID VARCHAR(255) PRIMARY KEY, Title VARCHAR(255),
               ArtistID VARCHAR(255), BPM INTEGER, KeyID VARCHAR(255), Length INTEGER,
               ISRC VARCHAR(255), FolderPath VARCHAR(255), AnalysisDataPath VARCHAR(255),
               ReleaseYear INTEGER, rb_local_deleted INTEGER DEFAULT 0);",
        )
        .unwrap();
        for (i, r) in rows.iter().enumerate() {
            c.execute(
                "INSERT INTO djmdArtist VALUES (?1, ?2)",
                (format!("a{i}"), r.2),
            )
            .unwrap();
            c.execute("INSERT OR IGNORE INTO djmdKey VALUES (?1, ?1)", (r.4,))
                .unwrap();
            c.execute(
                "INSERT INTO djmdContent (ID, Title, ArtistID, BPM, KeyID, Length, ISRC, \
                 FolderPath, AnalysisDataPath, ReleaseYear) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 2001)",
                (r.0, r.1, format!("a{i}"), r.3, r.4, r.5, r.6, r.7, r.8),
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The library reads each live track**, its tempo from hundredths,
    /// its key by name, a blank ISRC as none — and a deleted one not at all.
    #[test]
    fn the_library_reads_its_tracks() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("master.plain.db");
        fixture::library(
            &db,
            &[
                (
                    "101",
                    "One More Time",
                    "Daft Punk",
                    12250,
                    "F#m",
                    320,
                    "GBDUW0000059",
                    "tidal:tracks:1234567",
                    "/PIONEER/USBANLZ/a1b/2c3d/ANLZ0000.DAT",
                ),
                (
                    "102",
                    "Local",
                    "Someone",
                    12800,
                    "8A",
                    200,
                    " ",
                    "C:/Music/x.mp3",
                    "",
                ),
                ("103", "Gone", "Nobody", 12000, "Am", 100, "", "", ""),
            ],
        );
        Connection::open(&db)
            .unwrap()
            .execute(
                "UPDATE djmdContent SET rb_local_deleted = 1 WHERE ID = '103'",
                [],
            )
            .unwrap();
        let c = contents(&db).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(
            c[0],
            Content {
                id: "101".into(),
                title: "One More Time".into(),
                artist: "Daft Punk".into(),
                bpm: 122.5,
                key: Some("F#m".into()),
                length_s: 320,
                isrc: Some("GBDUW0000059".into()),
                folder_path: "tidal:tracks:1234567".into(),
                anlz: Some("/PIONEER/USBANLZ/a1b/2c3d/ANLZ0000.DAT".into()),
                year: Some(2001),
            }
        );
        assert_eq!(c[0].tidal_id().as_deref(), Some("1234567"));
        assert_eq!((c[1].isrc.clone(), c[1].anlz.clone()), (None, None));
        assert_eq!(c[1].tidal_id(), None, "a local file");
    }
}
