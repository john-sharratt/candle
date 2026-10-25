//! When this member was last joined to a cluster, and what an open drops when
//! it has been away too long (`docs/replicated_substrate.md` §7.8).
//!
//! `substrate/JOINED` holds one wall-clock reading in milliseconds, refreshed
//! by the replicator while the member is joined and caught up. An expired
//! tombstone is forgotten everywhere (§6.8), so a member away longer than the
//! retention cannot learn of a deletion it missed: its copy of the deleted
//! records would stand for ever. So a writable open of a member away too long
//! drops what it can no longer trust, compacts, and resyncs:
//!
//! - away more than [`FOREIGN_AFTER`]: every record another origin wrote. The
//!   applied set forgets those origins, so the catch-up refetches whatever the
//!   cluster still holds.
//! - away more than [`OWN_AFTER`]: also its own records written before it
//!   was last joined. They leave gaps in its own written set, which the catch-up
//!   fills from the members that hold them.
//!
//! What it wrote after it was last joined — work no other member has — is kept
//! either way. A store with no file has never been judged: the open writes one
//! and drops nothing.

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use super::identity::{wall_ms, RecordIdentity};

/// The file the reading is kept in, inside the substrate directory.
pub const JOINED_FILE: &str = "JOINED";

/// Away longer than this, a member drops what other origins wrote.
pub const FOREIGN_AFTER: Duration = Duration::from_secs(20 * 24 * 60 * 60);

/// Away longer than this — the tombstone retention — a member also drops its
/// own records written before it was last joined.
pub const OWN_AFTER: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// The reading in `dir`'s file, or `None` when there is none.
pub fn read(dir: &Path) -> io::Result<Option<u64>> {
    match fs::read_to_string(dir.join(JOINED_FILE)) {
        Ok(text) => text.trim().parse().map(Some).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{JOINED_FILE} holds {text:?}, not a millisecond reading: {e}"),
            )
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Record `at_ms` in `dir`'s file, through a temporary sibling and a rename,
/// so a crash leaves the old reading or the new one and never a torn one.
pub fn write(dir: &Path, at_ms: u64) -> io::Result<()> {
    let tmp = dir.join(format!("{JOINED_FILE}.tmp"));
    // Synced through the handle that wrote it: Windows refuses to flush a
    // handle opened for reading.
    let mut file = fs::File::create(&tmp)?;
    file.write_all(at_ms.to_string().as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, dir.join(JOINED_FILE))
}

/// What an open drops. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purge {
    /// Nothing: the member was joined recently, or has never been judged.
    Nothing,
    /// Every record another origin wrote.
    Foreign,
    /// Those, and its own records written before `joined_ms`.
    ForeignAndOwnBefore { joined_ms: u64 },
}

impl Purge {
    /// What an open at `now_ms` drops, for a member last joined at `joined`.
    pub fn after(joined: Option<u64>, now_ms: u64) -> Purge {
        let Some(joined_ms) = joined else {
            return Purge::Nothing;
        };
        let away = now_ms.saturating_sub(joined_ms);
        if away > OWN_AFTER.as_millis() as u64 {
            Purge::ForeignAndOwnBefore { joined_ms }
        } else if away > FOREIGN_AFTER.as_millis() as u64 {
            Purge::Foreign
        } else {
            Purge::Nothing
        }
    }

    /// Whether the open drops anything at all.
    pub fn drops_any(&self) -> bool {
        *self != Purge::Nothing
    }

    /// Whether the record carrying `id` is dropped by member `node`. A record
    /// with no identity is housekeeping no origin wrote, and is kept.
    pub fn drops(&self, node: u64, id: &RecordIdentity) -> bool {
        if id.is_none() {
            return false;
        }
        match *self {
            Purge::Nothing => false,
            Purge::Foreign => id.origin != node,
            Purge::ForeignAndOwnBefore { joined_ms } => {
                id.origin != node || wall_ms(id.hlc) < joined_ms
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::identity::COUNTER_BITS;

    const DAY: u64 = 24 * 60 * 60 * 1000;
    const NODE: u64 = 0xA;

    fn id(origin: u64, at_ms: u64) -> RecordIdentity {
        RecordIdentity {
            origin,
            oseq: 1,
            gen: 0,
            epoch: 0,
            hlc: at_ms << COUNTER_BITS,
        }
    }

    #[test]
    fn the_reading_is_written_as_decimal_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read(dir.path()).unwrap(), None);
        write(dir.path(), 1_792_884_808_478).unwrap();
        assert_eq!(
            fs::read(dir.path().join(JOINED_FILE)).unwrap(),
            b"1792884808478"
        );
        assert_eq!(read(dir.path()).unwrap(), Some(1_792_884_808_478));
        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![JOINED_FILE.to_string()], "no temporary left");
    }

    #[test]
    fn a_reading_that_is_not_a_number_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(JOINED_FILE), b"yesterday").unwrap();
        assert_eq!(
            read(dir.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    /// Twenty days away keeps everything; past it, other origins go; past
    /// thirty, own records from before the last join go too. Never judged,
    /// nothing goes.
    #[test]
    fn the_purge_grows_with_the_time_away() {
        let joined = 100 * DAY;
        assert_eq!(Purge::after(None, joined + 400 * DAY), Purge::Nothing);
        assert_eq!(
            Purge::after(Some(joined), joined + 20 * DAY),
            Purge::Nothing
        );
        assert_eq!(
            Purge::after(Some(joined), joined + 20 * DAY + 1),
            Purge::Foreign
        );
        assert_eq!(
            Purge::after(Some(joined), joined + 30 * DAY),
            Purge::Foreign
        );
        assert_eq!(
            Purge::after(Some(joined), joined + 30 * DAY + 1),
            Purge::ForeignAndOwnBefore { joined_ms: joined }
        );
        assert_eq!(Purge::after(Some(joined), joined - DAY), Purge::Nothing);
    }

    #[test]
    fn each_purge_drops_what_it_names_and_never_housekeeping() {
        let joined = 100 * DAY;
        let foreign = id(0xB, joined + DAY);
        let own_before = id(NODE, joined - 1);
        let own_after = id(NODE, joined);
        for purge in [
            Purge::Nothing,
            Purge::Foreign,
            Purge::ForeignAndOwnBefore { joined_ms: joined },
        ] {
            assert!(!purge.drops(NODE, &RecordIdentity::NONE), "{purge:?}");
            assert!(!purge.drops(NODE, &own_after), "{purge:?}");
        }
        assert!(!Purge::Nothing.drops(NODE, &foreign));
        assert!(Purge::Foreign.drops(NODE, &foreign));
        assert!(!Purge::Foreign.drops(NODE, &own_before));
        let long = Purge::ForeignAndOwnBefore { joined_ms: joined };
        assert!(long.drops(NODE, &foreign));
        assert!(long.drops(NODE, &own_before));
        assert!(!Purge::Nothing.drops_any());
        assert!(Purge::Foreign.drops_any() && long.drops_any());
    }
}
