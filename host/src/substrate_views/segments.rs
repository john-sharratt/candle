//! The segmented redo log's files on disk.

use std::path::Path;

use serde::Serialize;

/// One `seg-*` file.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SegmentView {
    pub id: u64,
    /// The highest-numbered segment: the one being appended to.
    pub active: bool,
    pub bytes: u64,
}

/// The substrate's segment files in `dir`, ascending by id (the last is the
/// active one), and their total size. Pure `fs` stat — no lock, so it answers
/// whether or not anything holds the substrate open. A directory that does not
/// exist holds no segments.
pub fn segment_files(dir: &Path) -> (Vec<SegmentView>, u64) {
    let mut segs: Vec<(u64, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(rest) = name.strip_prefix("seg-") else {
                continue;
            };
            let Some((num, ext)) = rest.rsplit_once('.') else {
                continue;
            };
            if ext != "log" && ext != "active" {
                continue;
            }
            let Ok(id) = num.parse::<u64>() else { continue };
            let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
            segs.push((id, bytes));
        }
    }
    segs.sort_by_key(|(id, _)| *id);
    let total: u64 = segs.iter().map(|(_, b)| *b).sum();
    let last = segs.len().saturating_sub(1);
    let views = segs
        .iter()
        .enumerate()
        .map(|(i, (id, bytes))| SegmentView {
            id: *id,
            active: i == last,
            bytes: *bytes,
        })
        .collect();
    (views, total)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    /// Segments sort by number, the last is active, and only `seg-<n>.log` and
    /// `seg-<n>.active` files count.
    #[test]
    fn the_segments_are_listed_in_order_with_the_newest_active() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("seg-000010.log"), [0u8; 5]).unwrap();
        fs::write(dir.path().join("seg-000002.log"), [0u8; 3]).unwrap();
        fs::write(dir.path().join("seg-000011.active"), [0u8; 7]).unwrap();
        fs::write(dir.path().join("seg-abc.log"), [0u8; 99]).unwrap();
        fs::write(dir.path().join("seg-000012.txt"), [0u8; 99]).unwrap();
        fs::write(dir.path().join("substrate.log"), [0u8; 99]).unwrap();

        let (segments, total) = segment_files(dir.path());
        assert_eq!(
            serde_json::to_value(&segments).unwrap(),
            json!([
                {"id": 2, "active": false, "bytes": 3},
                {"id": 10, "active": false, "bytes": 5},
                {"id": 11, "active": true, "bytes": 7},
            ])
        );
        assert_eq!(total, 15);
    }

    /// A substrate never written holds no segments, and that is not an error.
    #[test]
    fn a_missing_directory_holds_no_segments() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            segment_files(&dir.path().join("substrate")),
            (Vec::new(), 0)
        );
    }
}
