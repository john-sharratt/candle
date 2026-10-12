//! The sector windows a batch of fixed-size records is read through.
//!
//! Direct I/O reads whole, aligned sectors, and a PLE record is 170 bytes, so
//! a 4 KiB sector holds parts of about 24 of them. A batch's records are
//! sorted by offset, and each one whose sectors touch or overlap the window
//! being built joins it, so a sector two records share is read once. A window
//! stops growing at `cap` bytes, so a batch of clustered rows still spreads
//! across the reader's handles rather than becoming one long read.

/// Where a batch's records are read from, and where each lands.
#[derive(Debug, PartialEq, Eq)]
pub struct Windows {
    /// Each window's sector-aligned file offset and length, in file order.
    pub spans: Vec<(u64, usize)>,
    /// For each record, in the caller's order: its window, and its byte offset
    /// within that window.
    pub at: Vec<(usize, usize)>,
}

/// Plan the windows for records of `record` bytes at `offsets`, reading whole
/// `sector`-byte sectors, no window longer than `cap` bytes unless one record's
/// own sectors are.
pub fn plan(offsets: &[u64], record: usize, sector: u64, cap: usize) -> Windows {
    let mut order: Vec<usize> = (0..offsets.len()).collect();
    order.sort_by_key(|&i| offsets[i]);
    let mut spans: Vec<(u64, usize)> = Vec::new();
    let mut at = vec![(0usize, 0usize); offsets.len()];
    let mut open: Option<(u64, u64)> = None;
    for i in order {
        let off = offsets[i];
        let start = off / sector * sector;
        let end = (off + record as u64).div_ceil(sector) * sector;
        open = Some(match open {
            Some((s, e)) if start <= e && (end.max(e) - s) as usize <= cap => (s, end.max(e)),
            Some((s, e)) => {
                spans.push((s, (e - s) as usize));
                (start, end)
            }
            None => (start, end),
        });
        let (s, _) = open.expect("set just above");
        at[i] = (spans.len(), (off - s) as usize);
    }
    if let Some((s, e)) = open {
        spans.push((s, (e - s) as usize));
    }
    Windows { spans, at }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECTOR: u64 = 4096;

    /// Two records in one sector are read through one window.
    #[test]
    fn records_sharing_a_sector_share_a_window() {
        assert_eq!(
            plan(&[0, 170], 170, SECTOR, 65_536),
            Windows {
                spans: vec![(0, 4096)],
                at: vec![(0, 0), (0, 170)],
            }
        );
    }

    /// A record across a sector boundary reads both sectors.
    #[test]
    fn a_record_across_a_boundary_reads_both_sectors() {
        assert_eq!(
            plan(&[4000], 170, SECTOR, 65_536),
            Windows {
                spans: vec![(0, 8192)],
                at: vec![(0, 4000)],
            }
        );
    }

    /// Far-apart records get windows of their own, and each record is placed
    /// in the caller's order whatever order the windows are in.
    #[test]
    fn far_records_get_their_own_windows_in_file_order() {
        assert_eq!(
            plan(&[1_000_000, 0], 170, SECTOR, 65_536),
            Windows {
                spans: vec![(0, 4096), (999_424, 4096)],
                at: vec![(1, 576), (0, 0)],
            }
        );
    }

    /// Neighbouring sectors join a window until it would pass the cap.
    #[test]
    fn neighbouring_sectors_merge_up_to_the_cap() {
        assert_eq!(
            plan(&[0, 4096, 8192], 170, SECTOR, 8192),
            Windows {
                spans: vec![(0, 8192), (8192, 4096)],
                at: vec![(0, 0), (0, 4096), (1, 0)],
            }
        );
    }
}
