//! A creep set aside mid-sweep so a higher band's prompt can run first.
//!
//! A sweep cannot be restarted — its finished layers already hold its rows in
//! their carried state (`Scheduler::reset_wave_prefill`) — and finishing it at
//! once costs the waiting prompt the whole remainder of a group that may be
//! thousands of rows deep. Set aside instead, it keeps everything the sweep
//! needs to go on: the residual it holds, the layer it reached, its members
//! and each member's rows. Nothing else touches a paused member's state — it
//! is mid-prefill, so only its own creep writes it — and the group resumes at
//! the same layer once nothing above its band is waiting.

use std::collections::HashSet;

use candle::{Result, Tensor};

use super::WaveMember;
use crate::projection::DecodePriority;

/// A creep group mid-sweep, set aside.
pub(super) struct PausedCreep {
    pub residual: Option<Tensor>,
    pub cursor: usize,
    pub members: Vec<WaveMember>,
    pub rows: Vec<usize>,
    /// The band it ran at; it resumes once nothing above it waits.
    pub band: DecodePriority,
}

impl PausedCreep {
    /// Every member's sequence id.
    pub fn seq_ids(&self) -> HashSet<usize> {
        self.members.iter().map(|m| m.seq_id()).collect()
    }

    /// Whether `seq` is a member.
    pub fn holds(&self, seq: usize) -> bool {
        self.members.iter().any(|m| m.seq_id() == seq)
    }
}

/// The `(first row, rows)` runs of a held creep residual that survive dropping
/// the members `keep` marks `false`, given each member's rows in member order.
pub(super) fn kept_row_ranges(rows: &[usize], keep: &[bool]) -> Result<Vec<(usize, usize)>> {
    if rows.len() != keep.len() {
        candle::bail!(
            "creep residual: {} members' rows recorded against {} members",
            rows.len(),
            keep.len()
        );
    }
    let mut start = 0;
    let mut out = Vec::new();
    for (&r, &k) in rows.iter().zip(keep) {
        if k && r > 0 {
            match out.last_mut() {
                Some((s, len)) if *s + *len == start => *len += r,
                _ => out.push((start, r)),
            }
        }
        start += r;
    }
    Ok(out)
}

/// Take the members `keep` marks `false` out of a held creep — its residual,
/// its members and their rows — leaving the survivors' rows in order.
///
/// The one copy here, the survivors' rows gathered into a residual of their
/// own, is an exception to hot-path invariant 2 that pays for itself: it runs
/// only when a member errors or its slot is freed mid-sweep, and the
/// alternative is replaying every survivor's finished layers.
pub(super) fn drop_rows(
    residual: &mut Option<Tensor>,
    members: &mut Vec<WaveMember>,
    rows: &mut Vec<usize>,
    keep: &[bool],
) -> Result<()> {
    if keep.iter().all(|&k| k) {
        return Ok(());
    }
    if let Some(res) = residual.take() {
        let ranges = kept_row_ranges(rows, keep)?;
        if !ranges.is_empty() {
            let pieces = ranges
                .iter()
                .map(|&(start, len)| res.narrow(1, start, len))
                .collect::<Result<Vec<_>>>()?;
            *residual = Some(Tensor::cat(&pieces, 1)?);
        }
    }
    let mut k = keep.iter();
    members.retain(|_| *k.next().unwrap_or(&false));
    let mut k = keep.iter();
    rows.retain(|_| *k.next().unwrap_or(&false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle::{Device, Tensor};

    use super::{drop_rows, kept_row_ranges};
    use crate::scheduler::WaveMember;

    /// The surviving members' rows, as runs of the held residual — adjacent
    /// survivors merged into one run, a member with no rows contributing none.
    #[test]
    fn the_kept_rows_are_the_survivors_runs() {
        assert_eq!(
            kept_row_ranges(&[3, 2, 4], &[true, false, true]).unwrap(),
            vec![(0, 3), (5, 4)]
        );
        assert_eq!(
            kept_row_ranges(&[3, 2, 4], &[false, true, true]).unwrap(),
            vec![(3, 6)]
        );
        assert_eq!(
            kept_row_ranges(&[3, 0, 4], &[true, true, true]).unwrap(),
            vec![(0, 7)]
        );
        assert!(kept_row_ranges(&[3, 2], &[true]).is_err());
    }

    /// Dropping a member takes exactly its rows, and the others keep theirs in
    /// order.
    #[test]
    fn dropping_a_member_takes_exactly_its_rows() {
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let mut residual = Some(Tensor::from_vec(values, (1, 6, 2), &Device::Cpu).unwrap());
        let section = |seq_id| WaveMember::Section { seq_id, advance: 1 };
        let mut members = vec![section(1), section(2), section(3)];
        let mut rows = vec![2, 3, 1];
        drop_rows(&mut residual, &mut members, &mut rows, &[true, false, true]).unwrap();
        assert_eq!(
            residual
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0., 1., 2., 3., 10., 11.]
        );
        let ids: Vec<usize> = members.iter().map(|m| m.seq_id()).collect();
        assert_eq!(ids, vec![1, 3]);
        assert_eq!(rows, vec![2, 1]);
    }
}
