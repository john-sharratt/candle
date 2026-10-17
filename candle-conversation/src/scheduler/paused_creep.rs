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

use candle::Result;
use candle_transformers::models::residual_rows::CreepResidual;

use super::WaveMember;
use crate::projection::DecodePriority;

/// A creep group mid-sweep, set aside.
pub(super) struct PausedCreep {
    /// Its arena, holding the rows it reached — kept as it stands.
    pub residual: Option<CreepResidual>,
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
/// The one copy here, the survivors' rows moved into the arena's other half
/// ([`CreepResidual::retain`]), runs only when a member errors or its slot is
/// freed mid-sweep, and the alternative is replaying every survivor's finished
/// layers. With no survivor rows the arena goes.
pub(super) fn drop_rows(
    residual: &mut Option<CreepResidual>,
    members: &mut Vec<WaveMember>,
    rows: &mut Vec<usize>,
    keep: &[bool],
) -> Result<()> {
    if keep.iter().all(|&k| k) {
        return Ok(());
    }
    let ranges = kept_row_ranges(rows, keep)?;
    match residual.as_mut() {
        Some(r) if !ranges.is_empty() => r.retain(&ranges)?,
        Some(_) => *residual = None,
        None => {}
    }
    let mut k = keep.iter();
    members.retain(|_| *k.next().unwrap_or(&false));
    let mut k = keep.iter();
    rows.retain(|_| *k.next().unwrap_or(&false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle::{DType, Device, Tensor};
    use candle_transformers::models::residual_rows::{copy_rows, CreepResidual, RowCopy, SpanRows};

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

    /// A creep arena of six rows of width two, holding `0..12` in order.
    fn held_creep() -> CreepResidual {
        let mut c = CreepResidual::claim(&Device::Cpu, 6, 2, DType::F32).unwrap();
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let src = Tensor::from_vec(values, (6, 2), &Device::Cpu).unwrap();
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: 6,
        }];
        copy_rows(&SpanRows::new(&src, 2).unwrap(), c.pending(), &plan).unwrap();
        c.commit();
        c
    }

    /// Dropping a member takes exactly its rows, and the others keep theirs in
    /// order.
    #[test]
    fn dropping_a_member_takes_exactly_its_rows() {
        let mut residual = Some(held_creep());
        let section = |seq_id| WaveMember::Section { seq_id, advance: 1 };
        let mut members = vec![section(1), section(2), section(3)];
        let mut rows = vec![2, 3, 1];
        drop_rows(&mut residual, &mut members, &mut rows, &[true, false, true]).unwrap();
        let held = residual.unwrap();
        assert_eq!(held.rows(), 3);
        assert_eq!(
            held.committed()
                .to_tensor()
                .unwrap()
                .narrow(0, 0, 3)
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

    /// Dropping every member's rows gives the arena back.
    #[test]
    fn dropping_every_row_gives_the_arena_back() {
        let mut residual = Some(held_creep());
        let section = |seq_id| WaveMember::Section { seq_id, advance: 1 };
        let mut members = vec![section(1), section(2)];
        let mut rows = vec![2, 4];
        drop_rows(&mut residual, &mut members, &mut rows, &[false, false]).unwrap();
        assert!(residual.is_none());
        assert!(members.is_empty());
    }
}
