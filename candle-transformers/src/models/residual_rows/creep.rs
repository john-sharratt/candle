//! A creep's residual between steps: two halves, flipped on commit.
//!
//! A window reads the committed half and writes the other. The halves flip only
//! once the window has succeeded, so a window that fails mid-sweep — a refused
//! tier, a device OOM — leaves the committed half exactly as the failed attempt
//! found it, and the retry resumes from the same residual (`docs/wave_step.md`
//! §3.1).

use candle::{DType, Device, Result};

use super::arena::ResidualArena;
use super::row_map::{RowCopy, RowMap};
use super::store::{copy_rows, RowStore};

#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{arena_regions, plan_slot_moves, ArenaSlot, SlotTenant};

/// A creep's rows, held twice over.
pub struct CreepResidual {
    halves: [ResidualArena; 2],
    committed: usize,
    rows: usize,
}

impl CreepResidual {
    /// Claim both halves for `rows` rows of `width` elements of `dtype`.
    ///
    /// Between forwards, as every span claim is ([`ResidualArena::claim`]).
    pub fn claim(device: &Device, rows: usize, width: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            halves: [
                ResidualArena::claim(device, rows, width, dtype)?,
                ResidualArena::claim(device, rows, width, dtype)?,
            ],
            committed: 0,
            rows,
        })
    }

    /// Rows the creep holds, in caller order.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The layout the rows are held in — caller order, one row per index.
    pub fn layout(&self) -> RowMap {
        RowMap::identity(self.rows)
    }

    /// The half a window resumes from.
    pub fn committed(&self) -> &ResidualArena {
        &self.halves[self.committed]
    }

    /// The half a window writes.
    pub fn pending(&self) -> &ResidualArena {
        &self.halves[1 - self.committed]
    }

    /// The window that wrote [`Self::pending`] succeeded: it is now the residual
    /// the next window resumes from.
    pub fn commit(&mut self) {
        self.committed = 1 - self.committed;
    }

    /// Keep only the caller rows of `keep` — `(first row, rows)` runs, ascending
    /// — in order, and drop the rest: the survivors are copied into the pending
    /// half and committed.
    ///
    /// For a member that leaves mid-sweep. Its rows go; the survivors' finished
    /// layers stand, which is why this copies rather than restarting the sweep.
    /// The halves keep the room they were claimed with until the creep ends.
    pub fn retain(&mut self, keep: &[(usize, usize)]) -> Result<()> {
        let mut plan = Vec::with_capacity(keep.len());
        let mut at = 0;
        for &(start, len) in keep {
            if start + len > self.rows {
                candle::bail!(
                    "creep residual: keeping rows {start}..{} of {}",
                    start + len,
                    self.rows
                );
            }
            plan.push(RowCopy {
                src: start,
                dst: at,
                len,
            });
            at += len;
        }
        let (src, dst) = (self.committed(), self.pending());
        if dst.capacity() < at {
            candle::bail!(
                "creep residual: {at} rows kept in a {}-row half",
                dst.capacity()
            );
        }
        copy_rows(src, dst, &plan)?;
        self.rows = at;
        self.commit();
        Ok(())
    }

    /// Bytes a block's slot is claimed for, in both halves.
    pub fn block_bytes(&self) -> usize {
        self.halves[0].block_bytes()
    }

    pub fn device(&self) -> &Device {
        self.halves[0].device()
    }

    /// Move both halves' blocks that `moves` names — see
    /// [`ResidualArena::relocate`]. Between forwards, by the creep's holder.
    #[cfg(feature = "cuda")]
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        Ok(self.halves[0].relocate(moves)? + self.halves[1].relocate(moves)?)
    }
}

/// What one compaction of the creeps' arenas did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CreepCompaction {
    /// Moves the pass planned, each with its destination claimed.
    pub planned: usize,
    /// Blocks that moved onto their destination. Lower than `planned` only when a
    /// source was held by nothing passed in, whose claimed destination goes back.
    pub moved: usize,
    /// Regions every creep arena on the device held before the pass and after it.
    pub regions_before: usize,
    pub regions_after: usize,
}

impl CreepCompaction {
    /// Regions the pass handed back to the span.
    pub fn regions_released(&self) -> usize {
        self.regions_before.saturating_sub(self.regions_after)
    }
}

/// Compact the arenas `held` live in, the way every slot tenant is compacted:
/// plan the moves that empty the arenas holding the frontier
/// (`plan_slot_moves`), and move every block whose slot is a source onto its
/// destination.
///
/// **Between forwards, by the creeps' holder.** A move copies a block on the
/// primary stream, behind every launch that read or wrote it, and repoints the
/// block; a step resolves its blocks' addresses only while it runs, so between
/// steps nothing holds the old one. Planning claims the arena window, which a
/// forward refuses, so a pass asked for mid-forward fails rather than racing it.
///
/// Every creep that could hold a source must be in `held`: a source left behind
/// keeps its slot and its claimed destination goes back unused — safe, but
/// reclaim lost, which `moved` falling short of `planned` shows.
pub fn compact_creeps<'a>(
    held: impl IntoIterator<Item = &'a mut CreepResidual>,
    max_moves: usize,
) -> Result<CreepCompaction> {
    let mut held: Vec<&mut CreepResidual> = held.into_iter().collect();
    let Some(device) = held.first().map(|c| c.device().clone()) else {
        return Ok(CreepCompaction::default());
    };
    #[cfg(feature = "cuda")]
    {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(CreepCompaction::default());
        }
        let regions_before = arena_regions(&device, SlotTenant::CreepResidual);
        // Arenas are keyed by stride, so each block size is its own pass.
        let mut strides: Vec<usize> = held.iter().map(|c| c.block_bytes()).collect();
        strides.sort_unstable();
        strides.dedup();
        let (mut planned, mut moved) = (0, 0);
        for bytes in strides {
            let moves = plan_slot_moves(&device, SlotTenant::CreepResidual, bytes, max_moves)?;
            planned += moves.len();
            let mut by_source: HashMap<u64, ArenaSlot> =
                moves.into_iter().map(|m| (m.src, m.dst)).collect();
            for creep in held.iter_mut().filter(|c| c.block_bytes() == bytes) {
                moved += creep.relocate(&mut by_source)?;
            }
            // Destinations whose source nothing here held go back to their arenas.
            drop(by_source);
        }
        Ok(CreepCompaction {
            planned,
            moved,
            regions_before,
            regions_after: arena_regions(&device, SlotTenant::CreepResidual),
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (&mut held, device, max_moves);
        Ok(CreepCompaction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::residual_rows::store::SpanRows;
    use candle::Tensor;
    #[cfg(feature = "cuda")]
    use std::iter::repeat_n;

    /// `rows` rows of width 2, row `r` holding `r` in both elements.
    fn numbered(rows: usize) -> Tensor {
        let v: Vec<f32> = (0..rows).flat_map(|r| [r as f32, r as f32]).collect();
        Tensor::from_vec(v, (rows, 2), &Device::Cpu).unwrap()
    }

    fn write(half: &ResidualArena, t: &Tensor) {
        let rows = t.dims()[0];
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: rows,
        }];
        copy_rows(&SpanRows::new(t, 2).unwrap(), half, &plan).unwrap();
    }

    fn firsts(half: &ResidualArena, rows: usize) -> Vec<f32> {
        half.to_tensor()
            .unwrap()
            .narrow(0, 0, rows)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .map(|r| r[0])
            .collect()
    }

    /// A window that writes the pending half and is never committed leaves the
    /// committed half as it was — the retry reads what the failed attempt read.
    #[test]
    fn an_uncommitted_window_leaves_the_committed_half_alone() {
        let mut c = CreepResidual::claim(&Device::Cpu, 4, 2, DType::F32).unwrap();
        write(c.pending(), &numbered(4));
        c.commit();
        assert_eq!(firsts(c.committed(), 4), vec![0., 1., 2., 3.]);
        // The next window writes the pending half and fails: no commit.
        write(c.pending(), &(numbered(4) + 100.).unwrap());
        assert_eq!(firsts(c.committed(), 4), vec![0., 1., 2., 3.]);
        // It succeeds on retry: committed.
        c.commit();
        assert_eq!(firsts(c.committed(), 4), vec![100., 101., 102., 103.]);
    }

    /// Dropping a member keeps exactly the survivors' rows, in order.
    #[test]
    fn retaining_keeps_exactly_the_survivors_rows() {
        let mut c = CreepResidual::claim(&Device::Cpu, 6, 2, DType::F32).unwrap();
        write(c.pending(), &numbered(6));
        c.commit();
        c.retain(&[(0, 2), (5, 1)]).unwrap();
        assert_eq!(c.rows(), 3);
        assert_eq!(firsts(c.committed(), 3), vec![0., 1., 5.]);
    }

    #[test]
    fn retaining_past_the_rows_is_refused() {
        let mut c = CreepResidual::claim(&Device::Cpu, 2, 2, DType::F32).unwrap();
        assert!(c.retain(&[(1, 2)]).is_err());
    }

    /// `rows` rows of `width` on `device`, row `r` holding `r` throughout.
    #[cfg(feature = "cuda")]
    fn numbered_on(rows: usize, width: usize, device: &Device) -> Result<Tensor> {
        let v: Vec<f32> = (0..rows).flat_map(|r| repeat_n(r as f32, width)).collect();
        Tensor::from_vec(v, (rows, width), device)
    }

    /// The committed half's first element of each row.
    #[cfg(feature = "cuda")]
    fn committed_ids(c: &CreepResidual) -> Result<Vec<f32>> {
        c.committed()
            .to_tensor()?
            .narrow(0, 0, c.rows())?
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .to_vec1::<f32>()
    }

    /// **Compaction across two strides moves what it plans and keeps every
    /// row.** Two creeps whose blocks differ in size sit above the ground two
    /// dropped creeps gave back; one pass per stride moves their blocks down,
    /// and both still read their own rows from their committed halves.
    #[cfg(feature = "cuda")]
    #[test]
    fn compaction_across_strides_keeps_every_row() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        // 1 MiB rows fill a 16 MiB block; 3 MiB rows fit five to a 15 MiB one.
        // Two block sizes, so two arenas' passes.
        let (wa, wb) = ((1 << 20) / 4, (3 << 20) / 4);
        let (ra, rb) = (16, 20);
        let low_a = CreepResidual::claim(&device, ra, wa, DType::F32)?;
        let low_b = CreepResidual::claim(&device, rb, wb, DType::F32)?;
        let mut a = CreepResidual::claim(&device, ra, wa, DType::F32)?;
        let mut b = CreepResidual::claim(&device, rb, wb, DType::F32)?;
        assert_ne!(a.block_bytes(), b.block_bytes());
        for (c, rows, width) in [(&mut a, ra, wa), (&mut b, rb, wb)] {
            let plan = [RowCopy {
                src: 0,
                dst: 0,
                len: rows,
            }];
            copy_rows(
                &SpanRows::new(&numbered_on(rows, width, &device)?, width)?,
                c.pending(),
                &plan,
            )?;
            c.commit();
        }
        drop((low_a, low_b));
        let pass = compact_creeps([&mut a, &mut b], 64)?;
        assert!(pass.planned > 0, "the dropped creeps' ground draws moves");
        assert_eq!(
            pass.moved, pass.planned,
            "every planned source is held here"
        );
        assert!(pass.regions_after <= pass.regions_before);
        let ids = |rows: usize| (0..rows).map(|r| r as f32).collect::<Vec<_>>();
        assert_eq!(committed_ids(&a)?, ids(ra));
        assert_eq!(committed_ids(&b)?, ids(rb));
        Ok(())
    }
}
