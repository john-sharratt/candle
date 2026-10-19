//! A residual held between steps, in a span tenant's slot arena.
//!
//! A creep's residual crosses steps, so it cannot live on the transient tier,
//! which is re-placed every step and moves with the weight floor. It is held
//! the way every other per-sequence tenant of the span is — the recurrent state,
//! the QSA index pages, the provenance gallery — in fixed-stride slot arenas
//! (`candle_nn::kv_cache::tenant_arena`, [`SlotTenant::CreepResidual`]),
//! claimed between forwards, sized to the creep's rows, and given back slot by
//! slot as the blocks drop (`docs/wave_step.md` §3).
//!
//! **A block is whole rows in one slot**: as many rows as fit a region, so no
//! row ever crosses a slot boundary — between two slots of different arenas lies
//! another tenant's ground, and every address in the span is mapped, so a row
//! written across the gap would corrupt it without a fault. Each block's tensor
//! is anchored to its slot, so the slot cannot go back to its arena while any
//! view of the block is alive.
//!
//! **And it compacts like the other tenants.** `plan_slot_moves` plans moves only
//! for the arenas holding the frontier, with their destinations claimed;
//! [`ResidualArena::relocate`] copies each moved block on the primary stream and
//! repoints the block at its new slot (`relocate_tensor`), so the creep never
//! pins the frontier for the length of its sweep.
//!
//! Off CUDA the arena is one tensor: there is no reservation to claim from, and
//! the scheduler's tests drive the same paths on the CPU.

use candle::{DType, Device, Result, Tensor};

use super::store::RowStore;

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    claim_arena_slots, relocate_tensor, slot_stride, ArenaSlot, SlotTenant, SpanRegion,
};

/// One block of rows: physical rows `first..first + rows`.
struct Block {
    first: usize,
    rows: usize,
    /// Device address of the block's slot — the key a compaction's moves are
    /// planned by. Zero off CUDA, where nothing moves.
    base: u64,
    tensor: Tensor,
}

/// Rows of one residual, held in a tenant's slot arenas.
pub struct ResidualArena {
    device: Device,
    width: usize,
    dtype: DType,
    capacity: usize,
    /// Bytes a block's slot is claimed for — the stride its arenas are keyed by,
    /// and what a compaction of them is planned against.
    block_bytes: usize,
    blocks: Vec<Block>,
}

/// Whole rows of `row_bytes` that one slot holds: as many as fit a region.
#[cfg(feature = "cuda")]
fn rows_per_block(row_bytes: usize) -> Result<usize> {
    let region = SpanRegion::bytes();
    if row_bytes == 0 || row_bytes > region {
        candle::bail!(
            "residual arena: a {row_bytes} B row does not fit a {region} B region, so no slot \
             can hold one"
        );
    }
    // Rounded down so the slot's stride — `slot_stride` aligns it up — still fits
    // the region.
    let rows = region / row_bytes;
    Ok(if slot_stride(rows * row_bytes) > region {
        rows - 1
    } else {
        rows
    }
    .max(1))
}

impl ResidualArena {
    /// What [`Self::claim`] takes of the span for `rows` rows of `width`
    /// elements of `dtype`: on CUDA, one region-sized block per
    /// `rows_per_block` rows, each at its slot's share of the region; elsewhere
    /// the rows themselves.
    ///
    /// Not the rows' bytes. The blocks are whole slots, so a half's last block
    /// claims its region however few rows it holds: priced at the rows, a
    /// 22-member cohort's residual read under the eight regions its two halves
    /// claimed, and its last window was refused two regions short (WhiteWorld,
    /// c8221fe39, 2026-10-19).
    pub fn claim_bytes(device: &Device, rows: usize, width: usize, dtype: DType) -> usize {
        let row_bytes = width * dtype.size_in_bytes();
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => rows_per_block(row_bytes).map_or(0, |per_block| {
                rows.div_ceil(per_block)
                    * candle_nn::kv_cache::slot_share_bytes(per_block * row_bytes)
            }),
            _ => rows * row_bytes,
        }
    }

    /// Claim room for `rows` rows of `width` elements of `dtype`.
    ///
    /// **Between forwards.** A new arena is a region claim, which takes the arena
    /// window, and a forward refuses it. All or nothing: a claim the span cannot
    /// meet gives back the slots it took before the error is returned.
    pub fn claim(device: &Device, rows: usize, width: usize, dtype: DType) -> Result<Self> {
        if width == 0 {
            candle::bail!("residual arena: rows of no width");
        }
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Self::claim_slots(device, rows, width, dtype),
            _ => {
                let blocks = if rows == 0 {
                    Vec::new()
                } else {
                    vec![Block {
                        first: 0,
                        rows,
                        base: 0,
                        tensor: Tensor::zeros((rows, width), dtype, device)?,
                    }]
                };
                Ok(Self {
                    device: device.clone(),
                    width,
                    dtype,
                    capacity: rows,
                    block_bytes: rows * width * dtype.size_in_bytes(),
                    blocks,
                })
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn claim_slots(device: &Device, rows: usize, width: usize, dtype: DType) -> Result<Self> {
        let row_bytes = width * dtype.size_in_bytes();
        let per_block = rows_per_block(row_bytes)?;
        let block_bytes = per_block * row_bytes;
        let slots = if rows == 0 {
            Vec::new()
        } else {
            claim_arena_slots(
                device,
                SlotTenant::CreepResidual,
                block_bytes,
                rows.div_ceil(per_block),
            )?
        };
        let mut blocks = Vec::with_capacity(slots.len());
        let mut first = 0;
        for slot in slots {
            let n = per_block.min(rows - first);
            let base = slot.ptr();
            // Anchored to the slot: the slot goes back to its arena only once
            // every view of this block has dropped.
            let tensor = Arc::new(slot).tensor(0, dtype, (n, width), device)?;
            blocks.push(Block {
                first,
                rows: n,
                base,
                tensor,
            });
            first += n;
        }
        Ok(Self {
            device: device.clone(),
            width,
            dtype,
            capacity: rows,
            block_bytes,
            blocks,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Bytes a block's slot is claimed for — what a compaction of this arena's
    /// tenant is planned against.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Move every block whose slot is the source of a move in `moves` onto its
    /// destination, taking that destination out of the map; answers how many
    /// moved.
    ///
    /// **Between forwards, and only by the arena's holder.** Each move is one
    /// device copy of the block's rows on the primary stream — ordered behind
    /// every launch that wrote or read the block — and the block is rebuilt as a
    /// view of the new slot, which every later reader resolves. The old slot goes
    /// back once its last view drops. Between forwards nothing else holds a view:
    /// a step's input parts live only inside the step.
    #[cfg(feature = "cuda")]
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        let mut moved = 0;
        for b in &mut self.blocks {
            // The block's view starts at its slot's base, so its base is the key
            // `relocate_tensor` finds it by.
            let dst = moves.get(&b.base).map(ArenaSlot::ptr);
            if relocate_tensor(&mut b.tensor, moves)? {
                b.base = dst.expect("a block relocates only onto a planned destination");
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// The rows as one `[rows, width]` tensor, copied — for a reader that needs
    /// the values themselves rather than a copy into another residual.
    pub fn to_tensor(&self) -> Result<Tensor> {
        let parts: Vec<Tensor> = self.blocks.iter().map(|b| b.tensor.clone()).collect();
        match parts.as_slice() {
            [] => Tensor::zeros((0, self.width), self.dtype, &self.device),
            [one] => one.copy(),
            _ => Tensor::cat(&parts, 0),
        }
    }
}

impl RowStore for ResidualArena {
    fn width(&self) -> usize {
        self.width
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn block_at(&self, row: usize) -> Result<(&Tensor, usize, usize)> {
        let i = self.blocks.partition_point(|b| b.first + b.rows <= row);
        match self.blocks.get(i) {
            Some(b) if b.first <= row => Ok((&b.tensor, row - b.first, b.first + b.rows - row)),
            _ => candle::bail!("residual arena: row {row} of a {}-row arena", self.capacity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::residual_rows::row_map::RowCopy;
    use crate::models::residual_rows::store::{copy_rows, SpanRows};

    #[test]
    fn a_cpu_arena_holds_its_rows_and_refuses_past_them() {
        let a = ResidualArena::claim(&Device::Cpu, 3, 4, DType::F32).unwrap();
        assert_eq!(a.capacity(), 3);
        assert_eq!(a.block_at(2).unwrap().1, 2);
        assert!(a.block_at(3).is_err());
    }

    #[test]
    fn an_empty_arena_holds_nothing() {
        let a = ResidualArena::claim(&Device::Cpu, 0, 4, DType::F32).unwrap();
        assert_eq!(a.capacity(), 0);
        assert!(a.block_at(0).is_err());
    }

    /// Rows written into an arena read back exactly, through its own blocks.
    #[test]
    fn rows_round_trip_through_an_arena() {
        let src =
            Tensor::from_vec((0..12).map(|v| v as f32).collect(), (4, 3), &Device::Cpu).unwrap();
        let a = ResidualArena::claim(&Device::Cpu, 4, 3, DType::F32).unwrap();
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: 4,
        }];
        copy_rows(&SpanRows::new(&src, 3).unwrap(), &a, &plan).unwrap();
        assert_eq!(
            a.to_tensor().unwrap().to_vec2::<f32>().unwrap(),
            src.to_vec2::<f32>().unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_block_is_as_many_whole_rows_as_fit_a_region() -> Result<()> {
        let region = SpanRegion::bytes();
        assert_eq!(rows_per_block(1 << 20)?, region >> 20);
        assert_eq!(rows_per_block(region)?, 1);
        assert_eq!(rows_per_block(40 * 1024)?, region / (40 * 1024));
        assert!(rows_per_block(region + 1).is_err());
        Ok(())
    }

    /// `rows` rows of `width` on `device`, row `r` holding `r` throughout.
    #[cfg(feature = "cuda")]
    fn numbered_rows(rows: usize, width: usize, device: &Device) -> Result<Tensor> {
        Tensor::arange(0f32, rows as f32, device)?
            .reshape((rows, 1))?
            .broadcast_as((rows, width))?
            .contiguous()
    }

    /// Each row's first element, in order.
    #[cfg(feature = "cuda")]
    fn row_ids(a: &ResidualArena) -> Result<Vec<f32>> {
        a.to_tensor()?.narrow(1, 0, 1)?.flatten_all()?.to_vec1()
    }

    /// On a card, a residual wider than one region is laid over several slots
    /// — whole rows each — and rows written across slot boundaries read back
    /// exactly.
    #[cfg(feature = "cuda")]
    #[test]
    fn rows_cross_slot_blocks_exactly() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        // Rows of 1 MiB: 16 to a slot, so 40 rows are blocks of 16, 16 and 8.
        let width = (1 << 20) / 4;
        let rows = 40;
        let a = ResidualArena::claim(&device, rows, width, DType::F32)?;
        let per = SpanRegion::bytes() >> 20;
        assert_eq!(
            a.blocks.iter().map(|b| b.rows).collect::<Vec<_>>(),
            vec![per, per, rows - 2 * per]
        );
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: rows,
        }];
        copy_rows(
            &SpanRows::new(&numbered_rows(rows, width, &device)?, width)?,
            &a,
            &plan,
        )?;
        assert_eq!(
            row_ids(&a)?,
            (0..rows).map(|r| r as f32).collect::<Vec<_>>()
        );
        Ok(())
    }

    /// **The price is the blocks a claim takes, not its rows.** Forty 1 MiB rows
    /// are blocks of 16, 16 and 8, and the last block claims its whole region:
    /// three regions, where the rows alone are two and a half.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_claim_is_priced_at_its_blocks() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let width = (1 << 20) / 4;
        assert_eq!(
            ResidualArena::claim_bytes(&device, 40, width, DType::F32),
            3 * SpanRegion::bytes()
        );
        assert_eq!(
            ResidualArena::claim_bytes(&Device::Cpu, 40, width, DType::F32),
            40 << 20,
            "off CUDA the rows are the allocation"
        );
        Ok(())
    }

    /// **A compaction move keeps the rows exactly.** The block is copied onto
    /// its destination, read back from there, and only the moved block's slot
    /// changes; a block no move names stays where it was.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_relocated_block_keeps_its_rows() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let width = (1 << 20) / 4;
        let rows = 40;
        let mut a = ResidualArena::claim(&device, rows, width, DType::F32)?;
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: rows,
        }];
        copy_rows(
            &SpanRows::new(&numbered_rows(rows, width, &device)?, width)?,
            &a,
            &plan,
        )?;
        let before: Vec<u64> = a.blocks.iter().map(|b| b.base).collect();
        let dst = claim_arena_slots(&device, SlotTenant::CreepResidual, a.block_bytes(), 1)?
            .pop()
            .expect("one slot claimed");
        let dst_ptr = dst.ptr();
        let mut moves = HashMap::from([(before[1], dst)]);
        assert_eq!(a.relocate(&mut moves)?, 1);
        assert!(moves.is_empty(), "the move was taken");
        let after: Vec<u64> = a.blocks.iter().map(|b| b.base).collect();
        assert_eq!(after, vec![before[0], dst_ptr, before[2]]);
        assert_eq!(
            row_ids(&a)?,
            (0..rows).map(|r| r as f32).collect::<Vec<_>>()
        );
        Ok(())
    }
}
