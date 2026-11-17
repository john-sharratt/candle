//! A lift's bands cut into ring-slot batches.
//!
//! Every band of every layer is one entry: where its payload sits in the warm
//! tier, how long it is, and the device address it goes to. A batch is a run of
//! consecutive entries whose payloads, each started on a 16-byte boundary, fit
//! one ring slot and whose count fits the slot's descriptor table. Nothing
//! here touches memory; it is the arithmetic the lift and its tests share.

use std::ops::Range;

use candle::Result;

/// Where each payload starts in a slot: the scatter's wide copy wants 16 bytes.
pub(crate) const ALIGN: usize = 16;

/// One band to lift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiftBand {
    /// The layer whose backing holds the source.
    pub layer: usize,
    /// The source band's warm arena and slot.
    pub src_arena: usize,
    pub src_chunk: usize,
    /// Payload bytes.
    pub bytes: usize,
    /// The destination slot's device address.
    pub dst: u64,
}

/// One ring slot's share of a lift: `bands` of the plan, each at its offset in
/// the slot, `payload` bytes in all, alignment padding included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotBatch {
    pub bands: Range<usize>,
    pub offsets: Vec<usize>,
    pub payload: usize,
}

/// `bands` cut into batches of at most `slot_bytes` of payload and
/// `max_entries` entries, in order, a band never split. A band longer than a
/// slot cannot be lifted through the ring and is refused.
pub(crate) fn batches(
    bands: &[LiftBand],
    slot_bytes: usize,
    max_entries: usize,
) -> Result<Vec<SlotBatch>> {
    let mut out: Vec<SlotBatch> = Vec::new();
    let mut start = 0usize;
    let mut offsets: Vec<usize> = Vec::new();
    let mut at = 0usize;
    for (i, band) in bands.iter().enumerate() {
        if band.bytes > slot_bytes {
            candle::bail!(
                "warm lift: a {} B band is longer than the {slot_bytes} B ring slot",
                band.bytes
            );
        }
        let placed = at.next_multiple_of(ALIGN);
        if placed + band.bytes > slot_bytes || offsets.len() == max_entries {
            out.push(SlotBatch {
                bands: start..i,
                offsets: std::mem::take(&mut offsets),
                payload: at,
            });
            start = i;
            offsets.push(0);
            at = band.bytes;
            continue;
        }
        offsets.push(placed);
        at = placed + band.bytes;
    }
    if !offsets.is_empty() {
        out.push(SlotBatch {
            bands: start..bands.len(),
            offsets,
            payload: at,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(bytes: usize) -> LiftBand {
        LiftBand {
            layer: 0,
            src_arena: 0,
            src_chunk: 0,
            bytes,
            dst: 0,
        }
    }

    /// **Bands fill a slot in order, each on a 16-byte boundary**, and the
    /// next starts a new slot when it would not fit.
    #[test]
    fn bands_are_packed_aligned_and_cut_at_the_slot() {
        // 40 B at 0, 16 B at 48 (ends 64); 100 B would end at 164 > 128, so a
        // new slot: 100 B at 0; 30 B would start at 112 and end at 142, so a
        // third slot.
        let bands = [band(40), band(16), band(100), band(30)];
        assert_eq!(
            batches(&bands, 128, 64).unwrap(),
            vec![
                SlotBatch {
                    bands: 0..2,
                    offsets: vec![0, 48],
                    payload: 64,
                },
                SlotBatch {
                    bands: 2..3,
                    offsets: vec![0],
                    payload: 100,
                },
                SlotBatch {
                    bands: 3..4,
                    offsets: vec![0],
                    payload: 30,
                },
            ]
        );
    }

    /// **The descriptor table's room cuts a batch too**, and every band is in
    /// exactly one batch.
    #[test]
    fn the_table_bounds_a_batch_and_every_band_is_placed_once() {
        let bands: Vec<LiftBand> = (0..10).map(|_| band(16)).collect();
        let got = batches(&bands, 1 << 20, 4).unwrap();
        let spans: Vec<Range<usize>> = got.iter().map(|b| b.bands.clone()).collect();
        assert_eq!(spans, vec![0..4, 4..8, 8..10]);
        assert_eq!(got[0].offsets, vec![0, 16, 32, 48]);
        assert_eq!(got[2].payload, 32);
    }

    /// A band longer than a slot is refused; no bands is no batch.
    #[test]
    fn a_band_past_the_slot_is_refused_and_nothing_is_no_batch() {
        assert!(batches(&[band(129)], 128, 8).is_err());
        assert_eq!(batches(&[], 128, 8).unwrap(), Vec::new());
    }
}
