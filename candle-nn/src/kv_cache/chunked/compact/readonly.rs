//! Declares a pass's source slots immutable until the pass ends.
//!
//! While a pass runs, nothing should write the K/V it is moving. Its sources are
//! declared to the readonly-regions guard, which names any writer at the moment
//! of the write. The guard releases on drop, so every early return clears the
//! declaration, since a stale one would blame an innocent allocation for a later
//! write.

#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
use super::*;

/// Declares this pass's **source** slots immutable for the rest of the pass, and
/// gives them back on the way out however the pass ends.
///
/// # Why the sources, and only the sources
///
/// Every other check in this module verifies a *result* — an address is in bounds, a
/// pointer word holds what it was given, a destination holds what its source held.
/// None of them can answer "who", and by the time a byte is wrong the writer is
/// gone. This one asks the question the other way round: for the window in which
/// nothing at all is supposed to write K/V, declare the bytes this pass is about to
/// move and let `forbid_write` name any writer at the moment of the write.
///
/// Not the whole K/V band, though that is the stronger claim, because this pass
/// writes the band itself: `copy_records` fills every destination and already calls
/// `forbid_write` on each one, so declaring the band makes the pass panic on its own
/// copy. The sources are the half nothing should touch — the copy only reads them,
/// and they stay live until the holders naming them have been rewritten.
///
/// A guard rather than a pair of calls because the pass has several early returns,
/// and a declaration left standing past any of them reports the next legitimate K/V
/// write as a violation. That failure mode is in the harness's own notes: a stale
/// declared region blames an innocent allocation, which is worse than not checking,
/// because a guard that cries wolf is one you stop reading.
///
/// `release_below(weight_floor)` is the release, the same idiom `set_weight_floor`
/// uses: the floor is the top of the K/V side, so it clears exactly what was
/// declared while the expert zone above keeps its own declaration.
#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
pub(super) struct ReadonlySources {
    pub(super) floor: u64,
}

#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
impl ReadonlySources {
    pub(super) fn declare(device: &candle::Device, records: &CopyRecords) -> Option<Self> {
        let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
            return None;
        };
        let layout = super::super::region_pool::span_layout(gpu_id)?;
        let mut spans: Vec<(u64, usize)> = records
            .srcs
            .iter()
            .zip(&records.lens)
            .map(|(&s, &l)| (s as u64, l.max(0) as usize))
            .collect();
        if spans.is_empty() {
            return None;
        }
        // Merged, because the sources of one pass are thousands of slots that sit end
        // to end inside a handful of arenas: declared individually they would both
        // overflow the table and turn the check into a thousands-entry scan.
        candle::readonly_regions::declare_merged("kv compaction source", &mut spans);
        Some(Self {
            floor: layout.weight_floor,
        })
    }
}

#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
impl Drop for ReadonlySources {
    fn drop(&mut self) {
        candle::readonly_regions::release_below(self.floor);
    }
}
