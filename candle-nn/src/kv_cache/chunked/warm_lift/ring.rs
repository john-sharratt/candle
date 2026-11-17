//! The warm lift's pinned ring: two slots the host fills while the scatter
//! kernel drains the other (`docs/warm_lift.md` §3.2).
//!
//! Each slot is mapped, write-combined pinned memory — the CPU only writes it,
//! the kernel only reads it, at its host address — holding a slot's payloads
//! and, behind them, its descriptor table. Allocated once and never grown:
//! pinned host memory is the expert weights' tier, and what the lift takes of it
//! is this fixed number.
//!
//! A slot carries the event recorded after the scatter that read it. Refilling
//! a slot waits on that event, and that is the only wait the lift has.

use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaStream};
use candle::cuda_backend::kernels::simple::warm_lift::WarmLiftEntry;
use candle::cuda_backend::WrapErr;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{CudaDevice, Result};

/// The smallest piece a kernel pulling pinned memory moves at the link's rate:
/// an 819,200 B projection at ≈ 11.9 GB/s (`docs/moe_live_dispatch_design.md`
/// §0.10.1).
const LINK_RATE_PIECE_BYTES: usize = 819_200;

/// A slot is that piece times eight, so the ~15 µs a slot costs (a launch and
/// an event) is ~3 % of the time its bytes take on a 12 GB/s link — at one
/// piece it would be ~20 %. Past eight the pipeline only takes longer to fill
/// and the weights lose pinned memory. Rounded up to a power of two: 8 MiB.
pub(crate) const SLOT_BYTES: usize = (LINK_RATE_PIECE_BYTES * 8).next_power_of_two();

/// The smallest band a lift carries is a few hundred bytes (a sub-4-bit
/// palette band), so a slot's table has room for one entry per 256 bytes of
/// payload; a batch is cut at whichever bound it reaches first.
const BYTES_PER_ENTRY_BOUND: usize = 256;

/// The two slots.
const SLOTS: usize = 2;

/// One slot: its memory, and the event recorded after the last scatter that
/// read it.
struct RingSlot {
    buf: PinnedBuf,
    read: Option<CudaEvent>,
}

/// See the module documentation.
pub struct WarmLiftRing {
    slots: Vec<RingSlot>,
    next: usize,
    slot_bytes: usize,
    max_entries: usize,
}

impl WarmLiftRing {
    /// The production ring: two [`SLOT_BYTES`] slots.
    pub fn new() -> Result<Self> {
        Self::with_slot_bytes(SLOT_BYTES)
    }

    /// A ring of `slot_bytes` slots — production uses [`Self::new`]; a test
    /// takes small slots to cut a lift into many batches.
    pub fn with_slot_bytes(slot_bytes: usize) -> Result<Self> {
        let max_entries = (slot_bytes / BYTES_PER_ENTRY_BOUND).max(1);
        let table_bytes = max_entries * std::mem::size_of::<WarmLiftEntry>();
        let slots = (0..SLOTS)
            .map(|_| {
                Ok(RingSlot {
                    buf: PinnedBuf::alloc_owned_mapped(slot_bytes + table_bytes)?,
                    read: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            slots,
            next: 0,
            slot_bytes,
            max_entries,
        })
    }

    /// Payload bytes a slot holds.
    pub(crate) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Entries a slot's descriptor table holds.
    pub(crate) fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// The next slot, once the scatter that last read it has: its payload
    /// region and its table region, both writable. The wait is the ring's
    /// only one, and a slot no scatter has read since it was last filled has
    /// nothing to wait for.
    pub(crate) fn acquire(&mut self, device: &CudaDevice) -> Result<SlotFill<'_>> {
        let k = self.next;
        self.next = (self.next + 1) % SLOTS;
        let slot = &mut self.slots[k];
        if let Some(event) = slot.read.as_ref() {
            if !event.is_complete() {
                // An event wait is host protocol, refused on a thread recording
                // a wave; a lift never runs inside one, and the pause makes that
                // a no-op rather than an assumption.
                let _paused = device.pause_capture()?;
                event.synchronize().w()?;
            }
        }
        let slot_bytes = self.slot_bytes;
        let max_entries = self.max_entries;
        let (payload, table) = slot.buf.as_mut_slice().split_at_mut(slot_bytes);
        Ok(SlotFill {
            payload,
            table: &mut table[..max_entries * std::mem::size_of::<WarmLiftEntry>()],
            read: &mut slot.read,
        })
    }
}

/// A slot being filled: write its payload and table, launch its scatter, then
/// [`Self::scattered`] on the stream that ran it.
pub(crate) struct SlotFill<'a> {
    pub payload: &'a mut [u8],
    table: &'a mut [u8],
    read: &'a mut Option<CudaEvent>,
}

impl SlotFill<'_> {
    /// Write `entries` as the slot's descriptor table; the table's address.
    pub(crate) fn write_table(
        &mut self,
        entries: &[WarmLiftEntry],
    ) -> Result<*const WarmLiftEntry> {
        let bytes = std::mem::size_of_val(entries);
        if bytes > self.table.len() {
            candle::bail!(
                "warm lift: {} entries past the slot table's {}",
                entries.len(),
                self.table.len() / std::mem::size_of::<WarmLiftEntry>()
            );
        }
        // SAFETY: `WarmLiftEntry` is `repr(C)` plain data, and `bytes` of it
        // fit the table region checked above.
        let raw = unsafe { std::slice::from_raw_parts(entries.as_ptr() as *const u8, bytes) };
        self.table[..bytes].copy_from_slice(raw);
        Ok(self.table.as_ptr() as *const WarmLiftEntry)
    }

    /// The scatter reading this slot is on `stream`: record the event a
    /// refill waits on.
    pub(crate) fn scattered(self, stream: &CudaStream) -> Result<()> {
        if self.read.is_none() {
            *self.read = Some(stream.context().new_event(None).w()?);
        }
        if let Some(event) = self.read.as_ref() {
            event.record(stream).w()?;
        }
        Ok(())
    }
}
