//! The device-side slot table: one fixed allocation per device, for the life of
//! the process.
//!
//! Everything about this module exists to keep an assert from allocating. The
//! table is claimed once, on first use, and every later assert folds into a slot
//! that already exists — so an assert issued inside a wave adds a launch and
//! nothing else, and cannot move the wave arena's allocation layout underneath
//! the code it is observing.

use crate::cuda_backend::{CudaDevice, DeviceId};
use crate::Result;
use candle_kernels::simple::tensor_assert::run_tensor_assert_reset;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Distinct assert names a process may use. Names beyond this are reported once
/// and then ignored; the table is never grown, because growing it would mean
/// reallocating on a hot path.
pub const MAX_SLOTS: usize = 4096;

/// Host mirror of `AssertSlot` in `simple/tensor_assert.cu`. The layouts must
/// agree exactly — the drain reinterprets the device buffer as this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct AssertSlot {
    pub nan: u32,
    pub inf: u32,
    pub min_key: u32,
    pub max_key: u32,
    /// 1-based order stamp of the first bad observation; 0 means never bad.
    pub seq: u32,
    pub elems: u32,
    pub pad0: u32,
    pub pad1: u32,
}

impl AssertSlot {
    /// Whether this slot ever saw a non-finite value.
    pub fn is_bad(&self) -> bool {
        self.nan > 0 || self.inf > 0
    }

    /// The smallest finite value observed, or `None` if every element was
    /// non-finite (or the slot was never written).
    pub fn min(&self) -> Option<f32> {
        (self.min_key != u32::MAX).then(|| key_to_f32(self.min_key))
    }

    /// The largest finite value observed, or `None` as for [`Self::min`].
    pub fn max(&self) -> Option<f32> {
        (self.max_key != 0).then(|| key_to_f32(self.max_key))
    }
}

/// Inverse of the kernel's order-preserving float→u32 map. See the header
/// comment of `simple/tensor_assert.cu` for the forward direction.
pub fn key_to_f32(key: u32) -> f32 {
    let bits = if key & 0x8000_0000 != 0 {
        key ^ 0x8000_0000
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// One device's table plus its ordering counter.
///
/// The counter lives in slot `MAX_SLOTS` — one extra slot on the end of the
/// same allocation — so the whole instrument is a single buffer rather than two
/// that could get out of step.
pub struct DeviceSlots {
    buf: CudaSlice<u32>,
}

const WORDS_PER_SLOT: usize = 8;

impl DeviceSlots {
    fn new(dev: &CudaDevice) -> Result<Self> {
        let words = (MAX_SLOTS + 1) * WORDS_PER_SLOT;
        let buf = dev
            .alloc_zeros::<u32>(words)
            .map_err(|e| crate::Error::Msg(format!("tensor_assert: slot table alloc: {e}")))?;
        let me = Self { buf };
        me.reset(&dev.cuda_stream())?;
        Ok(me)
    }

    /// Device address of slot `idx`.
    pub fn slot_ptr(&self, idx: usize, stream: &CudaStream) -> u64 {
        let (base, _g) = self.buf.device_ptr(stream);
        base + (idx * WORDS_PER_SLOT * std::mem::size_of::<u32>()) as u64
    }

    /// Device address of the shared ordering counter.
    pub fn seq_ptr(&self, stream: &CudaStream) -> u64 {
        self.slot_ptr(MAX_SLOTS, stream)
    }

    /// Restore every slot to the reductions' identity. One small launch, no
    /// synchronisation — this is how an epoch boundary is drawn.
    pub fn reset(&self, stream: &CudaStream) -> Result<()> {
        let (ptr, _g) = self.buf.device_ptr(stream);
        unsafe {
            run_tensor_assert_reset(
                ptr as *mut std::ffi::c_void,
                (MAX_SLOTS + 1) as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }

    /// Copy the whole table to host. This is the ONLY read of the table, and it
    /// belongs at a synchronisation the caller already performs — see
    /// [`super::drain`].
    pub fn read(&self, dev: &CudaDevice) -> Result<Vec<AssertSlot>> {
        let words = dev
            .memcpy_dtov(&self.buf)
            .map_err(|e| crate::Error::Msg(format!("tensor_assert: slot table read: {e}")))?;
        Ok(words
            .as_chunks::<WORDS_PER_SLOT>()
            .0
            .iter()
            .take(MAX_SLOTS)
            .map(|w| AssertSlot {
                nan: w[0],
                inf: w[1],
                min_key: w[2],
                max_key: w[3],
                seq: w[4],
                elems: w[5],
                pad0: w[6],
                pad1: w[7],
            })
            .collect())
    }

}

type Registry = Mutex<HashMap<DeviceId, DeviceSlots>>;

fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Run `f` against `dev`'s slot table, creating it on first use.
///
/// The creation is the one allocation this instrument ever makes, and it
/// happens on the first assert of the process — which is why the first assert
/// of a run should be issued at load time, not from inside a wave.
pub fn with_slots<R>(dev: &CudaDevice, f: impl FnOnce(&DeviceSlots) -> Result<R>) -> Result<R> {
    let mut reg = registry()
        .lock()
        .map_err(|_| crate::Error::Msg("tensor_assert: slot registry poisoned".to_string()))?;
    let slots = match reg.entry(dev.id()) {
        std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => v.insert(DeviceSlots::new(dev)?),
    };
    f(slots)
}

#[cfg(test)]
mod tests {
    use super::{key_to_f32, AssertSlot};

    /// The kernel's forward map, duplicated here so the round trip is asserted
    /// against an independent statement of it rather than against itself.
    fn f32_to_key(v: f32) -> u32 {
        let b = v.to_bits();
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    }

    #[test]
    fn the_float_key_round_trips_and_preserves_order() {
        let vals = [
            f32::MIN,
            -1.0e30,
            -1.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.5,
            1.0e30,
            f32::MAX,
        ];
        for v in vals {
            let back = key_to_f32(f32_to_key(v));
            assert_eq!(
                back.to_bits(),
                v.to_bits(),
                "round trip changed {v} to {back}"
            );
        }
        // Ordering is the whole reason the map exists: integer atomicMin/Max on
        // the key must be float min/max.
        for w in vals.windows(2) {
            assert!(
                f32_to_key(w[0]) < f32_to_key(w[1]),
                "key order broke between {} and {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn an_untouched_slot_reports_no_range_and_is_not_bad() {
        let empty = AssertSlot {
            nan: 0,
            inf: 0,
            min_key: u32::MAX,
            max_key: 0,
            seq: 0,
            elems: 0,
            pad0: 0,
            pad1: 0,
        };
        assert!(!empty.is_bad());
        assert_eq!(empty.min(), None);
        assert_eq!(empty.max(), None);
    }

    #[test]
    fn a_slot_reports_the_exact_values_its_keys_encode() {
        let s = AssertSlot {
            nan: 3,
            inf: 1,
            min_key: f32_to_key(-2.5),
            max_key: f32_to_key(7.25),
            seq: 4,
            elems: 100,
            pad0: 0,
            pad1: 0,
        };
        assert!(s.is_bad());
        assert_eq!(s.min(), Some(-2.5));
        assert_eq!(s.max(), Some(7.25));
    }
}
