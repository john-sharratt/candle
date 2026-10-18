//! The word a live launch's workers report a lost cold expert in.
//!
//! A worker waiting for a cold expert gives it up when the abort word is raised
//! (the stager or the pipeline thread died) or when the wait passes
//! `SPIN_LIMIT_NS` (a drive that stopped answering). It does not trap — a trap
//! is a sticky error that poisons the CUDA context for the whole process —
//! but claims this mapped word with what it waited on, and every other cold
//! wait that sees it gives its expert up at once (`kernel.cuh`, "A live expert
//! table"). The forward's results are then garbage, so the host reads the word
//! once the forward has synchronised and fails it (`ExpertCache::take_fault`);
//! the wave driver rolls the wave back and the context stays usable. A wait
//! that ran out is a stall ([`EXPERT_STALL_MARKER`]), and the scheduler runs
//! the same work again; one that ended on the abort word fails it.
//!
//! While the word is set nothing builds on the faulted forward: the dispatch
//! refuses every launch ([`FaultWord::check`]) — so a caller that never takes
//! the fault gets an error, not a garbled forward, and the wave driver takes
//! it on that error path too — and the pipeline thread lands no demand
//! promotion ([`FaultWord::is_set`]), since a given-up item's promotion slot
//! was never written. The word is cleared only after the pipeline thread has
//! dropped those promotions.
//!
//! The bits (`candle_kernels::quantized::MOE_FAULT_*`): bit 63 set when the
//! wait ended on the abort word, the row in bits 48–62, the expert in bits
//! 32–47, the microseconds waited in bits 0–31, saturating.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{fence, AtomicU64, Ordering};

use candle::cuda_backend::cudarc::driver::sys;
use candle::{bail, Error, Result};
use candle_kernels::quantized::{
    MOE_FAULT_ABORTED, MOE_FAULT_CPU_EXPERT, MOE_FAULT_EXPERT_SHIFT, MOE_FAULT_ROW_SHIFT,
};

/// A cold expert a live launch gave up on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertFault {
    /// The wait ended on the abort word, not the spin limit.
    pub(crate) aborted: bool,
    pub(crate) row: usize,
    pub(crate) expert: usize,
    pub(crate) waited_us: u64,
}

impl ExpertFault {
    /// The fault a word holds; `None` for 0, no fault.
    pub(crate) fn decode(word: u64) -> Option<Self> {
        (word != 0).then_some(Self {
            aborted: word & MOE_FAULT_ABORTED != 0,
            row: ((word >> MOE_FAULT_ROW_SHIFT) & 0x7fff) as usize,
            expert: ((word >> MOE_FAULT_EXPERT_SHIFT) & 0xffff) as usize,
            waited_us: word & 0xffff_ffff,
        })
    }

    /// The word a wait that gave this up claims: [`Self::decode`]'s inverse,
    /// the row, expert and microseconds saturating to their fields.
    pub(crate) fn encode(self) -> u64 {
        (if self.aborted { MOE_FAULT_ABORTED } else { 0 })
            | ((self.row as u64 & 0x7fff) << MOE_FAULT_ROW_SHIFT)
            | ((self.expert as u64 & 0xffff) << MOE_FAULT_EXPERT_SHIFT)
            | self.waited_us.min(0xffff_ffff)
    }

    /// Whether the wait given up was the down launch's on the CPU expert pool,
    /// not a cold expert's.
    pub(crate) fn is_cpu(self) -> bool {
        self.expert as u64 == MOE_FAULT_CPU_EXPERT
    }

    /// The forward's error. A wait that ran out carries [`EXPERT_STALL_MARKER`]:
    /// the forward may simply run again. One that ended on the abort word does
    /// not — a dead stager or pipeline thread answers no retry.
    pub(crate) fn error(self) -> Error {
        let outcome = if self.aborted {
            "the forward's results are discarded"
        } else {
            EXPERT_STALL_MARKER
        };
        if self.is_cpu() {
            let why = if self.aborted {
                "the expert stager or pipeline thread died"
            } else {
                "the pool stopped or did not answer in time"
            };
            return Error::Msg(format!(
                "expert forward failed: MoE row {}'s CPU experts were given up after {:.1} ms \
                 ({why}); {outcome}",
                self.row,
                self.waited_us as f64 / 1e3
            ));
        }
        let why = if self.aborted {
            "the expert stager or pipeline thread died"
        } else {
            "the stager did not publish it in time — a pack read stalled"
        };
        Error::Msg(format!(
            "expert forward failed: cold expert {} of MoE row {} was given up after {:.1} ms \
             ({why}); {outcome}",
            self.expert,
            self.row,
            self.waited_us as f64 / 1e3
        ))
    }
}

/// Ends the error of a forward the device gave part of up because a wait ran
/// out — a cold expert its pack read published late, the CPU pool answering
/// late — rather than because a thread died. The forward was discarded whole
/// and rolled back, the context is usable, and the next attempt waits afresh,
/// so the scheduler runs the same work again ([`is_expert_stall`]) instead of
/// failing it: a pack read that stalls for a second under disk contention
/// otherwise failed zend's load, and in a decode, every turn in the step.
pub const EXPERT_STALL_MARKER: &str = "the forward is discarded and may run again";

/// Whether `err` is a forward given up on a stall — see [`EXPERT_STALL_MARKER`].
pub fn is_expert_stall(err: &Error) -> bool {
    err.to_string().contains(EXPERT_STALL_MARKER)
}

/// The mapped fault word.
pub(crate) struct FaultWord {
    host: *mut u64,
    dev: u64,
}

// SAFETY: a mapped word the device claims with an atomic compare-and-swap and
// the host reads and clears with volatile accesses, only after a synchronise.
unsafe impl Send for FaultWord {}
unsafe impl Sync for FaultWord {}

impl FaultWord {
    pub(crate) fn new() -> Result<Self> {
        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, 8, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            bail!("expert cache: mapped fault word allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe { sys::cuMemFreeHost(raw) };
            bail!("expert cache: fault word has no device address: {r:?}");
        }
        // SAFETY: eight bytes just allocated.
        unsafe { ptr::write_volatile(raw as *mut u64, 0) };
        Ok(Self {
            host: raw as *mut u64,
            dev,
        })
    }

    /// The word's device address, for `MoeLive::fault`.
    pub(crate) fn ptr(&self) -> u64 {
        self.dev
    }

    /// Whether a fault is claimed and not yet taken. A worker's claim is a
    /// compare-and-swap followed by a system fence, so a launch that started
    /// after it in stream order — which is what makes a promotion's ticket
    /// reclaimable — is ordered after the word reads set.
    pub(crate) fn is_set(&self) -> bool {
        // Ordered after whatever the caller read first (a started word).
        fence(Ordering::SeqCst);
        // SAFETY: the mapped word this struct owns.
        unsafe { ptr::read_volatile(self.host) != 0 }
    }

    /// `Err` while a fault is claimed and not yet taken: the forward it
    /// belongs to has not been failed, so nothing may be launched on top of it.
    pub(crate) fn check(&self) -> Result<()> {
        match self.peek() {
            Some(fault) => Err(fault.error()),
            None => Ok(()),
        }
    }

    /// The fault claimed and not yet cleared, if any. The word stays set: the
    /// caller clears it ([`Self::clear`]) once whatever must still see it set
    /// has run (`ExpertCache::take_fault`).
    pub(crate) fn peek(&self) -> Option<ExpertFault> {
        // SAFETY: the mapped word this struct owns.
        ExpertFault::decode(unsafe { ptr::read_volatile(self.host) })
    }

    /// Claim the word for `fault` unless a fault is already claimed — the host
    /// side of a worker's compare-and-swap, for a wait the CPU pool gives up
    /// (`cpu_pool::cold`). Every device wait that then polls the word gives its
    /// expert up at once, the down launch's wait on the pool among them.
    pub(crate) fn claim(&self, fault: ExpertFault) {
        // SAFETY: the mapped word this struct owns, 8-aligned, live as long
        // as `self`; the device claims it only with atomics.
        let word = unsafe { AtomicU64::from_ptr(self.host) };
        let _ = word.compare_exchange(0, fault.encode(), Ordering::SeqCst, Ordering::SeqCst);
    }

    /// Clear the word — only with every launch that could claim it complete
    /// (after the forward's synchronise), or a fault claimed after the read
    /// would be cleared unseen.
    pub(crate) fn clear(&self) {
        // SAFETY: the mapped word this struct owns; no launch is running to
        // claim it concurrently.
        unsafe { ptr::write_volatile(self.host, 0) };
        fence(Ordering::SeqCst);
    }
}

impl Drop for FaultWord {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; every launch that
        // could write it is done before the cache drops.
        unsafe { sys::cuMemFreeHost(self.host as *mut c_void) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// The word's bits, raw: abort flag, row, expert, microseconds waited.
    #[test]
    fn a_fault_word_decodes() {
        assert_eq!(ExpertFault::decode(0), None);
        let word = (1u64 << 63) | (37u64 << 48) | (511u64 << 32) | 1_000_123;
        assert_eq!(
            ExpertFault::decode(word),
            Some(ExpertFault {
                aborted: true,
                row: 37,
                expert: 511,
                waited_us: 1_000_123,
            })
        );
        assert_eq!(
            ExpertFault::decode(word).unwrap().encode(),
            word,
            "encode inverts"
        );
        let saturated = ExpertFault {
            aborted: false,
            row: 3,
            expert: 4,
            waited_us: 1 << 40,
        };
        assert_eq!(
            saturated.encode(),
            (3u64 << 48) | (4u64 << 32) | 0xffff_ffff
        );
        let timed_out = ExpertFault::decode((5u64 << 48) | (9u64 << 32) | 1_500).unwrap();
        assert!(!timed_out.aborted);
        assert_eq!(
            timed_out.error().to_string(),
            "expert forward failed: cold expert 9 of MoE row 5 was given up after 1.5 ms (the \
             stager did not publish it in time — a pack read stalled); the forward is discarded \
             and may run again"
        );
    }

    /// **A wait that ran out is a stall the scheduler retries; one that ended
    /// on the abort word is not** — a dead stager answers no retry.
    #[test]
    fn a_timed_out_wait_is_a_stall_and_an_aborted_one_is_not() {
        let timed_out = ExpertFault::decode((5u64 << 48) | (9u64 << 32) | 1_500).unwrap();
        assert!(is_expert_stall(&timed_out.error()));
        let cpu = ExpertFault::decode((7u64 << 48) | (MOE_FAULT_CPU_EXPERT << 32) | 2_500).unwrap();
        assert!(is_expert_stall(&cpu.error()));
        let aborted =
            ExpertFault::decode(MOE_FAULT_ABORTED | (5u64 << 48) | (9u64 << 32) | 7).unwrap();
        assert!(!is_expert_stall(&aborted.error()));
        assert_eq!(
            aborted.error().to_string(),
            "expert forward failed: cold expert 9 of MoE row 5 was given up after 0.0 ms (the \
             expert stager or pipeline thread died); the forward's results are discarded"
        );
        assert!(!is_expert_stall(&Error::Msg("out of memory".into())));
    }

    /// The CPU item's claim names the pool, not a cold expert: expert
    /// `0xffff`, the row, the wait.
    #[test]
    fn a_cpu_fault_names_the_pool() {
        let f = ExpertFault::decode((7u64 << 48) | (MOE_FAULT_CPU_EXPERT << 32) | 2_500).unwrap();
        assert!(f.is_cpu());
        assert_eq!(
            f.error().to_string(),
            "expert forward failed: MoE row 7's CPU experts were given up after 2.5 ms (the pool \
             stopped or did not answer in time); the forward is discarded and may run again"
        );
        assert!(!ExpertFault::decode((7u64 << 48) | (9u64 << 32) | 1)
            .unwrap()
            .is_cpu());
    }

    /// Reading the word leaves it set — everything that must still see it set
    /// runs between the read and the clear — and it refuses launches until
    /// cleared.
    #[test]
    fn a_claimed_word_stays_set_until_cleared() -> Result<()> {
        let _context = Device::new_cuda(0)?;
        let word = FaultWord::new()?;
        assert!(!word.is_set());
        assert!(word.check().is_ok());
        let claim = (12u64 << 48) | (300u64 << 32) | 1_000_001;
        word.claim(ExpertFault::decode(claim).unwrap());
        let fault = ExpertFault::decode(claim);
        assert_eq!(word.peek(), fault);
        // A second claim finds the word taken and leaves the first.
        word.claim(ExpertFault::decode((1u64 << 48) | (2u64 << 32) | 3).unwrap());
        assert_eq!(word.peek(), fault, "the first claim stands");
        assert_eq!(word.peek(), fault, "a peek does not clear");
        assert!(word.is_set());
        assert_eq!(
            word.check().map_err(|e| e.to_string()),
            Err(fault.unwrap().error().to_string())
        );
        word.clear();
        assert_eq!(word.peek(), None);
        assert!(!word.is_set());
        assert!(word.check().is_ok());
        Ok(())
    }
}
