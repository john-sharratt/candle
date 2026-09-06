//! The device ground a guest model stands on.
//!
//! # Why a guest is a tenant of the span and not a pool allocation
//!
//! The reservation is the budget. Anything this process allocates outside it
//! competes with the reservation for the same card, and on WDDM the loser is
//! not an error but a demotion to host RAM — measured at 17× on decode, with
//! nothing anywhere reporting it. A guest model's weights are gigabytes, so a
//! guest that allocated through the CUDA pool would be the largest such
//! competitor the engine has ever had, and the thing it demoted would be the
//! engine.
//!
//! So a guest takes [`SpanRegion`]s: ~16 MiB granules of the reservation
//! itself, claimed from the KV side and handed straight back when the drain
//! ends. That is the same mechanism the gallery arena and the recurrent state
//! use, and it is what `claim_span_region`'s own doc comment describes as the
//! home for "a long-lived device buffer of region size".
//!
//! # Why the claims happen between forwards
//!
//! A claim moves `live_end`, and the wave transient tier is placed flush
//! against `live_end` with no gap above it. A region claimed *during* a forward
//! is therefore carved out of ground the standing tier already occupies — the
//! guest's weights and the wave's scratch become the same bytes, and whichever
//! kernel reads them next dies somewhere else entirely. [`SpanClaims`] holds the
//! arena window open for the whole construction, which both makes the claim
//! legal and pays for one tier handback rather than one per region.
//!
//! # What the engine gets back
//!
//! Nothing here restores anything. Dropping the ground returns its regions to
//! the KV side, and the evicted working set comes back the way it always does —
//! a warm turn elevates on the demand that needs it, an expert pages in on the
//! layer that routes to it. That is the whole restoration story, and it is why
//! the drain has no restore step: there is no second mechanism to get wrong.

use candle::Device;
use candle_nn::kv_cache::{span_region_refusal, SpanClaims, SpanRegion};

/// One placed allocation: where it is and how much room it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Device address of the first byte.
    pub ptr: u64,
    pub len: usize,
}

/// A stretch of ground that really is contiguous.
///
/// Regions are ~16 MiB and a model's tensors are not: a Llama's embedding table
/// is 323 MB in one piece, and every projection is several regions. So the
/// allocator's unit cannot be the region — it has to be the **run**: a maximal
/// stretch of claimed regions whose addresses are consecutive, inside which an
/// allocation may span freely because there is no other tenant between them.
///
/// The KV side hands regions back in whatever order its free list holds them,
/// so a claim is a *set* of addresses, not a range. Coalescing them into runs
/// is what turns that set into somewhere a 323 MB tensor can live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub base: u64,
    pub bytes: usize,
}

/// The placement arithmetic, over region base addresses alone.
///
/// Split from [`GuestGround`] because it is the part with the invariant in it
/// and the part that must be testable without a card: every rule below — never
/// cross a run boundary, refuse rather than split, take the runs in address
/// order — is a rule about addresses, and a test that needs a GPU to check an
/// address is a test nobody runs.
#[derive(Clone, Debug)]
pub struct Bump {
    runs: Vec<Run>,
    /// The run placements are being cut from.
    run: usize,
    /// Bytes handed out of that run so far.
    ///
    /// A bump cursor, because a guest's weights are written once at load and
    /// read until the drain ends: there is nothing to free individually, and a
    /// free list would be machinery for an event that never happens.
    offset: usize,
}

/// Coalesce claimed region bases into contiguous runs.
///
/// Sorted first, because the claim order is the free list's order and says
/// nothing about adjacency — two regions claimed one after another may sit at
/// either end of the span. Sorting is what makes "is the next one adjacent?"
/// answerable at all.
pub fn runs_from(mut bases: Vec<u64>, region_bytes: usize) -> Vec<Run> {
    bases.sort_unstable();
    bases.dedup();
    let mut runs: Vec<Run> = Vec::new();
    for base in bases {
        match runs.last_mut() {
            Some(r) if r.base + r.bytes as u64 == base => r.bytes += region_bytes,
            _ => runs.push(Run {
                base,
                bytes: region_bytes,
            }),
        }
    }
    // Widest first. A model's largest tensor is placed first in practice, but
    // ordering the runs means it does not have to be: the biggest ask meets the
    // biggest stretch either way, and a load that would have failed on a
    // fragmented claim succeeds on the same regions.
    runs.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.base.cmp(&b.base)));
    runs
}

impl Bump {
    pub fn new(bases: Vec<u64>, region_bytes: usize) -> Self {
        Self {
            runs: runs_from(bases, region_bytes),
            run: 0,
            offset: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.runs.iter().map(|r| r.bytes).sum()
    }

    /// The largest single allocation this ground can serve.
    ///
    /// Not the capacity: a claim scattered across the span holds plenty of
    /// bytes and no one stretch big enough for a 323 MB embedding. The
    /// difference between the two is exactly what a refusal has to report, or
    /// "out of room" is read as "get a bigger card" when the card was never the
    /// problem.
    pub fn largest_run(&self) -> usize {
        self.runs.iter().map(|r| r.bytes).max().unwrap_or(0)
    }

    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// The largest contiguous stretch still placeable.
    ///
    /// [`Self::largest_run`] answers what the ground could serve when it was
    /// empty; this answers what it can serve *now*, with the cursor where the
    /// weights left it. The difference is what a caller sizing one big
    /// allocation out of the remainder needs — a guest's activation arena, which
    /// must be a single run because a bump cursor walking off the end of one
    /// would hand out an address in another tenant's ground.
    pub fn largest_free_run(&self) -> usize {
        let here = self
            .runs
            .get(self.run)
            .map(|r| r.bytes.saturating_sub(self.offset))
            .unwrap_or(0);
        let later = self
            .runs
            .iter()
            .skip(self.run + 1)
            .map(|r| r.bytes)
            .max()
            .unwrap_or(0);
        here.max(later)
    }

    /// Bytes still placeable.
    ///
    /// The tail of the run in progress plus every run after it. This is what is
    /// *available*, not what is unused: a placement that does not fit the
    /// current run moves on and abandons its remainder, so the two diverge as
    /// soon as an allocation does not divide a run evenly.
    pub fn free(&self) -> usize {
        let whole: usize = self.runs.iter().skip(self.run + 1).map(|r| r.bytes).sum();
        let here = self
            .runs
            .get(self.run)
            .map(|r| r.bytes.saturating_sub(self.offset))
            .unwrap_or(0);
        whole + here
    }

    /// Place `len` bytes, aligned to `align`.
    ///
    /// **An allocation never crosses a run boundary.** Two runs are separated by
    /// ground some other tenant holds — that is what makes them two runs rather
    /// than one — so a buffer written across the gap corrupts it, and the write
    /// succeeds: every address in the span is mapped, so nothing faults and the
    /// damage surfaces later as a wrong number somewhere unrelated.
    ///
    /// A request that does not fit what is left of the current run therefore
    /// moves to the next and abandons the remainder. A request larger than the
    /// *largest* run cannot be served at all and says so, rather than being
    /// quietly split.
    pub fn place(&mut self, len: usize, align: usize) -> Result<Placement, GroundError> {
        if len > self.largest_run() {
            return Err(GroundError::TooLarge {
                len,
                largest_run: self.largest_run(),
                capacity: self.capacity(),
            });
        }
        let align = align.max(1);
        // Walk forward until one fits. Forward only: a bump allocator that went
        // back to fill an earlier gap would hand out an address below its own
        // cursor, and nothing here tracks which of those are still free.
        while let Some(run) = self.runs.get(self.run) {
            let at = self.offset.next_multiple_of(align);
            if at + len <= run.bytes {
                self.offset = at + len;
                return Ok(Placement {
                    ptr: run.base + at as u64,
                    len,
                });
            }
            self.run += 1;
            self.offset = 0;
        }
        Err(GroundError::Exhausted {
            capacity: self.capacity(),
            wanted: len,
        })
    }
}

/// A run of span regions held for one guest, for the length of one drain.
///
/// Regions are individually ~16 MiB and **not** guaranteed adjacent: the KV
/// side hands back whatever is free. A guest that needs one contiguous buffer
/// larger than a region must therefore lay its weights out per region — which
/// is what a GGUF's per-tensor layout already is, and why the quantized loaders
/// take a destination address per tensor rather than one base.
pub struct GuestGround {
    /// Held for the drain. Dropping these is what returns the ground.
    _regions: Vec<SpanRegion>,
    bump: Bump,
}

impl GuestGround {
    /// Claim `bytes` worth of regions, rounded up, or say why not.
    ///
    /// Takes what it can and refuses as a whole: a guest holding two thirds of
    /// its weights is not a guest that runs slowly, it is one that reads
    /// whatever else occupies the third it did not get. On refusal the partial
    /// claim is dropped here, so the KV side has its ground back before the
    /// error reaches the caller.
    pub fn claim(device: &Device, bytes: usize) -> Result<Self, GroundError> {
        let want = bytes.div_ceil(SpanRegion::bytes());
        let claims = SpanClaims::open(device).map_err(|e| GroundError::Window(e.to_string()))?;
        let mut regions = Vec::with_capacity(want);
        for _ in 0..want {
            match claims.claim() {
                Ok(Some(r)) => regions.push(r),
                Ok(None) => {
                    let got = regions.len();
                    // Dropped before the error is built, so the ground is back
                    // on the KV side by the time anyone reads the message.
                    drop(regions);
                    return Err(GroundError::Short {
                        wanted_bytes: (want * SpanRegion::bytes()) as u64,
                        got_bytes: (got * SpanRegion::bytes()) as u64,
                        why: span_region_refusal(device),
                    });
                }
                Err(e) => {
                    drop(regions);
                    return Err(GroundError::Window(e.to_string()));
                }
            }
        }
        let bases = regions.iter().map(|r| r.base()).collect();
        Ok(Self {
            _regions: regions,
            bump: Bump::new(bases, SpanRegion::bytes()),
        })
    }

    pub fn capacity(&self) -> usize {
        self.bump.capacity()
    }

    pub fn regions(&self) -> usize {
        self._regions.len()
    }

    pub fn free(&self) -> usize {
        self.bump.free()
    }

    pub fn place(&mut self, len: usize, align: usize) -> Result<Placement, GroundError> {
        self.bump.place(len, align)
    }

    /// The coalesced stretches this ground occupies.
    ///
    /// For a tenant that wants to say something about its own extent — declaring
    /// it read-only once its weights are written, most usefully, so that a write
    /// landing here from anywhere else names itself at the FFI boundary instead
    /// of surfacing later as attention reading a weight.
    pub fn runs(&self) -> &[Run] {
        self.bump.runs()
    }

    /// The largest contiguous stretch still placeable — see [`Bump::largest_free_run`].
    pub fn largest_free_run(&self) -> usize {
        self.bump.largest_free_run()
    }

    /// Ground over addresses nobody owns, for testing the drain without a card.
    ///
    /// Holds no regions, so dropping it returns nothing — which is exactly
    /// right for a test that never claimed any. The addresses must not be
    /// written: this exists so the *sequence* around a guest is testable, and
    /// the placement arithmetic itself is [`Bump`]'s to prove.
    #[cfg(test)]
    pub fn for_test(bases: Vec<u64>, region_bytes: usize) -> Self {
        Self {
            _regions: Vec::new(),
            bump: Bump::new(bases, region_bytes),
        }
    }
}

impl std::fmt::Debug for GuestGround {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestGround")
            .field("regions", &self._regions.len())
            .field("capacity_mib", &(self.capacity() >> 20))
            .field("free_mib", &(self.free() >> 20))
            .finish()
    }
}

/// Why a guest could not be given ground.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroundError {
    /// The KV side had some regions but not enough.
    ///
    /// `why` distinguishes the two refusals, which have opposite fixes: a
    /// standing transient tier means the ground exists but belongs to a running
    /// wave, while exhaustion means the weight side would not concede.
    Short {
        wanted_bytes: u64,
        got_bytes: u64,
        why: &'static str,
    },
    /// The arena window could not be opened — a forward is in flight.
    Window(String),
    /// One allocation larger than the longest contiguous stretch of ground.
    ///
    /// Carries the capacity as well, because the two say different things and
    /// the fix differs: a `largest_run` far below `capacity` is a *fragmented*
    /// claim — the bytes are there, scattered — while the two being close is
    /// genuinely not enough ground. Reporting only "too large" reads as the
    /// second when it is usually the first.
    TooLarge {
        len: usize,
        largest_run: usize,
        capacity: usize,
    },
    /// The ground is fully placed.
    Exhausted { capacity: usize, wanted: usize },
}

impl std::fmt::Display for GroundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Short {
                wanted_bytes,
                got_bytes,
                why,
            } => write!(
                f,
                "the guest wanted {} MiB of span ground and got {} MiB — {why}",
                wanted_bytes >> 20,
                got_bytes >> 20
            ),
            Self::Window(e) => write!(f, "could not take the arena window: {e}"),
            Self::TooLarge {
                len,
                largest_run,
                capacity,
            } => write!(
                f,
                "a single {} MiB allocation is larger than the longest contiguous stretch of \
                 ground this guest holds ({} MiB), out of {} MiB claimed in total — the KV side \
                 hands regions back in free-list order, so a claim can be big enough and still \
                 too broken up for one tensor",
                len >> 20,
                largest_run >> 20,
                capacity >> 20
            ),
            Self::Exhausted { capacity, wanted } => write!(
                f,
                "the guest's {} MiB of ground is fully placed and {wanted} more bytes were asked \
                 for",
                capacity >> 20
            ),
        }
    }
}

impl std::error::Error for GroundError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three regions in two stretches, handed back in free-list order: two that
    /// happen to be adjacent (0x4000, 0x5000) and one on its own (0x1000). The
    /// claim order is deliberately not the address order, because the KV side
    /// never promised one.
    fn scattered() -> Bump {
        Bump::new(vec![0x4000, 0x1000, 0x5000], 0x1000)
    }

    /// **The whole reason runs exist.** A model's tensors are far bigger than a
    /// region — a Llama's embedding table is 323 MB in one piece — so an
    /// allocator whose unit is the region cannot host any model at all. It
    /// refused that table on the first real load, correctly and uselessly.
    #[test]
    fn adjacent_regions_coalesce_into_one_stretch() {
        let runs = runs_from(vec![0x4000, 0x1000, 0x5000], 0x1000);
        assert_eq!(
            runs,
            vec![
                Run {
                    base: 0x4000,
                    bytes: 0x2000
                },
                Run {
                    base: 0x1000,
                    bytes: 0x1000
                },
            ],
            "adjacent regions were not joined, so nothing larger than one can be placed"
        );
    }

    /// An allocation spanning two *adjacent* regions is served, because there is
    /// no other tenant between them — that is what makes them one run.
    #[test]
    fn an_allocation_may_span_regions_that_are_actually_adjacent() {
        let mut b = scattered();
        let p = b.place(0x1800, 1).unwrap();
        assert_eq!(p.ptr, 0x4000);
        assert_eq!(b.largest_run(), 0x2000);
    }

    /// **The invariant, restated for runs.** Two *runs* are separated by ground
    /// another tenant holds, so a placement that ran off the end of one would
    /// land in it — and the write would succeed, because every address in the
    /// span is mapped. The corruption surfaces later, somewhere unrelated.
    #[test]
    fn a_placement_never_crosses_a_run_boundary() {
        let mut b = scattered();
        // Fill the two-region run to within 8 bytes.
        b.place(0x2000 - 8, 1).unwrap();
        let next = b.place(16, 1).unwrap();
        assert_eq!(
            next.ptr, 0x1000,
            "the placement continued past the end of a run into another tenant's ground"
        );
        // And the eight-byte tail is abandoned, not reused: nothing else is
        // asking for it during a drain.
        assert_eq!(b.place(4, 1).unwrap().ptr, 0x1000 + 16);
    }

    /// Widest run first, so the biggest tensor meets the biggest stretch
    /// whatever order the model asks in.
    #[test]
    fn the_widest_run_is_used_first() {
        let b = Bump::new(vec![0x1000, 0x8000, 0x9000, 0xA000], 0x1000);
        assert_eq!(b.runs()[0].bytes, 0x3000);
        assert_eq!(b.runs()[0].base, 0x8000);
        assert_eq!(b.largest_run(), 0x3000);
    }

    #[test]
    fn placements_pack_within_a_run() {
        let mut b = scattered();
        b.place(16, 1).unwrap();
        assert_eq!(
            b.place(16, 1).unwrap(),
            Placement {
                ptr: 0x4000 + 16,
                len: 16
            }
        );
    }

    #[test]
    fn alignment_is_honoured_within_a_run() {
        let mut b = Bump::new(vec![0x4000], 0x1000);
        b.place(3, 1).unwrap();
        assert_eq!(b.place(8, 256).unwrap().ptr, 0x4000 + 256);
    }

    /// An alignment that pushes the cursor past a run's end moves to the next
    /// run, exactly as an oversized length does — otherwise the aligned address
    /// is outside the run it was cut from.
    #[test]
    fn an_alignment_that_overflows_a_run_moves_to_the_next() {
        let mut b = Bump::new(vec![0x4000, 0x9000], 0x1000);
        b.place(0x1000 - 4, 1).unwrap();
        assert_eq!(b.place(4, 512).unwrap().ptr, 0x9000);
    }

    /// **A refusal distinguishes "not enough" from "too broken up".** The two
    /// have different fixes, and reporting only the size reads as the first when
    /// it is usually the second — which is what sent the first real load looking
    /// for a bigger card.
    #[test]
    fn an_allocation_larger_than_the_widest_run_is_refused_rather_than_split() {
        let mut b = scattered();
        assert_eq!(
            b.place(0x2001, 1),
            Err(GroundError::TooLarge {
                len: 0x2001,
                largest_run: 0x2000,
                capacity: 0x3000,
            }),
            "a fragmented claim was reported as merely too small"
        );
        // And the message says both numbers, so the difference is readable.
        let msg = GroundError::TooLarge {
            len: 512 << 20,
            largest_run: 16 << 20,
            capacity: 4096 << 20,
        }
        .to_string();
        assert!(msg.contains("512 MiB") && msg.contains("16 MiB") && msg.contains("4096 MiB"));
    }

    #[test]
    fn running_out_of_ground_says_so() {
        let mut b = Bump::new(vec![0x4000], 0x1000);
        b.place(0x1000, 1).unwrap();
        assert_eq!(
            b.place(8, 1),
            Err(GroundError::Exhausted {
                capacity: 0x1000,
                wanted: 8
            })
        );
    }

    /// `free` sums what is left; this reports the widest single stretch of it.
    /// The two differ exactly when a claim is fragmented, which is the case an
    /// activation arena has to be sized against — it is one allocation, so the
    /// sum is not what it can have.
    #[test]
    fn the_largest_free_run_is_a_stretch_not_a_sum() {
        let mut b = scattered();
        let whole = b.largest_free_run();
        assert_eq!(whole, b.largest_run(), "nothing placed yet");
        assert!(
            b.free() > whole,
            "a scattered claim holds more in total ({}) than in any one run ({whole})",
            b.free()
        );
        // Cutting into the widest run shortens what remains of it by exactly
        // the placement, so long as that is still the widest.
        b.place(0x100, 1).unwrap();
        assert_eq!(b.largest_free_run(), whole - 0x100);
    }

    /// An exhausted ground offers no stretch at all, rather than the width of a
    /// run whose bytes are all handed out — which is what an arena carved from
    /// the remainder would otherwise be told it could have.
    #[test]
    fn an_exhausted_ground_has_no_free_run() {
        let mut b = scattered();
        let all = b.free();
        // Drain it a run at a time: a single `place` of `all` would be refused,
        // since no one run holds the whole capacity.
        while b.place(0x1000, 1).is_ok() {}
        assert_eq!(b.largest_free_run(), 0, "started with {all}");
    }

    #[test]
    fn free_counts_the_current_runs_tail_and_every_run_after_it() {
        let mut b = scattered();
        assert_eq!(b.free(), 0x3000);
        b.place(0x100, 1).unwrap();
        assert_eq!(b.free(), 0x3000 - 0x100);
        // Still inside the two-region run: 0x100 + 0x1E00 fits in 0x2000, so
        // nothing is abandoned and `free` drops by exactly the placement.
        b.place(0x1E00, 1).unwrap();
        assert_eq!(b.free(), 0x3000 - 0x1F00);
        // This one does not fit the 0x100 left in that run, so it moves to the
        // next — abandoning that tail, so `free` drops by far more than the
        // 0x200 placed.
        b.place(0x200, 1).unwrap();
        assert_eq!(b.free(), 0x1000 - 0x200);
    }

    /// A duplicate base would coalesce into a run twice as long as the ground
    /// really is, and the second half of every allocation in it would land
    /// outside the claim.
    #[test]
    fn a_repeated_base_does_not_invent_ground() {
        assert_eq!(
            runs_from(vec![0x4000, 0x4000], 0x1000),
            vec![Run {
                base: 0x4000,
                bytes: 0x1000
            }]
        );
    }

    #[test]
    fn an_empty_ground_places_nothing() {
        let mut b = Bump::new(Vec::new(), 0x1000);
        assert_eq!(b.free(), 0);
        assert_eq!(b.capacity(), 0);
        assert_eq!(b.largest_run(), 0);
        assert!(b.place(1, 1).is_err());
    }
}
