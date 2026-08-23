//! The host-timed named span, with its optional NVTX range.
//!
//! Measures WALL time around a region. For device work prefer
//! [`gpu_span`](super::gpu_span), which times the GPU without a host sync —
//! a host timer around an async launch measures the launch, not the work.

use super::mark::{profile_now, ProfileMark};
use super::pipeline::pipeline_record;
#[cfg(feature = "nvtx")]
use candle_kernels::simple::nvtx::{candle_nvtx_range_end, candle_nvtx_range_start};
#[cfg(feature = "nvtx")]
use std::ffi::CString;

/// A named pipeline span: wall-clock timing under `profile`, an NVTX range
/// under `nvtx`, and nothing at all under neither.
///
/// The two features are independent on purpose. `profile` costs an `Instant`
/// plus a `RefCell` borrow per span and produces the summary table; `nvtx`
/// costs a driver call and annotates the trace so
/// `nsys stats --report nvtx_kern_sum` attributes every kernel to the span that
/// launched it. A capture run wants the second without paying for the first.
///
/// # Why this exists rather than [`pipeline_record`] alone
///
/// [`pipeline_record`] receives the span name at the END of the span, which is
/// enough to accumulate a duration but not to open an NVTX range — that needs
/// the name at the start. `Span` captures both ends.
///
/// # Ordering
///
/// The span closes when the guard drops, so nested spans come out correctly
/// without any bookkeeping: Rust drops in reverse declaration order, so an
/// enclosing span declared first closes last. Spans that do NOT nest are also
/// fine — the underlying `nvtxRangeStartEx`/`nvtxRangeEnd` pair is the
/// overlapping-capable NVTX API, not the strictly-stacked push/pop one, so
/// ending out of order is legal.
///
/// # Usage
///
/// ```rust,ignore
/// let s = span("decode:select");
/// // ... work ...
/// s.end();          // records here, exactly like `pipeline_record` did
/// ```
///
/// Use [`Span::end`] rather than letting the guard fall out of scope wherever
/// the span ends before its enclosing block does — most of the hot path records
/// mid-block with more work following.
pub struct Span {
    name: &'static str,
    mark: ProfileMark,
    #[cfg(feature = "nvtx")]
    range: u64,
}

/// The NUL-terminated form of a span name, interned per name.
///
/// Span names come from a closed set of `&'static str` literals — a few dozen
/// across the wave — but a range is opened per layer per wave, so building the
/// `CString` on every open would be an allocation on the hot path for no reason.
/// Interning turns it into a pointer lookup. The cache is thread-local, matching
/// the timing accumulator, so there is no lock.
#[cfg(feature = "nvtx")]
fn nvtx_name(name: &'static str) -> &'static CString {
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static NAMES: RefCell<HashMap<&'static str, &'static CString>> =
            RefCell::new(HashMap::new());
    }
    NAMES.with(|n| {
        *n.borrow_mut().entry(name).or_insert_with(|| {
            // Leaked deliberately: one allocation per distinct span name per
            // thread, for the life of the process, so the pointer handed to NVTX
            // stays valid without a borrow outliving the call.
            let c = CString::new(name).expect("span names are literals without NUL bytes");
            Box::leak(Box::new(c))
        })
    })
}

/// Open a named pipeline span. See [`Span`].
#[inline(always)]
pub fn span(name: &'static str) -> Span {
    Span::new(name)
}

/// Open a span only when `cond` holds, for a phase that is skipped entirely on
/// some waves (decode-only waves have no prefill rows, and vice versa).
///
/// Without this the guard would record a span for the phase that did not run,
/// polluting both the timing table and the NVTX trace with zero-work entries.
/// `drop` the returned value where the phase ends.
#[inline(always)]
pub fn span_if(cond: bool, name: &'static str) -> Option<Span> {
    if cond {
        Some(Span::new(name))
    } else {
        None
    }
}

impl Span {
    /// Open the span: capture the start mark and push the NVTX range.
    #[inline(always)]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            mark: profile_now(),
            #[cfg(feature = "nvtx")]
            range: {
                // Span names are `&'static str` from a closed set, so the
                // NUL-terminated copy is built once per name and reused rather
                // than allocated on every open.
                let c = nvtx_name(name);
                unsafe { candle_nvtx_range_start(c.as_ptr()) }
            },
        }
    }

    /// Close the span at this exact point, rather than at end of scope.
    #[inline(always)]
    pub fn end(self) {
        drop(self);
    }
}

impl Drop for Span {
    // `ProfileMark` is `()` when `profile` is off — that IS the zero-cost
    // design, so passing it on is a unit arg by construction, not an oversight.
    #[allow(clippy::unit_arg)]
    #[inline(always)]
    fn drop(&mut self) {
        // Close the NVTX range before the host-side bookkeeping, so the
        // accumulator's `RefCell` borrow is not attributed to the span.
        #[cfg(feature = "nvtx")]
        unsafe {
            candle_nvtx_range_end(self.range)
        };
        pipeline_record(self.name, self.mark);
    }
}
