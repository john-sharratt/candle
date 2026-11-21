//! The forward-wide span that a forward's launch scratch is carved from.
//!
//! A forward's plan prices its own buffers together with a reserve for the
//! launchers' scratch, and both sit in one span. The scratch scope carves from
//! that span only within the budget the plan priced, because the forward's own
//! buffers share its cursor and a carve the plan did not price would take their
//! room. The span is registered when its generation opens and removed when the
//! generation closes, so a launcher only ever sees a span that is open.

use super::*;

/// An open forward-wide span the launchers' scratch is carved from.
pub(super) struct ForwardScratch {
    pub(super) span: Arc<Mutex<Inner>>,
    /// The bytes of the span its plan priced for launch scratch
    /// ([`WaveDomain::scratch_priced`]) — the most the launchers may carve.
    pub(super) budget: usize,
    /// What they have carved so far, alignment padding included.
    pub(super) carved: usize,
}

/// The forward-wide span of each device whose forward is open — what the
/// launchers' scratch is carved from while it is (`docs/held_ground_elimination.md`
/// §3). Registered when the span's generation opens, removed when it closes.
pub(super) fn forward_scratch() -> &'static Mutex<HashMap<usize, ForwardScratch>> {
    static SPANS: OnceLock<Mutex<HashMap<usize, ForwardScratch>>> = OnceLock::new();
    SPANS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The alignment a launcher's scratch is carved at — what a fresh device
/// allocation guarantees and its vectorised loads assume.
pub(super) const SCRATCH_ALIGN: usize = 256;

/// The scratch scope's carver: `bytes` from the forward-wide span open on device
/// `ordinal`, or `None` when none is open or its scratch budget cannot hold them.
/// Only the span's own lock is taken — never the domain map, which a scope is
/// opened under.
///
/// **Within the budget, never past it.** The span's cursor is shared with the
/// forward's own buffers, so a carve the plan did not price takes their room: the
/// first forwards after a start price no scratch (no scope has asked yet), the
/// launchers carved anyway, and the MTP head's priced matmul output found the span
/// spent — `wave arena exhausted`, and zend failed to load.
pub(super) fn carve_forward_scratch(ordinal: usize, bytes: usize) -> Option<u64> {
    let mut spans = forward_scratch().lock().unwrap_or_else(|e| e.into_inner());
    let open = spans.get_mut(&ordinal)?;
    let mut inner = open.span.lock().unwrap_or_else(|e| e.into_inner());
    if inner.live == 0 {
        return None;
    }
    let start = aligned_start(inner.base, inner.cursor, SCRATCH_ALIGN);
    let consumed = (start + bytes).saturating_sub(inner.cursor);
    if open.carved + consumed > open.budget {
        return None;
    }
    let at = bump_locked(
        &mut inner,
        "forward span (launch scratch)",
        bytes,
        SCRATCH_ALIGN,
    )?;
    open.carved += consumed;
    Some(at)
}

/// A forward's span on device `ordinal` has opened, priced with `budget` bytes of
/// launch scratch: its launchers carve up to that from `span` until
/// [`close_forward_scratch`].
pub(super) fn open_forward_scratch(ordinal: usize, span: Arc<Mutex<Inner>>, budget: usize) {
    install_scope_carve(carve_forward_scratch);
    forward_scratch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(
            ordinal,
            ForwardScratch {
                span,
                budget,
                carved: 0,
            },
        );
    open_scratch_scope(ordinal);
}

/// The forward's span on device `ordinal` is closing: nothing more is carved
/// from it, and every launcher takes a fresh block at its next launch.
pub(super) fn close_forward_scratch(ordinal: usize) {
    forward_scratch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&ordinal);
    close_scratch_scope(ordinal);
}
