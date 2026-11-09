//! The one release hook for every arena here that frees as its storages die.
//!
//! candle-core takes a single hook ([`install_wave_release`]) and calls it from
//! every wave lease's drop. Two arenas answer it — the guest heap and the asides
//! — so the hook is this dispatcher on the ticket's arena, installed by whichever
//! of them opens first. A bump arena frees nothing per storage and is not here.

use candle::cuda_backend::wave_provenance::{
    install_wave_release, WaveTicket, ASIDE_ARENA, GUEST_ARENA,
};

/// A storage over `ptr`, carved for `ticket`, has been dropped.
fn release(ticket: WaveTicket, ptr: u64) {
    match ticket.arena {
        GUEST_ARENA => super::guest_heap::release_carve(ticket, ptr),
        ASIDE_ARENA => super::aside::release(ticket, ptr),
        _ => {}
    }
}

/// Install the dispatcher. Idempotent.
pub(super) fn install() {
    install_wave_release(release);
}
