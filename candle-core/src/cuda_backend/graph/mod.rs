//! Decode graphs: each forward's launches recorded as a chain of graphs.
//!
//! This is the driver layer of `docs/decode_graphs.md` — the only place the
//! graph machinery calls the CUDA driver, and therefore where all its `unsafe`
//! lives.
//!
//! * [`CaptureHub`] — one per device. While a thread holds a [`WaveCapture`]
//!   its launches go to the hub's created, **non-blocking** capture stream in
//!   `ThreadLocal` mode, so other threads' work neither breaks the capture nor
//!   is recorded into it. The wave is cut into segments wherever the host has
//!   to meet the device; each is launched into the compute stream as it ends.
//! * `ExecLibrary` (`library`) — the executables kept across waves, found by
//!   a segment's topology (its node kinds in order) rather than its ordinal,
//!   so a run of layers folds into its own executables wherever the wave's
//!   cuts put it, whichever kernel variants its width chose. Kept within a fixed
//!   device-memory budget (`budget`), and decided by measurement (`regime`): a
//!   learn fills it, a trial times each shape graphed against eager and bans
//!   what is rarely hit or not faster, and once a trial keeps nothing new the
//!   library locks. Segment device time comes from event pairs (`timing`).
//! * `ExecSlot` (`slot`) — one topology's executables, one per shape in steady
//!   use, each recapture folded into them in place.
//! * [`GraphExec`] — an instantiated graph, launched only into a
//!   [`ComputeStream`]. A graph launched into the null stream orders against
//!   everything else on it exactly as the eager launches it replaces did.
//! * `staging` — the rings a recorded upload is staged through.
//!
//! A launch is recorded only if it is issued **on the capture stream**. A
//! kernel whose launcher takes no stream runs on the legacy default stream and
//! is executed immediately instead of being recorded, so only launchers that
//! take their stream from the caller can sit inside a recorded segment.

mod budget;
pub(crate) mod capture;
mod device_wide;
mod error;
mod exec;
mod hub;
mod library;
mod record_gate;
mod regime;
#[cfg(test)]
mod session;
mod slot;
mod staging;
mod stream;
#[cfg(test)]
mod tests;
mod thread_wave;
mod timing;

pub use budget::{budget_for as graph_budget_for, graph_budget_bytes};
pub use error::GraphError;
pub use exec::GraphExec;
pub(crate) use hub::CaptureHub;
pub use hub::{CaptureStats, Paused, WaveCapture};
pub use record_gate::{try_without_recording, without_recording, Quiet};
pub use regime::Phase as GraphPhase;
#[cfg(test)]
use session::CaptureSession;
#[cfg(test)]
use stream::CaptureStream;
pub use stream::ComputeStream;
pub(crate) use thread_wave::enter as enter_wave;
pub use thread_wave::outside_capture;
