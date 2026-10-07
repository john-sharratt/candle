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
//! * `ExecSlot` (`slot`) — the executables one segment ordinal keeps, one per
//!   wave shape, each wave's recapture folded into them in place.
//! * [`GraphExec`] — an instantiated graph, launched only into a
//!   [`ComputeStream`]. A graph launched into the null stream orders against
//!   everything else on it exactly as the eager launches it replaces did.
//! * `staging` — the rings a recorded upload is staged through.
//!
//! A launch is recorded only if it is issued **on the capture stream**. A
//! kernel whose launcher takes no stream runs on the legacy default stream and
//! is executed immediately instead of being recorded, so only launchers that
//! take their stream from the caller can sit inside a recorded segment.

mod error;
mod exec;
mod hub;
mod record_gate;
#[cfg(test)]
mod session;
mod slot;
mod staging;
mod stream;
#[cfg(test)]
mod tests;

pub use error::GraphError;
pub use exec::GraphExec;
pub(crate) use hub::CaptureHub;
pub use hub::{CaptureStats, Paused, WaveCapture};
pub use record_gate::try_without_recording;
#[cfg(test)]
use session::CaptureSession;
#[cfg(test)]
use stream::CaptureStream;
pub use stream::ComputeStream;
