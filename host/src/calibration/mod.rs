//! Tool calibration — the load phase that teaches provenance which tool a
//! question is for.
//!
//! A daemon's tools are one collection of its system prompt, and each turn
//! shows the few of them the belief scan scores highest. The scan scores a
//! tool by the turns that carry its tag, so a tool nobody has ever been seen
//! to choose has nothing to be scored by and the collection falls back to
//! catalog order. This phase writes those turns ahead of time, from what the
//! tool's author wrote down: its trajectories (a question, the reasoning, the
//! call) and its questions (the plain ways it is asked for).
//!
//! Every daemon that offers tools runs it, over its own definitions
//! (`docs/tool_calibration.md`):
//!
//! | Module | What it holds |
//! |---|---|
//! | [`exemplar`] | [`Calibrated`] — what a tool definition gives the phase, and the marker a case resumes by |
//! | [`projection`] | the hidden projection every case is written under: the whole catalog, one tool pinned a case |
//! | [`case`] | one unit of work, waiting and in flight |
//! | [`window`] | how many cases are in flight at once, and when the window refills |
//! | [`timing`] | where the phase's wall time goes |
//! | [`phase`] | [`run`] — retire what is stale, resume what is sealed, write the rest |

pub mod case;
pub mod exemplar;
pub mod phase;
pub mod projection;
pub mod timing;
pub mod window;

pub use exemplar::{Calibrated, MARKER_KEY, PROJECTION_MARKER};
pub use phase::{run, Phase, Report};
pub use projection::{Frame, TOOL_SCOPE_TAG, TOOL_SELECTOR};
