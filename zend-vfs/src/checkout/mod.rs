//! Running a tool against a conversation's files on a real checkout.
//!
//! A conversation's file changes live as deltas ([`FileChanges`]) over the
//! commit of the branch it works on. A tool that runs on the machine — a
//! build, a test run, a formatter — needs them as files on disk, and what it
//! changes has to come back as deltas. One checkout serves every conversation,
//! one run at a time (the caller holds whatever keeps a second run off it), so
//! a build cache over it stays warm across conversations. A run is:
//!
//! 1. **[`preserve`]** — whatever the checkout held, its owner's own work
//!    included, snapshotted byte for byte under a journal, to be put back
//!    however the run ends;
//! 2. **[`materialize`]** — the checkout put on the conversation's branch with
//!    its changes laid over the branch's commit, touching only files whose
//!    bytes are wrong;
//! 3. the tool runs;
//! 4. **[`capture`]** — what the tool changed, read back as deltas against the
//!    conversation's own state, to append to its changes;
//! 5. the set-aside state put back ([`Preserved::restore`]).
//!
//! Through a run the checkout carries a [`Ledger`]: its deviation from its base
//! commit, each file with the content it holds and the [`FileStamp`] it had
//! when that was recorded. A matching stamp vouches for a file without reading
//! it, which is what lets capture touch only what actually changed.
//!
//! | Module | Concern |
//! |---|---|
//! | [`stamp`] | A file's size and modification time, and when a stamp is too fresh to trust |
//! | [`ledger`] | The checkout's deviation from its base, stamped |
//! | [`target`] | Refusing any path that could leave the checkout or enter its git database |
//! | [`preserve`](mod@preserve) | The checkout's own state set aside for a run, and put back |
//! | [`materialize`](mod@materialize) | Branch plus changes onto the checkout |
//! | `reclaim` | The branch and `HEAD` back where the run left them, before capture |
//! | [`capture`](mod@capture) | The tool's changes back as deltas |
//!
//! [`FileChanges`]: crate::FileChanges

pub mod capture;
mod error;
pub mod ledger;
pub mod materialize;
pub mod preserve;
mod reclaim;
pub mod stamp;
pub(crate) mod target;

pub use capture::capture;
pub use error::CheckoutError;
pub use ledger::{Entry, Ledger};
pub use materialize::{materialize, Materialized};
pub use preserve::{preserve, recover, Preserved};
pub use stamp::FileStamp;
