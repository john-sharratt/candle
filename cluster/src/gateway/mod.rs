//! A gateway's cluster (§3–§5, §12). The state — who it admits, the members of
//! each app and their claims, where each request goes — has no I/O; [`Hub`]
//! runs member sockets over any transport and answers the `web` crate's
//! proxy and status page.

mod admission;
mod claims;
mod hub;
mod log;
mod registry;
mod status;
mod traffic;

pub use admission::{Admission, Admitted, MAX_SKEW_MS};
pub use claims::Claims;
pub use hub::{Hub, MemberStream, APPS, MEMBERS_EVERY, REFUSALS_KEPT};
pub use log::{CoordLog, Entry};
pub use registry::{MemberView, Pick, Registry};
pub use status::{state_name, AppView, ClaimView, HubStatus, MemberRow, Refusal};
pub use traffic::{Class, Failure, OpenStream, Traffic, TrafficView};
