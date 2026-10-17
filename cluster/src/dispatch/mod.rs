//! How a request is dispatched to a member (§5.2). Every route a member serves
//! declares it; the member sends its table in `ROUTES`, and the gateway matches
//! each request's path against it with the matcher axum itself uses.
//!
//! | Dispatch | Picks |
//! |---|---|
//! | `Best` | the highest-power member with idle capacity (§5.3) |
//! | `Sticky(keys)` | the member holding the key's claim; with none, `Best`, which takes it (§5.4) |
//! | `Owner(key)` | the member holding the lease for the key (§9.6) |
//! | `Aggregate(merge)` | every ready member, responses combined (§5.5) |
//! | `Any` | round robin |
//! | `Ui` | the member that served the page this request belongs to |
//!
//! A path no route declares is a UI file, dispatched `Ui`.

mod derived;
mod extract;
mod table;

pub use derived::{content_text, conversation_key, passthrough_key, Derived, PASSTHROUGH_MODEL};
pub use extract::{extract, Extracted, BODY_LIMIT};
pub use table::{Matched, Route, RouteTable};

/// Where a key is read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeySource {
    /// A path parameter, by its name in the route's pattern.
    Path(String),
    /// A request header.
    Header(String),
    /// A top-level string field of a JSON request body, read from at most
    /// [`BODY_LIMIT`] bytes.
    Body(String),
    /// A named key function computed from the buffered body.
    Derived(Derived),
}

/// How the responses of an aggregate are combined (§5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merge {
    /// `{ "members": [ { node, host, build, status, body | error } ] }`.
    Envelope,
    /// The first `2xx` body, unwrapped.
    FirstFound,
    /// One WebSocket, every member's lines interleaved and tagged.
    Lines,
    /// One body: every `2xx` answer's arrays merged by each item's `id`, a
    /// member still syncing included — what it wrote cut off is in no other
    /// member's answer yet. Where two members list one item, the one further
    /// along (`turn_count`, then `updated_ms`) is kept.
    Union,
}

/// A route's dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dispatch {
    Best,
    /// The first source that yields a key wins: `x-tokera-conversation` ahead
    /// of a body's `conv_id`, so the zend GUI never has its chat body buffered.
    Sticky(Vec<KeySource>),
    Owner(KeySource),
    Aggregate(Merge),
    Any,
    Ui,
}

impl Dispatch {
    pub fn sticky_path(param: &str) -> Dispatch {
        Dispatch::Sticky(vec![KeySource::Path(param.into())])
    }

    pub fn owner_path(param: &str) -> Dispatch {
        Dispatch::Owner(KeySource::Path(param.into()))
    }

    /// Whether the gateway must buffer the request body to route it.
    pub fn reads_body(&self) -> bool {
        let reads = |k: &KeySource| matches!(k, KeySource::Body(_) | KeySource::Derived(_));
        match self {
            Dispatch::Sticky(keys) => keys.iter().any(reads),
            Dispatch::Owner(key) => reads(key),
            _ => false,
        }
    }
}

/// How a route is mirrored to certify a new build (§3.8). A route declaring
/// nothing is `None`, so a route that writes is never mirrored by omission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Mirror {
    #[default]
    None,
    /// Compare the response's structure, not its values.
    Shape,
    /// Compare it exactly, ignoring the named volatile fields.
    Exact(Vec<String>),
}
