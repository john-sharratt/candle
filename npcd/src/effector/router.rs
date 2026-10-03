//! The `local` router — the effector device's one axum surface.
//!
//! Every character reaches the world at `http://local/...`, and this is the
//! router that answers it, behind the device-auth layer ([`crate::effector::
//! auth`]) rather than the operator roles. It is driven two ways from one build:
//! in-process by the engine's fast path ([`crate::engine::runtime::Runtime::
//! effector_invoke`]) and, mounted externally, over a real socket — the
//! identical service either way (effector design §8.1).
//!
//! # The near-you index, `GET http://local/`
//!
//! An internal read for external callers and the docs, not a model-facing tool
//! (a character surveys with `scan`): it returns *what is reachable from where the caller's body
//! stands*, recomputed each call from world state. It is a **pure function of
//! that state** (effector design §13) — no session, no cursor — so it is
//! `oneshot`-testable on the CPU with no model, which is exactly how it is
//! pinned. The answer is three parts:
//!
//! - the **personal routes** (`history`, `phone`, `self`), reachable wherever
//!   the body stands because they are the character's own, not the room's;
//! - the **lift**, when the body is standing on a landing (`at_landing`);
//! - one entry per **placed instance within reach** (`instances_at`) that a
//!   route serves ([`address_of`]), the machines the body could work from here
//!   — each addressed by its own instance id, so two same-kind stations in one
//!   room are two distinct URLs rather than one. A fixture that affords nothing
//!   (a seat) has no route and is not listed.

use std::sync::{Arc, OnceLock, Weak};

use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::from_fn_with_state;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde_json::{json, Value};

use npc_map::instance::PartInstance;

use std::collections::BTreeSet;

use crate::api::err;
use crate::effector::auth::{device_auth, DeviceCaller};
use crate::effector::here;
use crate::effector::history;
use crate::effector::lift;
use crate::effector::namespace::namespace_of;
use crate::effector::phone;
use crate::effector::plan;
use crate::effector::reshape;
use crate::effector::selfsurface;
use crate::effector::station;
use crate::effector::token::Tokens;
use crate::engine::runtime::Runtime;
use crate::engine::tools;
use crate::world::Hosted;

/// The vault's single lift shaft, named as the effector design names it (§C.1:
/// "one shaft, id `lift/command-shaft`" — one car serves every level).
///
/// A constant rather than a minted instance id because the lift is a portal
/// woven across levels, not a part placed in a node, so it has no
/// `instances_at` id. The vault has exactly one car, so naming the one shaft
/// here is the whole of its identity.
const SHAFT_ID: &str = "command-shaft";

/// What the `local` router needs to answer a call: who can be resolved, and the
/// world state to read a standpoint from.
///
/// The runtime is held [`Weak`] on purpose. The runtime retains this router (for
/// the fast path), so a strong handle here would be a cycle that never drops;
/// the router upgrades it per call, and a call arriving after the runtime is
/// gone is answered as "no world" rather than keeping it alive.
#[derive(Clone)]
pub struct Local {
    tokens: Arc<Tokens>,
    runtime: Weak<Runtime>,
}

impl Local {
    pub fn new(tokens: Arc<Tokens>, runtime: &Arc<Runtime>) -> Self {
        Self {
            tokens,
            runtime: Arc::downgrade(runtime),
        }
    }

    /// Resolve a caller to the world its body stands in and the body id it acts
    /// through — the same resolution [`near_you`] makes (`bodies.bound`, then
    /// `hosted.get`), factored out so the situated routes reach a standpoint the
    /// one way.
    ///
    /// `None` when the runtime has been dropped, the caller has no body, or that
    /// body's world is not hosted — the three states a situated route answers as
    /// "there is no such lift near you".
    pub(crate) fn body_of(&self, caller: &DeviceCaller) -> Option<(Arc<Hosted>, String)> {
        let runtime = self.runtime.upgrade()?;
        let bound = runtime.bodies.bound(caller.npc_id)?;
        let hosted = runtime.hosted.get(&bound.world)?;
        Some((hosted, bound.body))
    }

    /// The runtime, upgraded — `None` once it has been dropped. The personal
    /// projected routes ([`crate::effector::selfsurface`], [`crate::effector::
    /// plan`]) reach the shared cast through it rather than a standpoint, because
    /// a character's own plan and beliefs travel with it wherever it stands (§7.2).
    pub(crate) fn runtime(&self) -> Option<Arc<Runtime>> {
        self.runtime.upgrade()
    }
}

/// Build the `local` router: the near-you index behind the device-auth layer.
///
/// The auth layer carries its own state (the token store), independent of the
/// handler state, so it resolves the bearer token to a [`DeviceCaller`] before
/// any handler runs and inserts it into the request extensions.
pub fn router(local: Local) -> Router {
    let tokens = local.tokens.clone();
    let mut router = Router::new()
        .route("/", get(near_you))
        // The lift, the first situated namespace lifted onto this router: its
        // status and schema are reads, and calling or riding it is a POST run
        // through the real act dispatch (effector design §C.1). All behind the
        // same device-auth layer as the index. It keeps its dedicated module
        // rather than the generic station routes: the shaft is a portal woven
        // across levels, not a placed part with an instance id (§C.1).
        .route("/lift/:id", get(lift::status).options(lift::schema))
        .route("/lift/:id/call", post(lift::call))
        .route("/lift/:id/use", post(lift::ride))
        // The personal routes: the character's own, reachable wherever the body
        // stands (§7.2), so they carry no instance id — the caller resolves
        // straight to its body. The near-you index already emits their urls; these
        // make them real, behind the same device-auth layer as everything else.
        //
        // `/phone` is the messaging surface: the migrated phone acts over the real
        // dispatch. `sign_off` is addressed per-thread, so it has a two-segment
        // route of its own, distinct from the single-segment `:verb`.
        .route("/phone", get(phone::threads).options(phone::schema))
        .route("/phone/:verb", post(phone::invoke))
        .route("/phone/:thread/sign_off", post(phone::sign_off))
        // `/here` is the world-state acts as routes: the catalogue's
        // non-part-bound acts on things and rooms (`claim`, `operate`, `give`,
        // …), reachable wherever the body stands, with their live `Choices` sets
        // as the schema (§7.2, Appendix G). Personal like `/phone` and `/self`,
        // so it carries no instance id — the caller resolves straight to its
        // body and the acts name their targets by the names the world writes.
        .nest("/here", here::routes())
        // `/history` is the read-only window onto the body's witnessed past — a
        // non-destructive peek that never advances the engine's `looked` cursor.
        .route("/history", get(history::history))
        // `/self` is the character's own maintained state as a read view: its
        // plan, orders, beliefs and memory, from the shared cast (§7.2, C.27).
        .nest("/self", selfsurface::routes())
        // `/plan` and `/orders` are the projected store (§9.2, C.11/C.21): their
        // verbs write the substrate agency layer through the cast bridge, not the
        // world dispatch, so a plan the character sets projects into its next
        // turn's prompt. The *same* sub-router serves both — the namespace only
        // picks the verbs a character reaches for. `orders` is handled here
        // rather than as a generic station namespace for exactly this reason: its
        // write target is the agency plane, not `Sim.ledger` (§9.2, D.6 #2),
        // which is why it is excluded from [`station_namespaces`] below.
        .nest("/plan", plan::routes())
        .nest("/orders", plan::routes())
        // `/reshape/:world` is runtime topology mutation — the operator/embedder
        // surface onto `World::reshape` (§8.3, Appendix F). Direct-scope only,
        // and not in the near-you index: a character must never be shown a way to
        // reshape the world it lives in. Its own prefix, distinct from the Makers'
        // `/map` cartography station, which authors world-map lore, not the
        // walkable topology.
        .nest("/reshape", reshape::routes());
    // Every station namespace, mounted at once: one generic sub-router per
    // distinct namespace the catalogue attaches a verb to (effector design §9,
    // Appendix C). Each namespace is its own static prefix, so the nests cannot
    // conflict with each other or with the dedicated `/lift/:id` routes above.
    // The mount namespace rides along as an [`station::StationNamespace`]
    // extension, layered onto this one nest rather than the shared router, so
    // `station`'s handlers can refuse a request that reached a resolved part
    // through the wrong prefix (§9, "each namespace is its own static prefix")
    // instead of silently answering it anyway.
    for ns in station_namespaces() {
        router = router.nest(
            &format!("/{ns}"),
            station::routes().layer(Extension(station::StationNamespace(ns))),
        );
    }
    router
        .route_layer(from_fn_with_state(tokens, device_auth))
        .with_state(local)
}

/// The distinct station namespaces to mount, from the tool catalogue.
///
/// Driven by [`Tool::at`](crate::engine::tools::Tool::at): a namespace is
/// mounted iff some catalogue act attaches to a part that maps to it
/// ([`namespace_of`]). Excluded are the namespaces this router serves another
/// way or not yet:
///
/// - `lift` — its dedicated module keeps `/lift/:id` (§C.1); no station act
///   attaches to a part named `lift` in any case.
/// - `map` — topology-mutating, so its writes are not routed here (the generic
///   routes must not expose them).
/// - `plan`/`orders` — the projected store (§9.2, C.11/C.21): their verbs write
///   the substrate agency layer through [`crate::effector::plan`], not the world
///   dispatch, so they have their own nest and must not also mount here — a
///   generic station route would run them through `body::perform` into
///   `Sim.ledger`, the exact defect §9.2/D.6 #2 replaces. (`order-table`/
///   `muster-board` map to `orders` via [`namespace_of`], so without this
///   exclusion `orders` would double-mount and axum would panic.)
/// - the personal roots `here`/`history`/`phone`/`self` — not part-bound, so no
///   act's `at` ever produces them; excluded defensively so a future mapping
///   cannot silently shadow a personal route with a station nest.
fn station_namespaces() -> Vec<&'static str> {
    const EXCLUDED: &[&str] = &[
        "lift", "map", "plan", "orders", "here", "history", "phone", "self", "reshape",
    ];
    let mut namespaces: BTreeSet<&'static str> = BTreeSet::new();
    for tool in tools::CATALOG.iter() {
        for part_id in tool.at {
            let ns = namespace_of(part_id);
            if !EXCLUDED.contains(&ns) {
                namespaces.insert(ns);
            }
        }
    }
    namespaces.into_iter().collect()
}

/// `GET http://local/` — the near-you index (effector design §6).
async fn near_you(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    let Some(runtime) = local.runtime.upgrade() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_world",
            "the world is not running",
        );
    };

    // The caller's body, resolved from its id — the world it stands in and the
    // body it is (`Runtime::body_id` is `npc-{id}`; the binding stores it). A
    // character with no body has only its personal routes.
    let Some(bound) = runtime.bodies.bound(caller.npc_id) else {
        return Json(index(&[], false)).into_response();
    };
    let Some(hosted) = runtime.hosted.get(&bound.world) else {
        return Json(index(&[], false)).into_response();
    };

    // One acquisition of the world lock: read the standpoint, and every placed
    // instance reachable from it, before releasing. The composition stays a
    // pure read; each instance becomes its own namespaced URL and summary.
    let (at_landing, parts) = hosted.read(|w| {
        let at_landing = w.at_landing(&bound.body).is_some();
        let parts: Vec<(String, String)> = match w.actor(&bound.body).map(|a| a.at.clone()) {
            Some(at) => w
                .map()
                .instances_at(&at)
                .into_iter()
                .filter_map(|inst| Some((address_of(&inst)?, inst.name().to_string())))
                .collect(),
            None => Vec::new(),
        };
        (at_landing, parts)
    });

    Json(index(&parts, at_landing)).into_response()
}

/// Compose the near-you index from a standpoint: personal routes always, the
/// lift when at a landing, then one entry per reachable instance. `parts` is
/// already `(url, summary)` — the caller has resolved each instance to its
/// namespaced address.
fn index(parts: &[(String, String)], at_landing: bool) -> Value {
    // Personal first: they are the character's own and reachable wherever it
    // stands (effector design §7.2), the exception to "reachable from here".
    let mut routes = vec![
        route("http://local/here", "what you can do where you stand"),
        route("http://local/history", "what you have done and seen"),
        route("http://local/phone", "your phone"),
        route("http://local/self", "your own plan, orders and memory"),
    ];
    if at_landing {
        routes.push(route(&format!("http://local/lift/{SHAFT_ID}"), "the lift"));
    }
    for (url, summary) in parts {
        routes.push(route(url, summary));
    }
    json!({ "routes": routes })
}

/// One index entry: the address, and what lives at it in the character's
/// register.
fn route(url: &str, summary: &str) -> Value {
    json!({ "url": url, "summary": summary })
}

/// The namespaces whose routes answer for placed instances: every station
/// namespace the router mounts, plus `orders`, which the plan bridge serves.
fn served_namespaces() -> &'static BTreeSet<&'static str> {
    static SERVED: OnceLock<BTreeSet<&'static str>> = OnceLock::new();
    SERVED.get_or_init(|| {
        let mut served: BTreeSet<&'static str> = station_namespaces().into_iter().collect();
        served.insert("orders");
        served
    })
}

/// Whether a route answers for instances of this part.
///
/// A namespace is mounted only when some act attaches to a part under it, so a
/// part that affords nothing (a seat, a blast door, a wall turret) has no route
/// at all, and `map` is held back from the generic routes. An address under
/// either would `404`, so [`address_of`] gives such an instance none.
pub fn serves(part_id: &str) -> bool {
    served_namespaces().contains(namespace_of(part_id))
}

/// The `http://local/<ns>/<instance-id>` a reachable instance is addressed at,
/// or `None` when no route answers for its part ([`serves`]).
///
/// The namespace comes from the part-id → namespace table ([`namespace_of`],
/// Appendix C), so several parts writing one surface share a prefix; the
/// instance id is the map's own stable id (effector design §7, §9), so two
/// same-kind stations in one room get two distinct URLs. This is the one place
/// an instance's address is composed: the near-you index and the grammar's
/// survey and `invoke`'s enum both read it, so what a character is shown, what it
/// may name and what the router serves are one set.
pub fn address_of(inst: &PartInstance) -> Option<String> {
    serves(inst.part_id()).then(|| {
        format!(
            "http://local/{}/{}",
            namespace_of(inst.part_id()),
            inst.id()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composition is a pure function of a standpoint's facts, so it is
    /// testable without a world at all: the personal routes are always present,
    /// the lift appears only at a landing, and every reachable instance gets an
    /// entry at its namespaced instance-id URL.
    #[test]
    fn the_index_composes_personal_lift_and_parts() {
        let empty = index(&[], false);
        let urls: Vec<&str> = empty["routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["url"].as_str().unwrap())
            .collect();
        assert_eq!(
            urls,
            [
                "http://local/here",
                "http://local/history",
                "http://local/phone",
                "http://local/self"
            ],
            "the personal routes stand alone when nothing is reachable"
        );

        // Two character terminals in one room resolve to two distinct URLs,
        // both under the `character` namespace, each carrying its instance id.
        let parts = vec![
            (
                "http://local/character/character-terminal~0".to_string(),
                "character terminal".to_string(),
            ),
            (
                "http://local/character/character-terminal~1".to_string(),
                "character terminal".to_string(),
            ),
        ];
        let full = index(&parts, true);
        let urls: Vec<&str> = full["routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["url"].as_str().unwrap())
            .collect();
        assert_eq!(
            urls,
            [
                "http://local/here",
                "http://local/history",
                "http://local/phone",
                "http://local/self",
                "http://local/lift/command-shaft",
                "http://local/character/character-terminal~0",
                "http://local/character/character-terminal~1",
            ]
        );
    }

    /// `address_of` is where the namespace table and the instance id meet: the
    /// part's kind picks the surface, the placement picks the id.
    #[test]
    fn a_part_address_is_its_namespace_then_its_instance_id() {
        use npc_map::load::MapSet;
        use npc_map::schema::Where;

        let set = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the vault loads");
        let at = Where::new("vault-casting", "band-one");
        let instances: Vec<_> = set
            .instances_at(&at)
            .into_iter()
            .filter(|i| i.part_id() == "character-terminal")
            .collect();
        assert_eq!(instances.len(), 6, "band one holds six character terminals");
        // The url is the namespace then the instance's short id (`<part>~<n>`),
        // derived from the map so the test does not hard-code an ordinal the map
        // decides.
        assert_eq!(
            address_of(&instances[0]),
            Some(format!("http://local/character/{}", instances[0].id()))
        );
        assert!(
            instances[0].id().starts_with("character-terminal~"),
            "the short id form: {}",
            instances[0].id()
        );
    }

    /// **A part is addressable only where a route answers for it.** A seat, a
    /// blast door and a wall turret afford no act, so no namespace is mounted for
    /// them and an address under one would `404`; the map table's namespace is
    /// deliberately not routed. The command table and the muster board are served
    /// (the first by the station routes, the second by the plan bridge), as is
    /// every part that carries a verb.
    #[test]
    fn a_part_is_served_only_where_a_route_answers_for_it() {
        for part in [
            "seat",
            "blast-door",
            "wall-turret",
            "map-table",
            "gallery-rail",
        ] {
            assert!(!serves(part), "`{part}` has no route behind its address");
        }
        for part in [
            "order-table",
            "muster-board",
            "character-terminal",
            "chronicle-terminal",
            "fabricator",
            "bridge-console",
        ] {
            assert!(serves(part), "`{part}` is answered by a mounted namespace");
        }
    }

    /// An instance of an unserved part has no address at all, so nothing can
    /// offer it; an instance of a served part has one.
    #[test]
    fn an_unserved_instance_has_no_address() {
        use npc_map::load::MapSet;

        let set = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the vault loads");
        let mut seen_seat = false;
        for area in set.areas() {
            for node in &area.nodes {
                let at = npc_map::schema::Where::new(area.id.as_str(), node.id.as_str());
                for inst in set.instances_at(&at) {
                    assert_eq!(
                        address_of(&inst).is_some(),
                        serves(inst.part_id()),
                        "{} is addressed iff its part is served",
                        inst.id()
                    );
                    seen_seat |= inst.part_id() == "seat";
                }
            }
        }
        assert!(
            seen_seat,
            "the vault places seats, so the sweep is not vacuous"
        );
    }
}
