//! The lift's routes on `http://local` — the first situated namespace lifted
//! onto the effector device (effector design §C.1).
//!
//! A body reaches the lift the way it reaches everything the world supplies:
//! through the device, at an address carrying the shaft's id. The vault has one
//! car serving every level, so the id is the single `command-shaft` (the
//! `SHAFT_ID` constant in `router.rs`); the `:id` segment is accepted and, for
//! this slice, is not validated against it beyond what the handlers already do.
//!
//! Four routes, three verbs:
//!
//! - `GET /lift/:id` — the car's status from where the body stands: whether it
//!   is boardable here, the level it is at, and whether it is moving. One read
//!   lock.
//! - `OPTIONS /lift/:id` — the schema (§5.1). The `use` body's `floor` is an
//!   **enum of the levels the shaft serves**, computed from live world state at
//!   OPTIONS time (`floor_names`, §11) rather than compiled in — so a schema
//!   read a turn before the ride reflects the shaft as it stands now.
//! - `POST /lift/:id/call` and `POST /lift/:id/use` — call the car to this
//!   landing, or ride it to another level. Neither reimplements the lift: each
//!   synthesises the `lift_call` / `lift_use` act the grammar already offers and
//!   runs it through the real dispatch ([`body::perform`]), so the proximity
//!   rules the enact handlers enforce (refuse off a landing, refuse with the car
//!   away) are the route's rules too, mapped to `409` by [`enact_response`].
//!
//! Proximity is therefore not a separate gate here: a call from the wrong
//! standpoint is refused by `lift_call` itself and comes back as the mapped
//! refusal (§C.1, and the consequence loop of §10).

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::{json, Map, Value};

use crate::api::err;
use crate::effector::auth::DeviceCaller;
use crate::effector::enact_route::enact_response;
use crate::effector::router::Local;
use crate::engine::act::Act;
use crate::engine::body;

/// `GET /lift/:id` — the car's status from where the caller's body stands.
///
/// `here` is whether the car is open at this body's landing (so it can be
/// boarded now), `level` the name of the level the car is currently at (or
/// `null` when there is no lift), and `moving` whether it still has a stop to
/// reach. Read under one lock, a pure function of world state.
pub async fn status(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(_id): Path<String>,
) -> Response {
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_lift();
    };
    let status = hosted.read(|w| {
        let landing = w.at_landing(&body);
        let lift = w.lift();
        // Boardable *here* — the car is open at the floor this body is standing
        // on. Not on a landing, or no car open there, reads as `false`.
        let here = matches!((landing, lift), (Some(floor), Some(car)) if car.boardable_at(floor));
        let level = lift
            .and_then(|car| w.floor_name(car.floor()))
            .map(str::to_string);
        let moving = lift.is_some_and(|car| car.is_moving());
        json!({ "here": here, "level": level, "moving": moving })
    });
    Json(status).into_response()
}

/// `OPTIONS /lift/:id` — the resource schema (effector design §5.1).
///
/// The `POST` body's `floor` enum is filled from [`floor_names`](npc_map::world::World::floor_names)
/// under the read lock, so it names exactly the levels the shaft serves at the
/// moment the schema is read.
pub async fn schema(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(id): Path<String>,
) -> Response {
    let Some((hosted, _body)) = local.body_of(&caller) else {
        return no_lift();
    };
    // Live world state, not a compiled list: the enum is handler-computed at
    // OPTIONS time (§11/§15) so a floor added or removed shows up here at once.
    let floors = hosted.read(|w| w.floor_names());
    let schema = json!({
        "id": format!("lift/{id}"),
        "summary": "The lift. Call the car to your landing, then ride it to another level.",
        "methods": {
            "GET": {
                "returns": {
                    "here": "boolean",
                    "level": "string|null",
                    "moving": "boolean"
                }
            },
            "POST": {
                "summary": "Call or ride the lift.",
                "body": {
                    "type": "object",
                    "properties": {
                        "floor": { "type": "string", "enum": floors }
                    },
                    "required": ["floor"]
                }
            }
        }
    });
    Json(schema).into_response()
}

/// `POST /lift/:id/call` — call the car to this landing.
///
/// Synthesises the `lift_call` act and runs it through the real dispatch; the
/// enact handler refuses off a landing or with the car already here, and that
/// refusal is mapped to `409` (effector design §C.1).
pub async fn call(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(_id): Path<String>,
) -> Response {
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_lift();
    };
    let act = Act {
        tool: "lift_call",
        args: Map::new(),
    };
    enact_response(body::perform(&hosted, &body, &act))
}

/// `POST /lift/:id/use` — ride the car to another level.
///
/// Body `{ "floor": "<name>" }`. The floor, when present, is passed straight to
/// the synthesised `lift_use` act; a missing or malformed body leaves it off,
/// and the enact handler refuses with "did not say to which level" — the same
/// answer a character reaching the act any other way receives.
pub async fn ride(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(_id): Path<String>,
    payload: Option<Json<Value>>,
) -> Response {
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_lift();
    };
    let mut args = Map::new();
    if let Some(Json(value)) = payload {
        if let Some(floor) = value.get("floor").and_then(Value::as_str) {
            args.insert("floor".to_string(), Value::String(floor.to_string()));
        }
    }
    let act = Act {
        tool: "lift_use",
        args,
    };
    enact_response(body::perform(&hosted, &body, &act))
}

/// The refusal for a caller whose body cannot be resolved to a hosted world —
/// there is no lift near a body that is nowhere.
fn no_lift() -> Response {
    err(
        StatusCode::NOT_FOUND,
        "no_lift",
        "you have no body in a running world, so there is no lift within reach",
    )
}
