use axum::Json;
use serde::Serialize;

use crate::passthrough::PASSTHROUGH_MODEL;

/// `GET /v1/models`
///
/// OpenAI-compatible model listing.  Continue queries this on startup to
/// populate the model selector. `zen-code` runs the daemon's own projection;
/// [`PASSTHROUGH_MODEL`] runs the client's context as-is (see
/// [`crate::passthrough`]).
pub async fn list() -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: ["zen-code", PASSTHROUGH_MODEL]
            .into_iter()
            .map(|id| ModelObject {
                id: id.into(),
                object: "model",
                owned_by: "zend".into(),
            })
            .collect(),
    })
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub owned_by: String,
}
