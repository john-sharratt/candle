use axum::Json;
use serde::Serialize;

/// `GET /v1/models`
///
/// OpenAI-compatible model listing.  Continue queries this on startup to
/// populate the model selector.
pub async fn list() -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: "zen-code".into(),
            object: "model",
            owned_by: "zend".into(),
        }],
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
