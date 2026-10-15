//! An author's drafts, from the console (`docs/replicated_substrate.md` §10.3):
//! what they hold, publishing them, bringing the record's newer text into
//! them, and discarding them.
//!
//! **Publish runs the same merge an operation's publish step runs**: one
//! commit on the store's base, landed on the record under a lease, its push
//! made through this checkout or an agent holding the mind with push rights
//! (§11.5). A record that moved at other lines is taken in as the drafts land;
//! one that moved at the lines they change is a conflict, which `sync` marks
//! in the drafted documents for the author to settle in the editor before
//! publishing again.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use tokio::task::spawn_blocking;
use web::auth::Role;

use super::overlay::Drafted;
use crate::accounts::user_id_of;
use crate::api::{err, Authored};
use crate::identity::require;
use crate::opstore::merge::publish::Publish;
use crate::opstore::merge::sync::Synced;
use crate::opstore::owner::Owner;
use crate::opstore::MindStores;

/// What the console is told when the mind keeps no drafts to publish.
pub const NO_DRAFTS: &str = "The mind folder is not a git repository, so there are no drafts: \
                             what is written is what stands.";

/// The calling admin's store over the mind, with the account's name and
/// drafts — or the refusal that says there are none to act on.
struct Caller {
    stores: Arc<MindStores>,
    owner: Owner,
    name: String,
    drafted: Drafted,
}

async fn caller(s: &Authored, headers: &HeaderMap) -> Result<Caller, Box<Response>> {
    let id = require(headers, &s.roles, Role::Admin)?
        .into_identity()
        .ok_or_else(|| Box::new(err(StatusCode::FORBIDDEN, "forbidden", "sign in first")))?;
    let author = s.author(&user_id_of(&id));
    let (Some(owner), Some(rt)) = (author.owner().cloned(), s.runtime.as_ref()) else {
        return Err(Box::new(err(StatusCode::CONFLICT, "no_drafts", NO_DRAFTS)));
    };
    let name = s
        .accounts
        .read()
        .await
        .get(&id)
        .and_then(|me| me["unique_name"].as_str().map(str::to_string))
        .unwrap_or_else(|| "an author".to_string());
    Ok(Caller {
        stores: Arc::clone(&rt.ops),
        owner,
        name,
        drafted: author.drafted(),
    })
}

/// Each drafted path, and whether the draft removes it.
fn listing(drafted: &Drafted) -> Vec<Value> {
    drafted
        .iter()
        .map(|(path, text)| json!({ "path": path, "removed": text.is_none() }))
        .collect()
}

/// `GET /v1/mind/drafts` — what the caller has drafted and not published.
pub async fn get_drafts(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    match caller(&s, &headers).await {
        Ok(c) => {
            Json(json!({ "keeps_drafts": true, "drafts": listing(&c.drafted) })).into_response()
        }
        Err(r) if r.status() == StatusCode::CONFLICT => {
            Json(json!({ "keeps_drafts": false, "drafts": [] })).into_response()
        }
        Err(r) => *r,
    }
}

/// The message a published draft's commit carries: who published it, and
/// what it changes.
pub fn commit_message(name: &str, drafted: &Drafted) -> String {
    let lines: Vec<String> = drafted
        .iter()
        .map(|(path, text)| match text {
            Some(_) => format!("- {path}"),
            None => format!("- {path} (removed)"),
        })
        .collect();
    format!(
        "Published from the console by {name}\n\n{}\n",
        lines.join("\n")
    )
}

/// `POST /v1/mind/publish` — land the caller's drafts on the record.
pub async fn post_publish(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let c = match caller(&s, &headers).await {
        Ok(c) => c,
        Err(r) => return *r,
    };
    if c.drafted.is_empty() {
        return Json(json!({ "published": Value::Null, "paths": [] })).into_response();
    }
    let message = commit_message(&c.name, &c.drafted);
    let (stores, owner) = (Arc::clone(&c.stores), c.owner.clone());
    let published = spawn_blocking(move || stores.merge(&owner, |m| m.publish(&message))).await;
    let published = match published {
        Ok(Some(Ok(published))) => published,
        Ok(Some(Err(why))) => return err(StatusCode::CONFLICT, "refused", &why),
        Ok(None) => return err(StatusCode::CONFLICT, "no_drafts", NO_DRAFTS),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "publish_failed",
                &e.to_string(),
            )
        }
    };
    match published {
        Publish::Landed { commit, paths } => {
            tracing::info!(%commit, ?paths, "drafts PUBLISHED from the console by {}", c.name);
            // The push moved the store's record; the log takes it as the
            // member's push, and the mind moves to it — the worlds and
            // personalities read again — when its branch does
            // (`docs/substrate_git.md` §6.2, §8.2).
            Json(json!({ "published": commit, "paths": paths })).into_response()
        }
        Publish::Nothing => {
            // The drafts say what the record already says: nothing is left to
            // publish, so nothing is left drafted.
            c.stores.discard(&c.owner);
            Json(json!({ "published": Value::Null, "paths": [] })).into_response()
        }
        Publish::Conflicts(paths) => conflict(
            "conflicts",
            "The record has moved at the lines these drafts change. Sync brings its text in \
             with each overlap marked; settle every mark and publish again.",
            &paths,
        ),
        Publish::Unsettled(paths) => conflict(
            "unsettled",
            "These drafts still hold a merge's marks. Settle each and publish again.",
            &paths,
        ),
        Publish::Unreachable(why) => err(StatusCode::SERVICE_UNAVAILABLE, "unreachable", &why),
        Publish::Refused(why) => err(StatusCode::CONFLICT, "refused", &why),
    }
}

fn conflict(code: &str, detail: &str, paths: &[String]) -> Response {
    (
        StatusCode::CONFLICT,
        Json(json!({ "error": code, "detail": detail, "paths": paths })),
    )
        .into_response()
}

/// `POST /v1/mind/sync` — bring the record's newer text into the caller's
/// drafts, each overlap marked in the drafted document.
pub async fn post_sync(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let c = match caller(&s, &headers).await {
        Ok(c) => c,
        Err(r) => return *r,
    };
    let (stores, owner) = (Arc::clone(&c.stores), c.owner.clone());
    match spawn_blocking(move || stores.merge(&owner, |m| m.sync())).await {
        Ok(Some(Ok(Synced::UpToDate))) => {
            Json(json!({ "synced": "up_to_date", "conflicts": [] })).into_response()
        }
        Ok(Some(Ok(Synced::Merged { conflicts }))) => {
            Json(json!({ "synced": "merged", "conflicts": conflicts })).into_response()
        }
        Ok(Some(Ok(Synced::Unreachable(why)))) => {
            err(StatusCode::SERVICE_UNAVAILABLE, "unreachable", &why)
        }
        Ok(Some(Ok(Synced::Refused(why)) | Err(why))) => err(StatusCode::CONFLICT, "refused", &why),
        Ok(None) => err(StatusCode::CONFLICT, "no_drafts", NO_DRAFTS),
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sync_failed",
            &e.to_string(),
        ),
    }
}

/// `DELETE /v1/mind/drafts` — drop every draft the caller holds.
pub async fn delete_drafts(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    match caller(&s, &headers).await {
        Ok(c) => {
            c.stores.discard(&c.owner);
            StatusCode::NO_CONTENT.into_response()
        }
        Err(r) => *r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The commit names who published and what it changes.**
    #[test]
    fn the_commit_message_names_the_author_and_the_paths() {
        let drafted: Drafted = [
            (
                "worlds/ardh.yaml".to_string(),
                Some("name: Ardh\n".to_string()),
            ),
            ("worlds/old.yaml".to_string(), None),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            commit_message("wren", &drafted),
            "Published from the console by wren\n\n- worlds/ardh.yaml\n- worlds/old.yaml \
             (removed)\n"
        );
    }
}
