//! npcd's share of the host's shared endpoints (`docs/shared_model_host.md`
//! §3.5) — the thin adapter from npcd's own state to `/v1/status`, `/v1/tools`,
//! `/v1/schema/layers`, `/v1/substrate` and `/v1/me`.

use std::path::PathBuf;
use std::sync::Arc;

use candle_conversation::projection::Conversation;
use futures::future::BoxFuture;
use host::loading::{tokens_per_second, LoadingView};
use host::{CheckProgress, Maintenance, MeShare, Reports, TenantStatus};
use serde_json::{json, Map, Value};
use tokio::task::spawn_blocking;
use web::auth::session::Identity;
use web::auth::Role;

use crate::api::{account, schema_layers, Authored};
use crate::engine::loading::LoadingSnapshot;
use crate::engine::pulse::catalog;
use crate::substrate::SubstrateDir;

/// npcd's state, as the host reads it.
pub struct NpcdReports {
    authored: Arc<Authored>,
    substrate: SubstrateDir,
}

impl NpcdReports {
    pub fn new(authored: Arc<Authored>, data: PathBuf) -> Self {
        Self {
            authored,
            substrate: SubstrateDir::new(&data),
        }
    }
}

/// npcd's load, in the host's shape.
fn loading_view(l: LoadingSnapshot) -> LoadingView {
    LoadingView {
        current: l.current.id(),
        label: l.label,
        progress: l.progress,
        completed: l.completed.iter().map(|s| s.id()).collect(),
        completed_labels: l.completed.iter().map(|s| s.label()).collect(),
        progressed: l.progressed,
        total: l.total,
        unit: l.unit,
        detail: l.detail,
        elapsed_ms: l.elapsed_ms,
        prefill_tokens: l.prefill_tokens,
        prefill_tps: l
            .prefill_tps
            .or_else(|| tokens_per_second(l.prefill_tokens, l.elapsed_ms)),
        waiting_for: l.waiting_for,
    }
}

/// npcd's line: its load while it loads, and its substrate's maintenance once
/// an engine holds the substrate open.
fn status_of(authored: &Authored) -> TenantStatus {
    let runtime = authored.runtime.as_ref();
    let mut status = TenantStatus::of(
        runtime
            .and_then(|rt| rt.progress.snapshot())
            .map(loading_view),
    );
    let minds = runtime.and_then(|rt| rt.minds.read().unwrap().clone());
    status.maintenance = minds.map(|minds| Maintenance::of(minds.substrate().maintenance_status()));
    status
}

/// The substrate's storage: its files on disk, and — once an engine holds it
/// open — its live chunks and dead weight.
fn storage(authored: &Authored, dir: &SubstrateDir) -> Value {
    let mut storage = dir.read();
    let minds = authored
        .runtime
        .as_ref()
        .and_then(|rt| rt.minds.read().unwrap().clone());
    if let Some(minds) = minds {
        let conv = minds.substrate();
        storage.dead_ratio = conv.dead_ratio();
        storage.live_chunks = Some(conv.read().live_chunk_count() as u64);
    }
    json!(storage)
}

/// The catalog npcd's console reads, split into the tools and what is said of
/// the catalog as a whole.
fn split_catalog(described: Value) -> (Vec<Value>, Value) {
    match described {
        Value::Object(mut m) => {
            let tools = match m.remove("tools") {
                Some(Value::Array(t)) => t,
                _ => Vec::new(),
            };
            (tools, Value::Object(m))
        }
        _ => (Vec::new(), Value::Null),
    }
}

impl Reports for NpcdReports {
    /// On the blocking pool: the maintenance read takes the engine's lock.
    fn status(&self) -> BoxFuture<'_, TenantStatus> {
        let authored = Arc::clone(&self.authored);
        Box::pin(async move {
            spawn_blocking(move || status_of(&authored))
                .await
                .expect("the status read panicked")
        })
    }

    fn tools(&self) -> BoxFuture<'_, (Vec<Value>, Value)> {
        Box::pin(async move { split_catalog(catalog(&self.authored)) })
    }

    fn calibrate(&self) -> BoxFuture<'_, Result<Value, (u16, Value)>> {
        Box::pin(async {
            Err((
                409,
                json!({
                    "error": "calibrated_by_examples",
                    "detail": "an npcd act is calibrated by the examples authored for it in the \
                               catalog; there is no pass to run",
                }),
            ))
        })
    }

    fn layers(&self) -> BoxFuture<'_, Vec<Value>> {
        Box::pin(async move {
            schema_layers(&self.authored).unwrap_or_else(|e| {
                tracing::warn!("npcd's projection schema could not be read: {e}");
                Vec::new()
            })
        })
    }

    /// On the blocking pool: it lists the segment files and takes the engine's
    /// lock, neither of which may park a runtime worker.
    fn substrate(&self) -> BoxFuture<'_, Value> {
        let authored = Arc::clone(&self.authored);
        let dir = self.substrate.clone();
        Box::pin(async move {
            spawn_blocking(move || storage(&authored, &dir))
                .await
                .expect("the storage read panicked")
        })
    }

    /// The account, upserted for a signed-in caller. A failed write leaves the
    /// shared fields to the host and says so in npcd's own share, rather than
    /// a signed-in caller appearing with no account and no reason.
    fn me<'a>(&'a self, identity: Option<&'a Identity>, _: Role) -> BoxFuture<'a, MeShare> {
        Box::pin(async move {
            let Some(id) = identity else {
                return MeShare {
                    top: Map::new(),
                    own: Value::Null,
                };
            };
            match account(&self.authored, id).await {
                Ok(Value::Object(top)) => MeShare {
                    top,
                    own: Value::Null,
                },
                Ok(other) => MeShare {
                    top: Map::new(),
                    own: json!({"error": "account_unreadable", "detail": other.to_string()}),
                },
                Err(detail) => MeShare {
                    top: Map::new(),
                    own: json!({"error": "account_write_failed", "detail": detail}),
                },
            }
        })
    }

    fn conversation(&self) -> Option<Conversation> {
        let minds = self
            .authored
            .runtime
            .as_ref()
            .and_then(|rt| rt.minds.read().unwrap().clone())?;
        Some(minds.substrate())
    }

    fn self_check(&self, _: CheckProgress) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async {
            Err("self-check is zend's: npcd has no conversations of a user's to ask about".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use crate::accounts::Accounts;
    use crate::collections::Libraries;
    use crate::engine::loading::LoadStep;
    use crate::images::Images;
    use crate::mind::Mind;
    use crate::npcs::Npcs;
    use crate::projection::Source;
    use crate::registry::Registry;

    /// npcd with no engine, over a directory of the test's own.
    fn reports(base: &std::path::Path) -> NpcdReports {
        let npcs = Npcs::load(base).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &base.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &base.join("images")).unwrap();
        let authored = Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            accounts,
            npcs,
            serde_yaml::from_str("admins: []\n").unwrap(),
            Libraries::load(&Source::resolve(None).unwrap()),
            Mind::new(None),
            images,
            BTreeMap::new(),
        );
        NpcdReports::new(Arc::new(authored), base.to_path_buf())
    }

    /// npcd's load in the host's shape: steps by id, their labels one for one,
    /// and a rate worked out from the tokens when the step reported none.
    #[test]
    fn the_load_maps_to_the_hosts_shape() {
        let view = loading_view(LoadingSnapshot {
            current: LoadStep::Layers,
            label: LoadStep::Layers.label(),
            progress: 0.5,
            completed: vec![LoadStep::Model, LoadStep::Substrate, LoadStep::Tools],
            progressed: 3,
            total: 6,
            unit: "documents".to_string(),
            detail: "world/ammo.md".to_string(),
            elapsed_ms: 2_000,
            prefill_tokens: 8_000,
            prefill_tps: None,
            waiting_for: vec!["zend"],
        });
        assert_eq!(
            serde_json::to_value(&view).unwrap(),
            json!({
                "current": "layers",
                "label": "Ingesting mind layers",
                "progress": 0.5,
                "completed": ["model", "substrate", "tools"],
                "completed_labels": ["Loading model", "Replaying substrate", "Installing tools"],
                "progressed": 3,
                "total": 6,
                "unit": "documents",
                "detail": "world/ammo.md",
                "elapsed_ms": 2_000,
                "prefill_tokens": 8_000,
                "prefill_tps": 4_000.0,
                "waiting_for": ["zend"],
            })
        );
    }

    /// There is no calibration pass to run for npcd, and it says why.
    #[tokio::test]
    async fn calibration_is_refused_with_its_reason() {
        let dir = tempfile::tempdir().unwrap();
        let (code, body) = reports(dir.path()).calibrate().await.unwrap_err();
        assert_eq!(code, 409);
        assert_eq!(body["error"], "calibrated_by_examples");
    }

    /// A signed-out caller has no account and npcd says nothing; a signed-in
    /// one gets the account, created on first sight, at the top level.
    #[tokio::test]
    async fn me_is_the_account_for_a_signed_in_caller_only() {
        let dir = tempfile::tempdir().unwrap();
        let r = reports(dir.path());
        let none = r.me(None, Role::Unauthenticated).await;
        assert!(none.top.is_empty());
        assert_eq!(none.own, Value::Null);

        let id = Identity {
            provider: "google".to_string(),
            sub: "google-1".to_string(),
            email: "a@example.com".to_string(),
            name: "A".to_string(),
            picture: String::new(),
            exp: u64::MAX,
        };
        let share = r.me(Some(&id), Role::User).await;
        assert_eq!(share.own, Value::Null, "{}", share.own);
        assert_eq!(share.top["provider"], "google");
        assert_eq!(share.top["email"], "a@example.com");
        assert_eq!(share.top["display"], "A");
        assert!(share.top["user_id"].as_str().unwrap().starts_with("u_"));
        // The provider's subject never leaves npcd: `user_id` is its handle.
        assert!(!share.top.contains_key("sub"));
    }

    /// The catalog's tools go to the host's list; what it says of itself stays
    /// npcd's.
    #[test]
    fn the_catalog_splits_into_tools_and_the_rest() {
        let (tools, rest) = split_catalog(json!({
            "tools": [{"name": "say"}],
            "uncalibrated": 2,
            "acts_per_turn": 1,
        }));
        assert_eq!(tools, [json!({"name": "say"})]);
        assert_eq!(rest, json!({"uncalibrated": 2, "acts_per_turn": 1}));
        assert_eq!(split_catalog(Value::Null), (Vec::new(), Value::Null));
    }
}
