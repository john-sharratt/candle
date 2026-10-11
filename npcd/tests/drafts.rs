//! An admin's writes to the mind, called the way the console calls them
//! (`docs/replicated_substrate.md` §10.3).
//!
//! The mind is a clone of a bare `origin` — the record — so a save is a draft
//! in the admin's own store, seen by them and by no one else, and the
//! console's Publish lands it on the record through the same merge an
//! operation's publish step runs. The router is the daemon's own
//! ([`npcd::api::api`]) over a real [`Authored`] and a real [`Runtime`].

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use npcd::accounts::Accounts;
use npcd::api::{api, Authored};
use npcd::collections::Libraries;
use npcd::engine::runtime::Runtime;
use npcd::images::Images;
use npcd::mind::Mind;
use npcd::npcs::Npcs;
use npcd::projection::Source;
use npcd::registry::Registry;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

/// Two admins, so a draft can be shown to be one admin's alone.
const ADMIN: &str = "google-admin";
const OTHER_ADMIN: &str = "google-second";

const ARDH: &str = "# Ardh, as its author tells it.\nname: Ardh\nsetting: A kingdom.\n";

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn configure(dir: &Path) {
    for (k, v) in [
        ("core.autocrlf", "false"),
        ("user.name", "Mind"),
        ("user.email", "mind@example.com"),
        ("commit.gpgSign", "false"),
    ] {
        git(dir, &["config", k, v]);
    }
}

struct Rig {
    app: Router,
    state: Arc<Authored>,
    base: PathBuf,
    mind: PathBuf,
    origin: PathBuf,
    _dir: TempDir,
}

/// A daemon over a mind holding one world. Under git — a clone of a bare
/// `origin` — when `record` is set, a plain folder otherwise.
fn rig(record: bool) -> Rig {
    let dir = TempDir::new().unwrap();
    let base = dir.path().to_path_buf();
    let mind = base.join("mind");
    let origin = base.join("origin");
    if record {
        let seed = base.join("seed");
        std::fs::create_dir_all(seed.join("worlds")).unwrap();
        git(&seed, &["init", "-q", "--template="]);
        git(&seed, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        configure(&seed);
        std::fs::write(seed.join("worlds/ardh.yaml"), ARDH).unwrap();
        git(&seed, &["add", "-A"]);
        git(&seed, &["commit", "-q", "-m", "seed"]);
        git(&base, &["clone", "-q", "--bare", "seed", "origin"]);
        git(
            &base,
            &["clone", "-q", "-c", "core.autocrlf=false", "origin", "mind"],
        );
        configure(&mind);
    } else {
        std::fs::create_dir_all(mind.join("worlds")).unwrap();
        std::fs::write(mind.join("worlds/ardh.yaml"), ARDH).unwrap();
    }

    let data = base.join("data");
    let npcs = Npcs::load(&data).unwrap();
    let accounts = Accounts::open(npcs.substrate(), &data.join("accounts")).unwrap();
    let (images, _) = Images::open(npcs.substrate(), &data.join("images")).unwrap();
    let rt = Runtime::new(Mind::new(Some(mind.clone())), &data);
    rt.set_substrate(npcs.substrate());
    let state = Authored::new(
        Registry::load("world", mind.join("worlds")).unwrap(),
        Registry::load("personality", mind.join("personalities")).unwrap(),
        accounts,
        npcs,
        serde_yaml::from_str(&format!(
            "admins:\n  - sub: {ADMIN}\n  - sub: {OTHER_ADMIN}\n"
        ))
        .unwrap(),
        Libraries::load(&Source::resolve(None).unwrap()),
        Mind::new(Some(mind.clone())),
        images,
        Default::default(),
    )
    .with_runtime(rt);
    let app = api(state.roles.clone()).into_routes(state.clone()).router;
    Rig {
        app,
        state,
        base,
        mind,
        origin,
        _dir: dir,
    }
}

impl Rig {
    async fn call(
        &self,
        method: &str,
        path: &str,
        sub: Option<&str>,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut b = Request::builder().method(method).uri(path);
        if let Some(sub) = sub {
            b = b
                .header("x-tokera-user", sub)
                .header("x-tokera-provider", "google")
                .header("x-tokera-email", "wren@example.com")
                .header("x-tokera-name", "Wren S");
        }
        let req = match body {
            Some(v) => b
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap(),
            None => b.body(Body::empty()).unwrap(),
        };
        let res = self.app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    /// World `ardh`'s setting, as `sub` sees it.
    async fn setting(&self, sub: Option<&str>) -> Value {
        let (status, world) = self.call("GET", "/v1/world/ardh", sub, None).await;
        assert_eq!(status, StatusCode::OK, "{world}");
        world["setting"].clone()
    }

    async fn save(&self, setting: &str) {
        let (status, body) = self
            .call(
                "PUT",
                "/v1/world/ardh",
                Some(ADMIN),
                Some(json!({ "name": "Ardh", "setting": setting })),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    async fn drafts(&self, sub: &str) -> Value {
        let (status, body) = self.call("GET", "/v1/mind/drafts", Some(sub), None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    /// Another member publishes `ardh` with this setting, straight to `origin`.
    fn published_elsewhere(&self, setting: &str) {
        let other = self.base.join("other");
        git(
            &self.base,
            &[
                "clone",
                "-q",
                "-c",
                "core.autocrlf=false",
                "origin",
                "other",
            ],
        );
        configure(&other);
        std::fs::write(
            other.join("worlds/ardh.yaml"),
            format!("# Ardh, as its author tells it.\nname: Ardh\nsetting: {setting}\n"),
        )
        .unwrap();
        git(&other, &["commit", "-q", "-am", "elsewhere"]);
        git(&other, &["push", "-q", "origin", "main"]);
    }
}

/// **A save is a draft its admin alone sees, until they publish it**: the
/// record, the mind folder and every other caller keep the world as it was;
/// the publish lands it on the record — comments and all — under a message
/// naming who published it, brings the mind folder up, and the world is read
/// again for everyone.
#[tokio::test(flavor = "multi_thread")]
async fn a_save_is_a_draft_until_it_is_published() {
    let r = rig(true);
    r.save("A republic.").await;

    assert_eq!(
        std::fs::read_to_string(r.mind.join("worlds/ardh.yaml")).unwrap(),
        ARDH
    );
    assert_eq!(r.setting(Some(ADMIN)).await, "A republic.");
    assert_eq!(r.setting(Some(OTHER_ADMIN)).await, "A kingdom.");
    assert_eq!(r.setting(None).await, "A kingdom.");
    assert_eq!(
        r.drafts(ADMIN).await,
        json!({ "keeps_drafts": true, "drafts": [{ "path": "worlds/ardh.yaml", "removed": false }] })
    );
    assert_eq!(r.drafts(OTHER_ADMIN).await["drafts"], json!([]));

    let (status, published) = r.call("POST", "/v1/mind/publish", Some(ADMIN), None).await;
    assert_eq!(status, StatusCode::OK, "{published}");
    assert_eq!(published["paths"], json!(["worlds/ardh.yaml"]));
    assert_eq!(
        published["published"],
        git(&r.origin, &["rev-parse", "main"]).trim()
    );
    assert_eq!(
        git(&r.origin, &["show", "main:worlds/ardh.yaml"]),
        "# Ardh, as its author tells it.\nname: Ardh\nsetting: A republic.\n"
    );
    assert!(git(&r.origin, &["log", "-1", "--format=%B", "main"])
        .starts_with("Published from the console by "));
    assert_eq!(
        std::fs::read_to_string(r.mind.join("worlds/ardh.yaml")).unwrap(),
        "# Ardh, as its author tells it.\nname: Ardh\nsetting: A republic.\n",
        "the mind folder follows the record"
    );
    assert_eq!(r.setting(None).await, "A republic.");
    assert_eq!(
        r.state.worlds.read().await.get("ardh").unwrap().body["setting"],
        "A republic."
    );
    assert_eq!(r.drafts(ADMIN).await["drafts"], json!([]));
}

/// **A record that moved under the draft at its lines is a conflict**: the
/// publish is refused naming the document, sync brings the record's text in
/// with the overlap marked, and once the admin saves the settled world the
/// publish lands.
#[tokio::test(flavor = "multi_thread")]
async fn a_conflict_is_synced_settled_and_published() {
    let r = rig(true);
    r.save("A republic.").await;
    r.published_elsewhere("An empire.");

    let (status, refused) = r.call("POST", "/v1/mind/publish", Some(ADMIN), None).await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"], "conflicts");
    assert_eq!(refused["paths"], json!(["worlds/ardh.yaml"]));

    let (status, synced) = r.call("POST", "/v1/mind/sync", Some(ADMIN), None).await;
    assert_eq!(status, StatusCode::OK, "{synced}");
    assert_eq!(
        synced,
        json!({ "synced": "merged", "conflicts": ["worlds/ardh.yaml"] })
    );
    let (status, _) = r.call("POST", "/v1/mind/publish", Some(ADMIN), None).await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "the marks are not settled yet"
    );

    r.save("A republic, once an empire.").await;
    let (status, published) = r.call("POST", "/v1/mind/publish", Some(ADMIN), None).await;
    assert_eq!(status, StatusCode::OK, "{published}");
    assert!(git(&r.origin, &["show", "main:worlds/ardh.yaml"])
        .contains("setting: A republic, once an empire.\n"));
}

/// **Discarding drops every draft**, and the admin sees the record again.
#[tokio::test(flavor = "multi_thread")]
async fn discarding_drops_every_draft() {
    let r = rig(true);
    r.save("A republic.").await;
    let (status, _) = r.call("DELETE", "/v1/mind/drafts", Some(ADMIN), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(r.setting(Some(ADMIN)).await, "A kingdom.");
    assert_eq!(r.drafts(ADMIN).await["drafts"], json!([]));
}

/// **Over a mind folder that is not a git repository, what is written is
/// what stands**: there are no drafts to list or publish, and the save is
/// the file, read at once by everyone.
#[tokio::test(flavor = "multi_thread")]
async fn a_plain_mind_keeps_no_drafts() {
    let r = rig(false);
    r.save("A republic.").await;
    assert_eq!(
        std::fs::read_to_string(r.mind.join("worlds/ardh.yaml")).unwrap(),
        "# Ardh, as its author tells it.\nname: Ardh\nsetting: A republic.\n"
    );
    assert_eq!(r.setting(None).await, "A republic.");
    assert_eq!(
        r.drafts(ADMIN).await,
        json!({ "keeps_drafts": false, "drafts": [] })
    );
    let (status, refused) = r.call("POST", "/v1/mind/publish", Some(ADMIN), None).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(refused["error"], "no_drafts");
}

/// Drafts are an admin's: an ordinary user is refused.
#[tokio::test(flavor = "multi_thread")]
async fn drafts_are_an_admins() {
    let r = rig(true);
    for (method, path) in [
        ("GET", "/v1/mind/drafts"),
        ("POST", "/v1/mind/publish"),
        ("POST", "/v1/mind/sync"),
        ("DELETE", "/v1/mind/drafts"),
    ] {
        let (status, _) = r.call(method, path, Some("google-user"), None).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {path}");
    }
}
