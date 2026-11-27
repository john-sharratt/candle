//! The cast's backup through its routes: what `GET /v1/npcs/export` takes out
//! of one substrate, `POST /v1/npcs/import` puts back into a fresh one,
//! portraits and all.

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use candle_conversation::persistence::blob::hex;
    use candle_conversation::persistence::record::NpcPayload;
    use serde_json::json;
    use tower::ServiceExt;

    use super::super::route_table::router;
    use super::super::tests_accounts::tests::{signed_in, state, tmp, ADMIN};

    fn npc(id: u64, name: &str, portrait: Option<String>) -> NpcPayload {
        serde_json::from_value(json!({
            "npc_id": id,
            "owner_id": "u_1",
            "revision": 4,
            "created_ms": 1,
            "updated_ms": 2,
            "state": "idle",
            "name": name,
            "world_id": "moon",
            "personality_id": "commander",
            "hidden": false,
            "heartbeat_ms": 1000,
            "salience_gate": 0.5,
            "tags": [],
            "persona_description": "p",
            "persona_origin": "authored",
            "portrait_image_id": portrait,
            "portrait_origin": null,
        }))
        .unwrap()
    }

    async fn text(app: axum::Router, req: Request<Body>) -> (StatusCode, String) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 24).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    fn admin(method: &str, path: &str, body: String) -> Request<Body> {
        let mut b = Request::builder().method(method).uri(path);
        for (k, v) in signed_in(ADMIN) {
            b = b.header(k, v);
        }
        b.body(Body::from(body)).unwrap()
    }

    /// The export past its header line, which names the moment it was taken.
    fn body(export: &str) -> Vec<&str> {
        export.lines().skip(1).collect()
    }

    /// **A portrait survives the round trip.** A portrait is a blob only its
    /// character's record names, so no kept object carries it: exported as
    /// records alone, a restore lost all four of WhiteWorld's. Through the
    /// cast export the restored substrate exports exactly what the first did.
    #[tokio::test]
    async fn the_cast_and_its_portraits_come_back_through_the_routes() {
        let old = state(tmp("cast-old"));
        let sha = old
            .npcs
            .substrate()
            .put_blob(b"a portrait".to_vec())
            .unwrap();
        old.npcs
            .import(vec![
                npc(7, "Zen", Some(format!("{}.png", hex(&sha)))),
                npc(9, "Wick", None),
            ])
            .unwrap();
        let (s, out) = text(
            router(old.clone()),
            admin("GET", "/v1/npcs/export", String::new()),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{out}");
        assert_eq!(body(&out).len(), 3, "two characters and a portrait: {out}");
        assert!(
            out.contains(&format!("{{\"blob\":\"{}\"", hex(&sha))),
            "{out}"
        );

        let new = state(tmp("cast-new"));
        let (s, said) = text(
            router(new.clone()),
            admin("POST", "/v1/npcs/import", out.clone()),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{said}");
        let said: serde_json::Value = serde_json::from_str(&said).unwrap();
        assert_eq!(said, json!({ "imported": 2, "portraits": 1 }));
        assert_eq!(
            new.npcs.substrate().blob(&sha).unwrap().as_deref(),
            Some(&b"a portrait"[..])
        );
        let (_, again) = text(router(new), admin("GET", "/v1/npcs/export", String::new())).await;
        assert_eq!(body(&again), body(&out));
    }
}
