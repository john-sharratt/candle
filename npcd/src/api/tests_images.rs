//! Tests for upload links, cutouts and the portrait route with no engine.
//!
//! An upload without a valid link is refused before its body is read, and an
//! upload with one is kept as a JPEG and named. A cutout needs an engine, and a
//! file that is not a picture is refused at the boundary before it is queued.
//! The portrait route refuses a stranger and says why when no engine is present.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use axum::body::Body;

    use axum::http::Request;

    use super::super::tests_accounts::tests::{call, call_raw, get, send, sign_in, state, tmp};

    use super::super::tests_characters::tests::author;

    /// **A picture sent through an upload link is kept and named**, with
    /// nobody signed in; without a link, with one altered or expired, it is
    /// refused before it is read, and what is not a picture is named as such.
    #[tokio::test]
    async fn an_upload_through_its_link_is_kept_and_named() {
        let st = state(tmp("upload-route"));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::new(40, 20))
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        let png = buf.into_inner();
        let put = |path: &str, body: Vec<u8>| {
            Request::builder()
                .method("PUT")
                .uri(path)
                .body(Body::from(body))
                .unwrap()
        };

        // Through its link, with nobody signed in: kept as a JPEG and named.
        let link = st.image_links.upload_path(now_ms() / 1000);
        let (s, v) = call(router(st.clone()), put(&link, png.clone())).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let id = v["image"].as_str().unwrap().to_string();
        assert!(id.ends_with(".jpg"), "{id}");
        assert_eq!(
            (v["width"].as_u64(), v["height"].as_u64()),
            (Some(40), Some(20))
        );
        assert_eq!(v["kept_s"], 3600);
        let opened = v["link"].as_str().unwrap();
        let path = &opened[opened.find("/v1/image/link/").unwrap()..];
        let (s, got) = call_raw(router(st.clone()), get(path, None)).await;
        assert_eq!(s, StatusCode::OK, "its link opens it: {opened}");
        assert_eq!(&got[..2], &[0xFF, 0xD8]);
        let (bytes, mime) = st.images.get(&id).unwrap();
        assert_eq!((&bytes[..2], mime), (&[0xFF, 0xD8][..], "image/jpeg"));

        // Without one, with one altered, and with one trimmed: refused.
        let last = if link.ends_with('0') { '1' } else { '0' };
        let forged = format!("{}{last}", &link[..link.len() - 1]);
        for refused in [forged.as_str(), "/v1/image/upload"] {
            let (s, v) = call(router(st.clone()), put(refused, png.clone())).await;
            assert_eq!(
                (s, v["error"].as_str()),
                (StatusCode::FORBIDDEN, Some("link_invalid")),
                "{refused}"
            );
        }
        let expired = st.image_links.upload_path(0);
        let (s, v) = call(router(st.clone()), put(&expired, png.clone())).await;
        assert_eq!(v["error"], "link_expired", "{s}");

        // Not a picture: named as the upload's fault.
        let (s, v) = call(router(st), put(&link, b"not a picture".to_vec())).await;
        assert_eq!(
            (s, v["error"].as_str()),
            (StatusCode::BAD_REQUEST, Some("bad_upload"))
        );
    }

    /// **A cutout is a guest job, so it needs an engine.** It used to be a
    /// flood fill on the host and answered without one; it is a network now,
    /// and on a daemon with no engine the honest answer is the same 503 every
    /// other guest route gives rather than a colour algorithm's guess.
    #[tokio::test]
    async fn a_cutout_needs_an_engine_like_every_other_guest() {
        use base64::Engine as _;
        let st = state(tmp("cutout-route"));

        let mut img = image::RgbImage::from_pixel(32, 32, image::Rgb([250, 250, 250]));
        for y in 8..24 {
            for x in 8..24 {
                img.put_pixel(x, y, image::Rgb([20, 30, 60]));
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());

        let (s, out) = call(
            router(st.clone()),
            send(
                "/v1/image/cutout",
                "POST",
                "google-1",
                json!({ "png_base64": b64 }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{out}");
    }

    /// A malformed upload is still the caller's mistake, and is still answered
    /// **before** anything is queued — a drain evicts the engine's working set,
    /// and finding out inside one that a file was not a picture is the failure
    /// the boundary decode exists to prevent.
    #[tokio::test]
    async fn a_cutout_of_something_that_is_not_a_picture_is_refused_at_the_boundary() {
        let st = state(tmp("cutout-junk"));
        let (s, out) = call(
            router(st),
            send(
                "/v1/image/cutout",
                "POST",
                "google-1",
                json!({ "png_base64": "bm90IGEgcGljdHVyZQ==" }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{out}");
        assert_eq!(out["error"], "bad_reference");
    }

    /// **The portrait button reaches the image guest, and says so when there is
    /// none.**
    ///
    /// This state has no engine at all, which is the honest shape for a router
    /// test — what it pins is that the route exists, is reachable by the
    /// character's owner, refuses a stranger, and comes back with a machine-
    /// readable reason rather than a 404. Before this route the console called
    /// `/v1/image/models`, got a 404, and *inferred* "no image model is
    /// loaded"; a guess that happened to be right is still a guess.
    #[tokio::test]
    async fn a_portrait_can_be_asked_for_and_the_refusal_is_legible() {
        let st = state(tmp("portrait-generate"));
        let a = "google-1";
        sign_in(&st, a).await;
        author(&st).await;

        let (s, npc) = call(
            router(st.clone()),
            send(
                "/v1/npc",
                "POST",
                a,
                json!({
                    "name": "Hess", "world_id": "battle-cities",
                    "personality_id": "commander",
                    "persona_description": "a quartermaster in his fifties, scarred left hand",
                }),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{npc}");
        let id = npc["npc_id"].as_str().expect("created").to_string();

        // The models catalogue is **not** asserted here: it lives in
        // `engine::routes` — where the placeholder it replaced already was —
        // and this `router()` builds `api::routes` alone. Reaching for it from
        // here is what a duplicate registration looks like just before axum
        // panics at startup with "overlapping method route", which is exactly
        // how it was found.

        // The draw is refused, with a reason a console can branch on.
        let (s, v) = call(
            router(st.clone()),
            send(
                &format!("/v1/npc/{id}/portrait/generate"),
                "POST",
                a,
                json!({}),
            ),
        )
        .await;
        assert!(
            s == StatusCode::NOT_IMPLEMENTED || s == StatusCode::SERVICE_UNAVAILABLE,
            "a daemon with no image guest answered {s} to a portrait request: {v}"
        );
        assert!(
            v["error"].is_string(),
            "the refusal carries no machine-readable code: {v}"
        );

        // And it is somebody's own character: a stranger cannot draw over it.
        let (s, _) = call(
            router(st),
            send(
                &format!("/v1/npc/{id}/portrait/generate"),
                "POST",
                "google-2",
                json!({}),
            ),
        )
        .await;
        assert_eq!(
            s,
            StatusCode::NOT_FOUND,
            "a stranger could ask for a portrait of somebody else's character"
        );
    }
}
