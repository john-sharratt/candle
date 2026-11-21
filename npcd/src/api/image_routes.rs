//! The routes that store, serve and link the pictures of characters and
//! personalities.
//!
//! An upload is a raw body, and its format is decided from its bytes. A
//! portrait records whether it was uploaded or generated, because that
//! decides whether a later draw may replace it. Stored images are
//! content-addressed and cached hard, but they are served only to a signed-in
//! caller or through a link that carries its own signature and expiry. A
//! link-signed upload is checked before its body is read, and a kept picture
//! is normalised to a bounded JPEG.

use super::*;

/// Upload a portrait for a character.
///
/// The raw image as the body, not a multipart form: there is one file and no
/// other fields, so a boundary-encoded envelope would be ceremony around a byte
/// string. The format is decided from the bytes — see [`crate::images`].
///
/// The console called this "uploaded" and then dropped the file: `create()`
/// posted a name, a world, a personality and a description, and the image
/// existed only as an object URL that went away with the tab. The record has
/// carried `portrait_image_id` the whole time with nothing to put in it.
/// Where the bytes came from, for [`crate::npcs::Npcs::set_portrait`].
///
/// Defaults to `uploaded`, which is what a file chosen from disk is. The create
/// step needs the other value: it draws a portrait through the image guest and
/// shows it *before* there is a character to attach it to, then sends the bytes
/// it already has once the character exists. Those bytes are a generated
/// portrait and have to be recorded as one — filed as `uploaded`, the character
/// could never be redrawn afterwards, because [`crate::portrait::post_generate`]
/// refuses to replace an uploaded portrait without `force`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortraitOrigin {
    #[serde(default)]
    pub(super) origin: Option<String>,
}

pub(super) async fn put_portrait(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Query(q): Query<PortraitOrigin>,
    body: axum::body::Bytes,
) -> Response {
    // Two values, and anything else is refused rather than stored: the string
    // is written onto the record and read back as a rule about what may
    // overwrite it, so a third value would be a portrait nothing can classify.
    let origin =
        match q.origin.as_deref() {
            None | Some("uploaded") => "uploaded",
            Some("generated") => "generated",
            Some(other) => return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "bad_origin",
                    "detail": format!("origin must be `uploaded` or `generated`, not `{other}`"),
                })),
            )
                .into_response(),
        };
    let id = match s.images.put(&body) {
        Ok(id) => id,
        Err(e) => return image_err(e),
    };
    // A portrait can be edited too: placed among its holders like any picture.
    s.hand_over_image(&id);
    // Recorded on the character, so the portrait survives a restart and the
    // console can find it again from the listing.
    //
    // Through `set_portrait`, not `patch`: an image id is minted here from the
    // bytes just stored, and `patch` takes what a person types. A caller must
    // not be able to name one — every id in the store is valid, so there would
    // be nothing to reject.
    write_npc(s, headers, nid, move |n, npc_id, owner, now| {
        n.set_portrait(npc_id, owner, id, origin, now)
    })
    .await
}

/// Serve an uploaded image.
///
/// `User`, like everything else about a character. Not `unauthenticated`: an id
/// is a content hash and unguessable, but "unguessable" is not a permission,
/// and these are pictures of somebody's characters. A link that carries its
/// permission is [`get_linked_image`].
pub(super) async fn get_image(State(s): State<Arc<Authored>>, Path(id): Path<String>) -> Response {
    // Content-addressed, so the bytes at an id never change and this can be
    // cached hard. The one place in this daemon where that is true.
    served(&s, &id, "public, max-age=31536000, immutable")
}

/// `GET /v1/image/link/:iid?exp=…&sig=…` — the image, to whoever holds a link
/// this cluster signed, until it expires. A link missing either part, cut short
/// in passing on, is refused as any other that is not one.
pub(super) async fn get_linked_image(
    State(s): State<Arc<Authored>>,
    Path(id): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let exp = q.get("exp").and_then(|e| e.parse::<u64>().ok());
    let sig = q.get("sig").map(String::as_str);
    let checked = match (exp, sig) {
        (Some(exp), Some(sig)) => s.image_links.check(&id, exp, sig, now_ms() / 1000),
        _ => Err(Refused::Invalid),
    };
    match checked {
        // Private: a shared cache would go on serving it past its expiry.
        Ok(()) => served(&s, &id, "private, max-age=3600"),
        Err(Refused::Expired) => err(
            StatusCode::FORBIDDEN,
            "link_expired",
            "this image link has expired; ask for the picture again",
        ),
        Err(Refused::Invalid) => err(
            StatusCode::FORBIDDEN,
            "link_invalid",
            "this is not a link to this image",
        ),
    }
}

/// The most of an upload's body read: the largest picture a reference takes.
pub(super) const UPLOAD_BODY: usize = refimage::MAX_BYTES;

/// `PUT /v1/image/upload?exp=…&sig=…` — a picture handed in to be edited,
/// through a link `image_upload` signed ([`crate::image_links`]).
///
/// The link is checked **before a byte of the body is read**, so the open
/// route takes nothing from anyone without one. The picture is kept as Zen
/// keeps every upload ([`refimage::normalised`]: any reasonable format, turned
/// upright, within 2048 px, a JPEG) for an hour ([`Kept::Uploaded`]), as a
/// substrate blob that replicates to every member. Answers its id, which
/// `image_edit` takes.
pub(super) async fn put_upload(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    body: Body,
) -> Response {
    let public = origin::of(&headers);
    let exp = q.get("exp").and_then(|e| e.parse::<u64>().ok());
    let sig = q.get("sig").map(String::as_str);
    let checked = match (exp, sig) {
        (Some(exp), Some(sig)) => s.image_links.check_upload(exp, sig, now_ms() / 1000),
        _ => Err(Refused::Invalid),
    };
    match checked {
        Ok(()) => {}
        Err(Refused::Expired) => {
            return err(
                StatusCode::FORBIDDEN,
                "link_expired",
                "this upload link has expired; ask for another with image_upload",
            )
        }
        Err(Refused::Invalid) => {
            return err(
                StatusCode::FORBIDDEN,
                "link_invalid",
                "this is not an upload link",
            )
        }
    }
    let bytes = match to_bytes(body, UPLOAD_BODY).await {
        Ok(b) => b,
        Err(_) => {
            return err(
                StatusCode::PAYLOAD_TOO_LARGE,
                "too_large",
                &format!("a picture is at most {UPLOAD_BODY} bytes"),
            )
        }
    };
    /// Why an upload was not kept: the picture, or the store.
    enum NotKept {
        Picture(String),
        Store(ImageError),
    }
    // Decoding a camera original is a second of work: off the async workers.
    // Handed over where it is stored, so a caller gone meanwhile leaves no
    // stored picture unplaced.
    let kept = spawn_blocking(move || {
        let (jpeg, width, height) = refimage::normalised(&bytes).map_err(NotKept::Picture)?;
        // A Zen picture uploaded again keeps the seed that drew it, which the
        // re-encode drops — unless it was shrunk, when the seed no longer
        // names the noise these pixels came from.
        let jpeg = if refimage::dimensions(&bytes) == Ok((width, height)) {
            image_seed::carry(&bytes, jpeg)
        } else {
            jpeg
        };
        let now = now_ms();
        let id = s
            .images
            .put_kept(&jpeg, Kept::Uploaded, now)
            .map_err(NotKept::Store)?;
        s.hand_over_image(&id);
        let link = format!(
            "{public}{}",
            s.image_links.path(&id, now / 1000, Kept::Uploaded)
        );
        Ok((id, width, height, link))
    })
    .await;
    match kept {
        Ok(Ok((id, width, height, link))) => Json(json!({
            "image": id,
            "width": width,
            "height": height,
            "link": link,
            "kept_s": Kept::Uploaded.lifetime().as_secs(),
        }))
        .into_response(),
        Ok(Err(NotKept::Picture(why))) => err(StatusCode::BAD_REQUEST, "bad_upload", &why),
        Ok(Err(NotKept::Store(e))) => image_err(e),
        Err(e) => {
            tracing::error!(error = %e, "an upload's decode panicked");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed",
                "the picture could not be taken in",
            )
        }
    }
}

/// Image `id` with its type, cached as `cache` says.
pub(super) fn served(s: &Authored, id: &str, cache: &'static str) -> Response {
    match s.images.get(id) {
        Ok((bytes, mime)) => {
            let mut res = bytes.into_response();
            let h = res.headers_mut();
            if let Ok(v) = HeaderValue::from_str(mime) {
                h.insert(header::CONTENT_TYPE, v);
            }
            h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
            res
        }
        Err(e) => image_err(e),
    }
}
