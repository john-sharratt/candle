//! Putting a static file on the wire: an entity tag, a conditional answer, and
//! compression.
//!
//! # A short age, not none, and not for ever
//!
//! The consoles have **no build step**. `app.js` is served under that name for
//! ever, so a long `max-age` is a promise that a *changed* file will not be
//! fetched — a cached page talking to a newer API, and the end of the editing
//! loop the daemons are run in, where `--content` serves the console from disk
//! and a refresh is how a change is seen.
//!
//! This went too far the other way first. `no-store` forbade keeping a copy at
//! all, so every refresh re-downloaded the whole console — two hundred
//! kilobytes over the tunnel. Then `no-cache`, which keeps the copy and asks
//! every time: the bytes went away, and fifteen round trips per load stayed, to
//! be told nothing had changed. With the bodies at zero those round trips were
//! the entire remaining cost.
//!
//! So both halves are bounded instead. A file may be reused for a short age
//! without asking — see [`Cache`] for how long and why the split is where it is
//! — and when the age is up, the asking is still a conditional request answered
//! by **304 and no body**. Two levers: the age decides how often to ask, the
//! entity tag decides what the answer costs.
//!
//! # The tag is the content
//!
//! A hash of the bytes, not a modification time. Two boxes serving the same
//! console agree, a file restored from a copy does not look new, and a file
//! rewritten with identical bytes does not invalidate anything. It costs a hash
//! per request — twenty microseconds on a forty-kilobyte stylesheet, against
//! the forty kilobytes it saves sending.

use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, Response, StatusCode};

use crate::config::Cache;
use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};
use std::io::Write;

/// Below this, a compressed body is not reliably smaller than the original once
/// the gzip header and trailer are counted, and never enough to matter.
const MIN_COMPRESS: usize = 860;

/// How long this file may be reused, by what it is.
///
/// Two tiers — see [`Cache`]. The split is *how much a stale copy matters*, not
/// how often the file changes:
///
/// - Code and documents get the **short** one, because a page running last
///   minute's JavaScript against this minute's API is the failure the whole
///   policy exists to bound, and because `--content` serves them from disk so a
///   refresh can show an edit.
/// - Images and fonts get the **long** one. A stale picture is a slightly old
///   picture.
///
/// Anything else — an unrecognised type, an octet-stream — takes the short tier.
/// Being wrong towards freshness costs a round trip; being wrong the other way
/// costs correctness for as long as the age lasts.
///
/// `public`, so shared caches may hold these too. Every file this serves is the
/// same for every reader: the console's code and its pictures. Nothing
/// per-account goes through here — that is all `/v1`, which is not a file.
fn directive(mime: &str, cache: Cache) -> String {
    let base = mime.split(';').next().unwrap_or(mime).trim();
    let secs = if is_long(base) {
        cache.long_secs
    } else {
        cache.short_secs
    };
    // Zero means "always ask", and `max-age=0` says that in a way every cache
    // understands — including one that would otherwise apply a default of its
    // own to a response with no directive.
    format!("public, max-age={secs}")
}

/// Give a relayed response a cache policy **only if it arrived without one**.
///
/// # A fallback, never an override
///
/// The upstream is the authority on its own responses. It knows which of them
/// are content-addressed and may be kept for a year, which are a listing that
/// must never be reused, and which are files. This gateway knows none of that —
/// it sees bytes and a content type — so anything it decided would be a guess
/// overruling a fact.
///
/// So: if the response already carries `Cache-Control`, it is left exactly as it
/// is. `npcd` sets `immutable` on a portrait and the short tier on its console
/// files, and both survive the trip unchanged. This only fills the silence.
///
/// # What the silence is worth
///
/// A relayed response with nothing to say gets `no-store` unless its type is one
/// only a static asset has. That asymmetry is deliberate and is the opposite of
/// the rule for files served from disk: a file *is* static, so an unrecognised
/// one can safely take the short tier; a proxied response could be anything, and
/// the common thing it actually is, is an API answer. Caching a character
/// listing for a minute because nobody said not to is the kind of bug that
/// presents as the daemon ignoring a save.
pub fn relay_cache(h: &mut HeaderMap, cache: Cache) {
    if h.contains_key(header::CACHE_CONTROL) {
        return;
    }
    let mime = h
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let value = match asset_tier(mime, cache) {
        Some(secs) => format!("public, max-age={secs}"),
        None => "no-store".to_string(),
    };
    if let Ok(v) = HeaderValue::from_str(&value) {
        h.insert(header::CACHE_CONTROL, v);
    }
}

/// The tier for a type that is unmistakably a static asset, or `None` for
/// anything a proxied response might be.
fn asset_tier(mime: &str, cache: Cache) -> Option<u32> {
    let base = mime.split(';').next().unwrap_or(mime).trim();
    if is_long(base) {
        return Some(cache.long_secs);
    }
    match base {
        "text/html" | "text/css" | "text/javascript" | "application/javascript" => {
            Some(cache.short_secs)
        }
        _ => None,
    }
}

/// Types whose staleness is a slightly old picture.
fn is_long(base: &str) -> bool {
    base.starts_with("image/") || base.starts_with("font/") || base == "application/font-woff"
}

/// The answer for one file: 304 when the client already has it, otherwise the
/// bytes, compressed when that is both possible and worth it.
pub fn respond(name: &str, bytes: Vec<u8>, cache: Cache, req: &HeaderMap) -> Response<Body> {
    let mime = mime_guess::from_path(name).first_or_octet_stream();
    let mime = mime.as_ref();
    let cache_control = &directive(mime, cache);

    let gzip = accepts_gzip(req) && compressible(mime) && bytes.len() >= MIN_COMPRESS;
    // The tag identifies the *representation*, so the compressed body carries a
    // different one. A cache holding both must be able to tell them apart, and
    // `Vary` alone does not do that for a client that revalidates.
    let tag = etag(&bytes, gzip);

    if none_match(req, &tag) {
        let mut res = Response::new(Body::empty());
        *res.status_mut() = StatusCode::NOT_MODIFIED;
        headers(res.headers_mut(), None, cache_control, &tag, gzip);
        return res;
    }

    let body = if gzip { gz(&bytes) } else { bytes };
    let mut res = Response::new(Body::from(body));
    *res.status_mut() = StatusCode::OK;
    headers(res.headers_mut(), Some(mime), cache_control, &tag, gzip);
    res
}

/// The headers both answers share. A 304 has to repeat them: a client caching
/// the response uses these, not the ones from the 200 it no longer has.
fn headers(h: &mut HeaderMap, mime: Option<&str>, cache_control: &str, tag: &str, gzip: bool) {
    if let Some(mime) = mime {
        if let Ok(v) = HeaderValue::from_str(mime) {
            h.insert(header::CONTENT_TYPE, v);
        }
    }
    if let Ok(v) = HeaderValue::from_str(cache_control) {
        h.insert(header::CACHE_CONTROL, v);
    }
    if let Ok(v) = HeaderValue::from_str(tag) {
        h.insert(header::ETAG, v);
    }
    // Whether the body was compressed depends on the request, so a shared cache
    // must key on that header or it will hand a gzip body to a client that
    // asked for none.
    h.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    if gzip {
        h.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    }
}

/// A strong entity tag: the content, hashed.
///
/// Sixteen hex characters of SHA-256 — sixty-four bits, which is not a security
/// boundary and does not need to be. The consequence of a collision is one
/// stale file for one client until its next change, and the alternative is a
/// longer header on every response for ever.
fn etag(bytes: &[u8], gzip: bool) -> String {
    let digest = Sha256::digest(bytes);
    let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    if gzip {
        format!("\"{hex}-gz\"")
    } else {
        format!("\"{hex}\"")
    }
}

/// Whether the client says it already has this exact representation.
///
/// `*` matches anything the server has, which for a `GET` means "I hold a copy
/// of something at this URL" — answered as unchanged, since the tag we would
/// send is the one it would be checking against.
fn none_match(req: &HeaderMap, tag: &str) -> bool {
    let Some(value) = req.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    value.split(',').any(|candidate| {
        let candidate = candidate.trim();
        // A weak comparison is the one the spec requires here, and `W/"x"` and
        // `"x"` are the same entity under it.
        candidate == "*" || candidate.trim_start_matches("W/") == tag
    })
}

/// Whether the client will take gzip.
///
/// `gzip;q=0` is a refusal, not an offer — the one part of the grammar that
/// changes the answer rather than merely ordering it.
fn accepts_gzip(req: &HeaderMap) -> bool {
    let Some(value) = req
        .get(header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    value.split(',').any(|part| {
        let mut bits = part.split(';').map(str::trim);
        let coding = bits.next().unwrap_or_default();
        if !coding.eq_ignore_ascii_case("gzip") {
            return false;
        }
        !bits.any(|p| p.replace(' ', "") == "q=0" || p.replace(' ', "").starts_with("q=0."))
    })
}

/// Whether compressing this type is worth doing.
///
/// Text, and the structured formats that are text underneath. Deliberately not
/// images, fonts or archives: those arrive compressed already, and running them
/// through gzip spends time to make them very slightly larger.
fn compressible(mime: &str) -> bool {
    let base = mime.split(';').next().unwrap_or(mime).trim();
    base.starts_with("text/")
        || matches!(
            base,
            "application/javascript"
                | "text/javascript"
                | "application/json"
                | "application/manifest+json"
                | "image/svg+xml"
                | "application/xml"
                | "application/wasm"
        )
}

fn gz(bytes: &[u8]) -> Vec<u8> {
    // Level 6: the default, and the knee of the curve. Level 9 spends roughly
    // twice the time for about one percent fewer bytes on this kind of source.
    let mut enc = GzEncoder::new(Vec::with_capacity(bytes.len() / 3), Compression::default());
    if enc.write_all(bytes).is_err() {
        return bytes.to_vec();
    }
    enc.finish().unwrap_or_else(|_| bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped policy: a minute for code, fifteen for pictures.
    const CACHE: Cache = Cache {
        short_secs: 60,
        long_secs: 900,
    };

    fn req(pairs: &[(header::HeaderName, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(k.clone(), HeaderValue::from_str(v).unwrap());
        }
        h
    }

    fn css(n: usize) -> Vec<u8> {
        ".a { color: red; }\n".repeat(n).into_bytes()
    }

    /// **The refresh this exists for.** The first request sends the file and
    /// names it; the second, carrying that name, sends nothing at all.
    #[test]
    fn a_second_request_holding_the_tag_gets_no_body() {
        let bytes = css(200);
        let first = respond("app.css", bytes.clone(), CACHE, &req(&[]));
        assert_eq!(first.status(), StatusCode::OK);
        let tag = first.headers()[header::ETAG].to_str().unwrap().to_owned();

        let second = respond(
            "app.css",
            bytes,
            CACHE,
            &req(&[(header::IF_NONE_MATCH, &tag)]),
        );
        assert_eq!(second.status(), StatusCode::NOT_MODIFIED);
        // And it repeats what a cache needs, since the client no longer holds
        // the 200 those came on.
        assert_eq!(second.headers()[header::ETAG], tag.as_str());
        assert_eq!(
            second.headers()[header::CACHE_CONTROL],
            "public, max-age=60"
        );
    }

    /// **And the editing loop still works.** The tag is the content, so a file
    /// edited on disk is a different file and comes back whole.
    #[test]
    fn an_edited_file_is_sent_again_rather_than_confirmed() {
        let tag = {
            let r = respond("app.js", css(200), CACHE, &req(&[]));
            r.headers()[header::ETAG].to_str().unwrap().to_owned()
        };
        let mut edited = css(200);
        edited.extend_from_slice(b"/* one more line */\n");
        let after = respond(
            "app.js",
            edited,
            CACHE,
            &req(&[(header::IF_NONE_MATCH, &tag)]),
        );
        assert_eq!(
            after.status(),
            StatusCode::OK,
            "a change was confirmed as unchanged"
        );
    }

    /// The same bytes hash the same, wherever they are served from. Two boxes
    /// running the same console agree, and a file copied back into place does
    /// not look new.
    #[test]
    fn the_tag_is_the_content_and_nothing_else() {
        let a = respond("a.css", css(120), CACHE, &req(&[]));
        let b = respond("b.css", css(120), CACHE, &req(&[]));
        assert_eq!(a.headers()[header::ETAG], b.headers()[header::ETAG]);
        let c = respond("a.css", css(121), CACHE, &req(&[]));
        assert_ne!(a.headers()[header::ETAG], c.headers()[header::ETAG]);
    }

    #[test]
    fn a_client_that_takes_gzip_gets_gzip_and_one_that_does_not_gets_bytes() {
        let plain = respond("app.css", css(200), CACHE, &req(&[]));
        assert!(!plain.headers().contains_key(header::CONTENT_ENCODING));

        let zipped = respond(
            "app.css",
            css(200),
            CACHE,
            &req(&[(header::ACCEPT_ENCODING, "gzip, deflate, br")]),
        );
        assert_eq!(zipped.headers()[header::CONTENT_ENCODING], "gzip");
        // A shared cache must key on the request header, or it will hand this
        // body to the client above.
        assert_eq!(zipped.headers()[header::VARY], "accept-encoding");
        // And the two are different representations, so different tags.
        assert_ne!(
            plain.headers()[header::ETAG],
            zipped.headers()[header::ETAG]
        );
    }

    /// `gzip;q=0` is a refusal. Reading it as an offer sends a body the client
    /// said it cannot read.
    #[test]
    fn a_refusal_of_gzip_is_honoured() {
        for value in ["gzip;q=0", "gzip; q=0", "identity, gzip;q=0.0", "identity"] {
            let r = respond(
                "app.css",
                css(200),
                CACHE,
                &req(&[(header::ACCEPT_ENCODING, value)]),
            );
            assert!(
                !r.headers().contains_key(header::CONTENT_ENCODING),
                "compressed against `{value}`"
            );
        }
    }

    /// Already-compressed formats are left alone: gzip spends time to make a
    /// PNG very slightly larger.
    #[test]
    fn images_and_fonts_are_not_compressed_and_text_is() {
        let accept = req(&[(header::ACCEPT_ENCODING, "gzip")]);
        for name in ["mark.png", "photo.jpg", "font.woff2", "archive.zip"] {
            let r = respond(name, vec![7u8; 4000], CACHE, &accept);
            assert!(
                !r.headers().contains_key(header::CONTENT_ENCODING),
                "{name} was compressed"
            );
        }
        for name in ["app.css", "app.js", "index.html", "icon.svg", "data.json"] {
            let r = respond(name, css(200), CACHE, &accept);
            assert_eq!(
                r.headers()[header::CONTENT_ENCODING],
                "gzip",
                "{name} was not compressed"
            );
        }
    }

    /// A file too small to be worth it is sent as it is — the header and
    /// trailer would be most of the saving.
    #[test]
    fn a_tiny_file_is_not_compressed() {
        let r = respond(
            "small.css",
            b".a{color:red}".to_vec(),
            CACHE,
            &req(&[(header::ACCEPT_ENCODING, "gzip")]),
        );
        assert!(!r.headers().contains_key(header::CONTENT_ENCODING));
    }

    /// The list form, and the weak-comparison form a proxy may rewrite it into.
    #[test]
    fn a_tag_is_matched_inside_a_list_and_through_a_weak_prefix() {
        let bytes = css(200);
        let tag = {
            let r = respond("app.css", bytes.clone(), CACHE, &req(&[]));
            r.headers()[header::ETAG].to_str().unwrap().to_owned()
        };
        for value in [
            format!("\"other\", {tag}"),
            format!("W/{tag}"),
            "*".to_string(),
        ] {
            let r = respond(
                "app.css",
                bytes.clone(),
                CACHE,
                &req(&[(header::IF_NONE_MATCH, &value)]),
            );
            assert_eq!(r.status(), StatusCode::NOT_MODIFIED, "`{value}` missed");
        }
        // And a tag for something else is not a match.
        let r = respond(
            "app.css",
            bytes,
            CACHE,
            &req(&[(header::IF_NONE_MATCH, "\"nope\"")]),
        );
        assert_eq!(r.status(), StatusCode::OK);
    }

    /// **Code gets the short age, pictures get the long one.**
    ///
    /// The split is how much a stale copy matters. A page running last minute's
    /// JavaScript against this minute's API is the failure worth bounding; an
    /// old picture is an old picture.
    #[test]
    fn the_tier_follows_what_the_file_is() {
        let short = |name: &str| {
            let r = respond(name, css(200), CACHE, &req(&[]));
            r.headers()[header::CACHE_CONTROL]
                .to_str()
                .unwrap()
                .to_owned()
        };
        for name in ["app.js", "app.css", "index.html", "data.json"] {
            assert_eq!(short(name), "public, max-age=60", "{name}");
        }
        for name in [
            "mark.png",
            "photo.jpg",
            "icon.svg",
            "brand.webp",
            "text.woff2",
        ] {
            assert_eq!(short(name), "public, max-age=900", "{name}");
        }
    }

    /// An unrecognised type takes the **short** tier.
    ///
    /// Being wrong towards freshness costs a round trip. Being wrong the other
    /// way serves something out of date for as long as the age lasts, and the
    /// files that land here are the ones nobody classified — which is exactly
    /// when a guess should be the cheap mistake.
    #[test]
    fn an_unknown_type_is_treated_as_code_not_as_a_picture() {
        let r = respond("thing.unknown-extension", css(200), CACHE, &req(&[]));
        assert_eq!(r.headers()[header::CACHE_CONTROL], "public, max-age=60");
    }

    /// The age says when to *ask*; the tag still says what the answer costs.
    /// A file past its age is a conditional request and no body, not a
    /// re-download.
    #[test]
    fn an_expired_age_still_revalidates_to_nothing() {
        let bytes = css(200);
        let tag = {
            let r = respond("app.js", bytes.clone(), CACHE, &req(&[]));
            r.headers()[header::ETAG].to_str().unwrap().to_owned()
        };
        let again = respond(
            "app.js",
            bytes,
            CACHE,
            &req(&[(header::IF_NONE_MATCH, &tag)]),
        );
        assert_eq!(again.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(again.headers()[header::CACHE_CONTROL], "public, max-age=60");
    }

    /// **A relayed response that states a policy keeps it, exactly.**
    ///
    /// This is the whole contract of the fallback. The upstream knows things
    /// this gateway cannot: that a portrait is content-addressed and good for a
    /// year, that a listing must never be reused. Overruling it with a guess
    /// from a content type would be worse than doing nothing.
    #[test]
    fn a_relayed_policy_is_never_overwritten() {
        for stated in [
            "public, max-age=31536000, immutable", // npcd's portraits
            "no-store",
            "private, max-age=5",
            "no-cache",
        ] {
            let mut h = HeaderMap::new();
            h.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
            h.insert(
                header::CACHE_CONTROL,
                HeaderValue::from_str(stated).unwrap(),
            );
            relay_cache(&mut h, CACHE);
            assert_eq!(h[header::CACHE_CONTROL], stated, "the gateway overruled it");
        }
    }

    /// **Silence from an upstream is `no-store`, unless the type says asset.**
    ///
    /// The opposite of the rule for a file on disk, and deliberately: a file is
    /// static by definition, while a proxied response is most often an API
    /// answer. Caching a character listing for a minute because nobody said not
    /// to presents as the daemon ignoring a save.
    #[test]
    fn a_relayed_response_with_no_policy_gets_a_safe_one() {
        let relayed = |ct: Option<&str>| {
            let mut h = HeaderMap::new();
            if let Some(ct) = ct {
                h.insert(header::CONTENT_TYPE, HeaderValue::from_str(ct).unwrap());
            }
            relay_cache(&mut h, CACHE);
            h[header::CACHE_CONTROL].to_str().unwrap().to_owned()
        };

        // The things only a static asset is.
        assert_eq!(relayed(Some("text/css")), "public, max-age=60");
        assert_eq!(
            relayed(Some("application/javascript")),
            "public, max-age=60"
        );
        assert_eq!(
            relayed(Some("text/html; charset=utf-8")),
            "public, max-age=60"
        );
        assert_eq!(relayed(Some("image/png")), "public, max-age=900");
        assert_eq!(relayed(Some("font/woff2")), "public, max-age=900");

        // Everything else, including the common case: an API answer.
        assert_eq!(relayed(Some("application/json")), "no-store");
        assert_eq!(relayed(Some("text/event-stream")), "no-store");
        assert_eq!(relayed(Some("application/octet-stream")), "no-store");
        assert_eq!(relayed(None), "no-store");
    }

    /// The compressed body has to actually be the file.
    #[test]
    fn the_compressed_body_decompresses_to_the_original() {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let bytes = css(400);
        let out = gz(&bytes);
        assert!(out.len() < bytes.len() / 2, "barely compressed at all");
        let mut back = Vec::new();
        GzDecoder::new(&out[..]).read_to_end(&mut back).unwrap();
        assert_eq!(back, bytes);
    }
}
