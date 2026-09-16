//! Compression for everything this server sends, applied on the way out.
//!
//! A response that already carries a `Content-Encoding` goes out exactly as it
//! came: a static file [`asset::respond`] gzipped, or an upstream that
//! compressed its own reply. The client's `Accept-Encoding` is forwarded to the
//! upstream untouched, so an upstream able to compress does, and nothing here
//! compresses it a second time. Anything else goes out gzipped when the client
//! takes gzip and the type is text underneath — pages, JSON from a daemon's
//! API, event streams.
//!
//! # Streamed, and flushed chunk by chunk
//!
//! This used to be deliberately absent: a compression layer that buffers would
//! hold an SSE stream until it closed, and one of the proxied routes is exactly
//! that. So nothing here buffers. Each chunk the body yields is compressed and
//! sync-flushed before the next is read, which means the client can inflate
//! everything it has been sent at every point — a token streamed over SSE
//! arrives when it did before, and a megabyte of JSON arrives as a tenth of one.
//! A flush costs a few bytes and a little ratio on a stream of many small
//! chunks; neither is noticeable against sending the bytes uncompressed.

use std::io::Write;

use axum::body::{Body, BodyDataStream, Bytes, HttpBody};
use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::{Stream, StreamExt};

use crate::asset;

/// The middleware: compress the answer when the request allows it and the
/// answer is worth it.
pub async fn layer(req: Request, next: Next) -> Response {
    // Decided from the request before it is handed on; the handler owns it after.
    let wants = asset::accepts_gzip(req.headers()) && req.method() != Method::HEAD;
    let res = next.run(req).await;
    if wants && worth_compressing(&res) {
        gzip(res)
    } else {
        res
    }
}

/// Whether gzipping this answer is both allowed and worth doing.
fn worth_compressing(res: &Response) -> bool {
    let status = res.status();
    // No body to compress, a body that must stay as the range it names, or a
    // connection that is about to stop being HTTP.
    if status.is_informational()
        || status == StatusCode::NO_CONTENT
        || status == StatusCode::NOT_MODIFIED
        || status == StatusCode::PARTIAL_CONTENT
    {
        return false;
    }
    let h = res.headers();
    if h.contains_key(header::CONTENT_ENCODING) || h.contains_key(header::CONTENT_RANGE) {
        return false;
    }
    // `no-transform` forbids an intermediary from changing the representation,
    // and compressing it is exactly that.
    let no_transform = h
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.to_ascii_lowercase().contains("no-transform"));
    if no_transform {
        return false;
    }
    let Some(mime) = h.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    if !asset::compressible(mime) {
        return false;
    }
    // A body known to be small gains nothing once the gzip header and trailer
    // are counted. A body of unknown length — a stream — is compressed.
    let declared = h
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    match declared.or(res.body().size_hint().exact()) {
        Some(len) => len >= asset::MIN_COMPRESS as u64,
        None => true,
    }
}

/// `res` with its body gzipped as it streams, and the headers that describe
/// the new representation.
fn gzip(res: Response) -> Response {
    let (mut parts, body) = res.into_parts();
    parts.headers.remove(header::CONTENT_LENGTH);
    parts
        .headers
        .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    vary_on_encoding(&mut parts.headers);
    weaken_etag(&mut parts.headers);
    let body = Body::from_stream(gzip_stream(body.into_data_stream()));
    Response::from_parts(parts, body)
}

/// Compress `data` chunk by chunk, flushing after each so every byte sent can
/// be inflated at once, and close the gzip member when the body ends.
fn gzip_stream(data: BodyDataStream) -> impl Stream<Item = Result<Bytes, axum::Error>> {
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    futures::stream::unfold(Some((data, encoder)), |state| async move {
        let (mut data, mut encoder) = state?;
        match data.next().await {
            Some(Ok(chunk)) => {
                let out = encoder
                    .write_all(&chunk)
                    .and_then(|()| encoder.flush())
                    .map(|()| Bytes::from(std::mem::take(encoder.get_mut())))
                    .map_err(axum::Error::new);
                Some((out, Some((data, encoder))))
            }
            Some(Err(e)) => Some((Err(e), None)),
            None => Some((
                encoder.finish().map(Bytes::from).map_err(axum::Error::new),
                None,
            )),
        }
    })
}

/// Name `Accept-Encoding` in `Vary`, keeping whatever the answer already varied
/// on. A shared cache must key on it, or it hands a gzip body to a client that
/// asked for none.
fn vary_on_encoding(h: &mut HeaderMap) {
    let existing = h
        .get(header::VARY)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let value = match existing {
        Some(v)
            if v.split(',')
                .any(|p| p.trim() == "*" || p.trim().eq_ignore_ascii_case("accept-encoding")) =>
        {
            return
        }
        Some(v) if !v.trim().is_empty() => format!("{v}, accept-encoding"),
        _ => "accept-encoding".to_string(),
    };
    if let Ok(v) = HeaderValue::from_str(&value) {
        h.insert(header::VARY, v);
    }
}

/// A strong tag names exact bytes, and these are no longer the bytes it named.
/// The weak form still matches a conditional request — `If-None-Match` compares
/// weakly — so revalidation keeps working, without claiming the compressed body
/// is byte-for-byte the one the upstream tagged.
fn weaken_etag(h: &mut HeaderMap) {
    let Some(tag) = h.get(header::ETAG).and_then(|v| v.to_str().ok()) else {
        return;
    };
    if tag.starts_with("W/") {
        return;
    }
    if let Ok(v) = HeaderValue::from_str(&format!("W/{tag}")) {
        h.insert(header::ETAG, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::Router;
    use http_body_util::BodyExt;
    use std::io::Read;
    use tower::ServiceExt;

    fn app(res: fn() -> Response) -> Router {
        Router::new()
            .route("/", get(move || async move { res() }))
            .layer(axum::middleware::from_fn(layer))
    }

    async fn fetch(router: Router, accept: Option<&str>) -> (HeaderMap, Vec<u8>) {
        let mut req = Request::builder().uri("/");
        if let Some(a) = accept {
            req = req.header(header::ACCEPT_ENCODING, a);
        }
        let res = router
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let headers = res.headers().clone();
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (headers, body)
    }

    fn gunzip(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_end(&mut out)
            .unwrap();
        out
    }

    fn json_body() -> String {
        format!(
            "{{\"messages\":[{}\"end\"]}}",
            "\"a turn of the conversation\",".repeat(200)
        )
    }

    fn json_reply() -> Response {
        Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ETAG, "\"abc123\"")
            .header(header::VARY, "origin")
            .body(Body::from(json_body()))
            .unwrap()
    }

    /// **The reply this exists for.** A daemon's JSON leaves gzipped, and
    /// inflates to exactly the bytes the daemon sent.
    #[tokio::test]
    async fn a_text_reply_goes_out_gzipped_and_inflates_to_what_was_sent() {
        let (h, body) = fetch(app(json_reply), Some("gzip, deflate, br")).await;
        assert_eq!(h[header::CONTENT_ENCODING], "gzip");
        assert!(h.get(header::CONTENT_LENGTH).is_none());
        assert_eq!(h[header::VARY], "origin, accept-encoding");
        assert_eq!(h[header::ETAG], "W/\"abc123\"");
        assert!(
            body.len() < json_body().len() / 4,
            "{} of {}",
            body.len(),
            json_body().len()
        );
        assert_eq!(gunzip(&body), json_body().into_bytes());
    }

    /// An answer that is already encoded — a gzipped asset, or an upstream that
    /// compressed its own reply — goes out byte for byte as it came.
    #[tokio::test]
    async fn an_encoded_reply_passes_through_untouched() {
        fn pre_gzipped() -> Response {
            let mut enc = GzEncoder::new(Vec::new(), Compression::default());
            enc.write_all(json_body().as_bytes()).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::CONTENT_ENCODING, "gzip")
                .header(header::ETAG, "\"abc123-gz\"")
                .body(Body::from(enc.finish().unwrap()))
                .unwrap()
        }
        let (h, body) = fetch(app(pre_gzipped), Some("gzip")).await;
        let (_, original) = fetch(app(pre_gzipped), None).await;
        assert_eq!(h[header::CONTENT_ENCODING], "gzip");
        assert_eq!(h[header::ETAG], "\"abc123-gz\"");
        assert_eq!(body, original);
        assert_eq!(gunzip(&body), json_body().into_bytes());
    }

    /// A client that does not take gzip, or refuses it, gets the bytes as sent.
    #[tokio::test]
    async fn a_client_that_does_not_take_gzip_gets_the_bytes() {
        for accept in [None, Some("identity"), Some("gzip;q=0")] {
            let (h, body) = fetch(app(json_reply), accept).await;
            assert!(h.get(header::CONTENT_ENCODING).is_none(), "{accept:?}");
            assert_eq!(h[header::ETAG], "\"abc123\"", "{accept:?}");
            assert_eq!(body, json_body().into_bytes(), "{accept:?}");
        }
    }

    /// Pictures arrive compressed already, a tiny reply gains nothing, and
    /// `no-transform` forbids the change: all three go out as they came.
    #[tokio::test]
    async fn pictures_small_replies_and_no_transform_are_left_alone() {
        fn picture() -> Response {
            Response::builder()
                .header(header::CONTENT_TYPE, "image/png")
                .body(Body::from(vec![7u8; 4096]))
                .unwrap()
        }
        fn small() -> Response {
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{\"id\":\"c1\"}"))
                .unwrap()
        }
        fn pinned() -> Response {
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::CACHE_CONTROL, "no-store, no-transform")
                .body(Body::from(json_body()))
                .unwrap()
        }
        let (h, body) = fetch(app(picture), Some("gzip")).await;
        assert!(h.get(header::CONTENT_ENCODING).is_none());
        assert_eq!(body, vec![7u8; 4096]);
        let (h, body) = fetch(app(small), Some("gzip")).await;
        assert!(h.get(header::CONTENT_ENCODING).is_none());
        assert_eq!(body, b"{\"id\":\"c1\"}".to_vec());
        let (h, body) = fetch(app(pinned), Some("gzip")).await;
        assert!(h.get(header::CONTENT_ENCODING).is_none());
        assert_eq!(body, json_body().into_bytes());
    }

    /// **The stream it must not hold back.** Each chunk an event stream yields
    /// is compressed and flushed before the next is read: the first event can
    /// be inflated in full while the stream is still open, and the whole body
    /// closes as one valid gzip member.
    #[tokio::test]
    async fn an_event_stream_is_flushed_chunk_by_chunk() {
        let (tx, rx) = futures::channel::mpsc::unbounded::<Result<Bytes, std::io::Error>>();
        let res = Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(rx))
            .unwrap();
        let mut body = gzip(res).into_body();
        let first = b"event: prefill\ndata: {\"done\":512,\"total\":4096}\n\n";
        let second = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";

        tx.unbounded_send(Ok(Bytes::from_static(first))).unwrap();
        let sent = body.frame().await.unwrap().unwrap().into_data().unwrap();
        let mut inflated = flate2::write::GzDecoder::new(Vec::new());
        inflated.write_all(&sent).unwrap();
        inflated.flush().unwrap();
        assert_eq!(inflated.get_ref().as_slice(), first.as_slice());

        tx.unbounded_send(Ok(Bytes::from_static(second))).unwrap();
        drop(tx);
        let mut all = sent.to_vec();
        while let Some(frame) = body.frame().await {
            all.extend_from_slice(&frame.unwrap().into_data().unwrap());
        }
        assert_eq!(gunzip(&all), [first.as_slice(), second.as_slice()].concat());
    }
}
