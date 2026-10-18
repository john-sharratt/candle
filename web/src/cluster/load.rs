//! A member's answer, carrying the request's [`Load`] until its body is done
//! (§5.3). A chat streams its answer for as long as it decodes, so the request
//! is work for the member until the last byte, not until the headers: the load
//! rides inside the body and is dropped with it — finished, or abandoned by a
//! client that went away.

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use cluster::gateway::Load;
use futures::StreamExt;

/// `res` with `load` held until its body is finished or dropped. An upgraded
/// connection — a websocket — is a session, not a request, and holds none.
pub fn carry(res: Response, load: Option<Load>) -> Response {
    let Some(load) = load else {
        return res;
    };
    if res.status() == StatusCode::SWITCHING_PROTOCOLS {
        return res;
    }
    let (parts, body) = res.into_parts();
    let body = Body::from_stream(body.into_data_stream().map(move |chunk| {
        let _answering = &load;
        chunk
    }));
    Response::from_parts(parts, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    use http_body_util::BodyExt;

    /// No load leaves the response as it was; an upgrade holds none.
    #[tokio::test]
    async fn a_response_without_load_is_untouched() {
        let res = carry(Response::new(Body::from("ok")), None);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"ok");
    }
}
