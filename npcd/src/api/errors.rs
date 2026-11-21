//! Turns registry and store failures into responses that say what a caller
//! can act on, and nothing about this host.
//!
//! An author's own mistakes are returned verbatim, because the message is the
//! useful part. A server-side failure is logged and generalised, since its
//! detail can carry the absolute path of a file on this machine. A missing
//! character and a character owned by someone else both answer as not found,
//! so a 403 cannot be used to enumerate the ids of other people's casts.

use super::*;

/// Map a registry error to a response.
///
/// `NotFound` covers both "no such id" and "not yours" on purpose: a 403 would
/// confirm that an id exists, which is enough to enumerate somebody else's cast
/// one guess at a time — the §8.3 leak by another route.
pub(crate) fn npc_err(e: NpcError) -> Response {
    match e {
        NpcError::NotFound => err(StatusCode::NOT_FOUND, "npc_not_found", "no such character"),
        NpcError::Invalid(field) => err(
            StatusCode::BAD_REQUEST,
            "invalid_field",
            &format!("`{field}` is missing or out of range"),
        ),
        NpcError::Persist(detail) => {
            tracing::error!(error = %detail, "npc write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "npc_write_failed",
                &detail,
            )
        }
    }
}

pub(super) fn image_err(e: ImageError) -> Response {
    match e {
        ImageError::NotAnImage => err(StatusCode::BAD_REQUEST, "not_an_image", &e.to_string()),
        ImageError::TooLarge(..) => err(StatusCode::PAYLOAD_TOO_LARGE, "too_large", &e.to_string()),
        ImageError::NotFound => err(StatusCode::NOT_FOUND, "image_not_found", &e.to_string()),
        ImageError::Io(io) => {
            // The path is not in the reply: it is the estate's own shape, and a
            // stranger who can provoke a failed write should not get a map of
            // the disk out of it.
            tracing::error!(error = %io, "image store failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "the image could not be stored",
            )
        }
    }
}

pub fn err(status: StatusCode, code: &str, detail: &str) -> Response {
    (status, Json(json!({ "error": code, "detail": detail }))).into_response()
}

/// A listing failure, as a status and a code the console can branch on.
pub(super) fn catalog_err(e: CatalogError) -> Response {
    match e {
        CatalogError::NotFound => err(StatusCode::NOT_FOUND, "not_found", &e.to_string()),
        CatalogError::OutOfScope => err(StatusCode::FORBIDDEN, "out_of_scope", &e.to_string()),
    }
}

/// A document failure, likewise.
pub(super) fn doc_err(e: DocError) -> Response {
    match e {
        // A path error cannot come from an address that parsed — the names were
        // checked when it did — so this is a bug rather than a bad request, and
        // it is reported as one rather than blamed on the caller.
        DocError::Path(p) => {
            tracing::error!(error = %p, "an address resolved to a path the rules refuse");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "could not resolve that",
            )
        }
        DocError::NotFound => err(StatusCode::NOT_FOUND, "not_found", &e.to_string()),
        DocError::IsADirectory => err(StatusCode::BAD_REQUEST, "not_a_document", &e.to_string()),
        DocError::NotText => err(StatusCode::BAD_REQUEST, "not_text", &e.to_string()),
        DocError::TooLarge(_) => err(StatusCode::PAYLOAD_TOO_LARGE, "too_large", &e.to_string()),
        DocError::Exists => err(StatusCode::CONFLICT, "already_exists", &e.to_string()),
        DocError::CannotPatch => err(StatusCode::CONFLICT, "cannot_patch", &e.to_string()),
        DocError::Unkept(why) => {
            // The reason is deliberately not in the reply: it can carry the
            // absolute location of the mind on disk, which is the estate's
            // internal shape and not the caller's business.
            tracing::error!(error = %why, "mind write failed");
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "could not write that",
            )
        }
    }
}

/// Turn a registry failure into a response, deciding what may leave the machine.
///
/// The author's mistakes are returned verbatim, because the message is the
/// useful part — `id::check` names what is wrong with an id better than any
/// paraphrase. A server-side failure is **logged and generalised**: its detail
/// is the absolute path of a file on this host, and a stranger who can provoke
/// a failed write should not get a map of the filesystem out of it.
pub(super) fn registry_err(e: PutError) -> Response {
    if e.is_callers_fault() {
        return err(StatusCode::BAD_REQUEST, "invalid_arguments", &e.to_string());
    }
    tracing::error!(error = %e, "registry write failed");
    err(
        StatusCode::INTERNAL_SERVER_ERROR,
        "write_failed",
        "the document could not be written; see the daemon log",
    )
}
