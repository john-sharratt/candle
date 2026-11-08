//! `file_upload_and_commit {repo, message, files: [{path, base64}], branch?}`:
//! bytes the caller holds — a picture for the art library, a page a
//! preview built — landed as one commit (`docs/mcp_file_tools.md` §3.6),
//! through the path every committing tool shares ([`super::landing`]). Any
//! bytes: the repository's gate decides which paths may hold them (§3.8).

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};

use super::landing::{land, refused, target};
use super::{Answer, Ctx};
use crate::file_commit::change::Change;

pub const NAME: &str = "file_upload_and_commit";

/// The most bytes one file may carry, decoded.
pub const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;
/// The most bytes one call may carry, decoded.
pub const MAX_CALL_BYTES: usize = 16 * 1024 * 1024;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Commit files whose bytes you hold — a photo, a picture, a page an app preview \
            built — to one of the user's repositories, as base64, all at once or not at all. Use it \
            for anything that is not text, or a whole file you already have; to change text use \
            `file_edit_and_commit`, and for a file at a link use `file_download_and_commit` so its \
            bytes never pass through you. Each file is created or replaced. At most 8 MiB a file \
            and 16 MiB a call. The answer says `live` with the commit, or `refused` with why; \
            a repository that checks changes (Yen Bots) refuses binary files outside the paths \
            its rules allow.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "The repository, by name: `yenbot`, `candle`, …"},
                "message": {"type": "string", "description": "The commit message: what changed and why, in a line."},
                "branch": {"type": "string", "description": "The branch to commit on; the repository's main branch by default."},
                "files": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The file, relative to the repository's root."},
                            "base64": {"type": "string", "description": "The file's whole bytes, base64 (standard alphabet, padded)."}
                        },
                        "required": ["path", "base64"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["repo", "message", "files"],
            "additionalProperties": false
        }
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    path: String,
    base64: String,
}

/// The changes `files` make, or why one cannot be made.
fn changes(files: &[File]) -> Result<Vec<Change>, String> {
    let mut total = 0usize;
    files
        .iter()
        .map(|f| {
            let bytes = STANDARD
                .decode(f.base64.trim())
                .map_err(|e| format!("{}: not base64: {e}", f.path))?;
            if bytes.len() > MAX_FILE_BYTES {
                return Err(format!(
                    "{} is {} bytes; a file is at most {MAX_FILE_BYTES}",
                    f.path,
                    bytes.len()
                ));
            }
            total += bytes.len();
            if total > MAX_CALL_BYTES {
                return Err(format!(
                    "the files come to more than {MAX_CALL_BYTES} bytes"
                ));
            }
            Ok(Change::Bytes {
                path: f.path.clone(),
                bytes,
            })
        })
        .collect()
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (repo, _) = match target(ctx, args) {
        Ok(t) => t,
        Err(a) => return a,
    };
    let files: Vec<File> = match args.get("files").cloned().map(serde_json::from_value) {
        Some(Ok(files)) => files,
        Some(Err(e)) => return Answer::error(format!("`files` is not a list of files: {e}")),
        None => return Answer::error("`files` is missing"),
    };
    match changes(&files) {
        Ok(changes) => land(ctx, args, &changes),
        Err(why) => refused(&repo, &why),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, base64: &str) -> File {
        File {
            path: path.into(),
            base64: base64.into(),
        }
    }

    /// **Each file's bytes are its base64's, exactly**; bad base64, or a call
    /// past its bounds, is refused naming why.
    #[test]
    fn the_files_become_their_bytes() {
        assert_eq!(
            changes(&[file("library/a.png", "iVBORw0KGgo=")]).unwrap(),
            vec![Change::Bytes {
                path: "library/a.png".into(),
                bytes: b"\x89PNG\r\n\x1a\n".to_vec()
            }]
        );
        assert_eq!(
            changes(&[file("a.png", "not base64!")]).unwrap_err(),
            "a.png: not base64: Invalid symbol 32, offset 3."
        );
        let big = STANDARD.encode(vec![0u8; MAX_FILE_BYTES + 1]);
        assert!(changes(&[file("big.bin", &big)])
            .unwrap_err()
            .starts_with("big.bin is 8388609 bytes"));
        let half = STANDARD.encode(vec![0u8; MAX_FILE_BYTES]);
        assert_eq!(
            changes(&[file("a", &half), file("b", &half), file("c", "AA==")]).unwrap_err(),
            "the files come to more than 16777216 bytes"
        );
    }
}
