//! A site's file answered from what was prepared of it, prepared again only
//! when the file has changed.
//!
//! What is kept of a file: a compiled-in file whole, outside the cap; a disk
//! text file up to [`WHOLE_MAX`] whole — its bytes, gzipped bytes and tags;
//! any other disk file its type and tag only, so a `304` reads nothing and a
//! `200` reads the bytes and reuses the tag. A disk file's entry is kept only
//! when a stat after the read agrees with the one before it, so a file
//! written while it was read is not kept as the version it was not.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Response};
use tokio::fs;

use super::{stamp_of, Resolved, Roots};
use crate::asset::{self, compressible, mime_of, tag_of, Prepared};
use crate::config::Cache;

/// The largest disk text file kept whole.
pub const WHOLE_MAX: usize = 8 << 20;

/// What is kept of one file.
#[derive(Clone, Debug)]
pub enum Kept {
    /// Ready to answer with.
    Whole(Arc<Prepared>),
    /// A disk file's type and content hash: its bytes are read to answer.
    Tagged {
        real: PathBuf,
        mime: String,
        hex: String,
    },
}

impl Roots {
    /// What is prepared of `rel`, prepared now when nothing kept is of the
    /// file as it stands. `rel` must already be validated by
    /// [`super::safe_rel`].
    pub async fn prepared(&self, rel: &str) -> Option<Kept> {
        let resolved = self.resolve(rel).await?;
        let mime = mime_of(rel);
        let key = resolved.key(&mime);
        if let Some(kept) = self.stamped.get(&key, resolved.stamp()) {
            return Some(kept);
        }
        match resolved {
            Resolved::Embedded(f) => {
                let p = Arc::new(Prepared::typed(&mime, Bytes::from_static(f.contents())));
                self.stamped.put(key, None, Kept::Whole(Arc::clone(&p)), 0);
                Some(Kept::Whole(p))
            }
            Resolved::Disk { real, stamp } => {
                let bytes = fs::read(&real).await.ok()?;
                let unchanged = stamp_of(&real).await == Some(stamp);
                let p = Arc::new(Prepared::typed(&mime, Bytes::from(bytes)));
                if unchanged {
                    let whole = compressible(&mime) && p.plain.len() <= WHOLE_MAX;
                    let (kept, cost) = if whole {
                        let cost = p.plain.len() + p.gz.as_ref().map_or(0, Bytes::len);
                        (Kept::Whole(Arc::clone(&p)), cost)
                    } else {
                        let hex = p.tag.trim_matches('"').to_string();
                        let kept = Kept::Tagged {
                            real,
                            mime: mime.clone(),
                            hex,
                        };
                        (kept, 0)
                    };
                    self.stamped.put(key, Some(stamp), kept, cost);
                }
                Some(Kept::Whole(p))
            }
        }
    }

    /// The answer for `rel`: what was prepared of it, as `req` holds or takes
    /// it, aged by `cache`. `None` when no root has it.
    pub async fn serve(&self, rel: &str, cache: Cache, req: &HeaderMap) -> Option<Response<Body>> {
        match self.prepared(rel).await? {
            Kept::Whole(p) => Some(asset::respond(&p, cache, req)),
            Kept::Tagged { real, mime, hex } => {
                if let Some(res) = asset::not_modified(&mime, &tag_of(&hex, false), cache, req) {
                    return Some(res);
                }
                // The stamp this tag was kept at was checked a moment ago.
                let bytes = fs::read(&real).await.ok()?;
                let p = Prepared::tagged(&mime, Bytes::from(bytes), &hex);
                Some(asset::respond(&p, cache, req))
            }
        }
    }
}

#[cfg(test)]
mod tests;
