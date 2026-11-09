//! What is kept of a site's files, and when it is prepared again.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use axum::http::{header, HeaderValue, StatusCode};
use filetime::{set_file_mtime, FileTime};
use include_dir::{include_dir, Dir};
use tempfile::TempDir;

use super::*;

static COMMON: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/content/common");

fn css(n: usize) -> String {
    ".a { color: red; }\n".repeat(n)
}

fn whole(kept: Option<Kept>) -> Arc<Prepared> {
    match kept {
        Some(Kept::Whole(p)) => p,
        other => panic!("not whole: {other:?}"),
    }
}

/// Set `path`'s modification time `secs` after the epoch.
fn touch(path: &Path, secs: u64) {
    let at = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
    set_file_mtime(path, FileTime::from_system_time(at)).unwrap();
}

/// **A file is prepared once per version**: two lookups share one
/// preparation; a later modification time serves the new bytes under a new
/// tag.
#[tokio::test]
async fn a_disk_file_is_prepared_once_per_version() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("app.css");
    fs::write(&file, css(100)).unwrap();
    touch(&file, 1_000);
    let roots = Roots::disk(&[dir.path().to_path_buf()]);

    let a = whole(roots.prepared("app.css").await);
    let b = whole(roots.prepared("app.css").await);
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(a.plain, css(100).as_bytes());

    fs::write(&file, css(101)).unwrap();
    touch(&file, 2_000);
    let c = whole(roots.prepared("app.css").await);
    assert_eq!(c.plain, css(101).as_bytes());
    assert_ne!(c.tag, a.tag);
}

/// A compiled-in file is prepared once, and every clone of the roots shares
/// what was prepared.
#[tokio::test]
async fn an_embedded_file_is_prepared_once() {
    let roots = Roots::embedded(&[&COMMON]);
    let a = whole(roots.prepared("base.css").await);
    let b = whole(roots.clone().prepared("base.css").await);
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(a.plain, COMMON.get_file("base.css").unwrap().contents());
    assert_eq!(roots.stamped.len(), 1);
}

/// **A file added to an earlier root shadows the one already kept**: the
/// path is resolved through the roots on every request.
#[tokio::test]
async fn a_file_added_to_an_earlier_root_shadows_the_kept_one() {
    let (first, second) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    fs::write(second.path().join("app.css"), css(100)).unwrap();
    let roots = Roots::disk(&[first.path().to_path_buf(), second.path().to_path_buf()]);
    assert_eq!(
        whole(roots.prepared("app.css").await).plain,
        css(100).as_bytes()
    );
    fs::write(first.path().join("app.css"), css(50)).unwrap();
    assert_eq!(
        whole(roots.prepared("app.css").await).plain,
        css(50).as_bytes()
    );
}

/// **A picture keeps its tag, not its bytes**: the second lookup is the tag
/// alone, a client holding it is answered 304 without a read, and one that
/// does not gets the bytes under the same tag.
#[tokio::test]
async fn a_picture_keeps_its_tag_and_reads_its_bytes_to_answer() {
    let dir = TempDir::new().unwrap();
    let png = vec![9u8; 5_000];
    fs::write(dir.path().join("mark.png"), &png).unwrap();
    let roots = Roots::disk(&[dir.path().to_path_buf()]);
    let first = whole(roots.prepared("mark.png").await);
    let Some(Kept::Tagged { mime, hex, .. }) = roots.prepared("mark.png").await else {
        panic!("a picture is kept by its tag");
    };
    assert_eq!(mime, "image/png");
    assert_eq!(tag_of(&hex, false), first.tag);

    let cache = Cache::default();
    let mut holding = HeaderMap::new();
    holding.insert(
        header::IF_NONE_MATCH,
        HeaderValue::from_str(&first.tag).unwrap(),
    );
    let res = roots.serve("mark.png", cache, &holding).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_MODIFIED);
    assert_eq!(res.headers()[header::ETAG], first.tag.as_str());

    let res = roots
        .serve("mark.png", cache, &HeaderMap::new())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()[header::ETAG], first.tag.as_str());
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body, png);
}

/// A gzipped answer's 304 is matched on the gzipped tag.
#[tokio::test]
async fn a_gzipped_answer_revalidates_on_its_own_tag() {
    let roots = Roots::embedded(&[&COMMON]);
    let mut req = HeaderMap::new();
    req.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
    let first = roots
        .serve("base.css", Cache::default(), &req)
        .await
        .unwrap();
    let tag = first.headers()[header::ETAG].to_str().unwrap().to_owned();
    assert!(tag.ends_with("-gz\""), "{tag}");
    req.insert(header::IF_NONE_MATCH, HeaderValue::from_str(&tag).unwrap());
    let again = roots
        .serve("base.css", Cache::default(), &req)
        .await
        .unwrap();
    assert_eq!(again.status(), StatusCode::NOT_MODIFIED);
    assert!(roots
        .serve("none.css", Cache::default(), &req)
        .await
        .is_none());
}
