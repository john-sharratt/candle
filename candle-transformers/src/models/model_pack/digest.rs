//! The sources' SHA-256s, taken beside the build rather than before it.
//!
//! A pack records a whole-file SHA-256 of every source (`provenance`). The
//! build cannot take it from its own reads — those follow the composition and
//! the repack, not the file's order — so the hash is its own sequential read of
//! each source. Run first, that read was a full pass over up to 156 GB before
//! the build began. Run beside the build it overlaps the repack instead: each
//! source is hashed on its own thread from the moment the build starts.
//!
//! The metadata is laid out before the hashes exist, so each source's entry is
//! written as a placeholder of a digest's exact length ([`placeholder`]). Before
//! the pack is published the build waits for the hashes and writes each over
//! its placeholder in the temp file ([`patch_digests`]); nothing moves, and the
//! rename that publishes the pack still happens once, after it is complete.

use candle::quantized::gguf_file::Content;
use candle::Result;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// A lowercase hex SHA-256's length.
const DIGEST_LEN: usize = 64;

/// The stand-in for source `i`'s SHA-256 in a pack's metadata until it is
/// known: a digest's length, and text no digest can be.
pub fn placeholder(i: usize) -> String {
    let tag = format!("zen-sha256-pending-{i}-");
    format!("{tag:-<width$}", width = DIGEST_LEN)
}

/// The SHA-256s of a build's sources, each being taken on its own thread.
///
/// Dropped without [`Self::wait`] — a build that failed — it stops them: the
/// threads would otherwise go on reading every source to the end for a pack
/// that will never be published.
pub struct PendingDigests {
    handles: Vec<JoinHandle<Result<String>>>,
    cancel: Arc<AtomicBool>,
}

impl PendingDigests {
    /// Start hashing each of `paths`, in order.
    pub fn start(paths: Vec<PathBuf>) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let handles = paths
            .into_iter()
            .map(|path| {
                let cancel = Arc::clone(&cancel);
                std::thread::spawn(move || sha256_until(&path, &cancel))
            })
            .collect();
        Self { handles, cancel }
    }

    /// No digests to wait for: the records already carry theirs.
    pub fn none() -> Self {
        Self {
            handles: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Every source's digest, in order, once each is taken.
    pub fn wait(mut self) -> Result<Vec<String>> {
        std::mem::take(&mut self.handles)
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| candle::Error::Msg("model pack: a source hash panicked".into()))?
            })
            .collect()
    }
}

impl Drop for PendingDigests {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// `path`'s SHA-256, lowercase hex — or an error once `cancel` is set.
fn sha256_until(path: &Path, cancel: &AtomicBool) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        if cancel.load(Ordering::Relaxed) {
            candle::bail!("model pack: hashing {} stopped", path.display());
        }
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Write `digests[i]` over source `i`'s [`placeholder`] in the GGUF part of the
/// file at `path`, and sync it.
///
/// Each placeholder must occur exactly once in the metadata: one that is
/// missing, or that a stray metadata value happens to repeat, is refused rather
/// than written somewhere it does not belong.
pub fn patch_digests(path: &Path, digests: &[String]) -> Result<()> {
    if digests.is_empty() {
        return Ok(());
    }
    let header_len = {
        let mut r = BufReader::new(File::open(path)?);
        Content::read(&mut r)?.tensor_data_offset as usize
    };
    let mut header = vec![0u8; header_len];
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.read_exact(&mut header)?;
    for (i, digest) in digests.iter().enumerate() {
        if digest.len() != DIGEST_LEN {
            candle::bail!("model pack: source {i}'s digest is {} chars", digest.len());
        }
        let needle = placeholder(i);
        let mut at = header
            .windows(DIGEST_LEN)
            .enumerate()
            .filter(|(_, w)| *w == needle.as_bytes())
            .map(|(o, _)| o);
        let (Some(offset), None) = (at.next(), at.next()) else {
            candle::bail!(
                "model pack: source {i}'s digest placeholder is not in {} exactly once",
                path.display()
            );
        };
        f.seek(SeekFrom::Start(offset as u64))?;
        f.write_all(digest.as_bytes())?;
    }
    f.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::Value;
    use candle::quantized::gguf_writer::{GgufPlan, GgufStreamWriter, PlannedTensor};
    use candle::quantized::GgmlDType;

    /// "abc"'s SHA-256, the standard test vector.
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn a_placeholder_is_a_digests_length_and_never_hex() {
        for i in [0, 7, 12] {
            let p = placeholder(i);
            assert_eq!(p.len(), DIGEST_LEN);
            assert!(!p.bytes().all(|b| b.is_ascii_hexdigit()), "{p}");
        }
        let p = placeholder(3);
        assert_eq!(&p[..21], "zen-sha256-pending-3-");
        assert!(p[21..].bytes().all(|b| b == b'-'), "{p}");
        assert_ne!(placeholder(1), placeholder(11));
    }

    #[test]
    fn digests_are_taken_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        std::fs::write(&a, b"abc").unwrap();
        std::fs::write(&b, b"").unwrap();
        let got = PendingDigests::start(vec![a, b]).wait().unwrap();
        assert_eq!(
            got,
            [
                ABC,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ]
        );
    }

    /// A GGUF whose metadata carries both sources' placeholders reads back with
    /// the digests in their place, and its tensor data untouched.
    #[test]
    fn a_patch_lands_on_the_placeholders_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.gguf");
        let mut plan = GgufPlan::new(32).unwrap();
        plan.push_metadata("zen.source.0.sha256", Value::String(placeholder(0)))
            .unwrap();
        plan.push_metadata("zen.source.1.sha256", Value::String(placeholder(1)))
            .unwrap();
        plan.push_tensor(PlannedTensor {
            name: "a".into(),
            dtype: GgmlDType::F32,
            dims: vec![4],
        })
        .unwrap();
        let mut w = GgufStreamWriter::new(plan, File::create(&path).unwrap()).unwrap();
        w.write_tensor_bytes(&[9u8; 16]).unwrap();
        w.finish().unwrap();
        let len = std::fs::metadata(&path).unwrap().len();

        let other = "00".repeat(32);
        patch_digests(&path, &[ABC.to_string(), other.clone()]).unwrap();

        assert_eq!(std::fs::metadata(&path).unwrap().len(), len);
        let mut r = BufReader::new(File::open(&path).unwrap());
        let c = Content::read(&mut r).unwrap();
        let s = |k: &str| c.metadata[k].to_string().unwrap().clone();
        assert_eq!(s("zen.source.0.sha256"), ABC);
        assert_eq!(s("zen.source.1.sha256"), other);
        let bytes = std::fs::read(&path).unwrap();
        let at = c.tensor_data_offset as usize;
        assert_eq!(&bytes[at..at + 16], &[9u8; 16]);
    }

    /// A placeholder that is not there is refused, and the file is unchanged.
    #[test]
    fn a_missing_placeholder_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.gguf");
        let mut plan = GgufPlan::new(32).unwrap();
        plan.push_metadata("k", Value::String("v".into())).unwrap();
        let w = GgufStreamWriter::new(plan, File::create(&path).unwrap()).unwrap();
        w.finish().unwrap();
        let before = std::fs::read(&path).unwrap();
        let e = patch_digests(&path, &[ABC.to_string()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("exactly once"), "{e}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
}
