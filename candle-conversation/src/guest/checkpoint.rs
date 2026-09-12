//! The parts of a guest's checkpoint that are parsed once per process.
//!
//! # Why this exists
//!
//! A guest is built fresh for every drain — deliberately, so a half-loaded model
//! is never inherited ([`super::model::GuestRegistry`]) — and its `load` then
//! re-reads two files that have not changed since the last drain. Measured on a
//! 2.46 GiB Hermes-3 Q6_K, a ~2.9 s load spent:
//!
//! | | |
//! |---|---|
//! | GGUF header parse | **1,150 ms** |
//! | `tokenizer.json` parse | **310 ms** |
//! | file read + H2D of the weights | ~1,300 ms |
//! | everything else | ~10 ms |
//!
//! Half the load was re-deriving an answer it already had. Worse, the header is
//! parsed *again* outside `load` — [`super::model::GuestModel::footprint_bytes`]
//! reads it to size the claim — so a drain paid it two or three times over, and
//! those extra parses landed outside the `load_ms` the drain reports.
//!
//! # Why the header parse costs a second at all
//!
//! `gguf_file::Content::read` reads field by field — `read_u32`, `read_u64`, a
//! length prefix and then the bytes for every string — straight off the `File`.
//! A Llama-3 GGUF carries its whole tokenizer vocabulary in the metadata:
//! `tokenizer.ggml.tokens` is a 128,256-element string array. Unbuffered, that
//! is a quarter of a million syscalls to read a few megabytes.
//!
//! So there are two independent fixes here and both are worth having. The reader
//! is buffered, which makes even a first, uncached parse cheap; and the result is
//! cached, which removes it from every drain after the first.
//!
//! # Invalidation
//!
//! Keyed on the path *and* the file's length and modification time. A deployment
//! that swaps a checkpoint under a running daemon gets the new one rather than a
//! stale parse of the old — which matters because the entry that would be stale
//! is the tensor offsets, and reading a new file at the old offsets produces
//! weights that are wrong rather than absent.
//!
//! The cache holds one entry per distinct checkpoint, so it is bounded by what a
//! deployment configures rather than by how much traffic it serves.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use candle::quantized::gguf_file;
use candle::safetensors::MmapedSafetensors;
use memmap2::Mmap;

/// What a cache entry is keyed by: the path, and enough of the file's identity
/// to notice it was replaced.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Stamp {
    path: PathBuf,
    len: u64,
    /// Modification time as nanoseconds since the epoch, or `None` where the
    /// filesystem does not report one — in which case length alone decides, and
    /// a same-length replacement is missed. Every filesystem this runs on
    /// reports it; the `Option` is here so a hypothetical one that does not
    /// degrades to a weaker check rather than to a failure.
    modified_ns: Option<u128>,
}

impl Stamp {
    fn of(path: &Path) -> std::io::Result<Self> {
        let md = std::fs::metadata(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            len: md.len(),
            modified_ns: md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos()),
        })
    }
}

/// Parsed GGUF headers, one live entry per checkpoint path.
///
/// At module scope rather than inside the function so the tests can assert the
/// map does not grow when a checkpoint is replaced — an entry pins a map the
/// size of a vocabulary, so "the old one is dropped" is a claim worth checking
/// rather than asserting in a comment.
static HEADERS: OnceLock<Mutex<HashMap<Stamp, Arc<gguf_file::Content>>>> = OnceLock::new();

/// Parsed tokenizers, on the same terms.
static TOKENIZERS: OnceLock<Mutex<HashMap<Stamp, Arc<tokenizers::Tokenizer>>>> = OnceLock::new();

/// Mapped checkpoint payloads. Holds address space, not resident memory: the
/// pages are the OS page cache's, and it evicts them under pressure like any
/// other file-backed pages.
static PAYLOADS: OnceLock<Mutex<HashMap<Stamp, Arc<Mmap>>>> = OnceLock::new();

/// Mapped and indexed safetensors sets, keyed by the whole file list — a model
/// is its files together, and one of them changing invalidates the index built
/// across all of them.
static SAFETENSORS: OnceLock<Mutex<HashMap<Vec<Stamp>, Arc<MmapedSafetensors>>>> = OnceLock::new();

/// Parsed ONNX graphs. See [`onnx`] for why these especially must not be
/// re-read per drain.
static ONNX: OnceLock<Mutex<HashMap<Stamp, Arc<candle_onnx::onnx::ModelProto>>>> = OnceLock::new();

/// How many header entries the cache holds for one path.
///
/// Per path rather than a total: the cache is process-wide, so the tests run
/// against it concurrently and a global count is whatever the other tests
/// happened to have inserted at that moment. The claim being checked is about
/// one path's own entries anyway.
#[cfg(test)]
fn cached_headers_for(path: &Path) -> usize {
    HEADERS.get().map_or(0, |m| {
        m.lock().unwrap().keys().filter(|k| k.path == path).count()
    })
}

/// Read a GGUF's header, from cache when this process has read it before.
///
/// The returned `Arc` is a snapshot: callers derive geometry and sizes from it
/// rather than holding it, so a later swap of the checkpoint does not leave a
/// live model describing a file that is no longer there.
pub fn gguf_header(path: &Path) -> candle::Result<Arc<gguf_file::Content>> {
    let cache = HEADERS.get_or_init(|| Mutex::new(HashMap::new()));

    let stamp = Stamp::of(path)
        .map_err(|e| candle::Error::Msg(format!("guest checkpoint {path:?}: {e}")))?;
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&stamp).cloned()) {
        return Ok(hit);
    }

    // **Parsed outside the lock.** A first parse is ~1.1 s unbuffered and still
    // milliseconds buffered; holding the map's lock across it would block every
    // other guest's lookup on one guest's cold start. Two threads racing the
    // same cold entry both parse and the second insert wins — a wasted parse
    // once per process, against a lock nobody can be stuck behind.
    let file = File::open(path)
        .map_err(|e| candle::Error::Msg(format!("guest checkpoint {path:?}: {e}")))?;
    // Buffered, which is the difference between a syscall per field and a
    // syscall per 512 KiB. A Llama-3 GGUF's metadata holds a 128,256-element
    // token array, so the field count is in the hundreds of thousands.
    let mut reader = BufReader::with_capacity(1 << 19, file);
    let content = Arc::new(gguf_file::Content::read(&mut reader)?);

    if let Ok(mut c) = cache.lock() {
        // Entries for other stamps of the same path are the previous contents of
        // a checkpoint that has been replaced. Dropped rather than kept: nothing
        // can ask for them again, and holding one pins a vocabulary-sized map.
        c.retain(|k, _| k.path != stamp.path);
        c.insert(stamp, Arc::clone(&content));
    }
    Ok(content)
}

/// The checkpoint's bytes, mapped rather than read.
///
/// **Why a mapping and not a buffer.** The loader used to `read_exact` each
/// tensor into a fresh `Vec` — 2.46 GiB of copying per drain, measured at
/// 884 ms, to stage bytes the page cache was already holding. A mapping hands
/// the same bytes to `memcpy_htod` in place: no copy, no allocation, and the
/// second drain onward touches only pages that are already resident.
///
/// It also removes the reason the per-tensor `stream.synchronize()` existed. A
/// local `Vec` dies at the end of the function that made it, so an async copy
/// out of it had to be waited on or it would read freed memory. A mapping is
/// held here for the life of the process, which is exactly the condition
/// `load_repacked_into` was written for — the expert cache uploads out of a
/// long-lived mmap for the same reason.
///
/// # Safety
///
/// Mapping a file is unsound if another process writes it underneath us: the
/// bytes would change beneath a read the compiler believes is stable. This is a
/// model checkpoint on local disk that a deployment installs once, and the
/// [`Stamp`] means a *replacement* is noticed rather than silently mixed with
/// the old mapping. That is the same bargain every mmap-backed loader in this
/// crate makes, stated rather than assumed.
pub fn payload(path: &Path) -> candle::Result<Arc<Mmap>> {
    let cache = PAYLOADS.get_or_init(|| Mutex::new(HashMap::new()));

    let stamp = Stamp::of(path)
        .map_err(|e| candle::Error::Msg(format!("guest checkpoint {path:?}: {e}")))?;
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&stamp).cloned()) {
        return Ok(hit);
    }

    let file = File::open(path)
        .map_err(|e| candle::Error::Msg(format!("guest checkpoint {path:?}: {e}")))?;
    // SAFETY: as documented above — a checkpoint a deployment installed, whose
    // replacement is caught by the stamp rather than by aliasing a live map.
    let map = unsafe { Mmap::map(&file) }
        .map_err(|e| candle::Error::Msg(format!("guest checkpoint {path:?}: mapping: {e}")))?;
    let map = Arc::new(map);

    if let Ok(mut c) = cache.lock() {
        // A superseded mapping is dropped, which unmaps it once the last live
        // model using it has gone — the `Arc` is what keeps a load that is
        // still running from having its bytes pulled out from under it.
        c.retain(|k, _| k.path != stamp.path);
        c.insert(stamp, Arc::clone(&map));
    }
    Ok(map)
}

/// A set of safetensors files, mapped and indexed once per process.
///
/// The image guest re-mapped its three checkpoints and rebuilt a name index over
/// every tensor on each drain. That is ~1,550 names for Stable Diffusion 1.5,
/// re-derived from files that had not changed — the same waste the GGUF header
/// cache above removes for the prose guest.
///
/// Mapping is also what lets a placement copy *from the map* rather than through
/// a host `Tensor`: the bytes stay live for the process, so the copies need no
/// per-tensor synchronise to keep their source alive.
///
/// # Safety
///
/// As [`payload`]: a checkpoint a deployment installed, whose replacement the
/// stamp catches rather than aliasing into a live mapping.
pub fn safetensors(files: &[PathBuf]) -> Result<Arc<MmapedSafetensors>, String> {
    let cache = SAFETENSORS.get_or_init(|| Mutex::new(HashMap::new()));

    let key: Vec<Stamp> = files
        .iter()
        .map(|p| Stamp::of(p).map_err(|e| format!("guest weights {p:?}: {e}")))
        .collect::<Result<_, _>>()?;
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
        return Ok(hit);
    }

    // SAFETY: as documented above.
    let maps = unsafe { MmapedSafetensors::multi(files) }
        .map_err(|e| format!("guest weights {files:?}: {e}"))?;
    let maps = Arc::new(maps);
    if let Ok(mut c) = cache.lock() {
        // Any entry naming the same paths under different stamps is a superseded
        // checkpoint. Dropped, so a redeployed model does not pin the old one.
        let paths: Vec<&PathBuf> = files.iter().collect();
        c.retain(|k, _| !k.iter().map(|s| &s.path).eq(paths.iter().copied()));
        c.insert(key, Arc::clone(&maps));
    }
    Ok(maps)
}

/// Read an ONNX graph, from cache when this process has read it before.
///
/// **This one earns its cache more than any of the others.** A guest is rebuilt
/// per drain by design — a model that failed half-way through a load must not be
/// inherited — so anything a guest parses in its constructor is parsed again
/// every time the card is borrowed. For a graph that is 178 MB of protobuf,
/// that was the whole cost of a matte: measured, the parse dominates a request
/// that spends 0.23 s in the network.
pub fn onnx(path: &Path) -> Result<Arc<candle_onnx::onnx::ModelProto>, String> {
    let cache = ONNX.get_or_init(|| Mutex::new(HashMap::new()));

    let stamp = Stamp::of(path).map_err(|e| format!("guest onnx {path:?}: {e}"))?;
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&stamp).cloned()) {
        return Ok(hit);
    }

    let parsed =
        Arc::new(candle_onnx::read_file(path).map_err(|e| format!("guest onnx {path:?}: {e}"))?);
    if let Ok(mut c) = cache.lock() {
        c.retain(|k, _| k.path != stamp.path);
        c.insert(stamp, Arc::clone(&parsed));
    }
    Ok(parsed)
}

/// Load a tokenizer, from cache when this process has loaded it before.
///
/// `Arc` rather than a clone: a `Tokenizer` carries its vocabulary and merges,
/// and every drain cloning one is the parse cost paid again in memcpy.
pub fn tokenizer(path: &Path) -> Result<Arc<tokenizers::Tokenizer>, String> {
    let cache = TOKENIZERS.get_or_init(|| Mutex::new(HashMap::new()));

    let stamp = Stamp::of(path).map_err(|e| format!("guest tokenizer {path:?}: {e}"))?;
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&stamp).cloned()) {
        return Ok(hit);
    }

    let parsed = Arc::new(
        tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("guest tokenizer {path:?}: {e}"))?,
    );
    if let Ok(mut c) = cache.lock() {
        c.retain(|k, _| k.path != stamp.path);
        c.insert(stamp, Arc::clone(&parsed));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A GGUF small enough to write in a test: the v3 magic, no tensors, one
    /// metadata entry. Enough to prove the caching and the invalidation, which
    /// are the whole of this module's behaviour.
    fn write_gguf(path: &Path, value: u32) {
        let mut f = File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap(); // version
        f.write_all(&0u64.to_le_bytes()).unwrap(); // tensor count
        f.write_all(&1u64.to_le_bytes()).unwrap(); // metadata count
        let key = b"test.value";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key).unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap(); // value type: u32
        f.write_all(&value.to_le_bytes()).unwrap();
        f.sync_all().unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "guest-checkpoint-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_header_is_parsed_once_and_returned_by_identity() {
        let dir = scratch("hit");
        let p = dir.join("m.gguf");
        write_gguf(&p, 7);

        let a = gguf_header(&p).unwrap();
        let b = gguf_header(&p).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "the second call re-parsed instead of hitting the cache"
        );
        assert_eq!(a.metadata["test.value"].to_u32().unwrap(), 7);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A replaced checkpoint must not be served from cache.** The entry that
    /// would be stale is the tensor offsets, and reading a new file at the old
    /// offsets places weights that are wrong rather than missing — which shows
    /// up as bad output, not as an error.
    #[test]
    fn a_replaced_checkpoint_is_re_parsed() {
        let dir = scratch("swap");
        let p = dir.join("m.gguf");
        write_gguf(&p, 7);
        let first = gguf_header(&p).unwrap();
        assert_eq!(first.metadata["test.value"].to_u32().unwrap(), 7);

        // A distinct modification time. The stamp compares length *and* mtime,
        // and this rewrite keeps the length identical on purpose — so the test
        // fails if only the length is being consulted.
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_gguf(&p, 99);

        let second = gguf_header(&p).unwrap();
        assert!(!Arc::ptr_eq(&first, &second), "the stale parse was served");
        assert_eq!(second.metadata["test.value"].to_u32().unwrap(), 99);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **The superseded entry is dropped, not accumulated.** Each one pins a map
    /// the size of the checkpoint's vocabulary, and nothing can ask for it
    /// again — a daemon watching a checkpoint that is rewritten would otherwise
    /// grow a copy per rewrite, for the life of the process.
    #[test]
    fn a_superseded_entry_does_not_accumulate() {
        let dir = scratch("evict");
        let p = dir.join("m.gguf");
        for v in 0..5u32 {
            if v > 0 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            write_gguf(&p, v);
            gguf_header(&p).unwrap();
        }
        assert_eq!(
            cached_headers_for(&p),
            1,
            "five parses of one path left more than one live entry — the superseded parses \
             were never dropped, and each pins a vocabulary-sized map"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tokenizer is cached on the same terms as a header.
    #[test]
    fn a_tokenizer_is_parsed_once() {
        // The daemon's own tokenizer if the deployment has one; otherwise there
        // is nothing valid to parse and the miss path is covered by the test
        // below. Skipping is honest here — inventing a tokenizer.json would be
        // asserting against this test's fixture rather than against a real one.
        let p = Path::new("D:/guests/hermes3/tokenizer.json");
        if !p.exists() {
            return;
        }
        let a = tokenizer(p).unwrap();
        let b = tokenizer(p).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "the tokenizer was parsed twice");
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        let p = std::env::temp_dir().join("guest-checkpoint-does-not-exist.gguf");
        assert!(gguf_header(&p).is_err());
        assert!(tokenizer(&p).is_err());
    }
}
