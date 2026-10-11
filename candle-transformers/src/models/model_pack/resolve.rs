//! Finding a request's pack, or building it.
//!
//! ```text
//! model_pack(request)
//!   1. a pack under <root>/<repo>/ whose name the request owns, whose sources
//!      are the request's, whose mode and narrowing are this card's, and whose
//!      sections pass this build's checks           → its path
//!   2. otherwise: fetch the sources, build, release the sources that were
//!      fetched into a cache                         → the new pack's path
//! ```
//!
//! A pack that fails step 1 for any reason but its narrowing is stale — another
//! version, another repack, another source — and is removed once its
//! replacement is published. One of another narrowing belongs to a card of
//! another size and is left alone.

use super::compose::{checkpoint_bytes, MappedSource};
use super::digest::{placeholder, PendingDigests};
use super::family::{build_pack, narrowing, validate};
use super::open::ModelPack;
use super::provenance::{Provenance, SourceRecord};
use super::request::{PackRequest, SourceRef};
use candle::{Device, Result};
use std::path::{Path, PathBuf};

/// A source file on this machine.
pub struct Fetched {
    pub path: PathBuf,
    /// Whether the file lives in a cache this codebase manages (the hub cache,
    /// the model cache) and may be deleted once a pack holds it. `false` for a
    /// file the caller supplied by path: that is the user's, not ours.
    pub cached: bool,
}

/// Where sources and tokenizers come from.
pub trait SourceFetch {
    /// A local copy of `source`, downloading it if there is none.
    fn fetch(&self, source: &SourceRef) -> Result<Fetched>;
    /// `tokenizer.json`'s text from `repo` at `rev`.
    fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String>;
    /// Delete a cached source a pack now holds — the file and whatever cache
    /// object backs it.
    fn release(&self, fetched: &Fetched) -> Result<()>;
}

/// Why a pack on disk is not the answer.
enum Verdict {
    Serves,
    /// Built for a card of another size — another narrowing, or another mode
    /// picked for it: left for that card.
    OtherNarrowing,
    Stale(String),
}

fn judge(path: &Path, request: &PackRequest, device: &Device) -> Verdict {
    let pack = match ModelPack::open(path) {
        Ok(p) => p,
        Err(e) => return Verdict::Stale(e.to_string()),
    };
    if !request.matches(&pack.sources) {
        return Verdict::Stale("built from other sources".into());
    }
    if !request.tokenizer_matches(&pack.tokenizer_repo, &pack.tokenizer_rev) {
        return Verdict::Stale(format!(
            "carries the tokenizer of {}@{}",
            pack.tokenizer_repo, pack.tokenizer_rev
        ));
    }
    // A request that leaves the mode open takes the one this card picks for
    // this checkpoint; a pack of another mode is another card's, like another
    // narrowing.
    let want = request.mode_for(device, pack.checkpoint_bytes);
    if pack.int8_mode != want {
        return match request.int8_mode {
            Some(_) => Verdict::Stale(format!("packed for {:?}", pack.int8_mode)),
            None => Verdict::OtherNarrowing,
        };
    }
    match narrowing(request.family, &pack.content, pack.checkpoint_bytes, device) {
        Ok(want) if want != pack.narrow => return Verdict::OtherNarrowing,
        Ok(_) => {}
        Err(e) => return Verdict::Stale(e.to_string()),
    }
    match validate(request.family, &pack, device) {
        Ok(()) => Verdict::Serves,
        Err(e) => Verdict::Stale(e.to_string()),
    }
}

/// What is on disk for a request: the pack that serves it, if one does, and the
/// packs it owns that are stale.
struct Scan {
    serves: Option<PathBuf>,
    stale: Vec<PathBuf>,
}

fn scan(request: &PackRequest, root: &Path, device: &Device) -> Scan {
    let dir = request.dir(root);
    let mut stale = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut names: Vec<String> = entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| request.owns(n))
            .collect();
        names.sort();
        for name in names {
            let path = dir.join(&name);
            match judge(&path, request, device) {
                Verdict::Serves => {
                    return Scan {
                        serves: Some(path),
                        stale,
                    }
                }
                Verdict::OtherNarrowing => {}
                Verdict::Stale(why) => {
                    tracing::info!(
                        target: "candle_transformers::model_pack",
                        path = %path.display(),
                        %why,
                        "model pack is stale"
                    );
                    stale.push(path);
                }
            }
        }
    }
    Scan {
        serves: None,
        stale,
    }
}

/// The path of a pack under `root` that serves `request` on `device`, without
/// building one — for a caller choosing between requests by what is already
/// built.
pub fn existing_pack(request: &PackRequest, root: &Path, device: &Device) -> Option<PathBuf> {
    scan(request, root, device).serves
}

/// The path of `request`'s pack under `root`, building it if there is none.
pub fn model_pack(
    request: &PackRequest,
    root: &Path,
    device: &Device,
    fetch: &dyn SourceFetch,
) -> Result<PathBuf> {
    let dir = request.dir(root);
    let Scan { serves, stale } = scan(request, root, device);
    if let Some(path) = serves {
        return Ok(path);
    }

    let fetched: Vec<Fetched> = request
        .sources
        .iter()
        .map(|s| fetch.fetch(s))
        .collect::<Result<_>>()?;
    let mut sources = Vec::with_capacity(fetched.len());
    let mut records = Vec::with_capacity(fetched.len());
    for (i, (s, f)) in request.sources.iter().zip(&fetched).enumerate() {
        let len = std::fs::metadata(&f.path)
            .map_err(|e| {
                candle::Error::Msg(format!("model pack source {}: {e}", f.path.display()))
            })?
            .len();
        records.push(SourceRecord {
            role: s.role.clone(),
            repo: s.repo.clone(),
            rev: s.rev.clone(),
            file: s.file.clone(),
            len,
            sha256: placeholder(i),
        });
        sources.push(MappedSource::open(&f.path)?);
    }
    // Hashed beside the build, not ahead of it: see `digest`.
    let provenance = Provenance {
        records,
        digests: PendingDigests::start(fetched.iter().map(|f| f.path.clone()).collect()),
    };
    let tokenizer = fetch.tokenizer_json(&request.tokenizer_repo, &request.tokenizer_rev)?;
    let mode = request.mode_for(device, checkpoint_bytes(&sources[0].content));
    let out = build_pack(request, mode, &sources, provenance, tokenizer, device, &dir)?;
    drop(sources);
    // Through the same check a later load makes, before anything is released: a
    // pack this build cannot serve is never reported built, and its sources are
    // still there to build from again.
    match judge(&out, request, device) {
        Verdict::Serves => {}
        Verdict::OtherNarrowing => candle::bail!(
            "model pack {}: built for a narrowing or mode this card does not take",
            out.display()
        ),
        Verdict::Stale(why) => candle::bail!("model pack {}: built stale: {why}", out.display()),
    }

    for path in stale {
        if path != out {
            let _ = std::fs::remove_file(&path);
        }
    }
    // The pack is published and judged; what is left is disk the sources hold.
    // A source that cannot be deleted — another process has it open, which on
    // Windows refuses the delete — costs that disk and nothing else, so it is
    // reported and the rest are still released, rather than failing a load that
    // has a pack to serve.
    for f in fetched.iter().filter(|f| f.cached) {
        if let Err(e) = fetch.release(f) {
            tracing::warn!(
                target: "candle_transformers::model_pack",
                path = %f.path.display(),
                "model pack built, but its source could not be released: {e}"
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::request::PackFamily;
    use super::*;
    use crate::models::qwen4exp::prepare::store::sha256_file;
    use candle::quantized::gguf_file::Value;
    use candle::quantized::gguf_writer::{GgufPlan, GgufStreamWriter, PlannedTensor};
    use candle::quantized::{GgmlDType, Int8Mode};
    use std::cell::RefCell;
    use std::fs::File;

    /// A one-tensor GGUF checkpoint at `path`.
    fn write_checkpoint(path: &Path) {
        write_checkpoint_of(path, 4);
    }

    /// A one-tensor GGUF checkpoint at `path` whose tensor holds `elems` F32s.
    fn write_checkpoint_of(path: &Path, elems: usize) {
        let mut plan = GgufPlan::new(32).unwrap();
        plan.push_metadata("general.architecture", Value::String("test".into()))
            .unwrap();
        plan.push_tensor(PlannedTensor {
            name: "a.weight".into(),
            dtype: GgmlDType::F32,
            dims: vec![elems],
        })
        .unwrap();
        let mut w = GgufStreamWriter::new(plan, File::create(path).unwrap()).unwrap();
        w.write_tensor_bytes(&vec![1u8; elems * 4]).unwrap();
        w.finish().unwrap();
    }

    /// A hub stand-in: every fetch lays a fresh copy of the checkpoint into its
    /// cache directory, and every release deletes it — and both are counted.
    /// With `refuse_release` set, a release fails as a delete of an open file
    /// does on Windows.
    struct Hub {
        cache: PathBuf,
        fetched: RefCell<usize>,
        released: RefCell<Vec<PathBuf>>,
        refuse_release: bool,
    }

    impl SourceFetch for Hub {
        fn fetch(&self, s: &SourceRef) -> Result<Fetched> {
            *self.fetched.borrow_mut() += 1;
            std::fs::create_dir_all(&self.cache)?;
            let path = self.cache.join(&s.file);
            write_checkpoint(&path);
            Ok(Fetched { path, cached: true })
        }
        fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String> {
            Ok(format!("{{\"from\":\"{repo}@{rev}\"}}"))
        }
        fn release(&self, f: &Fetched) -> Result<()> {
            if self.refuse_release {
                candle::bail!("{} is open in another process", f.path.display());
            }
            std::fs::remove_file(&f.path)?;
            self.released.borrow_mut().push(f.path.clone());
            Ok(())
        }
    }

    fn hub(dir: &Path) -> Hub {
        Hub {
            cache: dir.join("hub"),
            fetched: RefCell::new(0),
            released: RefCell::new(Vec::new()),
            refuse_release: false,
        }
    }

    fn request(rev: &str) -> PackRequest {
        PackRequest::of(
            PackFamily::Plain,
            ("org/m", rev, "m.gguf"),
            ("org/tok", "t1"),
            Some(Int8Mode::Performance),
        )
    }

    /// A miss builds the pack under the request's directory, releases the
    /// source it fetched, and records where everything came from; the next
    /// call is a hit that fetches nothing.
    #[test]
    fn a_miss_builds_and_releases_then_a_hit_fetches_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models");
        let hub = hub(dir.path());
        let pack = model_pack(&request("r1"), &root, &Device::Cpu, &hub).unwrap();
        assert_eq!(pack, root.join("org--m").join("m.performance.pack.gguf"));
        assert_eq!(*hub.fetched.borrow(), 1);
        assert_eq!(*hub.released.borrow(), [hub.cache.join("m.gguf")]);
        assert!(!hub.cache.join("m.gguf").exists());

        // The provenance is the released source's: an identical file has its
        // length and its hash.
        let twin = dir.path().join("twin.gguf");
        write_checkpoint(&twin);
        let opened = ModelPack::open(&pack).unwrap();
        assert_eq!(opened.sources[0].rev, "r1");
        assert_eq!(
            opened.sources[0].len,
            std::fs::metadata(&twin).unwrap().len()
        );
        assert_eq!(opened.sources[0].sha256, sha256_file(&twin).unwrap());
        assert_eq!(
            opened.tokenizer_json().unwrap(),
            "{\"from\":\"org/tok@t1\"}"
        );

        let again = model_pack(&request("r1"), &root, &Device::Cpu, &hub).unwrap();
        assert_eq!(again, pack);
        assert_eq!(*hub.fetched.borrow(), 1, "a hit must not fetch");
        assert_eq!(
            existing_pack(&request("r1"), &root, &Device::Cpu),
            Some(pack)
        );
    }

    /// A pack built from another revision of the source is stale: it is not
    /// served, the request is built afresh, and the stale file is removed.
    #[test]
    fn a_provenance_mismatch_rebuilds_and_removes_the_stale_pack() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models");
        let hub = hub(dir.path());
        let first = model_pack(&request("r1"), &root, &Device::Cpu, &hub).unwrap();
        assert_eq!(existing_pack(&request("r2"), &root, &Device::Cpu), None);
        let second = model_pack(&request("r2"), &root, &Device::Cpu, &hub).unwrap();
        // Same name — the revision is provenance, not part of it — rebuilt over.
        assert_eq!(second, first);
        assert_eq!(*hub.fetched.borrow(), 2);
        assert_eq!(ModelPack::open(&second).unwrap().sources[0].rev, "r2");
    }

    /// A source the caller supplied by path is read in place and never
    /// released, and the pack lands in the cache root, not beside it.
    #[test]
    fn a_callers_file_is_never_released_or_written_beside() {
        use super::super::local::LocalFetch;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models");
        let theirs = dir.path().join("theirs");
        std::fs::create_dir_all(&theirs).unwrap();
        write_checkpoint(&theirs.join("m.gguf"));
        let hub = hub(dir.path());
        let fetch = LocalFetch {
            repo: "local/theirs".into(),
            dir: theirs.clone(),
            other: &hub,
        };
        let request = PackRequest::of(
            PackFamily::Plain,
            ("local/theirs", "", "m.gguf"),
            ("org/tok", "t1"),
            Some(Int8Mode::Performance),
        );
        let pack = model_pack(&request, &root, &Device::Cpu, &fetch).unwrap();
        assert!(pack.starts_with(&root), "{pack:?}");
        assert!(theirs.join("m.gguf").exists());
        assert_eq!(std::fs::read_dir(&theirs).unwrap().count(), 1);
        assert!(hub.released.borrow().is_empty());
        assert_eq!(*hub.fetched.borrow(), 0);
    }

    /// A caller's file reconverted in place under the same name is another
    /// source: pinned at its length and mtime, the old pack no longer serves it.
    #[test]
    fn a_callers_file_rewritten_in_place_is_rebuilt() {
        use super::super::local::{local_rev, LocalFetch};
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models");
        let theirs = dir.path().join("theirs");
        std::fs::create_dir_all(&theirs).unwrap();
        let file = theirs.join("m.gguf");
        write_checkpoint(&file);
        let hub = hub(dir.path());
        let fetch = LocalFetch {
            repo: "local/theirs".into(),
            dir: theirs.clone(),
            other: &hub,
        };
        let request = |rev: &str| {
            PackRequest::of(
                PackFamily::Plain,
                ("local/theirs", rev, "m.gguf"),
                ("org/tok", "t1"),
                Some(Int8Mode::Performance),
            )
        };
        let before = local_rev(&file);
        let pack = model_pack(&request(&before), &root, &Device::Cpu, &fetch).unwrap();

        write_checkpoint_of(&file, 8);
        let after = local_rev(&file);
        assert_ne!(after, before);
        assert_eq!(existing_pack(&request(&after), &root, &Device::Cpu), None);
        let rebuilt = model_pack(&request(&after), &root, &Device::Cpu, &fetch).unwrap();
        assert_eq!(rebuilt, pack);
        let opened = ModelPack::open(&rebuilt).unwrap();
        assert_eq!(opened.sources[0].rev, after);
        assert_eq!(opened.sources[0].sha256, sha256_file(&file).unwrap());
    }

    /// A pack carrying another pinned tokenizer is stale.
    #[test]
    fn another_tokenizer_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models");
        let hub = hub(dir.path());
        model_pack(&request("r1"), &root, &Device::Cpu, &hub).unwrap();
        let mut other = request("r1");
        other.tokenizer_rev = "t2".into();
        assert_eq!(existing_pack(&other, &root, &Device::Cpu), None);
        let rebuilt = model_pack(&other, &root, &Device::Cpu, &hub).unwrap();
        assert_eq!(ModelPack::open(&rebuilt).unwrap().tokenizer_rev, "t2");
    }

    /// A source that cannot be released after the build leaves its disk taken
    /// and nothing else: the pack is still the answer.
    #[test]
    fn a_failed_release_still_answers_the_pack() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models");
        let mut hub = hub(dir.path());
        hub.refuse_release = true;
        let pack = model_pack(&request("r1"), &root, &Device::Cpu, &hub).unwrap();
        assert!(pack.is_file());
        assert!(hub.cache.join("m.gguf").exists());
        assert_eq!(
            existing_pack(&request("r1"), &root, &Device::Cpu),
            Some(pack)
        );
    }
}
