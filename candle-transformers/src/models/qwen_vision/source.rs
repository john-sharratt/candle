//! Where a model's vision tower comes from: a twinned model pack
//! (`model_pack::twins`) built from the model's pinned `mmproj` GGUF, which is
//! fetched from the hub, held to its published length and digest, and released
//! once the pack holds it.
//!
//! The pack stores every projection as the KO twin the tower's matmuls read,
//! and the position table — which the tower gathers rows from — as its file's
//! F16. A load copies each tensor to the card as it is: no source upload, no
//! repack, no scratch.
//!
//! A model names its tower with a [`VisionTowerSource`] constant in its own
//! file (`docs/vision_models.md` §2.1). Resolved by whoever is about to need
//! the pack, on that caller's thread — never by the guest drain that reads it,
//! which runs with every wave stopped.

use candle::quantized::Int8Mode;

/// The int8 mode a tower's pack is built at: a tower's projections are F16,
/// whose twin is `Q8_KO` in either int8 mode, so one pack serves every card
/// with the int8 matmul.
pub const VISION_TOWER_MODE: Int8Mode = Int8Mode::Precision;

/// A model's vision tower: its name, and the pinned `mmproj` it is built from.
#[derive(Debug, PartialEq, Eq)]
pub struct VisionTowerSource {
    /// Names the tower wherever its output is kept: an image's rows are the
    /// tower's, so two towers' rows for one image are two records.
    pub name: &'static str,
    pub repo: &'static str,
    /// The commit the `mmproj` is pinned at.
    pub rev: &'static str,
    /// The `mmproj` GGUF: `(file, bytes, SHA-256)`.
    pub file: (&'static str, u64, &'static str),
    /// The text model's tokenizer `(repo, rev)` — the one whose
    /// `<|image_pad|>`s the tower's rows stand for, which its pack records.
    pub tokenizer: (&'static str, &'static str),
}

#[cfg(all(feature = "cuda", any(test, feature = "hub")))]
mod resolve {
    use std::fs::{self, File};
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::SystemTime;

    use candle::quantized::Int8Mode;
    use candle::{Device, Result};
    use hf_hub::{Repo, RepoType};
    use sha2::{Digest, Sha256};

    use super::VisionTowerSource;
    use crate::models::hub_download::repo_file;
    use crate::models::model_pack::{
        cache_root, existing_pack, model_pack, Fetched, HubFetch, PackFamily, PackRequest,
        SourceFetch, SourceRef,
    };
    use crate::models::qwen_vision::tower::POSITION_TABLE;

    /// The pack each `(tower, mode)` last resolved to in this process.
    ///
    /// Judging a pack opens its header, and the tokenizer it carries makes
    /// that megabytes of metadata. The drain asks for the path with every wave
    /// stopped, so it is judged once a process — and again only if the file is
    /// gone.
    static RESOLVED: Mutex<Vec<(&'static str, Int8Mode, PathBuf)>> = Mutex::new(Vec::new());

    /// The paths whose digest matched their pin, and their modification times
    /// then.
    static VERIFIED: Mutex<Vec<(PathBuf, SystemTime)>> = Mutex::new(Vec::new());

    impl VisionTowerSource {
        /// The tower's pack at `mode`, as the resolver is asked for it. The
        /// position table stays its file's F16: the tower gathers rows from
        /// it, and an int8 twin is a layout only the matmul reads.
        pub fn request(&self, mode: Int8Mode) -> PackRequest {
            PackRequest::of(
                PackFamily::Twinned(&[POSITION_TABLE]),
                (self.repo, self.rev, self.file.0),
                self.tokenizer,
                Some(mode),
            )
        }

        fn remembered(&self, mode: Int8Mode) -> Option<PathBuf> {
            let resolved = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
            resolved
                .iter()
                .find(|(n, m, p)| *n == self.name && *m == mode && p.is_file())
                .map(|(_, _, p)| p.clone())
        }

        fn remember(&self, mode: Int8Mode, path: &Path) {
            let mut resolved = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
            resolved.retain(|(n, m, _)| !(*n == self.name && *m == mode));
            resolved.push((self.name, mode, path.to_path_buf()));
        }

        /// The tower's pack at `mode`, if this machine already holds it, with
        /// no network and no build. The request names its mode, so no card is
        /// consulted to pick one, and the device the check is given is the
        /// host's.
        pub fn cached_pack(&self, mode: Int8Mode) -> Option<PathBuf> {
            if let Some(path) = self.remembered(mode) {
                return Some(path);
            }
            let path = existing_pack(&self.request(mode), &cache_root(), &Device::Cpu)?;
            self.remember(mode, &path);
            Some(path)
        }

        /// The tower's pack at `mode`, built — from the pinned `mmproj`,
        /// fetched when it is not cached — when there is none. The build is the
        /// host's: the twins are converted on the CPU.
        pub fn pack(&'static self, mode: Int8Mode) -> Result<PathBuf> {
            if let Some(path) = self.remembered(mode) {
                return Ok(path);
            }
            let path = model_pack(
                &self.request(mode),
                &cache_root(),
                &Device::Cpu,
                &PinnedFetch(self),
            )?;
            self.remember(mode, &path);
            Ok(path)
        }

        /// The `mmproj`, fetched when it is not cached, its length checked
        /// every call and its SHA-256 whenever the file is not the one last
        /// verified. Only a match is remembered, so a file fetched again after
        /// a failure is checked again.
        fn pinned_mmproj(&self) -> Result<PathBuf> {
            let (name, bytes, sha) = self.file;
            let repo =
                Repo::with_revision(self.repo.to_string(), RepoType::Model, self.rev.to_string());
            let path = repo_file(&repo, name)?;
            let meta = fs::metadata(&path)?;
            if meta.len() != bytes {
                candle::bail!("{path:?} is {} bytes, the pin is {bytes}", meta.len());
            }
            let modified = meta.modified()?;
            let mut verified = VERIFIED.lock().unwrap_or_else(|e| e.into_inner());
            if verified.iter().any(|(p, m)| *p == path && *m == modified) {
                return Ok(path);
            }
            let mut f = File::open(&path)?;
            let mut h = Sha256::new();
            io::copy(&mut f, &mut h)?;
            let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            if got != sha {
                candle::bail!("{path:?} has SHA-256 {got}, the pin is {sha}");
            }
            verified.retain(|(p, _)| *p != path);
            verified.push((path.clone(), modified));
            Ok(path)
        }
    }

    /// The hub, with a tower's `mmproj` held to its pin before a pack is built
    /// from it.
    struct PinnedFetch(&'static VisionTowerSource);

    impl SourceFetch for PinnedFetch {
        fn fetch(&self, source: &SourceRef) -> Result<Fetched> {
            if source.file != self.0.file.0 {
                candle::bail!(
                    "the {} tower's pack is built from {}, not {}",
                    self.0.name,
                    self.0.file.0,
                    source.file
                );
            }
            Ok(Fetched {
                path: self.0.pinned_mmproj()?,
                cached: true,
            })
        }

        fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String> {
            HubFetch.tokenizer_json(repo, rev)
        }

        fn release(&self, fetched: &Fetched) -> Result<()> {
            HubFetch.release(fetched)
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::model_pack::PackFamily;

    const TEST_TOWER: VisionTowerSource = VisionTowerSource {
        name: "test-tower",
        repo: "org/model-GGUF",
        rev: "0123456789abcdef0123456789abcdef01234567",
        file: ("mmproj-F16.gguf", 1, "00"),
        tokenizer: ("org/model", "fedcba9876543210fedcba9876543210fedcba98"),
    };

    /// A tower's pack is named for its `mmproj` and mode, in its repo's
    /// directory, records the text model's tokenizer, and keeps the position
    /// table out of its twins.
    #[test]
    fn a_towers_pack_is_named_for_its_mmproj() {
        let r = TEST_TOWER.request(VISION_TOWER_MODE);
        assert_eq!(
            r.file_name(VISION_TOWER_MODE, None),
            "mmproj-F16.precision.pack.gguf"
        );
        assert_eq!(r.family, PackFamily::Twinned(&["v.position_embd.weight"]));
        assert_eq!(r.checkpoint().repo, "org/model-GGUF");
        assert_eq!(r.checkpoint().rev, TEST_TOWER.rev);
        assert_eq!(
            (r.tokenizer_repo.as_str(), r.tokenizer_rev.as_str()),
            TEST_TOWER.tokenizer
        );
    }
}
