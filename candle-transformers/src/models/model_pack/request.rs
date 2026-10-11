//! What a caller asks for, and the file name the answer lives under.

use super::provenance::SourceRecord;
#[cfg(feature = "cuda")]
use crate::models::latent_moe::Arch;
use candle::quantized::Int8Mode;
use candle::Device;
use sha2::{Digest, Sha256};
use std::mem::discriminant;
use std::path::{Path, PathBuf};

/// Which model family's build a pack takes: what its sections are and how its
/// source files compose.
#[derive(Debug, Clone, Copy)]
pub enum PackFamily {
    /// No section: the checkpoint's tensors, the tokenizer and the provenance.
    Plain,
    /// A routed checkpoint of the qwen lineage, whose experts are the merged
    /// `blk.{n}.ffn_{gate,up,down}_exps` tensors: an expert section. Qwen3-MoE
    /// and Qwen3.8-Flash-Next's prepared artifact.
    Routed,
    /// A sparse-latent checkpoint (`latent_moe`): an expert section, with every
    /// expert tensor named through the model's own [`Arch`] — that family never
    /// names a tensor by another model's convention.
    #[cfg(feature = "cuda")]
    Latent(&'static dyn Arch),
    /// The Qwen3.5 lineage: an expert section when it routes, a layer section
    /// when it is dense; a draft head, gate donor or tensor overrides fold in.
    Qwen35,
}

/// Families are equal by kind, and a sparse-latent one by its arch's id — an
/// `Arch` is a model, and two references to it are the same model.
impl PartialEq for PackFamily {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            #[cfg(feature = "cuda")]
            (Self::Latent(a), Self::Latent(b)) => a.id() == b.id(),
            (a, b) => discriminant(a) == discriminant(b),
        }
    }
}

impl Eq for PackFamily {}

/// One source file, as the caller names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRef {
    /// `checkpoint`, `mtp`, `gate-donor`, `override:<tensor>`.
    pub role: String,
    pub repo: String,
    /// The pinned revision, or empty to take whatever the hub calls `main`.
    pub rev: String,
    pub file: String,
}

impl SourceRef {
    pub fn checkpoint(repo: &str, rev: &str, file: &str) -> Self {
        Self {
            role: "checkpoint".into(),
            repo: repo.into(),
            rev: rev.into(),
            file: file.into(),
        }
    }

    /// Whether `record` is the file this names: role, repo and file always; the
    /// revision when this pins one.
    pub fn names(&self, record: &SourceRecord) -> bool {
        self.role == record.role
            && self.repo == record.repo
            && self.file == record.file
            && (self.rev.is_empty() || self.rev == record.rev)
    }
}

/// A model, as a caller asks for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackRequest {
    pub family: PackFamily,
    /// The checkpoint first, then any file that completes it.
    pub sources: Vec<SourceRef>,
    pub tokenizer_repo: String,
    pub tokenizer_rev: String,
    /// The numeric mode the pack's sections are built for — or `None` for the
    /// one [`Int8Mode::auto_sized`] picks for this checkpoint on the device that
    /// loads it ([`Self::mode_for`]).
    pub int8_mode: Option<Int8Mode>,
}

/// The name a mode takes in a pack's file name.
fn mode_name(mode: Int8Mode) -> &'static str {
    match mode {
        Int8Mode::Off => "off",
        Int8Mode::Performance => "performance",
        Int8Mode::Precision => "precision",
    }
}

impl PackRequest {
    /// A request for one checkpoint, `(repo, rev, file)`, with its tokenizer
    /// `(repo, rev)`.
    pub fn of(
        family: PackFamily,
        (repo, rev, file): (&str, &str, &str),
        (tokenizer_repo, tokenizer_rev): (&str, &str),
        int8_mode: Option<Int8Mode>,
    ) -> Self {
        Self {
            family,
            sources: vec![SourceRef::checkpoint(repo, rev, file)],
            tokenizer_repo: tokenizer_repo.into(),
            tokenizer_rev: tokenizer_rev.into(),
            int8_mode,
        }
    }

    /// This request with another source file completing the checkpoint —
    /// `mtp`, `gate-donor` or `override:<tensor>`.
    pub fn with_source(mut self, role: &str, (repo, rev, file): (&str, &str, &str)) -> Self {
        self.sources.push(SourceRef {
            role: role.into(),
            repo: repo.into(),
            rev: rev.into(),
            file: file.into(),
        });
        self
    }

    /// The checkpoint the pack is named for.
    pub fn checkpoint(&self) -> &SourceRef {
        &self.sources[0]
    }

    /// Whether a pack built from `records` is the one asked for: every source
    /// named, in order, and nothing else.
    pub fn matches(&self, records: &[SourceRecord]) -> bool {
        records.len() == self.sources.len()
            && self.sources.iter().zip(records).all(|(s, r)| s.names(r))
    }

    /// Whether a pack whose tokenizer came from `repo` at `rev` carries the
    /// one asked for: the repo always, the revision when this pins one.
    pub fn tokenizer_matches(&self, repo: &str, rev: &str) -> bool {
        self.tokenizer_repo == repo && (self.tokenizer_rev.is_empty() || self.tokenizer_rev == rev)
    }

    /// The directory this request's packs live in: the checkpoint repo's, under
    /// `root`.
    pub fn dir(&self, root: &Path) -> PathBuf {
        root.join(self.checkpoint().repo.replace('/', "--"))
    }

    /// The stem every pack of this request shares — the checkpoint's file stem,
    /// and a digest of the other sources when there are any, so a fine-tune
    /// repaired from a donor never shares a name with the plain checkpoint.
    pub fn stem(&self) -> String {
        let file = &self.checkpoint().file;
        let base = Path::new(file)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.clone());
        if self.sources.len() == 1 {
            return base;
        }
        let mut h = Sha256::new();
        for s in &self.sources[1..] {
            for part in [&s.role, &s.repo, &s.rev, &s.file] {
                h.update(part.as_bytes());
                h.update([0u8]);
            }
        }
        let digest = h.finalize();
        let tag: String = digest[..4].iter().map(|b| format!("{b:02x}")).collect();
        format!("{base}.{tag}")
    }

    /// The numeric mode this request's pack takes on `device`, for a checkpoint
    /// of `checkpoint_bytes`: the one named, or the one [`Int8Mode::auto_sized`]
    /// picks.
    pub fn mode_for(&self, device: &Device, checkpoint_bytes: u64) -> Int8Mode {
        self.int8_mode
            .unwrap_or_else(|| Int8Mode::auto_sized(device, checkpoint_bytes as usize))
    }

    /// The pack's file name at numeric mode `mode` and layer narrowing `narrow`.
    pub fn file_name(&self, mode: Int8Mode, narrow: Option<usize>) -> String {
        let mode = mode_name(mode);
        match narrow {
            Some(n) => format!("{}.{mode}.n{n}.pack.gguf", self.stem()),
            None => format!("{}.{mode}.pack.gguf", self.stem()),
        }
    }

    /// Whether `name` is one of this request's packs — at the mode it names, or
    /// any mode when it leaves the choice open — at any narrowing.
    pub fn owns(&self, name: &str) -> bool {
        let modes: Vec<Int8Mode> = match self.int8_mode {
            Some(m) => vec![m],
            None => vec![Int8Mode::Off, Int8Mode::Performance, Int8Mode::Precision],
        };
        modes.iter().any(|&mode| {
            let wide = self.file_name(mode, None);
            let Some(prefix) = wide.strip_suffix(".pack.gguf") else {
                return false;
            };
            name == wide
                || name
                    .strip_prefix(prefix)
                    .and_then(|rest| rest.strip_prefix(".n"))
                    .and_then(|rest| rest.strip_suffix(".pack.gguf"))
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PackRequest {
        PackRequest {
            family: PackFamily::Routed,
            sources: vec![SourceRef::checkpoint(
                "unsloth/Qwen3-30B-A3B-GGUF",
                "",
                "Qwen3-30B-A3B-Q4_K_M.gguf",
            )],
            tokenizer_repo: "Qwen/Qwen3-30B-A3B".into(),
            tokenizer_rev: "main".into(),
            int8_mode: Some(Int8Mode::Performance),
        }
    }

    const PERF: Int8Mode = Int8Mode::Performance;

    #[test]
    fn a_pack_is_named_for_its_checkpoint_and_mode() {
        let r = request();
        assert_eq!(
            r.file_name(PERF, None),
            "Qwen3-30B-A3B-Q4_K_M.performance.pack.gguf"
        );
        assert_eq!(
            r.file_name(PERF, Some(64)),
            "Qwen3-30B-A3B-Q4_K_M.performance.n64.pack.gguf"
        );
        assert_eq!(
            r.dir(Path::new("root")),
            Path::new("root").join("unsloth--Qwen3-30B-A3B-GGUF")
        );
    }

    /// A checkpoint in a repo subfolder is named by its file stem alone.
    #[test]
    fn a_subfolder_checkpoint_is_named_by_its_stem() {
        let mut r = request();
        r.sources[0].file = "Q8_0/model-Q8_0.gguf".into();
        assert_eq!(r.file_name(PERF, None), "model-Q8_0.performance.pack.gguf");
    }

    /// Extra sources change the name, so the repaired fine-tune and the plain
    /// checkpoint never collide — and the same extras always give the same name.
    #[test]
    fn extra_sources_change_the_name_deterministically() {
        let mut r = request();
        r.sources.push(SourceRef {
            role: "gate-donor".into(),
            repo: "org/base".into(),
            rev: "r".into(),
            file: "base.gguf".into(),
        });
        let a = r.file_name(PERF, None);
        assert_ne!(a, request().file_name(PERF, None));
        assert_eq!(a, r.clone().file_name(PERF, None));
        assert!(a.starts_with("Qwen3-30B-A3B-Q4_K_M.") && a.ends_with(".performance.pack.gguf"));
    }

    #[test]
    fn a_request_owns_its_packs_at_any_narrowing_and_nothing_else() {
        let r = request();
        assert!(r.owns("Qwen3-30B-A3B-Q4_K_M.performance.pack.gguf"));
        assert!(r.owns("Qwen3-30B-A3B-Q4_K_M.performance.n64.pack.gguf"));
        assert!(!r.owns("Qwen3-30B-A3B-Q4_K_M.precision.pack.gguf"));
        assert!(!r.owns("Qwen3-30B-A3B-Q4_K_M.performance.n.pack.gguf"));
        assert!(!r.owns("Qwen3-30B-A3B-Q4_K_M.gguf"));
    }

    /// A request that leaves the mode open owns its packs at every mode.
    #[test]
    fn an_open_mode_owns_every_mode() {
        let mut r = request();
        r.int8_mode = None;
        assert!(r.owns("Qwen3-30B-A3B-Q4_K_M.performance.pack.gguf"));
        assert!(r.owns("Qwen3-30B-A3B-Q4_K_M.precision.n64.pack.gguf"));
        assert!(!r.owns("Other-Q4_K_M.precision.pack.gguf"));
    }

    /// A pinned request matches only its revision; an unpinned one any.
    #[test]
    fn a_record_matches_on_role_repo_file_and_a_pinned_rev() {
        let rec = SourceRecord {
            role: "checkpoint".into(),
            repo: "unsloth/Qwen3-30B-A3B-GGUF".into(),
            rev: "abc".into(),
            file: "Qwen3-30B-A3B-Q4_K_M.gguf".into(),
            len: 1,
            sha256: String::new(),
        };
        let mut r = request();
        assert!(r.matches(std::slice::from_ref(&rec)));
        r.sources[0].rev = "def".into();
        assert!(!r.matches(std::slice::from_ref(&rec)));
        r.sources[0].rev = "abc".into();
        assert!(r.matches(std::slice::from_ref(&rec)));
        assert!(!r.matches(&[rec.clone(), rec]));
    }

    /// The tokenizer's repo always counts; its revision only when pinned.
    #[test]
    fn a_tokenizer_matches_on_its_repo_and_a_pinned_rev() {
        let mut r = request();
        r.tokenizer_repo = "Qwen/Qwen3-30B-A3B".into();
        r.tokenizer_rev = String::new();
        assert!(r.tokenizer_matches("Qwen/Qwen3-30B-A3B", "anything"));
        assert!(!r.tokenizer_matches("Qwen/Qwen3-8B", "anything"));
        r.tokenizer_rev = "local-5-1".into();
        assert!(r.tokenizer_matches("Qwen/Qwen3-30B-A3B", "local-5-1"));
        assert!(!r.tokenizer_matches("Qwen/Qwen3-30B-A3B", "local-3-1"));
    }
}
