//! What a prepared engine artifact is built from, and the identity that names it.
//!
//! No repository publishes the engine GGUF: it is assembled on each machine from
//! pinned releases, as a **hybrid** — the trunk and the n-gram (PLE) table
//! verbatim at `Q8_0`, the MTP draft head's dense weights at `Q8_0`, and the
//! routed experts at the width the card's rung calls for
//! ([`crate::models::quant_ladder`]). The recipe is that whole description: every
//! source file (repo, revision, path, length, LFS SHA-256) plus every choice the
//! build makes about it.
//!
//! Its SHA-256 over a canonical text form is the artifact's identity. The build
//! stamps it into the artifact's GGUF metadata and puts its first
//! [`TAG_HEX`] hex digits in the filename, so a different recipe — a new quant
//! level, a new pin, a converter change — names a different file, and a machine
//! holding the old one builds the new one rather than loading bytes that no
//! longer match what the code would produce.

use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::GgmlDType;
use sha2::{Digest, Sha256};

/// The GGUF metadata key holding the recipe's hex SHA-256.
pub const STAMP_KEY: &str = "zen.prepare.recipe";

/// The GGUF metadata key holding the canonical recipe text itself, so an
/// artifact says what it was built from without the code that built it.
pub const CANONICAL_KEY: &str = "zen.prepare.canonical";

/// Hex digits of the digest carried in the artifact's filename.
pub const TAG_HEX: usize = 12;

/// What a source file is to the build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceRole {
    /// A shard of the `Q8_0` split: trunk, n-gram table and `Q8_0` experts.
    Trunk,
    /// The MTP draft head, merged in as the block past the trunk.
    DraftHead,
    /// A W4A16 safetensors shard the `Q4_KO` experts are imported from.
    ExpertImport,
}

impl SourceRole {
    fn name(self) -> &'static str {
        match self {
            Self::Trunk => "trunk",
            Self::DraftHead => "draft_head",
            Self::ExpertImport => "expert_import",
        }
    }
}

/// One pinned file the build reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub role: SourceRole,
    pub repo: &'static str,
    pub revision: &'static str,
    /// Path inside the repository at `revision`.
    pub path: &'static str,
    /// Published length in bytes.
    pub bytes: u64,
    /// The LFS object id: the file's SHA-256, lowercase hex.
    pub sha256: &'static str,
}

/// Where the routed experts' bytes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSource {
    /// Left at the split's own `Q8_0`.
    Verbatim,
    /// Requantized from the split's `Q8_0` to a KO format on the GPU.
    Requantized,
    /// Imported bit-exactly from the W4A16 release (`convert::convert_w4a16_experts`).
    AwqImport,
}

impl ExpertSource {
    fn name(self) -> &'static str {
        match self {
            Self::Verbatim => "verbatim",
            Self::Requantized => "requantized",
            Self::AwqImport => "awq_import",
        }
    }
}

/// The full description of one engine artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    pub sources: Vec<SourceFile>,
    /// Every non-expert tensor of the trunk, the n-gram table included.
    pub trunk: GgmlDType,
    /// The draft head's non-expert tensors.
    pub head_dense: GgmlDType,
    /// The trunk's routed experts.
    pub experts: GgmlDType,
    pub expert_source: ExpertSource,
    /// The draft head's routed experts.
    pub head_experts: GgmlDType,
    /// Bumped whenever the build's output bytes change for an unchanged recipe.
    pub converter_version: u32,
}

impl Recipe {
    /// The canonical text form: one `key=value` line per field, sources sorted
    /// by `(role, repo, path)` so the order they were listed in cannot change the
    /// identity.
    pub fn canonical(&self) -> String {
        let mut sources: Vec<&SourceFile> = self.sources.iter().collect();
        sources
            .sort_by(|a, b| (a.role.name(), a.repo, a.path).cmp(&(b.role.name(), b.repo, b.path)));
        let mut out = String::new();
        out.push_str(&format!("converter_version={}\n", self.converter_version));
        out.push_str(&format!("trunk={:?}\n", self.trunk));
        out.push_str(&format!("head_dense={:?}\n", self.head_dense));
        out.push_str(&format!("experts={:?}\n", self.experts));
        out.push_str(&format!("expert_source={}\n", self.expert_source.name()));
        out.push_str(&format!("head_experts={:?}\n", self.head_experts));
        for s in sources {
            out.push_str(&format!(
                "source={} {}@{} {} {} {}\n",
                s.role.name(),
                s.repo,
                s.revision,
                s.path,
                s.bytes,
                s.sha256
            ));
        }
        out
    }

    /// SHA-256 of [`Self::canonical`], lowercase hex.
    pub fn digest(&self) -> String {
        hex(&Sha256::digest(self.canonical().as_bytes()))
    }

    /// The artifact's filename: the expert width, then the digest's tag.
    pub fn artifact_name(&self) -> String {
        format!(
            "Qwen3.8-Flash-Next-{:?}EXP-{}.gguf",
            self.experts,
            &self.digest()[..TAG_HEX]
        )
    }

    /// The sources with `role`, in path order.
    pub fn sources_of(&self, role: SourceRole) -> Vec<&SourceFile> {
        let mut v: Vec<&SourceFile> = self.sources.iter().filter(|s| s.role == role).collect();
        v.sort_by_key(|s| s.path);
        v
    }

    /// The two metadata entries the build writes into the artifact.
    pub fn stamp(&self) -> [(String, Value); 2] {
        [
            (STAMP_KEY.to_string(), Value::String(self.digest())),
            (CANONICAL_KEY.to_string(), Value::String(self.canonical())),
        ]
    }

    /// Whether `content` carries this recipe's stamp.
    pub fn stamped_in(&self, content: &Content) -> bool {
        matches!(content.metadata.get(STAMP_KEY), Some(Value::String(s)) if *s == self.digest())
    }
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Recipe {
        Recipe {
            sources: vec![
                SourceFile {
                    role: SourceRole::DraftHead,
                    repo: "org/model",
                    revision: "abc",
                    path: "MTP/head.gguf",
                    bytes: 7,
                    sha256: "11",
                },
                SourceFile {
                    role: SourceRole::Trunk,
                    repo: "org/model",
                    revision: "abc",
                    path: "Q8_0/b.gguf",
                    bytes: 3,
                    sha256: "22",
                },
                SourceFile {
                    role: SourceRole::Trunk,
                    repo: "org/model",
                    revision: "abc",
                    path: "Q8_0/a.gguf",
                    bytes: 5,
                    sha256: "33",
                },
            ],
            trunk: GgmlDType::Q8_0,
            head_dense: GgmlDType::Q8_0,
            experts: GgmlDType::Q2_KO,
            expert_source: ExpertSource::Requantized,
            head_experts: GgmlDType::Q2_KO,
            converter_version: 1,
        }
    }

    #[test]
    fn the_canonical_form_is_exact_and_sorted() {
        assert_eq!(
            fixture().canonical(),
            "converter_version=1\n\
             trunk=Q8_0\n\
             head_dense=Q8_0\n\
             experts=Q2_KO\n\
             expert_source=requantized\n\
             head_experts=Q2_KO\n\
             source=draft_head org/model@abc MTP/head.gguf 7 11\n\
             source=trunk org/model@abc Q8_0/a.gguf 5 33\n\
             source=trunk org/model@abc Q8_0/b.gguf 3 22\n"
        );
    }

    #[test]
    fn the_digest_and_name_are_pinned() {
        let r = fixture();
        assert_eq!(
            r.digest(),
            "e07110279bbbc9d38c21d3d10a49f7e74ede3c4c83d671526538605d3f12673d"
        );
        assert_eq!(
            r.artifact_name(),
            "Qwen3.8-Flash-Next-Q2_KOEXP-e07110279bbb.gguf"
        );
    }

    /// Listing order is not identity: the same sources in another order name
    /// the same artifact.
    #[test]
    fn source_order_does_not_change_the_digest() {
        let a = fixture();
        let mut b = fixture();
        b.sources.reverse();
        assert_eq!(a.digest(), b.digest());
    }

    /// Every field is part of the identity — changing any one names a
    /// different artifact, which is what makes a machine rebuild.
    #[test]
    fn every_field_changes_the_digest() {
        let base = fixture().digest();
        let variants: [fn(&mut Recipe); 13] = [
            |r| r.experts = GgmlDType::Q3_KO,
            |r| r.head_experts = GgmlDType::Q3_KO,
            |r| r.trunk = GgmlDType::Q6_K,
            |r| r.head_dense = GgmlDType::Q6_K,
            |r| r.expert_source = ExpertSource::Verbatim,
            |r| r.converter_version = 2,
            |r| r.sources[0].revision = "abd",
            |r| r.sources[0].sha256 = "12",
            |r| r.sources[0].bytes = 8,
            |r| r.sources[0].path = "MTP/other.gguf",
            |r| r.sources[0].repo = "org/other",
            |r| r.sources[0].role = SourceRole::ExpertImport,
            |r| {
                r.sources.pop();
            },
        ];
        for (i, v) in variants.iter().enumerate() {
            let mut r = fixture();
            v(&mut r);
            assert_ne!(r.digest(), base, "variant {i} left the digest unchanged");
        }
    }

    #[test]
    fn sources_of_filters_by_role_in_path_order() {
        let r = fixture();
        let trunk: Vec<&str> = r
            .sources_of(SourceRole::Trunk)
            .iter()
            .map(|s| s.path)
            .collect();
        assert_eq!(trunk, ["Q8_0/a.gguf", "Q8_0/b.gguf"]);
        assert!(r.sources_of(SourceRole::ExpertImport).is_empty());
    }

    #[test]
    fn the_stamp_carries_the_digest_and_the_text() {
        let r = fixture();
        let [(k1, v1), (k2, v2)] = r.stamp();
        assert_eq!(k1, STAMP_KEY);
        assert!(matches!(v1, Value::String(s) if s == r.digest()));
        assert_eq!(k2, CANONICAL_KEY);
        assert!(matches!(v2, Value::String(s) if s == r.canonical()));
    }

    #[test]
    fn hex_is_lowercase_and_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
    }
}
