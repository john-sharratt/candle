//! Tensors read from a second checkpoint.
//!
//! A load can name individual tensors to take from another GGUF instead of the one being
//! loaded — one fine-tune's output head over another's trunk, say. A named tensor is read from
//! the other file's mapping by every [`Loader`] read of it, the banded repack included, so the
//! substitution is a property of reading the tensor rather than of one call site.
//!
//! # What may be overridden
//!
//! A tensor the load reads through its loader, with **the same shape** as the checkpoint's own.
//! The GGML type may differ — two conversions of one base need not quantize the head alike — and
//! the tensor is read and repacked as its own file stores it: the KO twin is chosen from the
//! override's type, not the checkpoint's.
//!
//! One thing is planned from the header before a byte is read, and it has to see the override:
//! the CUDA-pool headroom, which is bounded by the largest source tensor. The load sizes it from
//! [`TensorOverrides::effective_infos`] — the checkpoint's header with each override's own entry
//! in place — so a wider head from another file is inside the bound. The resident-narrowing
//! decision is deliberately *not* re-derived: it is a threshold on the checkpoint's total weight,
//! the streamed-layer build derives it again from the same header, and the two must agree.
//!
//! Two families are refused up front, because nothing reads them through a loader and an
//! override would be silently ignored:
//!
//! * the routed experts (`ffn_{gate,up,down}_exps`), which the expert cache packs straight from
//!   the checkpoint's mapping;
//! * `token_embd.weight`, which the embedding table reads the same way — and which a checkpoint
//!   without `output.weight` also uses as its head, so an override would change the head and
//!   leave the embedding as it was.
//!
//! Anything else a particular load does not read through its loader — a streamed dense layer's
//! projections, which come out of the layer pack — is caught after the load by
//! [`TensorOverrides::ensure_all_read`], so no override is ever dropped quietly.
//!
//! [`Loader`]: super::quantized_weights::Loader

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use candle::quantized::gguf_file::{Content, TensorInfo};
use candle::quantized::GgmlDType;
use candle::Result;

/// One tensor to read from another checkpoint instead of the one being loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorOverride {
    /// The tensor's GGUF name, e.g. `output.weight`.
    pub tensor: String,
    /// The checkpoint to read it from.
    pub path: PathBuf,
}

impl TensorOverride {
    pub fn new(tensor: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            tensor: tensor.into(),
            path: path.into(),
        }
    }
}

/// Suffixes of the tensors no loader reads — see this module's header.
const NOT_LOADER_READ: [&str; 4] = [
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
    "token_embd.weight",
];

struct Source<'a> {
    spec: &'a TensorOverride,
    content: &'a Content,
    bytes: &'a [u8],
    /// The checkpoint's own type for this tensor — what the override replaces.
    replaces: GgmlDType,
    /// Set by the first read. `ensure_all_read` refuses a load that leaves any of these clear.
    read: AtomicBool,
}

/// One load's overrides, validated against the checkpoint's header.
pub struct TensorOverrides<'a> {
    sources: Vec<Source<'a>>,
}

impl<'a> TensorOverrides<'a> {
    /// Check each override against `checkpoint`'s header. Headers only — nothing is read.
    ///
    /// `sources` pairs each override with its file's header and mapped bytes.
    pub fn new(
        checkpoint: &Content,
        sources: impl IntoIterator<Item = (&'a TensorOverride, &'a Content, &'a [u8])>,
    ) -> Result<Self> {
        let mut out: Vec<Source<'a>> = Vec::new();
        for (spec, content, bytes) in sources {
            let name = spec.tensor.as_str();
            let from = spec.path.display();
            if out.iter().any(|s| s.spec.tensor == name) {
                candle::bail!("tensor override: `{name}` is named more than once");
            }
            if NOT_LOADER_READ.iter().any(|s| name.ends_with(s)) {
                candle::bail!(
                    "tensor override: `{name}` cannot be taken from another checkpoint — it is \
                     read straight from the checkpoint's own mapping (the expert pack or the \
                     embedding table), so the override would be ignored"
                );
            }
            let Some(mine) = checkpoint.tensor_infos.get(name) else {
                candle::bail!("tensor override: the checkpoint has no `{name}` to replace");
            };
            let Some(theirs) = content.tensor_infos.get(name) else {
                candle::bail!("tensor override: {from} has no `{name}`");
            };
            if theirs.shape.dims() != mine.shape.dims() {
                candle::bail!(
                    "tensor override: {from}'s `{name}` is {:?} where the checkpoint's is {:?} — \
                     these are different models, not two conversions of one",
                    theirs.shape.dims(),
                    mine.shape.dims()
                );
            }
            out.push(Source {
                spec,
                content,
                bytes,
                replaces: mine.ggml_dtype,
                read: AtomicBool::new(false),
            });
        }
        Ok(Self { sources: out })
    }

    /// Whether `tensor` comes from another file. Does not count as reading it.
    pub(crate) fn names(&self, tensor: &str) -> bool {
        self.sources.iter().any(|s| s.spec.tensor == tensor)
    }

    /// The header and bytes of the file supplying `tensor`, if one does — recorded as read.
    pub(crate) fn take(&self, tensor: &str) -> Option<(&'a Content, &'a [u8])> {
        let s = self.sources.iter().find(|s| s.spec.tensor == tensor)?;
        s.read.store(true, Ordering::Relaxed);
        Some((s.content, s.bytes))
    }

    /// Refuse a load that never read one of its overrides.
    ///
    /// Called once the model is built. An unread override means the tensor reached the model
    /// by a route that reads the checkpoint's own copy, so the model on the card is not the one
    /// that was asked for.
    pub fn ensure_all_read(&self) -> Result<()> {
        let unread: Vec<String> = self
            .sources
            .iter()
            .filter(|s| !s.read.load(Ordering::Relaxed))
            .map(|s| format!("`{}` from {}", s.spec.tensor, s.spec.path.display()))
            .collect();
        if unread.is_empty() {
            return Ok(());
        }
        candle::bail!(
            "tensor override: {} never read — this load builds those tensors by a path that reads \
             the checkpoint's own copy (a streamed layer's projections come from the layer \
             pack), so the model would have run without them",
            unread.join(", ")
        )
    }

    /// The checkpoint's header entries with each override's own in place — what this load
    /// will actually read, for anything that plans from sizes before reading.
    pub fn effective_infos<'c>(
        &'c self,
        checkpoint: &'c Content,
    ) -> impl Iterator<Item = &'c TensorInfo> + 'c {
        checkpoint.tensor_infos.iter().map(move |(name, info)| {
            self.sources
                .iter()
                .find(|s| s.spec.tensor == *name)
                .and_then(|s| s.content.tensor_infos.get(name))
                .unwrap_or(info)
        })
    }

    /// Each override as `(tensor, file, its type, the checkpoint's type)`, for the load's log.
    pub fn describe(&self) -> Vec<(&'a str, &'a Path, GgmlDType, GgmlDType)> {
        self.sources
            .iter()
            .filter_map(|s| {
                let name = s.spec.tensor.as_str();
                let theirs = s.content.tensor_infos.get(name)?.ggml_dtype;
                Some((name, s.spec.path.as_path(), theirs, s.replaces))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::{TensorInfo, VersionedMagic};
    use candle::quantized::GgmlDType;
    use candle::Shape;
    use std::collections::HashMap;

    fn header(tensors: &[(&str, GgmlDType, (usize, usize))]) -> Content {
        let tensor_infos = tensors
            .iter()
            .map(|&(name, ggml_dtype, dims)| {
                (
                    name.to_string(),
                    TensorInfo {
                        ggml_dtype,
                        shape: Shape::from(dims),
                        offset: 0,
                    },
                )
            })
            .collect();
        Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos,
            tensor_data_offset: 0,
        }
    }

    const HEAD: (&str, GgmlDType, (usize, usize)) =
        ("output.weight", GgmlDType::Q6_K, (248_320, 2048));

    fn head_override() -> TensorOverride {
        TensorOverride::new("output.weight", "styletune.gguf")
    }

    #[test]
    fn a_matching_head_is_handed_out_and_counted_as_read() -> Result<()> {
        let checkpoint = header(&[HEAD]);
        let donor = header(&[HEAD]);
        let spec = head_override();
        let bytes = [7u8; 4];
        let o = TensorOverrides::new(&checkpoint, [(&spec, &donor, &bytes[..])])?;

        assert!(o.names("output.weight"));
        assert!(!o.names("output_norm.weight"));
        // Asking is not reading.
        assert!(o.ensure_all_read().is_err());

        let (content, got) = o.take("output.weight").expect("the head is overridden");
        assert!(
            std::ptr::eq(content, &donor),
            "read from the donor's header"
        );
        assert_eq!(got, &bytes[..], "and from the donor's bytes");
        assert!(o.take("output_norm.weight").is_none());
        o.ensure_all_read()
    }

    #[test]
    fn an_unread_override_fails_the_load_and_names_itself() -> Result<()> {
        let checkpoint = header(&[HEAD]);
        let donor = header(&[HEAD]);
        let spec = head_override();
        let o = TensorOverrides::new(&checkpoint, [(&spec, &donor, &[][..])])?;
        let err = o.ensure_all_read().unwrap_err().to_string();
        assert!(err.contains("`output.weight` from styletune.gguf"), "{err}");
        Ok(())
    }

    /// **Two conversions of one base need not quantize the head alike**, so a head of another
    /// type is taken — and the pool headroom the load claims before reading is bounded by the
    /// head it will actually read, not the one the checkpoint's header names.
    #[test]
    fn a_wider_head_is_accepted_and_bounds_the_load_headroom() -> Result<()> {
        use crate::models::dense_span::peak_load_pool_bytes;
        use candle::quantized::cuda::REPACK_BAND_BYTES;

        let checkpoint = header(&[HEAD, ("output_norm.weight", GgmlDType::F32, (1, 2048))]);
        let donor = header(&[("output.weight", GgmlDType::Q8_0, (248_320, 2048))]);
        let spec = head_override();
        let o = TensorOverrides::new(&checkpoint, [(&spec, &donor, &[][..])])?;

        let described = o.describe();
        assert_eq!(described.len(), 1);
        assert_eq!(described[0].2, GgmlDType::Q8_0, "the override's own type");
        assert_eq!(described[0].3, GgmlDType::Q6_K, "the type it replaces");

        // 248,320 × 2,048 elements: Q6_K is 210 B per 256, Q8_0 is 34 B per 32.
        assert_eq!(
            peak_load_pool_bytes(checkpoint.tensor_infos.values()),
            417_177_600 + 2 * REPACK_BAND_BYTES
        );
        assert_eq!(
            peak_load_pool_bytes(o.effective_infos(&checkpoint)),
            540_344_320 + 2 * REPACK_BAND_BYTES
        );
        // Every other entry is the checkpoint's own.
        assert_eq!(o.effective_infos(&checkpoint).count(), 2);
        Ok(())
    }

    #[test]
    fn a_different_shape_is_refused() {
        let checkpoint = header(&[HEAD]);
        let donor = header(&[("output.weight", GgmlDType::Q6_K, (151_936, 2048))]);
        let spec = head_override();
        assert!(TensorOverrides::new(&checkpoint, [(&spec, &donor, &[][..])]).is_err());
    }

    #[test]
    fn a_tensor_missing_from_either_file_is_refused() {
        let with = header(&[HEAD]);
        let without = header(&[("output_norm.weight", GgmlDType::F32, (1, 2048))]);
        let spec = head_override();
        let err = TensorOverrides::new(&without, [(&spec, &with, &[][..])])
            .err()
            .expect("nothing to replace");
        assert!(
            err.to_string().contains("no `output.weight` to replace"),
            "{err}"
        );
        let err = TensorOverrides::new(&with, [(&spec, &without, &[][..])])
            .err()
            .expect("nothing to read");
        assert!(err.to_string().contains("styletune.gguf has no"), "{err}");
    }

    #[test]
    fn tensors_no_loader_reads_are_refused_up_front() {
        for name in ["blk.3.ffn_up_exps.weight", "token_embd.weight"] {
            let t = [(name, GgmlDType::Q4_K, (256, 2048))];
            let (checkpoint, donor) = (header(&t), header(&t));
            let spec = TensorOverride::new(name, "styletune.gguf");
            let err = TensorOverrides::new(&checkpoint, [(&spec, &donor, &[][..])])
                .err()
                .unwrap_or_else(|| panic!("`{name}` would be ignored, so it is refused"));
            assert!(err.to_string().contains("would be ignored"), "{err}");
        }
    }

    #[test]
    fn a_tensor_named_twice_is_refused() {
        let checkpoint = header(&[HEAD]);
        let donor = header(&[HEAD]);
        let (a, b) = (head_override(), head_override());
        let sources = [(&a, &donor, &[][..]), (&b, &donor, &[][..])];
        assert!(TensorOverrides::new(&checkpoint, sources).is_err());
    }
}
