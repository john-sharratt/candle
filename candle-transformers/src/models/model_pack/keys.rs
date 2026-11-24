//! The metadata keys a model pack adds to its GGUF part.
//!
//! Every key is under `zen.`, which no published GGUF uses, so a source
//! checkpoint's own metadata passes through beside them untouched.

/// The model pack format's version. A reader refuses any other: a mismatch
/// rebuilds the pack, which is the whole of the upgrade path.
pub const VERSION_KEY: &str = "zen.pack.version";
pub const VERSION: u32 = 1;

/// The int8 mode the pack's repacked sections target, as `Int8Mode as u32`.
pub const INT8_MODE: &str = "zen.pack.int8_mode";

/// The layer stream's narrowing — the trunk depth its schedule was planned
/// against — or `0` for none. It changes record widths, so it is part of the
/// pack's identity.
pub const NARROW: &str = "zen.pack.narrow";

/// Every tensor the checkpoint carries, in bytes — the experts and streamed
/// projections the GGUF part no longer holds included. What a loader judges
/// "does this model fit the card" against, without the checkpoint.
pub const CHECKPOINT_BYTES: &str = "zen.pack.checkpoint_bytes";

/// The GGUF part's length: where the sections begin.
pub const GGUF_LEN: &str = "zen.pack.gguf_len";

/// The expert section's offset and length in the file, when the model routes.
pub const EXPERTS_OFFSET: &str = "zen.pack.experts.offset";
pub const EXPERTS_LEN: &str = "zen.pack.experts.len";

/// The layer section's offset and length in the file, when the model streams.
pub const LAYERS_OFFSET: &str = "zen.pack.layers.offset";
pub const LAYERS_LEN: &str = "zen.pack.layers.len";

/// The tokenizer — `tokenizer.json`'s text — and where it came from.
pub const TOKENIZER_JSON: &str = "zen.tokenizer.json";
pub const TOKENIZER_REPO: &str = "zen.tokenizer.repo";
pub const TOKENIZER_REV: &str = "zen.tokenizer.rev";

/// The KO twins a twinned pack's GGUF part stores: `zen.pack.twins.count`,
/// then `zen.pack.twins.{i}.{src,dst,fp}` for each `(source → twin)` pair —
/// the two dtypes as `GgmlDType::to_u32`, and the pair's fingerprint
/// (`model_pack::twins`).
pub const TWIN_COUNT: &str = "zen.pack.twins.count";

/// The key of twin pair `i`'s `field`.
pub fn twin_key(i: usize, field: &str) -> String {
    format!("zen.pack.twins.{i}.{field}")
}

/// The source files the pack was built from: `zen.source.count`, then
/// `zen.source.{i}.{role,repo,rev,file,len,sha256}` for each.
pub const SOURCE_COUNT: &str = "zen.source.count";

/// The key of source `i`'s `field`.
pub fn source_key(i: usize, field: &str) -> String {
    format!("zen.source.{i}.{field}")
}
