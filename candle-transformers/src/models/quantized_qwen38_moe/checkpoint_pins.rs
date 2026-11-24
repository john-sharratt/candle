//! The pinned Qwen3.8-Flash-Next checkpoints and their revisions.
//!
//! The bring-up checkpoint is the Q8_0 split, pinned by revision with each of its
//! six shards and their LFS digests. The W4A16 release is the source of the Q4_KO
//! expert import, which is bit-exact against that release's own quantisation. The
//! pinned revision is the one that carries the draft head, and it holds the same
//! Q8_0 shard objects as the earlier pin, so moving to it changed no weight. The
//! tokenizer is pinned to the canonical base repository.

use crate::models::qwen_vision::source::VisionTowerSource;

/// The tokenizer, pinned to the canonical base repo.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.8-Flash-Next";

pub const TOKENIZER_REV: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";

/// The bring-up checkpoint: unsloth's **plain Q8_0** split, revision-pinned.
///
/// Q8_0 rather than a Q4-class quant, and that is a measurement, not a
/// preference: both community sub-8-bit conversions (unsloth `UD-*`,
/// AtomicChat `AD-*`) carry IQ-family tensors this codebase cannot read —
/// verified from the shard headers, dtype codes 20/21, exactly the trap
/// `quantized_qwen38.rs` documents as "no rung is a `UD-` file". The §0.2
/// per-machine expert formats are produced by requantizing locally from this
/// file. Six shards; shard 1 is metadata-only, shard 3 is the isolated PLE
/// table.
///
/// The revision is the one that also carries the MTP draft head (`MTP/`). The
/// six `Q8_0` shards are byte-identical to the earlier `c8b5954a` pin — the same
/// LFS object ids — so moving the pin changed no weight.
pub const QWEN4EXP_REPO: &str = "unsloth/Qwen3.8-Flash-Next-GGUF";

pub const QWEN4EXP_REV: &str = "38bb39ee97821de2c9009abb7e93950eec396e66";

pub const QWEN4EXP_Q8_0_SHARDS: usize = 6;

/// The six `Q8_0` shards at [`QWEN4EXP_REV`]: `(path, bytes, LFS SHA-256)`.
pub const QWEN4EXP_Q8_0_FILES: [(&str, u64, &str); QWEN4EXP_Q8_0_SHARDS] = [
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
        10_946_624,
        "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf",
        682_434_912,
        "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf",
        54_400_261_312,
        "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf",
        49_446_841_216,
        "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf",
        49_668_930_400,
        "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97",
    ),
    (
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf",
        34_015_618_784,
        "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5",
    ),
];

/// The MTP draft head at [`QWEN4EXP_REV`], at `Q8_0`: `(path, bytes, LFS SHA-256)`.
/// Its dense weights stay `Q8_0` in the engine artifact; its experts take the
/// trunk's width (`quant_ladder::drafter_format`).
pub const QWEN4EXP_MTP_FILE: (&str, u64, &str) = (
    "MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
    4_137_429_120,
    "cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e",
);

/// The revision the vision tower is pinned at: the commit of the same
/// repository its `mmproj` header was read and its digest taken at.
pub const QWEN4EXP_MMPROJ_REV: &str = "f5d175de2cf7f720d45d6064019a81fcf056e986";

/// The vision tower at [`QWEN4EXP_MMPROJ_REV`], F16: `(path, bytes, LFS SHA-256)`.
/// llama.cpp's `mmproj` for this model — an unmodified Qwen3-VL ViT and merger
/// (`docs/flash_next_vision.md` §1.2).
pub const QWEN4EXP_MMPROJ_FILE: (&str, u64, &str) = (
    "mmproj-F16.gguf",
    904_004_000,
    "1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980",
);

/// Flash-Next's vision tower (`docs/vision_models.md` §2.1): the pinned
/// `mmproj`, its rows standing for this tokenizer's `<|image_pad|>`s.
pub const FLASH_NEXT_VISION: VisionTowerSource = VisionTowerSource {
    name: "qwen3.8-flash-next",
    repo: QWEN4EXP_REPO,
    rev: QWEN4EXP_MMPROJ_REV,
    file: QWEN4EXP_MMPROJ_FILE,
    tokenizer: (TOKENIZER_REPO, TOKENIZER_REV),
};

/// Shard `i` (1-based) of the pinned Q8_0 split.
pub fn q8_0_shard_name(i: usize) -> String {
    format!("Q8_0/Qwen3.8-Flash-Next-Q8_0-{i:05}-of-{QWEN4EXP_Q8_0_SHARDS:05}.gguf")
}

/// The W4A16 source for the Q4_KO expert import (`qwen4exp::convert`):
/// compressed-tensors int4, group_size 128, symmetric — the same per-128
/// affine quantization Q4_KO stores, so the import is bit-exact and carries
/// the release's AWQ calibration into our resident format. Only shards 2–5
/// hold the quantized trunk; shard 1 is the BF16 PLE table + embeddings and
/// is never fetched.
pub const QWEN4EXP_W4A16_REPO: &str = "wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16";

pub const QWEN4EXP_W4A16_REV: &str = "0939125b929543a783ce700c90e36dd1a575c00c";

pub const QWEN4EXP_W4A16_EXPERT_SHARDS: [usize; 4] = [2, 3, 4, 5];

/// Safetensors shard `i` of the W4A16 release.
pub fn w4a16_shard_name(i: usize) -> String {
    format!("model-{i:05}-of-00005.safetensors")
}

/// The W4A16 expert shards ([`QWEN4EXP_W4A16_EXPERT_SHARDS`]) at
/// [`QWEN4EXP_W4A16_REV`]: `(path, bytes, LFS SHA-256)`.
pub const QWEN4EXP_W4A16_FILES: [(&str, u64, &str); 4] = [
    (
        "model-00002-of-00005.safetensors",
        20_007_503_112,
        "dcb830243a6f1f56f8849cdc829724383c3fa104ba793e7f879a19927d4e8a32",
    ),
    (
        "model-00003-of-00005.safetensors",
        19_960_933_248,
        "00b849022ecb3fae6c8d6cdde768da899abb47c25bfc44d87e8c931f11b124cb",
    ),
    (
        "model-00004-of-00005.safetensors",
        20_008_579_224,
        "af83800aeef9ee47e49fc0a9769f67617108934626e79a2e30051ef48f338a7d",
    ),
    (
        "model-00005-of-00005.safetensors",
        13_134_032_728,
        "b4e2d8c38b466774703b831ccbfc89f1c78e726d2296bb7eaaf2d3b31933fadd",
    ),
];
