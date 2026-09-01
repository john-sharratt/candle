//! Comparing two GGUF checkpoints structurally, without loading either.
//!
//! # Why this exists
//!
//! Three third-party conversions of Qwen3.5-9B were tried in this engine and all three
//! generated garbage — stubs, word salad, and the prompt echoed back — while the reference
//! conversion generated correctly. Judging each one by loading it and reading its prose cost
//! half an hour, most of it download, and said only that something was wrong.
//!
//! This reads the header instead — seconds, no weights — and says *what differs*. It is what
//! found the answer: one row of the dtype table, `blk.N.ssm_alpha.weight: F32 vs BF16`, which
//! is the DeltaNet recurrent gate the architecture cannot tolerate quantized
//! (`check_recurrent_precision`). Every other difference between those files — the older chat
//! template, the eos and padding ids, the missing metadata keys — was a dead end, and the
//! quickest way to see that is still to print them all and watch them not matter.
//!
//! # What is compared, and why only these
//!
//! Everything the loader's correctness depends on and nothing that is merely descriptive:
//! the metadata the config is derived from, and the tensor inventory the weights are read
//! by. A checkpoint that agrees with the reference on both of those and still fails is a
//! bug below this layer; one that disagrees has told you where to look.
//!
//! Free text — `general.name`, descriptions, quantisation notes — is deliberately excluded.
//! It always differs and drowns the signal.

use std::collections::BTreeMap;
use std::path::Path;

use candle::quantized::gguf_file::{Content, Value};
use candle::Result;

/// One structural difference between two checkpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Difference {
    /// A metadata key one has and the other does not.
    MetadataOnlyIn { side: &'static str, key: String },
    /// A shared metadata key whose values differ.
    MetadataValue {
        key: String,
        left: String,
        right: String,
    },
    /// A tensor one has and the other does not.
    TensorOnlyIn { side: &'static str, name: String },
    /// A shared tensor whose shape differs — the loader reads by shape, so this is fatal.
    TensorShape {
        name: String,
        left: String,
        right: String,
    },
    /// A shared tensor stored in a different format. Expected between quantisations and
    /// reported separately from shape for that reason: it is usually benign, and a *lot* of
    /// them at once is the interesting case.
    TensorDtype {
        name: String,
        left: String,
        right: String,
    },
}

impl Difference {
    pub fn line(&self) -> String {
        match self {
            Difference::MetadataOnlyIn { side, key } => format!("metadata only in {side}: {key}"),
            Difference::MetadataValue { key, left, right } => {
                format!("metadata {key}: {left} vs {right}")
            }
            Difference::TensorOnlyIn { side, name } => format!("tensor only in {side}: {name}"),
            Difference::TensorShape { name, left, right } => {
                format!("tensor {name} shape: {left} vs {right}")
            }
            Difference::TensorDtype { name, left, right } => {
                format!("tensor {name} dtype: {left} vs {right}")
            }
        }
    }
}

/// Metadata keys excluded from the comparison: free text that always differs and says
/// nothing about whether the file can be read.
fn is_descriptive(key: &str) -> bool {
    const NOISE: &[&str] = &[
        "general.name",
        "general.basename",
        "general.description",
        "general.license",
        "general.organization",
        "general.finetune",
        "general.size_label",
        "general.quantization_version",
        "general.file_type",
        "general.url",
        "general.repo_url",
        "general.base_model.count",
        "quantize.imatrix.file",
        "quantize.imatrix.dataset",
        "quantize.imatrix.entries_count",
        "quantize.imatrix.chunks_count",
    ];
    NOISE.contains(&key)
        || key.starts_with("general.base_model.")
        || key.starts_with("general.tags")
        || key.starts_with("general.languages")
        || key.starts_with("general.datasets")
}

/// Render a metadata value compactly. Long arrays — the token tables — are reduced to their
/// length, because a 248,320-entry vocabulary printed in a diff is not a diff.
fn render(v: &Value) -> String {
    match v {
        Value::String(s) if s.len() > 80 => format!("<string, {} chars>", s.len()),
        Value::String(s) => format!("{s:?}"),
        Value::Array(a) => format!("<array, {} entries>", a.len()),
        other => format!("{other:?}"),
    }
}

/// The chat template a checkpoint carries, if it carries one.
///
/// This is the one long string [`render`] deliberately reduces to a length, and the one place
/// where "they differ by 236 characters" is not an answer: the template decides how a turn is
/// framed, so a checkpoint whose template disagrees with the framing the engine emits sees a
/// malformed prompt and answers accordingly.
pub fn chat_template(c: &Content) -> Option<&str> {
    match c.metadata.get("tokenizer.chat_template")? {
        Value::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Read a checkpoint's header.
pub fn header(path: &Path) -> Result<Content> {
    let mut f = std::fs::File::open(path)?;
    Content::read(&mut f)
}

/// Every structural difference between two checkpoints, in a stable order.
pub fn diff(left: &Content, right: &Content) -> Vec<Difference> {
    let mut out = Vec::new();

    let lm: BTreeMap<&String, &Value> = left
        .metadata
        .iter()
        .filter(|(k, _)| !is_descriptive(k))
        .collect();
    let rm: BTreeMap<&String, &Value> = right
        .metadata
        .iter()
        .filter(|(k, _)| !is_descriptive(k))
        .collect();

    for k in lm.keys() {
        if !rm.contains_key(*k) {
            out.push(Difference::MetadataOnlyIn {
                side: "left",
                key: (*k).clone(),
            });
        }
    }
    for k in rm.keys() {
        if !lm.contains_key(*k) {
            out.push(Difference::MetadataOnlyIn {
                side: "right",
                key: (*k).clone(),
            });
        }
    }
    for (k, lv) in &lm {
        if let Some(rv) = rm.get(*k) {
            let (a, b) = (render(lv), render(rv));
            if a != b {
                out.push(Difference::MetadataValue {
                    key: (*k).clone(),
                    left: a,
                    right: b,
                });
            }
        }
    }

    let lt: BTreeMap<&String, _> = left.tensor_infos.iter().collect();
    let rt: BTreeMap<&String, _> = right.tensor_infos.iter().collect();
    for n in lt.keys() {
        if !rt.contains_key(*n) {
            out.push(Difference::TensorOnlyIn {
                side: "left",
                name: (*n).clone(),
            });
        }
    }
    for n in rt.keys() {
        if !lt.contains_key(*n) {
            out.push(Difference::TensorOnlyIn {
                side: "right",
                name: (*n).clone(),
            });
        }
    }
    for (n, li) in &lt {
        if let Some(ri) = rt.get(*n) {
            if li.shape.dims() != ri.shape.dims() {
                out.push(Difference::TensorShape {
                    name: (*n).clone(),
                    left: format!("{:?}", li.shape.dims()),
                    right: format!("{:?}", ri.shape.dims()),
                });
            }
            if li.ggml_dtype != ri.ggml_dtype {
                out.push(Difference::TensorDtype {
                    name: (*n).clone(),
                    left: format!("{:?}", li.ggml_dtype),
                    right: format!("{:?}", ri.ggml_dtype),
                });
            }
        }
    }
    out
}

/// A one-screen summary of a checkpoint: what the loader will derive its config from.
pub fn summary(c: &Content) -> String {
    let get = |k: &str| c.metadata.get(k).map(render).unwrap_or_else(|| "-".into());
    let arch = c
        .metadata
        .get("general.architecture")
        .map(render)
        .unwrap_or_else(|| "-".into());
    // The same value unquoted: `render` is for display and wraps strings in quotes, which is
    // right in the header line and wrong as a metadata key prefix.
    let arch_key = c
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_default();
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    for i in c.tensor_infos.values() {
        *dtypes.entry(format!("{:?}", i.ggml_dtype)).or_default() += 1;
    }
    format!(
        "arch={arch} tensors={} dtypes={dtypes:?}\n  \
         block_count={} nextn={} vocab={} ctx={}\n  \
         bos={} eos={} eot={} pad={} chat_template={}",
        c.tensor_infos.len(),
        // **Keyed off the file's own arch string, not a literal.** These were spelled
        // `qwen3.5.*` — a plausible-looking key that no checkpoint carries — so every summary
        // printed `block_count=- nextn=- vocab=- ctx=-` and read as "this file declares
        // nothing", while `diff` was reporting `qwen35.block_count: U32(33) vs U32(32)` from
        // the same header a few lines later. Deriving the prefix cannot drift from the file.
        get(&format!("{arch_key}.block_count")),
        get(&format!("{arch_key}.nextn_predict_layers")),
        get(&format!("{arch_key}.vocab_size")),
        get(&format!("{arch_key}.context_length")),
        get("tokenizer.ggml.bos_token_id"),
        get("tokenizer.ggml.eos_token_id"),
        get("tokenizer.ggml.eot_token_id"),
        get("tokenizer.ggml.padding_token_id"),
        get("tokenizer.chat_template"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Find a cached GGUF by repo, without touching the network.
    ///
    /// The hub layout, not a guess: `hub/models--{org}--{name}/snapshots/{rev}/{file}`.
    /// Returns `None` when the checkpoint is not on this machine, which is the ordinary
    /// case on any box that has not fetched it.
    fn cached(repo: &str, suffix: &str) -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HF_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| dirs_home().map(|h| h.join(".cache").join("huggingface")))?;
        let dir = home
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots");
        for rev in std::fs::read_dir(dir).ok()?.flatten() {
            for f in std::fs::read_dir(rev.path()).ok()?.flatten() {
                let p = f.path();
                if p.to_str().is_some_and(|s| s.ends_with(suffix)) {
                    return Some(p);
                }
            }
        }
        None
    }

    fn dirs_home() -> Option<std::path::PathBuf> {
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(std::path::PathBuf::from)
    }

    /// **What differs between a checkpoint that generates and one that does not.**
    ///
    /// Reads two headers and prints the structural delta. Header-only, so it costs
    /// milliseconds and needs no GPU — which is the point: judging a candidate by loading it
    /// and reading its prose costs half an hour, most of it download.
    ///
    /// Skips when either file is absent rather than failing, because it names two specific
    /// checkpoints and most machines have neither.
    #[test]
    fn reference_and_candidate_checkpoints_agree_structurally() -> Result<()> {
        const REFERENCE: (&str, &str) = ("unsloth/Qwen3.5-9B-MTP-GGUF", "Q6_K.gguf");
        const CANDIDATES: &[(&str, &str)] = &[
            (
                "llmfan46/Qwen3.5-9B-Nikusui-v1-Uncensored-Heretic-Native-MTP-Preserved-GGUF",
                "Q6_K.gguf",
            ),
            (
                "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
                "Q6_K.gguf",
            ),
            (
                "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF",
                "Q6_K.gguf",
            ),
        ];

        let Some(a) = cached(REFERENCE.0, REFERENCE.1) else {
            println!("the reference checkpoint is not cached on this machine — nothing to diff");
            return Ok(());
        };
        let l = header(&a)?;
        println!("\nREFERENCE {}\n  {}\n", REFERENCE.0, summary(&l));
        for cand in CANDIDATES {
            let Some(p) = cached(cand.0, cand.1) else {
                println!("\n{}: not cached", cand.0);
                continue;
            };
            println!("\n════ {} ════", cand.0);
            diff_report(&l, &header(&p)?)?;
        }
        Ok(())
    }

    /// The per-candidate half of the report above.
    fn diff_report(l: &Content, r: &Content) -> Result<()> {
        println!("  {}\n", summary(r));
        let (l, r) = (l, r);

        let d = diff(&l, &r);
        let (structural, dtypes): (Vec<_>, Vec<_>) = d
            .iter()
            .partition(|x| !matches!(x, Difference::TensorDtype { .. }));

        println!("── structural differences ({}) ──", structural.len());
        for x in &structural {
            println!("  {}", x.line());
        }
        // **Is it the ids, or the whole table?** A checkpoint that merely numbers its
        // special tokens differently is one stop-token fix away. A checkpoint whose
        // vocabulary is genuinely different decodes every token slightly wrong and no stop
        // token will save it. Both embed their table, so this is answerable here.
        let toks = |c: &Content| match c.metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.to_string().ok().cloned().unwrap_or_default())
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        let (lt, rt) = (toks(&l), toks(&r));
        println!("── token tables ──");
        println!("  lengths: {} vs {}", lt.len(), rt.len());
        if !lt.is_empty() && !rt.is_empty() {
            let n = lt.len().min(rt.len());
            let mismatched = (0..n).filter(|i| lt[*i] != rt[*i]).count();
            println!(
                "  differing entries over the shared {n}: {mismatched} ({:.4}%)",
                mismatched as f64 * 100.0 / n as f64
            );
            for i in (0..n).filter(|i| lt[*i] != rt[*i]).take(6) {
                println!("    [{i}] {:?} vs {:?}", lt[i], rt[i]);
            }
            // The ids each side calls end-of-turn, and what the *other* side has there.
            for (label, id) in [("ref eos", 248046usize), ("cand eos", 248044)] {
                let a = lt.get(id).map(String::as_str).unwrap_or("<oob>");
                let b = rt.get(id).map(String::as_str).unwrap_or("<oob>");
                println!("    {label} id {id}: reference={a:?} candidate={b:?}");
            }
        }

        // **The framing, and why its size is not the interesting part.** These templates differ
        // by hundreds of characters — the third-party conversions ship the older Qwen3 one — and
        // it looks like a lead until you check what reads it. `builder.rs` never *runs* the
        // template: it substring-tests it for exactly the two facts below and takes the turn
        // format from the preset's dialect. Two templates that agree on both differ in ways
        // nothing consults, however many lines apart they are.
        match (chat_template(l), chat_template(r)) {
            (Some(a), Some(b)) if a == b => println!("── chat template ── identical"),
            (Some(a), Some(b)) => println!("── chat template ── {} vs {} chars", a.len(), b.len()),
            (a, b) => println!(
                "── chat template ── present: reference={} candidate={}",
                a.is_some(),
                b.is_some()
            ),
        }

        // **What the engine actually reads out of the template.** `builder.rs` does not run
        // the template — it derives exactly two things from it, by substring. A template that
        // differs in 44 lines but agrees on both of these differs in ways nothing consults,
        // and the difference is a dead end no matter how large it looks.
        println!("── derived signals (what builder.rs reads) ──");
        for (label, needle) in [
            ("thinking", "<think>"),
            ("ChatML", "<|im_start|>"),
            ("Llama3", "<|start_header_id|>"),
            ("Llama2", "[INST]"),
        ] {
            let a = chat_template(&l).is_some_and(|t| t.contains(needle));
            let b = chat_template(&r).is_some_and(|t| t.contains(needle));
            let flag = if a == b { "  " } else { "!!" };
            println!("  {flag} {label:<9} reference={a:<5} candidate={b}");
        }

        // **Grouped, because the outlier is the finding.** Printing the first twelve of
        // seventy-three showed twelve `ssm_*` rows and hid the fact that one tensor differed
        // for some other reason entirely — and a lone row in a table of near-identical ones is
        // exactly what a truncated list buries. Rolling each tensor name up to its family also
        // turns "73 differences" into the handful of *kinds* of difference actually present.
        println!("── dtype differences ({}) ──", dtypes.len());
        let mut kinds: BTreeMap<(String, String, String), Vec<String>> = BTreeMap::new();
        for x in &dtypes {
            if let Difference::TensorDtype { name, left, right } = x {
                // `blk.17.ssm_out.weight` → `blk.N.ssm_out.weight`, so the layers collapse.
                let family = name
                    .split('.')
                    .map(|p| {
                        if p.chars().all(|c| c.is_ascii_digit()) {
                            "N"
                        } else {
                            p
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(".");
                kinds
                    .entry((family, left.clone(), right.clone()))
                    .or_default()
                    .push(name.clone());
            }
        }
        for ((family, left, right), names) in &kinds {
            println!(
                "  {family}: {left} vs {right}  ×{}{}",
                names.len(),
                if names.len() == 1 {
                    format!("   ← the only one: {}", names[0])
                } else {
                    String::new()
                }
            );
        }
        Ok(())
    }
}
