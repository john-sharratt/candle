//! What quantization a GGUF actually holds, per tensor class.
//!
//! A checkpoint's *name* is a summary, not a manifest: a file called `Q8_0`
//! routinely mixes widths, and a converted artifact mixes them by construction —
//! this engine's own merged file carries pre-repacked `Q4_KO` experts beside
//! whatever the source shipped for everything else.
//!
//! The distinction decides real things. `Int8Mode::Precision` and
//! `Int8Mode::Performance` differ **only** in the weight twin `GgmlDType::to_ko`
//! picks, and that table is same-width at both ends of the ladder: an 8-bit
//! source takes `Q8_KO` in either mode, and a tensor already in a KO layout is
//! returned unchanged. So on a checkpoint that is all-8-bit plus repacked
//! experts, choosing between the modes changes nothing at all — and the way to
//! know is to read the file rather than its filename.
//!
//! Usage:
//!   cargo run -p candle-transformers --example gguf_dtypes \
//!       --features cuda --release -- <path-to.gguf> [more.gguf ...]

fn main() -> candle::Result<()> {
    use candle::quantized::gguf_file::Content;
    use std::collections::BTreeMap;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gguf_dtypes <path-to.gguf> [more.gguf ...]");
        return Ok(());
    }
    for path in &args {
        let mut f = std::fs::File::open(path)?;
        let content = Content::read(&mut f)?;
        println!("\n=== {path} ===");
        for key in [
            "general.architecture",
            "qwen4exp.block_count",
            "qwen4exp.nextn_predict_layers",
            "qwen4exp.expert_count",
        ] {
            if let Some(v) = content.metadata.get(key) {
                println!("  {key} = {v:?}");
            }
        }

        // Grouped by dtype, because the question is "what widths are in here"
        // rather than "what is tensor 417". Bytes come from the block
        // accounting, so the totals are what the loader will actually place.
        let mut by_dtype: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        // And by role, since the modes only bite on the non-expert weights.
        let mut experts = (0usize, 0usize);
        let mut other = (0usize, 0usize);
        for (name, info) in content.tensor_infos.iter() {
            let bytes = info.shape.elem_count() / info.ggml_dtype.block_size()
                * info.ggml_dtype.type_size();
            let e = by_dtype
                .entry(format!("{:?}", info.ggml_dtype))
                .or_insert((0, 0));
            e.0 += 1;
            e.1 += bytes;
            let slot = if name.contains("_exps") {
                &mut experts
            } else {
                &mut other
            };
            slot.0 += 1;
            slot.1 += bytes;
        }
        println!("  tensors: {}", content.tensor_infos.len());
        for (dt, (n, bytes)) in &by_dtype {
            println!(
                "    {dt:<10} {n:>5} tensors  {:>9.2} GiB",
                *bytes as f64 / (1u64 << 30) as f64
            );
        }
        println!(
            "  experts: {} tensors {:.2} GiB | everything else: {} tensors {:.2} GiB",
            experts.0,
            experts.1 as f64 / (1u64 << 30) as f64,
            other.0,
            other.1 as f64 / (1u64 << 30) as f64
        );

        // The actual question: does the Int8Mode choice move any weight?
        // A source already in a KO layout is returned unchanged by `to_ko`, and
        // an 8-bit source maps to `Q8_KO` in both modes — so only widths
        // between those two ends can differ.
        use candle::quantized::{GgmlDType, Int8Mode};
        let mut movable = 0usize;
        let mut movable_bytes = 0usize;
        for info in content.tensor_infos.values() {
            let dt = info.ggml_dtype;
            let (perf, prec) = (
                dt.to_ko(Int8Mode::Performance),
                dt.to_ko(Int8Mode::Precision),
            );
            if let (Ok(a), Ok(b)) = (perf, prec) {
                if a != b {
                    movable += 1;
                    movable_bytes += info.shape.elem_count() / dt.block_size() * dt.type_size();
                }
            }
        }
        let _ = GgmlDType::Q8_0;
        println!(
            "  tensors whose twin DIFFERS between Performance and Precision: {movable} \
             ({:.2} GiB) — zero means the mode is a no-op for this file",
            movable_bytes as f64 / (1u64 << 30) as f64
        );
    }
    Ok(())
}
