//! Every Qwen tokenizer in this machine's Hugging Face hub cache — real model
//! files, not fixtures — for the tests that must hold on a production
//! vocabulary. A test over them skips, loudly, when there is none.

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

/// Every Qwen tokenizer in the HF hub cache, as `(model name, path)`.
pub fn find_tokenizers() -> Vec<(String, PathBuf)> {
    let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) else {
        return Vec::new();
    };
    let hub = Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("hub");
    let Ok(models) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for m in models.flatten() {
        let name = m.file_name().to_string_lossy().to_string();
        if !name.contains("Qwen") {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(m.path().join("snapshots")) else {
            continue;
        };
        for s in entries.flatten() {
            let p = s.path().join("tokenizer.json");
            if p.is_file() {
                out.push((name.trim_start_matches("models--").to_string(), p));
                break;
            }
        }
    }
    out.sort();
    out
}

/// Run `check` on every cached Qwen tokenizer, naming each; skip, saying so,
/// when there is none.
pub fn for_each_tokenizer(check: impl Fn(&str, &Tokenizer)) {
    let found = find_tokenizers();
    if found.is_empty() {
        eprintln!("SKIP: no Qwen tokenizer in the HF hub cache on this machine");
        return;
    }
    for (name, path) in found {
        let tok = Tokenizer::from_file(&path)
            .unwrap_or_else(|e| panic!("{name}: tokenizer.json failed to parse: {e}"));
        eprintln!("── {name}");
        check(&name, &tok);
    }
}
