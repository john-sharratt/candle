//! Diagnostic: does `tokenizer.decode(ids, skip_special_tokens=true)`
//! preserve `<tool_call>` / `</tool_call>` / `<think>` / `</think>`?
//!
//! In `tokenizer.json` these are registered as added_tokens with
//! `special: False`, so HuggingFace's spec says `skip_special_tokens=true`
//! should NOT strip them.  This test verifies that's actually what the
//! Rust `tokenizers` crate does.

#[cfg(feature = "cuda")]
mod tokens {
    use std::path::PathBuf;

    /// Resolve the Qwen3-30B-A3B tokenizer.json the daemon uses.
    /// Falls back to the HF cache layout under `$HOME/.cache/huggingface`.
    fn tokenizer_path() -> Option<PathBuf> {
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .ok()?;
        let base = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--Qwen--Qwen3-30B-A3B")
            .join("snapshots");
        let snap = std::fs::read_dir(&base).ok()?.flatten().next()?;
        let candidate = snap.path().join("tokenizer.json");
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    }

    #[test]
    fn tool_call_tokens_survive_skip_special_tokens() {
        let Some(path) = tokenizer_path() else {
            eprintln!("skipping: Qwen3-30B-A3B tokenizer not in HF cache");
            return;
        };
        let tok = tokenizers::Tokenizer::from_file(&path).expect("tokenizer load");

        // Known IDs (from tokenizer.json added_tokens, `special: False`):
        //   <tool_call>  = 151657
        //   </tool_call> = 151658
        //   <think>      = 151667
        //   </think>     = 151668
        // Plus a few normal text tokens between them.
        let ids: Vec<u32> = vec![
            151667, // <think>
            198,    // \n
            151668, // </think>
            198,    // \n
            151657, // <tool_call>
            198,    // \n
            5018,   // "{"
            151658, // </tool_call>
        ];

        let skipped = tok.decode(&ids, true).expect("decode skip=true");
        let kept = tok.decode(&ids, false).expect("decode skip=false");

        eprintln!("decode(ids, skip_special=true) :\n{skipped}\n---");
        eprintln!("decode(ids, skip_special=false):\n{kept}\n---");

        // The decisive question: with skip_special=true, do
        // `<tool_call>` / `</tool_call>` / `<think>` / `</think>`
        // survive in the output?
        assert!(
            skipped.contains("<tool_call>"),
            "<tool_call> was stripped by skip_special_tokens=true — \
             the streaming decode path in session.rs would lose every \
             opener.  Output: {skipped:?}",
        );
        assert!(
            skipped.contains("</tool_call>"),
            "</tool_call> was stripped by skip_special_tokens=true — \
             Output: {skipped:?}",
        );
        assert!(
            skipped.contains("<think>"),
            "<think> was stripped by skip_special_tokens=true — \
             Output: {skipped:?}",
        );
        assert!(
            skipped.contains("</think>"),
            "</think> was stripped by skip_special_tokens=true — \
             Output: {skipped:?}",
        );
    }
}
