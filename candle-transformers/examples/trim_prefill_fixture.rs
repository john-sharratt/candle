//! One-time tool to trim a full `ZEND_PREFILL_CAPTURE` dump down to a small,
//! committable kernel-replay fixture: keeps only the single largest-prefix slot
//! (slicing the packed Q/K/V to its token range). CPU-only.
//!
//! Usage:
//!   cargo run -p candle-transformers --example trim_prefill_fixture --release -- \
//!       <input.bin> <output.bin>
//!
//! Defaults: input `prefill_fixture.bin` (cwd), output
//! `candle-transformers/tests/fixtures/prefill_fixture.bin`.

use candle_transformers::models::prefill_capture::PrefillCapture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "prefill_fixture.bin".to_string());
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "candle-transformers/tests/fixtures/prefill_fixture.bin".to_string());

    let bytes = std::fs::read(&input)?;
    eprintln!("read {} bytes from {input}", bytes.len());
    let cap: PrefillCapture = bincode::deserialize(&bytes)?;
    let total_q = cap
        .q
        .len()
        .checked_div(cap.n_head * cap.head_dim)
        .unwrap_or(0);
    eprintln!(
        "loaded: slots={} n_head={} n_kv_head={} head_dim={} total_q={} rope_cs_rows={}",
        cap.slots.len(),
        cap.n_head,
        cap.n_kv_head,
        cap.head_dim,
        total_q,
        cap.rope_cs_rows,
    );
    for (i, s) in cap.slots.iter().enumerate() {
        eprintln!(
            "  slot {i}: prefix(offset)={} q_len={} chunks={}",
            s.offset,
            s.q_len,
            s.chunks.len()
        );
    }

    let trimmed = cap.keep_largest_slot();
    let kept = &trimmed.slots[0];
    eprintln!(
        "kept largest slot: prefix={} q_len={} chunks={}",
        kept.offset,
        kept.q_len,
        kept.chunks.len()
    );

    let out = bincode::serialize(&trimmed)?;
    if let Some(parent) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, &out)?;
    eprintln!("wrote {} bytes to {output}", out.len());
    Ok(())
}
