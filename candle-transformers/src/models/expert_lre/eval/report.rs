//! A printed report of the predictor's coverage at cold start and once converged,
//! read from a captured routing fixture.
//!
//! `load_fixture` loads the captured fixture if it is present. `report` prints
//! the decode-coverage figure, the share of cold experts the next layer needs
//! that the predictor named, which is the value a prefetch can deliver. It also
//! prints a learning curve of decode coverage against training amount, to show
//! whether the predictor improves the longer it runs.

use super::*;

/// Load and decompress the captured fixture, if present.
pub(super) fn load_fixture() -> Option<Vec<RoutingRecord>> {
    let bytes = std::fs::read(FIXTURE_PATH).ok()?;
    let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut raw = Vec::new();
    decoder.read_to_end(&mut raw).ok()?;
    bincode::deserialize(&raw).ok()
}

#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn report() {
    let Some(records) = load_fixture() else {
        println!(
            "\n[expert predictor eval] no fixture at {FIXTURE_PATH}\n\
             capture one with:\n  cargo test --release --features cuda --lib \
             -p candle-transformers quantized_qwen3_moe::tests::capture_routing_trace \
             -- --ignored --nocapture\n"
        );
        return;
    };

    let trace = Trace::from_records(&records);
    println!("\n=== MoE Expert Predictor — Offline Replay ===");
    println!(
        "records={}  passes={}  layers={}  experts={}  avg|set|={:.1}",
        records.len(),
        trace.passes.len(),
        trace.num_layers,
        trace.num_experts,
        trace.avg_set_size(),
    );

    const DECODE_MAX: usize = 16;
    let variants: [(&str, Scoring); 4] = [
        ("raw", Scoring::Raw),
        ("conditional", Scoring::Conditional),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
        ("pmi(1.0)", Scoring::Pmi { alpha: 1.0 }),
    ];

    // ── Scoring comparison, at cold-start and converged ──
    // `decode-cov` (cold-expert coverage on decode passes) maps to real
    // prefetch value: of the experts the next layer needs that are not already
    // active, what fraction did we name.
    let row = |label: &str, k: usize, m: &Metrics, d: &Metrics| {
        println!(
            "{:<14} {:>3} {:>7.1}% {:>9.1}% {:>9.1}% {:>11.1}%",
            label,
            k,
            m.top1() * 100.0,
            m.precision() * 100.0,
            m.coverage() * 100.0,
            d.coverage() * 100.0,
        );
    };

    for (title, warmup) in [
        ("cold-start (online from zero)", 0usize),
        ("converged (~25 epochs)", 25),
    ] {
        println!("\n── {title} ──");
        println!(
            "{:<14} {:>3} {:>8} {:>10} {:>10} {:>12}",
            "predictor", "k", "top1", "precision", "coverage", "decode-cov"
        );
        println!("{}", "-".repeat(62));
        for k in [1usize, 2, 4] {
            let mut pop = Popularity::new(trace.num_layers, trace.num_experts);
            let (a, d) = replay(&trace, &mut pop, k, DECODE_MAX, warmup);
            row("popularity", k, &a, &d);
            for &(label, scoring) in &variants {
                let mut mp = MatrixPredictor::new(trace.num_layers, trace.num_experts, scoring);
                let (a, d) = replay(&trace, &mut mp, k, DECODE_MAX, warmup);
                row(label, k, &a, &d);
            }
            println!();
        }
    }

    // ── Learning curve: decode coverage vs amount of training ──
    // Shows the "improves the more it runs" behavior directly.
    let epochs = [0usize, 1, 2, 4, 8, 16, 32, 64];
    println!("── Learning curve: decode-cov vs training epochs (k=4) ──");
    print!("{:<14}", "epochs→");
    for e in epochs {
        print!("{e:>7}");
    }
    println!();
    for &(label, scoring) in &variants {
        print!("{label:<14}");
        for &w in &epochs {
            let mut mp = MatrixPredictor::new(trace.num_layers, trace.num_experts, scoring);
            let (_, d) = replay(&trace, &mut mp, 4, DECODE_MAX, w);
            print!("{:>6.1}%", d.coverage() * 100.0);
        }
        println!();
    }
    print!("{:<14}", "popularity");
    for &w in &epochs {
        let mut pop = Popularity::new(trace.num_layers, trace.num_experts);
        let (_, d) = replay(&trace, &mut pop, 4, DECODE_MAX, w);
        print!("{:>6.1}%", d.coverage() * 100.0);
    }
    println!();

    println!(
        "\ntop1 = top prediction routed next layer; precision = predicted∩next / predicted;\n\
         coverage = predicted∩(next\\current) / (next\\current); decode-cov = coverage on decode passes.\n\
         NOTE: one 64-token capture. Epochs simulate a longer/persisted session; the real\n\
         converged number needs a longer capture (more decode tokens / diverse prompts).\n"
    );
}
