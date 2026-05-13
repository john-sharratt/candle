use super::*;

#[test]
#[ignore]
fn real_dump_cpu_table_smoke() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("SKIP: dump file not found at {DUMP_REL_PATH}");
            return;
        }
    };
    let (header, chunks) = match load_dump(&path) {
        Some(v) => v,
        None => {
            println!("SKIP: failed to parse dump");
            return;
        }
    };

    let use_chunks = chunks.iter().take(8).collect::<Vec<_>>();
    let n_batch = use_chunks.len();
    let candidates = candidate_formats();

    for (flat_values, side) in [
        (
            use_chunks
                .iter()
                .flat_map(|c| c.k.iter().copied())
                .collect::<Vec<_>>(),
            SampleSide::Key,
        ),
        (
            use_chunks
                .iter()
                .flat_map(|c| c.v.iter().copied())
                .collect::<Vec<_>>(),
            SampleSide::Value,
        ),
    ] {
        let surface = sample_error_surface(
            &flat_values,
            n_batch,
            header.n_kv_head,
            header.chunk_size,
            header.head_dim,
            0,
            &candidates,
            side,
            None,
        )
        .expect("real-data cpu sampling");

        let mut prev_cr = 0.0f64;
        for thr in [1e-6_f32, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2] {
            let winners = select_smallest_passing(&surface, thr, &candidates, None);
            let summary = model_compression_from_surface(&surface, &winners, &candidates, None)
                .expect("summary");
            assert!(summary.ideal_cr >= summary.palette4_cr);
            assert!(summary.palette4_cr >= summary.head_cr);
            assert!(summary.palette4_cr >= prev_cr);
            prev_cr = summary.palette4_cr;
        }
    }
}
