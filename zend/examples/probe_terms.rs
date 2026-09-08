//! Inspect the probe layer's CPU machinery against a real workspace.
//!
//! Walks the tree, builds the directory-frequency index, and reports what it
//! considers distinctive — plus the rarity gate every specific probe must clear.
//! No model, no GPU, no substrate: this is the half of the probe pipeline that
//! can be checked in a second rather than in a four-hour ingest, and getting it
//! wrong is the failure that would silently waste that ingest.
//!
//! ```text
//! cargo run -p zend --example probe_terms --release -- --dirs 12
//! cargo run -p zend --example probe_terms --release -- --dir candle-nn/src/kv_cache/chunked/
//! ```

use std::collections::BTreeMap;

use zend::repo_scan::metadata;
use zend::repo_scan::probe::idf::TermIndex;
use zend::repo_scan::probe::{render, Register};
use zend::repo_scan::{build_units, walk_workspace};

fn main() {
    let mut root = std::path::PathBuf::from(".");
    let mut show_dirs = 15usize;
    let mut only: Option<String> = None;
    let mut prompt_for: Option<String> = None;
    let mut payload_for: Option<String> = None;
    let mut payload_out: Option<String> = None;
    let mut seed_metadata = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root" => root = std::path::PathBuf::from(args.next().expect("--root needs a path")),
            "--dirs" => show_dirs = args.next().expect("--dirs needs N").parse().expect("N"),
            "--dir" => only = Some(args.next().expect("--dir needs a path")),
            "--prompt" => prompt_for = Some(args.next().expect("--prompt needs a path")),
            "--seed-metadata" => seed_metadata = true,
            "--payload" => payload_for = Some(args.next().expect("--payload needs a dir")),
            "--payload-out" => payload_out = Some(args.next().expect("--payload-out needs a path")),
            other => panic!("unknown arg {other:?}"),
        }
    }

    let started = std::time::Instant::now();
    let map = walk_workspace(&root);
    let walk_ms = started.elapsed().as_millis();
    let units = build_units(&map, &root);
    let index = TermIndex::build(&units, &map);
    let build_ms = started.elapsed().as_millis();

    let total_symbols: usize = map.symbols.values().map(|v| v.len()).sum();
    println!("workspace       {}", root.display());
    println!("files walked    {}", map.files.len());
    println!("files w/ syms   {}", map.symbols.len());
    println!("symbols total   {total_symbols}");
    println!("directories     {}", units.len());
    println!("rarity gate     df <= {}", index.rarity_gate());
    println!("walk            {walk_ms} ms");
    println!("walk+index      {build_ms} ms");
    println!();

    // Write a ready-to-POST chat-completions body carrying this folder's real
    // generation prompt, so the generator can be exercised against a running
    // daemon without waiting for a directory to come round in the ingest — which
    // at full workspace width is over an hour per batch.
    // Seed `.substrate.yaml` skeletons without starting the daemon.
    //
    // Seeding is pure CPU — a walk, an index, and a file per folder — so it has
    // no business waiting behind a model load. Running it here is also how the
    // files get authored: the skeletons appear, a person fills in the questions,
    // and the daemon then finds them complete and generates nothing.
    if seed_metadata {
        let created = metadata::seed_skeletons(&root, &units, |unit| {
            index
                .distinctive(&unit.dir, render::seed_count())
                .into_iter()
                .map(str::to_string)
                .collect()
        });
        let complete = units
            .iter()
            .filter_map(|u| metadata::load(&root, u))
            .filter(|m| m.is_complete())
            .count();
        println!("seeded   {created} new .substrate.yaml files");
        println!(
            "complete {complete} of {} folders need no generation",
            units.len()
        );
        // Name the folders that are PARTLY authored, with the counts per
        // register. A folder one question short of complete still costs a full
        // generation, and nothing else says which register is short.
        let mut partial: Vec<(String, Vec<(&str, usize)>)> = Vec::new();
        for unit in &units {
            let Some(meta) = metadata::load(&root, unit) else {
                continue;
            };
            let counts: Vec<(&str, usize)> = Register::ALL
                .into_iter()
                .map(|r| (r.id(), meta.register(r).len()))
                .collect();
            let total: usize = counts.iter().map(|(_, n)| n).sum();
            if total > 0 && !meta.is_complete() {
                partial.push((unit.dir.clone(), counts));
            }
        }
        if !partial.is_empty() {
            println!("\npartly authored — each still costs a full generation:");
            for (dir, counts) in &partial {
                let cells: Vec<String> = counts
                    .iter()
                    .map(|(name, n)| format!("{name} {n}"))
                    .collect();
                println!("  {dir:<52} {}", cells.join("  "));
            }
        }
        return;
    }

    if let Some(dir) = &payload_for {
        let Some(unit) = units
            .iter()
            .find(|u| u.dir.trim_end_matches('/') == dir.trim_end_matches('/'))
        else {
            eprintln!("no unit for {dir:?}");
            std::process::exit(1);
        };
        let seeds = index.distinctive(&unit.dir, render::seed_count());
        let prompt = format!(
            "{}\n{}",
            render::evidence(unit, "(summary unavailable in this harness)", &seeds),
            render::instruction(unit, &seeds),
        );
        let body = serde_json::json!({
            "model": "zen-code",
            "messages": [{ "role": "user", "content": prompt }],
            "stream": false,
            "max_tokens": 3500,
        });
        let out = payload_out.as_deref().unwrap_or("probe_payload.json");
        std::fs::write(out, serde_json::to_vec_pretty(&body).expect("encode"))
            .expect("write payload");
        eprintln!("wrote {out} ({} seeds)", seeds.len());
        return;
    }

    if let Some(dir) = &prompt_for {
        let Some(unit) = units
            .iter()
            .find(|u| u.dir.trim_end_matches('/') == dir.trim_end_matches('/'))
        else {
            eprintln!("no unit for {dir:?}");
            std::process::exit(1);
        };
        let seeds = index.distinctive(&unit.dir, render::seed_count());
        println!(
            "{}",
            render::evidence(unit, "(the folder's decoded summary goes here)", &seeds)
        );
        println!("{}", render::instruction(unit, &seeds));
        return;
    }

    if let Some(dir) = &only {
        let Some(unit) = units
            .iter()
            .find(|u| u.dir == *dir || u.dir.trim_end_matches('/') == dir.trim_end_matches('/'))
        else {
            eprintln!("no unit for {dir:?}");
            eprintln!("try one of:");
            for u in units.iter().take(40) {
                eprintln!("  {}", u.dir);
            }
            std::process::exit(1);
        };
        report(&index, &unit.dir, unit.files.len(), 28);
        return;
    }

    // The biggest directories first — they carry the most vocabulary, so a
    // failure of the extractor shows there before anywhere else.
    let mut by_size: Vec<_> = units.iter().collect();
    by_size.sort_by_key(|u| std::cmp::Reverse(u.files.len()));
    for unit in by_size.iter().take(show_dirs) {
        report(&index, &unit.dir, unit.files.len(), 12);
    }

    // How many directories would have nothing to seed a specific probe with —
    // the population that can only carry systemic probes.
    let barren = units
        .iter()
        .filter(|u| index.distinctive(&u.dir, 1).is_empty())
        .count();
    println!(
        "── {barren} of {} directories have NO distinctive term (systemic register only)",
        units.len(),
    );

    // Term-frequency shape: how the corpus splits across the gate.
    let mut buckets: BTreeMap<&str, usize> = BTreeMap::new();
    for unit in &units {
        let n = index.distinctive(&unit.dir, 1000).len();
        let bucket = match n {
            0 => "0",
            1..=4 => "1-4",
            5..=19 => "5-19",
            20..=99 => "20-99",
            _ => "100+",
        };
        *buckets.entry(bucket).or_insert(0) += 1;
    }
    println!("── distinctive terms per directory: {buckets:?}");
}

fn report(index: &TermIndex, dir: &str, n_files: usize, k: usize) {
    let terms = index.distinctive(dir, k);
    println!(
        "{dir}  ({n_files} files, {} distinctive)",
        index.distinctive(dir, 100_000).len()
    );
    if terms.is_empty() {
        println!("    (nothing distinctive — systemic probes only)");
    } else {
        for chunk in terms.chunks(6) {
            println!("    {}", chunk.join(", "));
        }
    }
    println!();
}
