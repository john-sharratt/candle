//! `zend-journals-import <checkout> <person> <file>...` — each file a
//! `POST /v1/journals` body, applied in order onto the journal checkout as
//! `person` (`zend::api::journals::import`). Before each file the checkout
//! is brought level with origin, so the file is checked against what origin
//! holds; after it, what it committed is pushed. The run stops at the first
//! file refused, or the first write a push had to drop, and fails.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use journals::slug::Slug;
use journals::store::{Store, Synced};
use journals::time::Stamp;
use serde_json::Value;
use zend::api::journals::import::apply;

/// What `store` pushed, or why it lost writes: a write origin's newer tree
/// refused is gone, which a run must not pass over.
fn push(store: &mut Store, file: &str) -> Result<Synced, String> {
    let synced = store.sync().map_err(|e| format!("{file}: push: {e}"))?;
    if !synced.dropped.is_empty() {
        let lost: Vec<String> = synced
            .dropped
            .iter()
            .map(|(subject, why)| format!("  {subject}: {why}"))
            .collect();
        return Err(format!(
            "{file}: origin refused {} writes on replay, now gone:\n{}",
            lost.len(),
            lost.join("\n")
        ));
    }
    Ok(synced)
}

fn run(args: &[String]) -> Result<(), String> {
    let [dir, person, files @ ..] = args else {
        return Err("usage: zend-journals-import <checkout> <person> <file>...".into());
    };
    if files.is_empty() {
        return Err("name at least one file".into());
    }
    let person = Slug::parse(person)?;
    let mut store = Store::open(&PathBuf::from(dir)).map_err(|e| e.to_string())?;
    for file in files {
        let text = fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
        let body: Value = serde_json::from_str(&text).map_err(|e| format!("{file}: {e}"))?;
        push(&mut store, file)?;
        match apply(&mut store, &person, &body, Stamp::now()) {
            Ok(answer) => {
                let synced = push(&mut store, file)?;
                let applied = answer["applied"].as_array().map_or(0, Vec::len);
                println!(
                    "{file}: {applied} applied, {} pushed, {} replayed",
                    synced.pushed, synced.replayed
                );
            }
            Err(e) => {
                // A batch the journals moved under partway kept what it
                // committed before then: pushed, not left in the checkout.
                let synced = push(&mut store, file)?;
                return Err(format!(
                    "{file}: {e}\n{} written before it stopped, pushed",
                    synced.pushed
                ));
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
