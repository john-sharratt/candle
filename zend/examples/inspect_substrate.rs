//! Substrate census — tally what is actually persisted, so a rebuild can be
//! checked for completeness: how many turn (conversation) streams and prompt
//! sections exist, and which of them carry chunks / tokens /
//! wide-Q consensus windows.
//!
//! ```text
//! cargo run -p zend --example inspect_substrate --release -- <workspace>
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;

#[derive(Default)]
struct Tally {
    total: usize,
    with_chunks: usize,
    with_tokens: usize,
    with_wide_q: usize,
    with_proj_events: usize,
    chunk_total: usize,
}

impl Tally {
    fn add(&mut self, s: &candle_conversation::substrate::StreamRuntime) {
        self.total += 1;
        if !s.chunks.is_empty() {
            self.with_chunks += 1;
            self.chunk_total += s.chunks.len();
        }
        if s.tokens.is_some() {
            self.with_tokens += 1;
        }
        if s.wide_q_sigs.is_some() {
            self.with_wide_q += 1;
        }
        if s.projection_events.is_some() {
            self.with_proj_events += 1;
        }
    }
    fn print(&self, label: &str) {
        println!(
            "{label:<16} total {:>5}   chunks {:>5} ({} chunk-recs)   tokens {:>5}   wide-Q {:>5}   proj-events {:>5}",
            self.total, self.with_chunks, self.chunk_total, self.with_tokens, self.with_wide_q, self.with_proj_events,
        );
    }
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut substrate = Substrate::new();
    let _persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate at {}: {e}", workspace.display()))?;
    eprintln!("opened substrate at {}", workspace.display());

    let (mut turns, mut sections, mut other) =
        (Tally::default(), Tally::default(), Tally::default());
    // Per-timeline turn count (a "conversation" = one timeline of turn streams).
    let mut per_timeline: HashMap<u64, usize> = HashMap::new();
    // Which turns are missing a wide-Q record (the gather corpus needs these).
    let mut turns_missing_wideq = 0usize;

    for (_sid, e) in substrate.all_streams() {
        match &e.decl {
            Some(StreamDecl::Turn(t)) => {
                turns.add(e);
                *per_timeline.entry(t.timeline_id).or_default() += 1;
                if e.wide_q_sigs.is_none() {
                    turns_missing_wideq += 1;
                }
            }
            Some(StreamDecl::PromptSection(_)) => sections.add(e),
            _ => other.add(e),
        }
    }

    println!("\n=== substrate census ({}) ===", workspace.display());
    turns.print("turns");
    sections.print("prompt-sections");
    other.print("other");
    println!(
        "\nconversations (distinct timelines): {}   turn streams missing wide-Q: {}",
        per_timeline.len(),
        turns_missing_wideq,
    );
    Ok(())
}
