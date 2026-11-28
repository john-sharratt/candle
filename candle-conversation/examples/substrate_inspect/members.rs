//! `substrate_inspect members` — which knowledge-base members a restore would
//! take whole, read-only.
//!
//! A member is a section stream declared under `shared/files/…`. A restore
//! lays its chunks out as a per-layer grid and refuses one that stops short of
//! the member's tokens or has a hole in it (`resume::grid_gap`), sending the
//! member to a fresh seal. This asks the same of every member in a stopped
//! store, against the token count its own `Tokens` record holds, so the count
//! of members a start would restore — and of those it would read again — is
//! known before the start.

use std::path::Path;

use anyhow::{Context, Result};
use candle_conversation::files::is_member_stream_name;
use candle_conversation::persistence::resume::{
    decode_token_ids, grid_gap, recover_section_cold_refs,
};
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;

/// Count every member of the store at `substrate_dir` by what a restore would
/// make of it, against `kv_layers` K/V layers.
pub fn members(substrate_dir: &Path, kv_layers: usize) -> Result<()> {
    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_store_read_only(substrate_dir, &mut substrate)
        .with_context(|| format!("opening {} read-only", substrate_dir.display()))?;

    let (mut members, mut whole, mut no_chunks, mut no_tokens) = (0usize, 0usize, 0usize, 0usize);
    let mut partial: Vec<(String, String)> = Vec::new();
    for (sid, stream) in substrate.all_streams() {
        let Some(StreamDecl::PromptSection(decl)) = stream.decl.as_ref() else {
            continue;
        };
        if !is_member_stream_name(&decl.debug_name) {
            continue;
        }
        members += 1;
        let Some(bytes) = persistence.read_tokens(&substrate, sid)? else {
            no_tokens += 1;
            continue;
        };
        let tokens = decode_token_ids(&bytes)?.len();
        match recover_section_cold_refs(&substrate, sid, kv_layers) {
            Ok(None) => no_chunks += 1,
            Ok(Some(grid)) => match grid_gap(&grid, tokens) {
                None => whole += 1,
                Some(gap) => partial.push((decl.debug_name.clone(), gap)),
            },
            Err(e) => partial.push((decl.debug_name.clone(), e.to_string())),
        }
    }

    println!("kv layers: {kv_layers}");
    println!(
        "{members} members: {whole} whole, {} partial, {no_chunks} hold no chunk, \
         {no_tokens} hold no tokens",
        partial.len()
    );
    for (name, gap) in partial.iter().take(20) {
        println!("  partial {name}: {gap}");
    }
    Ok(())
}
