//! Print what a Maker remembers of the vault, and what it could reach from
//! each room in it.
//!
//! ```text
//! cargo run -p npc-map --example vault_memory
//! cargo run -p npc-map --example vault_memory -- vault-casting
//! cargo run -p npc-map --example vault_memory -- --tools
//! ```
//!
//! With no argument it prints the building memory and then every level. With a
//! level id, that level only. With `--tools`, the tool surface of every room
//! in the vault — which is a function of the parts standing in it, so it is
//! the same question as *what is within reach of a body standing here*.
//!
//! This is the loop for iterating on the generated prose: the maps are data,
//! the text is a pure function of them, so a change to a map file shows up
//! here immediately and in nothing else.

use anyhow::Result;
use npc_map::{describe, Known, MapSet};

fn main() -> Result<()> {
    let dir = std::env::var("VAULT_MAPS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/maps").to_string());
    let set = MapSet::load_dir(&dir)?;

    match std::env::args().nth(1).as_deref() {
        Some("--tools") => tools(&set),
        Some(level) => print!("{}", describe::level(&set, level)),
        None => {
            rule("THE BUILDING");
            print!(
                "{}",
                describe::building(&set, "creators-vault", &Known::All)
            );
            for level in set.children("creators-vault") {
                rule(&level.name.to_uppercase());
                print!("{}", describe::level(&set, &level.id));
            }
        }
    }
    Ok(())
}

/// Every room, and what stands in it for a body to work at.
///
/// A corridor comes out empty, which is the whole point: what a character can
/// do is read off the room rather than declared about the character, so walking
/// away from a terminal takes its acts with it.
///
/// **The parts, not the tools.** This asked the map which *acts* a room offered
/// until the attachment was inverted — an act now names the parts it belongs to
/// (`Tool::at`), and the map went back to saying only what a building contains.
/// So the room's answer is its furniture, and the catalogue is what turns that
/// into a vocabulary.
fn tools(set: &MapSet) {
    for level in set.children("creators-vault") {
        rule(&level.name.to_uppercase());
        for node in &level.nodes {
            let parts = set.part_ids_at(node);
            if parts.is_empty() {
                println!("  {:<24} —", node.name);
            } else {
                println!("  {:<24} {}", node.name, parts.join(", "));
            }
        }
    }
}

fn rule(title: &str) {
    println!("\n{}", "=".repeat(76));
    println!("{title}");
    println!("{}\n", "=".repeat(76));
}
