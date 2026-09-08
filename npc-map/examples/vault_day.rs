//! A few minutes in the vault, with five Makers in it.
//!
//! ```text
//! cargo run -p npc-map --example vault_day
//! ```
//!
//! A scripted scene rather than a simulation — there is no NPC behind any of
//! these bodies yet. What it shows is the shape of what one *would* be handed
//! each turn: a percept, and the tools the room puts within reach.

use anyhow::Result;
use npc_map::perceive::{percept, within_reach};
use npc_map::witness::{narrate, since};
use npc_map::world::{Where, World};
use npc_map::MapSet;

fn main() -> Result<()> {
    let dir = std::env::var("VAULT_MAPS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/maps").to_string());
    let mut w = World::new(MapSet::load_dir(&dir)?);

    let casting = |node: &str| Where::new("vault-casting", node);

    // Three arrive at the lift and go to work; one takes the watch; one sits
    // in the green room with nothing in hand.
    for (id, name) in [
        ("m1", "Maker-01"),
        ("m2", "Maker-02"),
        ("m3", "Maker-03"),
        ("m4", "Maker-04"),
        ("m5", "Maker-05"),
    ] {
        w.enter(id, name, casting("core"))?;
    }

    // Everybody sets off at once and the level empties into its rooms over the
    // next few steps, which is what the ring corridor is for.
    w.set_off("m1", casting("band-one"))?;
    w.set_off("m2", casting("band-one"))?;
    w.set_off("m3", casting("watch"))?;
    w.set_off("m4", casting("green-room"))?;
    w.set_off("m5", casting("ring-north"))?;
    w.settle();

    w.take("m1", Some("r-okonkwo"))?;
    w.take("m2", Some("m-adeyemi"))?;
    w.take("m3", None)?;

    for id in ["m1", "m2", "m3", "m4", "m5"] {
        w.mark_seen(id);
    }

    show(&w, "m1", "AT A TERMINAL");
    show(&w, "m5", "PASSING IN THE CORRIDOR");

    // Now something happens: Maker-02 gets up and says why, and Maker-04
    // wanders in from the green room.
    w.say("m2", "the redoubt burned twice and both are written")?;
    w.release("m2")?;
    w.set_off("m2", casting("green-room"))?;
    w.set_off("m4", casting("band-one"))?;
    w.settle();

    show(&w, "m1", "AFTER A FEW TURNS, STILL AT THE TERMINAL");
    show(&w, "m5", "AND FROM THE CORRIDOR, WHICH HEARD NONE OF IT");
    show(&w, "m3", "AT THE WATCH, WHICH SEES NEITHER ROOM");

    // Maker-05 starts the long walk upstairs, gets one room into it, and is
    // called back to where orders are given instead. Both halves of that reach
    // it as events, because by now neither is something it already knew.
    w.mark_seen("m5");
    let stops = w.set_off("m5", Where::new("vault-command", "command-room"))?;
    println!("\n\n[Maker-05 sets off for the command room: {stops} stops]");
    w.tick();
    show(&w, "m5", "ONE STOP INTO A LONG WALK");

    w.teleport("m5")?;
    show(&w, "m5", "TELEPORTED, WHICH IS THE ONE TRIP THAT IS FREE");

    Ok(())
}

fn show(world: &World, id: &str, title: &str) {
    println!("\n{}", "=".repeat(72));
    println!(
        "{title}  ({})",
        world.actor(id).map(|a| a.name.as_str()).unwrap_or("?")
    );
    println!("{}\n", "=".repeat(72));
    // What a body is handed each turn: a point in time, then the change since
    // it last looked. Two modules, composed here rather than tangled there.
    print!("{}", percept(world, id));
    if let Some(news) = narrate(world, &since(world, id)) {
        println!("\n{news}");
    }
    let tools = within_reach(world, id);
    println!(
        "\nWithin reach: {}",
        if tools.is_empty() {
            "nothing".to_string()
        } else {
            tools.join(", ")
        }
    );
}
