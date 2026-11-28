//! Hosts the worlds the cast stands in and builds the identity collections the
//! projection reads.
//!
//! Each world is hosted from its map directory and its file, and a world that
//! does not hold together is logged and left out, so one bad map does not stop the
//! rest from being hosted. Renames are applied first, so a renamed world takes its
//! former id in and the former id is never a world of its own. A world's journals
//! group comes from its file, and a world without one keeps no journal memory.
//! The identity collections record who each personality, character and world is,
//! and a character's prompt renders its anchor the same way whether or not the
//! fallback prompt is used.

use super::*;

/// The places the cast stands in, one world at a time — `map/<world_id>/`
/// beside `worlds/<world_id>.yaml`. A map that does not hold together is an
/// authoring mistake to fix, not a reason for the rest to go unhosted.
pub(super) async fn host_worlds(
    runtime: &Arc<Runtime>,
    authored: &Arc<Authored>,
    authored_dir: &Path,
) {
    // The renames first, so a world renamed takes its former in as it is
    // hosted, and a former id is never a world of its own
    // (`docs/world_rename.md` §2).
    authored.renamings(runtime).await;
    // Each world's journals group, from its file (`docs/npc_conversations.md`
    // §5.0): a world that names none keeps no journal memory.
    for record in authored.worlds.read().await.iter() {
        runtime
            .journals
            .set_group(&record.id, journals_group(&record.body));
    }
    let ids: Vec<String> = authored
        .worlds
        .read()
        .await
        .iter()
        .map(|r| r.id.clone())
        .filter(|id| {
            let former = runtime.standing.is_former(id);
            if former {
                tracing::error!(
                    "world `{id}`: refused — another world names it as its former id, so \
                     worlds/{id}.yaml must go"
                );
            }
            !former
        })
        .collect();
    for (id, hosted) in runtime.host_authored(authored_dir, ids.iter().map(String::as_str)) {
        match hosted {
            Ok(w) => tracing::info!(
                "world `{id}` hosted — {} places",
                w.read(|w| w.map().areas().map(|a| a.nodes.len()).sum::<usize>())
            ),
            Err(e) => tracing::error!("world `{id}`: its map did not load: {e:#}"),
        }
    }
}

/// The journals group a world's file names, `journals: {group: moon}`: `None`
/// for a world with none, or one whose group is not a name.
pub(super) fn journals_group(world: &Value) -> Option<Slug> {
    let named = world.get("journals")?.get("group")?.as_str()?;
    match Slug::parse(named) {
        Ok(group) => Some(group),
        Err(e) => {
            tracing::error!("a world's journals group: {e} — it keeps no journals");
            None
        }
    }
}

/// Who each personality, character and world is, for the projection's
/// identity collections — read here, where the registries live.
pub(super) async fn identities(authored: &Arc<Authored>) -> Identities {
    let (npcs, personalities, worlds) = (
        authored.npcs.view(),
        authored.personalities.read().await,
        authored.worlds.read().await,
    );
    let field = |b: &Value, k: &str| {
        b.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    // A word in `reaches:` that names no group grants nothing; said here, once
    // a reading, rather than at every turn that reads the persona.
    for p in personalities.iter() {
        let (_, unknown) = beyond::declared(&p.body);
        if !unknown.is_empty() {
            tracing::warn!(
                personality = %p.id,
                "`reaches:` names no group as {unknown:?}: the groups are outside, reckoning, \
                 memory and making"
            );
        }
    }
    Identities {
        anchors: personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, "anchor")))
            .collect(),
        // What a dream reads in the anchor's place.
        inner_lives: personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, "inner_life")))
            .collect(),
        // Rendered through the same function the fallback prompt uses, so a
        // character reads the same words either way. No anchor: it is its own
        // collection member.
        characters: npcs
            .cast()
            .iter()
            .filter_map(|c| {
                let payload = npcs.payload(c.npc_id)?;
                let owned = engine::persona::of(payload, "", "");
                Some((c.npc_id, engine::prompt::character(&owned.as_persona())))
            })
            .collect(),
        settings: worlds
            .iter()
            .map(|w| (w.id.clone(), field(&w.body, "setting")))
            .collect(),
        // A personality's own voice and way of working, read in the schema's
        // generic ones' place.
        voices: personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, "voice")))
            .collect(),
        actings: personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, "acting")))
            .collect(),
        unordered: personalities
            .iter()
            .filter(|p| p.body.get("takes_orders").and_then(Value::as_bool) == Some(false))
            .map(|p| p.id.clone())
            .collect(),
    }
}
