//! Shared fixtures for the runtime's tests: a hosted vault, bodies, and waiting
//! for a turn.
//!
//! The vault is hosted with its metronome held still, so a test that asserts what
//! hosting, binding or acting does is not racing a world moving underneath it. Tests
//! drive every tick by hand through these helpers. The device daemon is built with
//! the effector installed, so the in-process path is reachable from a test.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle_conversation::persistence::record::{Modulation, NpcPayload};

    pub(in super::super) fn rt() -> Arc<Runtime> {
        // A temp path: the tests exercise scheduling, not persistence, and an
        // in-repo one would leave a ledger file behind.
        Runtime::new(Mind::new(None), &std::env::temp_dir()).unwrap()
    }

    /// One character's record in `state`, for a cast a test installs.
    pub(in super::super) fn character(npc_id: u64, state: &str) -> NpcPayload {
        NpcPayload {
            npc_id,
            owner_id: "u_1a2b3c4d".into(),
            revision: 1,
            created_ms: 0,
            updated_ms: 0,
            state: state.into(),
            name: format!("npc {npc_id}"),
            world_id: "battle-cities".into(),
            personality_id: "commander".into(),
            hidden: false,
            heartbeat_ms: 120_000,
            salience_gate: 0.5,
            tags: Vec::new(),
            persona_description: "A quartermaster.".into(),
            persona_origin: "authored".into(),
            portrait_image_id: None,
            portrait_origin: None,
            at: None,
            mood: None,
            suspended_from: None,
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Modulation::default(),
        }
    }

    /// One device act, from the arguments a decode would have produced.
    pub(in super::super) fn device_act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().expect("an object").clone(),
        }
    }

    /// A daemon hosting the vault, held still, with the effector device installed
    /// — the token store and the `local` router the fast path drives. The same
    /// shape as `npcd/tests/effector.rs::daemon`, here inside the crate so the
    /// `pub(crate)` [`Runtime::enact_device`] is reachable.
    pub(in super::super) fn device_daemon() -> Arc<Runtime> {
        use crate::effector::router::{router, Local};
        use crate::effector::token::Tokens;

        const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
        const WORLD: &str = "creators-vault";

        let rt = rt();
        rt.host(WORLD, std::path::Path::new(ROOMS))
            .expect("the vault loads");
        rt.hold_world(WORLD, true);

        let dir = std::env::temp_dir().join(format!(
            "npcd-enact-device-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tokens = Arc::new(Tokens::load(dir).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());
        rt.set_effector_router(router(Local::new(tokens, &rt)));
        rt
    }

    pub(in super::super) async fn wait_for(mut cond: impl FnMut() -> bool, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    // ── the world, hosted ───────────────────────────────────────────────────

    pub(in super::super) const ROOMS: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");

    /// The id the vault is authored under. A world is hosted under the same id
    /// its document has, because that is what a character's `world_id` names.
    pub(in super::super) const WORLD: &str = "creators-vault";

    /// A runtime hosting the vault, with its metronome held still.
    ///
    /// Paused on purpose: these assert what hosting, binding and acting *do*,
    /// and a world moving underneath them would make every one of them a race.
    /// The metronome's own behaviour is [`crate::engine::driver`]'s to prove.
    ///
    /// Every test that hosts runs as `#[tokio::test]` because hosting needs a
    /// handle to spawn the metronome on — but on the current-thread flavour a
    /// body that never awaits never yields, so no spawned task runs and the
    /// tests keep driving every tick by hand, deterministically.
    pub(in super::super) fn vaulted() -> Arc<Runtime> {
        let rt = rt();
        rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
        rt.hold_world(WORLD, true);
        rt
    }

    pub(in super::super) fn at(node: &str) -> npc_map::world::Where {
        npc_map::world::Where::new("vault-casting", node)
    }

    /// Put a body in the world and give a character to it.
    pub(in super::super) fn embody(rt: &Arc<Runtime>, npc_id: u64, body: &str, room: &str) {
        let w = rt.hosted.get(WORLD).expect("hosted");
        w.with(|w| {
            w.enter(body, format!("Maker-{npc_id:02}"), at(room))
                .unwrap()
        });
        rt.scheduler.wake(npc_id, 0, 0);
        rt.embody(npc_id, WORLD, body, 0).expect("bound");
    }

    pub(in super::super) fn window(rt: &Arc<Runtime>, npc_id: u64) -> Vec<String> {
        rt.scheduler
            .window_of(npc_id, |w| {
                w.turns().map(|t| t.text.clone()).collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Let every due character take its turn, with no decode behind it.
    pub(in super::super) fn run(rt: &Arc<Runtime>, at_ms: u64) {
        for id in rt.scheduler.due_now(at_ms) {
            rt.scheduler.tick(id, at_ms, at_ms, |_, _| Vec::new());
        }
    }

    // ── acts, landing on the world ──────────────────────────────────────────

    pub(in super::super) fn act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().expect("an object").clone(),
        }
    }

    pub(in super::super) fn whole_of(hosted: &Hosted, body: &str) -> String {
        hosted
            .read_both(|w, s| crate::engine::whole::view(w, s, body))
            .expect("a mind with no body")
    }

    /// What reached character `npc` and waits unread, as it would read it.
    pub(in super::super) fn waiting(rt: &Runtime, npc: u64) -> Vec<String> {
        rt.scheduler
            .tick(npc, 0, 0, |_, _| Vec::new())
            .map(|rec| rec.perceived)
            .unwrap_or_default()
    }

    /// A body standing at `at`, bound to character `npc_id`.
    pub(in super::super) fn embody_at(rt: &Arc<Runtime>, npc_id: u64, body: &str, at: Where) {
        let w = rt.hosted.get(WORLD).expect("hosted");
        w.with(|w| {
            w.enter(body, format!("Maker-{npc_id:02}"), at)
                .expect("a real place")
        });
        rt.scheduler.wake(npc_id, 0, 0);
        rt.embody(npc_id, WORLD, body, 0).expect("bound");
    }
}
