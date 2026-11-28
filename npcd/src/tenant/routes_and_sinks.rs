//! Declares npcd's route tables and installs the runtime resolvers and sinks that
//! read its authored state back.
//!
//! The tables cover the authored corpus, the cast, the engine's surface, the
//! effector device and the worlds' authority, and are merged at start with the
//! host's own and the other tenants' tables, which refuses any overlap. The
//! effector device authenticates by its own device token and never by operator
//! roles, and its paths name no key a gateway could route by. The sinks read
//! state at the moment it is needed rather than at load: the clock resolver
//! reads the clocks' snapshot without a lock, so a character sees its own
//! world's time, and a belief edited through the authoring API reaches the next
//! tick. Fire-and-forget persistence writes go to the blocking pool when a
//! runtime is present and run inline otherwise.

use super::*;

/// npcd's route tables, each over the state it reads.
pub(super) struct Tables {
    /// The authored corpus and the cast.
    pub(super) api: Api<Arc<Authored>>,
    /// The engine's surface.
    pub(super) engine: Api<Arc<Authored>>,
    /// The effector device, under `/v1/local`.
    pub(super) effector: Api<()>,
    /// The worlds' authority, which members write to
    /// (`docs/replicated_substrate.md` §9.3).
    pub(super) authority: Api<Arc<Runtime>>,
}

pub(super) fn tables(roles: &Roles, effector_router: Router) -> Tables {
    // The effector device authenticates by its own device token, never the
    // operator roles: a client may send both a token and `x-tokera-*`, and
    // only the token is read. Its paths name no key a gateway could route by,
    // so a cluster sends it to the best member.
    let effector = Api::<()>::new(roles.clone()).nest_self_authenticating(
        "/v1/local",
        Dispatch::Best,
        effector_router,
    );
    Tables {
        api: api::api(roles.clone()),
        engine: engine::api(roles.clone()),
        effector,
        authority: authority::routes::api(roles.clone()),
    }
}

/// The tables as one, refused when two of them overlap.
pub(super) fn merged(
    api: Routes,
    engine: Routes,
    effector: Routes,
    authority: Routes,
) -> anyhow::Result<Routes> {
    api.merge("npcd api", engine, "npcd engine")?
        .merge("npcd", effector, "npcd effector")?
        .merge("npcd", authority, "npcd authority")
}

/// Every route npcd serves, and the role each needs — the table the host
/// merges with its own and the other tenants' at start, read without standing
/// npcd up.
pub fn declared(roles: &Roles) -> anyhow::Result<Vec<Guarded>> {
    let t = tables(roles, Router::new());
    let all = merged(
        Routes::table(t.api.declared()),
        Routes::table(t.engine.declared()),
        Routes::table(t.effector.declared()),
        Routes::table(t.authority.declared()),
    )?;
    Ok(all.declared)
}

/// npcd's routes, over its state.
pub(super) fn routes(
    authored: &Arc<Authored>,
    runtime: &Arc<Runtime>,
    roles: Roles,
    effector_router: Router,
    log_identity_on: bool,
) -> anyhow::Result<Routes> {
    let t = tables(&roles, effector_router);
    let mut all = merged(
        t.api.into_routes(authored.clone()),
        t.engine.into_routes(authored.clone()),
        t.effector.into_routes(()),
        t.authority.into_routes(runtime.clone()),
    )?;
    if log_identity_on {
        tracing::info!(
            "--log-identity: logging the identity headers and the peer on every request to \
             npcd's own routes, as they arrive after the server has cleared them from any \
             peer it does not trust"
        );
        let log_roles = Arc::new(roles);
        all.router = all
            .router
            .layer(middleware::from_fn(move |req: Request, next: Next| {
                let roles = Arc::clone(&log_roles);
                async move {
                    let peer = req
                        .extensions()
                        .get::<ConnectInfo<SocketAddr>>()
                        .map(|ConnectInfo(a)| *a);
                    log_identity(
                        req.headers(),
                        &roles,
                        peer,
                        req.method().as_str(),
                        req.uri().path(),
                    );
                    next.run(req).await
                }
            }));
    }
    Ok(all)
}

/// The runtime's resolvers and sinks, each reading the authored state back.
pub(super) fn install_sinks(
    runtime: &Arc<Runtime>,
    authored: &Arc<Authored>,
    feelings: Vec<String>,
) {
    // The clock resolver closes over the state: worlds run at their own pace,
    // and a character has to see its own world's time. It reads the clocks'
    // snapshot, so it takes no lock and runs on any thread alike.
    let clock_state = authored.clone();
    runtime.set_clock(Arc::new(move |npc_id: u64| clock_state.world_ms(npc_id)));

    runtime.set_feelings(feelings);

    // Who each character is, resolved at tick time: a belief edited through
    // the authoring API has to reach the next tick.
    let persona_state = authored.clone();
    runtime.set_persona_source(Arc::new(move |npc_id: u64| {
        let npcs = persona_state.npcs.view();
        let payload = npcs.payload(npc_id)?;
        // The world's own `setting`, for the prompt's setting section.
        let world = persona_state
            .worlds
            .blocking_read()
            .get(&payload.world_id)
            .and_then(|r| {
                r.body
                    .get("setting")
                    .and_then(|d| d.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        // The personality's anchor, as prose — a reflection has no gather —
        // and where its characters belong.
        let personalities = persona_state.personalities.blocking_read();
        let personality = personalities.get(&payload.personality_id);
        let field = |k: &str| {
            personality.and_then(|r| r.body.get(k).and_then(|a| a.as_str()).map(str::to_owned))
        };
        let anchor = field("anchor").unwrap_or_default();
        let mut persona = engine::persona::of(payload, &world, &anchor);
        persona.home = field("home");
        persona.unordered = personality
            .and_then(|r| r.body.get("takes_orders"))
            .and_then(Value::as_bool)
            == Some(false);
        persona.attends = field("moves").as_deref() == Some("attention");
        persona.idle = field("idle");
        // What it reaches beyond the world; a word naming no group is said
        // where the personalities are read (`worlds_and_identities`).
        persona.reaches = personality
            .map(|r| beyond::declared(&r.body).0)
            .unwrap_or_default();
        Some(persona)
    }));

    // How it feels: a mood is what a character *is* between one thought and
    // the next.
    let mood_state = authored.clone();
    runtime.set_mood_sink(Arc::new(move |npc_id: u64, mood: &str| {
        let state = mood_state.clone();
        let mood = mood.to_string();
        run_sink(move || {
            state.npcs.remember_mood(npc_id, &mood);
        });
    }));
}

/// Run a fire-and-forget persistence sink write wherever it is safe to block:
/// on a runtime, deferred to the blocking pool; off one, inline. Every sink
/// write is last-writer-wins per character, so ordering does not matter.
pub(super) fn run_sink(write: impl FnOnce() + Send + 'static) {
    match Handle::try_current() {
        Ok(handle) => {
            handle.spawn_blocking(write);
        }
        Err(_) => write(),
    }
}
