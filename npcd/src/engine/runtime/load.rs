//! The settings a load reads before it builds anything, and the phases it runs in.
//!
//! The engine is built from the daemon's one substrate handle rather than opening
//! a second, because a second writable handle to the same log loses characters.
//! The model's prefill width is set for the ingest, whose documents are long and
//! arrive in windows. The sampling settings are the checkpoint's own, since a
//! second opinion would run the model outside the settings it was tuned under.
//! The mind's schema is installed before any document is written under it, so
//! every document is gathered against the prompt a character will think under.

use super::*;

/// Tokens one prefill forward carries when the model can take them — the
/// scheduler's per-forward *target*.
///
/// See the note at the load site. In short: the wave's fixed cost is per slab,
/// not per token, so a budget close to one document's size makes every document
/// pay a whole sweep. At the ceiling a slab carries four or five documents and
/// pays it once.
///
/// **A target, not a guarantee.** The scheduler bounds every forward by the
/// model's own width cap as well (`prefill_pass_budget`), and on a routed
/// checkpoint that cap is the narrower of the two: the expert chain carries eight
/// rows per token. Read alone, this number sized the world ingest on the routed
/// Qwen3.6-35B-A3B for a 3.3 GB transient tier a 24 GB card's partition did not
/// have, and every document in the wave failed.
pub(super) const PREFILL_PASS_TOKENS: usize = 8192;

/// `builder` asking for npcd's engine-wide settings: what its load gives the
/// model builder, and what a host running zend without npcd seats in npcd's
/// place ([`crate::tenant::NpcdArgs::engine_requests`]), so the engine is the
/// same one either way.
pub(crate) fn engine_asks(builder: ModelBuilder) -> ModelBuilder {
    builder.prefill_pass_tokens(PREFILL_PASS_TOKENS)
}

/// Reads again what the daemon built from the mind before it had one, once it
/// arrives: a fresh member starts on a mind its log does not hold yet, which
/// the load pulls through an agent after the model is up (`docs/substrate_git.md`
/// §8). The registries, the libraries and the registers the tenant read from
/// nothing are read again from the mind, and what the load was handed from
/// them comes back to it.
///
/// A hook rather than a restart, because in one process with zend a restart
/// stops zend too, mid-load. The tenant's for the same reason the clock is:
/// what it reads lives in the authored state.
pub type MindArrived = Arc<dyn Fn() -> BoxFuture<'static, anyhow::Result<Arrived>> + Send + Sync>;

/// What the load was handed from the authored state, read again once the
/// mind arrived — see [`MindArrived`]. Each is the [`LoadPlan`] field of the
/// same name.
pub struct Arrived {
    pub cast: Vec<Casting>,
    pub characters: Vec<String>,
    pub authored: identity::Authored,
}

/// Where a world's rooms live, under the authored corpus.
///
/// `map/<world_id>/`, a sibling of `worlds/<world_id>.yaml` and keyed the same
/// way. **That key is the whole reason a hosted world has an id**: every
/// character is created with a `world_id` naming the world document it belongs
/// to, so the places it can stand in have to be findable by that same name. A
/// hosted world under any other id would be a world no character could be put
/// into without a second table to reconcile the two.
///
/// A world with no map directory is authored but has nowhere to stand — which
/// is the common case, and not an error. Its characters have lore and no bodies.
pub const MAPS: &str = "map";

/// What the loader needs that is not on the [`Runtime`].
pub struct LoadPlan {
    /// Where the engine comes from: built here, with every tenant's requests
    /// merged, when npcd ranks highest of the tenants that arrive; otherwise
    /// npcd's substrate is attached to the engine a higher one built
    /// (`candle_conversation::EngineSlot`).
    pub engine: EngineSource,
    /// The cluster the process joins with `--cluster`, connected once the
    /// substrate is open; `None` standalone.
    pub cluster: Option<Join>,
    /// The characters to wake, each with the world it belongs to and the room
    /// it was last standing in. From the substrate's character store.
    ///
    /// Both travel with the id because waking a character and putting it back
    /// in its body are one step on a restart: a character woken without its
    /// world would think for a while about a place it is not standing in, and
    /// one woken without its room would do it from the front door.
    pub cast: Vec<Casting>,
    /// Every personality id the mind declares.
    ///
    /// Not the cast: this is what a *layer directory* can be named after, which
    /// is a personality (`layers/memory/zen/`), not an instantiated character.
    /// A world's biographies exist whether or not anyone has cast them.
    pub characters: Vec<String>,
    /// Who each personality, character and world *is*, for the projection's
    /// identity collections. Read in the tenant's start, where the registries
    /// live, and installed on the loader thread — see
    /// [`crate::engine::identity`].
    pub authored: identity::Authored,
    /// Retire every character's conversation before the cast wakes, so each
    /// opens a fresh one — `--forget-conversations`. See
    /// [`Minds::forget_conversations`].
    pub forget_conversations: bool,
    /// Retire every dream every character has kept before the cast wakes —
    /// `--forget-dreams`. See [`Minds::forget_dreams`].
    pub forget_dreams: bool,
    /// Retire every journal entry every character has kept before the cast
    /// wakes — `--forget-journal`. See [`Minds::forget_journal`].
    pub forget_journal: bool,
    /// Retire every conversation in the substrate and clear the ingest ledger
    /// before the layers load — `--wipe-conversations`. See
    /// [`wipe_conversations`].
    pub wipe_conversations: bool,
    /// Whether this member writes the mind into the substrate — its layers'
    /// documents, the lives, and the retirement of what left the record
    /// (`ingest:` in its config). One that does not reads the mind another
    /// member wrote, as it arrives by replication.
    pub ingest: bool,
    /// The mind repository the agents hold, with `--mind-repo`, read from the
    /// substrate (`docs/substrate_git.md` §8): its fetches and pushes go
    /// through the agents once joined, and a mind the log did not hold at
    /// start is pulled and taken in by the load — see [`MindArrived`].
    pub mind_repo: Option<MindSource>,
    /// The agent in this process, with no cluster: what the mind is read and
    /// written through (`docs/substrate_git.md` §9).
    pub agent: Option<Arc<Jobs>>,
    /// The process's stop, asked for a restart onto a schema that moved.
    pub stopping: Arc<Stopping>,
}

/// Load the mind repository whole — pulled through an agent first when no
/// member has — then follow it: each move of its branch takes the mind to the
/// new commit (`docs/substrate_git.md` §8.2). Whether the mind arrived now,
/// having been pending at start: it is taken in here, before it is followed,
/// and the load reads what it builds from the mind again.
pub(super) fn load_mind(
    rt: &Arc<Runtime>,
    source: &MindSource,
    shared: &SharedSubstrate,
    access: Access,
) -> anyhow::Result<bool> {
    let p = &rt.progress;
    // The database the mind was read from at start: what the follower keeps
    // with the log is what every reader of the mind reads.
    let Some(MindRepo::Substrate(db)) = rt.mind_handle.repo() else {
        anyhow::bail!(
            "the mind repository `{}` is read from the substrate, and the mind holds no database \
             of it",
            source.name
        );
    };
    let repos = Repos::over(vec![db.clone()], shared.clone(), access)?;
    let stopping = || rt.stopping();
    repos.load(&stopping, &|state| {
        if let Some(l) = state.first() {
            p.set_detail(match &l.error {
                Some(e) => format!("the mind repository `{}`: {e}", l.repo),
                None if l.listed == 0 => format!(
                    "the mind repository `{}`: waiting for a member to pull it through an agent",
                    l.repo
                ),
                None => format!(
                    "the mind repository `{}`: {} of {} packs",
                    l.repo, l.held, l.listed
                ),
            });
        }
    })?;
    if rt.stopping() {
        anyhow::bail!("the process stopped while loading the mind");
    }
    // A repository whole with no such branch — origin was empty — leaves the
    // mind pending, and its first commit is a move like any other.
    let arrived = match rt.mind_handle.is_present() {
        true => None,
        false => match repos.store(&source.name) {
            Some(store) => mind_store::at_record(store.repo(), source)?,
            None => None,
        },
    };
    let took_in = arrived.is_some();
    if let Some(files) = arrived {
        rt.arrive(files);
    }
    let follower = Arc::downgrade(rt);
    let followed = repos.clone();
    let source_moved = source.clone();
    repos.on_change(Arc::new(move |_| {
        if let Some(rt) = follower.upgrade() {
            mind_moved(&rt, &followed, &source_moved);
        }
    }));
    let stopped = Arc::downgrade(rt);
    repos.follow(Arc::new(move || {
        stopped.upgrade().is_none_or(|rt| rt.stopping())
    }))?;
    let _ = rt.mind_repos.set(repos);
    Ok(took_in)
}

/// The mind's branch moved in the log: every reader moves to the new commit,
/// and what was read from the old is read again — the worlds and
/// personalities, and the documents ingested from it. A move that changes the
/// schema restarts the process onto it, since the grammar is built once.
///
/// A mind pending at start is not one of these: the load takes it in before
/// it follows the branch (see [`load_mind`]), so the follower's first round
/// finds it already here. Only a first commit landing after the load went on
/// without one — its repository empty at origin — still restarts: the
/// projection, the layers and the lives are the load's to build, and it has
/// passed them.
pub(super) fn mind_moved(rt: &Arc<Runtime>, repos: &Repos, source: &MindSource) {
    let Some(store) = repos.store(&source.name) else {
        return;
    };
    let files = match mind_store::at_record(store.repo(), source) {
        Ok(Some(files)) => files,
        Ok(None) => return,
        Err(e) => {
            tracing::warn!("mind: reading the new commit: {e:#}");
            return;
        }
    };
    let old = rt.mind_handle.files();
    if old.as_ref().and_then(|o| o.commit().cloned()) == files.commit().cloned() {
        return;
    }
    let schema_changed = old
        .as_ref()
        .is_none_or(|o| o.read(projection::SCHEMA_FILE) != files.read(projection::SCHEMA_FILE));
    tracing::info!(commit = ?files.commit(), "mind: moved to a newer commit");
    // Not replaced: the grammar is the old schema's until the restart, and
    // readers moved onto a commit it does not describe would read past it.
    if schema_changed {
        match old {
            Some(_) => tracing::warn!("mind: its schema changed — restarting onto it"),
            None => tracing::warn!(
                "mind: its first commit arrived after the load went on without one — restarting \
                 onto it"
            ),
        }
        rt.ask_relaunch();
        return;
    }
    rt.mind_handle.replace(Arc::clone(&files));
    rt.ops.moved();
    if let Some(old) = old {
        rt.reconcile_mind(&old, &files);
    }
}

/// Begin loading, on its own thread. Returns immediately, so the host's
/// server comes up while the model loads and `GET /v1/status` answers with a
/// loading snapshot rather than nothing until the model has finished.
pub fn start(rt: Arc<Runtime>, plan: LoadPlan) {
    let handle = Handle::try_current().ok();
    std::thread::Builder::new()
        .name("npcd-loader".into())
        .spawn(move || {
            #[cfg(feature = "memory-census")]
            set_thread(Scope::Loader);
            let Err(e) = load(&rt, &plan, handle) else {
                return;
            };
            // A load the process is stopping underneath — its engine shut down
            // mid-ingest — is the stop, not a failure.
            if rt.stopping() {
                tracing::info!("engine load ended by the stop: {e:#}");
                return;
            }
            // Fatal, and it has to be: a daemon that answers requests with a
            // half-loaded engine shows the console something present and
            // broken, which is harder to diagnose than something absent.
            fatal("npcd", &format!("engine load failed: {e:#}"));
        })
        .expect("spawn the loader thread");
}
