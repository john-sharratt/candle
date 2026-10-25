//! The stores over the mind repository (`docs/replicated_substrate.md` §10.3):
//! an operation's, and an author's.
//!
//! **An operation's work belongs to the operation, not to the mission.** Each
//! operation owns one `zend-vfs` store: its lower layer the mind repository
//! at the commit pinned when the operation opened, its upper layer every edit
//! any step has committed, in order. A step taken on another member — by a
//! character leased there — sees the edits of the steps before it.
//!
//! **An author's writes are drafts until they are published.** Each account
//! that writes the mind through the console — a world, a personality, an
//! entry, a life, a map — writes into its own store. An author's store with
//! nothing in it reads the record as it stands now; once it holds a draft it
//! reads the record at the commit its first draft was made on, until the
//! drafts are published or discarded.
//!
//! Every store lives as file events on its owner's own timeline
//! ([`Owner::timeline`]), so it replicates like any record. A store is read
//! back from its events at each use and saved after each change, so a member
//! never works from a stale copy of what another member committed. While an
//! operation's world runs disconnected (§9.4) a step's edits write no event:
//! the store is held in memory here instead, and scrapped with the local world
//! when the member rejoins.
//!
//! | Module | |
//! |---|---|
//! | [`owner`] | whose store, and the timeline it lives on |
//! | [`log`] | the substrate as the stores' file log |
//! | [`shelf`] | a store as a shelf to read and commit through |
//! | [`merge`] | landing a store on the record: status, publish and sync |

pub mod log;
pub mod merge;
pub mod owner;
pub mod shelf;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::projection::TimelineId;
use store_events::Mirror;
use tokio::sync::watch;
use zend_vfs::workspace::Repo as WorkspaceRepo;
use zend_vfs::{fetch_here, here, Fetch, Push, RepoFiles, RepoSpec, Rev, VfsStore, Workspace};

use crate::mind::MindRepo;

use self::log::Substrate;
use self::merge::Merge;
use self::owner::Owner;

/// What decides that a world runs disconnected: its stores write no event.
pub type Quiet = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// Every store over the mind on this member.
pub struct MindStores {
    /// The mind repository as a workspace of one — its files read through
    /// one source, which every store shares — and its name in it; `None` for
    /// a daemon with no mind folder, whose worlds have no documents.
    mind: Option<(RepoFiles, String)>,
    shared: OnceLock<SharedSubstrate>,
    quiet: OnceLock<Quiet>,
    /// The stores of disconnected worlds' operations, kept here instead of as
    /// events.
    held: Mutex<HashMap<Owner, Arc<RepoFiles>>>,
    /// What makes a publish's push: an agent under `--cluster` (§11.5), this
    /// checkout otherwise.
    pusher: OnceLock<Arc<dyn Push>>,
    /// What makes the record's fetches: an agent under `--cluster`, the mind
    /// a mirror of the repository it holds (§10.3); this checkout otherwise.
    fetcher: OnceLock<Arc<dyn Fetch>>,
    /// How many times the mind folder has been moved to a newer record, for
    /// whatever reads the folder to read it again.
    moved: watch::Sender<u64>,
}

impl MindStores {
    /// The stores over the mind folder at `mind`.
    pub fn new(mind: Option<&Path>) -> Arc<MindStores> {
        let mind = mind.and_then(|root| {
            let name = root.file_name()?.to_string_lossy().into_owned();
            let parent = root.parent()?;
            let files = Workspace::new(parent, vec![RepoSpec::named(&name)])
                .map_err(|e| e.to_string())
                .and_then(|ws| RepoFiles::overlay(ws).map_err(|e| e.to_string()));
            match files {
                Ok(files) => Some((files, name)),
                Err(e) => {
                    tracing::warn!("the mind folder cannot hold stores: {e}");
                    None
                }
            }
        });
        Self::over(mind)
    }

    /// The stores over the mind `repo` holds: a folder, or a repository read
    /// from the substrate — whose stores read its database, shared with
    /// everything else that holds it. `data` is the daemon's data folder,
    /// which a repository with no folder of its own is placed under.
    pub fn for_mind(repo: Option<&MindRepo>, data: &Path) -> anyhow::Result<Arc<MindStores>> {
        let db = match repo {
            None => return Ok(Self::new(None)),
            Some(MindRepo::Folder(root)) => return Ok(Self::new(Some(root))),
            Some(MindRepo::Substrate(db)) => db,
        };
        let name = db.name().to_string();
        let ws = Workspace::new(data, vec![RepoSpec::named(&name)])?.in_substrate(&name)?;
        ws.attach(&name, db.clone(), fetch_here())?;
        Ok(Self::over(Some((RepoFiles::overlay(ws)?, name))))
    }

    fn over(mind: Option<(RepoFiles, String)>) -> Arc<MindStores> {
        Arc::new(MindStores {
            mind,
            shared: OnceLock::new(),
            quiet: OnceLock::new(),
            held: Mutex::new(HashMap::new()),
            pusher: OnceLock::new(),
            fetcher: OnceLock::new(),
            moved: watch::channel(0).0,
        })
    }

    /// Each time the mind folder moves to a newer record.
    pub fn moves(&self) -> watch::Receiver<u64> {
        self.moved.subscribe()
    }

    /// The mind moved to a newer record: tell whatever watches [`Self::moves`].
    pub fn moved(&self) {
        self.moved.send_modify(|n| *n += 1);
    }

    /// Make every publish's push through `pusher`.
    pub fn set_pusher(&self, pusher: Arc<dyn Push>) {
        let _ = self.pusher.set(pusher);
    }

    fn pusher(&self) -> Arc<dyn Push> {
        self.pusher.get().cloned().unwrap_or_else(here)
    }

    /// Make every fetch of the record through `fetcher`.
    pub fn set_fetcher(&self, fetcher: Arc<dyn Fetch>) {
        let _ = self.fetcher.set(fetcher);
    }

    fn fetcher(&self) -> Arc<dyn Fetch> {
        self.fetcher.get().cloned().unwrap_or_else(fetch_here)
    }

    /// Run `merge` over `owner`'s store as it lands on the record, and keep
    /// what it changed. `None` when this daemon has no mind; `Err` when the
    /// store cannot be merged at all.
    pub fn merge<T>(
        &self,
        owner: &Owner,
        merge: impl FnOnce(&Merge<'_>) -> T,
    ) -> Option<Result<T, String>> {
        let repo = self.mind_repo()?.clone();
        let (pusher, fetcher) = (self.pusher(), self.fetcher());
        self.change(owner, |store| {
            Merge::open(&repo, store, pusher, fetcher).map(|m| merge(&m))
        })
    }

    /// The substrate the stores' events are kept in.
    ///
    /// **Every file timeline there is a store's**, and each is declared one
    /// ([`SharedSubstrate::declare_file_store`]): no conversation registers a
    /// store's timeline, so undeclared it reads as an orphan and a compaction
    /// drops it — on WW a draft committed at a step was gone by its reading.
    /// A store is declared before its first save (`keep`); this declares the
    /// ones written before declarations were — those still holding live
    /// events, so a discarded store is not brought back as one.
    pub fn set_substrate(&self, shared: SharedSubstrate) {
        for raw in shared.vfs_timelines() {
            if shared.vfs_events(raw).is_ok_and(|e| e.is_empty()) {
                continue;
            }
            if let Some(timeline) = TimelineId::from_raw(raw) {
                if let Err(e) = shared.declare_file_store(timeline) {
                    tracing::warn!(timeline = raw, "a store could not be declared: {e}");
                }
            }
        }
        let _ = self.shared.set(shared);
    }

    /// What says a world runs disconnected.
    pub fn set_quiet(&self, quiet: Quiet) {
        let _ = self.quiet.set(quiet);
    }

    fn is_quiet(&self, owner: &Owner) -> bool {
        owner
            .world()
            .is_some_and(|world| self.quiet.get().is_some_and(|q| q(world)))
    }

    fn held(&self) -> std::sync::MutexGuard<'_, HashMap<Owner, Arc<RepoFiles>>> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The mind repository's name in each store.
    pub fn repo_name(&self) -> Option<&str> {
        self.mind.as_ref().map(|(_, name)| name.as_str())
    }

    /// The mind repository.
    fn mind_repo(&self) -> Option<&WorkspaceRepo> {
        let (files, name) = self.mind.as_ref()?;
        files.workspace()?.repo(name)
    }

    /// The mind repository's folder; `None` for one read from the substrate.
    pub fn mind_dir(&self) -> Option<&Path> {
        self.mind_repo()?.folder()
    }

    /// Whether the mind folder is a git repository, which is what gives a
    /// store a record to land on. A mind folder that is not one has no drafts:
    /// what is written to it is what stands.
    pub fn under_git(&self) -> bool {
        let Some((files, name)) = self.mind.as_ref() else {
            return false;
        };
        files
            .fresh()
            .repo(name)
            .is_ok_and(|store| matches!(store.rev(), Some(Rev::Branch(_) | Rev::Ref(_))))
    }

    /// `owner`'s stores, as its events hold them — or as this member holds
    /// them while its world is disconnected — with the mirror to save them
    /// against.
    fn open(&self, owner: &Owner) -> Option<(Arc<RepoFiles>, Mirror)> {
        let (mind, _) = self.mind.as_ref()?;
        if let Some(held) = self.held().get(owner) {
            tracing::debug!(?owner, "store opened from what this member holds");
            return Some((Arc::clone(held), Mirror::unread()));
        }
        let files = Arc::new(mind.fresh());
        let mirror = Mutex::new(Mirror::unread());
        if let Some(shared) = self.shared.get() {
            store_events::restore(&Substrate(shared), owner.timeline(), &files, &mirror);
        }
        if owner.world().is_some() {
            tracing::debug!(?owner, "operation store opened");
        }
        let mirror = mirror.into_inner().unwrap_or_else(|e| e.into_inner());
        Some((files, mirror))
    }

    /// Keep what `files` holds as `owner`'s: saved as its events, or held here
    /// while its world runs disconnected.
    fn keep(&self, owner: &Owner, files: &Arc<RepoFiles>, mirror: Mirror) {
        if self.is_quiet(owner) {
            self.held().insert(owner.clone(), Arc::clone(files));
            if owner.world().is_some() {
                tracing::info!(
                    ?owner,
                    "operation store held in memory: its world is disconnected"
                );
            }
            return;
        }
        if let Some(shared) = self.shared.get() {
            // Declared before its first event lands, so no rewrite between the
            // two can take it for an orphan — see `set_substrate`. A store that
            // cannot be declared is saved all the same: unsaved, the change is
            // lost now; saved, it stands until a rewrite, and its next save or
            // the next start declares it.
            if let Err(e) = shared.declare_file_store(owner.timeline()) {
                tracing::error!(?owner, "the store could not be declared: {e}");
            }
            store_events::save(
                &Substrate(shared),
                owner.timeline(),
                files,
                &Mutex::new(mirror),
            );
        }
        if owner.world().is_some() {
            tracing::info!(
                ?owner,
                events = self.event_count(owner),
                "operation store saved"
            );
        }
    }

    /// How many file events `owner`'s timeline holds: what an operation's
    /// store was saved as, read straight from the substrate.
    fn event_count(&self, owner: &Owner) -> Option<usize> {
        let shared = self.shared.get()?;
        shared
            .vfs_events(owner.timeline().raw())
            .ok()
            .map(|e| e.len())
    }

    /// `owner`'s store, ready to use: an author's with nothing in it lets go
    /// of the base its last drafts were made on, and reads the record as it
    /// stands now.
    fn store(&self, owner: &Owner, files: &RepoFiles) -> Option<Arc<VfsStore>> {
        let store = files.repo(self.repo_name()?).ok()?;
        if let (Owner::Author(_), Some(Rev::Branch(branch))) = (owner, store.rev()) {
            if store.changed_paths().is_empty() {
                store.set_branch(branch);
            }
        }
        Some(store)
    }

    /// Read `owner`'s store; `None` when this daemon has no mind.
    pub fn read<T>(&self, owner: &Owner, read: impl FnOnce(&VfsStore) -> T) -> Option<T> {
        let (files, _) = self.open(owner)?;
        let store = self.store(owner, &files)?;
        Some(read(&store))
    }

    /// Change `owner`'s store and keep the change; `None` when this daemon
    /// has no mind.
    pub fn change<T>(&self, owner: &Owner, change: impl FnOnce(&VfsStore) -> T) -> Option<T> {
        let (files, mirror) = self.open(owner)?;
        let store = self.store(owner, &files)?;
        let out = change(&store);
        self.keep(owner, &files, mirror);
        Some(out)
    }

    /// Pin `owner`'s base — the mind as it stands now — when it opens.
    pub fn pin(&self, owner: &Owner) {
        self.change(owner, |store| {
            if let Err(e) = store.base() {
                tracing::warn!(?owner, "a store could not take its base: {e}");
            }
        });
    }

    /// Drop `owner`'s store: every change it holds goes, as its events'
    /// tombstones, which the next rewrite of the log reclaims. Its declaration
    /// as a store stands ([`SharedSubstrate::declare_file_store`]): a release could land
    /// on another member after its next save there, leaving that save to be
    /// dropped as an orphan's.
    pub fn discard(&self, owner: &Owner) {
        self.change(owner, |store| store.discard());
        self.held().remove(owner);
    }

    /// Scrap what a disconnected `world` held: on rejoining, its operations'
    /// stores are what their events say again.
    pub fn scrap(&self, world: &str) {
        self.held().retain(|owner, _| owner.world() != Some(world));
    }
}

#[cfg(test)]
mod tests;
