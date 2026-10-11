use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use candle_conversation::models::{Model, StrataTokens};
use candle_conversation::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;
use candle_conversation::EngineSource;
use host::cluster::Join;
use host::MODEL;
use web::auth::Roles;
use zend_tools::state::Secrets;
use zend_vfs::Workspace;

use crate::tenant::requires;

/// Runtime configuration for the zend daemon.
#[derive(Clone, Debug)]
pub struct DaemonConfig {
    /// The workspace being served: its folder (absolute) — where `substrate/`
    /// and an optional `projection.yaml` live — and its repositories, the
    /// uploads repository included ([`crate::workspace::open`]).
    pub workspace: Workspace,
    /// The API keys and tokens the tools and the git layer present, read once
    /// at launch from `~/.zend/secrets.yaml` or the file `--secrets` names
    /// ([`crate::secrets::load`]). Every context the daemon builds shares it.
    pub secrets: Arc<Secrets>,
    /// Projection layers taken OUT OF SERVICE (`--disable-layer <name>`,
    /// repeatable). A disabled layer still exists in the schema, but it is
    /// inert: not ingested from the branches, **excluded from the provenance
    /// gather** (`Builder::set_layer_gathered`), not normalization-warmed, and
    /// not swept for crashed partials. Its turns remain in the substrate
    /// untouched — nothing is deleted and dropping the flag restores them — but
    /// while disabled they cannot be selected into any projection.
    ///
    /// The one deliberate exception is an EXPLICIT UPLOAD: a bounded `read_file`
    /// into a disabled per-file layer still runs (see
    /// `InferenceState::ingest_uploaded_files`), because a user who uploads a
    /// file has asked for it to be read. Those turns land in the substrate like
    /// any other and become selectable once the flag is dropped.
    ///
    /// Also names section **collections** (`response`, `mood`), which have no
    /// ingest pass of their own.
    pub disabled_layers: HashSet<String>,
    /// Turn-sink layers to tombstone COMPLETELY before the background ingest
    /// worker's first pass (`--wipe-layer <name>`, repeatable) — every
    /// conversation in the layer, not just crashed partials, so that pass
    /// re-ingests it from scratch. A targeted alternative to
    /// [`Self`]-wide `--wipe-substrate`: every other layer's content (the live
    /// dialogue, an unnamed ingest layer, uploads) survives untouched. A layer
    /// also named by [`Self::disabled_layers`] is not wiped — a disabled layer
    /// gets no cleanup of any kind. `Raw` layers are not wipeable this way.
    pub wiped_layers: HashSet<String>,
    /// Folder overrides for derived ingest layers (`--ingest-dir
    /// <layer>=<path>`, repeatable), keyed by layer name. Each replaces the
    /// folder that layer ingests from, relative to the workspace folder — for
    /// the code layers, one folder inside a repository. Scopes a rebuild to a
    /// subtree (e.g. `code_reading=candle/zend/src`) so the substrate stays
    /// small instead of absorbing every repository.
    pub ingest_dirs: HashMap<String, String>,
    /// `--max-depth <N>`: how deep, in path components below each repository's
    /// root (or a layer's `--ingest-dir` folder), the `repo_map` and
    /// `code_reading` branch walks read (`1` = the root's own files,
    /// `2` = one folder down). Content ingested from deeper is not found by the
    /// walk and is retired. `None` = unbounded.
    pub max_depth: Option<usize>,
    /// Force a whole-store redo-log compaction once during load, after the
    /// substrate reload and before serving. Normally reclaim is incremental and
    /// background (the persistence-thread maintenance pass); this flag forces
    /// the eager whole-store rewrite instead of deferring it. Opt-in
    /// (`--compact-substrate`).
    pub compact_substrate: bool,
    /// Whether the load asks every stored conversation whether it is intact,
    /// and what it does with the ones that say no (`--self-check`,
    /// `--dry-run`). See [`crate::self_check`].
    pub self_check: SelfCheck,
    /// Open the workspace's substrate READ-ONLY and write nothing to disk
    /// (`ModelBuilder::read_only_substrate`): every turn lives in RAM, and the
    /// boot steps that exist to write — calibration, compaction, the upload
    /// reconcile and the background ingest worker — do not run. For a
    /// tool that reads a substrate the running daemon owns, beside it.
    pub read_only_substrate: bool,
    /// Which model the daemon runs: the host's prescribed one
    /// ([`host::MODEL`]). A test harness names a small preset instead; the
    /// daemon has no flag for it.
    pub model: Model,
    /// Where the engine comes from: built by the highest-ranked tenant of the
    /// process with every tenant's requests merged, or attached to it
    /// (`candle_conversation::EngineSlot`). A daemon on its own — and every
    /// harness — has a slot of its own.
    pub engine: EngineSource,
    /// `--qsa-selection-budget <N>`: run the QSA selection with `N` positions in
    /// place of the checkpoint's own budget (`ModelBuilder::qsa_selection_budget`);
    /// `None` keeps the checkpoint's. A budget of at least the checkpoint's
    /// context reads every cell — the dense control a selection failure is
    /// judged against. Refused at load for a model whose attention does not
    /// select, and for a budget the selection kernel cannot run.
    pub qsa_selection_budget: Option<usize>,
    /// `--qsa-window`, `--qsa-recent`, `--qsa-recent-mode`: how a selecting
    /// attention divides its candidates (`ModelBuilder::qsa_strata`,
    /// `docs/qsa_stratified_selection.md`) — windows of `window` positions that
    /// each spend the whole budget, with the system prompt and the `recent`
    /// positions nearest the query ranked in, or forced into, every one.
    /// [`StrataTokens::DEFAULT`] unless the flags say otherwise;
    /// [`StrataTokens::WHOLE`] is the checkpoint's single ranking.
    pub qsa_strata: StrataTokens,
    /// `--summarize`: let conversations launch background tree summaries.
    /// Off by default — every conversation the daemon opens is built from a
    /// config with every summarization trigger disabled
    /// (`ConversationTreeConfig::disable_summarization`).
    ///
    /// Off because a summary is not free to the conversation it summarizes: it
    /// re-reads its whole window from scratch as one more prefill on the same
    /// scheduler, and it runs ahead of the live turn. Measured on a tool-heavy
    /// conversation, the eighth turn launched a 23.8k-token summary prefill and
    /// the user's next tool round waited behind it — 18 s on one conversation,
    /// most of a minute on another.
    pub summarize: bool,
    /// `--tombstone-retention <DAYS>`: how long the substrate keeps a deleted
    /// conversation's tombstone before a compaction may shrink it to its
    /// deleted-set entry (`ModelBuilder::tombstone_retention`).
    pub tombstone_retention: Duration,
    /// Who is an admin — the table a caller's tools modes resolve from
    /// ([`crate::access`]). The host's estate table in the daemon; empty
    /// (nobody) by default, so a harness that builds a default config grants no
    /// admin mode over HTTP.
    pub roles: Roles,
    /// The cluster the process joins with `--cluster`
    /// (`docs/replicated_substrate.md` §3.2): the load connects through it in
    /// its `connecting` step, once the substrate is open. `None` standalone.
    pub cluster: Option<Join>,
}

impl DaemonConfig {
    /// A config serving `workspace` with every flag at its default: no
    /// secrets, no layer flags, no depth bound, the prescribed model on an
    /// engine of its own, nobody an admin.
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            secrets: Arc::new(Secrets::empty()),
            disabled_layers: HashSet::new(),
            wiped_layers: HashSet::new(),
            ingest_dirs: HashMap::new(),
            max_depth: None,
            compact_substrate: false,
            self_check: SelfCheck::Off,
            read_only_substrate: false,
            model: MODEL,
            engine: EngineSource::solo("zend", requires()),
            qsa_selection_budget: None,
            qsa_strata: StrataTokens::DEFAULT,
            summarize: false,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION,
            roles: Roles::default(),
            cluster: None,
        }
    }
}

/// The load's conversation self-check ([`crate::self_check`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelfCheck {
    /// Not run.
    #[default]
    Off,
    /// Every conversation asked, every verdict reported, nothing tombstoned.
    DryRun,
    /// Every conversation asked, and the corrupt ones tombstoned.
    Tombstone,
}
