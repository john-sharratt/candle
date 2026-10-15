# Git in the substrate: repositories replicate as packs

**Status:** stages 1–4 are built (§11): the records, the importer and adoption
(`substrate-git/`), the in-process object database every repository read from the
substrate is held in (`zend-vfs/src/odb/`), zend reading every repository from it, and
npcd's mind read from the substrate with `--mind-repo`. No folder and no `git` program is
involved on a member. Nothing has run end to end across members yet (§12).
**Scope:** where a member's git objects and refs come from. Today every zend member reads
each workspace repository from its own checkout, and every npcd member reads its mind
from a mirror it keeps under `--data`. This design moves both into the substrate: git's
packs and refs become replicated records, and a member reads every repository from what
its log holds. Agents remain the only machines that talk to `origin` (fetch, push) and the
only ones with working trees for commands. zend and npcd use the same mechanism.

§1 is the code as it stood before this design. §2 lists the decisions. §3–§9 are the design. §10 lists what
it removes and what it adds, §11 the build order, §12 the tests, §13 what is open.

---

## 1. Where things stand

### 1.1 zend reads each repository from a checkout in its workspace

- The manifest names folders. Every repository in `workspace.yaml` must be a folder
  directly under the workspace, and a missing one fails the launch
  (`docs/zend_workspace_execution.md` §3.1, `zend-vfs/src/workspace.rs`).
- A conversation's files are a union mount: its edits (`VfsEvent`, replicated,
  `docs/zend_vfs_events.md`) over a lower layer, which for a repository is
  `Lower::Branch { source: Arc<GitSource>, .. }` (`zend-vfs/src/vfs/mod.rs:427-441`).
  `GitSource::open(dir)` opens the checkout's `.git` (`zend-vfs/src/vfs/git_source.rs:52`).
- `Repo` holds libgit2 for lookups, and sends everything else to the `git` program:
  "the network (fetch, push, clone…), the working tree (checkout, merge, the index) and
  everything that writes" (`zend-vfs/src/library/mod.rs`, `zend-vfs/src/lib.rs:139`).
  Commits, merges, `git_log` and `git_grep` therefore need a repository on disk.
- `git_commit` / `git_merge` build objects in the member's own `.git` under scratch refs
  and push them to `origin` under a lease (`zend-vfs/src/write/commit.rs`,
  `zend-vfs/src/remote/push.rs`). The objects exist only on that member until the push
  lands and the other members fetch.
- Every member fetched for itself. One `origin_watch` task per repository probed `origin`
  every couple of seconds and fetched on a change (`zend/src/origin_watch/`, removed by §7).
- Routing has to care. `HELLO.repos` lists a member's repositories, and `Sticky`/`Best`
  consider only members holding every repository a conversation names
  (`docs/replicated_substrate.md` §10.1). D10 there exists because "machines with
  different checkouts contribute their repositories".

### 1.2 npcd reads the mind from a mirror

- With `--cluster`, the mind is `--mind-repo <name>`, mirrored at
  `<data>/mind/<name>` with `origin = agent:<name>` (`npcd/src/mind_repo.rs:26,42-64`).
  `--mind <dir>` is refused beside `--cluster`, and `--mind-repo` without it
  (`npcd/src/tenant.rs:326-336`).
- The mirror follows the record every 60 s and after a publish, through `AgentFetch`
  (`docs/replicated_substrate.md` §10.3). A member whose mirror is not ready starts with no
  mind, makes the mirror once joined, and restarts onto it (`MindRepo::bring_up`,
  `mind_repo.rs:88`).
- npcd reads the mind as files. The schema is `std::fs::read_to_string` of
  `projection.yaml` at start (`npcd/src/engine/schema.rs:67-69`). About 25 modules read the
  mind folder directly: the watcher, registry, corpus, console, API, mission generation
  and lifegen among them.
- The watcher turns file changes into substrate writes: unchanged costs nothing, changed
  tombstones the old turn and ingests the new one (`npcd/src/engine/watcher.rs`).
- An operation's store reads the mind "at the commit pinned when the operation opened"
  (`docs/replicated_substrate.md` §10.3), so every member serving a step needs that
  commit's objects.

### 1.3 What the substrate already has

- **Content-addressed records.** `Blob` (tag 29) is bytes named by their SHA-256, keyed in
  the header by the hash's first 128 bits, checked on every read, a duplicate dead on
  arrival, never superseded and never resident (`persistence/record.rs`,
  `persistence/blob/`). It is how images replicate.
- **Leases.** Paxos leases among an app's members, used by `ingest/<content key>` so one
  member reads a unit (`zend/src/ingest_lease.rs`).
- **Fetch and push behind traits.** `Fetch` and `Push` (`zend-vfs/src/remote/fetcher.rs:17`,
  `pusher.rs:16`) have local and agent implementations (`AgentFetch`, `AgentPush`).
- **Records that wait for their parents.** A record whose parent has not arrived waits for
  it (`docs/replicated_substrate.md` §6.5).

### 1.4 The cost

A repository exists on a member up to three times: its checkout or mirror, the objects
other members fetched separately, and the ingest of it in the substrate. Each copy moves
on its own schedule:

- npcd members can serve different mind tips for up to 60 s, and a member keeps the schema
  it started with until it restarts.
- zend commits sit in one member's `.git` until pushed and fetched elsewhere.
- Every member probes and fetches `origin` independently.
- A member needs every repository checked out to be routed a conversation.

Measured on 2026-10-15: candle's objects are 1.3 GB packed and **8.9 GB raw** (41,604
blobs, 26,505 trees, 6,547 commits); battle-cities' are 0.3 GB packed plus 7.7 GB of LFS
content; battle-mind's are under 0.1 GB.

---

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| G1 | **A repository's objects replicate as git packs.** Each pack is a `Blob` record, and a keyed object `git/pack/<repo>/<sha256>` says it is one (§3). Packs are capped at 64 MiB. | Raw objects would be 7× the size (§1.4). A pack is what git fetches and pushes anyway. `Blob` is already content-addressed, verified on read and replicated, so no new record type is needed. |
| G2 | **Branches replicate as keyed objects** `git/ref/<repo>/<ref>`, naming the packs they need. A branch is applied only once those packs are held, and only forward unless `forced` (§3). | "Latest" becomes a point in the log, the same on every member, not each member's last fetch. |
| G3 | **One member imports from `origin`**, holding the lease `git/<repo>`, fetching through an agent. Everyone else learns new commits by replication (§5). | One fetch per change for the whole group, instead of one per member. |
| G4 | **Every member publishes its own pushes**, packs first, then the branch (§6). | A commit that lands on `origin` reaches every member through the log, not through each member's next fetch. |
| G5 | **A member's repository is an object database in process**, its packs the log's, its refs and settings in memory, its record branches the log's (§4). There is no folder, and libgit2 over the database answers everything `git` answered for a checkout. | Unlike a checkout or a mirror, it has no source but the log, so it cannot diverge from it; and nothing is written to disk twice — the log's packs are the repository. |
| G6 | **Repositories are names, not folders.** A member reads each repository `workspace.yaml` names from its object database and needs no checkout (§7). | Members stop needing per-machine clones. |
| G7 | **npcd's mind is a repository like zend's.** npcd reads it through the same database, at a commit. A branch move replaces the file watcher's trigger, and the mirror is retired (§8). | One mechanism for both daemons. The schema, the corpus and an operation's base all come from one commit in the log. |
| G8 | **Standalone daemons use the same read path.** Without `--cluster`, the importer fetches through the agent in this process instead of the cluster's (§9). | One code path. |
| G9 | **Packs are never collected in this design**, as `Blob` records are not. LFS content is not imported (§13). | git history is append-only by intent. Reachability collection and LFS are separable work. |

---

## 3. Records

Built in `substrate-git/src/records.rs`. Both kinds are keyed objects (`CustomObject`,
last writer wins by version stamp), held in RAM and replicated; a pack's bytes are a
`Blob`.

| Key | Value |
|---|---|
| `git/pack/<repo>/<sha256>` | `PackEntry { checksum, objects, index }`: the `Blob` of that SHA-256 is a pack of `<repo>`, and the `Blob` named by `index` is its git `.idx` (version 2) |
| `git/ref/<repo>/refs/heads/<branch>` | `RefEntry { oid, needs, forced }`: where the branch points (`None` for deleted), the SHA-256s of the packs written with it, and whether it may replace a non-descendant |
| `git/slice/<repo>/<oid>` | A commit an unfinished sliced fetch has taken (§5.1), so the next lease holder resumes from it |

- **Packs.** At most 64 MiB, so a pack fits a relocation stripe
  (`persistence/stripes.rs:14`) and replicates as 64 `EVENT` fragments or fewer. A pack
  comes from one of three places: the objects a member made, packed when it publishes
  (§4.4); a bundle an agent answers a fetch with, written as it is taken (§5.1); and a
  folder repository adopted into the log (`adopt`, the tests' fixture). Every pack is
  self-contained — no delta in it names a base outside it (§4.2). A blob already held is
  not written twice.
- **Indexes.** The pack's `.idx` is written beside it as its own `Blob`, so a member reads
  where every object is without decoding a pack: about 28 bytes an object, 2 MB for
  candle's 75,000.
- **Entries replicate apart from their blobs.** A branch whose `needs` are not all held
  waits; a sync takes the pack once its blob arrives. A pack whose blob cannot be read or
  taken is reported by name (`Synced::failed`, the load's detail, the follower's warning)
  and passed over; every other pack is still taken, and no branch moves while it stands.
- **Never backwards.** A writer checks that a new tip descends from the log's before
  writing it (`substrate-git/src/publish.rs`), and a store never moves a branch back to an
  ancestor of what it holds unless the entry says `forced`. A tip on another line *is*
  taken: `forced` belongs to the one entry that rewrote the branch, so a store that missed
  that entry meets the commits written on top of it as plain entries. Last-writer-wins can
  still let a stale write land briefly; the next import tick (2 s) corrects it.
- Conversation branches are not entries. A conversation's branch per repository stays in
  its `ConvState` (`zend/src/conv_branches.rs`).

---

## 4. The object database

A repository read from the substrate is held in process and has no folder
(`zend-vfs/src/odb/`). A `Repo` over it answers every question a `Repo` over a checkout
does, through libgit2, and never starts the `git` program: `Invocation::command` refuses a
repository with no folder, so a path not ported fails loudly instead of running against
the wrong place. Commands that need files run on agents, which keep their own checkouts
(`docs/replicated_substrate.md` §11).

### 4.1 Reading objects

- **The packs are the log's.** A `PackStore` (`substrate-git/src/pack_store.rs`, over the
  records) lists the packs whose pack and index blobs have both arrived, reads either, and
  writes a new pair. The database loads every index once, and a pack's bytes when an
  object in it is first wanted.
- **Pack bytes are kept** in memory within a budget, least recently read let go first
  (`odb/packs.rs`; `DEFAULT_PACK_BUDGET`, 1 GiB, an argument of `ObjectDb::open`):
  candle's 21 packs fit, and a pack let go is read again from the log.
- **Decoding is ours** (`odb/pack/`): an entry's header, zlib (flate2), and git's delta
  format — an offset delta's base in the same pack, a ref delta's by id — with the decoded
  bases of recent chains cached.
- **libgit2 reads through it.** A custom `git_odb_backend` (`odb/backend.rs`) answers
  `read`, `read_header`, `read_prefix`, `exists`, `exists_prefix`, `refresh` and `foreach`
  from the packs, so every algorithm libgit2 already runs in process — revision parsing,
  history walks, merge bases, tree reads, blame, diffs, merges — runs over them unchanged.

### 4.2 Taking a pack

`ObjectDb::take_pack` indexes a pack's bytes in Rust (`odb/pack/indexer.rs`): every entry
decoded, every delta resolved, every object hashed, and the `.idx` written exactly as
`git index-pack` writes it. A ref delta whose base is not in the pack — a thin pack, which
a bundle may be — has the base appended to the pack, whole, from what the database holds,
and the header's count and the trailer rewritten (`git index-pack --fix-thin`'s rule), so
no pack in the log depends on another.

A pack larger than the log keeps (64 MiB) is never written: it is held in memory only
while its objects are packed again in parts of about half the limit, each kept, and then
let go (`odb/split.rs`). An answer from an agent that does not bound its bundles, or one
commit larger than the bound, is kept that way.

### 4.3 Refs and settings

- **A custom `git_refdb_backend`** (`odb/refdb.rs`) holds the refs in memory, with
  compare-and-swap locks for libgit2's transactions.
- **The record is `origin`'s tracking refs**, `refs/remotes/origin/<branch>`, which is
  where zend reads a repository's branches when it has an origin
  (`Repo::record_branches`). `Store::sync` refreshes the packs and then moves each record
  branch whose packs are held, never back to an ancestor unless `forced`.
- `HEAD` names `refs/heads/substrate-store`, a branch never made, so nothing is checked out
  and no branch the log moves is the current one.
- **Local refs** — the branch a landed push leaves, the layer's scratch refs, the
  importer's `refs/remotes/import/*` — are the member's own, held in memory and gone at a
  restart: everything they name is in the log or on `origin`.
- **Settings** a checkout keeps in `.git/config` — its remotes and its branches'
  upstreams — are held in memory beside the refs. `origin` and `import` are both
  `agent:<repo>`. The identity commits are made under is the user's global git
  configuration, read in process, as a checkout's falls back to it.

### 4.4 Writing objects

- **Pending objects.** What a member makes — blobs, trees, commits, tags — is written to
  the database's own memory (`odb/pending.rs`), where libgit2 reads it back like any other
  object. A blob is stored as it is given: a repository with no working tree has no
  `.gitattributes` or `core.autocrlf` of its own to filter it through.
- **Publishing packs them.** Before a member publishes a ref, `ObjectDb::flush` packs
  every pending object (libgit2's packbuilder), takes the pack (§4.2) into the log, and
  drops the pending objects it holds. The packs written since the last published ref are
  that ref's `needs`.
- **A bundle** for a push is cut in process: libgit2's packbuilder over the pushed tips,
  hiding what `origin`'s tracking refs reach, behind git's bundle header. A bundle taken —
  a fetch's answer — goes to the log as a pack at once (§4.2).

### 4.5 What runs where

| Operation | In a checkout | In the database |
|---|---|---|
| refs, revisions, ancestry, merge base, trees, blob reads | libgit2 | libgit2 |
| history (`log`, a file's history across renames) | `git log` | libgit2 revwalk, git's path simplification and rename following |
| `grep` at a revision | `git grep` | a walk of the revision's tree, each blob matched in process |
| blame | `git blame` | libgit2 blame |
| diffs and patches | `git diff-tree` | libgit2 diff, git's patch format |
| commits, merges, cherry-picks, reverts, applied patches | `fast-import`, `read-tree`, `merge-file`, `apply` | libgit2 tree builder, `merge_trees`, `merge_file`, `apply_to_tree` |
| tags | `mktag` | libgit2 tag objects |
| bundles | `git bundle` | the packbuilder and §4.2 |
| upstreams, remotes | `.git/config` | held in memory (§4.3) |

A repository's object format is SHA-1: libgit2 holds no other, and a `sha256` repository is
refused when it is first imported (§13).

---

## 5. Bringing commits in

### 5.1 The importer

Every member runs an `Importer` per repository (`substrate-git/src/import.rs`), ticked
every 2 s by zend's `substrate-git` thread (`zend/src/substrate_git.rs`). Each tick:

1. **Publishes this member's pushes**: a record branch ahead of the log (§6.2).
2. **Imports**, on the member holding the lease `git/<repo>` (Paxos, 120 s, renewed every
   30 s): fetches every branch into `import` through `Fetch` (`AgentFetch` with
   `--cluster`), and publishes every branch that moved ahead of the log. A fetch that
   fails (no agent, origin down) is reported and the tick carries on.
3. **Syncs** the store up to the log. A record that changed since the follower last
   reported it — a sync, or this member's own push landing — wakes zend's ingest worker
   and moves npcd's mind.

A first pull of a long history comes through the agent in slices of 256 commits
(`max_commits`). A slice is a prefix of the history, parents first, and its bundle is cut
from every *head* of it — each commit of the slice no other commit of it descends from —
so a slice across merges carries its whole budget. The member names those heads as held
when it asks again, and keeps them as `git/slice/<repo>/<oid>` records (§3) so a failed
run resumes.

Each answer becomes one pack in the log, at most 64 MiB, so an answer is bounded by bytes
as well (`max_bytes`, half of that): a slice whose bundle is larger is cut again, half its
commits each time, until it fits, leaving room for the bases a thin bundle is completed
with (§4.2). A single commit whose objects do not fit is answered alone, as it is, and the
member keeps it as several packs (§4.2).

### 5.2 Everyone else

Members apply the records as they arrive and sync each tick. A branch that moves wakes
whatever follows it: zend's branch ingest, and npcd's mind reload (§8.2).

### 5.3 Force pushes

`origin` can move a branch to a non-descendant, or delete it. The importer publishes that
with `forced: true` only when two consecutive ticks agree, so a fetch racing another
member's push cannot rewind a branch. The old commits stay in their packs (G9).

### 5.4 A member cut off from the gateway

A cut-off member cannot hold the lease, so it imports nothing. It reads what its log
holds, and its own pushes are published when it rejoins, like any record written while
cut off.

---

## 6. Writing

### 6.1 Objects

zend's write tools (`git_commit`, `git_merge`, `git_cherry_pick`, `git_revert`) build
commits in the store, as they built them in a checkout. A commit always lands on `origin`
(`docs/replicated_substrate.md` §10.1), so it reaches the log when its push does.

### 6.2 Pushing

`Push` is unchanged. The bundle sent to the agent is built from the store, whose record is
the log's. A push that `origin` accepts moves the store's record as it lands:
`Repo::push` moves each accepted branch's tracking ref after whichever pusher made it, as
`git push` does for its own — an agent's push touches none of the store's refs. The
member's next tick publishes it, packs first, and its follower reports the move, so the
publishing member moves onto its own push at once. Since the push was compare-and-swap on
`origin`, the new tip descends from the log's, and never-backwards keeps it.

---

## 7. zend

Built, with no setting and no checkout mode, in `zend/src/substrate_git.rs`.

- **Manifest.** Unchanged: `workspace.yaml` names repositories. The manifest is read
  without requiring the folders (`Workspace::manifest`), and each repository is marked
  as read from the substrate (`Workspace::in_substrate`): it has no folder. The name is
  unchanged, so every tool argument, workspace-relative path and ingest key is too.
- **One database per repository.** The workspace is built at launch, before the
  substrate opens, so a repository read from the substrate holds a slot every clone of
  the workspace shares. The load opens each database and attaches it there
  (`Workspace::attach`), so what the tools read and what the follower keeps with the log
  are one database, whose refs are one set.
- **Everything that opens a repository** — `GitSource` for the lower layer, branch
  ingest, conversation branches, the git tools, pushes through `AgentPush` — opens it
  through the workspace (`workspace::Repo::open`): a `Repo` over the database, fetching
  through the agent. A repository that has not been attached yet refuses to open. No
  sandbox is made for one: it has no folder, and its commands run on agents.
- **Starting.** After the cluster join and catch-up, before anything reads a branch, each
  repository is brought up to the log once, pulled through an agent (§9) when the log
  names no branch of it. One the log names is not fetched at all (`Importer::settle`):
  it is whole from the log, origin's newer commits are the follower's, and fetching
  there made every restart wait on origin — 76 s for candle — with nothing to bring.
  The repositories' rounds run side by side, the loading page has their packs held of
  listed before the first round, and each logs `repository loaded` with its publish,
  fetch and sync time. Then a `substrate-git` thread ticks every 2 s until shutdown
  stops and joins it, before the drain. A workspace naming no repository skips the step
  and needs no agent.
- **Conversation bases.** A store has no local branch, so a conversation's base is the
  record's `main` (else `master`) before `HEAD` (`zend/src/conv_branches.rs`).
- **The ingest worker** is woken by a record branch that moves, and once when it starts
  listening; `origin_watch` is removed.
- **Commands** (`run_command`) already run on agents with the conversation's overlay.
- **Advertising.** `HELLO.repos` still lists each repository, so routing is unchanged
  among members. The advertisement goes out before the repositories load, so one read
  from the substrate carries no origin URL: a member reaches origin only through agents,
  which advertise their own.

---

## 8. npcd

### 8.1 Reading the mind

Built in `npcd/src/mind/files.rs` and `npcd/src/mind_store.rs`.

- **`MindFiles`** is the mind's files at one commit of its database, or in a plain folder.
  A commit's tree is indexed once with `ls_tree_all`, and each file's bytes are read in
  process on demand. It answers `read`, `bytes`, `is_file`, `is_dir`, `size`, `list`,
  `walk` and `not_a_document`. A link in a commit is never read as a document, as a
  link in a folder is not.
- **`Mind`** holds a swappable `Arc<MindFiles>`. Every clone shares one holder, so a
  branch move moves every reader at once. A mind read from the substrate that has not
  arrived yet is *pending*: no files until it is pulled.
- **Everything reads through it**: the schema (`projection::Source::from_files`,
  `engine/schema.rs`), the response and mood libraries, the world and personality
  registries (`Registry::load_mind`, re-read at each reload), personality portraits,
  the layer sources and their documents (`engine/ingest.rs`), the lives
  (`engine/life.rs`), bearings, eras, the mission generator's corpus, glossary, lore
  and `missions.yaml`, the bench's canon index, and the published shelf (`Published`)
  that every API read goes through.
- **Maps** load from the mind's `map/<id>/` files (`MapSet::load_texts`,
  `Hosted::load_mind`). A reshape of one is kept as the owner's draft in the mind.
- **One database.** `Mind` holds the repository behind its files (`MindRepo`): a
  folder, or the object database the tenant opened. The operations' stores
  (`MindStores::for_mind`) and the follower (`Repos::over`) are given that same
  database, so a branch the follower moves is the branch every store reads.
- **Operations' stores** read through the database's record: `GitSource` defaults to
  `origin/main` (else `master`) when the checked-out branch has no commit, which is the
  database's case, and `Merge` publishes on that branch through the agent.
- **The ingest ledger** is keyed by mind paths. A key an earlier build wrote as an
  absolute path under a mind folder is cut to its `layers/…` part when the ledger
  opens, so a mind that moved into the substrate retires nothing.

### 8.2 Startup and following

- `--mind-repo <name>` (with `--cluster` and `--data`): the tenant opens the substrate
  first, opens the mind's object database over it and brings it up to the local log
  (`mind_store::from_log`), and reads the schema and the registries from the branch's
  commit. A log that holds no such branch yet starts the daemon on the bundled schema,
  with the mind pending.
- After the cluster join, `load_mind` (`engine/runtime.rs`) loads the repository whole
  through `substrate_git::member::Repos`, over the mind's database: pulled through an
  agent by the lease holder when no member has. A mind that was pending restarts the process onto it, since the
  schema and the grammar built from it are read once.
- Then the repository is followed. When its branch moves, every reader moves to the
  new commit (`Mind::replace`), the worlds and personalities are read again
  (`MindStores::moved`), and the documents added, changed and removed are reported as
  the folder watcher reports them. A move that changes `projection.yaml` restarts the
  process.
- **Runtime ingest is unchanged.** As with a watched folder, a changed document is read
  into the substrate at the next start; nothing ingests at runtime today, from a folder
  or a commit.

### 8.3 What went

`npcd/src/mind_repo.rs` (the mirror at `<data>/mind/<name>`, its bring-up and its
relaunch), the checkout-based `follow` for a mind in the substrate, and the store folder
at `<data>/git/<name>`: nothing of the mind is on disk outside the substrate.

---

## 9. Agents, with or without a cluster

**Agents are the only way to a repository.** Every fetch, push and command goes through
one; no daemon reads a checkout, a folder or `origin` of its own.

- **With `--cluster`**, the cluster's agents, reaching the one in this process while cut
  off.
- **With no cluster**, the agent in this process alone: the host opens the `agent:`
  file's checkouts and runs their jobs, joined to nothing (`host::cluster::standalone_agent`,
  `Agents::in_process`). Tenants get it as `TenantContext::agent`.
- **No agent at all** is a startup error naming the missing `agent:` file, unless the
  workspace names no repository.
- **A member never acts alone.** With `--cluster`, a member whose cluster member has not
  started, or has no replication to hold the import lease through, fails its load; it
  never falls back to the in-process agent and the `Alone` lease, which would import
  beside the other members.

So zend has no checkout mode: every repository `workspace.yaml` names is read from its
object database, and `origin_watch` is gone. npcd has no `--mind <folder>`: the mind is
`--mind-repo <name>` or there is none, and the folder watcher, the checkout-based record
follow (`record_follow`, `opstore/merge/follow.rs`) and direct folder writes in
production are gone. `MindFiles::folder` and `Mind::new(Some(folder))` remain as test
fixtures only; nothing in a daemon builds one.

---

## 10. What this removes, and what it adds

### Removes

| Before | After |
|---|---|
| A checkout per repository on every zend member | Nothing on the member; agents keep their own |
| npcd's mirror, its 60 s follow, and its fetch-then-relaunch first start | Branches and packs in the log; a pending mind pulled once and restarted onto |
| Every member probing and fetching `origin` | One importer per repository, holding a lease, fetching every 15 s |
| Members serving different mind tips for up to 60 s | One tip at each point in the log |
| A schema that differs between members until each restarts | A restart driven by the branch move that changed it |
| A first fetch of a long history as one agent answer | Slices of 256 commits and at most 32 MiB (`max_commits`, `max_bytes`, §5.1) |
| A `git` program and a `.git` folder per repository on every member | An object database in process, over the log's packs (§4) |

### Adds

- Pack and branch entries as keyed objects, pack bytes as `Blob` records — no new
  record type.
- The importer and its lease; the `Repositories` load step in zend.
- The object database (`zend-vfs/src/odb/`): pack decoding and indexing, libgit2 odb and
  refdb backends, and in-process paths for history, grep, blame, diffs and writes.
- About 1.7 GB of packs across the three repositories in every substrate, growing with
  history, and in a new member's catch-up.
- Never-backwards branches, and the two-look force-push rule (§5.3).

---

## 11. Build order

| Stage | Contents | State |
|---|---|---|
| 1 | Pack and branch records; the importer and its lease; adoption | Built: `substrate-git/` |
| 2 | zend reads every repository from the substrate; repositories loaded whole while zend loads; `origin_watch` removed | Built: `zend/src/substrate_git.rs` |
| 3 | npcd reads the mind at a commit through `MindFiles`; the mirror retired; the watcher's trigger a branch move | Built: `npcd/src/mind/files.rs`, `npcd/src/mind_store.rs` |
| 4 | The in-process object database: packs decoded and indexed in Rust, libgit2 over our odb and refdb backends, every `git` program read and write ported, the derived store on disk deleted, answers bounded by bytes | Built: `zend-vfs/src/odb/`, `zend-vfs/src/library/` |

Runtime ingest of a changed mind document is the next start's, as it was for a watched
folder (§8.2).

---

## 12. Tests

Built:

- **Packs and records** (`substrate-git/tests/members.rs`): an adopted checkout reaches
  another member; a branch waits for its packs; only a descendant replaces a branch
  unless forced; a deleted store is rebuilt from the log; the importer follows origin,
  publishing a rewrite only on two agreeing looks; a member's own push is published
  without the lease; a store reads as a repository for zend.
- **A real substrate** (`substrate-git/tests/member.rs`, feature `member`): a standalone
  checkout loads whole, and a second daemon with no checkout loads it from the log alone.
- **Sliced fetches** (`zend-vfs/src/remote/fetch_bundle.rs`): a long history comes in
  slices and the last moves the tracking ref to origin's commit; a slice across a merge
  carries its whole budget; a byte bound halves a slice until it fits, and a commit
  larger than the bound is answered alone.
- **The object database** (`zend-vfs/src/odb/`): entry headers, inflation and deltas;
  an index byte for byte equal to `git index-pack`'s; a thin pack completed and accepted
  by git; packs let go and read again within the budget; a pack whose index will not
  read is reported and passed over; refs and transactions through libgit2; objects
  written and read back; a flush split under the pack limit; a pack over the limit kept
  as parts that fit; bundles cut and taken.
- **Git-exact answers** (oracle tests beside each library path): history order on equal
  timestamps, path simplification and `--follow`, grep's binary rule, blame, and diffs
  with renames and type changes, each compared with the `git` program's own answer.
- **The workspace** (`zend-vfs/src/workspace.rs`): a repository read from the substrate
  has no folder, opens in every clone once attached, and is attached only once.
- **npcd** (`npcd/src/mind/files.rs`, `npcd/src/mind_store.rs`,
  `npcd/src/engine/watcher.rs`): a folder and a commit read the same; a mind adopted
  into the log reads with no checkout, from a database naming it; an absolute ledger key
  becomes a mind path.

Still to run: end to end on `.6` against `.5` — two members agreeing on a mind commit at
one log point, a pending mind pulled and restarted onto, and a conversation's commits
read on another member.

---

## 13. Open

1. **Collection.** Packs are kept forever (G9). Abandoned conversation commits and
   force-pushed history accumulate. A reachability pass over `GitRef`, `ConvState` and
   open operations, rewriting packs, is a later design.
2. **LFS.** battle-cities' 7.7 GB of LFS content is not imported; its pointer files are.
   Importing it as `Blob` records, keyed by the LFS SHA-256, would fit the existing
   record, if members need the assets.
3. **Catch-up order.** A new member should take a repository's packs before its turns, so
   it can serve that repository's conversations first. Whether catch-up can be ordered by
   record type is a question for `docs/replicated_substrate.md` §7.
4. **`sha256` repositories.** libgit2 cannot hold them (`zend-vfs/src/library/mod.rs`),
   and the database's pack format is SHA-1's, so one cannot be read from the substrate. A
   checkout of one is still served through the `git` program, on an agent.
5. **Agents' own checkouts.** Agents still clone from `origin`. They could seed from the
   log instead, which would make the importer the only reader of `origin` at all.
