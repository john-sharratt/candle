# Git in the substrate: repositories replicate as packs

**Status:** being built in §11's order. Stages 1 and 2 are built: the records, the
derived store, the importer and adoption (`substrate-git/`), and zend behind
`substrate_git: true`. Stages 3 and 4 are not.
**Scope:** where a member's git objects and refs come from. Today every zend member reads
each workspace repository from its own checkout, and every npcd member reads its mind
from a mirror it keeps under `--data`. This design moves both into the substrate: git's
packs and refs become replicated records, and a member reads every repository from what
its log holds. Agents remain the only machines that talk to `origin` (fetch, push) and the
only ones with working trees for commands. zend and npcd use the same mechanism.

§1 is the code as it stands. §2 lists the decisions. §3–§9 are the design. §10 lists what
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
- Every member fetches for itself. One `origin_watch` task per repository probes `origin`
  every couple of seconds and fetches on a change (`zend/src/origin_watch/mod.rs`).
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
| G5 | **A member's object store is derived from its log**: a repository with an empty working tree at `<substrate>/git/<repo>`, filled only from the log's packs, whose record branches are the log's (§4). Stage 4 replaces it with an in-process object database (§11). | `Repo` sends writes and some reads to the `git` program, which needs a repository on disk (§1.1). Unlike today's checkouts and mirror, it has no other source, so it cannot diverge from the log. |
| G6 | **Repositories are names, not folders.** With `substrate_git: true`, a member reads each repository `workspace.yaml` names from its store and needs no checkout (§7). | Members stop needing per-machine clones. |
| G7 | **npcd's mind is a repository like zend's.** npcd reads it through the same store, at a commit. A branch move replaces the file watcher's trigger, and the mirror is retired (§8). | One mechanism for both daemons. The schema, the corpus and an operation's base all come from one commit in the log. |
| G8 | **Standalone daemons use the same read path.** Without `--cluster`, the importer fetches from the checkout's own `origin` (§9). | One code path. |
| G9 | **Packs are never collected in this design**, as `Blob` records are not. LFS content is not imported (§13). | git history is append-only by intent. Reachability collection and LFS are separable work. |

---

## 3. Records

Built in `substrate-git/src/records.rs`. Both kinds are keyed objects (`CustomObject`,
last writer wins by version stamp), held in RAM and replicated; a pack's bytes are a
`Blob`.

| Key | Value |
|---|---|
| `git/pack/<repo>/<sha256>` | `PackEntry { checksum, objects }`: the `Blob` of that SHA-256 is a pack of `<repo>`, named by git as `pack-<checksum>.pack` |
| `git/ref/<repo>/refs/heads/<branch>` | `RefEntry { oid, needs, forced }`: where the branch points (`None` for deleted), the SHA-256s of the packs written with it, and whether it may replace a non-descendant |

- **Packs.** `Repo::pack_objects(tips, known, 64 MiB)` packs what the tips reach beyond
  the log's branches (`zend-vfs/src/remote/packs.rs`). At most 64 MiB, so a pack fits a
  relocation stripe (`persistence/stripes.rs:14`) and replicates as 64 `EVENT` fragments
  or fewer. candle's 1.3 GB adopts as about 21 packs. A blob already held is not written
  twice.
- **Entries replicate apart from their blobs.** A branch whose `needs` are not all held
  waits; a sync takes the pack once its blob arrives.
- **Only forward.** A writer checks that a new tip descends from the log's before writing
  it (`substrate-git/src/publish.rs`), and a store only moves a branch forward unless the
  entry says `forced`. Last-writer-wins can still let a stale write land briefly; the next
  import tick (2 s) corrects it.
- Conversation branches are not entries. A conversation's branch per repository stays in
  its `ConvState` (`zend/src/conv_branches.rs`).

---

## 4. The derived object store

`<substrate>/git/<repo>` (`substrate-git/src/store.rs`) is a repository with an empty
working tree, so `Repo` opens it unchanged:

- `HEAD` names `refs/heads/substrate-store`, a branch never made, so nothing is checked out
  and no branch the log moves is the current one.
- **The record is `origin`'s tracking refs**, `refs/remotes/origin/<branch>`, which is
  where zend reads a repository's branches when it has an origin
  (`Repo::record_branches`). `Store::sync` takes each pack it lacks once its blob has
  arrived (`git index-pack`), then moves each record branch whose packs are held,
  forward only unless `forced`.
- **Two remotes, one URL.** `origin` is the record and where pushes go. `import` is where
  the importer fetches. A fetch therefore never moves what zend reads before the log does.
- Local branches are the member's own and never touched.
- Deleting the folder loses nothing published: the next sync rebuilds it.

Commands that need files run on agents, which keep their own checkouts
(`docs/replicated_substrate.md` §11).

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
3. **Syncs** the store up to the log. A record branch that moved wakes zend's ingest
   worker.

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
the log's. A push that `origin` accepts moves the store's record as it lands; the
member's next tick publishes it, packs first. Since the push was compare-and-swap on
`origin`, the new tip descends from the log's, and only-forward keeps it.

---

## 7. zend

Built behind `substrate_git: true` (`--substrate-git`), in `zend/src/substrate_git.rs`.

- **Manifest.** Unchanged: `workspace.yaml` names repositories. With the setting, the
  manifest is read without requiring the folders (`Workspace::manifest`), and each
  repository's folder is its store (`Workspace::with_dir`). The name is unchanged, so every
  tool argument, workspace-relative path and ingest key is too.
- **Everything that opens `repo.dir`** — `GitSource` for the lower layer, branch ingest,
  conversation branches, pushes through `AgentPush` — opens the store with no other
  change.
- **Starting.** After the cluster join and catch-up, before anything reads a branch, each
  repository is brought up to the log once: a checkout at `<workspace>/<name>` the log
  has not seen is adopted (§9), and the importer ticks. Then a `substrate-git` thread
  ticks every 2 s.
- **Conversation bases.** A store has no local branch, so a conversation's base is the
  record's `main` (else `master`) before `HEAD` (`zend/src/conv_branches.rs`).
- **`origin_watch`** is not started; the importer wakes the ingest worker instead.
- **Commands** (`run_command`) already run on agents with the conversation's overlay.
- **Not yet:** `HELLO.repos` still advertises each repository, now with the URL
  `agent:<name>` on every member, so routing is unchanged among members with the setting.

---

## 8. npcd

### 8.1 Reading the mind

A `MindFiles` reader answers `read(path)` and `list(dir)` at a commit, through the same
`GitSource` as zend. Every module that reads the mind folder today reads through it: the
schema (`npcd/src/engine/schema.rs`), the registries, the corpus, the console and API
reads, mission generation, lifegen. An operation's store keeps its lower layer at its
pinned commit, now resolved from the derived store.

### 8.2 Following the record

The mind's branch is a `GitRef`. When it moves, npcd diffs the old tree against the new
one (`git diff-tree`) and hands each changed path to the watcher's existing reconcile:
unchanged costs nothing, changed tombstones and re-ingests, deleted is reported
(`npcd/src/engine/watcher.rs`). Only the trigger changes, from file events to the
`GitRef`. A move that changes `projection.yaml` restarts the tenant onto the new commit
(`TenantContext::stopping`), since the grammar is built once at start.

### 8.3 What goes

`<data>/mind/<name>`, `MindRepo::bring_up` and the 60-second follow. A first start no
longer runs without a mind and relaunches: it waits in its loading phase until the
mind's `GitRef` and packs are held, as it waits for any catch-up.

---

## 9. Standalone and adoption

- **Standalone** (no `--cluster`): the importer fetches with `FetchHere` from the
  repository's local folder: `--mind <dir>` for npcd, the workspace folders for zend.
  The read path is the cluster's. A folder that is not a git repository is imported as a
  synthetic commit of its tree, so `--mind` keeps working on a plain folder.
- **Adoption.** The first start of a member on this build packs each repository it can
  read locally (`git pack-objects --all --max-pack-size=64m`) and writes the packs and
  refs once, under the lease. Duplicate packs from two members adopting at once are kept,
  and harmless (§3.1). A member with nothing local waits for the packs to replicate.

---

## 10. What this removes, and what it adds

### Removes

| Today | After |
|---|---|
| A checkout per repository on every zend member | Nothing on the member; agents keep their own |
| npcd's mirror, its 60 s follow, and the "no mind, fetch, relaunch" first start | A `GitRef` and the packs it needs |
| `--mind` versus `--mind-repo`, and the refusals between them and `--cluster` | One read path; the flag says only where the importer fetches from |
| Every member probing and fetching `origin` | One importer per repository, holding a lease |
| Commits in one member's `.git` until pushed and fetched | Packs replicated before anything names them |
| Members serving different mind tips for up to 60 s | One tip at each point in the log |
| `HELLO.repos` restricting routing | Every caught-up member holds every repository |
| A schema that differs between members until each restarts | A restart driven by the `GitRef` that changed it |

### Adds

- Two record types, with the `Blob` machinery reused for `GitPack`.
- The importer and its lease.
- The derived store, until stage 4.
- About 1.7 GB of packs across the three repositories in every substrate, growing with
  history, and in a new member's catch-up.
- The descendant rule for `GitRef`, and the force-push handling in §5.3.

---

## 11. Build order

| Stage | Contents | Size |
|---|---|---|
| 1 | `GitPack` and `GitRef` records; the derived store; the importer and its lease; adoption. zend behind a flag. | Medium. Record types follow `Blob`; the importer reuses `Fetch` and `remote/fetch_bundle.rs` |
| 2 | zend reads from the derived store: manifest names, `GitSource`/`Repo` opened on it, branch ingest, objects emitted by every write (§6), `origin_watch` retired, `HELLO.repos` no longer restricting. Members drop their checkouts. | Medium. Seven files in `zend/src` open repositories today |
| 3 | npcd: `MindFiles` through the ~25 modules that read the mind, the watcher's new trigger, schema restart on a `GitRef` move, the mirror retired, `--mind` through `FetchHere`. | Large. The widest change, spread across npcd |
| 4 | An in-process object database: the remaining `git` program writes and reads (`merge_tree`, `commit`, `ref_txn`, `apply`, `log`, `grep`, `blame`) ported to libgit2 over packs read from the log, and the derived store deleted. | Large, and optional. Stages 1–3 stand without it |

Each stage leaves the cluster working: stage 1 adds records nothing yet reads, stage 2
moves zend, stage 3 moves npcd. Members on an older build ignore records of types they do
not decode, deferring them as today (`docs/replicated_substrate.md` §6.5).

---

## 12. Tests

- **Packs.** A pack written on one member is held, verified and indexed on another; a
  duplicate is dead on arrival; a corrupt one is refused on read; an import over 64 MiB
  splits.
- **Refs.** A `GitRef` arriving before its packs waits for them; a descendant wins
  whatever the stamps; a stale importer record cannot rewind a member's push; a forced
  move needs two agreeing probes.
- **Derived store.** Deleting it and reopening rebuilds the same refs and objects; a pack
  or ref the log lacks is removed at open.
- **zend.** A conversation committed on one member continues on another with its commits
  readable, before any push; branch ingest re-reads nothing after the switch; a member
  with no checkouts serves every repository once caught up.
- **npcd.** Two members report the same mind commit at the same log point; a
  `projection.yaml` change restarts the tenant onto it; a changed corpus file is
  tombstoned and re-ingested once, by the world's authority.
- **Standalone.** `--mind` on a plain folder and on a git checkout both read through the
  substrate, with the same results as before.

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
4. **`sha256` repositories.** libgit2 cannot hold them (`zend-vfs/src/library/mod.rs`).
   The derived store serves them through the `git` program, but stage 4 would not cover
   them.
5. **Agents' own checkouts.** Agents still clone from `origin`. They could seed from the
   log instead, which would make the importer the only reader of `origin` at all.
