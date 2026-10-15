# Git in the substrate: repositories replicate as packs

**Status:** stages 1–3 are built (§11): the records, the derived store, the importer and
adoption (`substrate-git/`), zend behind `substrate_git: true`, and npcd's mind read from
the substrate with `--mind-repo`. Stage 4 is not, and nothing has run end to end yet.
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

Built in `npcd/src/mind/files.rs` and `npcd/src/mind_store.rs`.

- **`MindFiles`** is the mind's files at one commit of its store, or in a plain folder.
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
- **Operations' stores** read through the store's record: `GitSource` defaults to
  `origin/main` (else `master`) when the checked-out branch has no commit, which is a
  store's case, and `Merge` publishes on that branch through the agent.
- **The ingest ledger** is keyed by mind paths. A key an earlier build wrote as an
  absolute path under a mind folder is cut to its `layers/…` part when the ledger
  opens, so a mind that moved into the substrate retires nothing.

### 8.2 Startup and following

- `--mind-repo <name>` (with `--cluster` and `--data`): the tenant opens the substrate
  first, brings the store at `<data>/git/<name>` up to the local log, and reads the
  schema and the registries from the branch's commit. A log that holds no such branch
  yet starts the daemon on the bundled schema, with the mind pending.
- After the cluster join, `load_mind` (`engine/runtime.rs`) loads the repository whole
  through `substrate_git::member::Repos`: pulled through an agent by the lease holder
  when no member has. A mind that was pending restarts the process onto it, since the
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
relaunch), and the checkout-based `follow` for a mind in the substrate: `MindStores`
marked `read_from_substrate` checks nothing out.

---

## 9. Standalone and adoption

- **zend standalone** (`substrate_git: true`, no `--cluster`): each store fetches from
  its checkout's own `origin` with this machine's credentials, and the checkout's
  branches are adopted once (`substrate_git::adopt`). With a cluster nothing is adopted:
  the agents' view of `origin` is the only source, so a stale local clone never leaks
  in.
- **npcd `--mind <dir>`** reads its folder as it stands (`MindFiles::folder`), and the
  folder watcher stays. A standalone author edits the folder and sees it at once, which
  a commit would not give.

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
| A first fetch of a long history as one agent answer | Slices of 256 commits (`max_commits`, §5.1) |

### Adds

- Pack and branch entries as keyed objects, pack bytes as `Blob` records — no new
  record type.
- The importer and its lease; the `Repositories` load step in zend.
- The derived store, until stage 4.
- About 1.7 GB of packs across the three repositories in every substrate, growing with
  history, and in a new member's catch-up.
- Only-forward branches, and the two-look force-push rule (§5.3).

---

## 11. Build order

| Stage | Contents | State |
|---|---|---|
| 1 | Pack and branch records; the derived store; the importer and its lease; adoption | Built: `substrate-git/` |
| 2 | zend reads from the derived store behind `substrate_git: true`; repositories loaded whole while zend loads; `origin_watch` replaced | Built: `zend/src/substrate_git.rs` |
| 3 | npcd reads the mind at a commit through `MindFiles`; the mirror retired; the watcher's trigger a branch move | Built: `npcd/src/mind/files.rs`, `npcd/src/mind_store.rs` |
| 4 | An in-process object database: the remaining `git` program writes and reads ported to libgit2 over packs read from the log, and the derived store deleted | Not started; optional |

Not yet done: `HELLO.repos` still advertises repositories (§7), and runtime ingest of a
changed mind document is the next start's, as it was for a watched folder (§8.2).

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
  slices and the last moves the tracking ref to origin's commit.
- **npcd** (`npcd/src/mind/files.rs`, `npcd/src/mind_store.rs`,
  `npcd/src/engine/watcher.rs`): a folder and a commit read the same; a mind adopted
  into the log reads with no checkout, its store holding no file; an absolute ledger key
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
4. **`sha256` repositories.** libgit2 cannot hold them (`zend-vfs/src/library/mod.rs`).
   The derived store serves them through the `git` program, but stage 4 would not cover
   them.
5. **Agents' own checkouts.** Agents still clone from `origin`. They could seed from the
   log instead, which would make the importer the only reader of `origin` at all.
