# Zend: Ingesting Branches

**Status:** Built and unit-tested (§11). Not yet run against the live daemon: the end-to-end pass over a real model is `zend/tests/zen_code_phase12_smoke.rs` (ignored, loads the model).
**Scope:** How the `repo_map` and `code_reading` layers are populated from the branches origin holds rather than from a repository's folder; how a change on origin is noticed within seconds without being rate-limited; how an ingested unit is keyed by its content so a branch moving forward re-reads only what changed; and how a conversation retrieves only the ingested content that matches its own base.

It replaces the folder walk of `docs/zend_workspace_execution.md` §5.1 and refines §4.1 (which commit a conversation pins) and `docs/zend_git.md` §7.8–§7.9.

---

## 1. Summary

Every repository in the workspace whose record is on `origin` (`docs/zend_git.md` §7.8) is ingested **from its branches as origin holds them**: each `refs/remotes/origin/*` ref the ingest's whitelist names (§3.1), read in place, with no local branch made for any of them and nothing written to the checkout. Origin is imported by one member per repository, the holder of the lease `git/<repo>`, which fetches through an agent every 15 s and publishes what moved to the substrate's log; every member learns the new tips by replication, and a record branch that moves wakes the ingest worker (§4, `docs/substrate_git.md` §5). No member probes origin of its own.

The ingest lists each branch tip's tree (one `ls-tree` per distinct tree, no file contents read), filters it by the same rules the folder walk used, and derives the units the two layers ingest. **A unit is keyed by what it shows, never by the commit it was found on.** A file's key is its repository, its path and its **blob id** — git's own hash of exactly the bytes `file_read` returns. A folder's key is a hash of its listing — the entries on its first page and how many it holds — and the blob ids of the files whose content its turns show. A commit only says where to read a unit's bytes from. So a branch moving forward re-ingests only the paths whose blob changed there; a file untouched since an older commit is never read again; and the same file on twenty branches is one conversation.

Whatever is ingested but no longer found on any whitelisted branch is tombstoned. A conversation retrieves only the ingested units whose keys its own base holds, so it never reads another branch's version of a file, and a conversation pins **origin's copy** of its branch — the same copy the ingest covers.

---

## 2. Goals and non-goals

### Goals

- Every whitelisted branch on origin ingested (§3.1), within `--max-depth` and the `--ingest-dir` scope, whatever is checked out.
- A change on origin noticed within seconds (the importer's fetch is every 15 s, §4.1), by one member for the whole group, without spending REST or GraphQL quota.
- One ingest per distinct unit of content, however many branches or commits carry it.
- A conversation's retrieval and fast path agree with the files it actually reads.
- Nothing ingested outlives the branches that hold it.

### Non-goals

- **Folders that are not git repositories.** They have no branches. A workspace repository that is a plain folder is read by conversations from disk (`docs/zend_workspace_execution.md` §4.1) and is not ingested. The uploads repository is ingested by its endpoint (§9).
- **Webhooks.** Creating one needs admin on the repository and a public address; GitHub's own forwarding is "not supported for use in production" and it offers clients no documented push channel. The watcher's cadence makes them unnecessary.
- **Remotes other than `origin`.** `origin` is the record (`docs/zend_git.md` §7.8); `upstream` and any other remote are the developer's.

---

## 3. The branches

### 3.1 The record

A repository's branches are its **record branches**: with an `origin` remote, every `refs/remotes/origin/<b>` except the symbolic `HEAD`; with none, every local branch. `Repo::record_branches()` returns them as `(BranchName, Oid)`, in name order. They are read locally and never touch the network — the importer (§4) is what keeps them current.

**The ingest reads only the record branches its whitelist names.** `zend/src/prompts/ingest_branches.yaml` lists branch names, matched whole and applied to every repository; `branch_ingest::record_branches` drops every other branch before a tree is walked (`keep_monitored`, `branch_ingest/monitored.rs`). A branch off the list contributes no unit, so every reading only such a branch held is planned as gone and tombstoned on the next pass (§1), and maintenance reclaims its bytes; a version a listed branch shares keeps its reading. The list ships as `main` and `master`. Reading every branch cost a reading per distinct file version on each of them: on BlackWorld (2026-11-09) ~127 branches of `candle` held 3,562 `code_reading` conversations for the ~1,177 files of `main`, and the substrate reached 694 GB.

No local branch is created for any of them. The ingest reads trees and blobs by id, which needs no branch at all, and a conversation that switches to one gets a local branch the ordinary way (`git_switch` pulls it, `docs/zend_git.md` §7.8).

### 3.2 What a conversation pins

A conversation's store pins its branch's **record** on first read (`GitSource::base_at`): `refs/remotes/origin/<b>` when origin has the branch, else `refs/heads/<b>`. That is the copy the ingest covered, the copy `git_status` already compares against, and the copy a commit must descend from to publish. A commit the developer made locally and never pushed is not on the record and is not what a new conversation starts from.

A conversation's base can therefore be ahead of its local branch — origin moved, the importer fetched, the local branch was left where it was. (A repository read from the substrate has no local branch and no sandbox, its commands run on agents — `docs/substrate_git.md` §7; what follows is for a repository with a checkout.) The sandbox, which checks the conversation's branch out, brings the local branch up first (`Repo::follow_record`): when the base is in origin's copy's history and the local branch is in the base's, the local branch is fast-forwarded to the base (created there if it does not exist), under a compare-and-swap. It does so only once it holds the checkout's lock and has set the owner's state aside, so no other job's checkout moves under it, and the owner's checkout comes back as a branch that moved on rather than as uncommitted work reverting what origin gained: the owner's own changes — staged, unstaged, untracked — are laid onto the branch as it now stands, against the commit they were made on (`checkout/preserve/snapshot.rs`). When the owner's checkout is on that branch with uncommitted changes to a file the followed commits change too, nothing moves and the job is refused, as `git merge --ff-only` refuses (`CheckoutError::OwnWorkInTheWay`). Any other disagreement is refused as before (`docs/zend_git.md` §7.7).

A local branch holding commits origin's copy lacks — the developer's own, never pushed, whether origin has moved on as well or not — is **set aside** for the job instead (`Preserved::set_branch_aside`). The job is judged against the part of the branch origin holds (`Repo::local_branch`: the tip, or where the unpushed commits leave origin's copy), so a conversation made on that commit, or on a later one of origin's, runs, and one origin has since moved past is refused as behind. Under the checkout's lock the branch's tip is held by a ref of the preservation's own (`refs/zend/preserved/<id>/branch`) and journalled, then the branch is moved onto the conversation's base with a compare-and-swap; the job runs on exactly what the conversation reads, without the unpushed commits; and the restore puts the branch back at its own commit — unless it only moved on from there since — before `HEAD` and the owner's changes come back over it. A job that fails or is abandoned puts it back the same way, and a crash leaves the journal, which zend finishes when it starts (`Sandbox::recover`, as it builds each repository's sandbox) and the next job would finish otherwise. The unpushed commits are never built on, published or lost.

---

## 4. Following origin

zend reads every repository from the substrate (`zend/src/substrate_git.rs`, `docs/substrate_git.md` §7): a repository's tracking refs are records in the replicated log, and nothing on a member fetches origin for itself. The per-repository origin watcher that once probed origin every couple of seconds (`zend/src/origin_watch/`) is gone.

### 4.1 The importer

A `substrate-git` thread ticks every **2 s** (`TICK`, `substrate-git/src/member.rs`). Each tick, the member holding the lease `git/<repo>` (120 s, renewed every 30 s) imports: it fetches every branch of origin through an agent, the only machine that talks to origin, and publishes each branch that moved ahead of the log as packs and refs. The fetch is made at most every **15 s** (`FETCH_EVERY`). Every member also publishes its own pushes and syncs its store up to the log; a record that changed since the follower last reported it — a sync, or this member's own push landing — wakes the ingest worker (`repositories.on_change`, `zend/src/session/lifecycle.rs`), and registering the callback wakes it once.

A repository the log already names is not fetched at startup; it is whole from the log and origin's newer commits are the importer's. A branch origin deletes or rewrites is published as forced only once two consecutive ticks agree (`docs/substrate_git.md` §5.3).

### 4.2 A tool round

A commit zend itself publishes moves the store's tracking ref as the push lands, so no import is needed to see it. A `git_*` tool round that can write a branch or fetch one (`git_commit`, `git_merge`, `git_ref`, `git_switch`, `git_reset`, `git_push`, `git_fetch`; `BRANCH_MOVERS`, `branch_ingest/mod.rs`) wakes the ingest worker itself.

### 4.3 The ls-refs probe

`zend-vfs/src/remote/probe/` still holds the protocol half of a git v2 `ls-refs` probe over HTTPS — pkt-line framing (`pkt_line.rs`), the request body and the response parsed to branch tips (`ls_refs.rs`), an origin's HTTPS form (`url.rs`) — with its tests. No daemon calls it: nothing on a member probes origin.

---

## 5. The walk

`zend/src/branch_ingest/walk.rs`. For each record branch of each git repository, the tip's tree is listed once (`ls_tree_all`, every entry with its size). A tree two branches share is listed once. What is kept is exactly what the folder walk kept, decided from the listing alone:

- **Files only**: mode `100644` or `100755`. A link is not a file a tool reads, and a submodule (`160000`) is another project.
- **Nothing hidden**: no path component beginning with `.`.
- **Nothing protected**: no `secrets` component — the VFS refuses to serve such a path, so nothing may ingest one.
- **The extension allowlist** (`Language::from_extension`, plus `go.mod` / `go.sum`).
- **The size cap**, `MAX_FILE_BYTES` (1 MiB), from the listing's size — above the largest hand-written source in the workspace (~710 KB), below the dumps that are text only by extension (SASS listings at 2.3–2.8 MB, a 13 MB JSON data file), each of which would be 600K+ tokens that no reading chain finishes. A file already ingested and now over the cap is retired, since the kept list passes the same gate.
- **The scope**, `--ingest-dir <layer>=<repo>/<folder>`: only paths under that folder.
- **The depth bound**, `--max-depth N`: path components below the walk's start — the repository's root, or the scope folder — so `1` is its own files. It bounds what is ingested, not what is kept (§7.2).
- **Nothing ignored**, `--ignore <pattern>` (repeatable; `motherd.yaml`'s `zend: ignore:`): a workspace key `<repo>/<path>` in which `*` matches any run of characters, `/` included (`zend/src/branch_ingest/ignore.rs`). Unlike the depth bound, it bounds what is kept as well: the scope at full depth refuses an ignored path too, so a unit of one is in neither the ingest nor the retain list, and the pass tombstones the reading the substrate held of it (§7.2) — it is ejected, not merely left unread. A `…/*/*` pattern covers only what lies below a folder's own files. `motherd.yaml` ships the patterns that keep out the records of past states — every measured result (`candle/docs/results/*`) and every archived document (`candle/docs/archived/*`), whose paths and names no longer exist in the code and which read as current would answer with them — beside the dependency battery's questions and the gates' expected files (`candle/zend/examples/dependency_battery/*`, `candle/docs/provenance_expectations/*`), upstream examples (`candle/candle-examples/examples/*/*`) and the NPC map data (`candle/npc-map/maps/*/*`).

Binary content cannot be told from a listing. It is sniffed when a file is about to be ingested (§7.3).

The walk's output is a **corpus**: every distinct `code_reading` unit and every distinct `repo_map` unit on any branch, each with one commit it was found on (`at`), the repository's default branch's first (`main`, else `master`), then the other branches in name order.

---

## 6. Content keys

Every key carries a **version** (`KEY_VERSION`, `branch_ingest/keys.rs`, now 6):
the version of what a unit's turns say beyond its content — the calls and
responses rendered around it. A change to that rendering bumps it, which re-reads
every unit: each new key is queued, the old conversation stays readable until its
replacement commits, and the next pass tombstones it (§7.2); a unit past the depth
bound, which nothing re-reads, is tombstoned at once. Version 2 counted pages
from 1; version 3 read a file's parts under inline anchors; version 4 shows each
part as a fenced block, its lines numbered from `+1`, and reads a file by
`lines` ranges of whole parts (`docs/archived/zen_coding_challenge_fixes.md` Cause 4);
version 5 seals every unit that is not a priming-chain link behind the system
prompt alone, never parented on the chain (`docs/zend_workspace_execution.md` §6);
version 6 names a part's `scope` only for a part with a label of its own, so no
paragraph opens on its first line cut at 72 characters and marked `…`. A
file unit records the file's line count (`lines`), which the working set checks
the chain's reads covered before it serves the reading.

Every key states its version as a `v<N>:` prefix — a key with none is version 1,
the form before versions — and **a member judges only conversations at its own
version or older**. One under a newer version was written by a peer upgraded
first: this build cannot key it, so it is never retained, and planned as this
member's it would be tombstoned — while the peer tombstones this member's older
units, and each re-ingests what the other removed. A newer unit is therefore
neither a holder nor tombstoned until this member upgrades and plans it as its own
(`plan`).

### 6.1 A file

```
v<version>:<repo>/<path>@<blob id>
```

`v2:candle/zend/src/main.rs@3f73f8722a390439c029daf7748a9ab24791a016`. The blob id is git's hash of the file's committed bytes, which are exactly what `file_read` returns: the store reads the blob and nothing converts it. The path is part of the key because the ingested conversation names it — in its opening request and in every `file_read` header — so the same bytes at another path are another conversation. (Measured on this repository: 22 branch tips hold 8,120 distinct path-and-blob pairs and 8,024 distinct blobs, so the path costs almost nothing.)

The key is known from a tree listing alone. Deciding whether a file is ingested reads no file.

### 6.2 A folder

A folder's key is `v<version>:` and then the hash below. A folder's turns show its listing (`file_list`'s first page, and how many entries the folder holds) and the manifest hint in its request (`(crate: candle-nn)`) — nothing else. The chain is one `file_list` round-trip and the summary; no file of the folder is read, so no file's content is part of what it shows (`repo_scan/render.rs` records why a README/module-doc read was removed). The listing is what `file_list` shows, not what the layer reads: every file and subfolder but hidden ones, a `LICENSE` or a `kernels/` folder as much as a `.rs` file. Its key is the SHA-256, in hex, of:

```
v<version>\n        the key version, `v2`
<dir>\n             workspace-relative, `candle/zend/src/`
<total>\n           how many entries the listing holds
<entry>\n           one per entry on the first page, workspace-relative, a folder ending in `/`, in the listing's order
\0hint\0<hint>       the hint the request shows, rendered (`crate: candle-nn`)
```

the last present only when one of the folder's manifests gives a hint — the first that does, in path order. **The hint, never the manifest's bytes**: the request shows the hint, so two versions of a `Cargo.toml` that give the same one — a dependency bumped, a version raised — make the same turns and must make the same key. Keyed on the manifest's blob, every such edit split the folder into another conversation saying the same thing, and branches that share a lineage but not every manifest byte multiplied it: `candle/` stood ten times over seven distinct listings across its 22 branches. The walk reads each distinct manifest version once, cached by blob id (`branch_ingest::manifest::Hints`), and the retrieval scope derives the same key the same way. The sizes the listing prints beside each file are left out for the same reason: they move with every edit of a file the folder only names, and a folder would be read again for each one while what its summary rests on — which files and folders it holds, what its manifest says — stood still. So an edit to a file the folder only names — a README or module root included — moves nothing, and neither does a manifest edit that leaves its hint alone; an entry added or removed (on the page, or past it through the count) or a changed hint does. The manifest hint states the folder's role and never a magnitude of the checkout: a Cargo workspace is `Cargo workspace root`, not a member count. Every folder unit is inside a repository: `file_list` lists inside one repository, so there is no unit for the workspace itself.

### 6.3 Where the keys are kept

On the unit's conversation, as metadata: `content_key` (the key), plus `path`, `blob` and `lines` (the line count `file_read` reports for it) for a file, `dir` for a folder, and `kind`. `branches` names every branch whose tip holds the unit's version, the default branch first — the walk collects it as it meets the unit on each tip. It is not part of the key: it is written when the unit commits and rewritten by each pass whose walk finds a different set (`branch_ingest/tie.rs`), since branches move without the content changing. A file's conversation also records `commit`, the commit it read the file at: written once and never moved, since it names the version the conversation holds. A pass gives one committed without it a commit the walk found holding the same blob. The substrate viewer's layer view shows `commit` on a file's row — a file's version is shared across branches, so its commit is what tells two readings of one path apart — and `branches` on a folder's row. `path`/`dir` is written when the conversation is created and `content_key` only once its ingest succeeds, so a conversation with the first and not the second is a partial. An attempt that fails retires its own conversation; one a crash left is retired at boot, before any pool runs — a pool never retires another's, since an attempt in flight carries exactly the same metadata. A conversation from before this design has no `content_key` and is retired the same way, uploads included: without a key it is in no conversation's scope (§8.2).

---

## 7. The pass

`zend/src/branch_ingest/mod.rs`, run by the ingest worker (`zend/src/ingest_worker.rs`) on every wake: once at startup over the tracking refs as they stand, then after every fetch that changed a ref.

### 7.1 Live, committed, queued

1. **Live** — the corpus's keys (§5), in two lists: **ingest**, within the scope and the depth bound — what the layer reads — and **retain**, the same scope at full depth — what the layer keeps. Every ingest key is a retain key: a unit's key does not depend on the bound (a folder's listing and hint are the same whatever depth it is walked to), so the full-depth walk finds exactly the keys a deeper pass ingested.
2. **Committed** — every live conversation of the layer carrying a `content_key`; for `code_reading`, only one whose chain **finished** (`candle_conversation::chain_health`). A file's key is written once its conversation stops calling tools, which is not the same as answering: a read cut off mid-deliberation, with no tool call and no summary, carries the key all the same. Such a chain is not committed, so its key is queued again; its conversation is deferred like a dead key's (§7.2) and retired once the rebuild commits. The resume snapshot and the working set's coverage check read the same predicate (`zend/src/code_read/chains.rs`).
3. **Queued** — ingest keys not committed, in corpus order.
4. **Dead** — committed keys not retained.

### 7.2 Tombstones

A dead key is tombstoned, except while its path (or folder) has a queued key in the same layer: then it stays until that replacement commits, so a path is never missing from the layer while it is being re-read. A path deleted everywhere has no replacement and goes at once. Uploads (§9) are never swept here.

"In scope" is literal: only the layers this boot ingests (not `--disable-layer`), and only within the scope folder the pass walked. A key outside it is not found, and so is dead. The depth bound is not part of it: a unit past the bound is retained for as long as a branch holds it — its branches and commit kept current like any other — so narrowing `--max-depth` drops nothing already ingested. Past the bound, only what no branch holds at any depth goes, and an old version of a deep file goes with nothing to wait for, since its new version is past the bound too and is never queued. Retrieval derives folder units at full depth for the same reason (§8.2): a retained folder stays retrievable.

A repository whose branches could not be read this pass — a folder with a `.git` that would not open, or a tree that would not list — **sits the pass out whole**: none of its committed units is dead, since nobody looked for them, and none of its units is queued. Whether a folder is under git is read from the folder (a `.git` in it), never from a failure to open it.

### 7.3 Ingesting a unit

A file's hidden conversation and a folder's chain run exactly as before (`code_read`, `repo_scan`) with one change: the tools they call read **the commit the unit was found on**. The unit's conversation is given a file set whose store for its repository is pinned to `at` (`RepoFiles::fresh_at`), so its `file_read` and `file_list` return that commit's bytes — the bytes the key names. A folder's manifest hint is read from its blob.

Before a file's conversation is minted, its blob is sniffed; a binary file is skipped and its key remembered for the life of the process, so it is not read again — and from the next pass on it is not a unit at all, so an older version of its path does not wait on it as a replacement.

On success the conversation's `content_key` is written, which commits it and marks the retrieval index stale (§8.2); then each other committed conversation of the same path (or folder) whose key is dead is tombstoned. Conversations of the same path with a live key — the file as another branch holds it — are left alone, and so are those with no key yet: another worker's attempt at the file as another branch holds it. A key that cannot be written leaves the attempt uncommitted, and it retires its own conversation.

---

## 8. Retrieval

### 8.1 No served reads

**A `file_read` and a `file_list` always run, and their answers are prefilled.** There is no screen that answers a call from an ingested unit (`zend/src/fast_path.rs` is gone): a listing is short, so serving it saved little, and handed the folder's `repo_map` unit as the answer to its own listing the model lost track of whether it had listed the folder at all. Ingested units reach a dialogue as seeds and through the gather (§8.2). A file's key is still looked up from the conversation's base — the blob id its base tree holds for that path (`VfsStore::content_id`), no read and no hash — to find which finished reading holds the version a `file_read` returned, so the loop can withhold those readings for the rest of the turn (`zend/src/working_set/corpus.rs`), and the working set checks the chain's reads covered the file's line count (`lines`, §6.3) before it serves the reading (`zend/src/working_set/coverage.rs`). A finished chain (`chain_health`) gates a reading. How a reading is held in context — the pool, seeds, the withheld set, the provenance tier beside them — is the working set, `docs/zend_working_set.md`.

### 8.2 The gather

`repo_map` and `code_reading` are in the provenance gather, **scoped to each conversation's base**. The engine marks their groups as scoped (`ConversationEngine::mark_group_scoped`), and a scoped group's candidates for a target are exactly the timelines the target's scope names (`set_retrieval_scope`) — none when it names none. Both the belief scan (`score_belief_groups`) and projection assembly (`TargetedRead::group_turns`) apply it, so an out-of-scope conversation is neither scored nor selected. An ingest conversation's own self-local projection is unaffected.

Before each dialogue turn zend computes the target's scope (`zend/src/retrieval_scope.rs`): for each repository, the keys of every file and folder in the conversation's base tree, derived by the same filters and unit rules as the walk — at full depth, like the retain list (§7.2) — looked up in an index of committed keys to timelines. A path the conversation has changed contributes no file key, and the folder holding it no folder key. Every upload is in scope.

**A file version not yet read stands at its path's latest reading.** A commit gives a file a new blob, and the pass reads the new version while the old one stays (§7.2); looked up by the new key alone, the file was out of every scope on the branch until the read committed — and for as long as it kept failing. On BlackWorld (2026-11-19) `candle-conversation/src/conversation.rs`, three lines of imports changed, was out of every conversation while its re-reads failed, and a question whose answer was a call on its fifteenth page was declined. So a file key with no committed reading is looked up by its path instead, and the path's latest committed reading — of whichever version, the one begun last as its timeline id orders it — stands in until the version's own reading commits and replaces it (`IngestIndex::latest_of`). This is the gather only: a `file_read` always returns the base's bytes (§8.1), and the working set's seeds are looked up by the exact key. A folder's key already covers only its listing and hint (§6.2), so an edit inside a file moves no folder.

Working the scope out reads no file and **takes no base**. A repository the conversation has not read yet is scoped at the base its first read would take (`VfsStore::peek_base`), and is left unpinned. A tree's units are kept per (repository, tree, index generation), and listing the tree is the only git work: a tree some conversation was already scoped on under this index costs nothing. The whole scope of a conversation that has changed nothing is kept per set of base trees and **shared**: every such conversation on the same bases holds the same two sets. Both caches keep the sixteen most recently used entries.

The index is rebuilt **lazily**. A unit committing marks it stale, and the next turn rebuilds it before scoping — at most once a second, since units commit continuously through a pass, and one rebuild at a time, publishing under the next generation; the stale mark is cleared before the substrate is read, so a unit committing during a rebuild leaves the index stale again rather than being missed. An upload rebuilds it at once, since the very next turn may ask about the file. A turn's scope is dropped when its conversation is archived, tombstoned, or evicted after a turn that ended without an answer; the next turn of a conversation sets its own.

---

## 9. Uploads

The uploads repository is a folder, ingested by its endpoint, never walked and never swept by the pass. Its files are keyed the same way: the blob id is computed from the bytes on disk (`ObjectFormat::blob_id`), so the key form is one across both. Uploads are in every conversation's scope.

---

## 10. Placement

```
zend-vfs/src/
  remote/probe/
    pkt_line.rs       pkt-line encode / decode
    ls_refs.rs        the v2 ls-refs request body; the response parsed to branch tips
    url.rs            an origin's HTTPS form
  read/record.rs      Repo::record_branches, Repo::record_tip, Repo::local_branch
  sandbox/follow.rs   a job's branch followed to its base, or its unpushed commits set aside
  checkout/preserve/  Preserved::set_branch_aside, and the restore that puts it back
  origin/follow.rs    Repo::follow_record
  types/oid.rs        ObjectFormat::blob_id
  vfs/git_source.rs   base_at pins the record
  vfs/store.rs        VfsStore::content_id, peek_base, tree_at
  files.rs            RepoFiles::fresh_at
substrate-git/src/
  member.rs           the tick, the lease, FETCH_EVERY
  import.rs           the importer
zend/src/
  substrate_git.rs    the repositories read from the substrate; the thread that ticks
  branch_ingest/
    mod.rs            the pass (§7)
    walk.rs           tree listings → corpus (§5)
    filter.rs         which files a layer reads (§5)
    keys.rs           content keys (§6)
    units.rs          folder units from a listing
    manifest.rs       manifest hints from bytes
    plan.rs           live, committed, queued, dead (§7.1–§7.2)
  code_read/lines.rs  a file's line count, as file_read reports it (§6.3)
  retrieval_scope.rs  a conversation's scope; the lazy rebuild (§8.2)
  retrieval_scope/
    index.rs          committed keys to timelines
    tree_scope.rs     one tree's ingested units
    kept.rs           the bounded caches
    reader.rs         whose conversation a scope is worked out for (memory files, `docs/zen_memory_and_guides.md`)
candle-conversation/src/
  substrate/registry.rs        scoped groups and per-target scopes; archiving drops a scope
  projection/resolver/         both readers honour them (belief_groups.rs, targeted_read.rs)
```

---

## 11. Testing

- **pkt-line and ls-refs** assert raw bytes both ways, including an `ERR` line and a response with no branches.
- **URL derivation** over every remote form `RemoteUrl` accepts.
- **The walk** against real repositories (`TestRepo`): branches sharing a tree listed once; each filter; depth counted from the scope folder; a file moved to a path under `.hidden/` dropped.
- **Keys** asserted as exact strings and exact SHA-256 hex.
- **The pass's plan** (live, queued, dead, deferred) as a pure function of a corpus and a committed set.
- **The record**: a conversation pins origin's copy when it is ahead of the local branch; `follow_record` fast-forwards, creates, and refuses a diverged branch. A branch with unpushed commits — ahead of origin's copy or diverged from it, checked out or not, with the owner's staged, unstaged and untracked work over it, an untracked file where origin's copy tracks one — runs its job on origin's copy and comes back at its own commit with nothing published; a failed job and a crash put it back too; a conversation origin has moved past is still refused as behind.
- **The importer** (`docs/substrate_git.md` §12): the importer follows origin; a record that moved wakes the worker.
- **The scope**: a scoped group returns only the target's timelines, and nothing for a target with no scope; archiving or tombstoning a target drops its scope. A conversation's scope is what its base holds; a changed path leaves it with its folder; conversations that changed nothing share one scope; working a scope out pins no base.
- **The lazy rebuild**: nothing committed, no rebuild; a stale index is rebuilt at most once a second, each rebuild under the next generation; a commit during a rebuild leaves the index stale; a rebuild lets go of what was worked out against the old index.
- **Partial repositories**: a repository that fails part-way through the walk, or whose `.git` will not open, sits the pass out whole.
