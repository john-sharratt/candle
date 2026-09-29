# Zend: Ingesting Branches

**Status:** Built and unit-tested (§11). Not yet run against the live daemon: the end-to-end pass over a real model is `zend/tests/zen_code_phase12_smoke.rs` (ignored, loads the model).
**Scope:** How the `repo_map` and `code_reading` layers are populated from the branches origin holds rather than from a repository's folder; how a change on origin is noticed within seconds without being rate-limited; how an ingested unit is keyed by its content so a branch moving forward re-reads only what changed; and how a conversation retrieves only the ingested content that matches its own base.

It replaces the folder walk of `docs/zend_workspace_execution.md` §5.1 and refines §4.1 (which commit a conversation pins) and `docs/zend_git.md` §7.8–§7.9.

---

## 1. Summary

Every repository in the workspace whose record is on `origin` (`docs/zend_git.md` §7.8) is ingested **from its branches as origin holds them**: every `refs/remotes/origin/*` ref, read in place, with no local branch made for any of them and nothing written to the checkout. A watcher asks origin every couple of seconds whether any branch moved — one small authenticated request per repository, on git's own wire protocol, which no API quota counts — and fetches only when something did. The fetch wakes the ingest worker.

The ingest lists each branch tip's tree (one `ls-tree` per distinct tree, no file contents read), filters it by the same rules the folder walk used, and derives the units the two layers ingest. **A unit is keyed by what it shows, never by the commit it was found on.** A file's key is its repository, its path and its **blob id** — git's own hash of exactly the bytes `file_read` returns. A folder's key is a hash of its listing — the entries on its first page and how many it holds — and the blob ids of the files whose content its turns show. A commit only says where to read a unit's bytes from. So a branch moving forward re-ingests only the paths whose blob changed there; a file untouched since an older commit is never read again; and the same file on twenty branches is one conversation.

Whatever is ingested but no longer found on any branch is tombstoned. A conversation retrieves only the ingested units whose keys its own base holds, so it never reads another branch's version of a file, and a conversation pins **origin's copy** of its branch — the same copy the ingest covers.

---

## 2. Goals and non-goals

### Goals

- Every branch on origin ingested, within `--max-depth` and the `--ingest-dir` scope, whatever is checked out.
- A change on origin noticed within seconds, on GitHub, GitLab, Gitea or any smart-HTTP host, without spending REST or GraphQL quota.
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

A repository's branches are its **record branches**: with an `origin` remote, every `refs/remotes/origin/<b>` except the symbolic `HEAD`; with none, every local branch. `Repo::record_branches()` returns them as `(BranchName, Oid)`, in name order. They are read locally and never touch the network — the watcher (§4) is what keeps them current.

No local branch is created for any of them. The ingest reads trees and blobs by id, which needs no branch at all, and a conversation that switches to one gets a local branch the ordinary way (`git_switch` pulls it, `docs/zend_git.md` §7.8).

### 3.2 What a conversation pins

A conversation's store pins its branch's **record** on first read (`GitSource::base_at`): `refs/remotes/origin/<b>` when origin has the branch, else `refs/heads/<b>`. That is the copy the ingest covered, the copy `git_status` already compares against, and the copy a commit must descend from to publish. A commit the developer made locally and never pushed is not on the record and is not what a new conversation starts from.

A conversation's base can therefore be ahead of its local branch — origin moved, the watcher fetched, the local branch was left where it was. The sandbox, which checks the conversation's branch out, brings the local branch up first (`Repo::follow_record`): when the base is in origin's copy's history and the local branch is in the base's, the local branch is fast-forwarded to the base (created there if it does not exist), under a compare-and-swap. It does so only once it holds the checkout's lock and has set the owner's state aside, so no other job's checkout moves under it, and the owner's checkout comes back as a branch that moved on rather than as uncommitted work reverting what origin gained: the owner's own changes — staged, unstaged, untracked — are laid onto the branch as it now stands, against the commit they were made on (`checkout/preserve/snapshot.rs`). When the owner's checkout is on that branch with uncommitted changes to a file the followed commits change too, nothing moves and the job is refused, as `git merge --ff-only` refuses (`CheckoutError::OwnWorkInTheWay`). Any other disagreement is refused as before (`docs/zend_git.md` §7.7) — including a local branch holding commits origin lacks, which a conversation pinned to origin's copy cannot run a job on until they are pushed.

---

## 4. Watching origin

`zend/src/origin_watch/`. One task per repository with an `origin`, started when the daemon is ready.

### 4.1 The probe

Every **2 s** (±20 % jitter, so repositories do not synchronise), the watcher asks origin for its branch tips and compares them with the repository's tracking refs. Only a difference costs a fetch.

The probe is **git protocol v2 `ls-refs`** over HTTPS: one `POST <url>/git-upload-pack` with `Git-Protocol: version=2` and the body

```
0014command=ls-refs\n
0001
001bref-prefix refs/heads/\n
0000
```

answered by one pkt-line per branch, `<oid> refs/heads/<b>\n`, then a flush. About 1 KB each way, one request, answered from git's refs directly — no API cache in front of it, and no REST or GraphQL quota spent. GitHub's documented guidance for git reads is at most 15 per second per repository; this is one every two seconds. A `GET <url>/info/refs?service=git-upload-pack` is made once when the watcher starts, to confirm the server speaks v2 and offers `ls-refs`; a server that does not is probed with git instead (§4.3).

The probe is the one place zend reaches origin without the git binary. Spawning `git ls-remote` on Windows every two seconds, per repository, with an SSH handshake each time, is both expensive locally and the connection pattern hosts throttle; one kept-alive HTTPS connection per host is neither. The protocol half — pkt-line framing, the request body, parsing the response, deriving the URL — lives in `zend-vfs/src/remote/probe/`, beside the rest of the code that speaks git; zend's watcher only moves the bytes (`reqwest`, HTTP/2 where the host offers it).

**The URL.** An `https://` origin is probed as it is. An SSH origin — `git@host:owner/repo.git`, `ssh://git@host/owner/repo.git` — is probed at `https://host/owner/repo.git`, which GitHub, GitLab and Gitea all serve. A `file://` or local-path origin has no HTTPS form (§4.3).

**Authentication.** Unauthenticated git reads are throttled on GitHub, so a probe to `github.com` carries the daemon's `github_token` (`~/.zend/secrets.yaml`, `docs/zend_workspace_execution.md` §3.5) as HTTP Basic, user `x-access-token`. Another host is probed without credentials, which serves public repositories.

### 4.2 Fetch

A probe that differs from the tracking refs runs `fetch(origin, AllBranches)` over the remote's configured transport (SSH on this machine). It writes only `refs/remotes/origin/*`, prunes branches origin deleted, never moves a local branch, and fetches no tags. A fetch that changed anything wakes the ingest worker.

A commit zend itself publishes needs no probe at all: a push moves the tracking refs as it lands, so a probe afterwards would find origin and the tracking refs level and fetch nothing. A `git_*` tool round that can write a branch or fetch one (`git_commit`, `git_merge`, `git_ref`, `git_switch`, `git_reset`, `git_push`, `git_fetch`) wakes the ingest worker itself.

### 4.3 When the probe cannot run

A repository whose origin has no HTTPS form, or whose HTTPS probe is refused with 401, 403 or 404 (a private repository on a host zend holds no token for), is probed with `git ls-remote origin` instead (`Repo::ls_remote`, its `refs/heads/*` compared) — the transport the fetch uses, so it authenticates the same way. Over SSH that is every **30 s**; over a local path, where nothing is throttled, every 2 s.

A probe that fails for any other reason — a network error, 429, a 5xx — backs off, doubling from 4 s to at most 5 minutes, and resets on the next success. A shutdown stops every watcher before the ingest worker.

---

## 5. The walk

`zend/src/branch_ingest/walk.rs`. For each record branch of each git repository, the tip's tree is listed once (`ls_tree_all`, every entry with its size). A tree two branches share is listed once. What is kept is exactly what the folder walk kept, decided from the listing alone:

- **Files only**: mode `100644` or `100755`. A link is not a file a tool reads, and a submodule (`160000`) is another project.
- **Nothing hidden**: no path component beginning with `.`.
- **Nothing protected**: no `secrets` component — the VFS refuses to serve such a path, so nothing may ingest one.
- **The extension allowlist** (`Language::from_extension`, plus `go.mod` / `go.sum`).
- **The size cap**, `MAX_FILE_BYTES` (16 MiB), from the listing's size.
- **The scope**, `--ingest-dir <layer>=<repo>/<folder>`: only paths under that folder.
- **The depth bound**, `--max-depth N`: path components below the walk's start — the repository's root, or the scope folder — so `1` is its own files.

Binary content cannot be told from a listing. It is sniffed when a file is about to be ingested (§7.3).

The walk's output is a **corpus**: every distinct `code_reading` unit and every distinct `repo_map` unit on any branch, each with one commit it was found on (`at`), the repository's default branch's first (`main`, else `master`), then the other branches in name order.

---

## 6. Content keys

### 6.1 A file

```
<repo>/<path>@<blob id>
```

`candle/zend/src/main.rs@3f73f8722a390439c029daf7748a9ab24791a016`. The blob id is git's hash of the file's committed bytes, which are exactly what `file_read` returns: the store reads the blob and nothing converts it. The path is part of the key because the ingested conversation names it — in its opening request and in every `file_read` header — so the same bytes at another path are another conversation. (Measured on this repository: 22 branch tips hold 8,120 distinct path-and-blob pairs and 8,024 distinct blobs, so the path costs almost nothing.)

The key is known from a tree listing alone. Deciding whether a file is ingested reads no file.

### 6.2 A folder

A folder's turns show its listing (`file_list`'s first page, and how many entries the folder holds) and the manifest hint in its request (`(crate: candle-nn)`) — nothing else. The chain is one `file_list` round-trip and the summary; no file of the folder is read, so no file's content is part of what it shows (`repo_scan/render.rs` records why a README/module-doc read was removed). The listing is what `file_list` shows, not what the layer reads: every file and subfolder but hidden ones, a `LICENSE` or a `kernels/` folder as much as a `.rs` file. Its key is the SHA-256, in hex, of:

```
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

1. **Live** — the corpus's keys (§5).
2. **Committed** — every live conversation of the layer carrying a `content_key`; for `code_reading`, only one whose chain **finished** (`candle_conversation::chain_health`). A file's key is written once its conversation stops calling tools, which is not the same as answering: a read cut off mid-deliberation, with no tool call and no summary, carries the key all the same. Such a chain is not committed, so its key is queued again; its conversation is deferred like a dead key's (§7.2) and retired once the rebuild commits. The resume snapshot and the `file_read` fast path read the same predicate.
3. **Queued** — live keys not committed, in corpus order.
4. **Dead** — committed keys not live.

### 7.2 Tombstones

A dead key is tombstoned, except while its path (or folder) has a queued key in the same layer: then it stays until that replacement commits, so a path is never missing from the layer while it is being re-read. A path deleted everywhere has no replacement and goes at once. Uploads (§9) are never swept here.

"In scope" is literal: only the layers this boot ingests (not `--disable-layer`), and only within the scope and depth the pass walked. A key outside them is not found, and so is dead.

A repository whose branches could not be read this pass — a folder with a `.git` that would not open, or a tree that would not list — **sits the pass out whole**: none of its committed units is dead, since nobody looked for them, and none of its units is queued. Whether a folder is under git is read from the folder (a `.git` in it), never from a failure to open it.

### 7.3 Ingesting a unit

A file's hidden conversation and a folder's chain run exactly as before (`code_read`, `repo_scan`) with one change: the tools they call read **the commit the unit was found on**. The unit's conversation is given a file set whose store for its repository is pinned to `at` (`RepoFiles::fresh_at`), so its `file_read` and `file_list` return that commit's bytes — the bytes the key names. A folder's manifest hint is read from its blob.

Before a file's conversation is minted, its blob is sniffed; a binary file is skipped and its key remembered for the life of the process, so it is not read again — and from the next pass on it is not a unit at all, so an older version of its path does not wait on it as a replacement.

On success the conversation's `content_key` is written, which commits it and marks the retrieval index stale (§8.2); then each other committed conversation of the same path (or folder) whose key is dead is tombstoned. Conversations of the same path with a live key — the file as another branch holds it — are left alone, and so are those with no key yet: another worker's attempt at the file as another branch holds it. A key that cannot be written leaves the attempt uncommitted, and it retires its own conversation.

---

## 8. Retrieval

### 8.1 The fast path

A `file_read` of a file the conversation has not changed is looked up by the file's key from the conversation's base: the blob id its base tree holds for that path (`VfsStore::content_id`), no read and no hash. A hit is carried into the projection as before (`zend/src/fast_path.rs`), and its answer reports the line count the ingest recorded (`lines`, §6.3) — the file is not read to count it. A conversation that recorded no count is not served from: an answer that cannot say how much the model already has is not given.

A `file_list` of a folder's first page is served the same way from `repo_map`: the folder's unit is found in the conversation's retrieval scope (§8.2) by its workspace-relative folder (`RetrievalScope::folder_unit`), so its key is derived exactly as the walk derives it. A folder is not served when the conversation has changed anything at or under it — stricter than the scope, because a new file in a new subfolder changes the listing of every folder above it — nor for a later page, since a unit holds only the first. Its answer is deliberately minimal, `{"status":"already_listed","note":"Listing already in context."}`: the listing it stands in for is small, and the answer is paid on every projection of the call turn. Both tools share one fast-path set per conversation, under `code_reading`'s `fast_path_window`, and a finished chain (`chain_health`) gates both.

### 8.2 The gather

`repo_map` and `code_reading` are in the provenance gather, **scoped to each conversation's base**. The engine marks their groups as scoped (`ConversationEngine::set_group_scoped`), and a scoped group's candidates for a target are exactly the timelines the target's scope names (`set_retrieval_scope`) — none when it names none. Both the belief scan (`score_belief_groups`) and projection assembly (`TargetedRead::group_turns`) apply it, so an out-of-scope conversation is neither scored nor selected. An ingest conversation's own self-local projection is unaffected.

Before each dialogue turn zend computes the target's scope (`zend/src/retrieval_scope.rs`): for each repository, the keys of every file and folder in the conversation's base tree, derived by the same filters and unit rules as the walk, looked up in an index of committed keys to timelines. A path the conversation has changed contributes no file key, and the folder holding it no folder key. Every upload is in scope.

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
  read/record.rs      Repo::record_branches, Repo::record_tip
  origin/follow.rs    Repo::follow_record
  types/oid.rs        ObjectFormat::blob_id
  vfs/git_source.rs   base_at pins the record
  vfs/mod.rs          VfsStore::content_id, peek_base, tree_at
  files.rs            RepoFiles::fresh_at
zend/src/
  origin_watch/
    mod.rs            one task per repository; the check and fetch
    probe.rs          the HTTPS transport; git ls-remote when it cannot run
    cadence.rs        interval, jitter, backoff
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
candle-conversation/src/
  substrate.rs        scoped groups and per-target scopes; archiving drops a scope
  projection/resolver.rs   both readers honour them
```

---

## 11. Testing

- **pkt-line and ls-refs** assert raw bytes both ways, including an `ERR` line and a response with no branches.
- **URL derivation** over every remote form `RemoteUrl` accepts.
- **The walk** against real repositories (`TestRepo`): branches sharing a tree listed once; each filter; depth counted from the scope folder; a file moved to a path under `.hidden/` dropped.
- **Keys** asserted as exact strings and exact SHA-256 hex.
- **The pass's plan** (live, queued, dead, deferred) as a pure function of a corpus and a committed set.
- **The record**: a conversation pins origin's copy when it is ahead of the local branch; `follow_record` fast-forwards, creates, and refuses a diverged branch.
- **The probe loop** against a local bare origin: a push is seen, fetched and wakes the worker; an unchanged origin costs no fetch.
- **The scope**: a scoped group returns only the target's timelines, and nothing for a target with no scope; archiving or tombstoning a target drops its scope. A conversation's scope is what its base holds; a changed path leaves it with its folder; conversations that changed nothing share one scope; working a scope out pins no base.
- **The lazy rebuild**: nothing committed, no rebuild; a stale index is rebuilt at most once a second, each rebuild under the next generation; a commit during a rebuild leaves the index stale; a rebuild lets go of what was worked out against the old index.
- **Partial repositories**: a repository that fails part-way through the walk, or whose `.git` will not open, sits the pass out whole.
