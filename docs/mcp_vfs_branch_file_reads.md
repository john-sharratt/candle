# MCP branch-aware file reads

**Status: design.** Nothing here is built. Every statement about existing code is cited to a
file and line; everything this design adds is marked *planned*.

## 1. The gap

Claude reaches zend through `/v1/mcp` (`zend/src/mcp/mod.rs:1`), and the surface it is offered
has no file tools at all: `tools::definitions()` lists forty-two tools — context, threads,
journal, groups, jobs, cluster, services, layers, timelines — and none of them is `file_*`
(`zend/src/mcp/tools/mod.rs:155`, and the list its own test pins at `zend/src/mcp/tools/mod.rs:317`).
The file tools exist and are the daemon's own: `file_read`, `file_edit`, `file_write`,
`file_list`, `file_search`, `file_grep`, `file_delete`, `file_move`, `file_image`,
`file_present` (`zend-tools/src/tools/file/mod.rs:179-188`), each reading and writing through the
conversation's overlay store (`zend-tools/src/tools/file/mod.rs:12`). So a higher-level
conversation — Claude, Grok, anything on the other side of the gateway — cannot read a file. It
can only delegate the read to a worker and wait for the worker's journal entry, which is the
wrong shape for "what is on line 40 of this file".

The pieces the feature needs are all in place, and none of them is MCP-specific:

- **The overlay.** `VfsStore` is a session-private in-memory upper layer over a read-only lower
  layer, which is a branch, a folder, or nothing (`zend-vfs/src/vfs/mod.rs:427`, `:444`). Reads
  resolve session-first and fall through; writes never touch the disk
  (`zend-tools/src/tools/file/mod.rs:12`).
- **One store per repository, per conversation.** `RepoFiles` holds one `VfsStore` per
  repository in manifest order (`zend-vfs/src/files.rs:62`), routes a call's `repo` argument to
  it and refuses a name the workspace does not list (`zend-vfs/src/files.rs:180`), and `fresh`
  makes a set with none of the session's changes (`zend-vfs/src/files.rs:124`).
- **Branch pinning.** A repository under git is read through a branch, never through its folder
  (`zend-vfs/src/files.rs:14-19`), and the branch a conversation starts on is `main`, else
  `master`, else what is checked out (`zend/src/conv_branches.rs:27`, `:84`).
- **The conversation's overlay, already assembled.** `InferenceState::saved_files` returns
  exactly what a conversation's reads see: a fresh set, its branches taken up, its saved events
  replayed into it, and nothing saved back (`zend/src/session.rs:3574-3580`).
- **Invalidation on commit.** `Committing::land` carries the store onto the commit it published
  (`zend-vfs/src/work/commit.rs:173`), and the mirror tombstones the paths the store let go of
  (`docs/zend_vfs_events.md` §4, `zend/src/conv_overlay/mod.rs:51`).

What is missing is the seam: nothing on the MCP surface turns a caller's identity into a
conversation's file stores and hands them to the existing tools.

## 2. What is decided

**The tools keep their names.** `file_read`, `file_grep`, `file_list`, `file_search` — the same
names, the same descriptions, the same rendering the daemon's own conversations get. The
`repo_` prefix marks work Claude has no other way to reach (`zend/src/mcp/tools/mod.rs:12`); a
file read is not that, and the corpus the model was conditioned on — the `code_reading` ingest,
the tool-case corpus — knows these tools by these names. Two MCP-only arguments are added, and
they are the whole design: which conversation's files, and what the answer says about the
branch.

**The reads are the existing tools, not new ones.** Each MCP tool resolves the conversation's
`Arc<RepoFiles>`, builds a `ToolContext` around it, and calls the registered tool. The rendering
is `anchored_page` (`zend-tools/src/tools/file/render.rs:68`), so an MCP read is byte-identical
to the `code_reading` ingest's prefilled response and to what the web chat gets; the page is
required and bounded at `PAGE_LINES` = 200 (`zend-vfs/src/vfs/mod.rs:285`), which is the rule
that keeps a read from turning the next turn into a 53,288-token prefill
(`zend-tools/src/tools/file/read.rs:17`).

**Nothing is written.** The four tools are read-only; the write tools stay off the MCP surface.
The file tools need no grant in any mode because the overlay absorbs the write
(`docs/tool-system.md:36`), but an MCP caller has no turn to seal, no journal entry to hold the
change, and no one to answer for it. A read is what a higher-level conversation needs; a write
is a task, and a task is `repo_task_start` (`zend/src/mcp/tools/task.rs:14`), which runs on a
worker with a journal and a commit path.

## 3. Which conversation's files

The argument is `conversation`, and it names one of two things:

1. **A worker's path** — `T:<thread>/<worker>`, or a part below it, as `mcp::names::Path`
   parses them (`zend/src/mcp/names.rs:49`). The worker's conversation id is
   `mcp:<owner>:<thread>/<name>` (`zend/src/mcp/threads.rs:338`), and the caller may name one
   only if they own it or are an Admin, which is `owner_of` / `may_open`
   (`zend/src/mcp/ownership.rs:31`, `:37`) — the same rule every other route that names a
   conversation applies.

2. **Nothing** — the caller's own reader conversation: the caller's key
   (`zend/src/mcp/caller.rs:40`) plus the thread the session is attached to
   (`zend/src/mcp/state.rs:39`, read by `Ctx::attached`, `zend/src/mcp/tools/mod.rs:102`). A
   conversation id of `mcp:<key>:<thread>/files` is built from those, and it is a conversation
   like any other: its own `RepoFiles`, its own overlay, its own saved events.

The second case is the one that makes the feature useful on its own: Claude reads a file, edits
nothing, and its own reads and the worker's writes both live in the substrate, so a second
`file_read` after a context compaction sees the same thing it saw before — the conversation's
overlay is saved as events and restored when the conversation is built again
(`zend/src/conv_overlay/mod.rs:51`, `:64`).

**The plumbing.** `InferenceState::saved_files` is `pub(crate)`
(`zend/src/session.rs:3574`). *Planned:* a `ZendSession::conversation_files(conv_id) ->
Option<Arc<RepoFiles>>` returning `None` when the model is not loaded — the shape
`ZendSession::conversation_branches` already uses (`zend/src/session.rs:7874`) — and the tool
answers `LOADING` (`zend/src/mcp/tools/mod.rs:84`) when it is `None`.

It reads the conversation's **saved** overlay, not its live one. A live conversation's stores
are saved after every tool round (`zend/src/session.rs:3595`), so the difference is the round
in flight; reading a live store from an MCP call would reach into a conversation mid-round and
see a half-applied edit. The answer says so when the conversation is live:
`"a round is running; this is what it has saved"`. That is a fact about the store, not a caveat
in prose.

## 4. Branch resolution

A store reads a branch at one commit, its base, taken from the branch the first time the store
reads it and kept (`zend-vfs/src/vfs/mod.rs:546`, `:618`), and the branch moves only when the
conversation moves it (`zend-vfs/src/vfs/mod.rs:504`). So:

- **A named worker** reads its own branch, from its own record
  (`zend/src/conv_overlay/mod.rs:71` puts each store on the branch the conversation recorded).
  Nothing new: `saved_files` does it.
- **The caller's own reader conversation** has no recorded branch, and seeding one needs a
  registered timeline (`conv_branches::seed` returns 0 for an unknown timeline,
  `zend/src/conv_branches.rs:84`). *Planned:* it does not get one. It reads
  `InferenceState.base_branches` — the workspace's base per repository, `main` else `master`
  else what is checked out, computed once at load (`zend/src/session.rs:3078`,
  `zend/src/conv_branches.rs:32`) — and passes it to `RepoFiles::set_branches`
  (`zend-vfs/src/files.rs:154`) on the fresh set. A reader conversation is a pair of eyes on
  the base branch, not a line of work; the moment it needs its own branch it is a task.

**The branch is named in the answer.** Every read ends with the repository, the branch, and the
commit the read came from — `candle @ main (a3f1c92)`. The commit comes from `peek_base`, which
answers where the store's base is, or where the branch stands for a store that has not taken
one, and leaves the store as it was (`zend-vfs/src/vfs/mod.rs:558`). It is read *after* the
read, so the commit named is the one the bytes came from, not a guess about where the branch
stood before.

## 5. Invalidation on commit

Three cases, and the design is that none of them needs a hook.

1. **The conversation commits its own work.** `Committing::land` carries the store onto the new
   commit (`zend-vfs/src/work/commit.rs:173`), so what it committed reads from the branch and
   the deltas that made it are gone; the mirror tombstones the dropped paths and writes the new
   base (`docs/zend_vfs_events.md` §4, `zend/src/conv_overlay/mod.rs:51`). The next read sees
   the commit. This is the existing path, unchanged.
2. **Someone else commits to the branch.** A reader conversation holds no `RepoFiles` between
   calls — each call builds one (`zend/src/session.rs:3574`), and a store takes its base only
   when it reads (`zend-vfs/src/vfs/mod.rs:618`) — so the next call reads the branch as it
   stands now. There is nothing to invalidate because there is nothing cached.
3. **A conversation with a base sees the branch move under it and does not follow.** That is
   the pin's whole purpose (`zend-vfs/src/vfs/mod.rs:546`, `zend-vfs/src/files.rs:14-19`), and
   the MCP read does not break it: the answer names the base it read and, when
   `peek_base` (`zend-vfs/src/vfs/mod.rs:558`) shows the branch further along, says so —
   `"candle @ main (a3f1c92); main is at 9be44d0 — your conversation is behind; git_merge
   brings it up"`. The reader is told, not moved.

**The one thing that must not happen:** one long-lived `RepoFiles` shared by the MCP surface.
It would pin a base on its first read and never move it (`zend-vfs/src/vfs/mod.rs:504` keeps the
base a store already took), which is exactly the staleness this feature is measured against. The
per-call build is the invalidation.

## 6. The tools

Four definitions, one file each under `zend/src/mcp/tools/`, each calling the
`zend-tools` implementation through a `ToolContext` built around the conversation's stores
(`ToolContext::with_files`, `zend-tools/src/context.rs:199`; `ToolHost::context_for`,
`zend/src/tools.rs:979`). Mode: Restricted, which is what a user's tasks run and what offers the
file tools and no network (`docs/tool-system.md:36`), and the context is built branches-only
(`ToolHost::bound`, `zend/src/tools.rs:958`) so a conversation owned through `/mcp` can never
push to a default branch (`zend/src/tools.rs:938`, `zend-tools/src/context.rs:131`). The
`branches_only` guard is a write guard, so it changes nothing about a read — it is there so the
same context is honest if a write tool is ever added.

| Tool | Wraps | Arguments |
|---|---|---|
| `file_read` | `FILE_READ` (`zend-tools/src/tools/file/read.rs:117`) | `repo`, `path`, `page`, optional `line`, optional `conversation` |
| `file_grep` | `FILE_GREP` (`zend-tools/src/tools/file/grep.rs:251`) | `repo` (or `*`), `pattern`, `page`, optional `conversation` |
| `file_list` | `FILE_LIST` (`zend-tools/src/tools/file/list.rs:132`) | `repo`, `path`, `page`, optional `conversation` |
| `file_search` | `FILE_SEARCH` (`zend-tools/src/tools/file/search.rs:177`) | `repo` (or `*`), `query`, `page`, optional `conversation` |

`*` covers every repository and the hits come back grouped by repository
(`zend-tools/src/tools/file/mod.rs:86`), which is the behaviour those two tools already have.

Every answer is a text block (`Answer::result`, `zend/src/mcp/tools/mod.rs:149`), one page at a
time, and the page's own fence carries the file, the page, the page count and the length — the
format `file_read` already renders (`zend-tools/src/tools/file/read.rs:54`). The MCP client's
result ceiling is about 150,000 characters on claude.ai and warns at 10K tokens on Claude Code
(`docs/claude_zend_gateway.md:1330`), and a `PAGE_LINES` page of a 200-line file is far inside
both.

**`readOnlyHint: true`** on all four, because they read nothing but the store and write nothing.
The aggregator records an episode for every call that does not say so
(`docs/mcp_user_context.md` §2.3, asserted by the list at
`zend/src/mcp/tools/mod.rs:382`), and a read is not an episode.

## 7. What is refused

| Case | Answer |
|---|---|
| Model not loaded | `LOADING` (`zend/src/mcp/tools/mod.rs:84`) |
| `conversation` names a worker the caller does not own and they are not Admin | `404`-shaped refusal: the conversation is not there to them (`zend/src/mcp/ownership.rs:31`, `:37`) |
| `conversation` is not a path | `mcp::names::Path::parse`'s own message (`zend/src/mcp/names.rs:49`) |
| `repo` names no repository | `unknown_repo`, with the names the workspace lists (`zend-vfs/src/files.rs:41`, `:180`) |
| Path under `secrets/` or `.git/` | `forbidden` (`zend-vfs/src/vfs/mod.rs:157`, `:174`, `:177`) |
| File over 4 MiB or not UTF-8 | `unreadable` (`zend-vfs/src/vfs/mod.rs:149`) |
| A path that is a URL | `invalid_arguments` (`zend-tools/src/tools/file/read.rs:91`, `zend-tools/src/tools/file/mod.rs:209`) |

The protected-path refusals are the ones that matter most here: `.git/` is closed because
hand-parsing it gives confidently wrong answers and because `.git/config` holds remote URLs with
their credentials in them, which the git layer redacts and this would not
(`zend-vfs/src/vfs/mod.rs:157-176`). The guard is enforced where the lower layer is built, not
at the call site (`zend-vfs/src/vfs/mod.rs:103`), so an MCP tool inherits it for free.

## 8. Tests

Tier 1, CPU, no model — the shape of the existing MCP tool tests:

- **A read of the caller's own reader conversation** returns the base branch's content and names
  the branch and commit in the answer; a second read after the branch moves returns the new
  content, because the store is built per call (§5, case 2).
- **A read of a worker's conversation** returns the worker's overlay: a delta the worker wrote
  is visible, and the same path in the caller's own reader conversation is the base's copy
  (`zend-vfs/src/files.rs:124` — a fresh set shares no session changes).
- **Ownership**: a worker whose id names another owner is refused as absent; an Admin is served
  (`zend/src/mcp/ownership.rs:37`).
- **An uncommitted delta is not a commit**: after `git_commit` the delta is gone and the read
  answers from the branch; before it, the read answers from the overlay.
- **A read past the base reports the branch's position and does not follow it** — the
  `peek_base` note of §4, asserted on the answer text.
- **The four tools appear in `definitions()`** and in the `readOnlyHint` list, which is the
  existing list at `zend/src/mcp/tools/mod.rs:317` and `:382` extended with four names.

Tier 2, `#[ignore]`d, against a real checkpoint: a worker edits a file, commits, and the MCP
read of the same conversation returns the committed content and no longer reports the delta.

## 9. Open questions

1. **Does the reader conversation's overlay ever need to be written?** As designed it is
   read-only and its overlay is always empty, so it saves nothing (`conv_overlay::save` writes
   nothing when nothing changed, `zend/src/conv_overlay/mod.rs:51`) and its only substrate
   footprint is the conversation record itself. If it later gains writes, the write tools are
   the ones that need a turn, a journal entry and a seal — which is a task, not a call.
2. **A reader conversation's timeline.** `timeline_for` hashes the conversation id
   (`zend/src/session.rs:8991`), so `mcp:<key>:<thread>/files` is a stable timeline without
   registering anything. Whether it should appear in the sidebar and the thread's list is a
   product question for the daemon's own listing, not a blocker: the tools work either way.
3. **`file_grep` over `*` on a large workspace** is the one call whose cost is not bounded by a
   page. It is bounded by the store's own hit ceiling and `files_searched` count
   (`zend-vfs/src/vfs/mod.rs:229`), which is the same bound the web chat already lives with.
