# MCP branch-aware file reads

**Status: design, decisions settled.** Nothing here is built. Every statement about existing code is
cited to a file and line; everything this design adds is marked *planned*. The settled decisions are
in §2. Of the three questions §9 used to leave open, two are closed here — where writes land, and
the `file_grep` ceiling — and the third, whether the reader conversation's timeline shows in the
daemon's listings, stays open and is not a blocker.

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
  it and refuses a name the workspace does not list (`zend-vfs/src/files.rs:186`), and `fresh`
  makes a set with none of the session's changes (`zend-vfs/src/files.rs:124`).
- **A set pinned at a commit.** `RepoFiles::fresh_at` makes a fresh set whose store for one git
  repository reads a given commit rather than a branch, and answers `None` when the workspace has
  no git repository of that name (`zend-vfs/src/files.rs:135`). The ingest already uses it for
  exactly this (`zend/src/code_read/mod.rs:710`, `zend/src/repo_scan/mod.rs:982`).
- **Branch pinning.** A repository under git is read through a branch, never through its folder
  (`zend-vfs/src/files.rs:14-19`), and the branch a conversation starts on is `main`, else
  `master`, else what is checked out (`zend/src/conv_branches.rs:27`, `:84`).
- **The conversation's overlay, already assembled.** `InferenceState::saved_files` returns
  exactly what a conversation's reads see: a fresh set, its branches taken up, its saved events
  replayed into it, and nothing saved back (`zend/src/session.rs:3575`).
- **Invalidation on commit.** `Committing::land` carries the store onto the commit it published
  (`zend-vfs/src/work/commit.rs:173`), and the mirror tombstones the paths the store let go of
  (`docs/zend_vfs_events.md` §4, `zend/src/conv_overlay/mod.rs:51`).

What is missing is the seam: nothing on the MCP surface turns a caller's branch argument into a
pinned set of file stores and hands them to the existing tools.

## 2. What is decided

**The tools keep their names.** `file_read`, `file_grep`, `file_list`, `file_search` — the same
names, the same descriptions, the same rendering the daemon's own conversations get. The
`repo_` prefix marks work Claude has no other way to reach (`zend/src/mcp/tools/mod.rs:12`); a
file read is not that, and the corpus the model was conditioned on — the `code_reading` ingest,
the tool-case corpus — knows these tools by these names. Two MCP-only arguments come with them:
`branch`, which is **mandatory**, and `conversation`, which is optional and attributes the read.
The branch is what the design is about, and §4 is its whole rule.

**The reads are the existing tools, not new ones.** Each MCP tool resolves the branch it was given
to a throwaway `RepoFiles` view (§4), builds a `ToolContext` around it, and calls the registered
tool. The rendering is `anchored_page` (`zend-tools/src/tools/file/render.rs:68`), so an MCP read
is byte-identical to the `code_reading` ingest's prefilled response and to what the web chat gets;
the page is required and bounded at `PAGE_LINES` = 200 (`zend-vfs/src/vfs/mod.rs:285`), which is
the rule that keeps a read from turning the next turn into a 53,288-token prefill
(`zend-tools/src/tools/file/read.rs:17`).

**The answer never says `in_context`.** That status means "the corpus already holds this page, so
the call is answered without the bytes" (`zend/src/fast_path.rs:58`, `docs/zend_working_set.md`
§4), and it is a promise about a conversation's own working set — the page is locked and projected
on every decode. An MCP read is a throwaway view of a branch that is thrown away when the call
returns: there is nothing to have already held, and nothing kept to read later. The four tools
answer with the page or with a refusal, never with the status word, and the fast path is not on
their route at all.

**The surface is read-only.** The four read tools are the whole of it: no write tool, no purge,
no branch creation. The write tools stay off the MCP surface — the file tools need no grant in any
mode because the overlay absorbs the write (`docs/tool-system.md:36`), but an MCP caller has no
turn to seal, no journal entry to hold the change, and no one to answer for it. A read is what a
higher-level conversation needs; a write is a task, and a task is `repo_task_start`
(`zend/src/mcp/tools/task.rs:14`), which runs on a worker with a journal and a commit path. This
closes the question §9 used to leave open about where a write would land: it does not land on this
surface at all.

## 3. Which conversation's files

The argument is `conversation`, and it names one of two things — only the second of which is
served:

1. **A worker's path** — `T:<thread>/<worker>`, or a part below it, as `mcp::names::Path` parses
   them (`zend/src/mcp/names.rs:49`) — and it is **refused**. Ownership rules exist for the
   routes that do name a worker (`owner_of` / `may_open`, `zend/src/mcp/ownership.rs:31`, `:37`),
   and they are moot here: a worker's store is its working set mid-round, and the answer to a
   question about that is the worker's journal, not a page of a store nobody sealed.

2. **Nothing** — the caller's own reader conversation: the caller's key
   (`zend/src/mcp/caller.rs:40`) plus the thread the session is attached to
   (`zend/src/mcp/state.rs:39`, read by `Ctx::attached`, `zend/src/mcp/tools/mod.rs:102`). A
   conversation id of `mcp:<key>:<thread>/files` is built from those. It is the only value the
   four tools accept: the read is attributed to it, and it is what `LOADING` is asked about — but
   the bytes come from the branch the call named, not from this conversation's stores (§4).

The reader conversation is the whole of the surface: Claude reads a file, edits nothing, and the
conversation record itself is what the substrate holds — its overlay is always empty, because MCP
writes nothing (§2), so there is nothing of the reader's to save or restore. A worker's writes
reach it the way they reach any other reader: once the worker commits, the branch carries them.
Where a conversation's overlay is saved as events and restored when the conversation is built
again, that machinery is untouched here (`zend/src/conv_overlay/mod.rs:51`, `:64`) — it serves
conversations that write, which on this surface none do.

**The plumbing.** *Planned:* the tool takes the workspace's own file stores — the same
`RepoFiles` a new conversation starts from, `ToolHost::conversation_files`
(`zend/src/tools.rs:969`) — and calls `fresh_at` on the one repository the call named (§4), so the
set it reads is built for the call and dropped with it. It answers `LOADING`
(`zend/src/mcp/tools/mod.rs:84`) when the model is not loaded, the shape
`ZendSession::conversation_branches` already uses (`zend/src/session.rs:7875`).
`InferenceState::saved_files` (`zend/src/session.rs:3575`) is not on this path: it is the
conversation's own overlay, with its own branch recorded and its own saved events replayed into
it, and the MCP surface is not that conversation's eyes.

So the MCP read has no live-versus-saved question to answer: it reads neither. It reads a branch,
at a commit it resolves itself, and the conversation's own overlay is not on the path — which is
why a caller can never see a half-applied edit of a round in flight
(`zend/src/session.rs:3595` saves a live conversation's stores after every round; nothing here
reads them).

## 4. Branch resolution

A store reads a branch at one commit, its base, taken from the branch the first time the store
reads it and kept (`zend-vfs/src/vfs/mod.rs:546`, `:618`), and the branch moves only when the
conversation moves it (`zend-vfs/src/vfs/mod.rs:504`). The MCP read does not use that mechanism.
It pins the branch the caller named, for the length of one call, with the primitive the ingest
already uses for exactly that — `RepoFiles::fresh_at` (`zend-vfs/src/files.rs:135`), which makes a
`fresh` set (`zend-vfs/src/files.rs:124`) whose store for one git repository reads a given commit
rather than a branch, and returns `None` when the workspace has no git repository of that name.

- **A named worker is refused.** A worker's store is on the branch its own record names
  (`zend/src/conv_overlay/mod.rs:71`), and reading it through MCP would show a caller a working
  set that is not a branch — the answer would name a branch while the bytes came from a worker's
  overlay. The worker's own journal is the place its work is read (`journal_read`,
  `zend/src/mcp/tools/journal_read.rs`), and the refusal says so.
- **The caller's own reader conversation gets no branch of its own.** It has no recorded branch,
  and seeding one needs a registered timeline (`conv_branches::seed` returns 0 for an unknown
  timeline, `zend/src/conv_branches.rs:84`), which this design does not register. It reads the
  branch the call names, and nothing else: a reader conversation is a pair of eyes on a branch
  named in the call, not a line of work; the moment it needs its own branch it is a task.

*Planned:* the tool resolves the branch name to an `Oid` with `BlobReader::commit_of` — the
`rev^{commit}` read that answers the commit a revision names and its root tree, or `None` when the
revision names nothing (`zend-vfs/src/read/blob_reader.rs:323`, reached through `Repo::blobs`,
`zend-vfs/src/read/blob_reader.rs:343`) — and then calls `fresh_at` with it. A branch that does not
exist is refused, not guessed at; a name that is not a branch is refused by `BranchName::parse`,
which is the same parser `RepoFiles::set_branches` uses (`zend-vfs/src/files.rs:154`).

**`repo: "*"` is refused.** A pinned set is per repository (`fresh_at` takes one name and one
commit, `zend-vfs/src/files.rs:135`), and a wildcard read across every repository would need one
commit per repository named by the caller — a call that answers nothing a caller can ask for. The
workspace's repositories are listed by the refusal, as `RepoFiles::repo` already lists them
(`zend-vfs/src/files.rs:186`, `UnknownRepo` at `:41`), and the caller makes one call per
repository.

**The branch is mandatory, and there is no default.** Every call names one, on all four tools.
The conversation's own branch is not the default, because there is no conversation branch to
default to: the reader conversation is not a line of work (§3), and defaulting to
`InferenceState.base_branches` (`zend/src/session.rs:3079`, `zend/src/conv_branches.rs:32`) would
make the same call mean different things on different members and at different times, which is the
one property a read that cites its source has to keep. A caller who wants the default branch says
`main`.

**The branch is named in the answer.** Every read ends with the repository, the branch, and the
commit the read came from — `candle @ main (a3f1c92)`. The commit is the one `fresh_at` pinned the
store to, and `peek_base` (`zend-vfs/src/vfs/mod.rs:558`) is read after the read to confirm it:
for a store pinned at a commit it answers that commit, and it leaves the store as it was. The
throwaway view is not in the answer as a status — the citation is the whole of what the answer
says about where it came from, and §2's rule against `in_context` is what keeps it honest.

## 5. Invalidation on commit

Three cases, and the design is that none of them needs a hook — because the MCP read holds nothing
between calls.

1. **A worker commits its own work.** `Committing::land` carries the store onto the new commit
   (`zend-vfs/src/work/commit.rs:173`), so what it committed reads from the branch and the deltas
   that made it are gone; the mirror tombstones the dropped paths and writes the new base
   (`docs/zend_vfs_events.md` §4, `zend/src/conv_overlay/mod.rs:51`). That is the worker's path,
   unchanged, and it is the reason the MCP read of the branch after a commit sees the commit: the
   branch is what the MCP read resolves, and the commit moved the branch.
2. **Someone else commits to the branch.** The MCP surface holds no `RepoFiles` between calls —
   each call resolves the branch name and builds a pinned set (`zend-vfs/src/files.rs:135`) — so
   the next call reads the branch as it stands now. There is nothing to invalidate because there
   is nothing cached.
3. **A conversation with a base sees the branch move under it and does not follow.** That is the
   pin's whole purpose for a conversation (`zend-vfs/src/vfs/mod.rs:546`, `zend-vfs/src/files.rs:14-19`),
   and it is not this surface's problem: an MCP read pins a commit for one call and drops it, so
   there is no view left standing behind the branch to be told about it. A caller who wants to
   follow a branch calls again; the answer of the next call names the commit it is at now.

**The one thing that must not happen:** one long-lived `RepoFiles` shared by the MCP surface.
It would pin a base on its first read and never move it (`zend-vfs/src/vfs/mod.rs:504` keeps the
base a store already took), which is exactly the staleness this feature is measured against. The
per-call pinned set is the invalidation: `fresh_at` is built for the call and dropped with it.

## 6. The tools

Four definitions, one file each under `zend/src/mcp/tools/`, each calling the
`zend-tools` implementation through a `ToolContext` built around the pinned set the call's branch
resolves to (`ToolContext::with_files`, `zend-tools/src/context.rs:199`; `ToolHost::context_for`,
`zend/src/tools.rs:979`). Mode: Restricted, which is what a user's tasks run and what offers the
file tools and no network (`docs/tool-system.md:36`), and the context is built branches-only
(`ToolHost::bound`, `zend/src/tools.rs:958`) so a conversation owned through `/mcp` can never
push to a default branch (`zend/src/tools.rs:938`, `zend-tools/src/context.rs:122`). The
`branches_only` guard is a write guard, so it changes nothing about a read — it is there so the
same context is honest if a write tool is ever added, and the decision in §2 is that it is not.

| Tool | Wraps | Arguments |
|---|---|---|
| `file_read` | `FILE_READ` (`zend-tools/src/tools/file/read.rs:117`) | `repo`, `branch`, `path`, `page`, optional `line`, optional `conversation` |
| `file_grep` | `FILE_GREP` (`zend-tools/src/tools/file/grep.rs:251`) | `repo`, `branch`, `pattern`, `page`, optional `conversation` |
| `file_list` | `FILE_LIST` (`zend-tools/src/tools/file/list.rs:132`) | `repo`, `branch`, `path`, `page`, optional `conversation` |
| `file_search` | `FILE_SEARCH` (`zend-tools/src/tools/file/search.rs:177`) | `repo`, `branch`, `query`, `page`, optional `conversation` |

`repo` and `branch` are required on all four, and `branch` is read with the same rule
`string` applies to every required string argument — trimmed, and refused when missing or empty
(`zend/src/mcp/tools/mod.rs:264`). A missing `branch` is therefore an `invalid_arguments` refusal,
not a silent default.

`*` is **not** accepted for `repo` on this surface: `RepoFiles::repo` refuses it even where a
workspace exists (`zend-vfs/src/files.rs:186`, `ALL_REPOS` at `zend-vfs/src/workspace.rs:348`),
and §4 pins one repository per call.

**The `file_grep` ceiling is closed as designed.** A grep that stops at its hit ceiling says so:
`GrepOutcome` carries `truncated` and the `files_searched` denominator that tells a caller whether
"no matches" means "searched a lot and found nothing" or "the prefix matched nothing to search"
(`zend-vfs/src/vfs/mod.rs:234`, `:237`, set at `:1560` and `:1573`), and the tool renders that
over the page it returns. The MCP surface inherits the same answer and raises nothing: the ceiling
is the bound, `page` walks it, and no further paging or per-branch cap is planned.

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
| `branch` missing or empty | `invalid_arguments` — `string`'s own message, "`branch` is missing: it is a non-empty string" (`zend/src/mcp/tools/mod.rs:264`) |
| `branch` names no commit in that repository | `invalid_arguments`, naming the branch and the repository (`commit_of` answers `None`, `zend-vfs/src/read/blob_reader.rs:323`) |
| `repo` is `*` | `invalid_arguments`, naming the repositories to call one at a time (`ALL_REPOS` refused by `RepoFiles::repo`, `zend-vfs/src/files.rs:186`) |
| `conversation` names a worker (`T:<thread>/<worker>`, or a part below it) | `invalid_arguments`: reads of a worker's working set are not on this surface — read its journal (`journal_read`), or start a task (`repo_task_start`) |
| `repo` names no repository | `unknown_repo`, with the names the workspace lists (`zend-vfs/src/files.rs:41`, `:186`) |
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

- **A read names its branch and commit.** Every answer ends `candle @ main (a3f1c92)`; the commit
  is the one `fresh_at` pinned, confirmed by `peek_base` after the read (§4).
- **A call with no `branch` is refused**, on all four tools, before anything is read — there is no
  default to fall back to (§2, §4).
- **A read of a branch that does not exist is refused**, and the refusal names the branch; a read
  of `main` is unaffected.
- **A read of the branch after it moves returns the new content**, because the store is built per
  call (§5, case 2) — the same call twice across a commit returns different bytes and different
  commit citations.
- **No answer ever carries `in_context`** — asserted on all four tools' answers, so a fast-path
  reply cannot reach an MCP read (§2).
- **`repo: "*"` is refused**, and the refusal names the repositories to call one at a time (§4).
- **A `conversation` naming a worker is refused**, whether or not the caller owns it, and the
  answer points at `journal_read` and `repo_task_start` (§3, §7).
- **The four tools appear in `definitions()`** and in the `readOnlyHint` list, which is the
  existing list at `zend/src/mcp/tools/mod.rs:317` and `:382` extended with four names.
- **No write tool is on the surface**: `definitions()` still lists no `file_write`, `file_edit`,
  `file_delete`, `file_move`, or purge, and `call` refuses them as "there is no tool" — the
  read-only decision of §2, asserted so a later addition cannot slip in unmarked.

Tier 2, `#[ignore]`d, against a real checkpoint: a worker edits a file and the MCP read of the same
branch does not see the delta; after the worker commits, the MCP read returns the committed
content and cites the new commit.

## 9. Open questions

1. **Closed: where a write would land.** There is no write on this surface (§2) — no write tool,
   no purge, no branch creation — so the question does not arise. Writes stay with
   `repo_task_start`, which runs on a worker with a turn to seal, a journal entry to hold the
   change, and a commit path (`zend/src/mcp/tools/task.rs:14`). If the surface ever gains a write,
   it gains those three things first, and this document is reopened.
2. **Open: a reader conversation's timeline.** `timeline_for` hashes the conversation id
   (`zend/src/session.rs:8991`), so `mcp:<key>:<thread>/files` is a stable timeline without
   registering anything. Whether it should appear in the sidebar and the thread's list is a
   product question for the daemon's own listing, not a blocker: the tools work either way, and
   nothing in §4 depends on it — the branch comes from the call, not from the timeline.
3. **Closed as designed: `file_grep`'s ceiling.** A grep over a large workspace is bounded by the
   store's own hit ceiling and reports `truncated` and the `files_searched` denominator
   (`zend-vfs/src/vfs/mod.rs:234`, `:237`), which is the same bound the web chat already lives
   with; the answer says when it is a prefix of what is there, and `page` walks it. With `repo: "*"`
   refused (§4), one call scans one repository at one commit, so no further cap is planned.
