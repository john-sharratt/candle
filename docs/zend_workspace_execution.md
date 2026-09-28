# Zend: Workspace, Tooling and Execution Architecture

**Status:** Workspace and tools built (§3–§4). Ingest (§5) reads the repositories' branches as origin holds them, as designed in `docs/zend_branch_ingest.md`; nothing ingests from a repository's folder, which is the sandbox's. The priming chain (§6) is removed. The sandbox that runs tooling on one machine is built and wired into the daemon as `run_command` (`zend-vfs::sandbox`, §7.4); cluster execution is designed, not built (§7–§8).
**Scope:** How zend serves several repositories as one workspace; how the model reads and changes code in them; how the base conversation is seeded with every repository; and how real tooling (cargo, node, python, tests) runs, on one machine and across a cluster.

---

## 1. Summary

Zend serves a **workspace**: a folder holding a `workspace.yaml` that lists the repositories in scope. Each repository is a folder directly under the workspace, named by its folder — a git checkout or any other working tree. The daemon's own state (`substrate/`) sits in the workspace folder beside them, and a repository it owns, `uploads`, holds the files users upload.

Every file tool takes a `repo` argument and a path relative to that repository. The repository names are fixed for the daemon's life, so the `repo` parameter is a JSON-Schema `enum` of them, written into the tool definitions at startup; the constrained decoder (the stencil) compiles that schema, so the model can only ever name a repository that exists.

Internally, everything the ingest layers store is keyed by its **workspace-relative path** (`candle/src/lib.rs`), which is simultaneously a real path on disk and — split at its first segment — the `(repo, path)` pair a tool call takes.

A conversation reads a git repository **through its branch, never its folder**. Each conversation works on its own branch per repository (`ConvState::branches`), at its own base commit on it; its `file_*` tools read that commit, with the conversation's own changes laid over it, and never the repository's folder on disk. The base moves only when the conversation commits, merges, switches or resets (`docs/zend_git.md` §7.9). The folder is whoever's working copy it is; the sandbox borrows it one job at a time and hands it back exactly as it found it (§7.4).

---

## 2. Goals and non-goals

### Goals

- Present several repositories as one workspace, addressed uniformly by `(repo, path)`.
- Keep the repositories where they are: local working trees, no copying, no checkout manager.
- Make an invalid repository name impossible to decode, not merely rejected after the fact.
- Seed the base conversation with every repository in scope.
- Keep zend's implementation simple: one manifest, one key form, one split.

### Non-goals (for now)

- **Git inside this document.** Reading repository state, committing the session overlay to a branch and pushing it to origin are the git layer's, designed in `docs/zend_git.md`. The working trees here are edited only through the conversation's overlay, never on disk.
- **Repositories outside the workspace folder, or nested ones.** A repository is exactly `<workspace>/<name>`.
- **Cluster execution.** Designed in §7–§8, not built.

---

## 3. The workspace

### 3.1 Manifest

`<workspace>/workspace.yaml`:

```yaml
repos:
  - name: candle
  - name: battle-cities
  - name: mind
```

Rules, enforced when the workspace is built (`zend-vfs/src/workspace.rs`):

- At least one repository.
- A name is one plain folder name: no separators, no `.`/`..`, nothing Windows would open as another name (a `:`, a trailing dot or space, an 8.3 `~N` tail), not `secrets`, and not `jobs` in any case — the folder the command sandboxes write their job logs to (§7.4). Names are unique, so no two repositories share a folder and none nests in another.
- Each folder must exist. A manifest key the schema does not know (a misspelt `name`, a `path:` field) is an error, not ignored.
- A missing manifest fails the launch immediately — before the model loads.

### 3.2 The uploads repository

The daemon adds `uploads` after the listed repositories (`zend/src/workspace.rs`) and creates its folder. The upload endpoint writes files there; the model reads them as `repo: uploads`. A manifest that lists a repository named `uploads` is refused. The uploads repository is never walked by the ingest layers and never part of the priming chain — uploads are ingested by the endpoint itself.

### 3.3 The workspace folder

| Path | What |
|---|---|
| `workspace.yaml` | The manifest. |
| `substrate/` | The redo log, logs, conversation files. Visible, not a dot-directory: it sits beside the repositories, not inside one. |
| `<repo>/` | Each listed repository. |
| `uploads/` | The daemon's uploads repository. |
| `jobs/` | The sandbox servers' job logs, `<job id>.log` — outside every repository, so no job ever checks out, captures or resets another's log. Made at startup (§7.4). |
| `projection.yaml`, `tools/`, `identities/`, `<collection>s/`, raw layer folders | A *mind*'s configuration (a workspace carrying its own `projection.yaml`). Read from the workspace folder, never from a repository. |

Anything else in the folder — other checkouts, stray files — is outside every repository and invisible to tools and the walk.

### 3.4 Two path forms, one split

| Form | Example | Used by |
|---|---|---|
| Workspace-relative key | `candle/zend/src/main.rs` | The walk, `repo_map` / `code_reading` metadata and resume hashes, uploads, the fast path. |
| `(repo, path)` | `repo: candle`, `path: zend/src/main.rs` | Every tool call and tool response. |

A repository's folder is its name, so the key's first segment *is* the repository, and `zend/src/repo_path.rs::split` converts one form to the other. There is no second mapping to keep consistent.

### 3.5 Secrets

The API keys and tokens the daemon presents to third-party services — the Tavily key for `web_search`, the GitHub token for pushing branches to a repository's origin and opening pull requests — belong to the person running the daemon, not to a workspace. They live in one per-user file, outside every workspace:

```yaml
# ~/.zend/secrets.yaml
github_token: ghp_...
tavily_api_key: tvly-...
```

`zend --secrets <path>` names another file for an operator who keeps it elsewhere; a named file that does not exist fails the launch. The daemon reads the file once at startup (`zend/src/secrets.rs`) into `zend_tools::state::Secrets`, which every tool context shares, and logs only which keys are set.

The model cannot reach the file: the `file_*` tools mount only the listed repositories, resolve every path with its symlinks and junctions followed — so a link committed to a repository cannot lead out of it — and the code sandbox reaches nothing beyond those mounts. A file that is not private to the daemon's user is refused, as OpenSSH refuses a loose private key: on Unix one owned by another user, readable by others, or in a folder others can write to; on Windows one whose access list grants read to Everyone, Authenticated Users or Users. The web-search refusal the model reads never names the file. Values never pass through the process environment, so no child process inherits them, and `Secrets`' `Debug` output redacts them. Inside a repository, the VFS still refuses any path with a `secrets` segment, for the repositories that keep their own (`web/secrets/`); that is also why a repository may not be named `secrets`.

---

## 4. Tools

### 4.1 One store per repository

`ToolContext.files` is a `RepoFiles` (`zend-vfs/src/files.rs`): one `VfsStore` per repository. A tool resolves `repo` to its store and the path inside it; `..` stops at the repository's root, so a path cannot reach a sibling repository or anything else in the workspace folder. Every store is an overlay — a conversation's changes are held in memory and recorded in the substrate, never written to the folder — and the `secrets/` refusal and the Windows spelling guards hold per store. A context built without a workspace (`ToolContext::new`, tests) is *detached*: each repository name gets an upper-only store on first use.

What a store reads beneath the conversation's changes (`zend-vfs/src/vfs/`):

- **A git repository — at the conversation's base.** The store reads a pinned commit's tree, never the folder and never a moving branch: `RepoFiles::set_branches` gives each store the conversation's branch when its state is built (`conv_overlay::restore`), and until then it reads the branch checked out when the daemon started. The first read pins the base — the commit the branch's record holds then (origin's copy, `refs/remotes/origin/<b>`, when origin has the branch; the local branch otherwise — `docs/zend_branch_ingest.md` §3.2), read over one long-running `cat-file --batch` per repository that every conversation's store shares (`GitSource`); a tree is listed once with `ls-tree -r` and kept, so listings, searches and existence checks run in memory. Only regular and executable files are there — a link or a submodule is not a file a tool reads — and hidden entries are left out of listings and searches and still read by exact path, as before.
- **Any other folder — as it stands on disk.** The uploads repository and a scratch workspace.

**A branch moving does not move a conversation.** Anyone's push, or another conversation's commit, changes nothing this conversation reads. Its base moves only when the conversation moves it — its own `git_commit`, `git_merge`, `git_switch` or `git_reset` — through `VfsStore::move_base`, which carries each uncommitted change onto the new tree (`vfs/carry.rs`, `docs/zend_git.md` §7.9): a change the new tree already holds is dropped, one whose edits still fit the new copy is kept as it is, and one that no longer fits is merged three ways, with overlaps left between markers and the path flagged as in conflict. A flag outlives later moves and clears only when a write or edit leaves the file without markers. The base is kept with the conversation's changes, as events on its timeline in the substrate (`docs/zend_vfs_events.md`) — its tree and parents — so a conversation restored after a restart reads exactly what it read before. `git_status` reports the commits the branch has gained beyond the base, which `git_merge` brings in.

### 4.2 Arguments

Every tool that touches files requires `repo`. The scope of a call is always stated, never implied by an omitted argument.

| Tool | `repo` values | Notes |
|---|---|---|
| `file_read`, `write`, `file_edit`, `file_delete`, `file_present` | one repository | `path` is repository-relative. |
| `file_list` | one repository, or `*` | `*` lists the repositories and takes no `path` (`invalid_arguments` otherwise). |
| `file_search`, `file_grep` | one repository, or `*` | `*` covers every repository; each result names its repository. `prefix` applies inside each repository searched. One hit ceiling across the whole call. |
| `code_run`, `code_session_exec` | one repository | The script's `vfs` global is that repository's store. |
| `remote_fs_session_get`, `remote_fs_session_put` | one repository | The repository the file is saved into / uploaded from. |

`*` (`ALL_REPOS`) can never be a repository's name: the manifest refuses any name holding a character a folder name may not. A tool that works in one repository refuses `*`. An unknown repository is error code `unknown_repo`, and its message lists every valid name.

Required rather than optional, deliberately. An optional scope is where a model drifts into searching everything when it knew which repository it meant — slower over a large repository, and likelier to hit the hit ceiling with matches from the wrong one. A required field is also the stronger grammar shape: its key is prefilled and the only decision is the value.

### 4.3 The `repo` enum and the stencil

The tool YAML (`zend/src/prompts/tools/*.yaml`) does not carry repository names — they belong to the deployment. A tool that can cover the whole workspace declares `enum: ["*"]`; the rest declare no enum. `tool_def::init` receives the workspace and puts its names at the head of every `repo` parameter's `enum` (`tool_def::constrain_repos`), so the search tools choose between the repositories and `*`, and every other tool between the repositories alone.

An enum field's grammar arm carries the first step of what follows it — the next key or the close — so the value's closing quote and the separator after it are one arm's text (`candle_conversation::stencil::tool_call`, `continuations`). A tokenizer writes those together (`",`); a quote committed on its own left a live model unable to write the comma and closing its arguments instead of adding the `path` it meant to. The same schema feeds both readers: it is rendered into the prompt, and `tools::tool_catalog` compiles it into the constrained decoder, where a string `enum` becomes a choice between the listed values. The model is shown the names it may use and cannot decode any other.

A call that bypasses the stencil (OpenAI-style passthrough) reaches the executor unconstrained, and `RepoFiles::repo` refuses an unknown name there.

Because the enum is part of each tool's rendered definition, it is part of its calibration marker: adding or removing a repository recalibrates exactly the tools that take `repo`.

### 4.4 Output

- A `file_read` excerpt is headed `path in repo (page P of N, lines a-b of T):` — the same header the `code_reading` and `repo_map` ingests prefill, byte for byte.
- `file_list` inside a repository carries a top-level `repo`; the workspace listing's entries are `{"repo": name, "dir": true}`.
- `file_search` returns `{repo, path}` objects, shortest path first; `file_grep` matches carry `repo`.
- `write` / `file_edit` / `file_delete` / `file_present` responses carry `repo`.

---

## 5. Ingest

The `repo_map` and `code_reading` layers are ingested from every branch origin holds, never from a repository's folder — the folder is whoever's working copy it is, borrowed by the sandbox. `docs/zend_branch_ingest.md` is the design: a watcher notices a moved branch within seconds and fetches, and the pass ingests each unit of content once, keyed by what it shows rather than by the commit it was found on. Uploads are ingested by their endpoint.

### 5.1 The walk

`branch_ingest::walk` lists each record branch's tree (`docs/zend_branch_ingest.md` §5) and keys every file workspace-relatively, with the file's blob id. Nothing outside the listed git repositories is visited, and the uploads repository is never walked.

- `scope` (`--ingest-dir <layer>=<folder>`) narrows a code layer to one workspace-relative folder inside a repository, e.g. `code_reading=candle/zend/src`. Keys stay workspace-relative.
- `--max-depth N` counts components below each walk's start — a repository's root, or the scope folder — so `1` is a repository's own files.

### 5.2 `repo_map` units

One unit per directory holding walked files (`candle/`, `candle/zend/src/`, …), plus a **workspace unit** (`.`) whenever anything was walked: its evidence is the set of repositories reached, and its listing turn is `file_list` with `repo: "*"`, which lists the repositories. Each unit's prefilled calls split its directory into `repo` and `path` (`file_list {repo: candle, path: zend/src}`; the workspace unit's is `{repo: "*"}`), and the request names the folder with its repository.

### 5.3 `code_reading`

A file's hidden conversation opens with ``Read the entire contents of `zend/src/main.rs` in the `candle` repository …``; the model decodes its own `file_read` calls, constrained to the repository enum, against the commit the file was found on. Its key is the workspace-relative path and the blob id (`docs/zend_branch_ingest.md` §6.1).

### 5.4 Fast path

The fast path, which serves a `file_read` from a `code_reading` conversation that already read the same bytes, joins the call's `repo` and `path` into the workspace-relative key the ingest used, with the blob id the conversation's base holds for that path. A file the conversation has changed is never served: its own copy is what it must read. A `repo` the workspace does not list is never joined onto the workspace; the call runs for real and the tool refuses it.

There is no file watcher: nothing reads a repository's folder for ingest. Origin is watched instead (`docs/zend_branch_ingest.md` §4).

---

## 6. Seeding the base conversation

The base conversation every dialogue forks from is seeded with nothing from the repositories. The priming chain that read the workspace listing, each repository's root listing and its anchor documents (README, ARCHITECTURE, AGENTS, CLAUDE) walked the repositories' folders, and is removed with the rest of the folder ingest; a chain built from the branches belongs to the ingest that replaces it.

---

## 7. Execution: leased build agents (not built)

Most agent activity — reading, searching, editing, reasoning — runs in-process against the working trees and the session overlay. Real execution (builds, tests) runs on leased build agents.

### 7.1 Agent layout

Each build agent holds a checkout of every repository in the manifest, laid out under its own workspace folder exactly as on the GPU node, so relative references between repositories (a cargo path dependency from `battle-cities` to `candle`) resolve identically. Build output and caches persist across leases.

### 7.2 Lease lifecycle

1. **Pick** a sticky agent (§8.3) and **lock** it (lease with expiry).
2. **Sync** each repository to the GPU node's working tree: the same commit, then the node's uncommitted changes and the session overlay's writes applied on top.
3. **Run** one or more commands under a timeout.
4. **Capture** the changes the run made in each repository (`git add -A; git diff --cached --binary`), excluding ignored paths, and return them to the session as overlay writes.
5. **Unlock.**

A lease covers a tool session (edit, build, test, fix) rather than a single call, keeping incremental builds warm.

### 7.3 Safety

- Commands run as a low-privilege user with write access only to the checkouts and caches; credentials are never readable by them.
- Every command runs under a timeout that kills the whole process tree (a Job Object on Windows, a process group on Linux).
- Leases expire slightly after the tool timeout, so a crashed holder's lease is reclaimed.
- Tool output is rewritten from the agent's workspace folder to repository-relative paths before the model sees it, so the same error reads identically on every agent.

### 7.4 One machine: the sandbox

On one machine the repository's own folder is the build agent: `zend-vfs::sandbox`. That folder may be someone's working copy — a developer's clone with a branch checked out and work not yet committed — so it is borrowed, never taken. A `Sandbox` runs one job at a time in it:

1. **Lock.**
2. **Set aside** what the folder holds (`checkout::preserve`, `docs/zend_git.md` §7.7), under a file lock no other process's job can share. The index, and every changed, untracked and flagged file, are snapshotted byte for byte as git objects held by refs of their own. What is ignored is listed, and nothing on the list is ever removed or captured, whatever the branch's ignore rules say. An ignored file or folder at a path the job writes or the branch adds is moved aside. A journal in the git folder records all of it before anything is disturbed.
3. **Put the checkout** on the conversation's branch at its base, and lay the conversation's changes down.
4. **Check the command** — git run directly is refused, pointing at the git tools — and run it.
5. **Read back** what it changed as deltas, and record them in the conversation's store.
6. **Put the folder back**: `HEAD` where it was, the job's untracked files and links cleared, the set-aside files returned, the snapshot's bytes and index written back with their flags, and only then the snapshot let go.
7. **Unlock.**

However the job ends — refused, failed, timed out, panicked or abandoned — the folder is back before the next job may take it: the guard that puts it back drops before the lock does. A put-back that fails is retried; one that still fails keeps the snapshot and the journal and says where they are. A daemon that dies mid-job leaves the journal, and the next job restores from it before touching anything — first keeping what the folder holds by then under `refs/zend/recovered/`, and refusing outright when someone has worked in the folder since. A folder part way through a merge, rebase, cherry-pick, revert or bisect is refused untouched. Ignored files are never set aside or removed beyond the job's own, so build outputs survive from one job to the next and the build cache stays warm.

A conversation whose base is not the commit its branch holds is refused before anything is touched: its changes were made on its base, and a checkout of the branch is not that. Each way it can differ has its own refusal naming the way on — **behind** (the branch gained commits the conversation has not merged: `git_merge`), **ahead** (a merge fast-forwarded the conversation past its branch: `git_commit` with `from: all_changes` publishes it, with no commit of its own when it has no change), and **merging** (a merge is being finished: commit it).

**`SandboxServer`** runs jobs in the background. `start_job` returns at once with the job's id (a random 64-bit number as URL-safe base64), its log (`<jobs dir>/<id>.log`, both output streams as they are printed, capped at 64 MiB), its output as a stream that follows the log from its first byte and ends with the job, and a handle that gives the outcome — and cancels the job, killing its process tree, when dropped first. `query_job` says whether a job is queued, running, exited, timed out, cancelled, refused or failed, and how many lines its log holds. The last 1000 jobs are kept; an older one is let go with its log.

A program named without a path is looked up the way a shell looks it up: on Windows through `PATHEXT` as well as `PATH` (`sandbox/resolve.rs`), so `npm` starts the `npm.cmd` Node installs beside `node.exe`, which the system alone would not find.

**In the daemon.** `ToolHost::new` builds one `SandboxServer` per git repository of the workspace (`zend_tools::sandboxes::Sandboxes`; a folder that is not a git repository — `uploads` — gets none), logging to the workspace folder's `jobs/`, and shares them with every tool context; only a context granted `Exec` reaches them (`ToolContext::sandboxes`). The programs a job may start are the deployment's allow-list, `zend/src/sandbox_programs.rs`: the build, test and packaging toolchains (`node`, `npm`, `npx`, `cargo`, `python`, `pytest`, `go`, `dotnet`, `make`, …) and no shell. The list decides which programs start, not what they do — an interpreter or a package manager on it runs whatever it is handed, with the daemon's rights — so it is not the security boundary; the tools mode is.

Two tools drive it, Comprehensive only (`Network` + `Exec` + `DiskWrite`, high-risk):

- **`run_command`** `{repo, program, args, timeout_secs}` — runs one program, no shell (`args` required, `[]` for none; shell syntax that would reach the program verbatim — an argument that is `&&`, `||`, `|`, `>`, `>>` or `2>&1`, or a value opened with a shell's escaped quote `\"` — is refused with the call it should have been, and a git command line goes straight to the git refusal), on the conversation's branch with its changes laid down, and waits for it (600 s by default, at most 1800). It returns the exit code, whether it timed out, the files it changed — now among the conversation's uncommitted changes, persisted like any edit — anything its changes could not hold, and the first page of its output: 200 lines, terminal colours stripped, a line past 500 characters cut. A refused command, a conversation behind its branch (`behind`: merge first), ahead of it (`unpublished`: commit first), or finishing a merge (`merging`), and a program that cannot start are each their own error code.
- **`run_output`** `{repo, job, page}` — any page of a run's log, and how the run stands; a page past the end is the last.

A tool call drives its job on a thread and runtime of its own (`sandboxes/block.rs`) and waits for it there, so the wait holds none of the daemon's async workers, and a call from any context — a blocking thread or an async task — works the same.

---

## 8. Cluster (not built)

### 8.1 Node roles

| Role | Holds | Needs |
|---|---|---|
| GPU node | Conversations, KV state, the substrate, the working trees | GPU, large host RAM, NVMe |
| Build agent | Checkouts of every repository, build caches | CPU, disk, toolchains |

Conversations are pinned to the GPU node holding their state. Only execution jobs fan out. GPU nodes may register as low-priority build agents with reserved engine cores and capped build memory.

### 8.2 Scheduling

- **Capabilities and demands:** agents advertise OS, toolchains and memory; jobs declare demands.
- **Sticky routing by lineage:** a conversation's jobs go to the agent that last ran them; a fork goes to its parent's agent.
- **Spill threshold:** if the sticky agent is busy, wait briefly, then use any free agent.

### 8.3 Pool management

Agents are long-lived VMs, added or removed on queue depth and wait time, and reset to a golden image on a schedule, since ignored build output drifts.

---

## 9. Open questions

1. **Adding or removing a repository while conversations are live.** The manifest is read once at startup; a change takes a restart, which recalibrates the tools that take `repo` (§4.3). A removed repository's ingested units are on no branch the next pass walks, so they are retired (`docs/zend_branch_ingest.md` §7.2).
2. **A mind workspace with no repositories.** The manifest requires at least one; a pure conversational mind still needs a folder to list.
3. **Cross-repository search ranking.** `file_search` orders by path length across repositories; whether a repository the conversation is already working in should rank first is open.
4. **Build-agent sync granularity** (§7.2): whole-tree sync per lease versus per-command deltas.

---

## 10. Testing priorities

**Workspace and tools** (CPU, `cargo test -p zend-tools`)

- Manifest rules: names, duplicates, missing folders, unknown keys.
- Each repository resolves against its own folder; `..` never reaches a sibling.
- An unknown repository is `unknown_repo` with the valid names listed.
- Every file tool refuses a call without `repo`; `file_list` / `file_search` / `file_grep` with `*` cover every repository and tag results, and a workspace grep represents every matching repository.
- The secrets file: the default under the home folder, the `--secrets` override, a missing named file failing the launch, redacted `Debug`, and an exposed file refused on Unix.
- The excerpt header's exact bytes.

**Stencil** (CPU, `cargo test -p zend`)

- `tool_def::constrain_repos` writes the exact enum, and the compiled `ToolSpec` carries it.
- Every tool that takes `repo` declares it in its YAML exactly as its executor requires it.

**Reading through branches** (CPU, `cargo test -p zend-vfs`)

- A git repository reads its branch as committed, never the folder: a file changed on disk, and one only on disk, are not seen.
- A store keeps its base while the branch moves; moving onto its own commit lets go of what landed; a three-way move merges and marks overlaps, flagged until settled; a snapshot restores the base and the conflicts; a failed move changes nothing.
- A commit lands whole or not at all — behind origin, origin moving during the push, conflicts unsettled — and a merge brings the other writer's commits into the conversation's copy: fast-forward, diverged history finished by a merge commit, origin moving on during a merge on either side, edit meeting delete, both-added, binary refusal, no origin (`tests/work.rs`, a bare repository as origin). A conflict outlives later merges; a commit is never undone by the next; local-only commits survive a rewind and a delete.
- The sandbox hands the folder back exactly as it found it — staged and unstaged edits byte for byte, line endings, permissions, skip-worktree, assume-unchanged and `add -N` entries, untracked files, removed and renamed files, files and folders swapped, links to folders, the branch (deleted or rewound by the job), an ignored file or folder at a path the job wrote or the branch adds, a file only uncommitted ignore rules ignore — after a job that succeeds, fails, is refused, is cancelled or is abandoned; a file the job names as no conversation could is cleared; a put-back that fails keeps everything and names the journal; a crash mid-job is recovered by the next, what stood kept first, and a folder worked in since is refused; a second preservation is refused while the first holds the lock; a link the command left is removed unfollowed; the server's log, stream, status, cancellation and eviction.

**Ingest** (CPU)

- The walk visits exactly the listed repositories, keys workspace-relatively, respects scope and depth, never walks uploads.
- The workspace unit lists the repositories and heads the pass; prefilled calls split `repo` and `path`.
- The fast path refuses a repository the workspace does not list, and never offers a file the conversation has changed.

**End to end** (GPU, `#[ignore]`d): a daemon over a multi-repository workspace answers a question whose answer lives in the second repository's files.
