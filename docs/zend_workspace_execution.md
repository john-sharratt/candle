# Zend: Workspace, Tooling and Execution Architecture

**Status:** Workspace, tools and ingest built (§3–§6). Cluster execution designed, not built (§7–§8).
**Scope:** How zend serves several repositories as one workspace; how the model reads and changes code in them; how the base conversation is seeded with every repository; and how real tooling (cargo, node, python, tests) will run across a cluster.

---

## 1. Summary

Zend serves a **workspace**: a folder holding a `workspace.yaml` that lists the repositories in scope. Each repository is a folder directly under the workspace, named by its folder — a git checkout or any other working tree. The daemon's own state (`substrate/`) sits in the workspace folder beside them, and a repository it owns, `uploads`, holds the files users upload.

Every file tool takes a `repo` argument and a path relative to that repository. The repository names are fixed for the daemon's life, so the `repo` parameter is a JSON-Schema `enum` of them, written into the tool definitions at startup; the constrained decoder (the stencil) compiles that schema, so the model can only ever name a repository that exists.

Internally, everything the ingest layers store is keyed by its **workspace-relative path** (`candle/src/lib.rs`), which is simultaneously a real path on disk and — split at its first segment — the `(repo, path)` pair a tool call takes.

The base conversation every dialogue forks from descends from a **priming chain** that reads the workspace listing and then, for each repository in manifest order, its root listing and its anchor documents. The `repo_map` and `code_reading` layers then ingest every repository in the background.

---

## 2. Goals and non-goals

### Goals

- Present several repositories as one workspace, addressed uniformly by `(repo, path)`.
- Keep the repositories where they are: local working trees, no copying, no checkout manager.
- Make an invalid repository name impossible to decode, not merely rejected after the fact.
- Seed the base conversation with every repository in scope.
- Keep zend's implementation simple: one manifest, one key form, one split.

### Non-goals (for now)

- **Git inside this document.** Reading repository state, committing the session overlay to a branch and pushing it to origin are the git layer's, designed in `docs/zend_git.md`. The working trees here are edited on disk (Mutable mode) or through the session overlay.
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

Rules, enforced when the workspace is built (`zend-tools/src/state/workspace.rs`):

- At least one repository.
- A name is one plain folder name: no separators, no `.`/`..`, nothing Windows would open as another name (a `:`, a trailing dot or space, an 8.3 `~N` tail), and not `secrets`. Names are unique, so no two repositories share a folder and none nests in another.
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
| `projection.yaml`, `tools/`, `identities/`, `<collection>s/`, raw layer folders | A *mind*'s configuration (a workspace carrying its own `projection.yaml`). Read from the workspace folder, never from a repository. |

Anything else in the folder — other checkouts, stray files — is outside every repository and invisible to tools, the walk and the watcher.

### 3.4 Two path forms, one split

| Form | Example | Used by |
|---|---|---|
| Workspace-relative key | `candle/zend/src/main.rs` | The walk, `repo_map` / `code_reading` metadata and resume hashes, the watcher, uploads, the fast path. |
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

`ToolContext.files` is a `RepoFiles` (`zend-tools/src/state/files.rs`): one `VfsStore` per repository, each rooted at the repository's folder. A tool resolves `repo` to its store and the path inside it; `..` stops at the repository's root, so a path cannot reach a sibling repository or anything else in the workspace folder. The overlay/direct distinction, the `secrets/` refusal and the Windows spelling guards hold per store exactly as they did for the single root. A context built without a workspace (`ToolContext::new`, tests) is *detached*: each repository name gets an upper-only store on first use.

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

### 5.1 The walk

`repo_scan::walk_workspace(workspace, scope, max_depth)` walks each listed repository from its own root — so a repository's own `.git` never prunes it, while nested checkouts and submodules *inside* a repository are still pruned — and keys every file workspace-relatively. Nothing outside the listed repositories is visited, and the uploads repository is never walked.

- `scope` (`--ingest-dir <layer>=<folder>`) narrows a code layer to one workspace-relative folder inside a repository, e.g. `code_reading=candle/zend/src`. Keys stay workspace-relative.
- `--max-depth N` counts components below each walk's start — a repository's root, or the scope folder — so `1` is a repository's own files. The map records the bound in workspace-relative components.

### 5.2 `repo_map` units

One unit per directory holding walked files (`candle/`, `candle/zend/src/`, …), plus a **workspace unit** (`.`) whenever anything was walked: its evidence is the set of repositories reached, and its listing turn is `file_list` with `repo: "*"`, which lists the repositories. Each unit's prefilled calls split its directory into `repo` and `path` (`file_list {repo: candle, path: zend/src}`; the workspace unit's is `{repo: "*"}`), and the request names the folder with its repository.

### 5.3 `code_reading`

A file's hidden conversation opens with ``Read the entire contents of `zend/src/main.rs` in the `candle` repository …``; the model decodes its own `file_read` calls, constrained to the repository enum. Resume hashes are path-qualified with the workspace-relative key.

### 5.4 Watcher and fast path

- The watcher watches each repository folder (the uploads repository included, which drives only the upload-deletion reconcile) and each raw layer folder — never the workspace folder itself, which may hold other checkouts.
- The fast path, which serves a `file_read` from a `code_reading` conversation that already read the same bytes, joins the call's `repo` and `path` into the workspace-relative key the ingest hashed under. A `repo` the workspace does not list is never joined onto the workspace; the call runs for real and the tool refuses it.

---

## 6. Seeding the base conversation

`priming_chain::build` runs once at startup, before any dialogue exists:

```text
workspace ls                          (the "." unit — the repositories)
  -> for each repository, in manifest order:
       repo ls                        (its root folder unit)
         -> README.md -> ARCHITECTURE.md -> AGENTS.md -> CLAUDE.md
  -> base_conv, every ingest base, every repo_map folder, every code_reading file
```

Each link is parented on the one before it, so the last link's projection carries the whole chain; `base_conv` is parented on the last link, so the first question ever asked already has every repository's listing and anchor documents behind it. Each link is also an ordinary `repo_map` / `code_reading` entry with the same resume key, so the background pass that follows finds it already done. Missing anchors, and repositories with no files at their root, are skipped.

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

1. **Adding or removing a repository while conversations are live.** The manifest is read once at startup; a change takes a restart, which recalibrates the tools that take `repo` (§4.3). Whether the removed repository's ingested conversations are retired, or kept frozen, is not yet decided — today they stay in the substrate and simply stop being refreshed.
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

**Ingest** (CPU)

- The walk visits exactly the listed repositories, keys workspace-relatively, respects scope and depth, never walks uploads.
- The workspace unit lists the repositories and heads the pass; prefilled calls split `repo` and `path`.
- The priming chain plans each repository in manifest order with its own anchors.
- The fast path refuses a repository the workspace does not list.

**End to end** (GPU, `#[ignore]`d): a daemon over a multi-repository workspace answers a question whose answer lives in the second repository's files.
