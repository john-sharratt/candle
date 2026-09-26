# Zend: Git Layer

**Status:** The library (§3–§8) is built and tested (§11) as the `zend-git` crate, and the model-facing tools (§10) are built and tested as the `git_*` family in `zend-tools`. The VFS purge (§9) is designed, not built.
**Scope:** A strongly typed Rust interface over the `git` command line, for the repositories a zend workspace lists (`docs/zend_workspace_execution.md` §3). It reads repository state, writes commits and branches **without touching the user's checkout**, and fetches from and pushes to origin. Its first consumer is the VFS purge (§9): turning a repository's session overlay into a commit on a branch.

---

## 1. Summary

Every repository in the workspace is already a git working tree on disk. The git layer drives the installed `git` binary against it, using plumbing commands whose output formats are stable and machine-readable, and parses that output into Rust types. Nothing outside the crate spawns `git`, formats a git argument, or parses git output.

Three rules shape the whole interface:

1. **The user's checkout is never touched.** No operation writes the working tree, the index or `HEAD`. A commit is written as objects — blobs through `hash-object`, then tree and commit through one `fast-import` run — and a branch moves by reference transaction. No step involves an index at all. The user can be mid-edit in the same repository while zend commits beside them.
2. **Every ref move is a compare-and-swap.** A local ref update names the value it expects to replace; a push names the value it expects origin to hold. A concurrent change is a typed refusal, never an overwrite.
3. **Values are validated at the type boundary.** An `Oid`, a `BranchName`, a `RepoPath`, a `Rev` cannot be constructed from a string git would misread. An argument therefore cannot become a flag, a path cannot leave the repository, and a branch name cannot be `-x` or `HEAD`.

---

## 2. Goals and non-goals

### Goals

- Read repository state: `HEAD`, branches, remotes, status, diffs, history, tree listings, file contents at any revision.
- Commit a set of file changes onto any base commit without the working tree or index.
- Create, move and delete branches atomically, with the expected old value checked.
- Fetch from and push to origin over the transport the repository is already configured for (SSH on this machine).
- Merge in the object store, to rebuild a change onto a branch that moved.
- Be testable end to end on the CPU, offline, against real repositories and a local bare "origin".

### Non-goals

- **Operations that change the user's checkout**: `checkout`, `switch`, `reset`, `stash`, `merge` into a working tree, `rebase` of the user's branch. The user owns their checkout.
- **Porcelain output.** Anything whose output depends on the user's pager, colour or locale settings.
- **Pull requests and other GitHub API calls.** A separate `github` client, authenticated with `github_token` from the secrets file (`zend-tools/src/state/secrets.rs`), consumes the branches this layer pushes.
- **A pure-Rust git implementation.** `git2` lacks partial fetch and `gix` lacks HTTPS push; the command line has everything, and the machine already has it.

---

## 3. Placement

The `zend-git` crate, depending on nothing else in the workspace. One concern per file:

```
zend-git/src/
  lib.rs              Repo: open (top-level check), dir, format, write lock
  version.rs          GitVersion, MINIMUM, installed()
  runner.rs           Invocation: args, -c overrides, env, stdin, timeout
  kill_tree.rs        Process-tree kill (Job Object / process group)
  error.rs            GitError
  classify.rs         exit status + stderr → GitError
  changeset.rs        ChangeSet, Change
  worktrees.rs        worktrees, add_worktree, remove_worktree
  testing.rs          (tests only) TestRepo: nested repos under scratch/
  types/
    oid.rs            Oid, ObjectFormat
    ref_name.rs       RefName, BranchName, RemoteName
    tag_name.rs       TagName
    remote_url.rs     RemoteUrl
    repo_path.rs      RepoPath, PROTECTED_SEGMENT
    rev.rs            Rev
    mode.rs           FileMode
    signature.rs      Signature, GitTime
  read/
    head.rs           Head
    refs.rs           resolve, ref_target, branches, remotes, merge_base, is_ancestor
    remote_branches.rs remote_branches
    tags.rs           tags
    status.rs         porcelain v2 parser
    diff.rs           raw diff parser, diff, diff_worktree
    patch.rs          patches, patches_worktree, patch_bytes
    log.rs            CommitInfo, LogRange, log, file_history
    blame.rs          blame
    grep.rs           grep at a revision
    tree.rs           ls_tree, tree_entries
    attrs.rs          ignored
    blob_reader.rs    long-lived cat-file --batch, size cap and deadline
  write/
    blobs.rs          write_blob (hash-object through filters)
    fast_import.rs    the commit stream
    commit.rs         commit_changes
    ref_txn.rs        update_refs, create/move/delete_branch
    upstream.rs       set_upstream, unset_upstream
    merge_tree.rs     merge_trees, commit_tree
    pick.rs           cherry_pick, revert
    tags.rs           create_tag, delete_tag
    apply.rs          apply_patch (private index)
  remote/
    ls_remote.rs
    fetch.rs
    push.rs           branches and tags, updates and deletes
    manage.rs         add_remote, remove_remote, set_remote_url
  setup/
    init.rs           Repo::init
    clone.rs          Repo::clone, CloneOptions (partial)
```

---

## 4. Invoking git

### 4.1 Version

`git --version` is read once per process when the first `Repo` opens. The minimum is **2.24**, the release that added `--end-of-options`; every other command and flag the layer uses is older. Older or missing git is `GitError::GitMissing` / `GitError::GitTooOld { found, need }`.

The floor is deliberately low: a deployment on an older distribution's git must work, so the layer does not depend on recent releases. Where a newer git offers a convenience, the layer uses the older equivalent — a three-way merge through `read-tree` and `merge-file` rather than `merge-tree --write-tree` (2.38), before-and-after ref snapshots rather than `fetch --porcelain` (2.41), newline-delimited `cat-file` and `worktree list` input rather than their `-z` modes (2.43, 2.36), plain `update-ref --stdin` rather than its `start`/`commit` verbs (2.27), `init` plus `symbolic-ref` rather than `init -b` (2.28), and the per-file grep limit applied in Rust rather than `grep --max-count` (2.38). The protections security releases added are built into the layer instead (§4.7), so they hold on every supported version. The test suite runs against Git 2.24, a current release and the machine's own (§11).

### 4.2 Arguments

Every invocation is built by `runner::Invocation`, never by formatting a string:

- `-C <repo dir>` first, so no call depends on the process's working directory.
- `--end-of-options` before the first positional argument, so a value that begins with `-` is never read as a flag. The types in §5 already refuse such values; this is the second guard. `rev-parse`, `grep` and the `remote` subcommands parse their own arguments and take no `--end-of-options` before 2.30; for them the typed values are the only guard, which suffices because no `Rev`, `RefName`, `RemoteName` or `RemoteUrl` can begin with `-` and a grep pattern always follows `-e`.
- Values are passed as separate arguments. Nothing is ever joined into a shell command, and no shell is involved.
- Bulk input (paths, refs, object ids, the fast-import stream) goes through stdin, never the command line, which on Windows is capped at 32 KiB.

### 4.3 Configuration

The user's git configuration is **inherited**, not disabled. It carries what must match their checkout: `core.autocrlf` (set in the system config by Git for Windows), `.gitattributes` handling, and the SSH settings their pushes already use. Disabling it would make zend's blobs differ byte-for-byte from the user's for the same file.

What is inherited but unsafe is overridden per invocation with `-c`, before the subcommand:

| Override | Why |
|---|---|
| `core.hooksPath=NUL` (Windows) / `/dev/null` | `push` runs `pre-push`; `update-ref` runs `reference-transaction`. A user hook must not run under zend. The null device can hold no hook file; a real empty folder would have to live somewhere, and in a shared temp directory another user could create it first. |
| `core.fsmonitor=` (empty) | No daemon spawned into the user's repository. Empty rather than `false`: before 2.36 the setting named a hook to run, and `false` would run a program of that name. |
| `protocol.ext.allow=never`, `protocol.fd.allow=never` | Transports that run commands instead of fetching (§5.2a). |
| `submodule.recurse=false` | Submodules are never recursed into (§4.7). |
| `transfer.bundleURI=false` | A server cannot point the client at further downloads (§4.7). |
| `core.quotePath=false` | Non-ASCII paths are printed as themselves. |
| `gc.auto=0`, `maintenance.auto=false` | Writing objects must not trigger a repack of the user's repository. |
| `commit.gpgSign=false`, `tag.gpgSign=false` | Signing prompts for a key agent; zend has no terminal. |
| `core.pager=cat`, `color.ui=false` | Plumbing output is never paged or coloured. |
| `advice.*=false` | No advisory text mixed into stderr the classifier reads. |

### 4.4 Environment

Every child inherits the daemon's environment **minus** the variables that would point git at another repository, index or object store, inject configuration, re-enable a disabled transport, or supply an identity the caller did not choose: `GIT_ALLOW_PROTOCOL` (which overrides `protocol.ext.allow`), `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_COMMON_DIR`, `GIT_NAMESPACE`, `GIT_CEILING_DIRECTORIES`, `GIT_DISCOVERY_ACROSS_FILESYSTEM`, `GIT_CONFIG_PARAMETERS`, `GIT_CONFIG_COUNT`, the six `GIT_AUTHOR_*` / `GIT_COMMITTER_*` variables, and `LANGUAGE`. It then sets:

| Variable | Value | Why |
|---|---|---|
| `LC_ALL`, `LANG` | `C` | English, stable messages for the classifier (§6). |
| `GIT_TERMINAL_PROMPT` | `0` | An auth failure fails; it never waits on a prompt. |
| `GIT_LITERAL_PATHSPECS` | `1` | A path such as `:(glob)*` names that file, never pathspec magic. `check-ignore` refuses the setting, so `ignored()` lifts it. |
| `GIT_NO_REPLACE_OBJECTS` | `1` | A read shows the objects a push would send, never a local `git replace` substitute. |
| `GIT_OPTIONAL_LOCKS` | `0` | On reads. Only `status` honours it: without it, `status` takes `index.lock` and rewrites the user's index. |
| `GIT_AUTHOR_*`, `GIT_COMMITTER_*` | from `Signature` | On `commit-tree` only; identity is always explicit (§5.6). |

No secret is ever placed in the environment or on the command line. The SSH transport authenticates through the user's own SSH setup (`~/.ssh`, the SSH agent), exactly as their manual pushes do.

### 4.5 Timeouts and failure

Every invocation has a timeout: 30 s for local operations, 120 s for network ones. On expiry the whole process tree is killed — a Job Object on Windows (which also kills anything the job still holds when its handle closes), a process group on Unix — because `ssh` is a grandchild of `git push` and a hook or alias runs a shell. The call returns `GitError::Timeout { args, after }`. After a normal exit the tree is killed too, so a leftover grandchild holding the pipes cannot stall the read. A non-zero exit is classified (§6); stdout is parsed only after an accepted exit status.

### 4.6 Concurrency

Reads run concurrently. Writes to one repository — object writes, ref transactions, merges, fetches and pushes — are serialised by a lock inside `Repo`, so two writers in one process never interleave. Ref correctness does not depend on the lock (§8.3 checks old values); the lock only prevents avoidable refusals.

### 4.7 Security that does not depend on the git version

The layer does not require a recent git to be safe. Each class of fix recent releases shipped is closed by the layer on every supported version:

| Risk | Fixed upstream in | Closed here by |
|---|---|---|
| Clone-time code execution through submodules (CVE-2024-32002, CVE-2025-48384) | 2.45.1 / 2.50.1 series | Submodules are never cloned or recursed into: `clone --no-recurse-submodules`, `fetch --no-recurse-submodules`, `submodule.recurse=false`. |
| Local-clone attacks through the hardlink/copy shortcut (CVE-2024-32004, -32020, -32021) | 2.45.1 series | `clone --no-local --no-hardlinks`: a local path goes through the ordinary transport. |
| Protocol injection through bundle URIs (CVE-2025-48385) | 2.50.1 series | `transfer.bundleURI=false`. |
| Credential leaks through URLs carrying a carriage return (CVE-2024-52006) | 2.48.1 series | `RemoteUrl` refuses every control character; prompts are off (`GIT_TERMINAL_PROMPT=0`). |
| Command execution through `ext::` URLs | — | Refused by `RemoteUrl`, disabled in config, and `GIT_ALLOW_PROTOCOL` scrubbed. |
| A hook, fsmonitor daemon or text-conversion filter from repository config | — | Hooks at the null device, `core.fsmonitor=` empty, `--no-textconv` on `blame`, `grep` and patches. |
| A remote name read as a path or URL | — | Every network operation requires a configured remote (`UnknownRemote` otherwise). |
| A credential in a URL leaking through output | — | User-info redacted to `***` in errors, argument lists and `remotes()`. |

Upgrading git remains worthwhile — these cover how zend uses git, not every way git can be used — but zend's safety does not wait on it.

---

## 5. Types

Every type below has a fallible constructor and no public way around it.

### 5.1 `Oid`

A full object id: 40 lowercase hex characters (SHA-1) or 64 (SHA-256). The repository's `ObjectFormat` is read at open from its `extensions.objectformat` config, absent meaning SHA-1. Abbreviated ids are refused; git is always asked for full ids (`--no-abbrev`, `%H`).

### 5.2 `RefName`, `BranchName`, `RemoteName`

`RefName` is a full name under `refs/`, and enforces `git check-ref-format`'s rules in Rust: no `..`, no ASCII control characters, none of `~^:?*[\` or space, no component beginning with `.` or ending with `.lock`, no `@{`, not `@`, no trailing `.`, no empty component. `BranchName` is a short name whose ref is `refs/heads/<name>`, additionally refusing a leading `-` and the name `HEAD`. `RemoteName` is a single component not beginning with `-`. `TagName` is a short name whose ref is `refs/tags/<name>`, under the same rules. A test runs the same table of hostile names through `git check-ref-format`, so the Rust rules and git's cannot drift apart.

### 5.2a `RemoteUrl`

An SSH, HTTPS or `file://` URL, an scp-like `user@host:path`, or a local path. It refuses a leading `-`, control characters, and the `ext::` and `fd::` transports — `ext::` runs an arbitrary command rather than fetching. Every invocation also sets `protocol.ext.allow=never` and `protocol.fd.allow=never`, so a URL that reached a repository's config some other way still cannot run.

### 5.3 `RepoPath`

A repository-relative path as git stores it: `/`-separated, no leading `/`, no empty, `.` or `..` component, no component that Windows would open as `.git` (case-insensitive, trailing dots and spaces ignored, and the short name `GIT~1`), no backslash, no control character. It is **structural only**: every path git prints satisfies it, so reading a repository never fails on a path. `is_protected()` reports a component that resolves to `secrets` (`PROTECTED_SEGMENT`, matching `VfsStore`'s); the rule that such a path may not be written is the `ChangeSet`'s (§8.1).

### 5.4 `Rev`

What a caller may name as a revision: `Head`, an `Oid`, a `BranchName`, a `TagName` or a `RefName` — never a free string. A branch or tag is spelled as its full ref, so a ref or file of the same name can never shadow it.

### 5.5 `FileMode`

`Regular` (`100644`), `Executable` (`100755`), `Symlink` (`120000`), `Submodule` (`160000`), `Tree` (`040000`, also read as `40000`). Parsed from and printed to git's octal form exactly.

### 5.6 `Signature`

`{ name, email, when: GitTime }`, where `GitTime` is seconds since the epoch plus a UTC offset in minutes. A name or email containing `<`, `>` or a control character is refused. Every commit carries an explicit author and committer; nothing falls back to the user's configured `user.name`. Author and committer are both the **human** on whose behalf zend acts — the signed-in user (`--local-signin`, or the gateway identity). No commit names a model or assistant, in any field or trailer (CLAUDE.md, "No AI attribution").

Explicit time makes commits reproducible: the same base, changes, message and signatures give the same commit id, which is what lets the tests assert exact ids.

---

## 6. Errors

```rust
enum GitError {
    GitMissing(io::Error),
    GitTooOld { found: GitVersion, need: GitVersion },
    NotARepository { dir: PathBuf },
    UnknownRevision { rev: String },
    NotABlob { object: String, kind: String },
    RefLocked { name: RefName },
    StaleRef { name: RefName, detail: String },
    CheckedOutBranch { branch: BranchName },
    UnknownRemote { remote: RemoteName },
    RemoteExists { remote: RemoteName },
    AuthFailed { remote: RemoteName, detail: String },
    RemoteUnreachable { remote: RemoteName, detail: String },
    InvalidInput(String),             // a type constructor's refusal
    Malformed { command: &'static str, detail: String },
    Timeout { args: Vec<String>, after: Duration },
    Io(io::Error),
    Unclassified { args: Vec<String>, status: Option<i32>, stderr: String },
}
```

`classify.rs` maps an exit status plus stderr (in the `C` locale) to a variant. Remote wording is matched only for an invocation that talks to a remote, and revision wording only for one that resolves a revision. Anything unrecognised is `Unclassified` with the full stderr — never folded into a nearby variant. Authentication failures cannot be produced offline, so their mappings are pinned against OpenSSH's and git's exact text; every other mapping is tested by producing the condition against a real repository.

A **push rejection is a result, not an error** (§7.3): an atomic push that one branch fails reports every branch's outcome. The errors of `push` are the transport's.

---

## 7. Operations

All on `Repo`, opened with `Repo::open(dir)`. It checks the version, and refuses `dir` unless it is a working tree's own top level (`rev-parse --show-toplevel`, compared canonically): a subfolder of a repository, or a plain folder nested inside one, would otherwise let git walk up and operate on the enclosing repository.

### 7.1 Reading

| Method | Returns | Plumbing |
|---|---|---|
| `head()` | `Head::{Branch { branch, oid }, Detached(Oid), Unborn(BranchName)}` | `symbolic-ref -q HEAD`, `rev-parse -q --verify HEAD^{commit}` |
| `resolve(&Rev)` | `Oid` of a commit | `rev-parse --verify --end-of-options <rev>^{commit}` |
| `resolve_object(&Rev)` | `Oid` of the object itself, unpeeled — an annotated tag's own id | `rev-parse --verify <rev>^{object}` |
| `first_parent_ancestor(&Rev, back)` | `Ancestor::{Found(Oid), PastRoot { depth }}`: `rev~back`, or how far back the first-parent history goes | `rev-parse -q --verify <oid>~<back>^{commit}`, then `rev-list --first-parent --count` |
| `ref_target(&RefName)` | `Option<Oid>` | `rev-parse -q --verify` |
| `branches()` | `Vec<Branch { name, oid, upstream: Option<Upstream { remote, branch, ahead, behind, gone }> }>`; `remote` is `None` for a branch tracking another local branch (git's remote `.`), and `tracking_ref()` names what it is compared against | `for-each-ref` with NUL-separated `%(upstream:remotename)`, `%(upstream:remoteref)`, `%(upstream:track,nobracket)` |
| `remotes()` | `Vec<Remote { name, fetch_url, push_url }>` | `config -z --get-regexp` |
| `status()` | `Vec<StatusEntry>` | `status --porcelain=v2 -z --untracked-files=all --ignored=no`, optional locks off |
| `diff(from, to, paths)` | `Vec<DiffEntry>`, limited to `paths` when any are given | `diff-tree -r -z --raw --no-abbrev -M [-- <paths>]` |
| `diff_worktree(from)` | `Vec<DiffEntry>` | `diff-index -z --raw --no-abbrev -M`, then `hash-object --stdin-paths` (below) |
| `log(&LogRange, limit)` | `Vec<CommitInfo { oid, parents, author, committer, message }>` | `log -z --date=raw` with NUL-delimited fields |
| `ls_tree(rev, dir)` | `Vec<TreeEntry { mode, kind, oid, path }>` | `ls-tree -z --full-tree` |
| `tree_entries(rev, paths)` | the entries at exactly those paths | `ls-tree -z -t --full-tree -- <paths>` |
| `merge_base(a, b)` | `Option<Oid>` | `merge-base` |
| `is_ancestor(a, b)` | `bool` | `merge-base --is-ancestor` (exit 1 is `false`) |
| `ignored(paths)` | the subset `.gitignore` excludes; tracked files never | `check-ignore --stdin -z` |
| `blobs()` | `BlobReader` | one long-lived `cat-file --batch` (§7.2) |
| `remote_branches()` | `Vec<RemoteBranch { remote, branch, oid }>`, as of the last fetch | `for-each-ref refs/remotes/` (symbolic `HEAD` entries skipped) |
| `tags()` | `Vec<Tag { name, oid, target, annotated }>` | `for-each-ref refs/tags/` with `%(*objectname)` for the peeled target |
| `patches(from, to, context, paths)` | `Vec<FilePatch>` (§7.4) | `diff-tree --patch-with-raw -z --full-index -U<n>` |
| `patches_worktree(from, context)` | `Vec<FilePatch>` | `diff_worktree`, then `diff-index --patch-with-raw` on the paths that really changed |
| `patch_bytes(from, to)` | the patch as `git apply` takes it | `diff-tree -p --binary --full-index` |
| `file_history(rev, path, limit)` | `Vec<CommitInfo>`, following renames | `log --follow -- <path>` |
| `blame(rev, path, lines)` | `Option<Vec<BlameLine>>` (§7.5); `None` when `rev` holds no such file | `blame --line-porcelain --no-textconv [-L a,b]` |
| `grep(rev, &GrepQuery)` | `Vec<GrepHit { path, line, text }>`; binary files skipped; the per-file limit applied in Rust | `grep -z -n -I --no-textconv --full-name -E|-F [-i] -e <pattern> <rev>` |

`LogRange` is `{ to, exclude, paths }`: commits reachable from `to` and not from `exclude`, limited to those touching `paths`. `file_history` differs from a one-path `LogRange` in following the file back through renames.

`StatusEntry` mirrors porcelain v2: `Changed { xy, path, head_mode, worktree_mode, head_oid, index_oid }`, `Renamed { xy, path, from, score }`, `Unmerged { xy, path }`, `Untracked { path }`, where `Xy` holds two typed `StatusCode`s. An untracked repository nested in the tree is listed by git as a folder, `sub/`, and read as `Untracked { path: "sub" }`.

**Reads take what the repository records.** A signature read from history — `log`, `blame` — is taken as stored (`Signature::recorded`), empty name or `Name <>` included, as imported SVN and CVS histories carry them; only a signature the layer writes is held to `Signature::new`'s rules. Likewise a blame `filename` git has C-quoted (it always quotes `"` and `\`, whatever `core.quotePath` says) is unquoted before it is parsed as a path. `DiffEntry` is `{ status: Added | Modified | Deleted | TypeChanged | Unmerged | Renamed(score) | Copied(score), old: Option<DiffSide>, new: Option<DiffSide> }`, a side being `{ mode, oid, path }`.

**`diff_worktree` never writes the index.** `git diff` refreshes stale stat information and writes the refresh back into the user's index, whatever `GIT_OPTIONAL_LOCKS` says — found by this layer's own test. `diff-index` writes nothing, but against an index nobody has refreshed it reports a file whose timestamps changed as modified, with an unhashed new side. Those candidates are hashed through the repository's filters (`hash-object --stdin-paths`), the same comparison a refresh makes, and dropped when their content matches `from`. A symlink or a submodule is never a candidate: `hash-object` cannot read either, and a submodule's unhashed side means its checkout moved or is dirty, which is a real change.

### 7.2 `BlobReader`

Starting a process per file read costs milliseconds on Windows, so file contents come from one `git cat-file --batch` per reader, kept running. `read_at(rev, path)` and `read_blob(oid)` write the request to its stdin, one per line — neither a revision spec nor a `RepoPath` can hold a newline — and read `<oid> <type> <size>\n<bytes>\n` back; a missing object is `Ok(None)` and a tree is `NotABlob`, with the reader left in step either way. A child that has exited is restarted on the next read, and the reader kills its process tree when dropped. Contents are bytes — the layer never decodes text.

Two bounds keep one read from taking the daemon with it. An object over `MAX_BLOB_BYTES` (64 MiB) is refused from its header as `BlobTooLarge`, before any buffer is allocated, and the child is restarted to drop the unread bytes. Each read has a deadline — the network one, since a read in a partial clone may fetch — enforced by a watchdog that kills the child's process tree, so a stalled fetch returns `Timeout` instead of holding the reader's lock forever.

### 7.3 Remotes

| Method | Returns | Plumbing |
|---|---|---|
| `ls_remote(remote)` | `RemoteRefs { head_branch, refs }` | `ls-remote --symref` (peeled tag entries dropped) |
| `fetch(remote, &FetchSpec)` | `Vec<RefUpdate { flag: New \| FastForward \| Forced \| Pruned, old, new, local }>` | `fetch --no-tags --no-recurse-submodules`, the tracking refs snapshotted before and after |
| `push(remote, &[PushSpec])` | `Vec<PushResult { target, outcome }>` | `push --porcelain --atomic --no-verify`, one `--force-with-lease=<ref>:<expected>` per spec |
| `add_remote(name, url)` | — (`RemoteExists` if taken) | `remote add` |
| `remove_remote(name)` | — (`UnknownRemote` if absent); its remote-tracking branches go too | `remote remove` |
| `set_remote_url(name, url, Fetch \| Push)` | — (`UnknownRemote` if absent) | `config remote.<name>.url` / `.pushurl` |

`FetchSpec` is `Branch(BranchName)` or `AllBranches` (which also prunes), and both write only under `refs/remotes/<remote>/`: a fetch can never move a local branch and fetches no tags. What changed is read from the tracking refs themselves — snapshotted before and after, a moved ref classified by ancestry — rather than from git's report, which only newer releases print in a machine-readable form.

Every network operation first requires `remote` to be configured (`UnknownRemote` otherwise): git reads a name no remote has as a URL or path, so `push("foo")` would push to a repository at `<repo>/foo`. `remotes()` reports URLs with any embedded credential redacted, and so does every error — a push that fails before reporting any ref goes through the same classifier as every other failure.

**Timeouts.** A call that only asks a remote something (`ls_remote`) has two minutes. One that moves objects — `clone`, `fetch`, `push` — takes as long as the repository is large, so it has an hour, and an HTTP transfer that stalls (under 1,000 bytes a second for 60 seconds, `http.lowSpeedLimit`/`lowSpeedTime`) is abandoned by git well before that. A clone cut off by a timeout cannot clean up after itself, so `clone` removes what it wrote — the folder it created, or the contents of the empty folder it was given — and a retry finds the target as the first attempt did.

A `PushSpec` is `{ action, target, lease }`. `PushTarget` is `Branch(BranchName)` or `Tag(TagName)`; `PushAction` is `Update(Oid)` or `Delete`; `Lease` is `Expect(Oid)` (the remote must hold this) or `Absent` (the ref must not exist). Constructors cover the common cases: `PushSpec::branch`, `PushSpec::tag` (always `Absent` — a tag is published once; it takes the tag's own object, from `resolve_object`, so an annotated tag arrives with its message and tagger rather than as a lightweight tag on its commit), `PushSpec::delete_branch` and `PushSpec::delete_tag` (always `Expect`). A delete with `Absent` is refused before any push. There is no unconditional force; a matching lease does allow a non-fast-forward, which is how a branch zen owns is rebuilt. `--atomic` makes a multi-ref push all-or-nothing. Each result is `Created`, `FastForward`, `Forced`, `Deleted`, `UpToDate`, or `Rejected(Stale | AtomicAborted | Remote(reason) | Other(summary))`, parsed from `--porcelain` output.

### 7.4 Patches

A `FilePatch` is `{ entry: DiffEntry, binary, hunks }`; a `Hunk` is `{ old_start, old_lines, new_start, new_lines, context, lines }`; a `PatchLine` is `{ kind: Context | Added | Removed, text, no_newline_at_end }`. `added()` and `removed()` give the line counts `--numstat` reports, `None` for a binary file.

One `--patch-with-raw -z` run gives the raw entries (NUL-delimited, so paths are exact) followed by the patch text, one section per file in the same order. Sections are paired with entries by position, never by parsing paths out of `diff --git` headers, which are ambiguous for paths with spaces; a type change is two sections in git's output and is paired as one. Each hunk takes exactly the lines its header counts, so a content line such as `--- x` can never be read as a header, and a count mismatch is `Malformed`.

### 7.5 Blame

A `BlameLine` is `{ commit, orig_line, final_line, orig_path, author, committer, summary, content }`: which commit last changed the line, where the line sat in that commit's version of the file — `orig_path` differs after a rename — and the line's bytes. `LineRange { start, end }` limits the blame to those lines.

### 7.6 Setup and worktrees

| Method | Result |
|---|---|
| `Repo::init(dir, branch)` | An empty repository, unborn on `branch`. |
| `Repo::clone(url, dir, &CloneOptions { branch, partial, checkout })` | A clone with `origin` configured. `partial` fetches commits and trees only (`--filter=blob:none`); file contents arrive from the remote on first read, through a checkout or a `BlobReader` alike. |
| `worktrees()` | `Vec<Worktree { path, head, branch, detached, bare, locked, prunable }>`, the main worktree first (`worktree list --porcelain`). |
| `add_worktree(path, &WorktreeCheckout)` | A linked worktree at `path` on a new branch, an existing branch, or a detached commit, opened as a `Repo`. |
| `remove_worktree(path, force)` | Removes it; without `force`, one with local changes is refused. |

`init`, `clone` and `add_worktree` take an absolute path that does not exist or is an empty folder; a worktree path may hold no control character, since `worktree list` is read line by line. A linked worktree shares the repository's object store and branches but is a separate checkout: a build can run there without touching the user's. `clone` always passes `--no-local`, `--no-hardlinks` and `--no-recurse-submodules` (§4.7). `init` points the new unborn `HEAD` at the branch with `symbolic-ref`.

---

## 8. Writing commits without the checkout

### 8.1 `ChangeSet`

```rust
struct ChangeSet { changes: BTreeMap<RepoPath, Change> }
enum Change {
    Write { content: Vec<u8>, mode: Option<FileMode> }, // None: the base's mode, else Regular
    Delete,
}
```

`write(path, content, mode)`, `symlink(path, target)` and `delete(path)` keep one change per path in path order. All refuse a protected path (§5.3) — nothing the file tools refuse to serve may be committed on the model's behalf. `write` takes `Regular` or `Executable` only: a symlink is written through its own `symlink` call, because a symlink's content is where it points, and one committed to a repository can lead any later reader of the checkout outside it.

### 8.2 `commit_changes`

```rust
fn commit_changes(&self, base: &Oid, changes: &ChangeSet, message: &str,
                  author: &Signature, committer: &Signature) -> Result<Oid, GitError>
```

Returns a new commit whose single parent is `base` and whose tree is `base`'s with `changes` applied. It moves no ref — the caller publishes it with §8.3 or §7.3.

1. **Modes.** Writes with no mode take the base's (one `tree_entries` call for all of them), or `Regular` for a new path. A write with no mode over a path the base holds as a symlink is refused: inheriting the link's mode would make the written text a link target the caller never asked for — the purge, writing overlay text, would otherwise turn a repository symlink into one pointing wherever the text says.
2. **Blobs, through the repository's filters.** Each write is stored with `hash-object -w --stdin --path=<path>` (`write_blob`), which applies the clean filters, `core.autocrlf` and `.gitattributes` for that path. A file read from a Windows checkout holds CRLF where the committed blob holds LF, and this step makes them agree. A symlink's target is stored with `--no-filters`.
3. **Tree and commit, in one process.** A `fast-import` stream (`write/fast_import.rs`) writes the commit onto a private scratch ref, `refs/zen/scratch/<pid>-<nanos>-<n>`: `from <base>`, then `M <mode> <blob> "<path>"` per write and `D "<path>"` per delete. Paths are always quoted, and the message is stored ending in exactly one newline. The stream names blobs only by id, because `fast-import` applies no filters to inline content.
4. **Read back.** The commit id is read from the scratch ref, and the ref is deleted in a compare-and-swap transaction.

The stream's bytes are pinned by a test, and the commit id is checked against an independent oracle: `git commit-tree` over the same tree, parent, identities and message gives the same id.

### 8.3 Reference transactions

```rust
fn update_refs(&self, txn: &RefTransaction) -> Result<(), GitError>
enum RefOp {
    Create { name: RefName, new: Oid },           // must not exist
    Update { name: RefName, new: Oid, old: Oid }, // must hold `old`
    Delete { name: RefName, old: Oid },           // must hold `old`
}
```

Runs `update-ref --no-deref --stdin -z` with one command per op — `--no-deref` so an op changes the ref it names and never the one a symbolic ref points at, which would walk around the checked-out check below; `update-ref` locks every ref and checks every expected value before changing any, so the script applies whole or not at all. Every op carries its expected old value, so the whole transaction either applies or fails as `StaleRef` naming the ref that did not hold its value; a ref another process has locked is `RefLocked`. `create_branch`, `move_branch` and `delete_branch` are one-op transactions. A branch checked out in any worktree — the main one or a linked one — is refused as a target (`CheckedOutBranch`): moving it would change that checkout under it.

### 8.4 Upstreams

```rust
fn set_upstream(&self, branch: &BranchName, remote: &RemoteName, remote_branch: &BranchName) -> Result<(), GitError>
fn unset_upstream(&self, branch: &BranchName) -> Result<(), GitError>
```

`set_upstream` records what `git push -u` records: `branch.<name>.remote` and `branch.<name>.merge`. The user's own `git status`, `pull` and `push` then work on the branch, and `branches()` reports how far apart it and its upstream are. The branch may track a differently named branch on any remote — the fork layout, where `upstream` is the shared repository.

The keys are written with `git config` rather than `git branch --set-upstream-to`, which refuses unless the remote-tracking ref already exists; an upstream set before the first push reads as `gone` until a fetch brings the tracking ref in. Git treats a branch as having an upstream only when `merge` is set, so `remote` is written first and removed last: a failure between the two writes never leaves a half-configured upstream. Both refuse a local branch that does not exist; unsetting a branch with no upstream is a no-op. It changes config only, so the checked-out branch may track too.

### 8.5 Rebuilding onto a moved base

```rust
fn merge_trees(&self, merge_base: &Oid, ours: &Oid, theirs: &Oid) -> Result<MergeOutcome, GitError>
enum MergeOutcome { Clean { tree: Oid }, Conflicted { paths: Vec<RepoPath> } }
fn commit_tree(&self, tree: &Oid, parents: &[&Oid], message: &str,
               author: &Signature, committer: &Signature) -> Result<Oid, GitError>
```

`merge_trees` is a three-way merge computed in a private index (§8.8), with plumbing every supported git has:

1. `read-tree -m -i --aggressive <base> <ours> <theirs>` settles every path only one side changed, both changed identically, or either deleted unchanged.
2. Each path left unmerged that both sides modified as a regular file is merged line by line with `merge-file`, the three versions written to scratch files in the git folder. Any other conflict — modify/delete, add/add, a mode or type clash, overlapping lines, binary content — is reported.
3. The merged files replace their unmerged entries (`update-index --index-info`), and `write-tree` stores the result.

No rename detection runs: a file renamed on one side and edited on the other is a conflict rather than merged across the rename. When a branch zen committed to has moved on origin, the change set's commit is merged against the new head; a clean tree is committed with `commit_tree` and pushed with a fresh lease. A conflict is returned as the paths that need a human, never written into a working tree.

### 8.6 Cherry-pick and revert

```rust
fn cherry_pick(&self, commit: &Oid, onto: &Oid, committer: &Signature) -> Result<PickOutcome, GitError>
fn revert(&self, commit: &Oid, onto: &Oid, author: &Signature, committer: &Signature) -> Result<PickOutcome, GitError>
enum PickOutcome { Clean(Oid), Conflicted { paths: Vec<RepoPath> } }
```

Both are three-way merges in the object store. A cherry-pick merges `commit` into `onto` over `commit`'s parent and keeps the original author and message, as `git cherry-pick` does. A revert merges `commit`'s parent into `onto` over `commit`, with `git revert`'s message: ``Revert "<subject>"`` and `This reverts commit <id>.`. Root and merge commits are refused, since neither has a single change to replay. The result is a commit for the caller to publish; nothing moves and the checkout is untouched.

### 8.7 Tags

```rust
fn create_tag(&self, name: &TagName, target: &Oid, annotation: Option<&TagAnnotation>) -> Result<Oid, GitError>
fn delete_tag(&self, name: &TagName, old: &Oid) -> Result<(), GitError>
struct TagAnnotation { message: String, tagger: Signature }
```

Without an annotation the tag is lightweight: the ref holds `target` itself. With one, the tag object is built as text (`object`, `type`, `tag`, `tagger`, blank line, message) and stored through `git mktag`, which validates it; the ref then holds the tag object. Both create the ref in a transaction that requires it not to exist, and delete requires its expected value. Publishing a tag is `PushSpec::tag` (§7.3).

### 8.8 Applying a patch

```rust
fn apply_patch(&self, base: &Oid, patch: &[u8]) -> Result<ApplyOutcome, GitError>
enum ApplyOutcome { Applied { tree: Oid }, Rejected { detail: String } }
```

Applies a patch — as `patch_bytes` produces, binary content included — to `base`'s tree. `git apply --cached` needs an index, so it gets a private one: a file named by `GIT_INDEX_FILE`, loaded with `read-tree <base>`, written out with `write-tree`, and deleted with its lock afterwards. Private indexes and merge scratch files live in the repository's own git folder (`write/scratch.rs`), never a shared temp folder where another user could create the path, or its lock, first. The user's index is never read or written. A patch that does not fit, and input that is not a patch at all, are `Rejected` with git's reason.

A patch is held to `ChangeSet`'s rules (§8.1), judged by what it actually changed — the applied tree diffed against `base`: a path under a protected folder, or a symlink or submodule created or changed, is refused as `InvalidInput`. A patch is otherwise a way round both.

---

## 9. First consumer: the VFS purge (not built)

A repository's overlay store (`zend-tools/src/state/vfs.rs`) holds session writes in `Upper.files` and deletions in `Upper.whiteouts`. A purge publishes them and empties the overlay:

1. **Snapshot.** Take the overlay's writes and whiteouts as a `ChangeSet` — writes as `Change::Write` (UTF-8 bytes of the stored text), whiteouts as `Change::Delete`.
2. **Filter.** Drop paths `ignored()` reports; they are listed back to the caller, not committed.
3. **Base.** `head()` of the repository. `Unborn` or `Detached` is reported, not guessed around.
4. **Warn on divergence.** `diff_worktree(base)` names any overlay path the user also changed on disk. Overlay content for such a path was read through the user's uncommitted edit, so the commit would carry that edit too; the purge reports the paths and its caller decides.
5. **Commit.** `commit_changes(base, …)` with the signed-in user as author and committer.
6. **Branch.** `create_branch(zen/<name>, commit)` locally, then `push` with `Lease::Absent` — or `Lease::Expect` of the last pushed head when the branch already exists, rebuilding through §8.5 when origin moved — and `set_upstream` so the user can check the branch out, pull and push it like one they pushed themselves.
7. **Empty.** Only after the push succeeds are the purged paths removed from the overlay, and only those whose content is unchanged since the snapshot. A failure at any step leaves the overlay as it was.

The overlay is currently one per repository for the whole daemon, shared across conversations (`zend/src/tools.rs`, `ToolHost`). Branch naming and per-conversation purges therefore depend on §12, question 1.

---

## 10. Model-facing tools

Nine `git_*` tools in `zend-tools/src/tools/git/`, each taking the workspace's `repo` enum (`docs/zend_workspace_execution.md` §4.3). They are request/response shells over this layer: nothing in `zend-tools` spawns or parses git. The catalog entry for each is a bundled definition in `zend/src/prompts/tools/git_*.yaml`; the tool-facing view — modes, paging, and why distinctions live inside tools rather than between them — is documented in `docs/tool-system.md` under "Git".

**Reads** — no capability, offered wherever tools are:

| Tool | Wraps |
|---|---|
| `git_status` | `status()`, `head()`, and the checked-out branch's upstream from `branches()` |
| `git_log` | `log(range, limit)` / `file_history` |
| `git_show` | by `what`: `diff` (`changes`), `patches` (`patch`), `BlobReader::read_at` (`file`), `ls_tree` (`tree`), `blame` (`blame`) |
| `git_grep` | `grep` at a revision |
| `git_refs` | by `kind`: `branches()`, `tags()`, `remotes()`, `remote_branches()` |

**Writes** — every one declares `DiskWrite`, which the Comprehensive tools mode's grants withhold, so they are Mutable's alone and are refused at dispatch anywhere else:

| Tool | Wraps |
|---|---|
| `git_commit` | by `from`: `commit_changes` (`files`), `apply_patch` + `commit_tree` (`patch`), `cherry_pick` / `revert`; then `move_branch` |
| `git_ref` | `create_branch` / `move_branch` / `delete_branch`, `create_tag` / `delete_tag` |
| `git_fetch` | `fetch` (also `Network`) |
| `git_push` | `push` (also `Network`) |

Writing on the model's request was not part of the original design, and what makes it acceptable is §8: no tool here touches the checkout, the index or `HEAD`, so a call cannot disturb uncommitted work, and a branch checked out in any worktree is refused. Every ref move is a compare-and-swap against the value it replaces — `expected` / `expected_head` / a push lease, each optional because the layer reads the current value itself when it is left out — so a concurrent change is a refusal rather than a loss.

Requests use the §5 types as their argument schemas, so an invalid revision or branch name is refused at validation. A revision is an object whose `kind` is an enum (`{"kind":"branch","name":…}`) rather than a string, so git's revision *syntax* is not expressible and no argument can begin with `-`. Object ids are returned in full, never abbreviated, because the id a tool prints is the id a later call must hand back.

One rule is layered on top of this crate rather than inside it: a `secrets` path segment is refused for reads as well as writes, and a wildcard read (`git_grep` with no paths, `git_show`'s file contents, a tree listing) has protected entries dropped on the way out. `ChangeSet` already refuses to commit one; the tool layer closes the read direction, because a key committed once stays in the object store forever.

Adding these changes the tool catalog, which recalibrates only the tools whose definitions changed; a deployment's substrate is rebuilt (`--wipe-substrate`) when they land, so routing is calibrated against the whole new catalog at once.

---

## 11. Testing

`cargo test -p zend-git`: CPU only, offline, 175 tests. The suite passes against Git for Windows 2.24.1 (the minimum), 2.45.1 and 2.55.0, each put first on `PATH` for the run; the portable builds are unpacked from Git for Windows' signed releases. Every repository a test uses is created with `git init` inside `zend-git/scratch/` — nested in the candle checkout and ignored by it (`zend-git/.gitignore`) — and deleted when the test ends. Setup asserts each test repository is its own top level, so no test can reach the candle repository around it. "Origin" is a bare repository in the same folder, reached over `file://` so fetch and push use git's smart transport. Setup commits use fixed identities and dates, so their ids are reproducible.

**Types and wire formats (exact bytes, no tolerances)**
- `Oid`, `RefName` / `BranchName` / `RemoteName` / `TagName`, `RemoteUrl`, `RepoPath`, `FileMode`, `GitTime` / `Signature` accept and refuse exactly the documented sets; the ref-name table agrees with `git check-ref-format` name for name.
- The `fast-import` stream, the `update-ref -z` transaction and the tag object serialise to exact expected bytes.
- Each parser — porcelain v2 status, raw diff, patch-with-raw (hunks, no-newline markers, binary, renames, header-like content lines, count mismatches), `for-each-ref` (branches, remote-tracking branches, tags), `config -z`, `log -z`, `blame --line-porcelain`, `grep -z`, `ls-tree -z`, `ls-files -u -z`, `ls-remote`, `worktree list --porcelain`, `push --porcelain` (including deletes and tags), the `cat-file` header — reproduces exact values from raw bytes, including renames, unmerged entries, non-ASCII paths and paths with spaces.
- Authentication and network stderr classify to their variants, with any credential in an echoed URL redacted; URL redaction keeps everything but the user-info.

**Behaviour against real repositories**
- `Repo::open` refuses a plain folder inside the candle checkout, a repository's subfolder, a folder outside any repository and a missing folder.
- `commit_changes` gives the same id as an independent `commit-tree`, and the same inputs give the same id twice.
- After `commit_changes`, the working tree, the index (bytes and modification time), `HEAD` and every ref are unchanged, with no scratch ref left behind.
- A CRLF file under `core.autocrlf=true` is stored as the same blob `git add` produces; `.gitattributes` `-text` keeps CRLF.
- A write with no mode keeps an executable bit; an explicit mode overrides it; a write with no mode over a base symlink is refused, an explicit mode replaces the link with a file, and a symlink is only written through its own call.
- `update_refs` fails whole, as `StaleRef`, when any op's expected value is wrong; a held lock is `RefLocked`; a branch checked out in the main or a linked worktree is refused.
- `push` with `Lease::Absent` or a stale `Lease::Expect` is rejected with origin unchanged; a matching lease allows a non-fast-forward; one rejection in an atomic push leaves every branch unchanged; an unreachable remote is an error, not a rejection.
- `fetch` reports new, fast-forwarded, rewritten and pruned tracking refs, reports nothing when nothing changed, and creates no local branch; `remote_branches` lists every fetched branch of every remote.
- An unconfigured remote name is refused by push, fetch and `ls_remote`, and nothing reaches a repository sitting at that path.
- A remote branch is deleted only when it holds the expected commit; a tag pushes once and deletes under a lease; a delete with an absent lease is refused before any push.
- Patch line counts agree with `git diff --numstat` for every file; hunks hold the exact lines; a file-to-symlink type change pairs both sections; a worktree patch covers real edits only and leaves the index untouched.
- `file_history` follows a rename where a path-limited `log` stops at it.
- Blame names the commit that last changed each line, honours a line range, blames an older revision's own content, reports the pre-rename path, and is `None` for a missing file or a folder.
- `grep` searches another branch without touching the checkout, and applies case, fixed strings, path limits and per-file limits; binary files are skipped; a pattern starting with `-` is a pattern.
- An annotated tag is byte-for-byte the object `git tag -a` makes; a cherry-pick has the same id as `git cherry-pick`'s and a revert as `git revert`'s; a conflicting pick names its paths and commits nothing; root and merge commits are refused.
- A patch from one commit to another, applied to the first in a repository holding only it, gives the second's tree id exactly — text, binary, added, deleted and renamed files; a patch that does not fit is rejected with the index untouched.
- Remotes are added, repointed (fetch and push URLs) and removed with their tracking branches; an `ext::` URL never runs, even when written straight into config.
- `init` creates an unborn branch and refuses an occupied or relative target; `clone` checks out the default or a named branch, or nothing; a partial clone lacks an old blob until a blob read fetches it.
- A worktree is added on a new branch, listed, committed in — moving the shared branch while the main checkout stays untouched — refuses a branch checked out elsewhere, and is removed; a detached worktree with local changes needs `force`.
- A branch zen creates, pushes and gives an upstream carries the exact config keys `git push -u` writes; git's own `@{upstream}` resolves it; ahead/behind follow a local commit and a fetched remote one; an upstream set before the remote branch exists is `gone` until fetched; a branch may track another remote under another name; unsetting is idempotent; a missing branch is refused.
- `merge_trees` merges disjoint files cleanly, merges edits to different lines of one file to the same tree `git merge` produces, names the conflicting path of overlapping edits, modify/delete and binary clashes, and leaves no scratch file behind.
- A `pre-push` hook that demonstrably runs for ordinary git, and a `reference-transaction` hook, do not run under the layer; hooks point at the null device.
- A text-conversion filter that demonstrably runs for plain `git blame --textconv` does not run under `blame` or `grep`.
- A `git replace` substitute plain git honours is ignored by the blob reader.
- `status()` succeeds while another process holds `index.lock`, leaving the lock and the index untouched; `diff_worktree` ignores stat-only changes and leaves the index untouched.
- A path that looks like pathspec magic is read literally.
- The blob reader returns exact bytes (including NUL and CRLF) at any revision, stays in step after a missing path and a folder, restarts a dead child, refuses an object over its size limit and recovers, and times out a read that never answers.
- Version checks refuse only releases before 2.24.
- A timeout returns at the deadline and kills the whole process tree: an alias's shell that would write a marker after the timeout never does.
- The child environment has every scrubbed variable removed and the pinned ones set, and every unsafe setting is overridden before the subcommand.

---

## 12. Open questions

1. **One overlay per conversation.** Per-conversation branches need per-conversation overlays; today one overlay per repository serves every conversation. Which conversation a purge belongs to — and so its branch name — is undecided until the overlay is split.
2. **Dirty-path policy** (§9 step 4): refuse the purge, commit anyway, or commit only the lines the model changed.
3. **When a purge runs**: on request, at the end of a tool round, or when a conversation closes.
4. **Branch naming**: `zen/<conversation label>` is readable but can collide; `zen/<conversation id>` is unique but opaque.
5. **Linear or merge history** when rebuilding onto a moved base (§8.5).
