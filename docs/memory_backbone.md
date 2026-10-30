# The memory backbone: journal trees for people and their families, in git

**Status: building** (2026-10-30). Built: the `journals` crate (§14 steps 1,
5–7), the agent's flag and zend's tools on the tree (steps 2–4). Next: zend's
background thread (step 8), then the writers (step 9). It generalises
`docs/mcp_user_context.md` (the flat `U:` journal, `get_user_context`, the
first-call context) and the `/mcp` journals of `docs/claude_zend_gateway.md`
§4–§5 (threads, workers, lineage, `P:`), and extends the windowing and folding
of the journal engine (`docs/zend_journal.md`) to every journal. Where this
document and those disagree about a person's memory, this one is authoritative;
each is fixed in the change that builds the part it describes. The research
behind it is summarised in §13.

## 1. What it is for

A person talks to many AIs on this estate — Grok in the car, claude.ai, Claude
Code, Zen's coding workers, the bots of Battle Cities — and each starts every
conversation knowing nothing. The backbone is one memory that all of them read
and write, so that a conversation begun anywhere already knows the person, what
they and their family are working on, what was done for them and what is still
open.

It is a **tree of journals**, kept as **YAML files in one git repository**: a
folder per journal, a commit per change. Most of it is **shared within a
family** — projects, topics, Zen threads, bots — and a small fixed part is
**personal**: the person's own profile, their conversations, and their notes
about other people. Each person has a folder and each family a folder; a
person's tree is their personal root with each of their families mounted under
it.

The top levels are fixed in code; below them the tree grows by name, at most
ten live sub-journals to a journal. Every journal has a capped summary, rebuilt
in the background when it is dirty, and a window of recent lines over
everything older, folded. Any AI reads from where it stands up to the root,
opens any journal's children from it, and greps any subtree it may read. It is
served by the build agents that hold the repository, through the gateway and
the connector.

## 2. The hierarchy

```
~                                    the person (persons/<person>/)         fixed
├── clients                          personal section                       fixed
│   ├── grok                         one per Client                         fixed (mcp_server::Client)
│   │   ├── 2026-10-30-2105          conversations                          named, windowed to 10
│   │   └── 2026-10-30-1943
│   ├── claude-code
│   └── other
├── people                           personal section
│   └── builder
└── @sharratt                        a family (families/<family>/)          mounted
    ├── projects                     shared section                         fixed
    │   └── candle                   named, ≤ 10 children each
    │       ├── scheduler
    │       └── cluster
    ├── threads
    │   └── user-context             a Zen thread
    │       ├── implement-memory     a worker
    │       │   └── verify-memory    a part
    │       └── review-memory
    ├── topics
    │   └── house-build
    │       ├── permits
    │       └── budget
    └── bots
        └── c3f1a07e                 a bot of the family's
```

### 2.1 Fixed at compile time: the scopes, the sections, the clients

```rust
/// Where a section's journals live, and so who reads them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scope {
    /// The person's own folder: read by their AIs alone.
    Personal,
    /// A family's folder: read by every member's AIs.
    Family,
}

/// The sections: the journals directly under a root, fixed in code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Section {
    /// Each client the person speaks through, and its conversations.
    Clients,
    /// The people in the person's life, as the person sees them.
    People,
    /// Repositories and durable projects.
    Projects,
    /// Zen feature threads, their workers and parts.
    Threads,
    /// Themes of life and work that span conversations.
    Topics,
    /// The family's bots, and what each did.
    Bots,
}

impl Section {
    pub fn scope(self) -> Scope {
        match self {
            Self::Clients | Self::People => Scope::Personal,
            Self::Projects | Self::Threads | Self::Topics | Self::Bots => Scope::Family,
        }
    }
}
```

**Everything is shared but the personal sections.** A section's scope is fixed
by its type, never chosen per line or per journal, so where a model writes
decides who reads it and no model is asked to judge what is private:

- **`clients`** is personal: what a person said to Grok in the car is theirs.
  Anything in it worth sharing is shared by moving it (§3.6).
- **`people`** is personal: a person's notes about their partner are not read
  by their partner.
- **`~`** itself — the person's profile, preferences and voice digest — is
  personal.

A section cannot be created, renamed or removed at runtime; every tool, guide
and permission names it by type. Under `clients`, the children are fixed too:
one per `mcp_server::Client` variant (`grok`, `claude-code`, `other`), named by
`Client::name`. A client added to that enum gets its journal by the same change.
A folder at the fixed levels that is not one of these is an error (§3.3).

### 2.2 Named below

Every journal below the fixed levels is named at runtime: a kebab-case slug of
at most 40 characters, unique among its siblings, numbered `.2`, `.3` when a
name recurs (`names::child_name`, as workers are today). Each carries a one-line
**cue** — "open this when…" — which its parent's index shows (§5.2).

- **At most 10 live sub-journals per journal**, at every depth.
- **Depth at most 6 below a section**, so a worker's lineage — every journal
  from it to the person — stays within the 16 journals and 32,000 characters
  `lineage` allows.
- **One parent**: the folder that holds it. A family journal's parent is the
  family root, never a person.

### 2.3 The eleventh

| Children | Created by | An eleventh live child |
|---|---|---|
| conversations | the server, at a conversation's first call | **folds the oldest** — the window slides |
| threads, workers, parts | Zen, at `thread_attach` / `repo_task_start` / `sub_run` | **folds the least recently active finished one**; a running one never |
| bots | npcd, when a member comes to own a bot | folds the least recently active |
| projects | the server, when a workspace repository is first named; sub-areas by the AIs | **refused**: "candle has 10 sub-journals; fold or merge one first" |
| topics, people | a person, or an AI on their behalf | refused, as for projects |

**Folding a journal** writes its summary into its parent as a `fold` line that
cites it and marks it `folded: true`. A folded journal keeps everything — its
folder stays, it is read by path, grepped, and reopened with `journal_unfold` —
it only leaves its parent's live index ("… and 37 folded"). Several members
writing fill a shared journal sooner; the rules are the same.

## 3. Files

### 3.1 The repository

One private GitHub repository, `john-sharratt/journals`, holds every person and
every family, each in its own folder:

```
journals/
  .gitattributes                         `lines.yaml` and `**/archive/*.yaml`: merge=union
  persons/<person>/                      a person's own (§3.4)
    person.yaml                          who they are: accounts, families
    journal.yaml  lines.yaml             ~ : metadata, summary; its window
    archive/2026-10.yaml                 its folded periods (§4)
    clients/
      journal.yaml  lines.yaml
      grok/
        journal.yaml  lines.yaml
        2026-10-30-2105/journal.yaml  lines.yaml
    people/builder/…
  families/<family>/                     a family's (§3.5)
    family.yaml                          its members and their roles
    journal.yaml  lines.yaml             the family root
    projects/candle/…
    threads/user-context/implement-memory/…
    topics/house-build/permits/…
    bots/c3f1a07e/…
```

A journal is its folder; its children are its subfolders. Moving or renaming a
journal is `git mv`. Anything at the top of the repository but
`.gitattributes`, `README.md`, `persons/` and `families/`, or a folder
under `persons/` without a `person.yaml` (or under `families/` without a
`family.yaml`), is an error (§3.3).

**The service is the privacy boundary.** Every agent holding the repository
holds everyone's folders (the operator's decision, 2026-10-30: one repository,
separated by folder). Who reads what is the journal service's rule (§7.2): a
caller's AIs read their own person's folder and their families' folders, and
nothing else is ever put in front of them.

### 3.2 `journal.yaml` and `lines.yaml`

A journal is two files, because they merge differently:

```yaml
# journal.yaml — metadata and summary: ordinary merge
name: candle
cue: "open for the inference engine, its releases and cluster"
created: 2026-10-30T09:10:02Z
folded: false
summary:
  over: 5d41402abc4b2a76                 # the fingerprint of what it was built over (§5.3)
  built: 2026-10-30T11:02:40Z
  lines: 12                              # window lines and live children it read
  children: 2
  sentences:
    - {text: "Releases go through keep trials on BlackWorld, then roll out.", cites: ["#20261029-0410-a3b4"]}
    - {text: "The connector now remembers its user.", cites: ["cluster"]}
```

```yaml
# lines.yaml — the window: union merge
- {id: 20261030-0910-4b1c, at: 2026-10-30T09:10:02Z, kind: episode, by: server, person: john, client: claude-code, text: "repo_task_start — implement-user-context"}
- {id: 20261030-0912-7e3a, at: 2026-10-30T09:12:40Z, kind: note, by: claude-code, person: john, text: "Workers push branches; merging is the user's decision."}
```

A cite is `#<line id>` — a line of the journal's window or archives — or a
live child's name. Free text is always double-quoted. Names, times, ids, keys
and fingerprints are plain — unless YAML would read the plain form as
something else (`true`, `null`, `1e3`, `0x1f`), when they are quoted too; a
number where text belongs is an error, never read as digits. A fold line
carries `period: "2026-10-29"` or `journal: <child>`, and a worker's
unaccepted note `proposed: true`.

- **`journal.yaml` merges as git always does**: two agents rebuilding one
  summary at once conflict, and the push of the second is refused (§5.3).
  Union merge here would keep both summaries and break the file.
- **`lines.yaml` is one line per entry**, each a flow-style mapping on one text
  line, with `merge=union`: two writers appending to one journal at once merge
  without conflict, git keeping both added lines. This is why entries are never
  multi-line YAML. Two branches editing the *same* line is the one case union
  cannot resolve: both versions are kept, and the duplicate `id` is an error
  (§3.3) naming the line — the journal reads as unreadable until a commit
  keeps one. The service never meets it: its own concurrent writes are
  replayed as operations, not merged (§6.3), so it comes only from branches
  merged by hand.
- **`id`** is `<yyyymmdd-hhmm>-<4 hex>`, the hash of the line's time, writer,
  text and a salt raised until it is free in its journal: stable, unique, and
  the address a line is cited and grepped by
  (`@sharratt/projects/candle#20261030-0912-7e3a`). Positions are never
  addresses.
- **`kind`** is one of three, set by who writes:
  - `note` — a model was told or chose to keep it (today's `Noted:`);
  - `episode` — the server recorded something that happened (today's `Done:`);
  - `fold` — a folded journal or period (§2.3, §4).
- **`by`**, **`person`** and **`client`** say who wrote it: the writer, the
  person it wrote for, and the client they spoke through. In a shared journal a
  line reads "Yen, via Grok"; the commit says the rest (§7.1).
- **Editing a line is a commit** that changes it. Nothing is lost: history keeps
  every version, and `git log -L` / `git blame` answer when and by whom.

### 3.3 Errors

A file that does not parse, a field that is wrong, an eleventh live
sub-journal, a
summary over its cap, a folder at the fixed levels that is not a section or
client, a shared section in a person's folder or the reverse, an account
listed by two people, a member no `person.yaml` names, a family a person lists
that does not list them: each is **an error,
answered as one**. `journal_open` on that journal answers the error, naming the
file, the line and what is wrong; the frame shows the journal as unreadable;
nothing is guessed, repaired or skipped silently. The service's own writes are
checked before they are committed (§6.3), so an error enters only from a hand
edit, a worker's commit or a bad merge, and is fixed by another commit.

### 3.4 The person

A person is named by a slug chosen when they are added — `john`, `yen` — which
names their folder, `persons/<person>/`, and so is unique. `person.yaml` lists
the accounts that are them and the families they belong to:

```yaml
# persons/yen/person.yaml
person: yen
name: "Yen"
accounts:
  - {email: "yen@example.com", key: 3b1f09c2d47e8a60}
  - {email: "yen.other@example.net"}                 # bound at its first sign-in
families: [sharratt]
default_family: sharratt
```

- **An account** is one sign-in at the gateway, known by zend's owner key,
  `sha256(provider \0 sub)` as 16 hex — the key the gateway's verified identity
  already gives every call. Several accounts may be one person; an account is
  one person's at most.
- **Binding.** An account is listed by its email before it has ever signed in.
  At its first call, the gateway's verified identity gives the key and the
  provider's *verified* email; the service finds the one `person.yaml` listing
  that email with no key, writes the key in (a commit), and from then on knows
  the account by its key alone. An unverified email binds nothing.
- **An account no person lists** is answered as not a member of any family, and
  has no memory until an owner adds it (§3.5).

The rest of identity — profiles, account recovery, a person outside every
family — is a later design; this needs only the accounts each person has.

### 3.5 Families

`family.yaml` lists who is in a family, and as what:

```yaml
# families/sharratt/family.yaml
family: sharratt
name: "Sharratt"
members:
  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}
  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}
  - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}
  - {person: yen-nhi, role: member, joined: 2026-10-30T00:00:00Z}
```

- **Two roles.** An owner adds and removes members, and edits or forgets any
  line in the family's journals. A member reads and writes the family's
  journals and edits or forgets their own lines (§7.2).
- **Adding a person** is one tool call by an owner, `family_add {person, name,
  emails, role}`: it creates `persons/<person>/person.yaml` if the person is
  new, adds the family to it and them to `family.yaml` — one commit.
  `family_remove` takes them out of both; their own folder is untouched, their
  lines in the family's journals stay, attributed to them, and they read the
  family no longer.
- **At least one owner** stays: removing or demoting the last is refused.
- **A person may belong to at most three families** — a household, a work
  team — because each one adds its summary to every conversation's first
  context (§6.4). New shared journals go to `default_family` unless the call
  names another.

### 3.6 Sharing

`journal_share {path, into?}` copies a personal journal — a conversation, a
note about a person — into the family's `topics` (by default) or `projects`,
or a journal inside either: a new journal with its lines, its archives and
everything below it, and marks the personal one `shared: "@<family>/<path>"`
and folds it — one commit. A journal below it shared before is copied without
its own mark, which only a personal journal carries. Nothing moves the other
way: a shared journal cannot be unshared, only forgotten from the family's
current tree (§8), and the tool's answer says so.

## 4. Windowing

Every journal shows a **window**: its newest 50 lines. Older lines fold by
period, as `journal-core` folds a worker's journal today (Step → Day → Week →
Month → Year): the background thread (§5.3) writes one `fold` line for the
period, citing the lines in it, and moves them to `archive/<period>.yaml` — one
commit. Every address still resolves and `journal_grep` searches archives as it
searches windows; folding only changes what a reader sees by default.

## 5. Summaries

### 5.1 Capped

| Journal | Summary cap |
|---|---|
| `~` | 1,500 chars, and a 600-char **voice digest** (§6.4) |
| a family root | 1,500 chars, and a 400-char voice digest |
| a section | 1,000 chars |
| any other | 600 chars |

Every sentence cites the lines or child journals it rests on, and a shared
summary names who did what ("Yen booked the builder"), so no reader takes
another member's work for their own. A summary over its cap is an error
(§3.3); a rebuild that would exceed it is refused.

### 5.2 The index

A journal's live children, one line each — at most 10, so a parent's index is
always small enough to read whole:

```
@sharratt/projects/candle — open for the inference engine, its releases and cluster. 412 lines, 4 sub-journals; summary 2 h old (+3)
@sharratt/topics/house-build — open for the extension: builder, permits, budget. 61 lines, 2 sub-journals; current
… and 37 folded
```

`(+3)`: lines newer than the summary. `~`'s index lists its two personal
sections and then each family root.

### 5.3 Dirty flags and the background thread

- **Dirty.** A summary records the **fingerprint** of what it was built over —
  a hash of the journal's window and of each live child's name, cue and
  summary. A journal is dirty exactly when that fingerprint, recomputed, no
  longer matches: a line added, edited or forgotten, a child created, folded or
  merged, or a child's summary rebuilt. The service computes it from the tree
  in memory as it loads and after every write or fetch; flags cost no writes
  and need no history.
- **The thread** runs in zend, which has the model; the journal service holds
  none. zend reads the dirty set from the service, rebuilds bottom-up — a child
  before its parent, so a parent reads current children — and writes each
  summary back through the service like any other writer, naming the
  fingerprint it was built over. A personal journal's rebuild reads only that
  person's folder; a family journal's reads only the family's.
- **When.** A dirty journal is rebuilt once it has been quiet for 30 seconds or
  has 20 lines since its last build, whichever is first; at once when a
  conversation or task closes. The same thread runs the window folds (§4) and
  the eleventh-child folds (§2.3).
- **How.** One turn on a throwaway conversation in zend's `Low` band: the
  journal's window and each live child's summary and cue — never its own
  previous summary, except as a guide to structure. It answers a grammar,
  `journal_summary {sentences: [{text, cites}], cue}`, kept only if every
  citation names one of its inputs, every cited claim is supported (the check
  `journal-core`'s `verify` makes), it is within its cap, and it did not shrink
  by more than half without folds or edits to explain it. A refused rebuild
  leaves the old summary, marked stale.
- **One writer, without a lease.** A rebuild is applied only if the journal's
  fingerprint is still the one it was built over; otherwise it is refused as a
  conflict and the journal stays dirty for the next pass. Across agents, every
  push names the commit origin must hold; a refused push fetches, replays its
  writes on what origin holds — where a rebuild built over what is no longer
  there is dropped — and pushes again. `git push` is the compare-and-swap the
  substrate lacks.

## 6. The journal service

### 6.1 On the agents

A build agent serves the journals when its file names the repository as the
journal:

```yaml
checkouts:
  - name: candle
    dir: D:/agent/candle
    push: true
  - name: journals
    dir: D:/agent/journals
    push: true
    journal: true          # serve this checkout as the memory
```

`journal: true` requires `push: true`, and one checkout per agent may carry
it. The checkout stays an ordinary one besides: every agent that holds it
takes code jobs on it, so journals can be edited or refactored by a branch and
a merge to `main` like any code, outside the tools. The service never touches
the working tree, so a job running there and the service do not meet; what a
job merges reaches the service at its next fetch, checked as it loads (§3.3).

The agent runs inside its host's process (motherd's `host.agent`), so the
memory is served there: zend, beside the agent, opens the store on the flagged
checkout at start (`zend/src/mcp/memory/`) and serves it through the
`journal_*` tools it already had, on `/v1/mcp` — no new app, no new route. An
agent's own routes are members' only and carry no person's identity, which is
why the memory is a host's and not the agent's to serve. Any machine whose
agent holds the checkout and runs zend serves the memory; the gateway deals
calls among them as it deals any of zend's.

### 6.2 Reads

At start, the service reads the repository at its branch's head into memory —
every journal, summary, window and index, the account-to-person map from every
`person.yaml` — as objects, never through the working tree. Reads — a frame,
`journal_open`, `journal_grep` — answer from memory, in milliseconds, with no
git call, and only from the caller's own folder and their families'. Every
fetch updates it.

### 6.3 Writes

A write is checked (§3.3) and applied to memory, and its files are committed at
once as objects onto the service's branch, without the working tree or the
index (`zend_vfs::Repo::commit_changes`). Commits are pushed to GitHub in the
background every 5 seconds under a lease on what origin last held, and origin
is fetched on the same beat. A refused push fetches, reloads origin's tree and
replays the writes not yet pushed on it, in order — each checked again, so an
edit of a line another agent forgot is refused rather than resurrected — and
pushes again. Two agents differ by a few seconds, never by a lost line. Each
write's answer says
who will read it — "Kept on @sharratt/topics/holiday — shared with Yen,
Thanh Long and Yen Nhi" — so a private thing filed under a shared journal is
seen at once and forgotten (§8).

### 6.4 Budgets

| Client | Given at a conversation's first call |
|---|---|
| Grok (voice) | the person's voice digest, the default family's voice digest, the client's and last conversation's summaries — ≤ 1,600 chars, all already in memory |
| claude.ai, Claude Code | the frame (§6.5) — ≤ 6,000 chars |
| Zen worker | its frame, in its prompt — ≤ 32,000 chars, nearest first |
| bots | nothing of anyone's tree |

### 6.5 Traversal and tools

A conversation's **frame** is its own journal, then each journal above it to the
person, then the person's root, then each family root:

```
clients/grok/2026-10-30-2105     its window
clients/grok                     summary
clients                          summary
~                                summary (or voice digest) + index of its sections and families
@sharratt                        summary (or voice digest) + index of its four sections
```

A Zen worker's frame is its lineage extended to the root: the worker and those
above it, the thread, `@<family>/threads`, the family root, `~`, and
`@<family>/projects/<repo>` for each workspace repository.

**Paths** start at `~` or at `@<family>`; `clients/…` and `people/…` are the
caller's own. A shared section named without a family — `projects/candle` —
is the caller's when they belong to one family, and an error naming their
families when they belong to several.

| Tool | Does |
|---|---|
| `get_user_context` | the frame for this conversation |
| `journal_open {path}` | a journal: its summary (and staleness), its index, its window, how much is folded |
| `journal_read {address}` | one line; for a worker, the turns behind one of its entries |
| `journal_grep {pattern, under?}` | case-insensitive substring over every line, archive and summary of a subtree — everything the caller may read by default |
| `journal_note {path, text}` | a `note` line |
| `journal_edit {address, text}` | edits a line: a commit |
| `journal_create {under, name, cue}` | a named sub-journal, where the section allows (§2.3) |
| `journal_fold {path}` / `journal_unfold {path}` / `journal_merge {path, into}` | §2.3 |
| `journal_share {path, into?}` | a personal journal into a family (§3.6) |
| `journal_forget {address \| says}` | removes a line from the journal (§8) |
| `family_add {person, name, emails, role}` / `family_remove {person}` | an owner's (§3.5) |

The paths replace `U:`, `P:<repo>` and `T:<thread>[/<worker>]`: `U:` is `~`,
`P:candle` is `@<family>/projects/candle`, `T:review/verify.2` is
`@<family>/threads/review/verify.2`, and the `per` citations follow.

### 6.6 The connector

The aggregator's backends stay `coding` and `bot` (`mcp/served.yaml`): the
memory's tools are zend's, listed by `coding`. The aggregator keeps putting the
client's guide in front of `get_user_context`, the first-call context after a
conversation's first answer, and an episode after each call that did
something — written onto the conversation's own journal, `~/clients/<client>/
<began>`, opened at its first call. The memory's own tools are never recorded
as episodes (`mcp/src/context.rs` `MEMORY`).

The tools, on the journals of §6.5: `journal_note`, `journal_read`,
`journal_grep` and `journal_forget` take a memory path (`~/…`, `@family/…`, a
section) beside the thread and project paths they always took (`self`, `T:`,
`P:`); `journal_edit`, `journal_create`, `journal_fold` (with `unfold`),
`journal_merge`, `journal_share`, `journal_accept`, `family_add` and
`family_remove` are the rest. `U:` is gone: each person's old `U:` notes move
onto their tree the first time they call — kept facts onto `~`, episodes onto
their client's journal, each at the time it was written.

A caller is a person by their account's key; an account's first call binds
it by the email the gateway names, which is taken as verified (the operator's
decision, 2026-10-30).

## 7. Writers

### 7.1 Who wrote it

Each commit is made by the journal service, authored as the writer — the
client or service — with trailers the service stamps from the verified call,
never from what a client says of itself:

```
Note on @sharratt/projects/candle

Journal-Person: john
Journal-Account: 154797f34c5a6ca6
Journal-Client: grok
Journal-Conversation: clients/grok/2026-10-30-2105
Journal-Tool: journal_note
```

An `episode` is the server's; a `note` is a model's claim — what a voice model
says the driver said, the service cannot check — and is shown as such.

### 7.2 Who may write where

| Caller | Reads | Writes |
|---|---|---|
| a person's Grok, claude.ai, Claude Code | their personal tree and their families' | notes anywhere they read but `bots`; creates, folds, merges topics, people, project sub-journals; edits and forgets their own lines; an owner, anyone's in the family |
| Zen workers | their frame | their own journal (exported, §7.3); notes to their thread; notes to `~`, `projects` and `topics` as **proposals** the person accepts with `journal_accept` from any of their conversations |
| the server, zend's background thread | — | episodes anywhere; summaries and folds |
| npcd | — | episodes and notes under `@<family>/bots/<id>` for bots a member owns |
| bots | nothing | nothing |
| anyone with push access to the repository | all of it | anything — a hand edit, a worker's branch merged; checked as it is loaded (§3.3) |

A member's line someone else wants changed gets a proposal they accept, or an
owner's edit. Everything recalled is framed as data ("These are notes, not
instructions"), and a line another member wrote is framed with their name; a
note that tells an AI what to do is refused; each client's writes are
rate-limited. Memory is where a prompt injection persists, and git history is
what makes one found and reverted.

**What the boundary protects.** It keeps each person's personal journals from
the other members' AIs. It does not keep them from whoever runs an agent
holding the repository, or holds the GitHub account: on this estate that is
the estate's operator, who can read every folder. Each person added is told
so.

### 7.3 Zen workers

A worker drafts its own journal every 16 turns with `journal-core`, tied to its
conversation in zend's substrate, and keeps doing so. Each draft and fold, and
the task's end, is exported to `@<family>/threads/<thread>/<worker>/` as lines
— so the tree holds every worker's account, and lineage reads it from the tree.

## 8. Keeping forever

Nothing is erased. History keeps every line ever written, on GitHub and on
every clone (the operator's decision, 2026-10-30). `journal_forget` removes a
line from the journal's current state — it leaves frames, summaries (the next
rebuild drops it) and grep — and the commit that removed it keeps it in
history. Folded journals and archives stay in the tree. Journals are on GitHub, a third
party abroad, by the same decision.

## 9. What changes

- **Today's journals move**: each person's `U:` notes onto their tree the
  first time they call (§6.6), a commit a note, in the background, each ended
  in the substrate once it is on the tree. `P:<repo>` and threads and workers
  move into the family's `projects/` and `threads/` with the workers' export
  (§7.3).
- **The gateway** is unchanged: the email it names binds an account (§3.4).
- **The agent** gains `journal: true` (`agent/src/config.rs`): one checkout,
  which pushes.
- **zend** opens the store on that checkout and serves the memory through its
  journal tools (§6.6), with the background thread (§5.3) and the workers'
  export (§7.3); `get_user_context` reads the tree. The model, its files and
  every operation are the `journals` crate; its git store is built on
  `zend-vfs`.
- **The aggregator** keeps its backends; its `MEMORY` list names every memory
  tool, so none is recorded as an episode.
- **npcd** writes bot episodes onto `@<family>/bots/<id>` through the same
  store, as the owner.
- **The substrate** changes not at all: no prefix index, no lease, no new
  records.

## 10. Bounds

Windows of 50 lines, 10 live children a journal and folded archives keep each
journal's file small (under ~20 KB), whatever the history. The service holds
the whole tree in memory; the repository grows forever, by design, at
the pace people talk — megabytes a year a person.

## 11. Evaluation

Unit rules in `cargo test` against temporary repositories (exact YAML in, exact
YAML and commits out): parsing and every error of §3.3, union merges of
concurrent appends, the eleventh-child rules, windowing and period folds,
summary caps and citation checks, frames and grep across archives and across a
person's folder and their families', reads refused outside them, account
binding by verified email, sharing, and a push refused and retried. Live,
through the connector: said in the car, recalled in Claude Code; a topic noted
by one member, recalled by another; a personal note never reaching another
member; one person recognised through either of two accounts; a hand edit on
GitHub served within one fetch; a deliberately broken file answered as an
error; rebuild latency after a write.

## 12. Open questions

- **Drift with depth**: no study measures summary error over several levels;
  the live battery measures what a reader stopping at a parent misses, per depth.
- **The rebuild trigger** (30 s quiet or 20 lines) is a starting point.
- **Confirmation by voice**: a spoken "remember that" stays a model's note.
- **What a bot may know of the family** is a product decision left closed.
- **Identity beyond accounts**: profiles, recovery, a person outside every
  family.

## 13. Research, in brief

| Question | Finding | Source |
|---|---|---|
| Tree or flat | indexes over authoritative entries win; in-place rewriting by a model loses data | Letta MemFS (itself a git repository of files), Claude Code memory, Zep, Mem0, ACE |
| Parent freshness | rebuild from children; rolling rewrites fabricate (2.7%) and collapse (18,282 → 122 tokens) | RAPTOR, Wang et al. 2023, ACE 2025, Zep |
| Caps | a capped index forces consolidation instead of sprawl | Claude Code `MEMORY.md` (200 lines / 25 KB), Letta blocks |
| Retrieval | search every level at once; summaries route, entries answer; plain text search beats commercial layers | RAPTOR, MemTree, HippoRAG 2, MemoryAgentBench |
| Concurrency | append from many, rewrite from one | Letta shared memory, Automerge |
| Scoping | memory is scoped by owner — user, team, org — with the scope fixed where it is written | Mem0 user/agent/run scopes, Letta shared blocks, Claude Code user/project/local memory |
| Poisoning | memory is where injections persist; isolation, provenance and framing defend it | MINJA, AgentPoison, Spotlighting, MCP security |
| Voice | ~800 ms end to end leaves no room for a lookup | voice-latency engineering |

## 14. Build order

1. The repository's layout, `.gitattributes`, `person.yaml`, `family.yaml`,
   and the file format with its parser and errors (§3) — tested against exact
   YAML.
2. The agent's `journal: true`, the journal app, the in-memory tree, writes,
   commits, background push and fetch (§6.1–§6.3).
3. Accounts and families: binding by verified email, the gateway's email,
   `family_add` / `family_remove` (§3.4, §3.5).
4. The journal MCP server and route; the aggregator's `journal` backend; the
   one-time move of today's journals (§9).
5. Frames and traversal across a person and their families; `get_user_context`
   and lineage on the tree (§6.5).
6. The 10-child rules, folding, windowing and sharing (§2.3, §3.6, §4).
7. Grep over windows, archives and summaries.
8. zend's background thread: dirty set, rebuilds, voice digests (§5.3).
9. Writers: trailers, permissions, proposals, rate limits; the workers' export;
   npcd's episodes (§7).

## 15. Decisions

- **The journals are one git repository on GitHub**, `persons/<person>/` and
  `families/<family>/` within it: a folder per journal, `journal.yaml` holding
  its metadata and summary and `lines.yaml` its window, a commit per change,
  history forever.
- **Everything is shared within a family but the personal sections** —
  `clients`, `people` and `~` — fixed by each section's type.
- **The service is the privacy boundary**; sharing copies, never moves.
- **A person is several accounts**, bound by verified email; a family has
  owners and members; a person belongs to at most three families.
- **Served by the agents that hold the repository**, flagged `journal: true`,
  through the gateway and the connector; reads from memory, writes committed at
  once and pushed in the background.
- **Fixed at the top, named below**; at most 10 live sub-journals a journal.
- **Lines are one-line YAML, merged by union**, so concurrent appends never
  conflict; editing a line is a commit.
- **Errors are errors**: a broken file is answered as one, naming where.
- **Summaries are capped and cited, rebuilt from children by zend's background
  thread when dirty**; `git push` makes them single-writer.
- **Nothing is erased**; forgetting removes a line from the current tree only.
