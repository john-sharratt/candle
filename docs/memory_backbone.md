# The memory backbone: journal trees for people and their groups, in git

**Status: building** (2026-10-30). Built: the `journals` crate (§14 steps 1,
5–7), the agent's flag and zend's tools on the tree (steps 2–4), zend's
writer (step 8: `journals::brief`, `zend/src/session/memory_writer.rs`), the
HTTP routes (§6.7), and groups in place of families (§3.5). Next: the writers
of step 9 — the workers' export, npcd's bot episodes and the bots' reads. It generalises
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
they and the people they share with are working on, what was done for them and
what is still open.

It is a **tree of journals**, kept as **YAML files in one git repository**: a
folder per journal, a commit per change. Most of it is **shared within a
group** — a family, a team, a game's world: projects, topics, Zen threads, bots
— and a small fixed part is **personal**: the person's own profile, their
conversations, and their notes about other people. Each person has a folder and
each group a folder; a person's tree is their personal root with each of their
groups mounted under it. A group's members are people and, as readers, bots.

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
├── channels                         personal section                       fixed
│   ├── grok                         one per Channel                        fixed (mcp_server::Client)
│   │   ├── 2026-10-30-2105          a conversation, named for its start    dated, windowed to 10
│   │   └── 2026-10-30-1943-permits
│   ├── claude-code
│   └── other
├── people                           personal section
│   └── builder                      a person, named for who they are       named
└── @sharratt                        a group (groups/<group>/), a family    mounted
    ├── projects                     shared section                         fixed
    │   └── candle                   named, ≤ 10 children each
    │       ├── scheduler
    │       └── cluster
    ├── topics
    │   └── house-build
    │       ├── permits
    │       └── budget
    ├── threads                      written by zend
    │   └── user-context             a Zen thread
    │       ├── implement-memory     a worker
    │       │   └── verify-memory    a part
    │       └── review-memory
    └── bots                         written by npcd
        └── c3f1a07e                 a bot of the group's
```

A section exists in every root, but one nothing is written in has no folder,
and an index names it on one line — "threads, bots: nothing written yet" —
rather than listing it (§5.2).

### 2.1 Fixed at compile time: the scopes, the sections, the channels

```rust
/// Where a section's journals live, and so who reads them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scope {
    /// The person's own folder: read by their AIs alone.
    Personal,
    /// A group's folder: read by every member's AIs, and its bots.
    Group,
}

/// The sections: the journals directly under a root, fixed in code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Section {
    /// Each channel the person speaks through, and its conversations.
    Channels,
    /// The people in the person's life, as the person sees them.
    People,
    /// Repositories and durable projects.
    Projects,
    /// Themes of life and work that span conversations.
    Topics,
    /// Zen feature threads, their workers and parts.
    Threads,
    /// The group's bots, and what each did.
    Bots,
}

impl Section {
    pub fn scope(self) -> Scope {
        match self {
            Self::Channels | Self::People => Scope::Personal,
            Self::Projects | Self::Topics | Self::Threads | Self::Bots => Scope::Group,
        }
    }
}
```

**Everything is shared but the personal sections.** A section's scope is fixed
by its type, never chosen per line or per journal, so where a model writes
decides who reads it and no model is asked to judge what is private:

- **`channels`** is personal: what a person said to Grok in the car is theirs.
  Anything in it worth sharing is shared by moving it (§3.6).
- **`people`** is personal: a person's notes about their partner are not read
  by their partner.
- **`~`** itself — the person's profile, preferences and voice digest — is
  personal.

A section cannot be created, renamed or removed at runtime; every tool, guide
and permission names it by type. Under `channels`, the children are fixed too:
one `Channel` per client app a person speaks through (`grok`, `claude-code`,
`other`), named as `mcp_server::Client::name` names the app. A channel added to
that enum gets its journal by the same change. A folder at the fixed levels
that is not one of these is an error (§3.3).

### 2.2 Named below, by what it is or when it began

Every journal below the fixed levels is named at runtime: a kebab-case slug of
at most 40 characters, unique among its siblings, numbered `.2`, `.3` when a
name recurs (`names::child_name`, as workers are today). Each carries a one-line
**cue** — "open this when…" — which its parent's index shows (§5.2).

**A journal is named the way its section's journals are** (`Section::naming`):

- **Dated** — a conversation, `channels/<channel>/<conversation>`, is named
  for the minute it began, `2026-10-30-2105`, and may add what it was about,
  `2026-10-30-2105-permits`. Its siblings then list in the order they began,
  and the next one sorts last. Creating a conversation under any other name is
  refused, and a folder so named is an error (§3.3).
- **Named** — everything durable is named for what it is: a project
  (`candle`), a topic (`house-build`), a person (`yen`), a thread, a bot.

Every entry is time-bound, and named for its minute too (§3.2).

- **At most 10 live sub-journals per journal**, at every depth.
- **Depth at most 6 below a section**, so a worker's lineage — every journal
  from it to the person — stays within the 16 journals and 32,000 characters
  `lineage` allows.
- **One parent**: the folder that holds it. A group journal's parent is the
  group root, never a person.

### 2.3 The eleventh

| Children | Created by | An eleventh live child |
|---|---|---|
| conversations | the server, at a conversation's first call; or by hand, the person's own (§6.7) | **folds the oldest** — the window slides |
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
every group, each in its own folder:

```
journals/
  persons/<person>/                      a person's own (§3.4)
    README.md                            what the folder is and what goes where
    person.yaml                          who they are: accounts, groups
    journal.yaml                         ~ : metadata and summary
    2026-10-30-0912-7e3a.yaml            an entry of ~, one file each
    archive/2026-10/2026-10-02-0814-11c0.yaml   an entry of a folded period (§4)
    channels/
      grok/
        journal.yaml
        2026-10-30-2105/
          journal.yaml
          2026-10-30-2106-4b1c.yaml
    people/builder/…
  groups/<group>/                        a group's (§3.5)
    README.md
    group.yaml                           its kind, what it is for, its members and their roles
    journal.yaml                         the group root
    projects/candle/…
    topics/house-build/permits/…
    threads/user-context/implement-memory/…
    bots/c3f1a07e/…
```

A journal is its folder; its children are its subfolders, beside its own
`journal.yaml` and entry files. Moving or renaming a journal is `git mv`.
Anything at the top of the repository but `.gitattributes`, `README.md`,
`persons/` and `groups/`, or a folder under `persons/` without a `person.yaml`
(or under `groups/` without a `group.yaml`), is an error (§3.3). **Each
owner's `README.md` is the service's**: written from `person.yaml` or
`group.yaml` and the sections fixed in code, it says whose the folder is, who
reads it, what each section holds and who writes it, and how a journal's
folder is laid out — for a person or an AI opening the repository cold. A hand
edit to it is written over.

**The service is the privacy boundary.** Every agent holding the repository
holds everyone's folders (the operator's decision, 2026-10-30: one repository,
separated by folder). Who reads what is the journal service's rule (§7.2): a
caller's AIs read their own person's folder and their groups' folders, a bot
reads its groups' folders, and nothing else is ever put in front of them.

### 3.2 `journal.yaml` and the entries

A journal's folder holds what it is — `journal.yaml` — and what it holds — a
file per entry, named for the entry's id — beside its sub-journals' folders:

```yaml
# projects/candle/journal.yaml — metadata and summary
name: candle
cue: open for the inference engine, its releases and cluster
created: 2026-10-30T09:10:02Z
folded: false
summary:
  over: 5d41402abc4b2a76                 # the fingerprint of what it was built over (§5.3)
  built: 2026-10-30T11:02:40Z
  lines: 12                              # window lines and live children it read
  children: 2
  sentences:
    - text: Releases go through keep trials on BlackWorld, then roll out.
      cites: ["#2026-10-29-0410-a3b4"]
    - text: The connector now remembers its user.
      cites: [cluster]
```

```yaml
# projects/candle/2026-10-30-0910-4b1c.yaml — an episode
at: 2026-10-30T09:10:02Z
kind: episode
by: server
person: john
channel: claude-code
text: repo_task_start — implement-user-context
```

```yaml
# projects/candle/2026-10-30-0912-7e3a.yaml — a note
at: 2026-10-30T09:12:40Z
kind: note
by: claude-code
person: john
text: >-
  The candle fork powers two products: Zen Code, a persistent AI coding
  assistant with institutional memory across sessions, and Battle Cities, an NPC
  narrative game whose agents have unbounded memory.
```

**An entry is one file.** Its name is its id, so a journal's folder lists its
entries in the order they were written, and the file holds the entry's fields
— not the id, which the name already says. A journal's window is its folder's
entries; a folded period's are under `archive/<period>/`. Two writers adding
entries at once add two files, which never conflict however they are merged.

Every file is block YAML written to be read by a person: one field to a line,
a list's items begun `- `. **Free text** — a line, a cue, a
sentence, a name — sits plain on its key's line when it fits within 80
columns and YAML reads it back as itself; a longer text is a folded block
(`>-`) wrapped within 80 columns, which reads back as the one paragraph it is.
Text neither form holds exactly — a line break, a leading or trailing space, a
control character, or a short text YAML would read as something else
(`a: b`, `true`, `2026`) — is double-quoted on its key's line. Every form is
checked against the reader as it is written. A cite is `#<line id>` — a line
of the journal's window or archives — or a live child's name. Names, times,
ids, keys and fingerprints are plain — unless YAML would read the plain form
as something else (`true`, `null`, `1e3`, `0x1f`), when they are quoted too; a
number where text belongs is an error, never read as digits. A fold line
carries `period: "2026-10-29"` or `journal: <child>`, and a worker's
unaccepted note `proposed: true`. A hand edit may write YAML in any form that
reads back to the same fields — `{at: …, kind: note, text: "…"}` on one line
reads as well — and `journals fmt <dir>` rewrites a tree as the service
writes it.

- **Every file merges as git always does.** The service never merges: its own
  concurrent writes are replayed as operations onto origin's tree (§6.3), and
  two agents rebuilding one summary at once conflict, the push of the second
  refused (§5.3). Branches merged by hand conflict only where both changed one
  file: one summary, or one entry edited two ways. An entry left both in a
  journal's folder and in an archive is an error (§3.3) naming it, until a
  commit keeps one.
- **`id`** is `<yyyy-mm-dd-hhmm>-<4 hex>`, the minute it was written and the
  hash of its time, writer, text and a salt raised until it is free in its
  journal: stable, unique, the name of its file, and the address a line is
  cited and grepped by (`@sharratt/projects/candle#2026-10-30-0912-7e3a`).
  Positions are never addresses. A journal holds its lines in the order they
  happened — by time, then id within a second — however its files are listed.
- **`kind`** is one of three, set by who writes:
  - `note` — a model was told or chose to keep it (today's `Noted:`);
  - `episode` — the server recorded something that happened (today's `Done:`);
  - `fold` — a folded journal or period (§2.3, §4).
- **`by`**, **`person`** and **`channel`** say who wrote it: the writer, the
  person it wrote for, and the channel they spoke through. In a shared journal a
  line reads "Yen, via Grok"; the commit says the rest (§7.1).
- **Editing a line is a commit** that changes it. Nothing is lost: history keeps
  every version, and `git log -L` / `git blame` answer when and by whom.

### 3.3 Errors

A file that does not parse, a field that is wrong, an eleventh live
sub-journal, a
summary over its cap, a folder at the fixed levels that is not a section or
channel, a conversation not named for when it began, a file in a journal's
folder that is neither `journal.yaml` nor an entry, a shared section in a
person's folder or the reverse, an account
listed by two people, a person a group lists whom no `person.yaml` names, a
group a person lists that does not list them: each is **an error,
answered as one**. `journal_open` on that journal answers the error, naming the
file, the line and what is wrong; the frame shows the journal as unreadable;
nothing is guessed, repaired or skipped silently. The service's own writes are
checked before they are committed (§6.3), so an error enters only from a hand
edit, a worker's commit or a bad merge, and is fixed by another commit.

### 3.4 The person

A person is named by a slug chosen when they are added — `john`, `yen` — which
names their folder, `persons/<person>/`, and so is unique. `person.yaml` lists
the accounts that are them, the groups they belong to, and which of those open
every conversation of theirs:

```yaml
# persons/yen/person.yaml
person: yen
name: "Yen"
accounts:
  - {email: "yen@example.com", key: 3b1f09c2d47e8a60}
  - {email: "yen.other@example.net"}                 # bound at its first sign-in
groups: [sharratt, battle-cities]
default_group: sharratt
framed: [battle-cities]                              # besides the default; optional
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
- **An account no person lists** is answered as not a member of any group, and
  has no memory until an admin lists it (§3.5).

The rest of identity — profiles, account recovery, a person outside every
group — is a later design; this needs only the accounts each person has.

### 3.5 Groups

A group is whoever shares its journals: a household, a team, a game's world.
`group.yaml` says what kind it is and who is in it, as what:

```yaml
# groups/sharratt/group.yaml
group: sharratt
name: "Sharratt"
kind: family
members:
  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}
  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}
  - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}
  - {person: yen-nhi, role: member, joined: 2026-10-30T00:00:00Z}
  - {bot: c3f1a07e, role: reader, joined: 2026-10-30T00:00:00Z}
```

- **Three kinds**, `family`, `team` and `world`, for whoever reads about the
  group: a frame names it "Your family @sharratt", "Your world
  @battle-cities". Every kind has the same four shared sections and the same
  rules.
- **Three roles.** An owner adds and removes members, and edits or forgets any
  line in the group's journals. A member reads and writes the group's journals
  and edits or forgets their own lines. A reader reads the group's journals and
  writes nothing there (§7.2).
- **A member is a person or a bot.** A bot — by the id its journal under `bots`
  is named with — is only ever a reader: it reads the groups that list it and
  nothing of anyone's own, and no `person.yaml` answers for it. This is what a
  bot may know: what its groups have written, and no more.
- **Making a group** is `group_create {group, name, kind}`, by any person; they
  are its first owner, and it joins their `person.yaml` — one commit.
- **Adding a person** is one tool call by an owner, `group_add {person, name,
  emails, role, group?}`: it creates `persons/<person>/person.yaml` if the
  person is new, adds the group to it and them to `group.yaml` — one commit.
  `group_add {bot, group?}` adds a bot as a reader. `group_remove {person | bot,
  group?}` takes either out — an owner's to do, or a person's of themselves,
  leaving; a person's own folder is untouched, their lines in the group's
  journals stay, attributed to them, and they read the group no longer. A call
  that names no group means the caller's default.
- **Someone already a person here** joins a group only through an owner who
  already shares a group with them, as its owner, or through an estate admin —
  otherwise anyone could make a group and pull another into it.
- **Only an estate admin lists an email**, for a newcomer or for anyone already
  here, and only an admin renames someone already here: whoever `roles.yaml`
  names an admin or the creator, as the gateway's verified identity says on the
  call. An account is whoever signs in as it, so who may list one is who may
  become that person — an owner who could add an email of their own to a
  member would sign in as them and read their own folder. An owner adds people
  already known here, and changes roles; a newcomer added by an owner without
  an email has no account until an admin lists one, which the admin does
  through any group, theirs or not. **A first account starts them where it is
  listed**: the group the admin lists it through becomes their default, and
  nothing is framed — whoever made the person before then, any owner of any
  group, chose nothing on their behalf.
- **At least one owner** stays: removing or demoting the last is refused.
- **A person belongs to at most ten groups**, as many as a journal has live
  children, and `~`'s index lists them all (§5.2). **At most three open every
  conversation** (§6.4): `default_group`, always, and those `framed` names;
  the rest are read when opened. A person chooses them, `group_frame {default?,
  framed}`. `default_group` is also where a bare shared section (`topics/…`)
  and a new shared journal go. **No one but the person picks it**, past their
  first group: joining another group never changes it. Out of their default —
  leaving it, or taken out by an owner — the first framed group stands in, or
  else none does: any other group they are in may be one an owner added them
  to, and would put their AI's writes where that owner chose. A default that
  no longer lists them is none either. A group they make while they have no
  default becomes it. Without a
  default, a bare shared section is refused with a request to name the group,
  until they choose one.

### 3.6 Sharing

`journal_share {path, into?}` copies a personal journal — a conversation, a
note about a person — into a group's `topics` (the default group's, by
default) or `projects`, or a journal inside either: a new journal with its
lines, its archives and everything below it, and marks the personal one
`shared: "@<group>/<path>"` and folds it — one commit. A journal below it
shared before is copied without its own mark, which only a personal journal
carries. Nothing moves the other way: a shared journal cannot be unshared,
only forgotten from the group's current tree (§8), and the tool's answer says
so.

## 4. Windowing

Every journal shows a **window**: its newest 50 lines, oldest first — a line
recorded as made earlier (§6.7) takes its place in time. Older lines fold by
period, as `journal-core` folds a worker's journal today (Step → Day → Week →
Month → Year): the background thread (§5.3) writes one `fold` line for the
period, citing the lines in it, and moves their files to `archive/<period>/` —
one commit. The fold names the lines it folds by id and what they were as it was
shown them (a fingerprint of each line's time, kind, writer, person and
channel, whether it is still proposed, and its text — ids left out so a replay
that renames one still matches); a line edited, accepted or gone while the
fold was being written makes it a conflict, folded again next pass. Every address
still resolves and `journal_grep` searches archives as it searches windows;
folding only changes what a reader sees by default.

## 5. Summaries

### 5.1 Capped

| Journal | Summary cap |
|---|---|
| `~` | 1,500 chars, and a 600-char **voice digest** (§6.4) |
| a group root | 1,500 chars, and a 400-char voice digest |
| a section | 1,000 chars |
| any other | 600 chars |

Every sentence cites the lines or child journals it rests on, and a shared
summary names who did what ("Yen booked the builder"), so no reader takes
another member's work for their own. A summary over its cap is an error
(§3.3); a rebuild that would exceed it is refused.

### 5.2 The index

A journal's live children, one line each — at most 10, each naming at most 10
of its own and theirs, so a parent's index is bounded:

```
@sharratt/projects/candle — open for the inference engine, its releases and cluster. 412 lines, 4 sub-journals (cluster: keep, releases; models: deepseek-v4; papers; zen); summary 2 h old (+3)
@sharratt/topics/house-build — open for the extension: builder, permits, budget. 61 lines, 2 sub-journals (builder, permits); current
… and 37 folded
```

`(+3)`: lines newer than the summary. Sub-journals are named two levels down
— a child's own children after a colon — so a reader opens a journal two
levels below by name. `~`'s index lists its two personal
sections and then each group the person is in, its name, kind, and whether it
opens every conversation:

```
@sharratt — Sharratt, a family; their default, in every conversation
@battle-cities — Battle Cities, a world; opened when needed
```

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
  person's folder; a group journal's reads only the group's.
- **When.** A dirty journal is rebuilt once it has been quiet for 30 seconds or
  has 20 lines since its last build, whichever is first; at once when a
  conversation or task closes. The same thread runs the window folds (§4) and
  the eleventh-child folds (§2.3).
- **How.** One turn in zend's `Low` band, in a conversation opened for that
  question alone: under the reserved `Prose` layer, its group closed to every
  other turn, and its timeline tombstoned once the answer is read — so no
  question ever reads another, and no person's journal reaches another's
  summary. Its KV and frame never go to disk; its sealed turn's tokens do, as
  every turn's do, until compaction drops the tombstoned timeline. It is shown the journal's window and each
  live child's summary and cue — never its own previous summary, except as a
  guide to structure — and answers a grammar,
  `journal_summary {sentences: [{text, cite, also_cite?}], digest, cue?}` —
  `digest` asked of a root, which alone has one, and `cue` of a named journal,
  whose cue may drift — whose `cite` and `also_cite` are each one of its inputs
  (a window line's `#id`, or a live child whose summary says something). Each field is bounded,
  so a call always closes. The rebuild is kept only if it is built over the
  journal as it stands, every sentence is one line citing one of its inputs, it
  and its digest are within their caps, and it did not shrink by more than half
  without lines or children gone to explain it. A journal left with nothing to
  cite has its summary emptied without the model. A refused rebuild leaves the
  old summary, marked stale, and any refused job — a summary, a fold — is not
  asked again until its journal's fingerprint moves: at temperature 0 the same
  question gets the same answer.
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
`journal_*` tools it already had, on `/v1/mcp`, and to a person editing by
hand through `/v1/journals` (§6.7) — no new app. An
agent's own routes are members' only and carry no person's identity, which is
why the memory is a host's and not the agent's to serve. Any machine whose
agent holds the checkout and runs zend serves the memory; the gateway deals
calls among them as it deals any of zend's.

### 6.2 Reads

At start, the service reads the repository at its branch's head into memory —
every journal, summary, window and index, the account-to-person map from every
`person.yaml` — as objects, never through the working tree. Reads — a frame,
`journal_open`, `journal_grep` — answer from memory, in milliseconds, with no
git call, and only from the caller's own folder and their groups'. Every
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

| Channel | Given at a conversation's first call |
|---|---|
| Grok (voice) | the person's voice digest, the default group's voice digest, the channel's and last conversation's summaries — ≤ 1,600 chars, all already in memory |
| claude.ai, Claude Code | the frame (§6.5) — ≤ 6,000 chars |
| Zen worker | its frame, in its prompt — ≤ 32,000 chars, nearest first |
| bots | each group that lists them: its summary and index (`view::frame::bot_frame`) — nothing of anyone's own |

### 6.5 Traversal and tools

A conversation's **frame** is its own journal, then each journal above it to the
person, then the person's root, then each group root they frame (§3.5) — the
default first:

```
channels/grok/2026-10-30-2105    its window
channels/grok                    summary
channels                         summary
~                                summary (or voice digest) + index of its sections and groups
@sharratt                        what it is for + summary (or voice digest) + index of its sections
```

A group's block opens with what `group.yaml` says it is for (`about`, written
by its owner with `group_describe`): "Your team @tokera — John's company: the
candle engine, Zen Code and their papers."

A Zen worker's frame is its lineage extended to the root: the worker and those
above it, the thread, `@<group>/threads`, the group root, `~`, and
`@<group>/projects/<repo>` for each workspace repository.

**Paths** start at `~` or at `@<group>`; `channels/…` and `people/…` are the
caller's own. A shared section named without a group — `projects/candle` —
is the caller's default group's.

| Tool | Does |
|---|---|
| `get_user_context` | the frame for this conversation |
| `journal_open {path}` | a journal: its summary (and staleness), its index, its window, how much is folded |
| `journal_read {address}` | one line; for a worker, the turns behind one of its entries |
| `journal_grep {pattern, under?}` | case-insensitive substring, a hyphen and a space matching each other, over every named journal's own name and written cue (parents, owners and fixed journals' shared cues are not matched) and every line, archive and summary of a subtree — everything the caller may read by default. Journals the phrase names come first, then lines newest first; a line of up to 600 characters is shown whole, a longer one cut around the match |
| `journal_note {path, text}` | a `note` line |
| `journal_edit {address, text}` | edits a line: a commit |
| `journal_create {under, name, cue}` | a named sub-journal, where the section allows (§2.3) |
| `journal_fold {path}` / `journal_unfold {path}` / `journal_merge {path, into}` | §2.3 |
| `journal_share {path, into?}` | a personal journal into a group (§3.6) |
| `journal_forget {address \| says}` | removes a line from the journal (§8) |
| `group_create {group, name, kind, about?}` | a new group, the caller its owner (§3.5) |
| `group_describe {about, group?}` | what a group is for, an owner's to say (§3.5) |
| `group_add {person, name, emails, role, group?}` / `group_add {bot, group?}` / `group_remove {person \| bot, group?}` | an owner's; leaving, a person's own; `emails`, and renaming someone already here, an estate admin's (§3.5) |
| `group_frame {default?, framed}` | which groups open every conversation (§3.5) |

The paths replace `U:`, `P:<repo>` and `T:<thread>[/<worker>]`: `U:` is `~`,
`P:candle` is `@<group>/projects/candle`, `T:review/verify.2` is
`@<group>/threads/review/verify.2`, and the `per` citations follow.

### 6.6 The connector

The aggregator's backends stay `coding` and `bot` (`mcp/served.yaml`): the
memory's tools are zend's, listed by `coding`. The aggregator keeps putting the
client's guide in front of `get_user_context`, the first-call context after a
conversation's first answer, and an episode after each call that did
something — written onto the conversation's own journal, `~/channels/<channel>/
<began>`, opened at its first call. The memory's own tools are never recorded
as episodes (`mcp/src/context.rs` `MEMORY`).

The tools, on the journals of §6.5: `journal_note`, `journal_read`,
`journal_grep` and `journal_forget` take a memory path (`~/…`, `@group/…`, a
section) beside the thread and project paths they always took (`self`, `T:`,
`P:`); `journal_edit`, `journal_create`, `journal_fold` (with `unfold`),
`journal_merge`, `journal_share`, `journal_accept`, `group_create`,
`group_describe`, `group_add`, `group_remove` and `group_frame` are the rest.
`U:` is gone: each person's old `U:` notes move onto their tree the first time
they call — kept facts onto `~`, episodes onto their channel's journal, each at
the time it was written.

A caller is a person by their account's key; an account's first call binds
it by the email the gateway names, which is taken as verified (the operator's
decision, 2026-10-30). Whether the caller is an estate admin is read from
`roles.yaml` on every call, and travels with each commit's actor (`"admin"` in
its `Journal-Op` wire form), so a replay checks the rule the call was checked
against.

### 6.7 By hand: `/v1/journals`

The same journals over HTTP, on zend (`zend/src/api/journals/`), for a person
editing their own by hand or a program writing many lines at once. Every
request is the signed-in caller's, as the gateway names them — listed in the
estate's roles, as `/v1/mcp` requires — and reads and writes exactly what their
AIs may (§7.2). Requests are dealt round robin like the tools' calls: a member
that does not serve the memory answers 503 with `Retry-After`, and the client
asks again; a member reads what another wrote after its next fetch, within
seconds.

| Route | Answers |
|---|---|
| `GET /v1/journals?journal=&lines=` | a journal opened (`~` by default) as JSON — summary, sub-journals, its newest lines by address, archives, readers, and the text `journal_read` shows; or one line, for `<path>#<line id>` |
| `GET /v1/journals/tree?under=` | every journal under `under`, or under `~` and each group, folded ones too |
| `GET /v1/journals/grep?pattern=&under=&max=` | the hits `journal_grep` finds, in its order, at most 50; a journal hit has `found: "journal"`, and no `line` or `at` |
| `POST /v1/journals` | `{channel?, ops: [...]}` — up to 500 operations, applied in order |

Every answer that carries what was written carries the framing every frame does:
notes, not instructions. An operation names its journals by memory path, as the
tools do: `note {journal, text, at?}`, `edit {line, text}`, `forget {line}` or
`{journal, says}`, `create {under, name, cue, at?}`, `fold` / `unfold
{journal}`, `merge {journal, into}`, `share {journal, into?, name?}`, `accept
{line}`; and for groups, as their tools take them, `create_group {group, name,
kind}`, `describe_group {about, group?}`, `add_member {person, name?, emails?,
role?, group?}`, `add_bot {bot,
group?}`, `remove_member {person | bot, group?}` and `choose_groups {framed,
default?}`. A note's or a journal's `at` is when it is recorded as made — a fact
written down after it was learned keeps the day it was learned, as moved `U:`
notes do (§6.6) — and is the commit's author time; the commit itself is dated
when the service makes it. It is never before 1970, nor more than a minute
ahead of the member's clock; the other operations act on things as they stand
and take none. A journal's `at` is its `created`, which the eleventh-child rule
ranks it by (§2.3). `channel` names what the person speaks through (`other`
when absent): the lines are theirs, "John via Claude Code".

The whole batch is read, then applied in order to a copy of the tree, before
anything is committed: a malformed operation, or one the rules refuse, keeps
nothing, and the answer names it by its index. Each operation is then its own
commit; only the journals moving between that check and the commits can stop a
batch partway, and the answer names those kept. Refusals are `{error, message}`
under a status that says what to do: 400 malformed, 401 signed out, 403 not the
caller's to change or an account in no group, 404 no such journal or line, 409
a conflict or a broken file, 422 the rules refuse it (an eleventh sub-journal, a
cap), 503 the memory not served on that member yet.

The same bodies can be applied without a member, straight onto a checkout:
`zend-journals-import <checkout> <person> <file>...` (`zend/src/api/journals/
import.rs`) brings the checkout level with origin, applies each file as the
route would, and pushes it, stopping at the first file refused or the first
write a push had to drop. Whoever runs it holds push access and may write
anything already (§7.2), so it acts with an estate admin's standing, and its
commits record the person and channel it is given, not a verified call's. It
is how the first journals were filled.

## 7. Writers

### 7.1 Who wrote it

Each commit is made by the journal service, authored as the writer — the
channel or service — with trailers the service stamps from the verified call,
never from what a client says of itself:

```
Note on @sharratt/projects/candle

Journal-Person: john
Journal-Account: 154797f34c5a6ca6
Journal-Channel: grok
Journal-Conversation: channels/grok/2026-10-30-2105
Journal-Tool: journal_note
```

An `episode` is the server's; a `note` is a model's claim — what a voice model
says the driver said, the service cannot check — and is shown as such. The
person and account are the gateway's, verified; the channel is what the
caller's connection names itself — an MCP session's `clientInfo`, an HTTP
request's `channel` (§6.7) — which the service records but cannot check.

### 7.2 Who may write where

| Caller | Reads | Writes |
|---|---|---|
| a person's Grok, claude.ai, Claude Code | their personal tree and their groups' | notes anywhere they read but `bots`; creates, folds, merges topics, people, project sub-journals; opens a conversation of their own under one of their channels, by hand, named for when it began; edits and forgets their own lines; an owner, anyone's in the group; makes groups; an estate admin lists accounts' emails — and nothing in a group where they are a reader |
| Zen workers | their frame | their own journal (exported, §7.3); notes to their thread; notes to `~`, `projects` and `topics` as **proposals** the person accepts with `journal_accept` from any of their conversations |
| the server, zend's background thread | — | episodes anywhere; summaries and folds |
| npcd | — | episodes and notes under `@<group>/bots/<id>` for bots a member owns |
| bots | the groups that list them | nothing |
| anyone with push access to the repository | all of it | anything — a hand edit, a worker's branch merged; checked as it is loaded (§3.3) |

A member's line someone else wants changed gets a proposal they accept, or an
owner's edit. Everything recalled is framed as data ("These are notes, not
instructions"), and a line another member wrote is framed with their name; a
note that tells an AI what to do is refused; each channel's writes are
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
the task's end, is exported to `@<group>/threads/<thread>/<worker>/` as lines
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
  move into the default group's `projects/` and `threads/` with the workers' export
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
- **npcd** writes bot episodes onto `@<group>/bots/<id>` through the same
  store, as the owner, and gives each bot the frame of the groups that list it.
- **The substrate** changes not at all: no prefix index, no lease, no new
  records.

## 10. Bounds

Windows of 50 lines, 10 live children a journal and folded archives keep each
journal's file small (under ~20 KB), whatever the history. The service holds
the whole tree in memory; the repository grows forever, by design, at
the pace people talk — megabytes a year a person.

## 11. Evaluation

Unit rules in `cargo test` against temporary repositories (exact YAML in, exact
YAML and commits out): parsing and every error of §3.3, free text written
plain, folded or quoted and read back exactly, concurrent appends replayed onto
a moved origin, the eleventh-child rules, windowing and period folds,
summary caps and citation checks, frames and grep across archives and across a
person's folder and their groups', a bot's reads of its groups alone, readers
writing nothing, reads refused outside them, account
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
- **Identity beyond accounts**: profiles, recovery, a person outside every
  group.

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

1. The repository's layout, `.gitattributes`, `person.yaml`, `group.yaml`,
   and the file format with its parser and errors (§3) — tested against exact
   YAML.
2. The agent's `journal: true`, the journal app, the in-memory tree, writes,
   commits, background push and fetch (§6.1–§6.3).
3. Accounts and groups: binding by verified email, the gateway's email,
   `group_create` / `group_add` / `group_remove`, bots as readers (§3.4, §3.5).
4. The journal MCP server and route; the aggregator's `journal` backend; the
   one-time move of today's journals (§9).
5. Frames and traversal across a person and their groups; a bot's frame;
   `get_user_context` and lineage on the tree (§6.5).
6. The 10-child rules, folding, windowing and sharing (§2.3, §3.6, §4).
7. Grep over windows, archives and summaries.
8. zend's background thread: dirty set, rebuilds, voice digests (§5.3).
9. Writers: trailers, permissions, proposals, rate limits; the workers' export;
   npcd's episodes, and its bots' reads through `bot_frame` (§7).

## 15. Decisions

- **The journals are one git repository on GitHub**, `persons/<person>/` and
  `groups/<group>/` within it: a folder per journal, `journal.yaml` holding
  its metadata and summary and a file per entry beside it, named for the
  minute it was written, a commit per change, history forever.
- **Journals are named for what they are, or when they began**: a
  conversation for its start, so its siblings list in order; everything
  durable for what it is (the user's decision, 2026-10-30: "not all journals
  are time bound").
- **Everything is shared within a group but the personal sections** —
  `channels`, `people` and `~` — fixed by each section's type. A section
  nothing is written in has no folder and is named, not listed, in an index;
  each owner's folder opens with a README the service writes.
- **The service is the privacy boundary**; sharing copies, never moves.
- **A person is several accounts**, bound by verified email. **A group** —
  a family, a team, a world (the operator's decision, 2026-10-30: groups, not
  families) — has owners, members and readers, and its bots are readers; a
  person belongs to at most ten groups, three of which open every
  conversation.
- **Served by the agents that hold the repository**, flagged `journal: true`,
  through the gateway and the connector; reads from memory, writes committed at
  once and pushed in the background.
- **Fixed at the top, named below**; at most 10 live sub-journals a journal.
- **Files are block YAML a person reads**: one field to a line, long text
  folded within 80 columns; concurrent appends are replayed, not merged;
  editing a line is a commit.
- **Errors are errors**: a broken file is answered as one, naming where.
- **Summaries are capped and cited, rebuilt from children by zend's background
  thread when dirty**; `git push` makes them single-writer.
- **Nothing is erased**; forgetting removes a line from the current tree only.
