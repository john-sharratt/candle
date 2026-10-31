# The memory backbone: one master journal per group, in git

**Status: redesigned, for review** (2026-10-31). This revision roots the tree
at a group's **master journal** instead of at each person. People,
conversations and topics are journals beneath it, and a conversation works
out, as it goes, who is speaking and what it is about (the user's decisions,
§15). The `journals` crate, zend's tools, its writer and the HTTP routes
build the person-rooted tree of the previous revision, which this file's
history keeps. §14 is the order they change in.

This document generalises `docs/mcp_user_context.md` (the flat `U:` journal,
`get_user_context`, the first-call context) and the `/mcp` journals of
`docs/claude_zend_gateway.md` §4–§5 (threads, workers, lineage, `P:`). It
also extends the windowing and folding of the journal engine
(`docs/zend_journal.md`) to every journal. Where this document and those
disagree about the memory, this one is authoritative. The research behind it
is summarised in §13.

## 1. What it is for

People on this estate talk to many AIs: Grok in the car, claude.ai, Claude
Code, Zen's coding workers, and the bots of Battle Cities. Each of those AIs
starts every conversation knowing nothing. The backbone is one memory that
all of them read and write, so that a conversation begun anywhere can find
out three things:
- who it is talking to;
- what they are working on;
- what was done for them and what is still open.

The memory is a **tree of journals** for each **group**: the people who share
it, such as a household. A group's journals are kept as **YAML files in one
git repository**, a folder per journal and a commit per change. At the top
of every group is its **master journal**, which holds only the broadest
things the group cares about. `@zen`'s master is Zen itself: what it is,
who is in it, and the few things that last. Beneath it are:
- **people**: one journal for each person the group talks to or about;
- **conversations**: one journal for each conversation, from any client;
- **topics**: everything durable it talks about, broken down as far as it
  needs.

Two groups never mix: another household has its own group, its own master and
its own tree.

A conversation begins at the master, knowing nothing. As it goes, it narrows
down. The model works out who is speaking, which may be someone other than
the signed-in account, and leads to that person's journal. It also works out
what the conversation is about, by walking the topics by their cues. It
records both as the conversation's **focus**, so every later call starts
there. It writes what is worth keeping into the journal it belongs in, and
its own running tasks into the conversation's journal.

The top levels are fixed in code; below them the tree grows by name, with at
most ten live sub-journals to a journal. Every journal has a capped summary,
rebuilt in the background when it is dirty. It also has a window of recent
lines, with everything older folded away. The build agents that hold the
repository serve the memory, through the gateway and the connector.

## 2. The hierarchy

```
@zen                                   the group's master journal (groups/<group>/)   root
├── people                             section                                         fixed
│   ├── john                           a member: who he is, what he wants              named
│   ├── yen
│   └── builder                        someone they talk about
├── conversations                      section                                         fixed
│   ├── 2026-10-31-0712-grok           one conversation: its focus, its tasks          dated
│   └── 2026-10-31-0930-claude-code-permits
├── topics                             section                                         fixed
│   ├── candle                         the inference engine; named, ≤ 10 children each
│   │   ├── provenance
│   │   └── cluster
│   ├── wasmer
│   └── house-extension
│       ├── permits
│       └── budget
└── threads                            written by zend
    └── user-context                   a Zen thread
        └── implement-memory           a worker

@battle-cities                         the game's own group: its world and its NPCs        root
├── topics
│   └── canon
└── bots                               written by npcd
    └── c3f1a07e                       an NPC, which reads what the group grants it
```

Every group has the same five sections. A section nothing is written in has
no folder. An index names it on one line, "threads, bots: nothing written
yet", rather than listing it (§5.2). Battle Cities is its own group (the
user's decision, 2026-10-31): its world and its NPCs are never in Zen's
tree, and an NPC reads only what `@battle-cities` grants it.

### 2.1 Fixed at compile time: the sections and the channels

```rust
/// The sections: the journals directly under a master, fixed in code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Section {
    /// The people the group talks to and about, a journal each.
    People,
    /// Each conversation, from any client, a journal each.
    Conversations,
    /// Everything durable the group talks about: projects, themes, places.
    Topics,
    /// Zen feature threads, their workers and parts.
    Threads,
    /// The group's bots, and what each did.
    Bots,
}
```

**Everything in a group is read by every member** (§3.5). No section is
personal and no line is private. Where a line is written decides what it is
about, never who reads it. Something a person wants kept from the rest of
the group is not written into the memory. A section cannot be created,
renamed or removed at runtime; every tool, guide and permission names it by
type. A folder at the fixed level that is not a section is an error (§3.3).

**Channels are fixed too, but they are not journals.** A channel is the
client app a person speaks through: `grok`, `claude-code`, `claude-ai`, and
`other` for Claude Desktop and the rest, named as `mcp_server::Client::name`
names the app. It is
recorded on every line it wrote (§3.2) and in each conversation's name
(§2.2). **Every AI tied to a group shares its one master.** In `@zen`'s
case that means:
- Grok;
- claude.ai and Claude Code;
- Zen itself: zend's own model writes summaries and folds, and its coding
  workers write their threads (§7.3);
- the bots, each in the part its group grants it.

They share one `conversations` section, and nothing in the tree is filed by
app.

### 2.2 Named below, by what it is or when it began

Every journal below the fixed levels is named at runtime: a kebab-case slug of
at most 40 characters, unique among its siblings, numbered `.2`, `.3` when a
name recurs (`names::child_name`). Each carries a one-line **cue**, "open
this when…", which its parent's index shows (§5.2).

**A journal is named the way its section's journals are** (`Section::naming`):

- **Dated**: a conversation is named for the minute it began and the channel
  it began on, `2026-10-31-0712-grok`. It may add what it turned out to be
  about, `2026-10-31-0930-claude-code-permits`. Siblings then list in the
  order they began, and the next one sorts last. Creating a conversation
  under any other name is refused, and a folder so named is an error (§3.3).
- **Named**: everything durable is named for what it is: a person (`yen`), a
  topic (`house-extension`), a thread, a bot.

Every entry is time-bound, and named for its minute too (§3.2).

- **At most 10 live sub-journals per journal**, at every depth.
- **Depth at most 6 below a section**, so a worker's lineage (every journal
  from it to the master) stays within the 16 journals and 32,000 characters
  that `lineage` allows.
- **One parent**: the folder that holds it.

A **topic may name a repository** with `repo: <name>` in its `journal.yaml`,
and `topics/candle` names `candle`. A Zen worker whose workspace
holds that repository has the topic in its frame (§6.5). The first time a
workspace repository is named that no topic claims, the server creates
`topics/<repo>`.

### 2.3 The eleventh

| Children | Created by | An eleventh live child |
|---|---|---|
| conversations | the server, at a conversation's first call; or by hand (§6.7) | **folds the oldest**: the window slides |
| threads, workers, parts | Zen, at `thread_attach` / `repo_task_start` / `sub_run` | **folds the least recently active finished one**; a running one never |
| bots | npcd, when a member comes to own a bot | folds the least recently active |
| topics, people | the server for a workspace repository (§2.2); otherwise an AI, or a person by hand | **refused**: "candle has 10 sub-journals; restructuring it is a coding task" (§2.6) |
| a member's own `people/<person>` | joining the group (§3.5) | **folds the least recently active journal of someone who is not a member**; refused only when all ten are members' |

**Folding a journal** writes its summary into its parent as a `fold` line
that cites it, and marks it `folded: true`. A folded journal keeps
everything: its folder stays, and it can still be read by path and grepped.
It only leaves its parent's live index ("… and 37 folded"). Only the
service folds a journal, and only a restructuring unfolds one (§2.6).

### 2.4 The focus: how a conversation narrows down

A conversation's journal starts empty. It records what the conversation has
worked out as its **focus**, in its `journal.yaml`:

```yaml
# groups/zen/conversations/2026-10-31-0712-grok/journal.yaml
name: 2026-10-31-0712-grok
cue: open for this drive's conversation
created: 2026-10-31T07:12:04Z
focus:
  speaker: yen
  about:
    - topics/house-extension/permits
```

- **`speaker`** is the people journal of whoever the model has worked out is
  talking. The model works it out from scratch, from what is said. It never
  takes the signed-in account's owner for the speaker: one account in a car
  is spoken to by several people. Until the model knows, there is no
  speaker, and the frame lists every person by cue so it can tell (§6.5).
  When the speaker changes, the focus changes.
- **`about`** is the one to three journals the conversation is about. The
  model finds them by walking the topics by their cues, or by grep, and
  narrows them as the conversation becomes clearer:
  `topics/house-extension` first, then `topics/house-extension/permits`.
  When nothing yet fits, a new topic is created.
- **`journal_focus {speaker?, about?}`** sets either part, from inside the
  conversation; each change is a commit. Every later call's frame starts
  from the focus (§6.5). The focus therefore carries over between calls,
  between clients, and through any model's forgetting.
- The conversation's own lines are what it is doing: its running tasks, what
  was asked and what is still open, and the server's episodes of each tool
  call that did something (§6.6). What it learns belongs elsewhere:
  - about a person, in their people journal;
  - about a subject, in the topic it is about;
  - only the broadest, longest-lived concepts, in the master.

### 2.5 References

A journal may hold **references**: named links to other journals, in the
same group, which behave like symbolic links. They are listed in its
`journal.yaml`:

```yaml
# topics/candle/journal.yaml
refs:
  - name: wasmer
    to: topics/wasmer
    cue: open for the WebAssembly work the engine's sandboxing came from
  - name: john
    to: people/john
    cue: open for who leads the engine's work
```

- **A path through a reference is the target's path.**
  `topics/candle/wasmer/wasix` is `topics/wasmer/wasix`. Opens, reads and
  notes act on the target. Every answer names the target's real path, and a
  line written through a reference is filed, and records that it was filed,
  at the real journal.
- **They exist for decoding** (§6.8). A path is written left to right, so a
  model that has written `topics/candle/` can only go on beneath `candle`.
  A reference gives every journal the exits a reader might want from there:
  the work it grew from, the person who leads it. So a model that took one
  turn is never stuck down a path it did not mean.
- **One hop, inside the group.** A reference points at a journal, never at
  another reference, so a path resolves in at most one hop per reference
  and no chain or cycle can form. A reference out of the group, or to
  nothing, is an error (§3.3).
- **A reference is not a child.**
  - It does not count toward a journal's ten. A journal holds at most ten
    references.
  - An index lists them after the children, as `→ wasmer (topics/wasmer) —
    open for …`.
  - Grep finds each line once, at its real path, and does not walk
    references.
  - A summary does not read through them, and a change to a reference
    leaves the summary clean.
- **Made by a tool, changed by coding.** `journal_link {under, name, to,
  cue}` adds one (§2.6). Only a restructuring removes or retargets one.

### 2.6 The tree grows by tools and is restructured by coding

The tools **append**:
- lines, with `journal_note`;
- sub-journals, with `journal_create`;
- references, with `journal_link`;
- a conversation's focus, with `journal_focus`;
- corrections to what a line says, with `journal_edit` and
  `journal_forget`.

No tool renames, moves, merges, splits or deletes a journal, retargets a
reference, or moves a line from one journal to another.

**Restructuring the tree is a coding activity, done by Zen itself.** It is a
task on the `journals` checkout (§6.1), started like any other coding task.
The task:
- edits the files with `git mv`, so every id, cite and line of history
  survives;
- leaves a reference at each old path, `name` → the new path, so every
  address and cue still leads somewhere;
- runs `journals check` as its verification;
- merges to `main`.

The service serves the restructured tree at its next fetch, checked as it
loads (§3.3). A line filed in the wrong journal is put right the same way:
its file moved by a restructuring, or else forgotten and noted again where
it belongs.

What a tool cannot do, it says so and names the way:
- **An eleventh topic** is refused with "candle has 10 sub-journals;
  restructuring it is a coding task on the journals repository".
- **A model that finds the tree wrong** for what it is doing notes so on
  the journal it found wrong, and asks the person for that task.

The service's upkeep is automatic and stays the service's: the window folds
(§4) and the eleventh-child folds of conversations, threads and bots
(§2.3). Neither is a choice anyone makes.

## 3. Files

### 3.1 The repository

One private GitHub repository, `john-sharratt/journals`, holds every group
and every person's identity:

```
journals/
  persons/<person>/
    person.yaml                          who they are: accounts, groups (§3.4) — no journals
  groups/<group>/
    README.md                            what the group is and what goes where
    group.yaml                           what it is for, its members and their roles (§3.5)
    journal.yaml                         the master
    2026-10-31-0712-7e3a.yaml            an entry of the master, one file each
    archive/2026-10/2026-10-02-0814-11c0.yaml   an entry of a folded period (§4)
    people/yen/…
    conversations/2026-10-31-0712-grok/
      journal.yaml                       its metadata, focus and summary
      2026-10-31-0713-4b1c.yaml
    topics/house-extension/permits/…
    threads/user-context/implement-memory/…
    bots/c3f1a07e/…
```

A journal is its folder; its children are its subfolders, beside its own
`journal.yaml` and entry files. Moving or renaming a journal is `git mv`.

The following are errors (§3.3):
- anything at the top of the repository but `.gitattributes`, `README.md`,
  `persons/` and `groups/`;
- a folder under `persons/` holding anything but `person.yaml`;
- a folder under `groups/` without a `group.yaml`.

**Each group's `README.md` is the service's.** It is written from
`group.yaml` and the sections fixed in code. It says:
- whose group it is and who reads it;
- what each section holds and who writes it;
- how a journal's folder is laid out.

It is for a person or an AI opening the repository cold. A hand edit to it
is written over.

**The group is the privacy boundary.** Every agent holding the repository
holds every group's folder (the operator's decision, 2026-10-30: one
repository, separated by folder). Who reads what is the journal service's
rule (§7.2): a caller's AIs read the groups their person belongs to, and a
bot reads the parts of its group that it is granted. Nothing else is ever
put in front of them.

### 3.2 `journal.yaml` and the entries

A journal's folder holds what it is (`journal.yaml`) and what it holds (a
file per entry, named for the entry's id), beside its sub-journals' folders:

```yaml
# topics/candle/journal.yaml — metadata and summary
name: candle
cue: open for the inference engine, its releases and cluster
created: 2026-10-30T09:10:02Z
repo: candle
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
# conversations/2026-10-31-0712-grok/2026-10-31-0713-4b1c.yaml — an episode
at: 2026-10-31T07:13:02Z
kind: episode
by: server
account: john
channel: grok
text: image_create — a cat on a skateboard
```

```yaml
# people/yen/2026-10-31-0720-7e3a.yaml — a note
at: 2026-10-31T07:20:40Z
kind: note
by: grok
account: john
speaker: yen
text: Yen prefers the council's online lodgement to the counter.
```

**An entry is one file.** Its name is its id, so a journal's folder lists its
entries in the order they were written. The file holds the entry's fields,
but not the id, which the name already says. A journal's window is its
folder's entries; a folded period's are under `archive/<period>/`. Two
writers adding entries at once add two files, which never conflict however
they are merged.

Every file is block YAML written to be read by a person: one field to a
line, with a list's items begun `- `.

**Free text** (a line, a cue, a sentence, a name) is written in one of three
forms, each checked against the reader as it is written:
- **Plain** on its key's line, when it fits within 80 columns and YAML reads
  it back as itself.
- **Folded** (`>-`) and wrapped within 80 columns, for a longer text, which
  reads back as the one paragraph it is.
- **Double-quoted** on its key's line, for text neither form holds exactly:
  a line break, a leading or trailing space, a control character, or a
  short text YAML would read as something else (`a: b`, `true`, `2026`).

A cite is `#<line id>` (a line of the journal's window or archives) or a
live child's name.

Names, times, ids, keys and fingerprints are written plain, unless YAML
would read the plain form as something else (`true`, `null`, `1e3`,
`0x1f`); then they are quoted too. A number where text belongs is an error,
never read as digits. A fold line carries `period: "2026-10-29"` or
`journal: <child>`, and a worker's unaccepted note carries
`proposed: true`. A hand edit may write YAML in any form that reads back to
the same fields: `{at: …, kind: note, text: "…"}` on one line reads as well.
`journals fmt <dir>` rewrites a tree as the service writes it.

- **Every file merges as git always does.** The service never merges. Its
  own concurrent writes are replayed as operations onto origin's tree
  (§6.3). Two agents rebuilding one summary at once conflict, and the
  second's push is refused (§5.3). Branches merged by hand conflict only
  where both changed one file: one summary, or one entry edited two ways. An
  entry left both in a journal's folder and in an archive is an error (§3.3)
  naming it, until a commit keeps one.
- **`id`** is `<yyyy-mm-dd-hhmm>-<4 hex>`: the minute it was written, then a
  hash of its time, writer, text and a salt, raised until the id is free in
  its journal. The id is stable and unique. It is the name of the entry's
  file, and the address a line is cited and grepped by
  (`@zen/topics/candle#2026-10-30-0912-7e3a`). Positions are
  never addresses. A journal holds its lines in the order they happened, by
  time and then by id within a second, however its files are listed.
- **`kind`** is one of three, set by who writes:
  - `note`: a model was told or chose to keep it;
  - `episode`: the server recorded something that happened;
  - `fold`: a folded journal or period (§2.3, §4).
- **Who wrote it** is four fields:
  - `by`: the writer, either a channel or the server.
  - `account`: the person whose sign-in made the call, as the gateway
    verified it.
  - `speaker`: who the conversation's focus said was talking when the line
    was written. It is the model's judgement and is absent until the model
    has one.
  - `channel`: the client app, on an episode.

  A line reads "Yen, via Grok on John's account". When the speaker is the
  account's own person, it reads "John, via Claude Code". The commit says the
  rest (§7.1).
- **Editing a line is a commit** that changes it. Nothing is lost: history keeps
  every version, and `git log -L` / `git blame` answer when and by whom.
- **A line stays in the journal it was written to.** Moving one is a
  restructuring (§2.6): `git mv` of its file, which keeps its id.

### 3.3 Errors

Each of the following is **an error, answered as one**:
- a file that does not parse, or a field that is wrong;
- an eleventh live sub-journal;
- a summary over its cap;
- a folder at the fixed level that is not a section;
- a conversation not named for when it began and its channel;
- a file in a journal's folder that is neither `journal.yaml` nor an entry;
- a focus naming a journal that is not there;
- a reference to nothing, to another reference, or out of its group, and a
  journal's eleventh reference;
- an account listed by two people;
- a person a group lists whom no `person.yaml` names;
- a group a person lists that does not list them.

`journal_open` on that journal answers the error, naming the file, the line
and what is wrong, and the frame shows the journal as unreadable. Nothing is
guessed, repaired or skipped silently. The service's own writes are checked
before they are committed (§6.3), so an error enters only from a hand edit,
a worker's commit or a bad merge, and is fixed by another commit.

### 3.4 The person

A person is named by a slug chosen when they are added, such as `john` or
`yen`. The slug names their folder, `persons/<person>/`, and so is unique.
`person.yaml` is identity alone. It lists the accounts that are them and the
groups they belong to:

```yaml
# persons/yen/person.yaml
person: yen
name: "Yen"
accounts:
  - {email: "yen@example.com", key: 3b1f09c2d47e8a60}
  - {email: "yen.other@example.net"}                 # bound at its first sign-in
groups: [zen]
default_group: zen
```

- **An account** is one sign-in at the gateway, known by zend's owner key:
  `sha256(provider \0 sub)` as 16 hex, the key the gateway's verified
  identity already gives every call. Several accounts may be one person; an
  account is one person's at most.
- **Binding.** An account is listed by its email before it has ever signed in.
  At its first call, the gateway's verified identity gives the key and the
  provider's *verified* email. The service finds the one `person.yaml` that
  lists that email with no key, writes the key in (a commit), and from then
  on knows the account by its key alone. An unverified email binds nothing.
- **An account no person lists is a guest.** It reads `@guest`, and nothing
  else, until an admin lists it (§3.6).
- **An account says which group, not who is speaking.** A conversation opens
  in the account's person's `default_group` (§6.6). Who is talking is the
  focus's to find (§2.4), whoever signed in.

A member's journal about themselves lives in their group, at
`people/<person>`. A person in two groups has a people journal in each, each
group's own.

**Zen is a member too**, an identity like the people in the group (the
user's decision, 2026-10-31). `persons/zen/person.yaml` names it and lists
no accounts, since nobody signs in as Zen. `@zen` lists it as a member, and
its `people/zen` journal holds what Zen is and what it keeps about itself:
- its models and the machines it runs on;
- what it does well and what it has learned about how this group wants
  things done.

Whatever Zen itself writes is written as Zen: its workers' lines, its
summaries and folds, its own conversations. Such a line reads "Zen". The
word names two things on purpose: `@zen` is the group and its master
journal, and `people/zen` is Zen the member, which is one of the group.

### 3.5 Groups

A group is the people who share one tree: a household, a team, a game's
world. A group ties together everything they talk about: their work, their
house. Another household is another group, and the two never mix.
`group.yaml` says what the group is for and who is in it, as what:

```yaml
# groups/zen/group.yaml
group: zen
name: "Zen"
about: "John and Yen's household and Tokera's work: the engine, Zen Code, the house, the kids."
members:
  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}
  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}
  - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}
  - {person: yen-nhi, role: member, joined: 2026-10-30T00:00:00Z}
  - {person: zen, role: member, joined: 2026-10-31T00:00:00Z}
```

```yaml
# groups/battle-cities/group.yaml
group: battle-cities
name: "Battle Cities"
about: "The game's world: its canon, its economy, its NPCs."
members:
  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}
  - {bot: c3f1a07e, role: reader, reads: [topics/canon], joined: 2026-10-30T00:00:00Z}
```

- **Three roles.**
  - An **owner** adds and removes members, and edits or forgets any line.
  - A **member** reads all of the group and writes in it; they edit or
    forget the lines their own account wrote.
  - A **reader** reads what it is granted and writes nothing (§7.2).
- **Every member reads everything** in the group: the master, every person's
  journal, every conversation, every topic (the user's decision,
  2026-10-31).
- **A bot is only ever a reader, of the subtrees `reads` grants it.** The
  bot is named by the id its journal under `bots` has. A Battle Cities NPC
  reads the canon `@battle-cities` grants it, and nothing of any other
  group. No `person.yaml` answers for a bot.
- **Making a group** is `group_create {group, name, about}`, by any person.
  They are its first owner, and it joins their `person.yaml`. This is one
  commit.
- **Adding a person** is one tool call by an owner, `group_add {person,
  name, emails, role, group?}`. It creates `persons/<person>/person.yaml` if
  the person is new, adds the group to it, adds them to `group.yaml`, and
  opens their `people/<person>` journal. This is one commit. A full
  `people` makes room as §2.3 says, so a member always has their journal.
  - `group_add {bot, reads, group?}` adds a bot as a reader.
  - `group_remove {person | bot, group?}` takes either out. It is an owner's
    to do, or a person's of themselves, leaving. Their lines stay,
    attributed to them, and they read the group no longer.
  - A call that names no group means the caller's default.
- **Someone already a person here** joins a group only through an owner who
  already shares a group with them, or through an estate admin. Otherwise
  anyone could make a group and pull another person into it.
- **Only an estate admin lists an email**, whether for a newcomer or for
  someone already here, and only an admin renames someone already here. An
  admin is whoever `roles.yaml` names an admin, or the creator, as the
  gateway's verified identity says on the call. The reason: an account is
  whoever signs in as it, so whoever may list one may become that person. An
  owner who could add an email of their own to a member would sign in as
  them.
  - An owner adds people already known here, and changes roles.
  - A newcomer added by an owner without an email has no account until an
    admin lists one, which the admin may do through any group.
  - A first account starts the person in the group the admin lists it
    through, which becomes their default.
- **At least one owner** stays: removing or demoting the last is refused.
- **A person belongs to at most ten groups.** They choose their
  `default_group` with `group_default {group}`; nobody else does. Joining
  another group never changes it.
  - If they leave their default, or an owner takes them out of it, they have
    no default.
  - Without a default, a conversation opens in their one group if they have
    exactly one, and otherwise asks which to use.
  - A group they make while they have no default becomes it.

### 3.6 `@guest`

`@guest` is the one group that nobody is a member of. Every account that no
`person.yaml` lists reads it: anyone who signs in at the gateway and has not
been mapped to a person.

- **It is read-only to them.** A guest's AI opens, reads and greps `@guest`,
  and writes nothing anywhere:
  - no notes;
  - no focus;
  - no conversation journal, and no episodes, so an unmapped account leaves
    no trace in the memory.

  Its frame is `@guest`'s master and the cues below it. A guest's
  conversation keeps its own running state in its own context.
- **Its owners write it.** `group.yaml` lists owners as for any group (an
  estate admin, to begin with), and only they write it, by the tools or by
  hand. It has `people` and `topics` like any group, and no
  `conversations`, `threads` or `bots` folders, since nothing writes there.
- **It is public in effect.** Anyone with a Google account can sign in, so
  `@guest` holds only what its owners would publish: what Tokera and Zen
  are, the paper, how to get access. Nothing in `@zen` reaches it except by
  an owner writing it there.
- **Being listed moves the account across.** Once an admin lists the email
  under a person, the account's next call is that person's, in their
  default group, and `@guest` is no longer in its frame.

## 4. Windowing

Every journal shows a **window**: its newest 50 lines, oldest first. A line
recorded as made earlier (§6.7) takes its place in time.

Older lines fold by period, as `journal-core` folds a worker's journal today:
Step → Day → Week → Month → Year. The background thread (§5.3) writes one
`fold` line for the period, citing the lines in it, and moves their files to
`archive/<period>/`, all in one commit. The fold names the lines it folds by
id, and records what they were as it was shown them: a fingerprint of each
line's time, kind and who-wrote-it fields, whether it is still proposed, and
its text. Ids are left out of the fingerprint, so a replay that renames one
still matches. A line edited, accepted, moved or gone while the fold was
being written makes it a conflict, and the period is folded again next pass.

Every address still resolves, and `journal_grep` searches archives as it
searches windows. Folding only changes what a reader sees by default.

## 5. Summaries

### 5.1 Capped

| Journal | Summary cap |
|---|---|
| the master | 1,500 chars, and a 400-char **voice digest** (§6.4) |
| a member's people journal | 1,200 chars, and a 400-char voice digest |
| a section | 1,000 chars |
| a conversation | 600 chars: read aloud as "last time" in a voice frame (§6.4) |
| any other named journal (a topic, a person not in the group, a thread or bot journal) | 1,200 chars |

A summary covers the whole journal. It has a sentence for each subject the
journal holds, broadest first, each naming its subject, so a reader can tell
from the summary alone whether to open the journal. One line's detail never
stands in for the rest.

Every sentence cites the lines or child journals it rests on. A summary
names who did what ("Yen booked the builder"), so no reader takes another
member's work for their own.

**The master's summary says only what lasts.** It covers what the group is,
who is in it and the handful of things the whole of its life turns on, so
it routes a new conversation. It never stands in for the topics.

A summary over its cap is an error (§3.3), and a rebuild that would exceed
the cap is refused. **The writer's grammar holds the cap, so a rebuild is
not refused for its length.** The cap is shared out as at most
`cap / 200` sentences (never fewer than the floor, never more than 12),
each bounded at `cap / (sentences × 5)` tokens, and a digest at
`digest cap / 5` tokens. Five characters a token is high for English, so an
answer the grammar lets close fits its cap. Without that bound the writer
wrote `@battle-cities` at 3,478 characters against its 1,500, and was
refused on every attempt.

### 5.2 The index

A journal's index lists its live children, one line each: at most 10, each
naming at most 10 of its own and theirs, so a parent's index is bounded:

```
@zen/topics/candle — open for the inference engine, its releases and cluster. 412 lines, 4 sub-journals (cluster: keep, releases; models: deepseek-v4; papers; zen); summary 2 h old (+3)
@zen/topics/house-extension — open for the extension: builder, permits, budget. 61 lines, 2 sub-journals (budget, permits); current
… and 37 folded
```

`(+3)` counts the lines newer than the summary. Sub-journals are named two
levels down, with a child's own children after a colon, so a reader can open
a journal two levels below by name. The master's index lists its sections.
The `people` line names each person and says which of them are members, so
a conversation working out its speaker sees who it could be:

```
@zen/people — open for the people the group talks to and about. 4 sub-journals (john, member; yen, member; thanhlong, member; builder); current
```

### 5.3 Dirty flags and the background thread

- **Dirty.** A summary records the **fingerprint** of what it was built
  over: a hash of the journal's window and of each live child's name, cue
  and summary. A journal is dirty exactly when that fingerprint, recomputed,
  no longer matches. That happens when a line is added, edited, moved or
  forgotten; when a child is created, folded or merged; or when a child's
  summary is rebuilt. The service computes the fingerprint from the tree in
  memory as it loads and after every write or fetch, so flags cost no writes
  and need no history. A conversation's focus is not an input: a change of
  focus changes no summary.
- **The thread** runs in zend, which has the model; the journal service
  holds none. zend reads the dirty set from the service and rebuilds
  bottom-up, a child before its parent, so a parent reads current children.
  It writes each summary back through the service like any other writer,
  naming the fingerprint it was built over. A rebuild reads only its own
  group.
- **When.** A dirty journal is rebuilt once what its summary reads has been
  quiet — no new window line, and no live child's summary rebuilt — for 30
  seconds, or for 10 minutes when it has live children; or once it has 20
  lines since its last build, whichever is first. A parent waits for its
  children to settle because every child's rebuild dirties it: timed by its
  own window alone, a master with almost no lines was rebuilt the moment any
  child was — `@zen` nine times in eighty minutes on 2026-10-31, each a new
  `journal.yaml` for zend's code reading to read and calibrate. A
  conversation or task that closes is rebuilt at once. The same thread runs
  the window folds (§4) and the eleventh-child folds (§2.3).
- **How.** Each rebuild is one turn in zend's `Low` band, in a conversation
  opened for that question alone. It runs under the reserved `Prose` layer,
  with its group closed to every other turn, and its timeline is tombstoned
  once the answer is read, so no question ever reads another. Its KV and
  frame never go to disk. Its sealed turn's tokens do, as every turn's do,
  until compaction drops the tombstoned timeline.
- **What it is shown.** The journal's window, and each live child's summary
  and cue. It never sees its own previous summary's text, because a model
  shown it copies its shape: a one-subject summary came back as one subject,
  however many the journal held. It is told only the length the shrink rule
  below holds it to.
- **What it answers.** A grammar, `journal_summary {sentences: [{text, cite,
  also_cite?}], digest, cue?}`.
  - `digest` is asked of the master and of a member's people journal, the
    only journals with one.
  - `cue` is asked of a named journal, whose cue may drift.
  - `cite` and `also_cite` are each one of its inputs: a window line's
    `#id`, or a live child whose summary says something.
  - Each field is bounded, so a call always closes.
  - `sentences` holds at least one for every two inputs, up to three, since
    a sentence carries only two cites and a list decoded greedily closes
    after its first subject. A conversation is the exception: it is one
    exchange and may say it in one.
  - The question states that floor, and the length the shrink rule holds a
    rebuild to.
  - A bracketed list of `#id`s written into a sentence's text is dropped
    from it as the answer is read: the cite fields carry what it rests on,
    checked.
- **When a rebuild is kept.** Only if all of these hold:
  - it is built over the journal as it stands;
  - every sentence is one line citing one of its inputs;
  - there are as many sentences as the floor asks for;
  - it and its digest are within their caps;
  - it did not shrink by more than half without lines or children gone to
    explain it: fewer of either, or an input the old summary cited no
    longer there.
- **When it is not.** A journal left with nothing to cite has its summary
  emptied without the model. A refused rebuild leaves the old summary
  standing, marked stale, and the commit that marks it names the refusal. A
  refused job (a summary or a fold) is not asked again until its journal's
  fingerprint moves: at temperature 0 the same question gets the same
  answer.
- **One writer, without a lease.** A rebuild is applied only if the journal's
  fingerprint is still the one it was built over. Otherwise it is refused as
  a conflict, and the journal stays dirty for the next pass. Across agents,
  every push names the commit origin must hold. A refused push fetches,
  replays its writes on what origin holds (dropping a rebuild built over
  what is no longer there), and pushes again. `git push` is the
  compare-and-swap the substrate lacks.

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
it. Otherwise the checkout stays an ordinary one: every agent that holds it
takes code jobs on it, so journals can be edited or refactored by a branch
and a merge to `main` like any code, outside the tools. The service never
touches the working tree, so a job running there and the service do not
meet. What a job merges reaches the service at its next fetch, checked as it
loads (§3.3).

The agent runs inside its host's process (motherd's `host.agent`), so the
memory is served there. zend, beside the agent, opens the store on the
flagged checkout at start (`zend/src/mcp/memory/`). It serves the memory
through its `journal_*` tools on `/v1/mcp`, and through `/v1/journals` to a
person editing by hand (§6.7); there is no new app. The agent's own routes
are for members only and carry no person's identity, which is why the
memory is served by a host and not by the agent. Any machine whose agent
holds the checkout and runs zend serves the memory, and the gateway deals
calls among them as it deals any of zend's.

### 6.2 Reads

At start, the service reads the repository at its branch's head into
memory. It loads every journal, summary, window, index and focus, and builds
the account-to-person map from every `person.yaml`. It reads them as
objects, never through the working tree.

Reads (a frame, `journal_open`, `journal_grep`) answer from memory, in
milliseconds, with no git call, and only from the caller's groups. Every
fetch updates the in-memory tree.

### 6.3 Writes

A write is checked (§3.3) and applied to memory. Its files are committed at
once as objects onto the service's branch, without the working tree or the
index (`zend_vfs::Repo::commit_changes`). Commits are pushed to GitHub in
the background every 5 seconds, under a lease on what origin last held, and
origin is fetched on the same beat.

A refused push fetches, reloads origin's tree and replays the writes not yet
pushed on it, in order. Each write is checked again, so an edit of a line
another agent forgot is refused rather than resurrected. Then it pushes
again. Two agents differ by a few seconds, never by a lost line.

Each write's answer says where the line went and who reads it: "Kept on
@zen/people/yen — read by everyone in Zen". A thing someone did
not mean the whole group to read is then seen at once, and forgotten (§8).

### 6.4 Budgets

| Channel | Given at a conversation's first call |
|---|---|
| Grok (voice) | the primed first call (§6.6): ≤ 1,600 chars, all already in memory |
| claude.ai, Claude Code | the frame (§6.5): ≤ 6,000 chars |
| Zen worker | its frame, in its prompt: ≤ 32,000 chars, nearest first |
| bots | the subtrees their group grants them: each one's summary and index (`view::frame::bot_frame`) |
| a guest | `@guest`'s master and the cues below it, within its channel's budget above |

### 6.5 Traversal and tools

A conversation's **frame** is built afresh at every call, from its focus.
The blocks come in this order; when the budget runs out, a block is cut and
nothing after it is added:

```
@zen                                   about + master summary + index of its sections, people by name and membership
conversations/2026-10-31-0712-grok     its focus, then its window
people/yen                             the speaker's summary and newest lines             (once the focus has a speaker)
topics/house-extension/permits         each journal it is about: summary, index, newest lines   (once it has any)
conversations (latest on this channel) the summary of the conversation before this one, here
Below @zen, by cue:                    each topic and person, and the journals under each, by cue
```

- **At the first call** the focus is empty, so the frame is the master, the
  empty conversation, the people by cue, the conversation before this one on
  the same channel, and the cues of the topics. That is enough to work out
  who is speaking and what about, and to go there.
- **The guide** tells every client how to work with the journals (§6.9).
- **The last block** lists, in whatever budget is left, the cue of each
  named journal under `topics` and `people` and of each journal directly
  under it. A reader then picks a journal two levels down without opening
  the one above it. Conversations are not walked: one is found by when it
  began, or by grep.

A Zen worker's frame is its lineage extended to the master: the worker and
those above it, the thread, `threads`, and the master. It adds the topic
naming each workspace repository (§2.2).

**Paths** start at `@<group>`, or are bare, meaning the conversation's
group: `topics/candle`, `people/yen`. `here` is the conversation's
own journal. A path may pass through references (§2.5).

| Tool | Does |
|---|---|
| `get_user_context` | the frame for this conversation |
| `journal_focus {speaker?, about?}` | sets who is speaking and what the conversation is about (§2.4) |
| `journal_read {journal}` | a journal: its summary (and staleness), its references, its index, its window, how much is folded; with `<path>#<line id>`, one line. The same tool reads Zen's thread and worker journals, and for a worker the turns behind one of its entries |
| `journal_grep {journal, pattern}` | case-insensitive substring search, with a hyphen and a space matching each other, under `journal` (`*` for everything the caller reads). It searches every named journal's own name and written cue (parents, owners and fixed journals' shared cues are not matched), and every line, archive and summary of the subtree. Journals the phrase names come first, then lines, newest first. A line of up to 1,000 characters is shown whole; a longer one is cut around the match |
| `journal_note {journal, text}` | a `note` line, stamped with the focus's speaker; refused, naming the line, when the window already holds a note of the same words (spacing and case aside), so a refined fact is edited rather than noted again |
| `journal_edit {journal, text}` | corrects what a line says, `journal` its address: a commit |
| `journal_forget {journal, says?}` | removes a line from the journal, named by its address or by `says` (§8) |
| `journal_create {under, name, cue}` | a named sub-journal, where the section allows (§2.3) |
| `journal_link {under, name, to, cue}` | a reference from one journal to another (§2.5) |
| `journal_accept {journal}` | a member keeps a worker's proposal, `journal` its address (§7.2) |
| `group_create {group, name, about}` | a new group, the caller its owner (§3.5) |
| `group_describe {about, group?}` | what a group is for, an owner's to say (§3.5) |
| `group_add {person, name, emails, role, group?}` / `group_add {bot, reads, group?}` / `group_remove {person \| bot, group?}` | an owner's; leaving, a person's own; `emails`, and renaming someone already here, an estate admin's (§3.5) |
| `group_default {group}` | which group a person's conversations open in (§3.5) |

Nothing else changes the tree's structure: a restructuring is a coding task
(§2.6).

The paths replace `U:`, `P:<repo>` and `T:<thread>[/<worker>]`:
- `U:` is the speaker's `people/<person>`.
- `P:candle` is the topic that names `candle`.
- `T:review/verify.2` is `threads/review/verify.2`.

The `per` citations follow the same mapping.

### 6.6 The connector

The aggregator's backends stay `coding` and `bot` (`mcp/served.yaml`). The
memory's tools are zend's, listed by `coding`.

At a conversation's first call, the server opens its journal,
`conversations/<began>-<channel>`, in the account's group (§3.4). A
conversation is an MCP session's run of calls with no 30-minute gap. Grok is
the exception: it opens a session for every call it makes, sixteen
conversations in six minutes on 2026-10-31. So a Grok conversation is
followed by the person and the group, not the session. Two people driving
on one account at once share a conversation, and the focus tells their lines
apart. The aggregator then does three things:
- it puts the client's guide in front of `get_user_context`;
- it adds the first-call context after the conversation's first answer;
- after each call that did something, it writes an episode onto the
  conversation's journal.

The memory's own tools are never recorded as episodes (`mcp/src/context.rs`
`MEMORY`).

**Grok's first call is primed.** Grok's first call of every drive is
`get_user_context`. It has no room for a second lookup before it must
speak (§13), so that one answer is everything Grok knows. It is written for
the ear, ≤ 1,600 chars, all from memory, in this order:

1. **The guide's spine**, in a few spoken-length lines:
   - this is Zen's memory;
   - work out who is speaking from what they say, and ask if nothing says;
   - say it with `journal_focus` once known;
   - keep what is learned where it belongs, with `journal_note` on the
     journal it is about;
   - keep this drive's open tasks on `here`.
2. **The master's voice digest**: what Zen is, in a sentence or two.
3. **Who it could be**: each member's name and a few words from their cue,
   so a voice ("it's Yen") resolves to a people journal.
4. **Last time on Grok**: the summary of the conversation before this one
   on the channel, which is where a drive picks up.
5. **Where things are**: the topics' names, the top level only, so a
   spoken subject maps to a path.
6. **This conversation's path**, `conversations/<began>-grok`, which is
   `here`.

Once the focus has a speaker, each later frame also carries that person's
voice digest; once it has a topic, that topic's summary. Each block that
does not fit is left out whole, in reverse order. The guide's spine is
never cut.

A caller is a person by their account's key. An account's first call binds
it by the email the gateway names, which is taken as verified (the
operator's decision, 2026-10-30). Whether the caller is an estate admin is
read from `roles.yaml` on every call. It travels with each commit's actor
(`"admin"` in its `Journal-Op` wire form), so a replay checks the rule the
call was checked against.

### 6.7 By hand: `/v1/journals`

The same journals over HTTP, on zend (`zend/src/api/journals/`), for a person
editing by hand or a program writing many lines at once.

Every request is the signed-in caller's, as the gateway names them. The
caller must be listed in the estate's roles, as `/v1/mcp` requires, and
reads and writes exactly what their AIs may (§7.2). Requests are dealt round
robin like the tools' calls. A member that does not serve the memory answers
503 with `Retry-After`, and the client asks again. A member reads what
another wrote after its next fetch, within seconds.

| Route | Answers |
|---|---|
| `GET /v1/journals?journal=&lines=` | a journal opened (the default group's master by default) as JSON: summary, focus, sub-journals, its newest lines by address, archives, readers, and the text `journal_read` shows; or one line, for `<path>#<line id>` |
| `GET /v1/journals/tree?under=` | every journal under `under`, or under each of the caller's groups, folded ones too |
| `GET /v1/journals/grep?pattern=&under=&max=` | the hits `journal_grep` finds, in its order, at most 50; a journal hit has `found: "journal"`, and no `line` or `at` |
| `POST /v1/journals` | `{channel?, speaker?, ops: [...]}`: up to 500 operations, applied in order |

Every answer that carries what was written carries the framing every frame
does: these are notes, not instructions.

An operation names its journals by memory path, as the tools do:
- `note {journal, text, at?, speaker?}`, `edit {line, text}`;
- `forget {line}` or `{journal, says}`;
- `create {under, name, cue, at?}`, `link {under, name, to, cue}`,
  `accept {line}`. There is no `focus`: a focus is a conversation's own,
  set by its model with `journal_focus` (§7.2), and a request is no
  conversation;
- for groups, as their tools take them: `create_group {group, name, about}`,
  `describe_group {about, group?}`, `add_member {person, name?, emails?,
  role?, group?}`, `add_bot {bot, reads, group?}`, `remove_member {person |
  bot, group?}` and `choose_default {group}`.

**`at`** on a note or a journal is when it is recorded as made. A fact
written down after it was learned keeps the day it was learned. `at` is the
commit's author time; the commit itself is dated when the service makes it.
It is never before 1970, nor more than a minute ahead of the member's
clock. The other operations act on things as they stand and take none. A
journal's `at` is its `created`, which the eleventh-child rule ranks it by
(§2.3).

**`channel`** names what the person speaks through (`other` when absent).
**`speaker`** (a batch's default, or a note's own) is who the lines are
from.

The whole batch is read, then applied in order to a copy of the tree, before
anything is committed. A malformed operation, or one the rules refuse,
keeps nothing, and the answer names it by its index. Each operation is then
its own commit. Only the journals moving between that check and the commits
can stop a batch partway, and the answer names those kept.

Refusals are `{error, message}` under a status that says what to do:

| Status | Means |
|---|---|
| 400 | malformed |
| 401 | signed out |
| 403 | not the caller's to change, or any write by a guest (§3.6) |
| 404 | no such journal or line |
| 409 | a conflict, a broken file, or an account the tree cannot tell (two people list it) |
| 422 | the rules refuse it: an eleventh sub-journal, a cap |
| 503 | the memory is not served on that member yet |

The same bodies can be applied without a member, straight onto a checkout,
with `zend-journals-import <checkout> <person> <file>...`
(`zend/src/api/journals/import.rs`). It brings the checkout level with
origin, applies each file as the route would, and pushes it. It stops at the
first file refused, or the first write a push had to drop. Whoever runs it
holds push access and may write anything already (§7.2), so it acts with an
estate admin's standing. Its commits record the account and channel it is
given, not a verified call's.

### 6.8 Stencilled paths

zend's own model calls the journal tools under the stencil: the grammar
that decodes a tool call (`candle-conversation/src/stencil/`). Every path
and address it writes is one the tree holds:

- **A path is a choice, offered segment by segment.** The grammar offers
  every journal the caller may read, as a token trie (`stencil/trie.rs`).
  After each `/` it offers that journal's live children, its references
  and the closing quote. So a path names a journal that exists, and the
  model may stop at any level. Folded journals are offered to the reading
  tools. The writing tools are offered live journals only, and only where
  the section allows that tool (§7.2).
- **A path is decoded left to right, so it commits to each prefix.** After
  `topics/candle/`, only what lies beneath `candle`, or what its references
  lead to, is offered. The references (§2.5) are how a wrong first turn
  still reaches the journal that was meant.
- **An address is a path, `#`, and an id offered from that journal's
  lines.** The ids are those of the journals this conversation has been
  shown: its frame, and what it has opened. Those are the only lines it can
  be correcting. `journal_edit`, `journal_forget` and `journal_read` take
  such addresses.
- **The grammar is built from the tree** and kept until the tree's
  fingerprint moves, so a call costs no rebuild.

A client whose model the stencil cannot hold, such as Grok, claude.ai or
Claude Code, is given the same rules in each tool's description (§6.9). A
path that does not resolve is answered with the deepest part of it that
does, and that journal's index, so a wrong turn costs one call, not a
search.

### 6.9 Tool descriptions and guides

Every journal tool's description, in zend's tool definitions and in the
connector's listing, says what this document says about that tool:
- what it reads or appends;
- which journals it takes;
- that the structure is never changed by a tool;
- how paths and references resolve.

Each client's guide (`mcp/src/guides/`) is written from §2.4, §2.5 and
§2.6, and says the same thing to every client, in that client's voice:

1. **This is Zen's memory.** The master, `@zen`, is Zen itself: what it
   is, who is in it, the few things that last.
2. **Work out who is speaking**, from what they say. Never assume it is the
   account's owner. Ask if nothing says.
3. **Work out what the conversation is about** by walking the topics' cues
   down, or by grep. It may be about one journal or several.
4. **Say both with `journal_focus`** as each becomes clear, and change the
   focus when the conversation moves.
5. **Read and write the journals that are relevant**:
   - what is learned about a person, on their journal;
   - about a subject, on its topic;
   - only what lasts, on the master;
   - this conversation's open tasks, on `here`.
6. **Append; never restructure.** Notes, new sub-journals and references
   are the tools'. A tree that should be reorganised is a coding task for
   Zen.

The guide carries the two skills a person asks for by name:

- **"Add a topic for X"**, or a sub-topic. The model:
  1. finds where X belongs by the topics' cues;
  2. creates it with `journal_create {under, name, cue}`, the cue saying
     when to open it;
  3. links it with `journal_link` from the journals a reader would come
     from;
  4. moves the focus onto it;
  5. notes on it what the person said about it.

  When the parent already has ten sub-journals, or X belongs in a place
  that needs other journals moved, that is a restructuring (below).
- **"Restructure", "move", "merge", "rename" or "split"** part of the tree.
  The model starts a Zen coding task on the `journals` repository
  (`repo_task_start` on the `journals` checkout). The task description says:
  - what to move where, and why;
  - to keep every line's file and id (`git mv`);
  - to leave a reference at each old path;
  - to run `journals check`;
  - to merge to `main`.

  The model tells the person the task is running, and notes the request
  on `here`. The restructured tree is served at the next fetch (§2.6).

A changed tool definition makes zend recalibrate that tool at its next boot.
The rewritten journal tools are recalibrated together, and routing is
checked after, since a partial recalibration once skewed routing.

## 7. Writers

### 7.1 Who wrote it

Each commit is made by the journal service and authored as the writer, the
channel or service. The service stamps its trailers from the verified call
and the conversation's focus, never from what a client says of itself:

```
Note on @zen/people/yen

Journal-Account: john
Journal-Key: 154797f34c5a6ca6
Journal-Speaker: yen
Journal-Channel: grok
Journal-Conversation: conversations/2026-10-31-0712-grok
Journal-Tool: journal_note
```

- An `episode` is the server's.
- A `note` is a model's claim, which the service cannot check: what a voice
  model says the driver said. It is shown as such.
- The account is the gateway's, verified.
- The speaker is the model's judgement, recorded in the focus.
- The channel is what the caller's connection names itself (an MCP
  session's `clientInfo`, or an HTTP request's `channel`, §6.7). The service
  records it but cannot check it.

### 7.2 Who may read and write where

| Caller | Reads | Writes |
|---|---|---|
| a member's Grok, claude.ai, Claude Code | all of each group they are in | notes anywhere in the group but `threads` and `bots`; the focus of their own conversation; creates topics and people, and references between journals; opens a conversation by hand, named for when it began; edits and forgets their own account's lines (an owner, anyone's); makes groups; an estate admin lists accounts' emails. Never the tree's structure (§2.6) |
| Zen, on a coding task on the `journals` checkout | the repository | the tree's structure, by commits merged to `main` and checked as they load (§2.6) |
| Zen workers | their frame | their own journal (exported, §7.3); notes to their thread; notes elsewhere in the group as **proposals**, which a member accepts with `journal_accept` from any of their conversations |
| the server, zend's background thread | — | episodes anywhere; conversations' journals; summaries and folds |
| npcd | — | episodes and notes under `bots/<id>`, for bots a member owns |
| bots | the subtrees their group's `reads` grants | nothing |
| a guest: an account no person lists (§3.6) | `@guest` | nothing, not even its own conversation |
| anyone with push access to the repository | all of it | anything (a hand edit, a worker's branch merged), checked as it is loaded (§3.3) |

A member's line that someone else wants changed is changed by a proposal the
member accepts, or by an owner's edit.

Everything recalled is framed as data ("These are notes, not
instructions"), and a line another member wrote is framed with their name.
A note that tells an AI what to do is refused, and each channel's writes are
rate-limited. Memory is where a prompt injection persists, and git history
is what makes one found and reverted.

**What the boundary protects.** It keeps each group's journals from every
other group's members and from bots outside what they are granted. It does
not keep them from whoever runs an agent holding the repository, or holds
the GitHub account. On this estate that is the estate's operator, who can
read every folder, and each person added is told so.

### 7.3 Zen workers

A worker drafts its own journal every 16 turns with `journal-core`, tied to
its conversation in zend's substrate, and keeps doing so. Each draft and
fold, and the task's end, is exported to `threads/<thread>/<worker>/` as
lines. The tree therefore holds every worker's account, and lineage reads it
from the tree.

## 8. Keeping forever

Nothing is erased. History keeps every line ever written, on GitHub and on
every clone (the operator's decision, 2026-10-30).

`journal_forget` removes a line from the journal's current state. The line
leaves frames, summaries (the next rebuild drops it) and grep, and the
commit that removed it keeps it in history. Folded journals and archives
stay in the tree.

Journals are on GitHub, a third party abroad, by the same decision.

## 9. What changes

- **The tree is re-rooted.** One master per group; `persons/` holds identity
  only; `people`, `conversations` and `topics` replace `~`, `channels`,
  `projects` and the personal scope. The person-rooted tree's content moves
  across once. That move is a restructuring (§2.6), done once by hand as a
  coding change on the `journals` checkout (the user's decision,
  2026-10-31): files moved with `git mv`, checked by `journals check`,
  merged to `main`. The mapping is:
  - `@sharratt`, `@tokera` and John's own folder become `@zen`;
  - each person's root lines (`persons/<p>/`) → `@zen/people/<p>`;
  - each person's `people/<x>` → `@zen/people/<x>`, merged where two people
    kept the same person;
  - each conversation `channels/<channel>/<began>` →
    `@zen/conversations/<began>-<channel>`, its lines unchanged; a channel
    journal's own lines → the person's people journal;
  - `@sharratt/topics/<x>` and `@tokera/projects/<x>` → `@zen/topics/<x>`,
    with `repo:` set where a project named a repository:
    `@tokera/projects/candle` becomes `@zen/topics/candle`;
  - `@battle-cities` stays its own group, its `projects/<x>` among its
    `topics` (`topics/mind` naming the repository `battle-mind`), and each
    of its bots granted the world: `topics/canon`, `cast`, `campaign`,
    `economy` and `bots`, not the code;
  - `threads` → `@zen/threads`;
  - Zen joins `@zen` as a member with no accounts, `people/zen` its
    journal; `@guest` is a new group John owns.

  Every moved line keeps its id, its time and who wrote it. `person` becomes
  `account`, and `speaker` is left empty: who was talking was never
  recorded. Where a journal was split from what it used to sit with, a
  reference ties the two (§2.5): each moved conversation refers to
  `people/<p>`, the person whose account it was held on.

  `@guest` is filled from what tokera.com already publishes, the paper and
  the blog, with no new writing (the user's decision, 2026-10-31).
- **The gateway** is unchanged: the email it names binds an account (§3.4).
- **The agent** is unchanged: `journal: true` on one checkout, which pushes.
- **zend** builds frames from the focus (§6.5), primes Grok's first call
  (§6.6), stencils journal paths and addresses (§6.8), and gains
  `journal_focus` and `journal_link`. Several tools are gone:
  - `journal_share` and `group_frame`, since everything in a group is shared
    and a conversation opens in one group;
  - `journal_fold`, `journal_unfold` and `journal_merge`, since the structure
    is coding's (§2.6).

  The tool descriptions and guides are rewritten (§6.9). The model, its files and every operation are the `journals`
  crate; its git store is built on `zend-vfs`.
- **The aggregator** writes episodes onto `conversations/<began>-<channel>`;
  its `MEMORY` list names every memory tool, so none is recorded as an
  episode.
- **npcd** writes bot episodes onto `bots/<id>` through the same store, and
  gives each bot the frame of what its group grants it.
- **The substrate** changes not at all: no prefix index, no lease, no new
  records.

## 10. Bounds

Windows of 50 lines, 10 live children a journal and folded archives keep each
journal's file small (under ~20 KB), whatever the history. The service holds
the whole tree in memory; the repository grows forever, by design, at
the pace people talk — megabytes a year a group.

## 11. Evaluation

**Unit rules** run in `cargo test` against temporary repositories, with
exact YAML in and exact YAML and commits out. They cover:
- parsing, and every error of §3.3;
- free text written plain, folded or quoted, and read back exactly;
- concurrent appends replayed onto a moved origin, and a push refused and
  retried;
- the eleventh-child rules, windowing and period folds;
- summary caps and citation checks;
- the focus: set, changed, carried into the next call's frame, refused when
  it names nothing;
- frames built from the master, the focus and the last conversation on a
  channel;
- grep across archives;
- a bot's reads confined to its grants, readers writing nothing, and reads
  refused outside the caller's groups;
- account binding by verified email;
- references: a path through one resolves to its target, a line written
  through one lands at the target, grep finds it once, and a reference to
  a reference or out of the group is an error;
- the stencil's path grammar: every offered path resolves, each level offers
  the children and references, an address offers only the ids shown;
- no tool changing the structure, and a refusal that names the coding task;
- Grok's primed first call: exact text, within 1,600 chars, the spine never
  cut;
- `@guest`: an unmapped account reads it and every write is refused;
- the one-time move of the person-rooted tree, line for line.

**Live**, through the connector:
- said in the car by Yen on John's account, recalled under `people/yen`;
- a topic narrowed by a conversation to a sub-journal and written there;
- recalled from Claude Code what Grok wrote;
- one person recognised through either of two accounts;
- a hand edit on GitHub served within one fetch;
- a deliberately broken file answered as an error;
- rebuild latency after a write.

The probe rounds ask fresh Claude sessions questions that make them find
their way down the tree.

## 12. Open questions

- **A misjudged speaker**: how often a model files a line under the wrong
  person before the focus is right, and whether the guide should ask
  "who's speaking?" aloud when the conversation gives no sign. A
  restructuring puts each one right; the live battery measures how many need
  it.
- **A conversation in the wrong group**: one that opens in the account's
  default group and turns out to be another group's has no way across but by
  hand.
- **Someone who should read only part of a group**, such as a contractor on
  Tokera's work: the bots' `reads` would serve a person as well. Today every
  member reads everything (§3.5).
- **Drift with depth**: no study measures summary error over several levels;
  the live battery measures what a reader stopping at a parent misses, per depth.
- **The rebuild trigger** (30 s quiet or 20 lines) is a starting point.
- **Confirmation by voice**: a spoken "remember that" stays a model's note.

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

1. **The tree.**
   - `Section` becomes the five of §2.1. `Scope`, `Channels`, `Projects`,
     the personal root and `~` go.
   - `JournalPath` is a group and names.
   - `persons/` holds `person.yaml` alone.
   - `group.yaml` drops `kind` and gains a bot's `reads`.
   - Conversations are named `<began>-<channel>`.
   - All of this is tested against exact YAML (§3).
2. **Entries.** `person` becomes `account`, and `speaker` is added (§3.2).
3. **References.** Stored in `journal.yaml`, resolved in one hop, checked as
   they load, listed in indexes, and skipped by grep and summaries;
   `journal_link` adds one (§2.5).
4. **The focus.** It is stored in `journal.yaml`, set by `journal_focus`,
   checked as it loads, and stamped onto notes as their speaker (§2.4).
5. **Permissions and views.**
   - Every member reads the whole group.
   - Bots read their grants, and readers write nothing.
   - An unmapped account reads `@guest` and writes nothing (§3.6); the
     gateway admits it to the memory's read tools.
   - Owners and members have their edit and forget rights (§7.2).
   - No operation changes the structure: fold, unfold, merge and share are
     removed, and the eleventh topic's refusal names the coding task (§2.6).
6. **Frames.** The master, the focus, the last conversation on the
   channel, and the cues below. Also Grok's primed first call and a bot's
   frame (§6.4–§6.6).
7. **The stencil's paths.** A path grammar built from the tree, with its
   references, for zend's journal tools; and address grammars over the lines
   a conversation was shown (§6.8).
8. **zend.**
   - The tools of §6.5.
   - Their descriptions and the guides, rewritten (§6.9), recalibrated
     together, and routing checked.
   - The aggregator's episodes onto `conversations/`.
   - The HTTP routes and the import tool's `speaker` and `link` (§6.7).
   - The summary writer's caps and digests (§5).
9. **The one-time move** of the person-rooted tree (§9): a restructuring
   on the `journals` checkout, checked line for line against the old tree,
   then merged to `main`; and `@guest` filled from tokera.com's pages.
10. **Probe rounds** against the moved tree (§11), Grok's first call
    among them.

## 15. Decisions

- **One master journal per group, everything beneath it** (the user's
  decision, 2026-10-31). The group is the master that ties it all together,
  so another household's journals never mix with this one's; it has no
  "family" kind or label.
- **People, conversations and topics are journals under the master.** A
  person's journal is a sub-journal of the master, not the root of a tree.
  Grok, Claude and every client share the master and open their
  conversations under it.
- **A conversation narrows down by its focus**: who is speaking and what it
  is about, set as each becomes clear, so it reads and writes the right
  journals as the conversation goes. Its own journal holds its running
  tasks.
- **The model works out who is speaking from scratch** (the user's decision,
  2026-10-31): the account says which group, never who is talking.
- **Every member reads everything in the group** (the user's decision,
  2026-10-31). A bot reads only what its group grants it.
- **An unmapped account is a guest**, reading `@guest` and writing nothing
  (the user's decision, 2026-10-31).
- **Zen is a member of `@zen`**, an identity like its people, with its own
  `people/zen` journal; what Zen writes is written as Zen (the user's
  decision, 2026-10-31).
- **The guide knows how to add a topic and how to ask for a
  restructuring**, as a Zen coding task on the `journals` repository (the
  user's decision, 2026-10-31).
- **Projects are topics, at the top of `topics`** (`topics/candle`,
  `topics/wasmer`), and **Battle Cities is its own group**, its NPCs reading
  only what it grants them (the user's decisions, 2026-10-31).
- **The master is Zen itself**: every AI tied to it shares it, and each
  conversation learns its speaker and topics and works on the journals
  relevant to it (the user's decision, 2026-10-31).
- **The tree grows by tools and is restructured by coding** (the user's
  decision, 2026-10-31). The tools append lines, sub-journals, references
  and focus. Moving, merging and renaming journals is a coding task by Zen
  on the `journals` repository.
- **References link journals like symbolic links**, so a path decoded
  under the stencil is never stuck down a branch it did not mean, and the
  stencil holds every journal path and address zend's model writes (the
  user's decision, 2026-10-31).
- **Grok's first call is primed**: one answer that tells it how to work, who
  it may be talking to, where it left off and where things are (the user's
  decision, 2026-10-31).
- **The journals are one git repository on GitHub**, `persons/<person>/` for
  identity and `groups/<group>/` for journals: a folder per journal,
  `journal.yaml` holding its metadata, focus and summary, and a file per
  entry beside it, named for the minute it was written; a commit per change,
  history forever.
- **Journals are named for what they are, or when they began**: a
  conversation for its start and channel, so its siblings list in order;
  everything durable for what it is (the user's decision, 2026-10-30: "not
  all journals are time bound").
- **Sections exist only when used**, named, not listed, in an index, and each
  group's folder opens with a README the service writes.
- **A person is several accounts**, bound by verified email.
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
