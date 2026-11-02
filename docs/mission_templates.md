# Mission Templates

A mission template is a reusable, parameterised mission — a skill the world knows
how to carry out, written once and filled in with arguments each time it is
issued. Zen keeps a library of them; it issues one to a drone by naming the
template and its arguments, and the drone carries the concrete mission the
template produces. This document is the authoritative design; where the code
disagrees with it, the code is wrong unless this document is, and then this
document is fixed in the same change.

The point is to turn the drones from characters that *talk about* work into ones
that *do* real work — play a film on the family's television, patch the machines,
add a feature to zend, deploy a release — by naming the work Zen wants done and
letting the drone carry it out with the acts the world already has.

The design is **deliberately small**. A template is not a new kind of object: it
is the existing `Routine` with parameters. Everything that already reads, gates,
signs off and reports a mission works unchanged, and the grammar the drones decode
against is the safety net it already is.

---

## 1. What this is built on

- **`Mission`** (`npcd/src/engine/mission.rs`) — the unit of work: a `prompt`
  (the brief the carrier reads as part of who it is), ordered `todo` steps,
  optional `work` (documents to read/write with `checks` gates), `grants` (the
  world-building acts the carrier is lent while it carries the mission), and the
  `report` that closes it. The engine signs steps off **from facts it observes**
  — the body is in the room, it scanned the machine, the document was committed —
  never on the carrier's word.
- **The order table** (`npcd/src/engine/order_table.rs`, `docs/zen_moon_mind.md`
  §8) — a character that takes no orders (Zen) `order_post`s work for a drone, as
  an ad-hoc `{brief, steps}` or a named standing `{job}`; reports come back
  through `order_results`.
- **Routines** (`npcd/src/engine/mission_gen/world.rs`, `sim/missions.rs`) —
  authored missions a world hands out, `Routine { id, brief, steps }`, read from
  the world's repo and offered through `Choices::Jobs`. **These are templates
  already** — they just can't take arguments yet.
- **The grammar gate** (`performable` and `stencil_param` in
  `npcd/src/engine/tools.rs`) — a drone can only emit an act that is *performable
  here*, and only with argument values that are *live here*. A device act is
  dropped where no device stands; where one does, the emitted device is
  constrained to the real names. This is what ended the invented-device looping,
  and it holds **whatever a mission's prose says** — which is why template
  parameters need no validation machinery of their own (§4).
- **The handset threads** (`npcd/src/sim/phone.rs`) — the Discord messaging the
  drones already do, over which a coding mission drives zend (§7).

---

## 2. The template: a `Routine` with parameters

`Routine` gains three optional fields; nothing else changes, and a routine
written today keeps working:

```
id:       kebab-case, unique within its world's library
brief:    "Put {title} on {device}."          # {param} substitution
steps:    [ "go to {room}", "put {title} on {device}" ]
params:   [ title, device, room ]              # NEW — names that may appear in {…}
grants:   [ act, ... ]                         # NEW — lent while carried (Mission.grants)
work:     { writes, reads, checks } | null     # NEW — for doc/coding/deploy templates
```

- **`params`** are plain names. At issue time every `{name}` in `brief` and
  `steps` is replaced by the given argument. Substitution is total: a template
  that references a `{name}` it did not declare, or that is issued with a declared
  parameter left unfilled, is refused rather than carried with a hole in it. A
  template with no `params` is exactly a routine as it works today.
- **`grants` / `work`** are the mission fields of the same name, carried through
  onto the produced `Mission` (`Mission::granting`, `Mission::with_work`), so a
  template can lend world-building acts or name a document to write with its
  completion checks — all existing machinery.

There is **no separate `MissionTemplate` type**, and **no `kind` field**. A
template produces an ordinary `Mission`; the kind of work it is lives only in
which folder it sits in (§5), because the engine never branches on it — the steps
carry the acts, and completion is gated by `work.checks` where a template needs a
gate.

---

## 3. Issuing a template

`order_post` already posts a standing job whole by name (`{job}`); it gains one
thing — arguments:

```
order_post { job: "play-a-film", args: { title: "Dune", device: "the living room TV", room: "the front room" } }
```

`post` (`order_table.rs`) fetches the routine by id as it does now, substitutes
`args` into its brief and steps, and posts the concrete mission. A missing or
undeclared argument is refused there, with the parameter named, the same way a
post with no step is refused today. No new post form and no new concept: a
template *is* a job, and this is the job path taking arguments.

---

## 4. Why parameters need no binding machinery

A template parameter is a placeholder in prose, nothing more. It does **not** need
to be bound to a live set, because the grammar already is: when the drone comes to
emit the act a step describes, `performable` has dropped the act if its device
isn't here, and `stencil_param` has constrained the device argument to the names
that *are* here. So even if a template is issued with a device that doesn't exist,
the drone cannot act on it — it either has no such act to emit or emits it against
a real device. The template's prose guides; the grammar binds.

This means v1 ships with **no change to the grammar** and **no shared live-set
module**. (An earlier draft proposed lifting `Choices`/`Within` out of
`engine/tools.rs` so templates could validate arguments early and Zen could be
offered only the fillable templates. That is a genuine nicety — a wrong argument
is caught at issue rather than quietly redirected by the grammar — but it is not
needed for correctness, and it touches the just-stabilised grammar. It is
deferred, to be added only if issuing-time validation proves worth it in use.)

---

## 5. Where the library lives

Templates are stored per world, in that world's repo, foldered for browsing, one
template to a file:

```
<world-repo>/missions/<group>/<template-id>.yaml
            /missions/devices/play-a-film.yaml
            /missions/coding/add-a-feature-to-zend.yaml
            /missions/ops/patch-our-machines.yaml
            /missions/README.md
```

The folder name is for people reading the library; it carries no behaviour. The
loader walks `missions/` with `MindFiles::walk` (which exists) and validates each
file as `WorldMissions::parse` does now — a template that would produce a broken
mission is refused at load, naming the file and the fault.

**The library's folder holds templates and nothing else.** The world's other
mission files keep their own places, so no loader meets a file of another kind:
its standing missions under `standing/<world>/` (`docs/zen_moon_mind.md` §12)
and its round in `rounds/<world>.yaml` (§17.3).

The two worlds keep their templates in different repos, along the line the repos
already divide on:

- **Battle Cities** is lore and fiction — its templates live in the **mind repo**
  (`battle-mind`).
- **Zen and the moon** are the real-world mirror — their templates live in the
  **journals repo**, which this design **promotes to the unified moon-world repo**
  (the moon world's memory *and* its capabilities), with a top-level `README.md`
  describing what the repo now holds. The repo should be **renamed `journals` →
  `moon`** to match the world it serves (coordinated with the standing `life` →
  `moon` world rename; the rename is its own operation, not part of the first code
  landing).

Both repos are git and replicated across the cluster, so the library gets
versioning, replication and edit-via-thread for free.

---

## 6. Authoring, and the `critical` gate

**Authoring.** Zen writes a template with a new order-giver act, `mission_write`,
offered where `order_post` is. It renders what Zen gives — `id`, `group`, `brief`,
`steps`, `params` — to a `Routine` YAML and opens an operation on the built-in
**`write-template`** workflow: a Drone writes the file at a bench (the reused
`map-table` kind — a world shapes its own capabilities as it shapes its map) and
publishes it at the `merge-table` into the world's repo
(`missions/<world>/<group>/<id>.yaml`), a `routine-parses` check refusing a
template that would not load. The loader reads it next cycle, and it is then
`order_post`-able by name. **A world with its own missions runs no authored
workflows — only this one built-in**: an own-missions world otherwise clears the
record's workflows, so without it `mission_write` would have no pipeline to carry
a template onto the record. The Creator or Yen can also author and edit templates
by editing the repo directly; the same files, the same history.

**Autonomy is the default.** A drone carries a mission to completion on its own
and asks no one — plays the film, patches the machines, deploys the release. The
only brake is a step marked **`critical`**: one whose effect cannot be taken back
— deleting data, wiping a substrate, force-pushing or rewriting pushed history,
dropping a device's pairing or credentials, a factory reset, destroying a world or
room. The flag lives on the **step**, not the template, so an otherwise-autonomous
mission can have one guarded move (respond-to-intrusion runs free, but a step that
*wipes* a compromised machine is critical).

Reaching a critical step, the drone **messages an authorizer on Discord** — states
what it intends and why, holds the mission open, and proceeds only on an
affirmative. The authorizers are a configured list, currently the Creator and
**Yen**, each by Discord identity; either may clear it, whichever answers first. No
reply, or a refusal, leaves the step undone and the mission reported
blocked-on-authorization. A critical action fails **safe** (not done), never
**open**. The reach-out is an ordinary `message` act and the mission is gated on
the reply landing — no new channel plumbing, and the same muscle that makes the
drones actually *wait on* a Discord reply instead of ignoring it.

This mirrors the lines the system already draws: the irreversible and
outward-facing wants a human; the reversible does not; and it is the in-world twin
of the cluster's "Decide commands need approval" split. A deploy is reversible
(keep restores the last-good binary), so it runs free.

---

## 7. Coding and deploy — driving zend over a thread

A coding or deploy mission does not build anything itself. The drone **drives a
zend task as a correspondent**, over the Discord messaging it already uses: it
opens a thread to zend and posts the task (the feature, the fix, the release), the
task's turns arrive as messages, and the drone nudges it and answers its questions
across as many ticks as it takes, until it reports done. The mission's terminal
step is gated on that — the engine signs it off when the task completes, not on the
drone's word, the same shape as the doc-commit and publish gates in `mission.rs`.

**This is the one part with real new plumbing, and it does not exist yet.** npcd
has no in-process path to zend's repo-task machinery (`repo_task` lives in
`zend/src/mcp/`); the drones' threads today are handset/Discord threads between
world characters. Bridging a world thread to a zend repo task — posting the task,
piping its turns back onto the thread as messages, learning when it is done — is a
self-contained integration, and it is scheduled **last** (§8) so every other kind
of template ships without it. Until it lands, coding/deploy templates can still be
authored and issued; they simply wait on this bridge to actually run.

---

## 8. Build order

Each step is landable on its own, smallest first; steps 1–4 are world machinery
with no GPU and no live devices, fully unit-testable.

1. **Parameters.** Add `params`/`grants`/`work` to `Routine`, `{name}`
   substitution, and the foldered loader (walk `missions/`). Tests assert raw
   produced briefs and step lists, refusal of an undeclared or unfilled
   parameter, and refusal of a malformed file.
2. **Issue.** `order_post { job, args }` — substitute and post; refuse a bad
   argument at the table.
3. **Authoring.** The built-in `write-template` workflow + a `routine-parses`
   check, and the `mission_write` act that renders the template and opens an
   operation on it (a world's own-missions path runs this one workflow).
4. **The `critical` gate.** Step flag, the authorizer list, the Discord reach-out,
   the fail-safe hold.
5. **Coding/deploy (separate phase).** The world-thread ↔ zend-repo-task bridge
   (§7), then the completion gate on top.
6. One template of each group proven end to end; then fill the library toward a
   hundred.

---

## 9. To settle at wiring time

- **The authorizers' Discord identities** — the Creator's and Yen's handles, for
  the configured list (step 4).
- **The `journals` → `moon` repo rename and its `README.md`** — with the standing
  `life` → `moon` world rename, as its own operation.
- **Whether issuing-time argument validation is worth the `Choices` lift** (§4) —
  revisited once templates are in use, not before.
