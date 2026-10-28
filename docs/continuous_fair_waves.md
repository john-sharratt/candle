# Continuous Fair Waves

**Thesis.** Stop *time-sharing* decode and prefill. Run them **concurrently** by
decoupling the rate at which each advances through the model's layers: fast
decode sweeps every layer on every wave (a token per sweep) while a large prefill
**creeps** through the layers in the background at a throttled rate. Because
decode re-touches every layer's experts continuously, its hot expert working set
is never evicted by the background prefill — so the "cold decode after a big
prefill" penalty disappears, and decode/prefill overlap becomes continuous rather
than an either/or flip.

This supersedes the small⇄large **flip + recovery cooldown** of
[`unified_wave_inference_engine.md`](unified_wave_inference_engine.md). That design
still ran one homogeneous mode at a time and then spent a cooldown re-warming
decode after each large prefill smashed the LRU. Continuous fair waves removes the
smash at its source, so there is nothing to cool down from.

---

## 1. The problem it solves

The current loop dispatches one forward at a time and flips between a small
(decode) forward and a large (prefill/glue) forward. A large prefill forward runs
**all layers, all 128 experts/layer** in one shot. On the streaming box (weights
don't fit; experts DMA from pinned RAM), that one forward loads every expert of
every layer into the LRU and **evicts the tail — which is exactly the working set
the live decode conversations were riding.** The next several decodes are cold:
they re-stream their experts from RAM and run far below steady state.

The flip design papers over this with a *recovery cooldown* — after a large wave,
run decode-only waves until decode tps recovers before allowing the next large
wave. That is a scheduling band-aid over a structural problem: **decode and the
big prefill are fighting over the same resident-expert LRU because they never
share the layer traversal.**

## 2. Core idea — decouple the layer wave from the inference wave

Two distinct notions of "wave" are conflated today:

- **Layer wave** — the traversal `layer 0 → 1 → … → N-1`. This is where the
  expensive work is: per layer, load/prefetch this layer's experts (Markov
  predictor, `transition.rs`) and run one grouped GEMM over whatever tokens are
  present.
- **Inference wave** — the logical unit of *output*: one decode token, or one
  completed prefill forward.

Today these are locked together: an inference wave *is* one full layer traversal.
Continuous fair waves **decouples them**. The layer wave runs continuously; two
independent cursors move through it at different speeds:

- **Decode cursor** advances one layer per step and completes a full sweep
  `0→N` every wave — one token per wave, exactly as today.
- **Prefill / glue cursor** advances through the layers **slowly**, governed by a
  priority ratio (§3). Its inter-layer hidden state (the residual stream between
  layers) is **persisted across waves** rather than living only inside a single
  forward call — so a prefill can pause at layer `L`, let many decode sweeps
  overtake it, and resume at `L` later.

Wherever the two cursors coincide at a layer, their tokens **co-batch through that
layer's experts in one grouped GEMM** — the expert load is shared. Where they
don't, decode runs that layer alone (it was going to pay that layer's decode-set
load anyway).

The result: decode touches every layer every wave, so its top-8 experts/layer stay
at the head of the LRU and never age out. The background prefill still activates
all 128 experts/layer, but it does so **one layer at a time, spread across many
waves**, and each of those loads lands in the LRU *behind* decode's continuously
re-touched set. The prefill's expert churn no longer evicts the decode working
set — it fills the cold slots around it.

## 3. Decode priority — the throttle

New per-layer metadata in `projection.yaml`:

- Each **layer** declares a `decode_priority`: `Low` | `Normal` | `High`.
- A conversation inherits its layer's priority when it is created and attached.
- **Default is `Low`** for every layer; the **dialogue** layer is `High`.

The priority is the **decode-to-prefill airtime ratio** `R` — how many decode
tokens are produced for each prefill forward that completes:

| priority | R (decode tokens per completed prefill) | effect |
|---|---:|---|
| Low    | 1  | prefill drains at full speed; least decode protection (bulk ingest) |
| Normal | 16 | prefill stretched ~16× while 16 decode tokens land |
| High   | 64 | prefill heavily throttled; decode latency maximally protected (dialogue) |

Concretely, with `N` layers the prefill cursor is given a budget of `~N/R` layers
of forward progress per wave (fractional, accumulated):

- **Low (R=1):** `N/1 = N` — the prefill clears all layers in one wave, i.e. it
  fully overlaps a single decode token. Maximum ingest throughput, minimum decode
  shielding.
- **Normal (R=16):** `N/16 ≈ 3` layers/wave — the prefill completes over ~16
  decode sweeps.
- **High (R=64):** `N/64 < 1` — the prefill advances roughly one layer every one
  or two waves, creeping in the deep background while ~64 decode tokens land.

`R` is the single knob that trades ingest throughput against decode latency, set
per layer by how interactive that layer's conversations are. The layer budget
keeps each step's slice of the creep small; how often a slice rides is set by
airtime — decode gets `R` times the creep's share of the device (§6).

**Beside decode rows the slice is also bounded in time** (`scheduler/creep_slice.rs`):
no more than one decode-only step's worth of layers, at what a layer-row of creep
has cost on the steps it rode (learned per step; a single layer until measured).
`N / R` alone made the `R = 1` slice the whole sweep: an 8K-row ingest group at
depth rode one decode step for 1.7 s, and every decode beside it stood still for
each of them — ~5 tokens a second where it ran ~100 alone (`.5`, 2026-10-17;
`decode:cobatched:step` 1,737 ms against 34 ms decode-only). Bounded, the same
group's co-batched steps measured 133 ms. The share `R` sets is unchanged — the
debt is still the slice times `R` — it is only paid in steps rather than seconds.
With nothing decoding there is nothing to protect, and the creep runs its `N / R`.

**An outranking decode is protected, and a `High` one from `High` work too.**
The creep yields to the decodes that outrank it — and, when it is `High`, to the
`High` decodes — at the highest of their ratios; with none, `R = 1` and it clears
every layer in one wave. `R` is re-read every wave, so the creep returns to full
speed the wave the last such decode finishes. A creep runs at the band of its
highest-ranked prompt (a decode or prompt whose layer does not resolve counts as
`High`, the band it is gated in), and section chunks, which are background
ingest, ride at the band of the prompt they joined — alone they are `Low`
(`scheduler::prefill::throttle_ratio`).

**Below `High`, a band never holds back its own work.** Latency matters less
than throughput there, so a `Normal` or `Low` creep beside a decode of its own
band runs whole, co-batched with it.

**A `High` creep beside a `High` decode shares airtime at `R = 2`**
(`HIGH_BESIDE_HIGH_RATIO`). A conversation is never on both sides: while it
prefills (a turn, a tool result) it is not decoding. The collision is between two
conversations of one band — two zend dialogues, or a dialogue and a client's
passthrough conversation. When only an outranking decode counted, a same-band
creep rode the step whole — a full chunk through every layer — and the decode
beside it waited all of it: measured on `.5`, 2026-10-16, a dialogue's decode
step that took 17–19 ms alone took 3.2 s (profile build) whenever an npcd turn's
chunk, with the section chunks riding it, shared the step.

So the creep keeps its chunk whole — each layer's experts are still loaded once
for all its rows, which matters when they stream from RAM or NVMe — and takes the
same time-bounded slices and airtime debt as any other outranked creep, with the
decode keeping two thirds of the device. Pacing it in tokens instead (9 prefill
tokens per decode token) starved it: a tool result's prefill beside another
dialogue's decode took 8–27 s to reach its first token, which is the pause a user
sees at the end of a tool call. Airtime keeps the decode's step bounded by the
slice and the prefill's progress bounded below by the ratio, on every machine.

Every decode has a band. zend's dialogue and a client's passthrough conversation
are `High`. npcd's characters (its perception, action and interaction layers) and
its prose jobs are `Normal`: zend outranks them, so a character's turn creeps
beside a zend dialogue's decode, and a zend prompt is never paced by a
character's. The titler and summary probes are `Normal` too: background to the
dialogue they serve. Each creep group logs
its band, ratio, budget and rows at debug on
`candle_conversation::scheduler::creep`.

### 3.1 Bands share; nothing is paused

The throttle is the whole of priority. No band is held out of a forward for
another's sake: every band's decode rows ride every decode step — decode is
width-bound, so a row beside the dialogue's costs it little and is the other
row's whole progress — and each band's prefill creeps at the ratio of the
decodes that outrank it, a `High` one beside a `High` decode paced in tokens. Priority orders the rows (the highest band leads, so
the one-adapter rule keeps the dialogue's), admission's per-band offer, and the
keep-one-alive head (the highest band's first queued turn).

**Why there is no pause.** One was built: while any higher band had work, and
for 15 s after, every lower band was held out of every forward. It was meant to
keep one person's dialogue the device through its tool rounds. Beside npcd's
cast it was a mode switch that never switched back — the characters tick every
few seconds and are `High`, so the gap the pause waited for never came. Measured
on `.5`, 2026-10-16, with two characters running and zend's repo-map ingest
queued: decode one row wide, forwards on the device ~12% of the wall clock, and
30 folder workers finishing 0.7 folders a minute. Its measured motivation — a
dialogue's belief-scan index invalidated by ingest uploads, its forwards widened
by ingest rows — is a cost the throttle bounds; stopping the ingest is not the
price of it.

**Admission is not held behind a band it outranks.** The admission floors defend
the residency the running slots decode at, and when those slots are a lower
band's, they are what spent it — and they hold it until they finish. A prompt of
a band some running slot yields to is judged on the forward's row cap alone
(`admit::WaveRate::try_outranking`, `admit::Ground::outranks_active`), and the
width backstop does not turn it away either. Measured on `.5`, 2026-10-17: with
fourteen ingest and character slots in flight, a dialogue's two-row continuation
was refused at the decode floor on every pass (21.5 GiB resident against a
25.3 GiB floor), and the person's tool result waited for background work to drain
before it was even admitted. Among equals the floors hold as before.

The normalization warm-up replay (`scheduler/norm_warm.rs`) is the one piece of
background work that does not ride a forward: it runs on the loop thread, one
replayed turn per loop iteration, which a decode quantum bounds to one per
`WAVE_SLICE`.

## 4. Glue rides with prefill

Glue (boundary gap-fill, `paged-glue`) is treated as **prefill-class** by the
wave loop: it inherits the prefill ratio and its cursor advances on the same
throttle. It differs only in the attention kernel that runs at each layer
(`paged_glue_attn` vs `paged_prefill_batched` vs `paged_decode`) — the layer
traversal, expert load, and grouped GEMM are identical. Glue *is* part of getting
a reprojected context prefilled, so it belongs on the prefill cursor.

## 5. What stays exactly the same

- **Slot allocation and the single background scheduler thread** — unchanged.
- **Concurrency levels** (active decode / prefill / section counts) — unchanged.
- **Attention kernels** — `paged_decode`, `paged_prefill`, `paged_glue` are
  untouched; each still fills its own disjoint rows.
- **The layer traversal is still single-threaded, start → end**, and still uses
  the **Markov expert prediction + per-layer expert streaming** exactly as today
  (`quantized_qwen3_moe.rs`, `expert_lre/`).
- **Decode state** — the sampler, DRY window, sampling-state-per-sequence — is
  unchanged; it is simply applied to the decode cursor's rows within the unified
  waves.

The novelty is entirely in **(a)** lifting the prefill/glue inter-layer buffers
out of the per-forward scope so they persist across waves, and **(b)** the
per-cursor layer-advance throttle driven by `decode_priority`.

## 6. Why it works

- **Expert cache pressure is fixed at the source.** Continuous decode sweeps keep
  the interactive experts resident *through* a large background prefill. No
  wholesale eviction, so no cold-restart penalty — and therefore **no recovery
  cooldown** to schedule around.
- **Continuous decode/prefill overlap.** Decode and prefill are no longer mutually
  exclusive modes; a large ingest prefill runs *underneath* live dialogue decode,
  cutting the latency of mixed decode+prefill workloads instead of serialising
  them.
- **One knob, per-layer, matched to interactivity.** `High` on dialogue protects
  first-token and inter-token latency; `Low` on bulk-ingest layers lets prefill
  run at full speed when nothing interactive is in flight. The same loop serves
  both without a mode switch.

## 7. Wave time-slicing and mid-wave admission

The decode *quantum* — the batch of decode steps one scheduler-loop iteration
runs before returning to the top — is bounded by **wall-clock, not step count**.
It is clipped to `WAVE_SLICE` (2 s) regardless of whether the in-flight
generations have finished; unfinished sequences persist in `active_decodes` and
resume in the next quantum, so clipping simply passes the remaining work forward.
(A large `MAX_DECODE_STEPS` backstop only engages if steps ever run far under the
~74 ms/step WDDM launch floor.) Time-slicing replaces the old fixed 32-step
budget so the loop returns to the top on a predictable cadence no matter how fast
or slow the steps run.

The quantum is the cadence of the loop top — relief, guests, the warm-up,
telemetry. It is **not** the cadence of the creep: every decode step is a wave.

- **The creep rides every step its airtime allows.** A step that carried the
  creep (or a section chunk) owes decode its share times `R` of decode-only
  airtime, its share being the step's length less what a decode-only step takes
  now (an average over recent decode-only steps, `decode_only_step`); decode-only
  steps pay it down, and the next step after it is paid carries the creep again —
  across quantum boundaries, the debt kept on the scheduler (`creep_owed`). At
  `R = 1` decode and creep get equal airtime; at `R = 64` decode gets 64 times the
  creep's. The creep used to ride only a quantum's first step: whenever anything
  decoded, a prefill advanced once per `WAVE_SLICE` — an 8K chunk that runs in
  120 ms went once every 2 s, and a dialogue's tool result waited out the rest of
  another conversation's slice before its prefill moved.
- **The debt is the creep's share, not the step's length over `R`.** Owing
  `d / R` assumed the creep's `N/R` layers were a sliver of the step they rode.
  At depth they are not: one layer of an eight-member ingest group at 300K
  positions cost ~40 ms beside a ~30 ms dialogue decode, owed 1 ms at `R = 64`,
  and so rode every step — the dialogue decoded 13–15 steps a second where it ran
  33 alone (`.5`, 2026-10-17). Owed as its share times `R`, the ingest rides about
  once per 2.5 s of dialogue decoding and at full speed the moment the dialogue
  stops.
- **Mid-wave admission.** After every step the quantum runs the same admission it
  runs at the loop top — `drain_submissions` → `promote_new_prefills`, under the
  identical `admit_window`/VRAM entry criteria —
  so a conversation that queues *while a wave is executing* is projected and
  seated in `active_prefills` immediately. It joins the creep the next time
  `form_wave_group` re-forms: the next step the creep rides with no group in
  flight. The creep's held residual is a fixed-member-order invariant, so new
  work is never injected mid-sweep — only picked up at the group's sweep
  boundary.
- **An outranked creep pauses.** Every creep step is bounded in time — by the
  decode cadence, or by the decode-less slice — so a group takes many steps to
  reach its sweep boundary whether it is throttled or not, and a prompt that
  outranks it — a dialogue's tool result beside an ingest group — would wait all
  of them. A section counts at its own band: a schema's sections are `Low`, and
  a section a conversation submits for itself (a journal entry its next turn
  reads) runs at the band that conversation decodes at. Measured on `.5`,
  2026-10-24, before sections had a band: a dialogue's 38-row journal entry
  waited out a 6,213-row ingest sweep beside four `Low` decodes — a creep no
  decode throttled, so nothing paused it — then rode the next, 41 s to seal. The
  group is set aside at the layer it reached with its residual, members and rows
  (`scheduler::paused_creep`); the prompt's band forms a group of its own and
  sweeps; the paused group resumes at the same layer once nothing above its band
  is waiting. Nothing touches a paused member's state meanwhile — it is
  mid-prefill — and a paused member is in no other group and no standalone
  section pass. With a group already paused, the one in flight finishes its
  remaining layers in one step instead (`creep_outranked_by_a_waiting_prompt`).
- **A `High` group carries `High` prompts only.** Every row of a group rides
  every layer of its sweep, so a person's prompt beside eight characters' and
  ingest chunks waits for all of their rows: measured on `.5`, 2026-10-17, a
  10-member, 8,190-row group creeping a layer a step held a dialogue's first
  token ~24 s. A waiting `High` section makes the group `High` too, carrying
  only `High` sections; with no `High` prompt beside it, it runs in the
  standalone section pass. Below `High`, throughput matters more than latency,
  and the bands and section chunks mix as before.
- **Admission sizes a turn against the group it will run in.** One function
  answers which group forms next and what each member advances in it
  (`scheduler::next_group`), and both the group's formation
  (`form_wave_group`) and admission read it. The formation then fits the group
  to the ground its windows will place (`fit_cohort_to_ground`), which
  admission prices separately, and leaves sections out on a wave whose
  standalone section pass already ran. What admission asks it, for an offer at
  band `b`:
  - *Which work stays set aside.* A creep in flight — one whose sweep has run
    a layer; a group formed and not yet run re-forms from scratch — that `b`,
    or any prompt already waiting, outranks is paused, unless a group is
    paused already; then it finishes its sweep and its members re-enter the
    formation the rows and positions that sweep advanced them. A creep `b`
    does not outrank also re-enters at its sweep boundary, advanced. The
    paused group's members are in no formation.
  - *Whether a paused group resumes instead.* It does when nothing in the
    formation — the offer included — is above its band; the offer then waits
    behind a sweep it is not in, and is not offered rows.
  - *The formation itself*, exactly as `form_wave_group` runs it: highest band
    leading, then nearest its first token; a live band's group carrying its own
    band only; a background gang (`docs/prefill_gangs.md`); one adapter; the
    group budget shared by `gang::gang_shares`; sections in what the prompts
    leave. A section the prompts leave no room for runs in the standalone
    section pass — unless a prompt in flight is above it, which holds that
    pass (`section_batch_band`) — and is packed as that pass packs
    (`section_pack`).

  The offer's rows are its share of that group. A share of nothing is a full
  forward, and a full forward takes no more turns (`admit_ground::room_left`);
  nor does a group that leaves the offer out — a band below its own, another
  adapter, or a slot already prefilling, which runs one prefill at a time and
  which a submission may not fork mid-prefill.

  **Only a live band takes rows from the members behind it.** A `High` or
  `Urgent` turn that leads its group is admitted though the forward is full,
  taking rows from the members after it — that is the starvation this rule
  ends — but never the last of a member's rows at its own band or above,
  which would leave that member holding its store in no forward. Below the
  live bands throughput matters more than latency, and a turn is admitted
  only into rows the forward has spare, as it always was: taking rows from
  the members ahead it would join without the rate model weighing it.

  A live turn with no place is walked past, as another band's is, so the
  turns behind it that would lead the group are still offered — where that
  reorders nothing on their slot: no turn changes a slot before work queued
  ahead of it has forked it, and no fork — a turn-group case's view, a
  submission's — reads a slot a turn queued ahead of it has yet to change.
  Below the live bands, and once a pass has closed its rows, the first turn
  with no place ends the band as it always did: none behind has spare rows
  either. A closed pass re-offers a live member as a join, riding the group's
  forwards at its share without widening them. Sections are not walked past:
  one can name the sections before it as its prefix, and ingested ahead of
  them it would be sealed against what of them had landed.

  The rate model weighs the rows the offer *widens* its group by, not its
  share: a turn that leads a re-formed group takes rows from the members
  behind it in a forward no wider, and joins — charged its share on top of
  the standing creep, it was refused for the row cap the creep alone already
  filled. Rows and tier are both measured against the same formation without
  it, at the width the pass stands at: a turn that widens its group clears
  the cohort's width, every member widens with it, and that is the turn's to
  pay for. A turn that only re-shares a full forward's rows adds next to no
  tier, and one whose group is smaller than the one it displaces adds none,
  those members having been priced at their own admission. It owes the creep
  residual for its own rows. A turn admitted mid-sweep claims its store now
  and runs from the boundary, the claim priced as any other.

  An open pass places each turn at the deployment's target first, so it can
  still find wider rows paying; a turn that would not widen its group there
  is placed at the width the pass stands at. A turn sized to the floor is
  priced at the rows it is clipped to; it, and a turn whose rows the rate
  refuses, close the pass at the cohort's width as the rate model judged it
  — the rows it charged before any offer and those the pass admitted. The
  width is the cohort's, read by every formation until the cohort shrinks: a
  narrower one for the turn's own group would bind every other group, a long
  creep's included, for as long as that turn prefilled. Within it the
  formation fits each group it forms to the ground its windows place
  (`fit_cohort_to_ground`).

  Sized against the standing cohort instead — the held group first, at its
  fixed rows — one long member starved everything: on WhiteWorld (2026-10-27
  22:46–22:48) a single `High` member 158k tokens long held all 3,072 rows of
  every forward, four `Urgent` content checks were never offered for ninety
  seconds, and an 87-token `High` turn waited six minutes.
- **A sweep is never restarted.** Its finished layers have taken its rows into
  their carried state — the QSA index, the DeltaNet state, the PLE history — and
  nothing rolls that back; only the K/V is position-addressed, so only the K/V
  survives a rerun. Restarted, an outranked group ran its rows into those layers
  twice: measured on `.5`, 2026-10-16, three section slots' indexes stood 2,545
  tokens ahead of their K/V after two restarts, and the next append needed an
  index page inside the forward, which the partition refuses — failing the wave
  and the dialogue decode riding it. So a member that leaves mid-sweep (its slot
  freed, its prompt errored) takes its rows out of the held residual and the
  survivors carry on (`Scheduler::drop_wave_members`); a window that fails
  mid-sweep on a device OOM or a refused tier is retried in place, at most four
  times, with its residual put back; and anything else mid-sweep fails the
  creep's members rather than replaying them.
- **Per-slice relief.** The cheap ingest throttle (`regulate_ingest_admission` +
  the gentle warm-backed demote) also runs after each step, self-gated on an
  active ingest, so KV production is bound to the hot→warm drain rate *across* a
  long slice rather than only once per quantum.

## 8. Open questions

- **Per-cursor layer budget arithmetic.** `N/R` is the clean statement of intent;
  the exact fractional-accumulator rule (and whether prefill advances in
  contiguous layer runs or single layers per wave) is the primary tuning knob and
  should be validated against realised decode tps and ingest throughput.
- **Co-batch dtype at the shared layer.** Where decode (int8 context) and prefill
  (float context) coincide at a layer, their o_proj inputs must reconcile to one
  hidden dtype before the shared FFN — the same reconciliation the unified-wave
  design calls out (cheap; o_proj output is common hidden).
- **Persisted prefill activation VRAM.** Holding a large prefill's residual stream
  between waves costs `[prefill_tokens, hidden]` of resident activation for the
  duration of its (now stretched) traversal. On the 16 GB box this competes with
  KV; the throttle must fall back toward `Low` (drain fast, hold briefly) under
  VRAM pressure.
- **Bounded ingest context (prerequisite, unchanged from the unified-wave design
  §4.7).** Each prefilled scope must attend a bounded window, not the whole
  growing file, or the persisted prefill activation and its swept KV grow without
  limit.
