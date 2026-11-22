# Incremental chunk exclusion

The anomalous-chunk exclusion (`docs/provenance_scoped_signatures.md` §2.6) is
learned **incrementally**: a fixed random sample of 4,000 calibration questions,
each sample token's scan statistics kept as an exact, mergeable summary, and
every signature appended to the group folded in once — against the sample, never
against the whole gallery again. Each sample question's statistics are its own
keyed record, and the group's cut and cursor one more, all replicated in the
substrate, so a restart, a deploy or a change of member resumes where the last
fold stopped instead of learning from nothing.

It replaces the from-scratch learn (`Conversation::learn_chunk_exclusion`,
`GroupScanner::learn_chunk_exclusion`, `InferenceState::learn_chunk_exclusions`).
The live scan's use of the exclusion (`belief_files::cut_excluded`), the cut rule
and its constants do not change.

## 1. The problem

Measured on BlackWorld, 2026-11-16 (`code_reading`, 6,729 files):

- **A restart learns from nothing.** The exclusion is held in memory only, so the
  first ingest pass after every start learns it in full. After the 20:12 restart it
  ran 20:20:08 → 20:37:52 (**17 m 44 s**, 13,373 questions, `kept=false`, cut 269 of
  1,240,078 content chunks and 1,543 of 471,235 calibration chunks). After the
  22:03 restart it started at 22:05:36 and was still running at 22:17.
- **It runs on the wave thread, in front of live turns.** Each of its ~1,672 jobs
  (`ScanGroupVotes`, 16 questions each, two competitions) is one background job the
  scheduler runs to completion between waves (`step_norm_warm`). The step census
  put the `warm` activity at 51.2 s of the minute to 22:08:17 and 170 s of the three
  minutes to 22:16:19; the phase timeline, which does not time it, drew it as
  `blocked`, and a seal scan's launches waited 1.97 s for the wave thread.
- **It is quadratic in the corpus, twice.** The probes are 2 per file and the
  gallery is every file, so the kernel work is O(F²). And every job re-assembles all
  of the group's files and lays out the whole ~190k-window segment before scanning
  its 16 questions — O(F) setup repeated O(F) times.

## 2. What the learn computes

For one competition (content, or calibration — §2.6), every **probe token** `t`
of a question, in every fold group `g`:

- the scan's per-case statistics over the competition's gallery: the best
  agreement `a1` and its case (file) `c1`, the runner-up case's best agreement `a2`
  and its case `c2`, and over every scanned gallery token the agreement sum `s`,
  sum of squares `q` and count `n`;
- the vote `bdp_vote(a1, a2, s, q, n) = max(0, (a1 − mean)/std) × (a1 − a2)`
  (`candle-kernels/src/provenance/bdp_vote.cuh`), for case `c1`;
- per question, the needle gate keeps the tokens whose summed vote is in the top
  `NEEDLE_KEEP_FRAC` (0.25) (`window_score::gate`);
- a kept vote for a file not the question's own is credited to the 32-token chunk
  holding the gallery token that won it — the best agreement in its fold group
  among `c1`'s tokens (`chunk_exclusion::best_token`);
- a chunk's **mass** is the sum of what it wins; the cut is today's rule
  (`provenance/chunk_exclusion/anomaly.rs`): log mass more than `K` = 5 (content) or
  `K_CALIBRATION` = 3 (calibration) robust deviations above the median of the chunks
  that won anything, each file keeping at least `KEEP` = ½ of its chunks.

## 3. The property the design stands on

**Every per-token statistic merges exactly.** `s`, `q`, `n` are integer sums. `(a1,
c1, a2, c2)` is the top two of a max over cases, where a case's best only rises as
its tokens are added. So the statistics of a gallery are the merge of the
statistics of any partition of it, in any order:

```
merge((a1,c1,a2,c2,s,q,n), (b1,d1,b2,d2,s',q',n')) =
    (top-2 by agreement of {(a1,c1),(a2,c2),(b1,d1),(b2,d2)} with a case's own
     agreements combined by max, s+s', q+q', n+n')
```

A case outside both parts' top two is beaten in its own part by two distinct
cases at least as good, so it cannot be in the merged top two: the merge is exact.
Ties rank by the lower case — the order the kernel's ascending case loop gives
(`bdp_ranks_offer` displaces only on a strictly greater agreement). With that order
fixed, folding a gallery stream by stream yields **bit-identical** statistics to
scanning it whole, and the vote, gate, mass and cut computed from them are
identical too.

The statistics split by what they describe:

| Part | Describes | Changes |
|---|---|---|
| `a1, c1, a2, c2` | the question against its best two files | when a new file beats one of them — rare once built |
| `s, q, n` | the question against the whole gallery (z's mean and spread) | on every appended stream |

The vote, gate and mass are recomputed from the statistics each time the cut is
taken, on the host.

## 4. The design

### 4.1 The sample

- **4,000 questions, chosen by hash.** The sample is the 4,000 live calibration
  questions of the group, at the galleries' signature width, whose `hash(stream
  id)` is smallest (bottom-k sampling; a question at another width has no token
  to scan, so it never takes a place).
  It is uniform, needs no random state, and every member computes the same one
  from the same substrate.
- **It changes one question at a time.** A new question enters exactly when its
  hash is below the 4,000th smallest, and the 4,000th then leaves; a sample
  question that is tombstoned or archived leaves and the next-smallest hash
  enters. With 161,502 calibration questions in `code_reading` today, about 2.5% of
  new questions enter.
- **Its entries.** Each sample question's tokens × the competition's fold groups
  are its entries; each remembers the question's owner file, whose wins are never
  credited.

### 4.2 The records

Every piece of state is keyed by what it describes, so a question that leaves is
one tombstone:

| Key | Holds | Written |
|---|---|---|
| `chunk_exclusion/<group>/q/<question stream>` | the question's entries in both competitions — per token per fold group `a1, c1, a2, c2, s, q` and the winning token `(stream, position)` — and the **cursor it was written at** | when the question enters the sample, and in rotation: each fold rewrites the 200 records written longest ago (`ROTATE`, a twentieth of the sample); tombstoned when the question leaves the sample, or when a learner loading it finds its question gone |
| `chunk_exclusion/<group>` | the cut — both competitions' chunks `(stream, range)` — the **cursor** and the gallery counts | after a fold that changes any of them |

A question's record is ~75 tokens × 3 fold groups × 2 competitions × 49 B ≈ 22 KB;
the group record a few tens of KB. Cases (`c1`, `c2`) are timeline ids, the same on
every member.

**A record is written in rotation, not when it moves.** Almost every append
moves some entry of almost every question — a pass of 994 streams moved 3,992 of
4,000 (BlackWorld, 2026-11-17) — so writing on change rewrote the sample every
fold, ~116 MB replicated. A record instead carries its own cursor, the learner
holds every entry exactly, and the sample turns over every 20 folds, ~4 MB a
fold. A reload groups the questions by their records' cursors — about 20 — and
catches each group up on the streams above its own cursor alone (`catch_up`),
which is exact for every stream appended since.

**The cursor is the highest signature stamp folded.** Every signature record
carries a hybrid-logical-clock reading `hlc` (`docs/replicated_substrate.md` §6.2),
the same on every member. A fold takes every group stream and question whose
signature `hlc` is above the cursor, and every tombstone above it, folds them, and
raises the cursor to the highest it folded. A record that arrives by replication
after the cursor has passed its stamp is not folded, and a takeover whose cursor
trails the records it loads folds a stream twice; both are a few streams out of
tens of thousands, and the statistic they touch is over the whole gallery, so the
design takes that loss for one number of state.

### 4.3 Folding

The learning member holds, besides the records, the `SigWindow` of every stream
it folded (a shared reference to the substrate's own blob — no copy):

| Change since the cursor | What is scanned | Merge |
|---|---|---|
| **Stream appended** (a new turn, or a new file) | its tokens against the sample | merge its statistics into every entry; an entry whose `c1` it now holds has its winning token found within it |
| **Stream removed** (turn tombstoned, file archived or tombstoned) | its held `SigWindow` against the sample | subtract `s`, `q`, `n` exactly; every entry whose `c1` or `c2` was its case is **rescanned** against the whole gallery, one batched launch for all of them |
| **Stream rewritten** (a new signature) | old window, then new | removal, then append |
| **Question enters the sample** | its tokens against the whole gallery | new entries, its record written |
| **Question leaves the sample** | — | its entries dropped, its record tombstoned |

In the calibration competition the gallery is every calibration question not in
the sample, so a question entering the sample also leaves that gallery (removal
above), and one leaving enters it (append).

**A rescan reads the whole gallery, so it is batched.** An entry a removal
leaves waiting is kept out of the cut, and once waiting entries pass 2% of the
sample's entries (`RESCAN_SHARE`) one pass rescans all of them — each waiting
token against every folded stream, its leaders found again from nothing, its sums
(which are exact) kept.

A stream removed while no member held its window — a removal a takeover finds —
cannot be subtracted: there is no window left to scan, so its `s`, `q` stay
counted, a few streams out of tens of thousands. Every fold leaves each entry
whose leader or runner-up names a file no longer in its gallery waiting on a
rescan, so no entry votes for a file that is gone.

### 4.4 The cut

After each fold the learner recomputes, on its own thread: each entry's vote from
its statistics (the same `f32` expression as `bdp_vote`), the gate per question,
the mass per chunk, and the cut by today's rule with today's constants. At 4,000
questions the masses are sparse — about 0.3 votes a content chunk — so the rule
keeps only chunks far above one typical vote: the strongest outliers, which is
where the measured gain is (`K` = 5 beat 4, `anomaly.rs`). Each cut chunk's vote
count is logged with the cut, so the margin at this sample size is on record.

A winning token is found only when the cut needs it: an entry keeps its winner
until its `c1` or `a1` changes, and the cut resolves the entries left without one
by searching `c1`'s tokens on the host — a file at a time, each file's signatures
decoded once, files in parallel.

### 4.5 The kernel

A **statistics mode** of the paged BDP scan: for each probe token, fold group and
segment it writes `(a1, c1, a2, c2, s, q)` (`BdpStats`) instead of the vote — the
values the finalize already holds when it calls `bdp_vote` — on every backend.
The scalar kernel writes them in its own finalize; the tensor backends (BMMA,
IMMA) share a finalize kernel of their own for it, `bdp_stats_finalize_kernel`,
so the vote finalize's code and registers are untouched. Every vote-mode kernel
compiles to the registers and spills it did before, on sm_86, sm_89 and sm_120.
The arena runs it as `GalleryArena::scan_stats`, through the live scans' index
cache: a slice is scanned once per chunk of the sample's tokens, so its index is
built for the first chunk and reused, its turns checked current, by the rest;
once the learner moves on, it idles out of the cache like any other.

### 4.6 Scheduling

- **One learner per group in the cluster.** It runs under the lease
  `exclusion/<group>` (`zend/src/ingest_lease.rs`, renewed while it works, 120 s
  TTL), after each ingest pass. A member that finds the lease held installs the
  replicated cut (§4.7), scans nothing and lets go of any learner state it held.
  A member in a cluster that cannot propose leaves the learner alone, as it does
  a worker's turns: two learners folding one group would write its records from
  two states.
- **Jobs are small in both directions.** A job is up to 8,192 probe tokens of the
  sample (`PROBE_CHUNK`) against a slice of up to 262,144 gallery tokens
  (`SLICE_TOKENS`), laid out as segments of at most 1,024 files
  (`SEGMENT_FILES`, under the scalar kernel's 1,536-case bound) of **that slice
  only**: no job assembles the whole group. The probe chunk keeps a job's upload
  (~1.5 MB) and output (~0.8 MB a segment) inside the scratch the live scans
  already hold (`with_synced_upload` grows by doubling and never shrinks), and the
  slice bounds a job's kernel at ~2 × 10⁹ token pairs, ~10 ms. A slice's index
  is built once and reused by its later probe chunks (§4.5).
- **A job is background work on the wave thread, one a pass.** The learner runs
  on zend's own thread and sends each job there as a background held launch
  (`HeldScan::stats` over `SchedulerRequest::ArenaScan` with `background`), which
  the loop queues with its other background scans — the normalization warm-ups,
  the offline harness's group-vote scans — and serves one a pass
  (`step_norm_warm`), between waves, in the `warm` band. The merging and the cut
  stay on the learner's thread. Served as requests the moment they arrived, a
  learner's run of jobs went back to back between two waves: 19 questions
  entering the sample held decode at ~60 t/s for ~10 s (BlackWorld, 2026-11-17).
- **The first build** — no records for the group yet — is the whole gallery folded
  as appends, in the same jobs.

### 4.7 Replication

- Every member installs the group record's cut as it applies the record — written
  here, replayed at load or replicated (`Substrate::apply_custom_object`); the
  greatest stamp holds, as for any keyed value, and a tombstone leaves no cut.
- A member that takes the lease reads the question records and the cursor, and
  folds what is above the cursor (§4.3). With no records it runs the first build.
  A learner whose cursor is behind the group record's — another member folded
  since — drops its state and reads the records again.
- The group record is written before the question records: a fold that dies
  between them leaves a cursor whose missing question records are questions
  entering the sample on reload, scanned whole.
- A record's entries are exact as of its cursor, and a takeover catches each
  question up on what is above it: what stays approximate is only what a removal
  while no member held the state left in the sums (§4.3).
- On load, the calibration gallery taken as folded holds no question that has a
  record or is in today's sample; today's sample's others enter it and are
  scanned whole.

## 5. Cost

Estimated from the live scan rate (2,213 probe tokens against the ~40 M-token
content gallery in ~350 ms, ≈ 2.5 × 10¹¹ token pairs/s); the per-job event (§4.6)
measures it:

| Work | Token pairs | GPU time |
|---|---|---|
| A new 2,000-token file against the sample (~300 k probe tokens) | 6 × 10⁸ | ~2.4 ms, in 37 probe chunks |
| A question entering the sample (~75 tokens × ~53 M gallery tokens) | 4 × 10⁹ | ~16 ms |
| First build, both competitions | ~1.6 × 10¹³ | ~64 s, in ≤ 10 ms jobs |
| The from-scratch learn it replaces (13,373 questions) | ~5 × 10¹³ | ~3.5 min of kernel, stretched to ~18 min by per-job setup |

Steady state is O(appended tokens × sample), constant per appended token, and
writes only the question records an append changed. A restart costs nothing.

## 6. What is removed

`learn_chunk_exclusion` and its batching (`in_batches`, `PROBES_PER_SCAN`),
`ChunkExclusion::stands_for` / `learned_from` and the question count they keep,
`EXCLUSION_QUESTIONS_PER_FILE`, and `InferenceState::learn_chunk_exclusions` /
`learn_chunk_exclusions_behind`. `GroupScanner::scan` and `ScanGroupVotes` stay for
the offline harness (`zend/src/api/scan.rs`).

## 7. Tests

All `#[test]`s, asserting raw values:

- **Merge** — the statistics of a gallery equal the merge of its parts' in every
  order, ties included (`stats.rs`).
- **Fold equals scan** — a gallery folded a third, two thirds, then all of its
  files gives entries and counts identical to folding it at once.
- **Removal** — a fold then a removal equals never having folded the stream, sums
  and leaders; a file that vanished while no member held the state has the
  entries it led rescanned to the leaders a fold without it finds.
- **Sample** — bottom-k by a fixed SplitMix64 mix: the same sample from any order;
  one question in and the largest out (`sample.rs`).
- **Records** — each record is its bytes, and reads back as itself; an older
  layout is not read (`records.rs`); a restarted learner reads them, scans
  nothing and writes nothing; a fold writes the new questions and the 200 oldest
  records; a question reloaded behind the cursor catches up to exactly the
  entries a continuous fold holds; a group record installs its cut on apply and
  its tombstone clears it; an orphaned record is tombstoned.
- **Background** — a background launch is queued, not run, as it arrives, and
  answered one a step; a background handle sends background launches.
- **Cursor** — the cursor rises to the highest stamp folded; a fold with nothing
  new launches nothing; a turn appended after a restart is the one stream folded.
- **Statistics mode** — each backend's rows equal the definitions on a fixture of
  empty cases, descending case order and a dropped out-of-range case, and the
  backends' rows are identical (`gallery_arena/scan/tests_votes.rs`).
- **Cut** — a hub token planted in one chunk and asked by every question is the
  content cut.

The learner's tests fold against `HostArena`, which answers statistics launches
from their definitions on the host.

## 8. Code map

| Concern | Where |
|---|---|
| Entry statistics and merge | `provenance/chunk_exclusion/stats.rs` |
| Sample (bottom-k by hash) | `provenance/chunk_exclusion/sample.rs` |
| Question and group records, and their keys | `provenance/chunk_exclusion/records.rs` |
| The cut installed on apply | `substrate/marking_dead.rs` (`install_chunk_exclusion`) |
| A group's galleries and questions | `projection/chunk_exclusion/gallery.rs` |
| Bounded jobs and their merge | `projection/chunk_exclusion/scan_jobs.rs` |
| Learner state and folding | `projection/chunk_exclusion/fold.rs` |
| Cut from state | `projection/chunk_exclusion/cut.rs` |
| The learner, its load and its writes | `projection/chunk_exclusion/learner.rs` |
| Statistics-mode kernel | `candle-kernels/src/provenance/bdp_vote.cuh`, `bdp_scan.cu`, `bdp_bmma.cu`, `bdp_imma.cu`, `gallery_arena/scan/scan_api.rs` |
| A job sent to the wave thread, one a pass | `gallery_arena/held_scan.rs`, `scheduler/remote_arena.rs` (`connect_background`), `scheduler/norm_warm.rs`, `group_scanner.rs` |
| The `warm` band | `scheduler/phase_ring.rs`, `scheduler/mod.rs`, `web/content/common/lib/perf.js` |
| Learner and lease | `zend/src/session/ingesting.rs`, `zend/src/ingest_lease.rs` |
