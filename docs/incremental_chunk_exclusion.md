# Incremental chunk exclusion

The anomalous-chunk exclusion (`docs/provenance_scoped_signatures.md` §2.6) is
learned **incrementally**: a fixed random sample of 4,000 calibration questions,
each sample token's scan statistics kept as an exact, mergeable summary, and
every signature appended to the group folded in once — against the sample, never
against the whole gallery again. The state, its cursor and the cut it yields are
replicated in the substrate, so a restart, a deploy or a change of member resumes
where the last fold stopped instead of learning from nothing.

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

Ties rank by the lower case — the order the full kernel's ascending case loop
gives (`bdp_ranks_offer` displaces only on a strictly greater agreement), with
cases numbered by ascending `TimelineId`. A winning token ties by the lower
`(stream, position)`. With those orders fixed, folding a gallery stream by stream
yields **bit-identical** statistics to scanning it whole, and the vote, gate, mass
and cut computed from them are identical too.

The vote, gate and mass are recomputed from the statistics each time the cut is
taken: they move whenever `n`, `s`, `q` move, which is every fold, but that is a
host pass over the sample's entries, not a scan.

## 4. The design

### 4.1 The sample

- **4,000 questions, chosen by hash.** The sample is the 4,000 live calibration
  questions of the group whose `hash(stream id)` is smallest (bottom-k sampling).
  It is uniform, needs no random state, and every member computes the same one
  from the same substrate.
- **It changes by one question at a time.** A new question enters exactly when
  its hash is below the 4,000th smallest, and the 4,000th then leaves; a sample
  question that is tombstoned or archived leaves and the next-smallest hash
  enters. With 161,502 calibration questions in `code_reading` today, about 2.5% of
  new questions enter.
- **Its entries.** Each sample question's tokens × the competition's fold groups
  are its entries; each remembers the question's owner file, whose wins are never
  credited.

### 4.2 The state

Per group, per competition:

| Part | Content | Size (4,000 questions, ~75 tokens each) |
|---|---|---|
| Entries | per sample token per fold group: `a1, c1, a2, c2, s, q` and the winning token `(stream, position)` | ~1.8 M entries × 36 B ≈ 65 MB for both competitions |
| `n` | gallery tokens scanned — one count per competition, the same for every entry | 8 B |
| Folded set | every gallery stream folded: its stream id, case, the identity `(origin, oseq)` of its signature record, and its `SigWindow` (a shared reference to the substrate's own blob — no copy) | ~200 k streams × 40 B |
| Cursor | the per-origin `oseq` set of the signature records folded — an `AppliedSet` (`persistence/replica/`) | small |

**The cursor is the redo log's own position.** A record's local `(segment,
offset)` moves with every maintenance pass and other members' records arrive in
any order (`docs/replicated_substrate.md` §1.3, §6.9), so neither is a cursor. Its
identity `(origin, oseq)` survives both; the substrate already holds it beside each
turn's signature (`streams.entry(key).ids.wide_q_sigs`). A stream is *folded* when
its signature's identity is in the cursor.

### 4.3 Folding

The learner compares the group against the folded set:

| Change | What is scanned | Merge |
|---|---|---|
| **Stream appended** (a new turn, or a new file) | its tokens against the sample | merge its statistics into every entry; a new `c1` takes its winning token from the stream itself (`best_token` over its tokens) |
| **Stream removed** (turn tombstoned, file archived or tombstoned) | its held `SigWindow` against the sample | subtract `s`, `q`, `n` exactly; every entry whose `c1` or `c2` was its case is **rescanned** against the whole gallery, one batched launch for all of them |
| **Stream rewritten** (new signature identity) | old window, then new | removal, then append |
| **Question enters the sample** | its tokens against the whole gallery | new entries |
| **Question leaves the sample** | — | its entries dropped |

In the calibration competition the gallery is every calibration question not in
the sample, so a question entering the sample also leaves that gallery (removal
above), and one leaving enters it (append).

**Finding the change is a pointer compare.** Each active timeline's `FileGallery`
is held on its entry and replaced whenever anything it reads changes
(`substrate/file_gallery.rs`); the learner keeps the `Arc` it folded and visits only
the files whose `Arc` differs — ~7,500 compares, well under a millisecond — and
reads the calibration records the same way (`calibration_snapshot`). The cursor is
what a fold resumes from after a restart or a change of member, where no `Arc` was
kept: every stream whose signature identity is outside it is appended.

**A removal after a restart.** A stream removed while no member held the state
has no window left to scan, so its `s`, `q`, `n` cannot be subtracted exactly.
They are taken as its token count times the entry's current mean (and the mean of
squares), and the entries whose winner it was are rescanned exactly as above. The
state counts the tokens removed this way; when they pass 1% of `n`, the moments are
recomputed exactly by one moments-only pass over the gallery (§4.6's first build,
without the winners).

### 4.4 The cut

After each fold the learner recomputes, on its own thread: each entry's vote from
its statistics (the same `f32` expression as `bdp_vote`), the gate per question,
the mass per chunk, and the cut by today's rule with today's constants. At 4,000
questions the masses are sparse — about 0.3 votes a content chunk — so the rule
keeps only chunks far above one typical vote: the strongest outliers, which is
where the measured gain is (`K` = 5 beat 4, `anomaly.rs`). Each cut chunk's vote
count is logged with the cut, so the margin at this sample size is on record.

### 4.5 The kernel

A **statistics mode** of the paged BDP scan: for each probe token and fold group
it writes `(a1, c1, a2, c2, s, q, n)` instead of the vote — the values the
finalize already holds when it calls `bdp_vote` (`bdp_ranks_emit` →
`bdp_stats_emit`), on every backend (scalar, BMMA, IMMA). The winning token's
position is found on the host within the one winning stream.

### 4.6 Scheduling

- **One learner per group in the cluster.** It runs under the lease
  `exclusion/<group>` (`zend/src/ingest_lease.rs`, renewed while it works, 120 s
  TTL), after each ingest pass and each calibration pass. A member that finds the
  lease held installs the replicated cut (§4.7) and scans nothing.
- **Jobs are bounded by work, not by count.** A job is a slice of the delta — the
  streams of up to a token-pair budget against the sample — sent as one
  `SchedulerRequest` and laid out as one segment of **that slice only**: no job
  assembles the whole group. The budget is set so a job's kernel is ≤ 10 ms.
- **The wave thread's time is named.** The `warm` step's time gets its own band in
  the phase window, so it is never drawn as `blocked`, and each job notes an engine
  event with its split (slice streams, tokens, setup, kernel, readback).
- **The first build** — no replicated state for the group yet — is the whole gallery
  folded as appends, in the same bounded jobs.

### 4.7 Replication

- **The cut and where it stands** — a keyed custom object `exclusion/<group>`:
  the cut chunks (stream, range) of both competitions, the cursor, the sample's
  hash bound, and the checkpoint's blob hash. A few tens of KB. Every member
  installs its cut on apply (`Substrate::set_chunk_exclusion`); the greatest stamp
  holds, as for any keyed value.
- **The state** — a content-addressed blob (`persistence/blob/`), written as a
  checkpoint when the learner has folded 10% of the gallery's tokens since the last
  one, every 30 minutes while it changes, and when it hands back the lease. A
  superseded checkpoint's blob is retired by its tombstone.
- **Taking over.** A member that takes the lease reads the newest checkpoint and
  folds the delta outside its cursor (§4.3). With no checkpoint it runs the first
  build.

## 5. Cost

Estimated from the live scan rate (2,213 probe tokens against the ~40 M-token
content gallery in ~350 ms, ≈ 2.5 × 10¹¹ token pairs/s); the per-job event (§4.6)
measures it:

| Work | Token pairs | GPU time |
|---|---|---|
| A new 2,000-token file against the sample (~300 k probe tokens) | 6 × 10⁸ | ~2.4 ms |
| A question entering the sample (~75 tokens × ~53 M gallery tokens) | 4 × 10⁹ | ~16 ms |
| First build, both competitions | ~1.6 × 10¹³ | ~64 s, in ≤ 10 ms jobs |
| The from-scratch learn it replaces (13,373 questions) | ~5 × 10¹³ | ~3.5 min of kernel, stretched to ~18 min by per-job setup |

Steady state is O(appended tokens × sample), constant per appended token. A restart
costs nothing; a takeover costs the delta since the last checkpoint.

## 6. What is removed

`learn_chunk_exclusion` and its batching (`in_batches`, `PROBES_PER_SCAN`),
`ChunkExclusion::stands_for` / `learned_from` and the question count they keep,
`EXCLUSION_QUESTIONS_PER_FILE`, and `InferenceState::learn_chunk_exclusions` /
`learn_chunk_exclusions_behind`. `GroupScanner::scan` and `ScanGroupVotes` stay for
the offline harness (`zend/src/api/scan.rs`).

## 7. Tests

All `#[test]`s, asserting raw values:

- **Merge** — the statistics of a gallery equal the merge of its parts' in every
  order, ties included (`a1`/`a2` equal, cases ordered by id).
- **Fold equals scan** — a fixture group folded stream by stream, in shuffled
  orders, gives entries, masses and cut identical to one scan of the whole gallery.
- **Removal** — a fold then a removal equals never having folded the stream; a
  removed winner's entries are rescanned and match a whole scan.
- **Sample** — bottom-k by hash: the same sample from any order of questions; one
  question in and the 4,000th out; a tombstoned sample question replaced by the
  next.
- **Cursor** — a state restored from a checkpoint and folded to the present equals
  one folded continuously.
- **Statistics mode** — each backend's `(a1, c1, a2, c2, s, q, n)` matches the
  scalar reference on a fixture, and the vote computed from them matches the vote
  mode's.
- **Cut** — today's rule on fixture masses gives today's cut.

## 8. Code map

| Concern | Where |
|---|---|
| Entry statistics and merge | `provenance/chunk_exclusion/stats.rs` |
| Sample (bottom-k by hash) | `provenance/chunk_exclusion/sample.rs` |
| State, folded set, cursor | `projection/chunk_exclusion/state.rs` |
| Folding and change detection | `projection/chunk_exclusion/fold.rs` |
| Cut from state | `projection/chunk_exclusion/cut.rs` |
| Checkpoint and keyed object | `projection/chunk_exclusion/replicated.rs` |
| Statistics-mode kernel | `candle-kernels/src/provenance/bdp_vote.cuh` (`bdp_stats_emit`), the three backends, `gallery_arena/scan.rs` |
| Scheduler job and its event | `scheduler/norm_warm.rs`, `scheduler/mod.rs` |
| Learner and lease | `zend/src/session.rs`, `zend/src/ingest_lease.rs` |
