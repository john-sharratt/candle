# Gallery dedup overlay

The provenance gallery holds every scanned turn's folded signatures on the
card (`docs/gallery_residency.md`). Most of a turn's tokens are near
duplicates of tokens before them in the same turn — code repeats its
identifiers, prose its framing — and a near duplicate votes as the token it
repeats. The overlay keeps those off the card. **The substrate keeps every
token**: the overlay decides only which of a turn's tokens the arena uploads,
and a reader asking for them all gets them all.

Code: `candle-conversation/src/provenance/gallery_arena/dedup/`.

## 1. Naming a token: the codebook

A token's signature is a sign vector (24 words, 1,536 bits); the scan's
measure of two tokens voting alike is Hamming agreement. A **two-level binary
codebook** names each token by its nearest code:

- binary k-means under Hamming distance, each code the per-bit majority of its
  members, starts chosen farthest-point from a seeded sample;
- 64 coarse codes, each split into up to 64 fine codes over its members —
  4,096 codes for 128 comparisons a token;
- trained over **every fold group** of the signature, so one mask per turn
  serves every reader whatever groups it weighs;
- 4 rounds a level, on at most 200,000 tokens: every `k`-th turn whole, `k`
  spreading the sample over the corpus. ~1 s.

The arena trains its codebook once, from the first corpus a dedup reader
scans or preloads, and names every turn with it for the arena's life.
Signatures are already bits: nothing is hashed.

## 2. Keeping a turn's tokens: the mask

`Keep::FirstPerCode(n)` keeps, within each turn, the **first** `n` tokens of
each code to appear. The first, not the nearest the code: the nearest are a
code's most average members, and kept instead they cost retrieval in every
setting measured.

The mask is a bit per token with per-word prefix counts, so a kept token's
place among the kept ones — its **rank** — is O(1). A turn's pages pack its
kept tokens in order, kept token `r` at page `r / 32`, slot `r % 32`; the scan
index maps each window token it keeps to its rank and skips the rest. A
deduped scan is therefore exactly the raw scan of its kept tokens
(`a_deduped_scan_is_the_raw_scan_of_its_kept_tokens`).

Masks are computed in parallel before a scan index's or preload's uploads,
and cached per turn under (content fingerprint, keep, codebook), so a turn
evicted and uploaded again is not renamed.

Measured (`provenance_gates dedup`, 200 calibrated `code_reading` files,
1.54 M tokens, 600 questions, production rule `[0,1,1] + 0.5 × [0,0,1]`):

| setting | kept | R@1 | R@10 | MRR |
|---|---|---|---|---|
| every token | 100% | 0.355 | 0.687 | 0.460 |
| every 4th (control) | 25.0% | 0.292 | 0.637 | 0.395 |
| random 1 in 4 (control) | 25.0% | 0.278 | 0.627 | 0.383 |
| 64×64 over groups 1–2, 1 per code per turn | 23.0% | 0.362 | 0.668 | 0.457 |
| **64×64 over every group, 1 per code per turn** | **27.9%** | **0.363** | **0.677** | **0.463** |
| 64×64 over every group, 2 per code per turn | 44.2% | 0.367 | 0.682 | 0.465 |

Dropping a code found in many files, as carrying no signal, was measured and
refused: at every cutoff it cost most of the retrieval (a token sits ~22% of
its bits from its code, so codes are regions every file reaches).

## 3. Readers, and a turn several read

Each scan segment carries its `keep` from the reading policy:

- a belief group's content competitions: `scan.gallery_keep`;
- its calibration competition: `scan.calibration_gallery_keep`;
- a collection (tools): its policy's `scan.gallery_keep`.

YAML: `gallery_keep: all` or `gallery_keep: 1` (tokens per code), likewise
`calibration_gallery_keep`. Default `all`.

Keeps **cover** like fold groups do: `All` covers everything, more per code
covers fewer. A resident turn serves a read whose keep its own covers; one
held narrower is uploaded again under the union. A turn's keep is therefore
never narrower than any reader's — a reader never sees fewer tokens than it
asked for — and a turn two readers read under different keeps settles on the
wider once both have read it. The index fingerprint includes each segment's
keep, so an index over a turn's kept tokens never stands in for a reader
asking for more.

A keep with no codebook trained holds every token and is recorded as `All`.

The overlay is the arena's: a host scan (no CUDA) reads every token.

## 4. The scan route

`POST /v1/substrate/zend/scan` takes `per_code` (absent or `0` for every
token), so the offline gates measure the overlay on the daemon's own arena.
The chunk-exclusion learn scans under the live keeps.
