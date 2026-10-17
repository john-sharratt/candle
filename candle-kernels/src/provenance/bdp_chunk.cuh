// How many probe tokens one pass of a tensor-core BDP scan takes.
//
// The scan's global accumulators hold max, sum and sum-of-squares — 20 bytes —
// for every (probe token, group, case). Sized for the whole probe at once they
// grew with the gallery: ~11K cases, a few thousand probe tokens and three
// groups asked a single launcher scratch for 3.2 GB (`.5`, 2026-10-17), refused
// by the reservation's held ground and taken from the CUDA pool outside it —
// past WDDM's residency budget, so the process was paged to host RAM. A probe
// token's statistics are its own and its outputs are contiguous in the probe,
// so the launchers run the probe in chunks of this many tokens against a
// scratch of at most `BDP_SCRATCH_BYTES`: identical results, bounded memory,
// and the gallery re-read once per chunk.
#pragma once

// The accumulator scratch one pass may use.
#define BDP_SCRATCH_BYTES ((size_t)96 << 20)

// Bytes of accumulator per (probe token, group, case): u32 max, u64 sum, u64 sumsq.
#define BDP_ACC_BYTES_PER_SLOT 20

// Probe tokens per pass over `n_groups` groups and `n_cases` cases: as many as
// `BDP_SCRATCH_BYTES` holds, and never fewer than one.
inline int bdp_probe_chunk(int n_groups, int n_cases) {
    const size_t per_token = (size_t)n_groups * (size_t)n_cases * BDP_ACC_BYTES_PER_SLOT;
    const size_t fits = per_token == 0 ? 1 : BDP_SCRATCH_BYTES / per_token;
    return fits < 1 ? 1 : (fits > (size_t)0x7fffffff ? 0x7fffffff : (int)fits);
}
