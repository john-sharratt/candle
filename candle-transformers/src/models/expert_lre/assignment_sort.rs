//! The grouped-GEMM dispatch sort: `(token, expert)` assignments grouped by
//! ascending expert id.
//!
//! This is the host half of [`ExpertCache::submit_moe_work`]'s contract
//! (`super::handle::ExpertCache`): same-expert tokens must be contiguous, in
//! ascending expert order, each assignment carrying its token index and its
//! flat routing-weight index. Every MoE model that routes through the cache
//! runs exactly this grouping between its router readback and its submit —
//! the router *math* differs per model (softmax top-k, sigmoid+bias,
//! hash-routed layers), the grouping does not, so it lives here once.
//!
//! A counting sort, **O(A + E)** (A = token→expert assignments, E = experts):
//! expert id is a small bounded integer, so bucketing beats any comparison
//! sort and stays linear at the prefill widths the expert-stream amortization
//! wants. The scatter is stable in token order, which is what makes the
//! result identical to a stable sort-by-expert.
//!
//! Sentinel rule: the router kernels write `n_experts` itself into any top-k
//! slot that found no valid expert (a token whose logits were all −inf/NaN),
//! so ids `>= n_experts` are skipped in both passes — they are not real
//! experts and would index past the pipeline's expert arrays. `slot_k` keeps
//! the original top-k position, so the flat weight index stays aligned even
//! when a sentinel slot is skipped.

/// One routed assignment: `(expert_id, token_idx, flat_weight_idx)` with
/// `flat_weight_idx = token_idx * k + slot_k`.
pub type ExpertAssignment = (u32, u32, u32);

/// Group the per-token top-k routing `idx_cpu` (`[n_tokens][<=k]` expert ids)
/// by ascending expert id. Returns the ascending active expert ids and the
/// grouped assignments.
pub fn sort_assignments_by_expert(
    idx_cpu: &[Vec<u32>],
    k: usize,
    n_experts: usize,
) -> (Vec<usize>, Vec<ExpertAssignment>) {
    let k_u = k as u32;

    // Pass 1: count assignments per expert (skipping sentinels).
    let mut counts = vec![0u32; n_experts];
    for idxs in idx_cpu {
        for &eid in idxs {
            if (eid as usize) < n_experts {
                counts[eid as usize] += 1;
            }
        }
    }
    // Prefix-sum into per-expert bucket starts; collect the ascending active
    // expert ids in the same pass.
    let mut cursor = vec![0u32; n_experts];
    let mut expert_ids: Vec<usize> = Vec::new();
    let mut running = 0u32;
    for (e, &c) in counts.iter().enumerate() {
        cursor[e] = running;
        running += c;
        if c > 0 {
            expert_ids.push(e);
        }
    }
    // Pass 2: scatter each assignment into its expert's bucket (stable in
    // token order).
    let mut assignments: Vec<ExpertAssignment> = vec![(0, 0, 0); running as usize];
    for (tok, idxs) in idx_cpu.iter().enumerate() {
        let tok_u = tok as u32;
        for (slot_k, &eid) in idxs.iter().enumerate() {
            if (eid as usize) >= n_experts {
                continue;
            }
            let pos = cursor[eid as usize] as usize;
            assignments[pos] = (eid, tok_u, tok_u * k_u + slot_k as u32);
            cursor[eid as usize] += 1;
        }
    }
    (expert_ids, assignments)
}

#[cfg(test)]
mod tests {
    use super::sort_assignments_by_expert;

    /// The contract, asserted on raw expected values: ascending expert groups,
    /// stable token order within a group, flat weight indices preserving the
    /// original top-k slot.
    #[test]
    fn groups_by_ascending_expert_stable_in_token_order() {
        // 3 tokens, k = 2, 4 experts. Token 0 → {2, 0}, token 1 → {0, 3},
        // token 2 → {2, 2} (a router may repeat under degenerate logits).
        let idx = vec![vec![2u32, 0], vec![0, 3], vec![2, 2]];
        let (experts, asg) = sort_assignments_by_expert(&idx, 2, 4);
        assert_eq!(experts, vec![0, 2, 3]);
        assert_eq!(
            asg,
            vec![
                // expert 0: token 0 slot 1 (widx 1), token 1 slot 0 (widx 2)
                (0, 0, 1),
                (0, 1, 2),
                // expert 2: token 0 slot 0 (widx 0), token 2 slots 0+1 (widx 4, 5)
                (2, 0, 0),
                (2, 2, 4),
                (2, 2, 5),
                // expert 3: token 1 slot 1 (widx 3)
                (3, 1, 3),
            ]
        );
    }

    /// Sentinel ids (>= n_experts) are dropped from both passes, and the flat
    /// weight index of the surviving slots is untouched by the skip.
    #[test]
    fn sentinels_are_skipped_and_weight_indices_stay_aligned() {
        // n_experts = 4; the router wrote the sentinel `4` into token 0 slot 0.
        let idx = vec![vec![4u32, 1], vec![1, 4]];
        let (experts, asg) = sort_assignments_by_expert(&idx, 2, 4);
        assert_eq!(experts, vec![1]);
        // Token 0's surviving assignment keeps slot 1's weight index (1), not 0.
        assert_eq!(asg, vec![(1, 0, 1), (1, 1, 2)]);
    }

    #[test]
    fn empty_routing_yields_empty_dispatch() {
        let (experts, asg) = sort_assignments_by_expert(&[], 8, 16);
        assert!(experts.is_empty());
        assert!(asg.is_empty());
        let idx = vec![Vec::new(), Vec::new()];
        let (experts, asg) = sort_assignments_by_expert(&idx, 8, 16);
        assert!(experts.is_empty());
        assert!(asg.is_empty());
    }
}
