//! Token classes, and the biases composed from them.
//!
//! The generic core. A **class** is any property of a token worth steering on —
//! the writing system it emits, whether it is tool syntax, whether a caller
//! banned it this turn. Each class owns a [`TokenBitset`], built once and never
//! moved; steering is a matter of choosing which sets to union and how hard to
//! weight the result.
//!
//! Nothing here knows what a class means. [`super::script`] supplies one
//! taxonomy; a second tenant supplies its own and shares every line below.

use super::bitset::TokenBitset;
use std::collections::BTreeMap;

/// The two token sets a selection implies: what to push down, and what to nudge
/// up.
///
/// **Two sets rather than one signed set**, because they carry different
/// weights and the sampler applies each bitset with its own. Folding them would
/// force one magnitude on suppression and encouragement, which are not the same
/// size: suppression fights a whole class, encouragement only has to break a
/// tie.
///
/// Disjoint by construction — [`Self::boost`] is drawn from the kept classes
/// and [`Self::suppress`] from everything else — so no token is ever pushed in
/// both directions.
#[derive(Clone, Debug)]
pub struct TokenBias {
    pub(super) suppress: TokenBitset,
    pub(super) boost: TokenBitset,
    pub(super) frame: TokenBitset,
}

impl TokenBias {
    /// Tokens of a class the caller did not select. Weighted POSITIVE by the
    /// sampler — subtracted from the logit.
    pub fn suppress(&self) -> &TokenBitset {
        &self.suppress
    }

    /// Tokens of the selected class. Weighted NEGATIVE — a boost.
    ///
    /// Empty when the selection is only the always-kept class, which is not a
    /// special case so much as the absence of one: that class is the floor
    /// every selection stands on, and boosting the floor pushes against
    /// whatever else was asked for.
    ///
    /// **Gentle, and the magnitude is not arbitrary.** Measurements at
    /// CJK/ASCII boundaries put ~86% of positions within 1.0 logit of flipping,
    /// so a weight around 0.5–1.0 decides the genuinely marginal cases and
    /// leaves anything the model is confident about alone. A large weight does
    /// not *prefer* a class, it forces one — and the model will produce it even
    /// where it has nothing to say in it.
    pub fn boost(&self) -> &TokenBitset {
        &self.boost
    }

    /// The declared protocol tokens, offered for a small lift of their own.
    ///
    /// **Compensation, not preference.** Boosting a class raises it against
    /// everything, the frame included — so a model steered toward Chinese finds
    /// `</think>` fractionally harder to reach than before the bias was
    /// applied. Weighting this set the same as [`Self::boost`] restores the
    /// frame to where it stood, rather than putting it anywhere new.
    ///
    /// **It is a separate set because the risk is separate.** These are
    /// terminators: `</think>`, `</tool_call>`, `<|im_end|>`. A lift that is
    /// merely generous makes a turn end early — a truncated answer, which is a
    /// worse failure than the drift it was correcting, and one that only shows
    /// up as "the model stopped mid-sentence". Give it its own weight so it can
    /// be tuned, or zeroed, without touching the language bias.
    ///
    /// Suggested: **equal to the boost weight when a boost is active, zero when
    /// it is not.** With no boost there is no tilt to correct, and lifting the
    /// frame against an unbiased distribution is a preference nobody asked for.
    ///
    /// Only the tokens the tokenizer *declares* special, not everything the
    /// markup heuristic exempted: `<div>` and `<Account` are exempt from
    /// suppression because that is free, but they are not protocol and have no
    /// business being encouraged.
    pub fn frame(&self) -> &TokenBitset {
        &self.frame
    }

    /// Nothing to apply.
    pub fn is_empty(&self) -> bool {
        self.suppress.is_empty() && self.boost.is_empty() && self.frame.is_empty()
    }
}

/// Which tokens of a vocabulary belong to which class.
///
/// Built once and then immutable.
#[derive(Clone, Debug)]
pub struct TokenClasses<C: Copy + Ord> {
    vocab: usize,
    sets: BTreeMap<C, TokenBitset>,
    exempt: TokenBitset,
    declared: TokenBitset,
}

impl<C: Copy + Ord> TokenClasses<C> {
    /// Classify a vocabulary.
    ///
    /// `tokens` yields `(id, decoded bytes)` — the bytes the detokenizer would
    /// emit, **not** a byte-level-BPE surface form. Classifying surface forms
    /// reads `ä¸­` as Latin and inverts the whole result.
    ///
    /// A token is **exempt** if its id is in `exempt_ids` or `is_exempt` says
    /// so. Exempt tokens join no class, which puts them in the same
    /// never-biased category as tokens no class claims. Both routes matter:
    /// the predicate catches what a token looks like, and the id list catches
    /// what only the tokenizer knows — a special token that happens to look
    /// like an ordinary word is invisible to any predicate.
    pub fn build<'a, I>(
        vocab: usize,
        tokens: I,
        exempt_ids: &[u32],
        is_exempt: &dyn Fn(&[u8]) -> bool,
        classify: &dyn Fn(&[u8]) -> Vec<C>,
    ) -> Self
    where
        I: IntoIterator<Item = (u32, &'a [u8])>,
    {
        let mut exempt = TokenBitset::new(vocab);
        // Kept apart from the heuristic's findings: these are the tokens the
        // tokenizer calls protocol, and they are the only ones worth
        // ENCOURAGING. What the markup rule catches is exempted because that is
        // free, not because it is structural.
        let mut declared = TokenBitset::new(vocab);
        for id in exempt_ids {
            if declared.insert(*id) {
                exempt.insert(*id);
            }
        }
        let mut sets: BTreeMap<C, TokenBitset> = BTreeMap::new();
        for (id, bytes) in tokens {
            if is_exempt(bytes) {
                exempt.insert(id);
            }
            if exempt.contains(id) {
                continue;
            }
            for c in classify(bytes) {
                sets.entry(c)
                    .or_insert_with(|| TokenBitset::new(vocab))
                    .insert(id);
            }
        }
        Self {
            vocab,
            sets,
            exempt,
            declared,
        }
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The tokens of one class.
    pub fn set_of(&self, c: C) -> Option<&TokenBitset> {
        self.sets.get(&c)
    }

    /// Tokens excluded from every class — declared specials and whatever the
    /// exemption predicate caught.
    pub fn exempt(&self) -> &TokenBitset {
        &self.exempt
    }

    /// The subset of [`Self::exempt`] the caller *declared*, rather than the
    /// predicate inferred. The protocol proper.
    pub fn declared(&self) -> &TokenBitset {
        &self.declared
    }

    /// Classes this vocabulary contains, ascending.
    pub fn classes(&self) -> impl Iterator<Item = C> + '_ {
        self.sets.keys().copied()
    }

    /// The raw set to suppress when keeping `keep`.
    ///
    /// Union of every class not kept, minus everything the kept classes claim.
    /// Exempt and unclassified tokens are in no set, so neither the union nor
    /// the subtraction can reach them.
    ///
    /// **Private, and it stays private.** A caller who forgets the always-kept
    /// class gets a set that is wrong in a way nothing reports — see
    /// [`Self::bias_for`], which cannot express the mistake.
    fn suppression_for(&self, keep: &[C]) -> TokenBitset {
        let mut out = TokenBitset::new(self.vocab);
        for (c, set) in self.sets.iter() {
            if !keep.contains(c) {
                out.union_with(set);
            }
        }
        // Take back everything the kept classes claim. Without this a token
        // holding two classes would be suppressed for carrying the one the
        // caller did not ask for, even though it also carries the one they did.
        for k in keep {
            if let Some(set) = self.sets.get(k) {
                out.subtract(set);
            }
        }
        out
    }

    /// The bias for selecting `select`, with `always` kept unconditionally.
    ///
    /// `always` is the class a caller can never afford to lose — for scripts,
    /// the one every protocol is written in. It is added to the kept set and
    /// **excluded from the boost**: it is the floor, not a preference, and
    /// encouraging it would work against whatever `select` asked for.
    pub fn bias_for(&self, select: &[C], always: &[C]) -> TokenBias {
        let mut keep = select.to_vec();
        for a in always {
            if !keep.contains(a) {
                keep.push(*a);
            }
        }
        let mut boost = TokenBitset::new(self.vocab);
        for c in select {
            if always.contains(c) {
                continue;
            }
            if let Some(set) = self.sets.get(c) {
                boost.union_with(set);
            }
        }
        TokenBias {
            suppress: self.suppression_for(&keep),
            boost,
            frame: self.declared.clone(),
        }
    }

    /// What stays unsuppressed under [`Self::bias_for`].
    ///
    /// A query, not a policy: materialising it costs a full pass and no bias is
    /// applied through it. It exists because the interesting assertions are
    /// about what survives, and "the tool-call closer is still sayable" reads
    /// better than a claim about a complement.
    pub fn sayable_for(&self, select: &[C], always: &[C]) -> TokenBitset {
        let bias = self.bias_for(select, always);
        let mut out = TokenBitset::new(self.vocab);
        for t in 0..self.vocab as u32 {
            if !bias.suppress.contains(t) {
                out.insert(t);
            }
        }
        out
    }
}
