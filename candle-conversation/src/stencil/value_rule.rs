//! What a string value may spell: a regular language held to under the mask.
//!
//! A string span is otherwise free text — the grammar writes around it and the
//! model writes anything inside. A [`ValueRule`] holds the span to a pattern one
//! token at a time: the value is a walk through an automaton, and each step's
//! mask is the tokens that keep the walk alive.
//!
//! **Compiled once, then only read.** Every state's mask is worked out when the
//! rule is built and the rule is immutable after, so a turn's only cost is an
//! index, and no decode ever waits on anything.
//!
//! **A rule can never leave a turn with nothing to say.** A mask with no token in
//! it would stop the decode dead, so a rule is built only if every state it can
//! reach can be finished in this vocabulary: from each one, every byte that keeps
//! the walk alive has a token of exactly that byte, and each accepting state has
//! the closing quote. Anything else is refused at build ([`ValueRule::new`]) and
//! the value decodes free. A value the executor would refuse is an error the
//! model reads; a value that cannot be written is a turn that never ends.
//!
//! **A value under a rule has no escapes.** A backslash would make the next quote
//! content rather than the close, which a walk over raw bytes cannot see, so no
//! token holding one is admitted. Every value a rule describes — an id, a ref, a
//! scheme and host — has no use for one.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;
use regex_automata::dfa::{dense, Automaton};
use regex_automata::nfa::thompson;
use regex_automata::util::start::Config;
use regex_automata::util::syntax;
use regex_automata::{Anchored, MatchKind};

use super::mask::AllowedSet;
use super::vocab::{TokenId, Vocab};

/// The rules built so far, by vocabulary and pattern.
type RuleMap = HashMap<(u64, String), Arc<ValueRule>>;

/// The most automaton states a rule may have: each one holds a mask.
const MAX_STATES: usize = 512;

/// A state with no way on.
const DEAD: u32 = u32::MAX;

/// Why a pattern cannot be a rule.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuleError {
    #[error("the pattern does not compile: {0}")]
    Pattern(String),
    #[error("the pattern needs more than {MAX_STATES} states")]
    TooLarge,
    #[error("the pattern matches nothing")]
    Empty,
    #[error("the vocabulary has no token for byte {byte:#04x}, which the pattern needs")]
    Unspellable { byte: u8 },
}

/// Where a walk stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueState {
    /// Before the value's opening quote.
    Open,
    /// Past the one space a value may open with: only the quote is next, so
    /// no run of whitespace can spend the span before the value starts.
    Spaced,
    /// Inside the value, at this automaton state.
    Inside(u32),
}

/// One regular language over a value's bytes, with every step's mask.
#[derive(Debug)]
pub struct ValueRule {
    pattern: String,
    /// Each state's successor by byte, [`DEAD`] where the walk ends.
    next: Vec<[u32; 256]>,
    /// Whether a value may end at each state.
    accepting: Vec<bool>,
    /// The tokens allowed before the opening quote.
    open_mask: AllowedSet,
    /// The tokens allowed after the opening space.
    spaced_mask: AllowedSet,
    /// The tokens allowed at each state.
    masks: Vec<AllowedSet>,
}

impl ValueRule {
    /// The rule for `pattern`, a whole-value regular expression, over `vocab`.
    /// `Err` when it cannot be held to in this vocabulary: its caller leaves the
    /// value free.
    pub fn new(pattern: &str, vocab: &dyn Vocab) -> Result<ValueRule, RuleError> {
        let (next, accepting) = states_of(pattern)?;
        let tokens = vocab.extensions(b"");
        let mut rule = ValueRule {
            pattern: pattern.to_string(),
            next,
            accepting,
            open_mask: AllowedSet::from_tokens(Vec::new()),
            spaced_mask: AllowedSet::from_tokens(Vec::new()),
            masks: Vec::new(),
        };
        rule.check_spellable(&tokens)?;
        let mask_at = |rule: &ValueRule, state| {
            let admitted = tokens
                .iter()
                .filter(|(_, bytes)| rule.walk(state, bytes).is_some())
                .map(|(token, _)| *token);
            AllowedSet::from_tokens(admitted.collect())
        };
        rule.open_mask = mask_at(&rule, ValueState::Open);
        rule.spaced_mask = mask_at(&rule, ValueState::Spaced);
        rule.masks = (0..rule.next.len() as u32)
            .map(|state| mask_at(&rule, ValueState::Inside(state)))
            .collect();
        Ok(rule)
    }

    /// The process's one rule for `pattern` over `vocab`. A mode's tree compiles
    /// the same patterns as the others', and a rule's masks are the expensive
    /// part: built once, and the map swapped whole, so no reader ever waits.
    pub fn shared(pattern: &str, vocab: &dyn Vocab) -> Result<Arc<ValueRule>, RuleError> {
        static RULES: OnceLock<ArcSwap<RuleMap>> = OnceLock::new();
        let rules = RULES.get_or_init(ArcSwap::default);
        let key = (vocab.fingerprint(), pattern.to_string());
        if let Some(rule) = rules.load().get(&key) {
            return Ok(Arc::clone(rule));
        }
        let rule = Arc::new(ValueRule::new(pattern, vocab)?);
        rules.rcu(|held| {
            let mut held = RuleMap::clone(held);
            held.insert(key.clone(), Arc::clone(&rule));
            held
        });
        Ok(rule)
    }

    /// Every state can be finished one byte at a time, which is what makes an
    /// empty mask unreachable whatever tokens the walk took to get there.
    fn check_spellable(&self, tokens: &[(TokenId, Vec<u8>)]) -> Result<(), RuleError> {
        let mut single = [false; 256];
        for (_, bytes) in tokens {
            if let [byte] = bytes.as_slice() {
                single[*byte as usize] = true;
            }
        }
        let need = |byte: u8| match single[byte as usize] {
            true => Ok(()),
            false => Err(RuleError::Unspellable { byte }),
        };
        need(b'"')?;
        for transitions in &self.next {
            for (byte, target) in transitions.iter().enumerate() {
                if *target != DEAD {
                    need(byte as u8)?;
                }
            }
        }
        Ok(())
    }

    /// The pattern this rule holds to.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// How many tokens the widest of this rule's masks allows: what a step
    /// under it sends to the device.
    pub fn widest_mask(&self) -> usize {
        self.masks.iter().map(AllowedSet::len).max().unwrap_or(0)
    }

    /// How many states the rule has.
    pub fn states(&self) -> usize {
        self.masks.len()
    }

    /// The mask at `state`.
    pub fn mask(&self, state: ValueState) -> &AllowedSet {
        match state {
            ValueState::Open => &self.open_mask,
            ValueState::Spaced => &self.spaced_mask,
            ValueState::Inside(s) => &self.masks[s as usize],
        }
    }

    /// Whether a token that ran `rest` past a key into this value's span keeps
    /// to the rule, from `start`.
    pub fn admits(&self, start: ValueState, rest: &[u8]) -> bool {
        self.walk(start, rest).is_some()
    }

    /// Where `bytes` take a walk from `state`: `Some(None)` when they close the
    /// value, `None` when they are not allowed there.
    fn walk(&self, state: ValueState, bytes: &[u8]) -> Option<Option<ValueState>> {
        let mut at = state;
        for (i, &byte) in bytes.iter().enumerate() {
            at = match at {
                ValueState::Open if byte == b' ' => ValueState::Spaced,
                ValueState::Open | ValueState::Spaced => match byte {
                    b'"' => ValueState::Inside(0),
                    _ => return None,
                },
                ValueState::Inside(s) => match byte {
                    b'"' => return self.accepting[s as usize].then_some(None),
                    b'\\' => return None,
                    _ => match self.next[s as usize][byte as usize] {
                        DEAD => return None,
                        target => ValueState::Inside(target),
                    },
                },
            };
            debug_assert!(i < bytes.len());
        }
        (!bytes.is_empty()).then_some(Some(at))
    }

    /// `state` after a token of `bytes`. `None` when the value closed, or the
    /// token was not one the mask allows, which leaves the rest of it free.
    pub fn advance(&self, state: ValueState, bytes: &[u8]) -> Option<ValueState> {
        self.walk(state, bytes).flatten()
    }
}

/// `pattern`'s states as a dense table, state 0 the start and every state alive.
fn states_of(pattern: &str) -> Result<(Vec<[u32; 256]>, Vec<bool>), RuleError> {
    let bad = |e: &dyn std::fmt::Display| RuleError::Pattern(e.to_string());
    // Anchored at the end as well: a DFA says a match one byte late, so an
    // open-ended one reads any byte after a match as a state with a way on.
    let whole = format!("^(?:{pattern})$");
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .minimize(true)
                .match_kind(MatchKind::All)
                .dfa_size_limit(Some(1 << 24)),
        )
        .syntax(syntax::Config::new().utf8(false).unicode(false))
        .thompson(thompson::Config::new().utf8(false))
        .build(&whole)
        .map_err(|e| bad(&e))?;
    let start = dfa
        .start_state(&Config::new().anchored(Anchored::Yes))
        .map_err(|e| bad(&e))?;
    let accepts = |s| dfa.is_match_state(dfa.next_eoi_state(s));

    // Every state the walk can reach, in discovery order so the start is 0.
    let mut index = HashMap::from([(start, 0u32)]);
    let mut order = vec![start];
    let mut next: Vec<[u32; 256]> = Vec::new();
    let mut at = 0;
    while at < order.len() {
        let mut row = [DEAD; 256];
        for byte in 0..=255u8 {
            let target = dfa.next_state(order[at], byte);
            if dfa.is_dead_state(target) || dfa.is_quit_state(target) {
                continue;
            }
            let id = *index.entry(target).or_insert_with(|| {
                order.push(target);
                (order.len() - 1) as u32
            });
            row[byte as usize] = id;
        }
        next.push(row);
        if order.len() > MAX_STATES {
            return Err(RuleError::TooLarge);
        }
        at += 1;
    }
    let accepting: Vec<bool> = order.iter().map(|&s| accepts(s)).collect();
    prune_stranded(next, accepting)
}

/// Take out the states no path leads on from to an end. A byte into one would
/// be a walk that can never close.
fn prune_stranded(
    mut next: Vec<[u32; 256]>,
    accepting: Vec<bool>,
) -> Result<(Vec<[u32; 256]>, Vec<bool>), RuleError> {
    let mut live = accepting.clone();
    loop {
        let mut grew = false;
        for state in 0..next.len() {
            let onward = |t: &u32| *t != DEAD && live[*t as usize];
            if !live[state] && next[state].iter().any(onward) {
                live[state] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    if !live[0] {
        return Err(RuleError::Empty);
    }
    for row in &mut next {
        for target in row.iter_mut() {
            if *target != DEAD && !live[*target as usize] {
                *target = DEAD;
            }
        }
    }
    Ok((next, accepting))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::vocab::TestVocab;

    fn id(vocab: &TestVocab, text: &str) -> TokenId {
        vocab.encode(text)[0]
    }

    fn rule(pattern: &str) -> (TestVocab, ValueRule) {
        let vocab = TestVocab::new();
        let rule = ValueRule::new(pattern, &vocab).unwrap();
        (vocab, rule)
    }

    /// **A value is walked a token at a time**, each mask holding only what
    /// keeps it alive, and the quote only where the value may end.
    #[test]
    fn a_walk_holds_each_step_to_the_pattern() {
        let (vocab, rule) = rule("[0-9a-f]{2}");
        let open = ValueState::Open;
        assert!(rule.mask(open).contains(id(&vocab, "\"")));
        assert!(!rule.mask(open).contains(id(&vocab, "a")), "a bare value");

        let start = rule.advance(open, b"\"").unwrap();
        assert!(rule.mask(start).contains(id(&vocab, "a")));
        assert!(!rule.mask(start).contains(id(&vocab, "g")));
        assert!(!rule.mask(start).contains(id(&vocab, "\"")), "too short");

        let one = rule.advance(start, b"a").unwrap();
        let two = rule.advance(one, b"f").unwrap();
        assert!(rule.mask(two).contains(id(&vocab, "\"")), "may close");
        assert!(!rule.mask(two).contains(id(&vocab, "0")), "too long");
        assert_eq!(rule.advance(two, b"\""), None, "closed");
    }

    /// **A token may open and close a value on its own**, and an empty value is
    /// allowed only where the pattern is.
    #[test]
    fn one_token_may_cover_a_whole_value() {
        let (_, rule) = rule("[a-z]+");
        assert!(rule.walk(ValueState::Open, b" \"abc\"").is_some());
        assert_eq!(rule.walk(ValueState::Open, b"\"\""), None, "empty refused");
        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        assert!(
            rule.walk(start, b"ab\",").is_some(),
            "what follows a close is the grammar's"
        );

        let (_, blank) = self::rule("[a-z]*");
        assert_eq!(blank.walk(ValueState::Open, b"\"\""), Some(None));
    }

    /// **A backslash is never admitted**: it would turn the next quote into
    /// content, and the walk would read the value as closed.
    #[test]
    fn an_escape_is_never_admitted() {
        let (vocab, rule) = rule(r"[a-z\\]+");
        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        assert!(!rule.mask(start).contains(id(&vocab, "\\")));
    }

    /// **A state that cannot reach an end is cut off**, so no mask leads into
    /// one. Every mask of every rule has a token in it.
    #[test]
    fn no_state_has_an_empty_mask() {
        for pattern in ["a|bc", "[0-9]{4}", "(ab)+", "x*y", "https?://[a-z]+"] {
            let (_, rule) = rule(pattern);
            assert!(!rule.mask(ValueState::Open).is_empty(), "{pattern}");
            for state in 0..rule.next.len() as u32 {
                let mask = rule.mask(ValueState::Inside(state));
                assert!(!mask.is_empty(), "{pattern} state {state}");
            }
        }
    }

    /// **A pattern that cannot be held to is refused, not half applied**.
    #[test]
    fn what_cannot_be_held_to_is_refused() {
        let vocab = TestVocab::new();
        let refused = |pattern| ValueRule::new(pattern, &vocab).map(|_| ()).unwrap_err();
        assert!(matches!(refused("("), RuleError::Pattern(_)));
        assert_eq!(refused("[a&&b]"), RuleError::Empty);
        assert_eq!(refused("[a-z]{1000}"), RuleError::TooLarge);
    }

    /// **A byte with no token of its own refuses the rule**, as the walk could
    /// reach a state only that byte leaves.
    #[test]
    fn an_unspellable_byte_refuses_the_rule() {
        struct NoZ(TestVocab);
        impl Vocab for NoZ {
            fn encode(&self, text: &str) -> Vec<TokenId> {
                self.0.encode(text)
            }
            fn token_bytes(&self, token: TokenId) -> Vec<u8> {
                self.0.token_bytes(token)
            }
            fn eos(&self) -> TokenId {
                self.0.eos()
            }
            fn end_tokens(&self) -> Vec<TokenId> {
                self.0.end_tokens()
            }
            fn fingerprint(&self) -> u64 {
                self.0.fingerprint()
            }
            fn extensions(&self, prefix: &[u8]) -> Vec<(TokenId, Vec<u8>)> {
                let mut all = self.0.extensions(prefix);
                all.retain(|(_, bytes)| bytes != b"z");
                all
            }
        }
        let refused = ValueRule::new("az", &NoZ(TestVocab::new())).map(|_| ());
        assert_eq!(refused.unwrap_err(), RuleError::Unspellable { byte: b'z' });
    }

    /// A multi-byte token is admitted whole or not at all.
    #[test]
    fn a_token_past_the_pattern_is_not_admitted() {
        let (_, rule) = rule("ab");
        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        assert!(rule.walk(start, b"ab").is_some());
        assert_eq!(rule.walk(start, b"abc"), None);
    }
}
