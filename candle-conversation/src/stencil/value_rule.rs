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
//! it would stop the decode dead, so a state is *finishable* when some token from
//! it closes the value or lands on another finishable state, and a mask admits
//! only tokens of that kind. Every state a masked walk stands on therefore has a
//! way out, in whole tokens of this vocabulary: a byte no token can reach is
//! simply never offered, and a state only passed through inside one token needs
//! no way out of its own, which is how a character of several bytes is written.
//! A rule whose start cannot finish is refused at build ([`ValueRule::new`]) and
//! the value decodes free. A value the executor would refuse is an error the
//! model reads; a value that cannot be written is a turn that never ends.
//!
//! **Built over the vocabulary in byte order.** A token sharing a prefix with the
//! last one resumes from that prefix's state, and a prefix the rule refuses skips
//! every token under it by binary search, so a narrow rule visits only the few
//! tokens it could admit.
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

/// A vocabulary's non-empty tokens in byte order.
type Sorted = Vec<(TokenId, Vec<u8>)>;

/// Each vocabulary's tokens in byte order, by fingerprint.
type SortedMap = HashMap<u64, Arc<Sorted>>;

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
    #[error("the vocabulary has no token that opens a value")]
    NoQuote,
    #[error("no value the pattern matches can be written in this vocabulary's tokens")]
    Unfinishable,
}

/// One byte of a walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// The byte is not allowed here.
    Dead,
    /// The byte closes the value.
    Closes,
    On(ValueState),
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
        Self::over(pattern, &sorted(vocab))
    }

    fn over(pattern: &str, tokens: &Sorted) -> Result<ValueRule, RuleError> {
        let (next, accepting) = states_of(pattern)?;
        let mut rule = ValueRule {
            pattern: pattern.to_string(),
            next,
            accepting,
            open_mask: AllowedSet::from_tokens(Vec::new()),
            spaced_mask: AllowedSet::from_tokens(Vec::new()),
            masks: Vec::new(),
        };
        if !tokens.iter().any(|(_, bytes)| bytes == b"\"") {
            return Err(RuleError::NoQuote);
        }
        let finishable = rule.finishable(tokens);
        if !finishable[0] {
            return Err(RuleError::Unfinishable);
        }
        let mask_at = |rule: &ValueRule, state| {
            let mut admitted = Vec::new();
            rule.scan(state, tokens, |token, land| {
                let keeps = match land {
                    Step::On(ValueState::Inside(s)) => finishable[s as usize],
                    // The opening space, whose only way on is the quote.
                    Step::On(_) => true,
                    Step::Closes => true,
                    Step::Dead => false,
                };
                if keeps {
                    admitted.push(token);
                }
            });
            AllowedSet::from_tokens(admitted)
        };
        rule.open_mask = mask_at(&rule, ValueState::Open);
        rule.spaced_mask = mask_at(&rule, ValueState::Spaced);
        rule.masks = (0..rule.next.len() as u32)
            .map(|state| mask_at(&rule, ValueState::Inside(state)))
            .collect();
        Ok(rule)
    }

    /// Which states some run of whole tokens can finish from. Each state's
    /// landings are read once, and the set grown over them until it stops.
    fn finishable(&self, tokens: &Sorted) -> Vec<bool> {
        let mut closes = vec![false; self.next.len()];
        let mut lands: Vec<Vec<u32>> = Vec::with_capacity(self.next.len());
        for state in 0..self.next.len() {
            let mut targets = Vec::new();
            self.scan(
                ValueState::Inside(state as u32),
                tokens,
                |_, land| match land {
                    Step::Closes => closes[state] = true,
                    Step::On(ValueState::Inside(s)) => targets.push(s),
                    _ => {}
                },
            );
            targets.sort_unstable();
            targets.dedup();
            lands.push(targets);
        }
        let mut done = closes;
        loop {
            let mut grew = false;
            for state in 0..done.len() {
                if !done[state] && lands[state].iter().any(|s| done[*s as usize]) {
                    done[state] = true;
                    grew = true;
                }
            }
            if !grew {
                return done;
            }
        }
    }

    /// Hand `visit` every token that survives from `start`, with where it lands.
    /// `tokens` are in byte order: one sharing a prefix with the last resumes at
    /// that prefix's state, and a refused or closing prefix decides every token
    /// under it at once.
    fn scan(&self, start: ValueState, tokens: &Sorted, mut visit: impl FnMut(TokenId, Step)) {
        let mut path = vec![start];
        let mut last: &[u8] = &[];
        let mut i = 0;
        'tokens: while i < tokens.len() {
            let bytes = tokens[i].1.as_slice();
            let shared = last.iter().zip(bytes).take_while(|(a, b)| a == b).count();
            path.truncate(shared.min(path.len() - 1) + 1);
            while path.len() <= bytes.len() {
                let depth = path.len() - 1;
                match self.step(path[depth], bytes[depth]) {
                    Step::On(next) => path.push(next),
                    settled => {
                        let prefix = &bytes[..=depth];
                        let under = tokens[i..].partition_point(|(_, b)| b.starts_with(prefix));
                        if settled == Step::Closes {
                            for (token, _) in &tokens[i..i + under] {
                                visit(*token, Step::Closes);
                            }
                        }
                        last = bytes;
                        i += under;
                        continue 'tokens;
                    }
                }
            }
            visit(tokens[i].0, Step::On(path[bytes.len()]));
            last = bytes;
            i += 1;
        }
    }

    /// Where one byte takes a walk from `state`.
    fn step(&self, state: ValueState, byte: u8) -> Step {
        match state {
            ValueState::Open if byte == b' ' => Step::On(ValueState::Spaced),
            ValueState::Open | ValueState::Spaced => match byte {
                b'"' => Step::On(ValueState::Inside(0)),
                _ => Step::Dead,
            },
            ValueState::Inside(s) => match byte {
                b'"' if self.accepting[s as usize] => Step::Closes,
                b'"' | b'\\' => Step::Dead,
                _ => match self.next[s as usize][byte as usize] {
                    DEAD => Step::Dead,
                    target => Step::On(ValueState::Inside(target)),
                },
            },
        }
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
        let rule = Arc::new(ValueRule::over(pattern, &shared_sorted(vocab))?);
        rules.rcu(|held| {
            let mut held = RuleMap::clone(held);
            held.insert(key.clone(), Arc::clone(&rule));
            held
        });
        Ok(rule)
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
        for &byte in bytes {
            at = match self.step(at, byte) {
                Step::On(next) => next,
                Step::Closes => return Some(None),
                Step::Dead => return None,
            };
        }
        (!bytes.is_empty()).then_some(Some(at))
    }

    /// `state` after a token of `bytes`. `None` when the value closed, or the
    /// token was not one the mask allows, which leaves the rest of it free.
    pub fn advance(&self, state: ValueState, bytes: &[u8]) -> Option<ValueState> {
        self.walk(state, bytes).flatten()
    }
}

/// `vocab`'s non-empty tokens in byte order.
fn sorted(vocab: &dyn Vocab) -> Sorted {
    let mut tokens = vocab.extensions(b"");
    tokens.retain(|(_, bytes)| !bytes.is_empty());
    tokens.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    tokens
}

/// The process's one [`sorted`] copy of `vocab`, swapped in whole.
fn shared_sorted(vocab: &dyn Vocab) -> Arc<Sorted> {
    static SORTED: OnceLock<ArcSwap<SortedMap>> = OnceLock::new();
    let held = SORTED.get_or_init(ArcSwap::default);
    let key = vocab.fingerprint();
    if let Some(tokens) = held.load().get(&key) {
        return Arc::clone(tokens);
    }
    let tokens = Arc::new(sorted(vocab));
    held.rcu(|map| {
        let mut map = SortedMap::clone(map);
        map.insert(key, Arc::clone(&tokens));
        map
    });
    tokens
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

    /// A byte-level vocabulary with `drop`'s tokens gone and `add`'s present.
    struct Edited {
        base: TestVocab,
        drop: &'static [u8],
        add: Vec<(TokenId, Vec<u8>)>,
    }

    impl Vocab for Edited {
        fn encode(&self, text: &str) -> Vec<TokenId> {
            self.base.encode(text)
        }
        fn token_bytes(&self, token: TokenId) -> Vec<u8> {
            self.base.token_bytes(token)
        }
        fn eos(&self) -> TokenId {
            self.base.eos()
        }
        fn end_tokens(&self) -> Vec<TokenId> {
            self.base.end_tokens()
        }
        fn fingerprint(&self) -> u64 {
            self.base.fingerprint()
        }
        fn extensions(&self, prefix: &[u8]) -> Vec<(TokenId, Vec<u8>)> {
            let mut all = self.base.extensions(prefix);
            all.retain(|(_, bytes)| bytes.len() != 1 || !self.drop.contains(&bytes[0]));
            all.extend(self.add.iter().cloned());
            all
        }
    }

    /// **A rule no value of which can be written is refused**; one with another
    /// way through simply never offers the byte it cannot finish after.
    #[test]
    fn a_rule_is_refused_only_when_no_value_can_be_written() {
        let no_z = || Edited {
            base: TestVocab::new(),
            drop: b"z",
            add: vec![],
        };
        let refused = ValueRule::new("az", &no_z()).map(|_| ());
        assert_eq!(refused.unwrap_err(), RuleError::Unfinishable);

        let vocab = no_z();
        let rule = ValueRule::new("az|b", &vocab).unwrap();
        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        assert!(rule.mask(start).contains(id(&vocab.base, "b")));
        assert!(
            !rule.mask(start).contains(id(&vocab.base, "a")),
            "an `a` here would strand the turn"
        );
    }

    /// **A negated class is written in whole tokens**: a lead byte with no token
    /// of its own is reached inside a token that carries the whole character,
    /// and its lone continuation is never offered. This refused every `[^"]`
    /// rule on the live vocabulary, leaving those values free.
    #[test]
    fn a_character_of_several_bytes_needs_no_token_per_byte() {
        let high: Vec<u8> = (0x80..=0xff).collect();
        let vocab = Edited {
            base: TestVocab::new(),
            drop: high.leak(),
            add: vec![(9_000, "é".as_bytes().to_vec())],
        };
        let rule = ValueRule::new(r#"[^"]*\.png"#, &vocab).unwrap();
        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        assert!(rule.mask(start).contains(9_000));
        let on = rule.advance(start, "é".as_bytes()).unwrap();
        assert!(
            !rule.mask(on).contains(id(&vocab.base, "\"")),
            "not yet .png"
        );
        assert!(rule.mask(on).contains(id(&vocab.base, ".")));
    }

    /// **The sorted scan admits exactly what a token-by-token walk does**, on
    /// tokens that share prefixes, close part-way, and run past the end.
    #[test]
    fn the_scan_agrees_with_the_walk() {
        let tokens: Vec<&[u8]> = vec![
            b"a", b"ab", b"abc", b"ab\"", b"ab\",", b"b", b"ba", b"\"", b"\",", b" ", b" \"",
            b"\\", b"c",
        ];
        let mut sorted: Sorted = tokens
            .iter()
            .enumerate()
            .map(|(i, b)| (i as TokenId, b.to_vec()))
            .collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));
        for pattern in ["ab", "a|ab", "[ab]+", "a*b"] {
            let rule = ValueRule::over(pattern, &sorted).unwrap();
            let mut starts = vec![ValueState::Open, ValueState::Spaced];
            starts.extend((0..rule.next.len() as u32).map(ValueState::Inside));
            for start in starts {
                let mut scanned = Vec::new();
                rule.scan(start, &sorted, |token, _| scanned.push(token));
                scanned.sort_unstable();
                let mut walked: Vec<TokenId> = sorted
                    .iter()
                    .filter(|(_, bytes)| rule.walk(start, bytes).is_some())
                    .map(|(token, _)| *token)
                    .collect();
                walked.sort_unstable();
                assert_eq!(scanned, walked, "{pattern} from {start:?}");
            }
        }
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
