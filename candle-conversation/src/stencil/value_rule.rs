//! What a string value may spell: a regular language held to under the mask.
//!
//! A string span is otherwise free text — the grammar writes around it and the
//! model writes anything inside. A [`ValueRule`] holds the span to a pattern one
//! token at a time: the value is a walk through an automaton, and each step's
//! mask is the tokens that keep the walk alive.
//!
//! **Two automata, one walk.** A pattern's rule ([`ValueRule::new`],
//! [`ValueRule::shared`]) compiles a regular expression, and works out every
//! state's mask when it is built: a turn's only cost is an index, and no decode
//! ever waits on anything. A held set's rule ([`ValueRule::of_set`]) walks the
//! set's byte trie ([`ValueSet`]) and makes a node's mask the first time a walk
//! stands on it, kept for the rule's life — a set of any size, since only the
//! nodes walked are ever made (`docs/stencil_tree.md` §9.2.1).
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
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use arc_swap::ArcSwap;
use regex_automata::dfa::{dense, Automaton};
use regex_automata::nfa::thompson;
use regex_automata::util::start::Config;
use regex_automata::util::syntax;
use regex_automata::{Anchored, MatchKind};

use super::mask::AllowedSet;
use super::value_set::ValueSet;
use super::vocab::{TokenId, Vocab};

/// The rules built so far, by vocabulary and pattern.
type RuleMap = HashMap<(u64, String), Arc<ValueRule>>;

/// The set rules held anywhere, by vocabulary and the set's digest and size.
type SetRules = HashMap<(u64, u64, usize), Weak<ValueRule>>;

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

/// What a rule walks, and when its masks are made.
#[derive(Debug)]
enum Machine {
    /// A pattern's automaton, every state's mask made at build.
    Pattern {
        /// Each state's successor by byte, [`DEAD`] where the walk ends.
        next: Vec<[u32; 256]>,
        /// Whether a value may end at each state.
        accepting: Vec<bool>,
        /// The tokens allowed at each state.
        masks: Vec<AllowedSet>,
    },
    /// A held set's trie, each node's mask made the first time a walk stands
    /// on it.
    Set {
        set: Arc<ValueSet>,
        /// What closes a value: its quote when `None`; else one of these
        /// strings, written after a value of the set ([`ValueRule::closed_by`]).
        /// Their trie's nodes follow the set's in the state numbering.
        closes: Option<Arc<ValueSet>>,
        /// Which states a walk can finish from: the set's nodes, then the
        /// closing strings'.
        finishable: Vec<bool>,
        /// The vocabulary a mask is scanned over.
        tokens: Arc<Sorted>,
        /// The masks made so far, by node.
        masks: Mutex<HashMap<u32, AllowedSet>>,
        /// This set's rule for each way a raw value is closed, made the first
        /// time a span closed that way stands on it.
        forms: Mutex<HashMap<Vec<String>, Arc<ValueRule>>>,
    },
}

/// One regular language over a value's bytes, with every step's mask.
#[derive(Debug)]
pub struct ValueRule {
    pattern: String,
    machine: Machine,
    /// The tokens allowed before the opening quote.
    open_mask: AllowedSet,
    /// The tokens allowed after the opening space.
    spaced_mask: AllowedSet,
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
            machine: Machine::Pattern {
                next,
                accepting,
                masks: Vec::new(),
            },
            open_mask: AllowedSet::from_tokens(Vec::new()),
            spaced_mask: AllowedSet::from_tokens(Vec::new()),
        };
        if !tokens.iter().any(|(_, bytes)| bytes == b"\"") {
            return Err(RuleError::NoQuote);
        }
        let finishable = rule.pattern_finishable(tokens);
        if !finishable[0] {
            return Err(RuleError::Unfinishable);
        }
        rule.open_mask = rule.mask_over(ValueState::Open, tokens, &finishable);
        rule.spaced_mask = rule.mask_over(ValueState::Spaced, tokens, &finishable);
        let states = rule.states();
        let built: Vec<AllowedSet> = (0..states as u32)
            .map(|state| rule.mask_over(ValueState::Inside(state), tokens, &finishable))
            .collect();
        if let Machine::Pattern { masks, .. } = &mut rule.machine {
            *masks = built;
        }
        Ok(rule)
    }

    /// The rule holding a value to one of `values`, over `vocab`
    /// (`docs/stencil_tree.md` §9.2.1): their byte trie, each node's mask made
    /// the first time a walk stands on it. One rule per set while anything
    /// holds it — a second call with the same values, from any caller, is the
    /// same rule, and its masks are shared. `Err` when no value of the set can
    /// be written in this vocabulary: its caller leaves the field to its own
    /// rule.
    pub fn of_set<S: AsRef<str>>(
        values: &[S],
        vocab: &dyn Vocab,
    ) -> Result<Arc<ValueRule>, RuleError> {
        static RULES: OnceLock<Mutex<SetRules>> = OnceLock::new();
        let (digest, len) = ValueSet::key_of(values);
        let key = (vocab.fingerprint(), digest, len);
        let rules = RULES.get_or_init(Mutex::default);
        if let Some(rule) = rules
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .and_then(Weak::upgrade)
        {
            return Ok(rule);
        }
        let rule = Arc::new(ValueRule::over_set(
            Arc::new(ValueSet::of(values)),
            shared_sorted(vocab),
        )?);
        let mut held = rules.lock().unwrap_or_else(PoisonError::into_inner);
        held.retain(|_, rule| rule.strong_count() > 0);
        held.insert(key, Arc::downgrade(&rule));
        Ok(rule)
    }

    fn over_set(set: Arc<ValueSet>, tokens: Arc<Sorted>) -> Result<ValueRule, RuleError> {
        Self::over_closed(set, None, tokens)
    }

    /// A rule over `set`, its values closed by their quote when `closes` is
    /// `None`, else by one of `closes` written after them.
    fn over_closed(
        set: Arc<ValueSet>,
        closes: Option<Arc<ValueSet>>,
        tokens: Arc<Sorted>,
    ) -> Result<ValueRule, RuleError> {
        if set.is_empty() {
            return Err(RuleError::Empty);
        }
        let mut single = [false; 256];
        for (_, bytes) in tokens.iter() {
            if let [byte] = bytes.as_slice() {
                single[*byte as usize] = true;
            }
        }
        let finishable = match &closes {
            None => {
                if !single[b'"' as usize] {
                    return Err(RuleError::NoQuote);
                }
                set.finishable(&single, true)
            }
            Some(closes) => {
                let closing = closes.finishable_on_arrival(&single);
                let mut finishable = set.finishable(&single, closing[0]);
                finishable.extend(closing);
                finishable
            }
        };
        if !finishable[0] {
            return Err(RuleError::Unfinishable);
        }
        let mut rule = ValueRule {
            pattern: format!("one of {} values", set.len()),
            machine: Machine::Set {
                set,
                closes,
                finishable: finishable.clone(),
                tokens: Arc::clone(&tokens),
                masks: Mutex::default(),
                forms: Mutex::default(),
            },
            open_mask: AllowedSet::from_tokens(Vec::new()),
            spaced_mask: AllowedSet::from_tokens(Vec::new()),
        };
        rule.open_mask = rule.mask_over(ValueState::Open, &tokens, &finishable);
        rule.spaced_mask = rule.mask_over(ValueState::Spaced, &tokens, &finishable);
        Ok(rule)
    }

    /// The tokens allowed at `state`, scanned over `tokens`: those that close
    /// the value, or land on a state some run of whole tokens finishes from.
    fn mask_over(&self, state: ValueState, tokens: &Sorted, finishable: &[bool]) -> AllowedSet {
        let mut admitted = Vec::new();
        self.scan(state, tokens, |token, land| {
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
    }

    /// Which states of a pattern's automaton some run of whole tokens can
    /// finish from. Each state's landings are read once, and the set grown over
    /// them until it stops.
    fn pattern_finishable(&self, tokens: &Sorted) -> Vec<bool> {
        let states = self.states();
        let mut closes = vec![false; states];
        let mut lands: Vec<Vec<u32>> = Vec::with_capacity(states);
        for (state, closes) in closes.iter_mut().enumerate() {
            let mut targets = Vec::new();
            self.scan(
                ValueState::Inside(state as u32),
                tokens,
                |_, land| match land {
                    Step::Closes => *closes = true,
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
        if let (
            Machine::Set {
                set,
                closes: Some(closes),
                ..
            },
            ValueState::Inside(s),
        ) = (&self.machine, state)
        {
            return Self::raw_step(set, closes, s, byte);
        }
        match state {
            ValueState::Open if byte == b' ' => Step::On(ValueState::Spaced),
            ValueState::Open | ValueState::Spaced => match byte {
                b'"' => Step::On(ValueState::Inside(0)),
                _ => Step::Dead,
            },
            ValueState::Inside(s) => match byte {
                b'"' if self.accepting(s) => Step::Closes,
                b'"' | b'\\' => Step::Dead,
                _ => match self.next(s, byte) {
                    None => Step::Dead,
                    Some(target) => Step::On(ValueState::Inside(target)),
                },
            },
        }
    }

    /// One byte of a raw value closed by one of `closes`: state `s` is a node
    /// of `set` below its node count, and of `closes` past it. A value of the
    /// set that has ended goes on into a closing string when the byte begins
    /// one, so a value running on into a longer one through `<` is not
    /// written: no path, anchor or range has one.
    fn raw_step(set: &ValueSet, closes: &ValueSet, s: u32, byte: u8) -> Step {
        let base = set.nodes() as u32;
        let closing = |c: u32| match closes.step(c, byte) {
            Some(next) if closes.accepting(next) => Step::Closes,
            Some(next) => Step::On(ValueState::Inside(base + next)),
            None => Step::Dead,
        };
        if s >= base {
            return closing(s - base);
        }
        if set.accepting(s) {
            let found = closing(0);
            if found != Step::Dead {
                return found;
            }
        }
        match set.step(s, byte) {
            Some(next) => Step::On(ValueState::Inside(next)),
            None => Step::Dead,
        }
    }

    /// This set's rule for a raw value, a function block's, which has no
    /// quotes: closed by `marker` and then one of `tails`, or by `marker`
    /// alone when there are none (`Terminator::Until`). Each value of the set
    /// can be written, and after it only what closes it, so the span's
    /// terminator fires on the byte the rule closes on. Made once per way of
    /// closing and kept. `None` for a pattern's rule, which holds a quoted
    /// value only, and when no value can be closed in this vocabulary.
    pub fn closed_by(&self, marker: &str, tails: &[&str]) -> Option<Arc<ValueRule>> {
        let Machine::Set {
            set,
            closes: None,
            tokens,
            forms,
            ..
        } = &self.machine
        else {
            return None;
        };
        let strings: Vec<String> = match tails {
            [] => vec![marker.to_string()],
            tails => tails.iter().map(|tail| format!("{marker}{tail}")).collect(),
        };
        let mut forms = forms.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(form) = forms.get(&strings) {
            return Some(Arc::clone(form));
        }
        let literal: Vec<&str> = strings.iter().map(String::as_str).collect();
        let closes = Arc::new(ValueSet::literal(&literal));
        let form = Self::over_closed(Arc::clone(set), Some(closes), Arc::clone(tokens))
            .map_err(|e| tracing::debug!("a held set has no raw form closed by {marker:?}: {e}"))
            .ok()?;
        let form = Arc::new(form);
        forms.insert(strings, Arc::clone(&form));
        Some(form)
    }

    /// Whether a value may end at automaton state `s`.
    fn accepting(&self, s: u32) -> bool {
        match &self.machine {
            Machine::Pattern { accepting, .. } => accepting[s as usize],
            Machine::Set { set, .. } => set.accepting(s),
        }
    }

    /// The state `byte` leads to from `s`, when one does.
    fn next(&self, s: u32, byte: u8) -> Option<u32> {
        match &self.machine {
            Machine::Pattern { next, .. } => match next[s as usize][byte as usize] {
                DEAD => None,
                target => Some(target),
            },
            Machine::Set { set, .. } => set.step(s, byte),
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

    /// The pattern this rule holds to; a held set's rule names how many values
    /// it holds.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// How many tokens the widest of this rule's masks allows: what a step
    /// under it sends to the device. A held set's counts the masks made so far.
    pub fn widest_mask(&self) -> usize {
        match &self.machine {
            Machine::Pattern { masks, .. } => masks.iter().map(AllowedSet::len).max(),
            Machine::Set { masks, .. } => masks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .values()
                .map(AllowedSet::len)
                .max(),
        }
        .unwrap_or(0)
    }

    /// How many states the rule's automaton has: a held set's, the nodes of
    /// its trie.
    pub fn states(&self) -> usize {
        match &self.machine {
            Machine::Pattern { next, .. } => next.len(),
            Machine::Set { set, closes, .. } => {
                set.nodes() + closes.as_ref().map_or(0, |c| c.nodes())
            }
        }
    }

    /// How many masks a held set's rule has made so far; a pattern's makes all
    /// of them at build.
    pub fn masks_made(&self) -> usize {
        match &self.machine {
            Machine::Pattern { masks, .. } => masks.len(),
            Machine::Set { masks, .. } => {
                masks.lock().unwrap_or_else(PoisonError::into_inner).len()
            }
        }
    }

    /// The mask at `state`. A held set's is made here the first time a walk
    /// stands on its node, and kept.
    pub fn mask(&self, state: ValueState) -> AllowedSet {
        let s = match state {
            ValueState::Open => return self.open_mask.clone(),
            ValueState::Spaced => return self.spaced_mask.clone(),
            ValueState::Inside(s) => s,
        };
        match &self.machine {
            Machine::Pattern { masks, .. } => masks[s as usize].clone(),
            Machine::Set {
                finishable,
                tokens,
                masks,
                ..
            } => {
                if let Some(made) = masks.lock().unwrap_or_else(PoisonError::into_inner).get(&s) {
                    return made.clone();
                }
                let made = self.mask_over(state, tokens, finishable);
                masks
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(s)
                    .or_insert(made)
                    .clone()
            }
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

/// `vocab`'s tokens a value may be written in, in byte order. An end token's
/// text would pass any open class, and would end the turn inside the call.
fn sorted(vocab: &dyn Vocab) -> Sorted {
    let mut tokens = vocab.extensions(b"");
    let ends = vocab.end_tokens();
    tokens.retain(|(token, bytes)| !bytes.is_empty() && !ends.contains(token));
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
            for state in 0..rule.states() as u32 {
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

    /// **An end token is never part of a value**: its text passes an open class
    /// byte by byte, and sampling it would end the turn inside the call.
    #[test]
    fn an_end_token_is_never_admitted() {
        let vocab = TestVocab::new().with_special("<end>", 1000).with_end(1000);
        let rule = ValueRule::new(r#"[^"]*"#, &vocab).unwrap();
        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        assert!(!rule.mask(start).contains(1000));
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
            starts.extend((0..rule.states() as u32).map(ValueState::Inside));
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

    /// **A held set is walked a token at a time, its masks made as the walk
    /// reaches them**: each step offers exactly the bytes that continue some
    /// value, the quote only where one ends, and a node's mask is made once.
    #[test]
    fn a_set_rule_holds_each_step_to_its_values() {
        let vocab = TestVocab::new();
        let rule = ValueRule::of_set(&["src/lib.rs", "src/main.rs", "README.md"], &vocab).unwrap();
        assert_eq!(rule.pattern(), "one of 3 values");
        assert!(rule.mask(ValueState::Open).contains(id(&vocab, "\"")));
        assert_eq!(rule.masks_made(), 0, "no node walked yet");

        let start = rule.advance(ValueState::Open, b"\"").unwrap();
        let first = rule.mask(start);
        assert!(first.contains(id(&vocab, "s")) && first.contains(id(&vocab, "R")));
        assert!(!first.contains(id(&vocab, "c")), "no value begins with `c`");
        assert!(
            !first.contains(id(&vocab, "\"")),
            "the empty value is not one"
        );
        assert_eq!(rule.masks_made(), 1);
        rule.mask(start);
        assert_eq!(rule.masks_made(), 1, "made once, then kept");

        let dir = b"src/"
            .iter()
            .try_fold(start, |at, b| rule.advance(at, &[*b]))
            .unwrap();
        let next = rule.mask(dir);
        assert!(next.contains(id(&vocab, "l")) && next.contains(id(&vocab, "m")));
        assert!(
            !next.contains(id(&vocab, "t")),
            "`src/turn_layout.rs` is no value"
        );
        let whole = b"lib.rs"
            .iter()
            .try_fold(dir, |at, b| rule.advance(at, &[*b]))
            .unwrap();
        assert!(
            rule.mask(whole).contains(id(&vocab, "\"")),
            "a value ends here"
        );
        assert_eq!(rule.advance(whole, b"\""), None, "closed");
    }

    /// **One rule per set while anything holds it**: the same values, in any
    /// order, are the same rule and share its masks; once let go, a set is
    /// built afresh.
    #[test]
    fn the_same_set_is_one_rule_while_held() {
        let vocab = TestVocab::new();
        let a = ValueRule::of_set(&["one-rule/b.rs", "one-rule/a.rs"], &vocab).unwrap();
        let b = ValueRule::of_set(&["one-rule/a.rs", "one-rule/b.rs", "one-rule/a.rs"], &vocab)
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let other = ValueRule::of_set(&["one-rule/a.rs"], &vocab).unwrap();
        assert!(!Arc::ptr_eq(&a, &other));
        let start = a.advance(ValueState::Open, b"\"").unwrap();
        a.mask(start);
        assert_eq!(b.masks_made(), 1, "one rule, one set of masks");
        drop((a, b));
        let again = ValueRule::of_set(&["one-rule/a.rs", "one-rule/b.rs"], &vocab).unwrap();
        assert_eq!(again.masks_made(), 0, "built afresh");
    }

    /// **A set none of whose values can be written is refused**, and the
    /// field keeps its own rule.
    #[test]
    fn an_unwritable_set_is_refused() {
        let vocab = TestVocab::new();
        let refused = |values: &[&str]| ValueRule::of_set(values, &vocab).map(|_| ()).unwrap_err();
        assert_eq!(refused(&[]), RuleError::Empty);
        assert_eq!(
            refused(&["a\"b"]),
            RuleError::Empty,
            "no value it could write"
        );
    }

    /// **A set rule's scan admits exactly what a token-by-token walk does**, as
    /// a pattern's does, on tokens that share prefixes, close part-way, and run
    /// past the end.
    #[test]
    fn a_set_rules_scan_agrees_with_the_walk() {
        let tokens: Vec<&[u8]> = vec![
            b"a", b"ab", b"abc", b"ab\"", b"ab\",", b"b", b"ba", b"\"", b"\",", b" ", b" \"",
            b"\\", b"c", b"/", b"a/", b"a/b",
        ];
        let mut sorted: Sorted = tokens
            .iter()
            .enumerate()
            .map(|(i, b)| (i as TokenId, b.to_vec()))
            .collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));
        let sorted = Arc::new(sorted);
        for values in [
            vec!["ab"],
            vec!["a", "ab"],
            vec!["a/b", "ab", "c"],
            vec!["", "b"],
        ] {
            let rule =
                ValueRule::over_set(Arc::new(ValueSet::of(&values)), Arc::clone(&sorted)).unwrap();
            let mut starts = vec![ValueState::Open, ValueState::Spaced];
            starts.extend((0..rule.states() as u32).map(ValueState::Inside));
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
                assert_eq!(scanned, walked, "{values:?} from {start:?}");
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

    /// **A raw value is held to the set and then to what closes it**: a
    /// function block's value has no quotes, so its held rule takes the
    /// element's close — `</parameter>` and the tag after it — in the quote's
    /// place, and closes on that tag's last byte, where the span's terminator
    /// fires. `src/cancel.rs` is no value, even where `src` is one.
    #[test]
    fn a_raw_value_is_held_to_its_set_and_its_close() {
        let vocab = TestVocab::new();
        let quoted =
            ValueRule::of_set(&["candle-conversation/src/cancel.rs", "src"], &vocab).unwrap();
        let tails = ["\n<parameter=", "\n</function>"];
        let rule = quoted.closed_by("</parameter>", &tails).unwrap();
        assert!(
            Arc::ptr_eq(&rule, &quoted.closed_by("</parameter>", &tails).unwrap()),
            "one form per way of closing"
        );
        let start = ValueState::Inside(0);
        assert_eq!(
            rule.walk(
                start,
                b"candle-conversation/src/cancel.rs</parameter>\n<parameter="
            ),
            Some(None),
            "the value, then its close"
        );
        assert_eq!(
            rule.walk(start, b"src</parameter>\n</function>"),
            Some(None)
        );
        assert_eq!(rule.walk(start, b"src/cancel.rs"), None);
        assert_eq!(rule.walk(start, b"src</parameter>x"), None);
        assert_eq!(rule.walk(start, b"src\""), None, "no quote closes it");
        let src = rule.advance(start, b"src").unwrap();
        let at_end = rule.mask(src);
        assert!(at_end.contains(id(&vocab, "<")));
        assert!(!at_end.contains(id(&vocab, "\"")) && !at_end.contains(id(&vocab, "/")));
        let open = rule.advance(start, b"src</parameter>\n").unwrap();
        let tag = rule.mask(open);
        assert!(tag.contains(id(&vocab, "<")) && !tag.contains(id(&vocab, "\n")));
        let alone = quoted.closed_by("</parameter>", &[]).unwrap();
        assert_eq!(alone.walk(start, b"src</parameter>"), Some(None));
        let (_, pattern) = rule_arc("[a-z]+");
        assert!(pattern.closed_by("</parameter>", &tails).is_none());
    }

    fn rule_arc(pattern: &str) -> (TestVocab, Arc<ValueRule>) {
        let (vocab, rule) = rule(pattern);
        (vocab, Arc::new(rule))
    }
}
