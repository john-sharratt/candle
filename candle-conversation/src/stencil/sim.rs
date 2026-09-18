//! A model-free driver for a [`StencilSession`].
//!
//! An [`Oracle`] supplies the token at each masked/free decode; the simulator
//! records the full action stream and the emitted token sequence (prefilled +
//! decoded) so tests can assert exact behaviour and re-parse the output as JSON
//! — all on CPU, no forward pass.

use std::sync::Arc;

use super::driver::{healing, Healed};
use super::error::WalkError;
use super::mask::AllowedSet;
use super::session::{Observe, StencilAction, StencilSession};
use super::tree::StencilTree;
use super::vocab::{TokenId, Vocab};

/// Picks the token to emit at each decode.
pub enum Oracle {
    /// Emit exactly these tokens, in order, one per decode.  Errors if it runs
    /// out before the session exits.
    Scripted(Vec<TokenId>),
    /// A policy: given the allowed set (`Some` at a branch, `None` in free text),
    /// return a token.  Must respect the mask at branches.
    Policy(Box<dyn FnMut(Option<&AllowedSet>) -> TokenId>),
}

impl Oracle {
    fn next(&mut self, allowed: Option<&AllowedSet>, step: usize) -> Result<TokenId, SimError> {
        match self {
            Oracle::Scripted(toks) => toks
                .get(step)
                .copied()
                .ok_or(SimError::ScriptExhausted { step }),
            Oracle::Policy(f) => Ok(f(allowed)),
        }
    }
}

/// A recorded simulation.
#[derive(Debug, Clone)]
pub struct SimRun {
    pub actions: Vec<StencilAction>,
    pub observes: Vec<Observe>,
    /// All tokens that entered the KV — prefilled static runs and decoded tokens,
    /// in order. A decoded token is committed as the decode loop commits it: a
    /// dropped token never enters, and a rewritten one enters as its
    /// re-tokenized bytes.
    pub tokens: Vec<TokenId>,
    /// Decoded tokens the session dropped.
    pub dropped: Vec<TokenId>,
    /// Total leftover bytes across span closes (a non-zero value means a close
    /// fell mid-token and the integration would heal).
    pub healed_bytes: usize,
    pub forced_closes: usize,
}

impl SimRun {
    /// The decoded text of the emitted token stream.
    pub fn text(&self, vocab: &dyn Vocab) -> String {
        String::from_utf8_lossy(&vocab.decode(&self.tokens)).into_owned()
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SimError {
    #[error("oracle script exhausted at decode step {step}")]
    ScriptExhausted { step: usize },
    #[error("walk error: {0}")]
    Walk(#[from] WalkError),
    #[error("exceeded {0} steps without exiting (runaway)")]
    Runaway(usize),
}

/// Drive `session` to completion with `oracle`, capping at `max_steps`.
pub fn simulate(
    tree: Arc<StencilTree>,
    vocab: &dyn Vocab,
    mut oracle: Oracle,
    max_steps: usize,
) -> Result<SimRun, SimError> {
    let mut session = StencilSession::new(tree);
    let mut run = SimRun {
        actions: Vec::new(),
        observes: Vec::new(),
        tokens: Vec::new(),
        dropped: Vec::new(),
        healed_bytes: 0,
        forced_closes: 0,
    };
    let mut step = 0usize;
    loop {
        if step > max_steps {
            return Err(SimError::Runaway(max_steps));
        }
        let action = session.next_action();
        run.actions.push(action.clone());
        match action {
            StencilAction::Prefill(toks) => {
                run.tokens.extend_from_slice(&toks);
            }
            StencilAction::MaskedDecode(set) => {
                let token = oracle.next(Some(&set), step)?;
                step += 1;
                let obs = commit(&mut session, vocab, &mut run, token)?;
                run.observes.push(obs);
            }
            StencilAction::FreeDecode { .. } => {
                let token = oracle.next(None, step)?;
                step += 1;
                let obs = commit(&mut session, vocab, &mut run, token)?;
                run.observes.push(obs);
                match obs {
                    Observe::SpanClosed { leftover } => run.healed_bytes += leftover,
                    Observe::SpanForcedClosed => run.forced_closes += 1,
                    _ => {}
                }
            }
            StencilAction::Exit => break,
        }
    }
    Ok(run)
}

/// Observe `token` and commit it the way the decode loop would.
fn commit(
    session: &mut StencilSession,
    vocab: &dyn Vocab,
    run: &mut SimRun,
    token: TokenId,
) -> Result<Observe, SimError> {
    let bytes = vocab.token_bytes(token);
    let obs = session.observe(token, &bytes)?;
    match healing(obs, session.take_rewrite(), &bytes) {
        Healed::No => run.tokens.push(token),
        Healed::Drop => run.dropped.push(token),
        Healed::Rewrite { bytes } => run
            .tokens
            .extend(vocab.encode(&String::from_utf8_lossy(&bytes))),
    }
    Ok(obs)
}

/// A convenience policy that, at a branch, always picks the lowest-id allowed
/// token (deterministic path enumeration), and in free text emits a fixed token.
pub fn lowest_arm_policy(free_token: TokenId) -> Oracle {
    Oracle::Policy(Box::new(move |allowed| match allowed {
        Some(set) => *set.tokens().first().expect("non-empty frontier"),
        None => free_token,
    }))
}
