//! A member's dial loop for one socket (§3.2): connect, handshake, run the
//! session; after any close that is not terminal, wait the backoff and dial
//! again. An admission resets the backoff. 4409 and 4426 stop this socket;
//! 4410 stops every socket of the app, which the loop reports for its owner
//! to act on.

use std::future::Future;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tokio::time::sleep;

use crate::backoff::Backoff;
use crate::close::{AfterClose, Close};
use crate::frame::{Admit, Hello};
use crate::handshake::{join, GroupKey, HandshakeError};
use crate::pipe::{PipeRx, PipeTx};
use crate::session::{start, Ended, Role, Session};

/// Opens the transport to one cluster.
pub trait Connect: Send {
    type Tx: PipeTx;
    type Rx: PipeRx;
    fn connect(&mut self) -> impl Future<Output = Result<(Self::Tx, Self::Rx), String>> + Send;
}

/// Why a dial loop stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stopped {
    /// This socket stops; the member's others carry on (4409, 4426).
    Socket(Close),
    /// The app stops on every cluster until the binary is replaced (4410).
    App(Close),
}

/// What one attempt came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attempt {
    ConnectFailed(String),
    Handshake(HandshakeError),
    /// Admitted, then the session ended.
    Session(Ended),
}

impl Attempt {
    fn after(&self) -> (AfterClose, Option<Close>) {
        let close = match self {
            Attempt::Handshake(HandshakeError::Closed(c)) | Attempt::Session(Ended::Closed(c)) => c,
            _ => return (AfterClose::Backoff, None),
        };
        (close.after(), Some(close.clone()))
    }
}

/// Dial until a terminal close. `hello` builds each attempt's `HELLO`;
/// `serve` runs an admitted session and returns how it ended; `attempted`
/// sees every attempt's outcome and how long the loop waits before the next,
/// `None` when it stops (for the loading page, §7.6).
pub async fn dial<C, H, S, F>(
    mut connector: C,
    key: GroupKey,
    mut hello: H,
    mut serve: S,
    mut attempted: impl FnMut(&Attempt, Option<Duration>) + Send,
) -> Stopped
where
    C: Connect,
    H: FnMut() -> Hello + Send,
    S: FnMut(Admit, Session) -> F + Send,
    F: Future<Output = Ended> + Send,
{
    let mut backoff = Backoff::new();
    let mut rng = StdRng::from_os_rng();
    loop {
        let attempt = match connector.connect().await {
            Err(e) => Attempt::ConnectFailed(e),
            Ok((mut tx, mut rx)) => match join(&mut tx, &mut rx, &key, hello()).await {
                Err(e) => Attempt::Handshake(e),
                Ok(admit) => {
                    backoff.reset();
                    let session = start(Role::Member, tx, rx);
                    Attempt::Session(serve(admit, session).await)
                }
            },
        };
        let (after, close) = attempt.after();
        let wait = backoff.after(after, &mut rng);
        attempted(&attempt, wait);
        match wait {
            Some(wait) => sleep(wait).await,
            None => {
                let close = close.expect("only a close stops a socket");
                return match after {
                    AfterClose::StopApp => Stopped::App(close),
                    _ => Stopped::Socket(close),
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use tokio::time::Instant;

    use super::*;
    use crate::close::CloseCode;
    use crate::frame::BuildId;
    use crate::handshake::{admit, PROTO};
    use crate::pipe::memory::{pair, MemoryRx, MemoryTx};

    fn key() -> GroupKey {
        GroupKey::from_bytes([1; 32])
    }

    fn hello() -> Hello {
        Hello {
            proto: PROTO,
            app: "npcd".into(),
            node: 3,
            host: String::new(),
            build: BuildId {
                count: 1,
                hash: String::new(),
                local: String::new(),
            },
            rejected: false,
            model: None,
            power: 0,
            repos: vec![],
            now_ms: 0,
            nonce: [0; 32],
        }
    }

    fn admission() -> Admit {
        Admit {
            session: 1,
            members: vec![],
            claims: vec![],
            floor_count: 1,
            floor_hash: String::new(),
            routes_wanted: false,
        }
    }

    /// What the fake gateway does with each connection, in turn.
    #[derive(Clone)]
    enum Gate {
        Unreachable,
        Close(CloseCode),
        Admit,
    }

    struct Script {
        gates: Arc<Mutex<VecDeque<Gate>>>,
        dialled: Arc<Mutex<Vec<Instant>>>,
    }

    impl Connect for Script {
        type Tx = MemoryTx;
        type Rx = MemoryRx;
        async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
            self.dialled.lock().unwrap().push(Instant::now());
            let gate = self.gates.lock().unwrap().pop_front().expect("scripted");
            if let Gate::Unreachable = gate {
                return Err("connection refused".into());
            }
            let (m, g) = pair(1 << 20);
            tokio::spawn(async move {
                let (mut tx, mut rx) = (g.tx, g.rx);
                let _ = admit(
                    &mut tx,
                    &mut rx,
                    &key(),
                    |_| {},
                    |_| match gate {
                        Gate::Close(code) => Err(Close::new(code, "scripted")),
                        _ => Ok(admission()),
                    },
                )
                .await;
                // An admitted session is dropped at once: the member sees the
                // socket go.
            });
            Ok((m.tx, m.rx))
        }
    }

    async fn run(gates: Vec<Gate>) -> (Stopped, Vec<Duration>, Vec<Attempt>) {
        let dialled = Arc::new(Mutex::new(Vec::new()));
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let script = Script {
            gates: Arc::new(Mutex::new(gates.into())),
            dialled: dialled.clone(),
        };
        let seen = attempts.clone();
        let stopped = dial(
            script,
            key(),
            hello,
            |_, s: Session| async move { s.ended.await.unwrap_or(Ended::Dropped) },
            move |a, _| seen.lock().unwrap().push(a.clone()),
        )
        .await;
        let times = dialled.lock().unwrap().clone();
        let gaps = times.windows(2).map(|w| w[1] - w[0]).collect();
        let attempts = attempts.lock().unwrap().clone();
        (stopped, gaps, attempts)
    }

    /// Failures back off 250, 500, 1000, 2000, 4000, 8000, 8000 ms, each drawn
    /// within `[d/2, d]`; an admission starts the schedule over; 4410 stops the
    /// app.
    #[tokio::test(start_paused = true)]
    async fn failures_back_off_and_an_admission_resets_it() {
        let mut gates = vec![Gate::Unreachable; 7];
        gates.push(Gate::Admit);
        gates.push(Gate::Unreachable);
        gates.push(Gate::Close(CloseCode::Superseded));
        let (stopped, gaps, attempts) = run(gates).await;
        let ms: Vec<u64> = gaps.iter().map(|d| d.as_millis() as u64).collect();
        let bounds = [250u64, 500, 1000, 2000, 4000, 8000, 8000];
        for (gap, d) in ms.iter().zip(bounds) {
            assert!(
                (d / 2..=d).contains(gap),
                "{gap} outside [{}, {d}] in {ms:?}",
                d / 2
            );
        }
        // After the admission (the 8th dial) the schedule starts over: the
        // socket drop waits within [125, 250], the next failure [250, 500].
        assert!((125..=250).contains(&ms[7]), "{ms:?}");
        assert!((250..=500).contains(&ms[8]), "{ms:?}");
        assert!(matches!(stopped, Stopped::App(c) if c.code == 4410));
        assert!(matches!(attempts[7], Attempt::Session(Ended::Dropped)));
    }

    #[tokio::test(start_paused = true)]
    async fn model_and_skew_refusals_wait_at_the_cap_and_4426_stops_the_socket() {
        let (stopped, gaps, _) = run(vec![
            Gate::Close(CloseCode::Model),
            Gate::Close(CloseCode::Skew),
            Gate::Close(CloseCode::Protocol),
        ])
        .await;
        for gap in gaps {
            assert!(gap >= Duration::from_secs(4) && gap <= Duration::from_secs(8));
        }
        assert!(matches!(stopped, Stopped::Socket(c) if c.code == 4426));
    }
}
