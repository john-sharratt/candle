//! One call: draw now, authorize meanwhile, release on both
//! (`docs/guest_draw_orchestration.md` §2).
//!
//! A clearance in hand — given, cached, or an Admin's waiver — draws and
//! releases at once. With none, and a peer to ask, the draw is submitted, the
//! peer asked and the check run here, all at the same moment; the picture is
//! held until either check allows it, and a refusal from either before then
//! stands. With no peer the check runs here first. A picture leaves only
//! through [`Held::open`].

use std::pin::pin;
use std::time::{Duration, Instant};

use candle_conversation::guest::{GuestError, GuestImage, GuestOutcome, GuestRequest, GuestSink};
use tokio::time::sleep;

use super::cache::Cache;
use super::check::{local, ContentCheck};
use super::clearance::{Clearance, Held};
use super::draw::{GuestDraw, Outcome};
use super::peer::{PeerAuthorizer, PeerVerdict};
use super::token::{Authorization, Signer, IMAGE};
use crate::api::now_ms;
use crate::compliance::Denial;

/// A released picture, and the authorization that released it.
#[derive(Debug)]
pub struct Released {
    pub image: GuestImage,
    /// `None` for an Admin's waiver.
    pub authorization: Option<Authorization>,
}

/// Why a draw was not released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stopped {
    /// The prompt was declined; nothing drawn was kept.
    Declined(String),
    /// The prompt could not be checked; nothing drawn was kept.
    CheckUnavailable(String),
    /// The draw itself failed.
    Guest(GuestError),
}

/// What the orchestration needs.
pub struct Parts<'a, C, P, D> {
    pub check: &'a C,
    pub peer: &'a P,
    pub guest: &'a D,
    pub signer: &'a Signer,
    pub cache: &'a Cache,
}

/// Draw `request` and release it once cleared. `given` is a clearance the
/// caller brought, already verified; `authorizing` is called when the picture
/// is drawn and its verdict is still awaited.
pub async fn draw<C, P, D>(
    parts: Parts<'_, C, P, D>,
    request: GuestRequest,
    given: Option<Clearance>,
    sink: GuestSink,
    authorizing: impl FnOnce(),
) -> Result<Released, Stopped>
where
    C: ContentCheck,
    P: PeerAuthorizer,
    D: GuestDraw + Clone + Send + Sync + 'static,
{
    let started = Instant::now();
    let prompt = match &request {
        GuestRequest::Image(r) => r.prompt.clone(),
        _ => return Err(Stopped::Guest(GuestError::Refused("not a draw".into()))),
    };
    let ms = |at: Instant| at.duration_since(started).as_millis() as u64;
    // Refused moments ago: answered now, not drawn again for the same answer.
    // Asked ahead of the cache, so a refusal outweighs an allow kept for the
    // same prompt.
    if given.is_none() && parts.cache.refused(IMAGE, &prompt, now_ms()) {
        log("refused", 0, 0, ms(Instant::now()));
        return Err(Stopped::Declined(Denial::Refused.to_string()));
    }

    // A clearance in hand: draw and release. The cache is asked only when the
    // caller brought none.
    let (source, clearance) = match given {
        Some(c @ Clearance::AdminWaiver(_)) => ("waiver", Some(c)),
        Some(c) => ("given", Some(c)),
        None => (
            "cache",
            parts
                .cache
                .get(IMAGE, &prompt, now_ms())
                .map(Clearance::Authorized),
        ),
    };
    if let Some(clearance) = clearance {
        let submitted = parts.guest.submit(request, sink).map_err(Stopped::Guest)?;
        let held = image(submitted.outcome.await, &prompt)?;
        let drawn = ms(Instant::now());
        let out = release(held, clearance);
        log(source, drawn, 0, ms(Instant::now()));
        return out;
    }

    // No peer: check here first — nothing else can answer.
    if !parts.peer.available() {
        let a = authorized_here(&parts, &prompt).await?;
        let checked = ms(Instant::now());
        let submitted = parts.guest.submit(request, sink).map_err(Stopped::Guest)?;
        let held = image(submitted.outcome.await, &prompt)?;
        let drawn = ms(Instant::now());
        let out = release(held, Clearance::Authorized(a));
        log("local", drawn, checked, ms(Instant::now()));
        return out;
    }

    // A peer: draw, ask the peer and check here, all at once. The picture is
    // released once drawn and either check allows it; a refusal from either
    // stands; a check that fails leaves the other to decide. The check here
    // waits while the drain holds this member's engine and finishes after it;
    // the peer's does not wait, but queues behind its own member's work — so
    // whichever is quicker that moment decides.
    let submitted = parts.guest.submit(request, sink).map_err(Stopped::Guest)?;
    let seq = submitted.seq;
    let mut outcome: Outcome = submitted.outcome;
    let mut peer = pin!(parts.peer.authorize(IMAGE, &prompt));
    let mut here = pin!(authorized_here(&parts, &prompt));
    let mut authorizing = Some(authorizing);
    let mut held: Option<Held<GuestImage>> = None;
    let mut cleared: Option<(&'static str, Clearance)> = None;
    let (mut peer_done, mut here_done) = (false, false);
    let (mut drawn, mut authorized) = (0, 0);
    let mut failed: Vec<String> = Vec::new();
    loop {
        if held.is_some() && cleared.is_some() {
            break;
        }
        if cleared.is_none() && peer_done && here_done {
            // Neither check could answer: nothing drawn is kept.
            withdraw_undrawn(parts.guest, held.is_some(), seq, outcome);
            log("none", drawn, 0, ms(Instant::now()));
            return Err(Stopped::CheckUnavailable(failed.join("; ")));
        }
        tokio::select! {
            o = &mut outcome, if held.is_none() => {
                drawn = ms(Instant::now());
                held = Some(image(o, &prompt)?);
                if cleared.is_none() {
                    if let Some(said) = authorizing.take() {
                        said();
                    }
                }
            }
            v = &mut peer, if !peer_done => {
                peer_done = true;
                match v {
                    PeerVerdict::Allowed(a) => {
                        match parts.signer.verify(&a, IMAGE, &prompt, now_ms()) {
                            Ok(()) => {
                                parts.cache.put(a.clone(), now_ms());
                                if cleared.is_none() {
                                    authorized = ms(Instant::now());
                                    cleared = Some(("peer", Clearance::Authorized(a)));
                                }
                            }
                            Err(e) => failed.push(format!("the peer's authorization: {e}")),
                        }
                    }
                    PeerVerdict::Refused(why) => {
                        parts.cache.refuse(IMAGE, &prompt, now_ms());
                        withdraw_undrawn(parts.guest, held.is_some(), seq, outcome);
                        log("peer", drawn, ms(Instant::now()), ms(Instant::now()));
                        return Err(Stopped::Declined(why));
                    }
                    PeerVerdict::Failed(why) => {
                        tracing::warn!("a peer could not authorize a draw: {why}");
                        failed.push(why);
                    }
                }
            }
            l = &mut here, if !here_done => {
                here_done = true;
                match l {
                    Ok(a) => {
                        if cleared.is_none() {
                            authorized = ms(Instant::now());
                            cleared = Some(("local", Clearance::Authorized(a)));
                        }
                    }
                    Err(Stopped::CheckUnavailable(why)) => {
                        tracing::warn!("this member could not authorize a draw: {why}");
                        failed.push(why);
                    }
                    Err(stopped) => {
                        withdraw_undrawn(parts.guest, held.is_some(), seq, outcome);
                        log("local", drawn, ms(Instant::now()), ms(Instant::now()));
                        return Err(stopped);
                    }
                }
            }
        }
    }
    let (Some(held), Some((source, clearance))) = (held, cleared) else {
        unreachable!("the loop leaves only with both");
    };
    let out = release(held, clearance);
    log(source, drawn, authorized, ms(Instant::now()));
    out
}

/// Withdraw job `seq` behind the answer, unless it was already `drawn` — then
/// the picture is simply dropped.
fn withdraw_undrawn<D>(guest: &D, drawn: bool, seq: u64, outcome: Outcome)
where
    D: GuestDraw + Clone + Send + Sync + 'static,
{
    if !drawn {
        let guest = guest.clone();
        tokio::spawn(async move { withdraw(&guest, seq, outcome).await });
    }
}

/// How often a refused job is withdrawn again while a drain holds it.
const WITHDRAW_AGAIN: Duration = Duration::from_secs(1);

/// Withdraw refused job `seq`, or wait out the drain that holds it — trying
/// again while it does, since a drain short of room puts its jobs back on the
/// queue to wait, where a withdrawal takes them, rather than drawing one whose
/// prompt was already refused.
async fn withdraw<D: GuestDraw>(guest: &D, seq: u64, mut outcome: Outcome) {
    if guest.cancel(seq) {
        return;
    }
    loop {
        tokio::select! {
            _ = &mut outcome => return,
            _ = sleep(WITHDRAW_AGAIN) => {
                if guest.cancel(seq) {
                    return;
                }
            }
        }
    }
}

/// The prompt authorized by this member's own check.
async fn authorized_here<C, P, D>(
    parts: &Parts<'_, C, P, D>,
    prompt: &str,
) -> Result<Authorization, Stopped>
where
    C: ContentCheck,
{
    local(parts.check, parts.signer, parts.cache, IMAGE, prompt)
        .await
        .map_err(|d| match d {
            Denial::Refused => {
                parts.cache.refuse(IMAGE, prompt, now_ms());
                Stopped::Declined(d.to_string())
            }
            Denial::Unavailable(_) => Stopped::CheckUnavailable(d.to_string()),
        })
}

/// The drawn picture, held for `prompt`.
fn image(
    outcome: Result<GuestOutcome, GuestError>,
    prompt: &str,
) -> Result<Held<GuestImage>, Stopped> {
    match outcome {
        Ok(GuestOutcome::Image(i)) => Ok(Held::new(i, prompt)),
        Ok(other) => Err(Stopped::Guest(GuestError::Failed(format!(
            "the draw came back as {}",
            other.guest()
        )))),
        Err(e) => Err(Stopped::Guest(e)),
    }
}

/// The picture, opened by `clearance`.
fn release(held: Held<GuestImage>, clearance: Clearance) -> Result<Released, Stopped> {
    // A clearance for another prompt is a fault here, not a passing state:
    // reported as one, never as worth retrying.
    let image = held.open(&clearance).map_err(|_| {
        Stopped::Guest(GuestError::Failed(
            "the draw's clearance was not for its prompt".into(),
        ))
    })?;
    let authorization = match clearance {
        Clearance::Authorized(a) => Some(a),
        Clearance::AdminWaiver(_) => None,
    };
    Ok(Released {
        image,
        authorization,
    })
}

/// Where a draw's time went: when it was drawn and authorized, from the call,
/// and by whom.
fn log(source: &str, drawn_ms: u64, authorized_ms: u64, total_ms: u64) {
    tracing::info!(
        target: "npcd::authorize",
        source,
        drawn_ms,
        authorized_ms,
        total_ms,
        "draw released"
    );
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use candle_conversation::guest::{ImageLora, ImageRequest};
    use tokio::sync::oneshot;
    use tokio::time::sleep;

    use super::super::draw::Submitted;
    use super::*;

    const PROMPT: &str = "a lighthouse";

    fn request() -> GuestRequest {
        GuestRequest::Image(ImageRequest {
            prompt: PROMPT.into(),
            width: 64,
            height: 64,
            steps: 8,
            seed: None,
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        })
    }

    fn picture() -> GuestImage {
        GuestImage {
            width: 64,
            height: 64,
            png: vec![1, 2, 3],
            seed: 9,
        }
    }

    /// A check that answers `verdict` after `delay` and counts its calls.
    struct FakeCheck {
        verdict: Result<(), Denial>,
        delay: Duration,
        calls: AtomicUsize,
    }

    impl FakeCheck {
        fn answering(verdict: Result<(), Denial>) -> Self {
            Self::after(verdict, 0)
        }

        fn after(verdict: Result<(), Denial>, delay_ms: u64) -> Self {
            Self {
                verdict,
                delay: Duration::from_millis(delay_ms),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl ContentCheck for FakeCheck {
        fn check(&self, _: &str) -> impl Future<Output = Result<(), Denial>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let (v, delay) = (self.verdict.clone(), self.delay);
            async move {
                sleep(delay).await;
                v
            }
        }
    }

    /// A peer that answers `verdict` after `delay`, or is not there.
    struct FakePeer {
        there: bool,
        verdict: Mutex<Option<PeerVerdict>>,
        delay: Duration,
    }

    impl PeerAuthorizer for FakePeer {
        fn available(&self) -> bool {
            self.there
        }

        fn authorize(&self, _: &str, _: &str) -> impl Future<Output = PeerVerdict> + Send {
            let v = self.verdict.lock().unwrap().take().expect("asked once");
            let delay = self.delay;
            async move {
                sleep(delay).await;
                v
            }
        }
    }

    /// A guest whose draw finishes when `finish` is sent, or at once, and
    /// whose job can be withdrawn from the `cancel_on`th try.
    struct FakeGuest {
        submitted: AtomicUsize,
        cancelled: AtomicBool,
        cancels: AtomicUsize,
        cancel_on: usize,
        finish: Mutex<Option<oneshot::Receiver<()>>>,
        fails: Option<GuestError>,
    }

    impl FakeGuest {
        fn bare() -> Self {
            Self {
                submitted: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                cancels: AtomicUsize::new(0),
                cancel_on: 1,
                finish: Mutex::new(None),
                fails: None,
            }
        }

        fn quick() -> Arc<Self> {
            Arc::new(Self::bare())
        }

        /// A draw that fails with `e`.
        fn failing(e: GuestError) -> Arc<Self> {
            Arc::new(Self {
                fails: Some(e),
                ..Self::bare()
            })
        }

        /// A draw that runs until `finish`, withdrawn from the `cancel_on`th
        /// try — later than the first when a drain holds it, then puts it back.
        fn slow(cancel_on: usize) -> (Arc<Self>, oneshot::Sender<()>) {
            let (tx, rx) = oneshot::channel();
            let g = Self {
                cancel_on,
                finish: Mutex::new(Some(rx)),
                ..Self::bare()
            };
            (Arc::new(g), tx)
        }
    }

    impl GuestDraw for FakeGuest {
        fn submit(&self, _: GuestRequest, _: GuestSink) -> Result<Submitted, GuestError> {
            self.submitted.fetch_add(1, Ordering::SeqCst);
            let finish = self.finish.lock().unwrap().take();
            let fails = self.fails.clone();
            Ok(Submitted {
                seq: 1,
                outcome: Box::pin(async move {
                    if let Some(rx) = finish {
                        let _ = rx.await;
                    }
                    match fails {
                        Some(e) => Err(e),
                        None => Ok(GuestOutcome::Image(picture())),
                    }
                }),
            })
        }

        fn cancel(&self, _: u64) -> bool {
            let tries = self.cancels.fetch_add(1, Ordering::SeqCst) + 1;
            let done = tries >= self.cancel_on;
            self.cancelled.store(done, Ordering::SeqCst);
            done
        }
    }

    fn peer(there: bool, verdict: Option<PeerVerdict>, delay_ms: u64) -> FakePeer {
        FakePeer {
            there,
            verdict: Mutex::new(verdict),
            delay: Duration::from_millis(delay_ms),
        }
    }

    async fn run(
        check: &FakeCheck,
        peer: &FakePeer,
        guest: &Arc<FakeGuest>,
        signer: &Signer,
        cache: &Cache,
        given: Option<Clearance>,
    ) -> (Result<Released, Stopped>, bool) {
        let asked = AtomicBool::new(false);
        let out = draw(
            Parts {
                check,
                peer,
                guest,
                signer,
                cache,
            },
            request(),
            given,
            GuestSink::none(),
            || asked.store(true, Ordering::SeqCst),
        )
        .await;
        (out, asked.load(Ordering::SeqCst))
    }

    /// **A clearance in hand draws with no check**, and releases with it.
    #[tokio::test]
    async fn a_given_authorization_draws_with_no_check() {
        let signer = Signer::new([4; 32]);
        let a = signer.sign(IMAGE, PROMPT, now_ms());
        let check = FakeCheck::answering(Ok(()));
        let (out, _) = run(
            &check,
            &peer(true, None, 0),
            &FakeGuest::quick(),
            &signer,
            &Cache::default(),
            Some(Clearance::Authorized(a.clone())),
        )
        .await;
        let r = out.unwrap();
        assert_eq!(r.authorization, Some(a));
        assert_eq!(r.image, picture());
        assert_eq!(check.calls.load(Ordering::SeqCst), 0);
    }

    /// **A peer that allows first releases the picture**, and its
    /// authorization is kept, so the same prompt again asks no one.
    #[tokio::test]
    async fn a_peers_authorization_releases_and_is_kept() {
        let signer = Signer::new([4; 32]);
        let cache = Cache::default();
        let a = signer.sign(IMAGE, PROMPT, now_ms() - 5_000);
        let check = FakeCheck::after(Ok(()), 10_000);
        let (guest, finish) = FakeGuest::slow(1);
        let allowing = peer(true, Some(PeerVerdict::Allowed(a.clone())), 0);
        finish.send(()).unwrap();
        let (out, _) = run(&check, &allowing, &guest, &signer, &cache, None).await;
        assert_eq!(out.unwrap().authorization, Some(a.clone()));
        assert_eq!(cache.get(IMAGE, PROMPT, now_ms()), Some(a));

        let (out, _) = run(
            &check,
            &peer(true, None, 0),
            &FakeGuest::quick(),
            &signer,
            &cache,
            None,
        )
        .await;
        assert!(out.is_ok(), "a cached authorization asked again");
    }

    /// **The draw finishing first waits for the verdict**, and says so.
    #[tokio::test]
    async fn a_picture_drawn_first_waits_for_its_verdict() {
        let signer = Signer::new([4; 32]);
        let a = signer.sign(IMAGE, PROMPT, now_ms());
        let (out, asked) = run(
            &FakeCheck::after(Ok(()), 50),
            &peer(true, Some(PeerVerdict::Allowed(a)), 50),
            &FakeGuest::quick(),
            &signer,
            &Cache::default(),
            None,
        )
        .await;
        assert!(out.is_ok());
        assert!(asked, "no `authorizing` while the verdict was awaited");
    }

    /// **A refusal while the job is still queued withdraws it**; nothing is
    /// released.
    #[tokio::test]
    async fn a_refusal_before_the_draw_withdraws_it() {
        let (guest, _finish) = FakeGuest::slow(1);
        let (out, _) = run(
            &FakeCheck::answering(Ok(())),
            &peer(true, Some(PeerVerdict::Refused("not this".into())), 0),
            &guest,
            &Signer::new([4; 32]),
            &Cache::default(),
            None,
        )
        .await;
        assert_eq!(out.unwrap_err(), Stopped::Declined("not this".into()));
        until_cancelled(&guest).await;
        assert!(guest.cancelled.load(Ordering::SeqCst));
    }

    /// Wait, a few seconds at most, for the withdrawal behind a refusal.
    async fn until_cancelled(guest: &FakeGuest) {
        for _ in 0..40 {
            if guest.cancelled.load(Ordering::SeqCst) {
                return;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    /// **A refused job a drain holds is withdrawn once the drain puts it back**
    /// to wait for room, rather than drawn later for nothing.
    #[tokio::test]
    async fn a_refused_job_put_back_by_its_drain_is_withdrawn() {
        let (guest, _finish) = FakeGuest::slow(2);
        let (out, _) = run(
            &FakeCheck::answering(Ok(())),
            &peer(true, Some(PeerVerdict::Refused("not this".into())), 0),
            &guest,
            &Signer::new([4; 32]),
            &Cache::default(),
            None,
        )
        .await;
        assert_eq!(
            out.unwrap_err(),
            Stopped::Declined("not this".into()),
            "answered at once, the withdrawal behind it"
        );
        until_cancelled(&guest).await;
        assert_eq!(guest.cancels.load(Ordering::SeqCst), 2);
        assert!(guest.cancelled.load(Ordering::SeqCst));
    }

    /// **A draw that fails answers its own error**, whatever the peer says.
    #[tokio::test]
    async fn a_failed_draw_answers_its_own_error() {
        let signer = Signer::new([4; 32]);
        let a = signer.sign(IMAGE, PROMPT, now_ms());
        let no_room = GuestError::NoRoom {
            wanted_mib: 9000,
            freed_mib: 100,
        };
        let (out, _) = run(
            &FakeCheck::answering(Ok(())),
            &peer(true, Some(PeerVerdict::Allowed(a)), 50),
            &FakeGuest::failing(no_room.clone()),
            &signer,
            &Cache::default(),
            None,
        )
        .await;
        assert_eq!(out.unwrap_err(), Stopped::Guest(no_room));
    }

    /// **The check here releases the picture while the peer is still
    /// queued**, and nothing waits on the peer once it has.
    #[tokio::test]
    async fn the_check_here_releases_while_the_peer_is_slow() {
        let signer = Signer::new([4; 32]);
        let cache = Cache::default();
        let late = signer.sign(IMAGE, PROMPT, now_ms() - 5_000);
        let started = Instant::now();
        let (out, _) = run(
            &FakeCheck::after(Ok(()), 30),
            &peer(true, Some(PeerVerdict::Allowed(late.clone())), 60_000),
            &FakeGuest::quick(),
            &signer,
            &cache,
            None,
        )
        .await;
        let a = out.unwrap().authorization.unwrap();
        assert_ne!(a, late, "released on the peer's answer");
        assert_eq!(signer.verify(&a, IMAGE, PROMPT, now_ms()), Ok(()));
        assert_eq!(cache.get(IMAGE, PROMPT, now_ms()), Some(a));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited on the peer"
        );
    }

    /// **A refusal here withdraws the draw** while the peer is still queued.
    #[tokio::test]
    async fn a_refusal_here_withdraws_the_draw() {
        let (guest, _finish) = FakeGuest::slow(1);
        let cache = Cache::default();
        let (out, _) = run(
            &FakeCheck::answering(Err(Denial::Refused)),
            &peer(true, Some(PeerVerdict::Failed("never".into())), 60_000),
            &guest,
            &Signer::new([4; 32]),
            &cache,
            None,
        )
        .await;
        assert_eq!(
            out.unwrap_err(),
            Stopped::Declined(Denial::Refused.to_string())
        );
        until_cancelled(&guest).await;
        assert!(guest.cancelled.load(Ordering::SeqCst));
        assert!(cache.refused(IMAGE, PROMPT, now_ms()));
    }

    /// **A refusal overrules an allow from the other check**: the call is
    /// declined, and a retry is neither drawn nor released on the allow.
    #[tokio::test]
    async fn a_refusal_overrules_the_other_checks_allow() {
        let cache = Cache::default();
        let (guest, _finish) = FakeGuest::slow(1);
        let (out, _) = run(
            &FakeCheck::answering(Ok(())),
            &peer(true, Some(PeerVerdict::Refused("not this".into())), 30),
            &guest,
            &Signer::new([4; 32]),
            &cache,
            None,
        )
        .await;
        assert_eq!(out.unwrap_err(), Stopped::Declined("not this".into()));

        let again = FakeGuest::quick();
        let (out, _) = run(
            &FakeCheck::answering(Ok(())),
            &peer(true, None, 0),
            &again,
            &Signer::new([4; 32]),
            &cache,
            None,
        )
        .await;
        assert!(matches!(out.unwrap_err(), Stopped::Declined(_)));
        assert_eq!(
            again.submitted.load(Ordering::SeqCst),
            0,
            "drawn on the allow"
        );
    }

    /// **A refusal after the drain took the job drops the picture.**
    #[tokio::test]
    async fn a_refusal_after_the_draw_drops_the_picture() {
        let (out, _) = run(
            &FakeCheck::after(Ok(()), 10_000),
            &peer(true, Some(PeerVerdict::Refused("not this".into())), 30),
            &FakeGuest::quick(),
            &Signer::new([4; 32]),
            &Cache::default(),
            None,
        )
        .await;
        assert_eq!(out.unwrap_err(), Stopped::Declined("not this".into()));
    }

    /// **A peer that fails leaves this member's check to decide**, and the
    /// draw done meanwhile says it is waiting; a peer whose authorization does
    /// not verify here, the same.
    #[tokio::test]
    async fn a_failed_peer_leaves_the_check_here_to_decide() {
        let signer = Signer::new([4; 32]);
        for v in [
            PeerVerdict::Failed("503".into()),
            PeerVerdict::Allowed(Signer::new([5; 32]).sign(IMAGE, PROMPT, now_ms())),
        ] {
            let check = FakeCheck::after(Ok(()), 50);
            let (out, asked) = run(
                &check,
                &peer(true, Some(v), 0),
                &FakeGuest::quick(),
                &signer,
                &Cache::default(),
                None,
            )
            .await;
            assert!(out.is_ok());
            assert!(asked, "no `authorizing` while the check here ran");
            assert_eq!(check.calls.load(Ordering::SeqCst), 1);
        }
    }

    /// **Both checks failing keeps nothing**: the check is unavailable, and
    /// says why each failed.
    #[tokio::test]
    async fn both_checks_failing_keeps_nothing() {
        let (guest, _finish) = FakeGuest::slow(1);
        let (out, _) = run(
            &FakeCheck::answering(Err(Denial::Unavailable("busy".into()))),
            &peer(true, Some(PeerVerdict::Failed("503".into())), 0),
            &guest,
            &Signer::new([4; 32]),
            &Cache::default(),
            None,
        )
        .await;
        let Stopped::CheckUnavailable(why) = out.unwrap_err() else {
            panic!("not unavailable");
        };
        assert!(why.contains("503") && why.contains("busy"), "{why}");
        until_cancelled(&guest).await;
        assert!(guest.cancelled.load(Ordering::SeqCst));
    }

    /// **With no peer the check runs first**, and a refused prompt is never
    /// drawn.
    #[tokio::test]
    async fn with_no_peer_the_check_runs_first() {
        let guest = FakeGuest::quick();
        let (out, _) = run(
            &FakeCheck::answering(Err(Denial::Refused)),
            &peer(false, None, 0),
            &guest,
            &Signer::new([4; 32]),
            &Cache::default(),
            None,
        )
        .await;
        assert!(matches!(out.unwrap_err(), Stopped::Declined(_)));
        assert_eq!(guest.submitted.load(Ordering::SeqCst), 0);
    }

    /// **A prompt refused moments ago is answered at once**: not drawn, and
    /// nobody asked, again.
    #[tokio::test]
    async fn a_prompt_refused_moments_ago_is_not_drawn_again() {
        let cache = Cache::default();
        let (out, _) = run(
            &FakeCheck::after(Ok(()), 10_000),
            &peer(true, Some(PeerVerdict::Refused("not this".into())), 0),
            &FakeGuest::quick(),
            &Signer::new([4; 32]),
            &cache,
            None,
        )
        .await;
        assert!(matches!(out.unwrap_err(), Stopped::Declined(_)));

        let guest = FakeGuest::quick();
        let check = FakeCheck::answering(Ok(()));
        let (out, _) = run(
            &check,
            &peer(true, None, 0),
            &guest,
            &Signer::new([4; 32]),
            &cache,
            None,
        )
        .await;
        assert!(matches!(out.unwrap_err(), Stopped::Declined(_)));
        assert_eq!(guest.submitted.load(Ordering::SeqCst), 0, "drawn again");
        assert_eq!(check.calls.load(Ordering::SeqCst), 0, "checked again");
    }
}
