//! Stops npcd in an order that loses no committed turn, and runs the periodic
//! upkeep that retires stale pictures and warms the content check.
//!
//! Shutdown latches the flag and stops the guardian first, then cancels and
//! awaits every character task, because a cast killed mid-tick leaves a turn in
//! the redo log that was never sealed. The engine shuts down last: its
//! persistence drains queued calls before the substrate closes, and the writer
//! lock is released at the end. The upkeep task holds the state weakly, so it
//! ends with the state; each pass retires kept pictures past their time, which
//! every member does, and a picture another member already retired is simply
//! skipped. At load, one content-check prompt is sent so its frames are sealed
//! before a first draw has to wait on them.

use super::*;

/// Stop npcd, in the order that loses nothing.
///
/// 1. The flag is latched and the guardian stopped, then every character task
///    is cancelled and awaited out — not cosmetic: a character's decode writes
///    turns to the substrate, and a cast killed mid-tick leaves the redo log
///    with a turn whose sealing never happened. Every dropped turn future has
///    released its scheduler slot when this returns.
/// 2. The engine shuts down: its persistence drains — every call already
///    queued on the owner is served first, so a cast write a character left in
///    flight lands before the substrate is closed — its writer flushes, the
///    substrate detaches — stopping the scheduler when npcd is the engine's
///    last attachment — and the writer lock is released.
pub(super) async fn stop(runtime: Arc<Runtime>, guardian: Option<JoinHandle<()>>) {
    runtime.stop();
    if let Some(g) = guardian {
        g.abort();
    }
    runtime.stop_characters().await;
    let shut = spawn_blocking(move || runtime.shutdown_engine()).await;
    match shut {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::error!("npcd: {e:#}"),
        Err(e) => tracing::error!("npcd: engine shutdown panicked: {e}"),
    }
}

/// How often the pictures kept for a while — drawn for a day, uploaded for an
/// hour (`crate::images::Kept`) — are looked over for ones past their time:
/// ten minutes, so an upload's hour is not stretched to two.
pub(super) const RETIRE_KEPT_EVERY: Duration = Duration::from_secs(10 * 60);

/// Retire, every [`RETIRE_KEPT_EVERY`], the kept pictures past their time that
/// no portrait uses. Every member does it; a picture another member retired
/// first has no note left to act on here. Holds the state weakly, so the task
/// ends with it.
pub(super) fn retire_kept_pictures(authored: &Arc<Authored>) {
    let authored = Arc::downgrade(authored);
    tokio::spawn(async move {
        let mut every = interval(RETIRE_KEPT_EVERY);
        loop {
            every.tick().await;
            let Some(authored) = authored.upgrade() else {
                return;
            };
            if authored.runtime.as_ref().is_some_and(|rt| rt.stopping()) {
                return;
            }
            let named = authored.portrait_image_ids().await;
            let images = authored.images.clone();
            let now = api::now_ms();
            match spawn_blocking(move || images.retire_kept(now, &named)).await {
                Ok(Ok(0)) => {}
                Ok(Ok(n)) => tracing::info!("images: {n} kept pictures retired after their time"),
                Ok(Err(e)) => tracing::warn!("images: kept pictures could not be retired: {e}"),
                Err(e) => tracing::error!("images: retiring kept pictures panicked: {e}"),
            }
        }
    });
}

/// How often the engine is looked at while it loads, before the content check
/// is asked its first question.
pub(super) const WARM_CHECK_EVERY: Duration = Duration::from_secs(5);

/// What the content check is asked once at load: a picture of nothing anybody
/// could object to.
pub(super) const WARM_CHECK_PROMPT: &str = "A red apple on a wooden table.";

/// Ask the content check one prompt as soon as the engine is up, so its four
/// questions' frames are sealed before anybody waits on a draw
/// (`crate::compliance`). A frame is kept once sealed; left to the first draw,
/// sealing them was thirty-seven seconds of that draw's wait. Holds the state
/// weakly, so the task ends with it.
pub(super) fn warm_content_check(authored: &Arc<Authored>) {
    let authored = Arc::downgrade(authored);
    tokio::spawn(async move {
        let mut every = interval(WARM_CHECK_EVERY);
        loop {
            every.tick().await;
            let Some(authored) = authored.upgrade() else {
                return;
            };
            let Some(rt) = authored.runtime.as_ref() else {
                return;
            };
            if rt.stopping() {
                return;
            }
            if rt.minds.read().unwrap().is_none() {
                continue;
            }
            let started = Instant::now();
            match compliance::check(&authored, WARM_CHECK_PROMPT).await {
                Ok(()) => tracing::info!(
                    ms = started.elapsed().as_millis() as u64,
                    "content check: its questions' frames are sealed"
                ),
                Err(e) => tracing::warn!("content check: the first question did not answer: {e}"),
            }
            return;
        }
    });
}
