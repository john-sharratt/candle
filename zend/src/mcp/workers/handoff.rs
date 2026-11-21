//! Moves a running worker from one member to another at a round boundary, with a
//! signed proof that the receiving member may take it.
//!
//! A handing names both members and the commit the handing member holds, and the
//! receiver answers only once it holds that commit and every record the handing
//! member wrote up to it. A receiver that lacks records is asked again, carrying
//! them if there are few enough. A worker whose handing failed stays where it was,
//! unless its record says the other member took it after all. A handing sent
//! twice, or by anyone other than the worker's runner, runs nothing.

use super::*;

/// What a member hands a worker on with (`crate::api::worker_continue`).
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Handing {
    /// The member handing it on, and the member it is handed to: signed, so
    /// the proof names both and stands alone.
    pub from: u64,
    pub to: u64,
    pub owner: String,
    pub thread: String,
    pub name: String,
    pub call_turn: u32,
    /// The round of its turn the call turn was.
    pub iteration: usize,
    /// The handing member's newest commit of the worker's conversation.
    pub through: u64,
    /// Every record the handing member wrote up to this `oseq` is in its log:
    /// the commit, and the round's resume entry and the worker's record after
    /// it. A member lacking any of them says which (`crate::hand_records`).
    pub written: u64,
    /// When it was signed, Unix ms (`worker_hop::fresh`).
    pub signed_ms: u64,
}

/// The ranges a refusal says its member lacks of the handing member's
/// records, when it is one that says so ([`MISSING`]).
pub fn lacked(status: u16, body: &[u8]) -> Option<Vec<(u64, u64)>> {
    if status != 409 {
        return None;
    }
    let v: Value = serde_json::from_slice(body).ok()?;
    if v.get("error").and_then(Value::as_str) != Some(MISSING) {
        return None;
    }
    serde_json::from_value(v.get("missing")?.clone()).ok()
}

/// Hand `worker`, whose turn stopped before running round `iteration`'s call
/// turn `call_turn`, to member `to` (`crate::worker_hop`): once this member's
/// commit of it has landed and its writes are in the log, `to` is asked to
/// take it, signed with the members' handing key, and answers once it holds
/// the commit and runs the worker. When `to` answers that it lacks records of
/// this member's, it is asked again carrying them, if they are few enough
/// (`crate::hand_records`). `Err` keeps the worker here.
pub(super) async fn hand_on(
    session: &ZendSession,
    owner: &str,
    worker: &Worker,
    to: u64,
    call_turn: u32,
    iteration: usize,
) -> Result<(), String> {
    let member = worker_hop::member().ok_or("this member is in no cluster")?;
    let through = session
        .committed_through(&worker.conv_id, HAND_COMMIT_WAIT)
        .await
        .ok_or("its turn's commit did not land in time")?;
    let written = session
        .written_through()
        .await
        .ok_or("its writes could not be waited for")?;
    let from = worker_hop::own_node().ok_or("this member is in no cluster")?;
    let mut carrying = Carrying::default();
    loop {
        let handing = serde_json::to_vec(&Handing {
            from,
            to,
            owner: owner.to_string(),
            thread: worker.thread.clone(),
            name: worker.name.clone(),
            call_turn,
            iteration,
            through,
            written,
            signed_ms: now_ms(),
        })
        .map_err(|e| e.to_string())?;
        let body = hand_records::encode(&handing, &carrying.records);
        let proof = worker_hop::sign(&body).ok_or("this member holds no handing key")?;
        let headers = [
            (
                HeaderName::from_static(NODE_HEADER),
                HeaderValue::from_str(&format!("{to:016x}")).map_err(|e| e.to_string())?,
            ),
            (
                HeaderName::from_static(worker_hop::PROOF_HEADER),
                HeaderValue::from_str(&proof).map_err(|e| e.to_string())?,
            ),
        ];
        let reply = send_up(
            &member,
            Method::POST,
            CONTINUE_PATH,
            &headers,
            body,
            Some(HAND_ANSWER_WAIT),
        )
        .await
        .map_err(|e| format!("the gateway could not be reached: {e:#}"))?;
        if reply.status == 202 {
            return Ok(());
        }
        let refused = format!(
            "{to:016x} answered {}: {}",
            reply.status,
            String::from_utf8_lossy(&reply.body)
        );
        // Asked once carrying what it lacked, it is not asked again.
        let missing = match lacked(reply.status, &reply.body) {
            Some(missing) if carrying.records.is_empty() => missing,
            _ => return Err(refused),
        };
        carrying = session
            .carrying(&worker.conv_id, missing)
            .await
            .map_err(|why| format!("{refused} — and none are carried: {why}"))?;
        if carrying.records.is_empty() {
            return Err(format!("{refused} — and this member holds none of them"));
        }
        tracing::info!(
            conv_id = %worker.conv_id,
            to = %format!("{to:016x}"),
            "mcp: the member taking the worker lacks records of this member's, so the handing \
             carries them: {}",
            carrying.told()
        );
    }
}

/// Whether `worker`'s record says member `to` runs it now: a handing whose
/// answer was lost took effect all the same.
pub(super) fn taken_by(records: &dyn Records, owner: &str, worker: &Worker, to: u64) -> bool {
    threads::worker(records, owner, &worker.thread, &worker.name)
        .is_some_and(|w| w.state == State::Running && w.runner == Some(to))
}

/// [`taken_by`], looked at until it holds or [`TAKEN_WAIT`] passes.
pub(super) async fn taken_within(
    records: &dyn Records,
    owner: &str,
    worker: &Worker,
    to: u64,
) -> bool {
    let until = Instant::now() + TAKEN_WAIT;
    loop {
        if taken_by(records, owner, worker, to) {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Take `handing`'s worker, which member `by` handed here at a round
/// boundary: run its round's tools and carry the worker on — its record, its
/// beat, its settling — as if it had started here. Taken only while it runs on
/// `by`, its round still to run, so a handing sent twice, or by anyone but its
/// runner, runs nothing.
pub fn continue_handed(
    session: &Arc<ZendSession>,
    handing: &Handing,
    by: u64,
) -> Result<Worker, String> {
    if handing.from != by || Some(handing.to) != worker_hop::own_node() {
        return Err(format!(
            "the handing is from {:016x} to {:016x}, not from {by:016x} to this member",
            handing.from, handing.to
        ));
    }
    let records = session.shared_substrate().ok_or_else(loading)?;
    let _held = holding();
    let Some(mut worker) =
        threads::worker(&records, &handing.owner, &handing.thread, &handing.name)
    else {
        return Err(format!(
            "there is no worker `T:{}/{}`",
            handing.thread, handing.name
        ));
    };
    if worker.state != State::Running || worker.runner != Some(by) {
        return Err(format!(
            "`{}` is not running on {by:016x}, so it is not {by:016x}'s to hand on",
            worker.path()
        ));
    }
    if !session.round_pending(&worker.conv_id, handing.call_turn) {
        return Err(format!(
            "`{}` has no round at call turn {} still to run",
            worker.path(),
            handing.call_turn
        ));
    }
    // Its turns were written on `by` after anything this member keeps of the
    // conversation in memory.
    session.forget(&worker.conv_id);
    worker.runner = worker_hop::own_node();
    worker.beat_ms = now_ms();
    threads::put_worker(&records, &handing.owner, &worker).map_err(|e| e.to_string())?;
    if let Some(Err(e)) = session.set_decode_band(&worker.conv_id, worker.priority.band()) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the task's priority could not be kept: {e}");
    }
    session.mcp().bump();
    tokio::spawn(run(
        Arc::clone(session),
        handing.owner.clone(),
        worker.clone(),
        First::Round {
            call_turn: handing.call_turn,
            iteration: handing.iteration,
        },
        None,
    ));
    Ok(worker)
}
