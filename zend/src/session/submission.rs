//! The entry points that submit a user message to a conversation and return
//! its reply as a stream.
//!
//! `submit` takes the message and returns a stream of status and token items.
//! `submit_with_sampling` also takes an explicit sampling config, which tests
//! use for deterministic greedy decoding, and an optional collection to force
//! to full resolution for capture. A worker's turn resumes a sealed call turn
//! by running its tools, and may move to a faster member at a round boundary.
//! The commit helpers report whether a member holds a conversation complete
//! through an origin commit, which a handover waits on.

use super::*;

impl ZendSession {
    /// Submit the latest user message and return a stream of status + token items.
    // See the note on `run_inference_stream`.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        force_hires: Option<String>,
        assistant_prefill: Option<String>,
        lossless_kv: bool,
        tools_mode: ToolMode,
        identity: Option<String>,
        selection: candle_conversation::SelectionState,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        self.submit_with_sampling(
            messages,
            max_tokens,
            conv_id,
            None,
            force_hires,
            assistant_prefill,
            lossless_kv,
            tools_mode,
            identity,
            selection,
        )
        .await
    }

    /// Same as [`Self::submit`] but accepts an explicit
    /// [`candle_conversation::SamplingConfig`] override and an optional
    /// `force_hires` collection name (zend capture: force that collection to
    /// full resolution for this conversation — see [`ChatCompletionRequest`]).
    ///
    /// Used by tests that want deterministic generation (e.g.
    /// `SamplingConfig::argmax()` for greedy decoding) so the
    /// pass/fail signal isn't subject to top-k/top-p sampling noise.
    // See the note on `run_inference_stream`.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_with_sampling(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        sampling: Option<candle_conversation::SamplingConfig>,
        force_hires: Option<String>,
        assistant_prefill: Option<String>,
        lossless_kv: bool,
        tools_mode: ToolMode,
        identity: Option<String>,
        selection: candle_conversation::SelectionState,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        self.submit_turn(
            messages,
            max_tokens,
            conv_id,
            sampling,
            force_hires,
            assistant_prefill,
            lossless_kv,
            tools_mode,
            identity,
            selection,
            TurnStart::default(),
        )
        .await
    }

    /// A worker's turn (`crate::mcp::workers`): from `message`, or — `resume`
    /// naming a sealed call turn, its tools not run or cut off part way, here
    /// or on the member that handed it here — by running that round's tools.
    /// With `may_hand` it moves to a markedly faster member at a round boundary
    /// ([`StreamItem::Handed`]).
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_worker(
        &self,
        message: String,
        images: Vec<String>,
        conv_id: String,
        tools_mode: ToolMode,
        selection: candle_conversation::SelectionState,
        resume: Option<Resume>,
        may_hand: bool,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        let mut asked = ChatMessage::new(Role::User, message);
        asked.images = images;
        self.submit_turn(
            vec![asked],
            None,
            conv_id,
            None,
            None,
            None,
            false,
            tools_mode,
            None,
            selection,
            TurnStart {
                resume,
                may_hand,
                images: Vec::new(),
            },
        )
        .await
    }

    /// Whether `conv_id`'s round at call turn `call_turn` is still to run:
    /// its entry names that turn, and no answer to it is in the substrate.
    pub fn round_pending(&self, conv_id: &str, call_turn: u32) -> bool {
        let Some(state) = self.inference.load_full() else {
            return false;
        };
        let timeline = timeline_for(conv_id);
        resume::marked_call_turn(&state.engine, timeline) == Some(call_turn)
            && !resume::round_answered(&state.engine, timeline, call_turn)
    }

    /// The call turn of `conv_id`'s round still to run, when one is.
    pub fn pending_round(&self, conv_id: &str) -> Option<u32> {
        let state = self.inference.load_full()?;
        let timeline = timeline_for(conv_id);
        resume::marked_call_turn(&state.engine, timeline)
            .filter(|&call_turn| !resume::round_answered(&state.engine, timeline, call_turn))
    }

    /// The `oseq` of this member's newest commit of `conv_id`, once every turn
    /// it wrote of it is committed — what a member taking the conversation
    /// must hold it through (`crate::handover`); waited for as
    /// [`turn_claims::release`] waits, up to `within`. `None` when it is not
    /// committed by then, or this member wrote none of it.
    pub async fn committed_through(&self, conv_id: &str, within: Duration) -> Option<u64> {
        let state = self.inference.load_full()?;
        let conversation = state.engine.conversation();
        let timeline = timeline_for(conv_id);
        let node = conversation.node();
        let until = Instant::now() + within;
        while !conversation.own_turns_committed(timeline, node) {
            if Instant::now() >= until {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        conversation.last_commit_oseq(timeline, node)
    }

    /// Whether this member holds `conv_id` complete through `origin`'s commit
    /// `through`, waiting up to `within` for it to arrive.
    pub async fn holds_through(
        &self,
        conv_id: &str,
        origin: u64,
        through: u64,
        within: Duration,
    ) -> bool {
        let Some(state) = self.inference.load_full() else {
            return false;
        };
        let conversation = state.engine.conversation();
        let timeline = timeline_for(conv_id);
        let until = Instant::now() + within;
        loop {
            if conversation.holds_complete_through(timeline, origin, through) {
                return true;
            }
            if Instant::now() >= until {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// The `oseq` this member's own records are written through once every
    /// write queued so far is in the log: what a member taking a worker from
    /// this one must hold, its turn's commit and the round's resume entry and
    /// the worker's record after it among them.
    pub async fn written_through(&self) -> Option<u64> {
        let conversation = self.inference.load_full()?.engine.conversation();
        tokio::task::spawn_blocking(move || {
            conversation.wait_written(conversation.write_ticket());
            conversation.own_through()
        })
        .await
        .ok()
    }

    /// What this member lacks of `origin`'s records `1..=written`
    /// ([`hand_records::missing`]).
    pub async fn lacks_of(&self, origin: u64, written: u64) -> Vec<(u64, u64)> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let conversation = state.engine.conversation();
        tokio::task::spawn_blocking(move || {
            hand_records::missing(conversation.applied_set().origin(origin), written)
        })
        .await
        .unwrap_or_default()
    }

    /// What a handing of `conv_id` carries of this member's records in
    /// `missing`, the ranges its taker lacks ([`hand_records::gather`]).
    pub async fn carrying(
        &self,
        conv_id: &str,
        missing: Vec<(u64, u64)>,
    ) -> Result<Carrying, String> {
        let state = self
            .inference
            .load_full()
            .ok_or("the engine is not loaded")?;
        let conversation = state.engine.conversation();
        let timeline = timeline_for(conv_id);
        tokio::task::spawn_blocking(move || {
            let origin = conversation.node();
            hand_records::gather(&conversation, timeline, origin, &missing)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    // See the note on `run_inference_stream`.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn submit_turn(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        sampling: Option<candle_conversation::SamplingConfig>,
        force_hires: Option<String>,
        assistant_prefill: Option<String>,
        lossless_kv: bool,
        tools_mode: ToolMode,
        identity: Option<String>,
        selection: candle_conversation::SelectionState,
        mut start: TurnStart,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        let last = messages.iter().rev().find(|m| m.role == Role::User);
        let last_user = last.map(|m| m.content.clone()).unwrap_or_default();
        // The message's images, read before its first round. The HTTP route
        // refuses a url that names nothing before it gets here.
        let parsed = images::parse_all(last.map_or(&[][..], |m| &m.images[..]));
        let refused = parsed.as_ref().err().cloned();
        start.images = parsed.unwrap_or_default();
        // A turn starts having read nothing from outside
        // (`docs/zen_tool_reach.md` §6.2).
        self.mcp.ops().turn_began(&conv_id);

        let inference = Arc::clone(&self.inference);
        let mut ready_rx = self.ready_tx.subscribe();
        let mut status_rx = self.status_tx.subscribe();
        let live_turns = Arc::clone(&self.live_turns);

        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);

        tokio::spawn(async move {
            if let Some(why) = refused {
                let _ = tx.send(Err(anyhow::anyhow!("the message's images: {why}"))).await;
                return;
            }
            // Phase 1 — emit status updates while model is loading.
            if inference.load().is_none() {
                let cur = status_rx.borrow_and_update().clone();
                if !cur.is_empty() && tx.send(Ok(StreamItem::Status(cur))).await.is_err() {
                    return;
                }

                loop {
                    if *ready_rx.borrow_and_update() {
                        break;
                    }
                    tokio::select! {
                        _ = ready_rx.changed() => { break; }
                        _ = status_rx.changed() => {
                            let msg = status_rx.borrow_and_update().clone();
                            if !msg.is_empty()
                                && tx.send(Ok(StreamItem::Status(msg))).await.is_err()
                            {
                                return;
                            }
                        }
                    }
                }

                if tx
                    .send(Ok(StreamItem::Status("Processing…".into())))
                    .await
                    .is_err()
                {
                    return;
                }
            }

            // Phase 2 — generate tokens. `run_inference_stream` handles
            // the fork (which registers the timeline in the substrate),
            // then persists the conv_id and spawns the titler in
            // parallel — all in the right order so the in-RAM substrate
            // sees both the timeline registration and the conv_id by
            // the time the next sidebar refresh runs.
            let state = inference.load_full();
            if let Some(state) = state {
                let history_before = dialogue_history(&state, &conv_id).len();
                // The conversation stays this member's until the turn's last
                // record is written, however soon its client goes.
                let claim = turn_claims::hold(&conv_id);
                let live = live_turns.begin(&conv_id, Some(last_user.clone()), history_before);
                let mut ts = run_inference_stream(
                    state,
                    conv_id.clone(),
                    last_user.clone(),
                    max_tokens,
                    sampling,
                    force_hires,
                    assistant_prefill,
                    lossless_kv,
                    tools_mode,
                    identity,
                    selection,
                    start,
                );
                relay_live(&mut ts, Some(&tx), &live, &conv_id).await;
                live_turns.end(&conv_id, &live);
                // The reply is over: its stream ends now, and the claim waits on
                // the turn's commit behind it, where no client waits.
                drop(ts);
                drop(tx);
                turn_claims::release(claim, &conv_id).await;
            } else {
                tx.send(Ok(StreamItem::Status("Model unavailable.".into())))
                    .await
                    .ok();
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
