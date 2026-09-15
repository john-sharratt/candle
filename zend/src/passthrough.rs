//! OpenAI **passthrough**: a client that sends its whole context on every call —
//! its own system prompt, every prior message, and the new one (Cline and most
//! agent frameworks) — runs that context as-is. None of the daemon's own prompt
//! applies: no persona, no tool catalog, no retrieval, no tool orchestration.
//! What stays is the engine: each turn prefills and decodes through the batched
//! scheduler, seals into the substrate, and resumes on the next call — or after a
//! restart — without prefilling the history again.
//!
//! Selected by the request's `model` ([`PASSTHROUGH_MODEL`]), so a client that
//! sends a system message to the daemon's own conversations is unaffected.
//!
//! A conversation is identified by its system prompt and first user message
//! ([`conversation_key`]). Each call's history is matched against the exchanges
//! the conversation already holds ([`extends`]) and only the unmatched tail is
//! prefilled. A history that no longer extends what is held — the client edited
//! or trimmed it — opens a new conversation: a hybrid's recurrent state cannot be
//! rewound to the point of divergence.
//!
//! Live conversations stay cached for [`IDLE_TTL`] after their last call, each
//! behind its own async lock: calls to one conversation queue, calls to
//! different conversations never wait on each other.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;

use candle_conversation::projection::{
    Builder, GroupId, LayerId, Reserved, SectionId, TimelineId, TurnIndex,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig, TurnText};

use crate::openai_tools;
use crate::types::{ChatMessage, Role};

/// The `model` a request names to run as a passthrough.
pub const PASSTHROUGH_MODEL: &str = "passthrough";

/// Timeline metadata key holding a passthrough conversation's [`conversation_key`].
pub const METADATA_KEY: &str = "passthrough";

/// A passthrough conversation's `conv_id` is this prefix and its timeline id —
/// the handle `/v1/conversations` lists and addresses it by.
pub const CONV_ID_PREFIX: &str = "passthrough-";

/// Longest sidebar label, in characters.
const LABEL_CHARS: usize = 60;

/// How long a conversation stays live after its last call.
pub const IDLE_TTL: Duration = Duration::from_secs(5 * 60);

/// How often idle conversations are looked for.
const SWEEP_EVERY: Duration = Duration::from_secs(30);

/// Frame-section partition for passthrough system prompts: `[2^31, 2^31 + 2^30)`,
/// clear of the YAML schema's low ids and the tool ids above them, the
/// calibration corpus just under the reserved band, and the band itself. Each
/// prompt takes a pair — its frame and the summary framing
/// `for_reserved_corpus` places at frame + 1.
const SECTION_BAND_BASE: u32 = 1 << 31;
const SECTION_BAND_PAIRS: u32 = 1 << 29;

/// One prior user half and the assistant reply to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exchange {
    /// Its tool results' content is literal text — see [`TurnText`].
    pub user: TurnText,
    pub assistant: String,
}

/// A request's messages as the engine takes them: the system prompt, the prior
/// exchanges, and the new user half.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub system: String,
    pub history: Vec<Exchange>,
    pub message: TurnText,
    /// The client's tool definitions: listed in the system prompt, and the
    /// source of the grammar its calls are held to ([`LiveConv::tool_stencil`]).
    pub tools: Vec<Value>,
}

impl Transcript {
    /// Fold OpenAI messages into exchanges, in the model's own tool format
    /// ([`openai_tools`]).
    ///
    /// A turn's user half is everything between two assistant replies: the user
    /// message, and any tool results (each in `<tool_response>`) or further user
    /// messages that followed it, joined in order. Consecutive assistant
    /// messages answer the same user half, and an assistant's `tool_calls`
    /// follow its text as `<tool_call>` blocks. System messages, wherever they
    /// sit, form the system prompt, which the client's `tools` close as the
    /// `# Tools` block. The last message must be on the user side — it is what
    /// this call answers.
    pub fn from_messages(messages: &[ChatMessage], tools: &[Value]) -> Result<Self, String> {
        let mut system: Vec<&str> = Vec::new();
        let mut history: Vec<Exchange> = Vec::new();
        let mut user: Vec<TurnText> = Vec::new();
        for m in messages {
            match m.role {
                Role::System => system.push(&m.content),
                Role::User => user.push(TurnText::from(m.content.as_str())),
                Role::Tool => user.push(openai_tools::render_response(&m.content)),
                Role::Assistant if user.is_empty() => match history.last_mut() {
                    Some(last) => {
                        last.assistant.push_str("\n\n");
                        last.assistant.push_str(&assistant_text(m));
                    }
                    None => return Err("an assistant message comes before any user message".into()),
                },
                Role::Assistant => {
                    history.push(Exchange {
                        user: TurnText::join(std::mem::take(&mut user), "\n\n"),
                        assistant: assistant_text(m),
                    });
                }
            }
        }
        if user.is_empty() {
            return Err("the last message must be a user message or a tool result".into());
        }
        let mut system = system.join("\n\n");
        if !tools.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&openai_tools::tools_prompt(tools));
        }
        Ok(Self {
            system,
            history,
            message: TurnText::join(user, "\n\n"),
            tools: tools.to_vec(),
        })
    }

    /// The conversation's opening user half.
    pub fn first_user(&self) -> String {
        self.history
            .first()
            .map_or_else(|| self.message.text(), |e| e.user.text())
    }

    /// See [`conversation_key`].
    pub fn key(&self) -> String {
        conversation_key(&self.system, &self.first_user())
    }
}

/// An assistant message as the model wrote it: its text, then each of its
/// `tool_calls` as a `<tool_call>` block.
fn assistant_text(m: &ChatMessage) -> String {
    let mut parts: Vec<String> = Vec::new();
    let text = m.content.trim_end();
    if !text.is_empty() {
        parts.push(text.to_string());
    }
    parts.extend(
        m.tool_calls
            .iter()
            .map(|c| openai_tools::render_call(&c.function.name, &c.function.arguments)),
    );
    parts.join("\n")
}

/// A conversation's identity: its system prompt and first user message. Hex of
/// the first 16 bytes of their SHA-256, the system prompt length-prefixed so the
/// boundary between the two cannot shift.
pub fn conversation_key(system: &str, first_user: &str) -> String {
    let mut h = Sha256::new();
    h.update((system.len() as u64).to_le_bytes());
    h.update(system.as_bytes());
    h.update(first_user.as_bytes());
    h.finalize()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Text as compared across calls: every `<think>…</think>` block removed (the
/// substrate keeps a reply's reasoning, and a client may or may not send it back),
/// surrounding whitespace trimmed, and each tool call in one fixed rendering
/// ([`openai_tools::canonical`]). An unclosed block runs to the end, less a
/// tool call written inside it, which was the reply's answer
/// ([`crate::reasoning_split`]).
pub fn normalize(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    const CALL: &str = "<tool_call>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(OPEN) {
        out.push_str(&rest[..open]);
        rest = match rest[open..].find(CLOSE) {
            Some(close) => &rest[open + close + CLOSE.len()..],
            None => rest[open..].find(CALL).map_or("", |c| &rest[open + c..]),
        };
    }
    out.push_str(rest);
    openai_tools::canonical(&out)
}

/// Whether `incoming` begins with every exchange of `held`, compared
/// [`normalize`]d — i.e. the conversation can take the call by prefilling only
/// `incoming[held.len()..]`.
pub fn extends(held: &[Exchange], incoming: &[Exchange]) -> bool {
    held.len() <= incoming.len()
        && held.iter().zip(incoming).all(|(h, i)| {
            normalize(&h.user.text()) == normalize(&i.user.text())
                && normalize(&h.assistant) == normalize(&i.assistant)
        })
}

/// The `probe`-th candidate frame-section id for `system`: an even id in the
/// passthrough partition, derived from the prompt so identical prompts share a
/// frame (and its sealed K/V) across conversations.
fn frame_candidate(system: &str, probe: u32) -> u32 {
    let digest = Sha256::digest(system.as_bytes());
    let h = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    SECTION_BAND_BASE + 2 * (h.wrapping_add(probe) % SECTION_BAND_PAIRS)
}

/// The frame-section id `system`'s conversations use: the first candidate that
/// is unused, or already holds exactly this prompt. The substrate keys a
/// section's K/V by id alone, so a candidate holding another prompt must be
/// passed over — reusing it would attend to the other prompt's K/V.
fn frame_section_base(engine: &ConversationEngine, system: &str) -> anyhow::Result<u32> {
    let tokens = engine
        .tokenizer()
        .encode(system, false)
        .map_err(|e| anyhow::anyhow!("tokenizing the system prompt: {e}"))?
        .get_ids()
        .to_vec();
    let conv = engine.conversation();
    let view = conv.read();
    (0..SECTION_BAND_PAIRS)
        .map(|probe| frame_candidate(system, probe))
        .find(|&base| {
            let id = SectionId::new(base);
            !view.section_exists(id) || view.section_tokens_of(id).as_slice() == tokens.as_slice()
        })
        .ok_or_else(|| anyhow::anyhow!("the passthrough frame partition is full"))
}

/// The `conv_id` a passthrough conversation is listed under.
pub fn conv_id_of(timeline: TimelineId) -> String {
    format!("{CONV_ID_PREFIX}{}", timeline.raw())
}

/// The timeline a passthrough `conv_id` names, or `None` for any other id.
pub fn timeline_of(conv_id: &str) -> Option<TimelineId> {
    let raw = conv_id.strip_prefix(CONV_ID_PREFIX)?.parse::<u64>().ok()?;
    TimelineId::from_raw(raw)
}

/// A sidebar label: the first non-blank line of the opening user message,
/// clipped to [`LABEL_CHARS`].
pub fn label_for(first_user: &str) -> String {
    let line = first_user
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    match line.char_indices().nth(LABEL_CHARS) {
        Some((at, _)) => format!("{}…", &line[..at]),
        None => line.to_string(),
    }
}

/// List `timeline` under its passthrough `conv_id`, labelled by its opening
/// message. Idempotent, so a conversation stored before it carried one gains
/// it the next time it is opened.
fn name(engine: &ConversationEngine, timeline: TimelineId, first_user: &str) -> anyhow::Result<()> {
    engine
        .set_conversation_conv_id(timeline, &conv_id_of(timeline))
        .map_err(|e| anyhow::anyhow!("naming passthrough conversation: {e}"))?;
    engine
        .set_conversation_label(timeline, &label_for(first_user))
        .map_err(|e| anyhow::anyhow!("labelling passthrough conversation: {e}"))
}

/// The exchanges `timeline` holds, in turn order.
pub fn held_history(engine: &ConversationEngine, timeline: TimelineId) -> Vec<Exchange> {
    let conv = engine.conversation();
    let view = conv.read();
    let mut turns: Vec<TurnIndex> = view.turn_indices(timeline).collect();
    turns.sort_unstable();
    turns
        .into_iter()
        .map(|i| Exchange {
            user: TurnText::from(view.user_text_of(timeline, i)),
            assistant: view.assistant_text_of(timeline, i),
        })
        .collect()
}

/// A live passthrough conversation and the exchanges it holds.
pub struct LiveConv {
    pub seq: Sequence,
    pub system: String,
    pub history: Vec<Exchange>,
    /// The grammar a call is held to once the model opens one: the client's
    /// tool names and each tool's declared arguments, so an argument its schema
    /// does not declare cannot be written. Empty when the client sent no tools.
    pub tool_stencil: Arc<TriggerRegistry>,
}

/// Compile the client's tool definitions into the call grammar
/// [`LiveConv::tool_stencil`] holds.
fn tool_stencil(
    engine: &ConversationEngine,
    tools: &[Value],
) -> anyhow::Result<Arc<TriggerRegistry>> {
    let specs = openai_tools::specs(tools);
    if specs.is_empty() {
        return Ok(Arc::new(TriggerRegistry::new()));
    }
    engine
        .compile_tool_stencil(&specs)
        .map_err(|e| anyhow::anyhow!("compiling the client's tool grammar: {e}"))
}

/// Resume the newest stored conversation `transcript` extends, or open a new
/// one. The newest first: a conversation reopened after a divergence supersedes
/// the one it left.
pub fn open(
    engine: &ConversationEngine,
    mut config: SequenceConfig,
    transcript: &Transcript,
) -> anyhow::Result<LiveConv> {
    // The prompt is the client's transcript and nothing else — one frame
    // section, no collections — so a mid-decode reprojection has no selection
    // to refresh, and its view rebuild would be pure cost.
    config.reproject_every_n_tokens = 0;
    config.reproject_trigger_texts.clear();
    let key = transcript.key();
    let prompt = config.dialect.format_system_prompt(&transcript.system);
    let tool_stencil = tool_stencil(engine, &transcript.tools)?;
    let base = frame_section_base(engine, &transcript.system)?;
    // A client's tool arguments are quotations — paths, commands, identifiers
    // that are only correct if they repeat — so the penalties are lifted while
    // a call is written. Measured on a Cline turn with them in force: three
    // commands each naming `c:\Users\johna\prog\candle`, and every repeat of
    // the path came out with a space spliced in (`c:\ Users`, `\ candle`) —
    // the unused space-prefixed token outranking the one presence penalty had
    // already demoted.
    let builder = || {
        Builder::for_reserved_corpus(&transcript.system, Reserved::Passthrough, base)
            .free_tool_calls_from_penalties()
    };

    let mut stored = engine.find_conversations_by_metadata(METADATA_KEY, &key);
    stored.sort_unstable_by(|a, b| b.cmp(a));
    for timeline in stored {
        let history = held_history(engine, timeline);
        if extends(&history, &transcript.history) {
            let seq = engine
                .resume_conversation_with_projection(timeline, &prompt, builder(), config)
                .map_err(|e| anyhow::anyhow!("resuming passthrough conversation: {e}"))?;
            name(engine, timeline, &transcript.first_user())?;
            return Ok(LiveConv {
                seq,
                system: transcript.system.clone(),
                history,
                tool_stencil,
            });
        }
    }

    let seq = engine
        .new_conversation_with_projection(
            &prompt,
            builder(),
            LayerId::reserved(Reserved::Passthrough),
            GroupId::reserved(Reserved::Passthrough),
            config,
        )
        .map_err(|e| anyhow::anyhow!("opening passthrough conversation: {e}"))?;
    engine
        .set_conversation_metadata(seq.timeline_id(), METADATA_KEY, &key)
        .map_err(|e| anyhow::anyhow!("tagging passthrough conversation: {e}"))?;
    name(engine, seq.timeline_id(), &transcript.first_user())?;
    Ok(LiveConv {
        seq,
        system: transcript.system.clone(),
        history: Vec::new(),
        tool_stencil,
    })
}

/// The live passthrough conversations, keyed by [`conversation_key`].
pub struct PassthroughCache {
    slots: Mutex<HashMap<String, Arc<Slot>>>,
    sweeping: AtomicBool,
}

struct Slot {
    conv: Arc<AsyncMutex<Option<LiveConv>>>,
    last_used: Mutex<Instant>,
}

impl PassthroughCache {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            slots: Mutex::new(HashMap::new()),
            sweeping: AtomicBool::new(false),
        })
    }

    /// The lock guarding `key`'s conversation — `None` inside until a call opens
    /// it. Calls on one conversation queue on it; calls on different
    /// conversations never meet.
    pub fn conversation(&self, key: &str) -> Arc<AsyncMutex<Option<LiveConv>>> {
        let mut slots = self.slots.lock().unwrap();
        let slot = slots.entry(key.to_string()).or_insert_with(|| {
            Arc::new(Slot {
                conv: Arc::new(AsyncMutex::new(None)),
                last_used: Mutex::new(Instant::now()),
            })
        });
        *slot.last_used.lock().unwrap() = Instant::now();
        Arc::clone(&slot.conv)
    }

    /// Restart `key`'s idle clock — at the end of a call, so a long turn is not
    /// idle the moment it finishes.
    pub fn touch(&self, key: &str) {
        if let Some(slot) = self.slots.lock().unwrap().get(key) {
            *slot.last_used.lock().unwrap() = Instant::now();
        }
    }

    /// Drop every conversation idle for at least `ttl` that no call holds or
    /// waits on (the map's own reference is the only one), freeing its slot.
    /// Returns how many were dropped.
    pub fn sweep(&self, ttl: Duration) -> usize {
        let mut slots = self.slots.lock().unwrap();
        let before = slots.len();
        slots.retain(|_, slot| {
            let idle = slot.last_used.lock().unwrap().elapsed() >= ttl;
            let unheld = Arc::strong_count(&slot.conv) == 1;
            !(idle && unheld)
        });
        before - slots.len()
    }

    /// Drop every conversation — at shutdown, while the scheduler still answers
    /// the slot frees.
    pub fn clear(&self) {
        self.slots.lock().unwrap().clear();
    }

    /// Start the idle sweep, once. It ends when the cache is dropped.
    pub fn ensure_sweeper(self: &Arc<Self>) {
        if self.sweeping.swap(true, Ordering::AcqRel) {
            return;
        }
        let cache: Weak<Self> = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut every = tokio::time::interval(SWEEP_EVERY);
            loop {
                every.tick().await;
                let Some(cache) = cache.upgrade() else {
                    return;
                };
                let dropped = cache.sweep(IDLE_TTL);
                if dropped > 0 {
                    tracing::info!(dropped, "passthrough: released idle conversations");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, MessageToolCall};

    /// The listed id carries the timeline, and no other id is mistaken for one.
    #[test]
    fn a_conv_id_names_its_timeline_and_nothing_else() {
        let tl = TimelineId::from_raw(1789300485193544).unwrap();
        let id = conv_id_of(tl);
        assert_eq!(id, "passthrough-1789300485193544");
        assert_eq!(timeline_of(&id), Some(tl));
        assert_eq!(timeline_of("default"), None);
        assert_eq!(timeline_of("passthrough-abc"), None);
        assert_eq!(timeline_of("passthrough-0"), None);
    }

    /// The label is the opening message's first non-blank line, clipped.
    #[test]
    fn a_label_is_the_first_line_clipped() {
        assert_eq!(
            label_for("\n  hi - how are you?\nmore"),
            "hi - how are you?"
        );
        assert_eq!(
            label_for(&"x".repeat(70)),
            format!("{}…", "x".repeat(LABEL_CHARS))
        );
        assert_eq!(label_for(""), "");
    }

    fn msg(role: Role, content: &str) -> ChatMessage {
        ChatMessage::new(role, content)
    }

    fn fold(messages: &[ChatMessage]) -> Result<Transcript, String> {
        Transcript::from_messages(messages, &[])
    }

    fn calling(text: &str, name: &str, arguments: &str) -> ChatMessage {
        let mut m = msg(Role::Assistant, text);
        m.tool_calls = vec![MessageToolCall {
            id: "c1".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: Value::String(arguments.to_string()),
            },
        }];
        m
    }

    /// Calls and results reach the model as its own format, and the client's
    /// tools close the system prompt.
    #[test]
    fn tool_calls_and_results_fold_into_the_models_format() {
        let tools = [serde_json::json!({"type": "function", "function": {"name": "read_files"}})];
        let t = Transcript::from_messages(
            &[
                msg(Role::System, "sys"),
                msg(Role::User, "task"),
                calling("Reading.", "read_files", r#"{"paths":["a.md"]}"#),
                msg(Role::Tool, "file body"),
            ],
            &tools,
        )
        .unwrap();
        assert!(t.system.starts_with("sys\n\n# Tools\n\n"), "{}", t.system);
        assert!(t
            .system
            .contains(r#"{"type":"function","function":{"name":"read_files"}}"#));
        assert_eq!(
            t.history,
            vec![ex(
                "task",
                "Reading.\n<tool_call>\n{\"name\": \"read_files\", \"arguments\": {\"paths\":[\"a.md\"]}}\n</tool_call>"
            )]
        );
        // The wrapper is markup; what the tool returned is literal.
        assert_eq!(
            t.message,
            TurnText::markup("<tool_response>")
                .then_literal("\nfile body\n")
                .then_markup("</tool_response>")
        );
    }

    /// A call the model left inside a block it never closed was its answer, so
    /// the call the client sends back still extends the held conversation.
    #[test]
    fn a_call_left_in_an_open_block_extends_as_the_answer() {
        let held = vec![ex(
            "task",
            "<think>\n\nLet me look:\n\n<tool_call>\n\
             {\"name\": \"run_commands\", \"arguments\": {\"commands\": [\"ls\"]}}\n</tool_call>",
        )];
        let t = fold(&[
            msg(Role::User, "task"),
            calling("", "run_commands", r#"{"commands":["ls"]}"#),
            msg(Role::Tool, "out"),
        ])
        .unwrap();
        assert!(extends(&held, &t.history));
        assert_eq!(normalize("<think>\nstill thinking"), "");
    }

    /// The call a client sends back matches the one the model wrote, so the
    /// next request extends the held conversation instead of reopening it.
    #[test]
    fn a_call_sent_back_extends_the_call_the_model_wrote() {
        let held = vec![ex(
            "task",
            "<think>\nplan\n</think>\n\nReading.\n\n<tool_call>\n\
             {\"arguments\": {\"paths\": [\"a.md\"]}, \"name\": \"read_files\"}\n</tool_call>",
        )];
        let t = fold(&[
            msg(Role::User, "task"),
            calling("Reading.", "read_files", r#"{"paths":["a.md"]}"#),
            msg(Role::Tool, "body"),
        ])
        .unwrap();
        assert!(extends(&held, &t.history));
    }

    fn ex(user: &str, assistant: &str) -> Exchange {
        Exchange {
            user: TurnText::from(user),
            assistant: assistant.to_string(),
        }
    }

    #[test]
    fn messages_fold_into_exchanges_and_the_new_user_half() {
        let t = fold(&[
            msg(Role::System, "sys"),
            msg(Role::User, "task"),
            msg(Role::Assistant, "calling a tool"),
            msg(Role::Tool, "tool result"),
            msg(Role::User, "and more"),
            msg(Role::Assistant, "first"),
            msg(Role::Assistant, "second"),
            msg(Role::User, "next"),
        ])
        .unwrap();
        assert_eq!(t.system, "sys");
        assert_eq!(t.history.len(), 2);
        assert_eq!(t.history[0], ex("task", "calling a tool"));
        // The tool result's content is literal; its wrapper and the user's own
        // words are markup.
        assert_eq!(
            t.history[1].user,
            TurnText::markup("<tool_response>")
                .then_literal("\ntool result\n")
                .then_markup("</tool_response>\n\nand more")
        );
        assert_eq!(t.history[1].assistant, "first\n\nsecond");
        assert_eq!(t.message.text(), "next");
        assert_eq!(t.first_user(), "task");
    }

    #[test]
    fn a_call_must_end_on_the_user_side() {
        assert!(fold(&[msg(Role::User, "q"), msg(Role::Assistant, "a")]).is_err());
        assert!(fold(&[msg(Role::Assistant, "a"), msg(Role::User, "q")]).is_err());
        let first = fold(&[msg(Role::User, "only")]).unwrap();
        assert_eq!(first.first_user(), "only");
        assert!(first.history.is_empty());
    }

    #[test]
    fn the_key_names_the_prompt_and_the_opening_message() {
        let a = conversation_key("sys", "task");
        assert_eq!(a.len(), 32);
        assert_eq!(a, conversation_key("sys", "task"));
        assert_ne!(a, conversation_key("sys", "other task"));
        assert_ne!(a, conversation_key("other sys", "task"));
        // The length prefix fixes the boundary between the two.
        assert_ne!(conversation_key("ab", "c"), conversation_key("a", "bc"));
    }

    #[test]
    fn normalizing_drops_reasoning_and_edge_whitespace() {
        assert_eq!(normalize("<think>\nplan\n</think>\n\nanswer "), "answer");
        assert_eq!(normalize("a<think>x</think>b<think>y</think>c"), "abc");
        assert_eq!(normalize("answer<think>unclosed"), "answer");
        assert_eq!(normalize("plain"), "plain");
    }

    #[test]
    fn a_history_extends_what_is_held_or_it_does_not() {
        let held = vec![ex("q1", "<think>r</think>a1")];
        assert!(extends(&held, &[ex("q1", "a1"), ex("q2", "a2")]));
        assert!(extends(&held, &[ex("q1", "a1")]));
        assert!(extends(&[], &[ex("q1", "a1")]));
        assert!(!extends(&held, &[ex("q1", "edited"), ex("q2", "a2")]));
        assert!(!extends(&held, &[]));
    }

    #[test]
    fn frame_candidates_are_even_ids_inside_the_partition() {
        for probe in [0, 1, SECTION_BAND_PAIRS - 1, SECTION_BAND_PAIRS] {
            let id = frame_candidate("a system prompt", probe);
            assert!(id >= SECTION_BAND_BASE);
            assert!(id + 1 < SECTION_BAND_BASE + 2 * SECTION_BAND_PAIRS);
            assert_eq!((id - SECTION_BAND_BASE) % 2, 0);
        }
        assert_eq!(
            frame_candidate("same", 0),
            frame_candidate("same", 0),
            "identical prompts share a frame"
        );
        assert_ne!(frame_candidate("same", 0), frame_candidate("same", 1));
    }

    #[tokio::test]
    async fn the_sweep_releases_only_idle_unheld_conversations() {
        let cache = PassthroughCache::new();
        let held = cache.conversation("held");
        let _guard = Arc::clone(&held).lock_owned().await;
        drop(held);
        let _ = cache.conversation("idle");
        assert_eq!(
            cache.sweep(Duration::from_secs(3600)),
            0,
            "nothing is idle yet"
        );
        assert_eq!(cache.sweep(Duration::ZERO), 1, "the held one stays");
        assert_eq!(cache.slots.lock().unwrap().len(), 1);
        assert!(cache.slots.lock().unwrap().contains_key("held"));
    }

    #[tokio::test]
    async fn calls_on_one_conversation_queue_and_others_do_not() {
        let cache = PassthroughCache::new();
        let first = cache.conversation("a").lock_owned().await;
        assert!(
            cache.conversation("a").try_lock().is_err(),
            "a second call on `a` queues"
        );
        assert!(
            cache.conversation("b").try_lock().is_ok(),
            "`b` is not held up by `a`"
        );
        drop(first);
        assert!(cache.conversation("a").try_lock().is_ok());
    }
}
