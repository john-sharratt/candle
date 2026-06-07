//! YAML append-log conversation persistence.
//!
//! The file is a multi-document YAML stream.  Each `---`-delimited document
//! is a [`LogEntry`]: the conversation header (first document, written once),
//! turn records, or tombstones.  Documents are appended and flushed after
//! every write so the file is always parseable up to the last flush — killing
//! the process mid-run leaves a complete, readable log.
//!
//! ## File structure
//!
//! ```text
//! ---
//! kind: conversation
//! system_prompt:
//!   - kind: static
//!     text: |
//!       You are Mira, a thoughtful archivist…
//!   - kind: section
//!     name: mood
//!   - kind: section
//!     name: response_template
//!   - kind: section
//!     name: conversation_history
//! created_at: "epoch+20507d 14:22:01Z"
//! ---
//! kind: user_turn
//! turn_id: 1
//! text: |
//!   A letter arrives on your desk.
//! view: []
//! ---
//! kind: assistant_turn
//! turn_id: 2
//! text: |
//!   I hold the envelope for a long moment.
//! view:
//!   - 1
//! signature:
//!   k_lower: "AAEC…"
//!   k_mid: "ZmVk…"
//!   k_upper: "9/7+…"
//!   q_lower: "AQID…"
//!   q_mid: "VGVz…"
//!   q_upper: "c2ln…"
//!   scale_kl: 0.0234
//!   scale_km: 0.0188
//!   scale_ku: 0.0301
//!   scale_ql: 0.0174
//!   scale_qm: 0.0213
//!   scale_qu: 0.0398
//! ---
//! kind: summary_turn
//! turn_id: 3
//! turn_type: sleep
//! text: |
//!   The letter exchange felt weighty. Mira suspects the sender knew too much.
//! covers:
//!   - 1
//!   - 2
//! view:
//!   - 1
//!   - 2
//! ---
//! kind: tombstone
//! turn_id: 1
//! ```
//!
//! ## Turn types
//!
//! | `kind`            | Produced by            | `covers` | Notes |
//! |---|---|---|---|
//! | `user_turn`       | Live user input         | —        | Role implicit in variant name |
//! | `assistant_turn`  | Live inference          | —        | Optional `signature` |
//! | `summary_turn`    | Async cognitive pipeline| required | `turn_type`: `sleep` / `thought` / `reason` |
//! | `tombstone`       | Session management      | —        | Suppresses the referenced turn |
//!
//! ## Session resumption
//!
//! Each turn record's `view` field lists the ordered `turn_id`s that formed
//! the assembled KV-cache context for that turn's prefill and decode.
//! Reading `view` from the most recent assistant turn and reloading those KV
//! blocks exactly restores the warm-start context for the next turn — no
//! bootstrap approximation required.

use crate::tree::TurnType;
use crate::turn::Role;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

// ── Base64Bytes ───────────────────────────────────────────────────────────────

/// A binary blob serialized as a standard-alphabet Base64 string in YAML.
///
/// Used for:
/// - Three-tier INT8 fingerprint band vectors (one per K/Q × band).
/// - Packed token-ID arrays (each `u32` stored as 4 little-endian bytes).
///
/// Storing binary as Base64 keeps YAML files human-inspectable while avoiding
/// the verbosity of YAML integer sequences (128-entry lists per fingerprint band).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Base64Bytes(Vec<u8>);

impl Base64Bytes {
    /// Wrap raw bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Borrow the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Consume and unwrap to raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Build from a signed INT8 slice.  Sign bits are preserved in the byte
    /// representation (`i8 as u8`).
    pub fn from_i8_slice(values: &[i8]) -> Self {
        Self(values.iter().map(|&v| v as u8).collect())
    }

    /// Interpret stored bytes as signed INT8 (sign bit preserved).
    pub fn to_i8_vec(&self) -> Vec<i8> {
        self.0.iter().map(|&b| b as i8).collect()
    }
}

impl Serialize for Base64Bytes {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        s.serialize_str(&STANDARD.encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for Base64Bytes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let s = String::deserialize(d)?;
        STANDARD
            .decode(&s)
            .map(Base64Bytes)
            .map_err(serde::de::Error::custom)
    }
}

// ── Format types ──────────────────────────────────────────────────────────────

/// A segment of the system prompt template.
///
/// The assembled system prompt is an ordered sequence of these segments.
/// Static segments are injected verbatim; section segments are placeholders
/// whose content is resolved at inference time from the named candidate library.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SystemPromptSegment {
    /// Fixed text, written to the assembled context as-is.
    Static { text: String },
    /// Named dynamic section.  Only the name is stored; the resolved content
    /// is selected at inference time by the probe-and-scan mechanism.
    Section { name: String },
}

/// Three-tier attentional provenance fingerprint for a single turn.
///
/// Captures six INT8 band vectors and six f32 dequantization scales following
/// the three-band aggregation scheme (Section 3.1 of the API paper).
///
/// **Band boundaries:** lower = layers `0..N/3`, mid = `N/3..2N/3`,
/// upper = `2N/3..N`, where `N` is the total layer count of the model.
///
/// For Qwen3-30B-A3B-AWQ (d_head = 128), each vector is 128 bytes (128 Base64
/// chars ≈ 172 Base64-encoded chars).  Total fingerprint: ~780 bytes on disk.
///
/// ## K vectors — content semantics
///
/// Aggregated via Q·K score ranking (top 10% of tokens contribute), which
/// mitigates the Q→K distributional gap (OOD problem, Section 2.6) at
/// construction time by selecting the K tokens most visible from Q-space.
///
/// ## Q vectors — cognitive / reasoning state  
///
/// Recency-weighted mean over generated tokens.  `q_upper` is the dominant
/// retrieval signal for conversation history (w_Qu = 0.50) and mood selection
/// (w_Qu = 0.45) because it reflects the model's fully accumulated relational
/// reasoning state at the terminal decode token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSignature {
    // ── K band vectors (content semantics, INT8 × d_head) ────────────────
    /// Mean K, layers 0..N/3 — lexical identity, surface form, syntax.
    pub k_lower: Base64Bytes,
    /// Mean K, layers N/3..2N/3 — semantic category, topic, emotion vocabulary.
    pub k_mid: Base64Bytes,
    /// Mean K, layers 2N/3..N — relational meaning, coreference, context.
    pub k_upper: Base64Bytes,

    // ── Q band vectors (cognitive / reasoning state, INT8 × d_head) ──────
    /// Mean Q, layers 0..N/3 — immediate lexical intent, surface query state.
    pub q_lower: Base64Bytes,
    /// Mean Q, layers N/3..2N/3 — semantic reasoning state, emotion register.
    pub q_mid: Base64Bytes,
    /// Recency-weighted mean Q, layers 2N/3..N — accumulated relational context
    /// and full generation intent.  Dominant signal for history and mood scan.
    pub q_upper: Base64Bytes,

    // ── Per-band dequantization scales ────────────────────────────────────
    pub scale_kl: f32,
    pub scale_km: f32,
    pub scale_ku: f32,
    pub scale_ql: f32,
    pub scale_qm: f32,
    pub scale_qu: f32,
}

/// A turn record stored in the YAML log.
///
/// Token IDs are intentionally not stored: they can be reconstructed from
/// [`text`](Self::text) at inference time via the model's tokenizer, which
/// removes the need to keep a redundant packed-integer column in the log.
#[derive(Debug, Clone)]
pub struct TurnRecord {
    /// Monotonic turn identifier within the conversation.
    pub turn_id: u64,

    /// Role of the speaker — derived from which [`LogEntry`] variant was used
    /// (`user_turn` → [`Role::User`], `assistant_turn` → [`Role::Assistant`]).
    /// For `summary_turn` entries this is always [`Role::Assistant`].
    pub role: Role,

    /// Raw text content of the turn.
    pub text: String,

    /// Ordered list of `turn_id`s that formed the assembled KV-cache context
    /// view for this turn's prefill and decode, recorded at generation time.
    ///
    /// On session resume, reading this field from the most recent assistant turn
    /// and reloading exactly those KV blocks restores the warm-start context for
    /// the next turn's history probe without any bootstrap approximation.
    pub view: Vec<u64>,

    /// Attentional provenance fingerprint, captured at end-of-decode.
    ///
    /// Absent for turns produced before fingerprinting was enabled, or for
    /// turns where fingerprinting was explicitly skipped (e.g. system prompt).
    /// Only meaningful for [`Role::Assistant`] turns.
    pub signature: Option<TurnSignature>,

    /// The cognitive pipeline that produced this turn.
    ///
    /// [`TurnType::Reality`] for all live user↔assistant exchanges.
    /// [`TurnType::Sleep`], [`TurnType::Thought`], or [`TurnType::Reason`] for
    /// async cognitive-pipeline turns that synthesize or summarize prior turns.
    pub turn_type: TurnType,

    /// The turn IDs covered / synthesized by this turn.
    ///
    /// Non-empty only for non-[`TurnType::Reality`] turns.  Lists the ordered
    /// `turn_id`s that this summary or synthesis was derived from — which may
    /// themselves be Reality turns, other summary turns, or a mixture.
    pub covers: Vec<u64>,
}

impl TurnRecord {
    /// Constructor for [`TurnType::Reality`] records without a signature.
    pub fn new(turn_id: u64, role: Role, text: impl Into<String>, view: Vec<u64>) -> Self {
        Self {
            turn_id,
            role,
            text: text.into(),
            view,
            signature: None,
            turn_type: TurnType::Reality,
            covers: Vec::new(),
        }
    }

    /// Constructor for async cognitive-pipeline turns (Sleep, Thought, Reason).
    ///
    /// The `covers` list records which prior `turn_id`s this turn synthesizes
    /// or was derived from.  `turn_type` must **not** be
    /// [`TurnType::Reality`] — use [`new`](Self::new) for live exchanges.
    pub fn new_summary(
        turn_id: u64,
        turn_type: TurnType,
        text: impl Into<String>,
        covers: Vec<u64>,
        view: Vec<u64>,
    ) -> Self {
        debug_assert!(
            turn_type != TurnType::Reality,
            "new_summary requires a non-Reality TurnType; use TurnRecord::new for live turns"
        );
        Self {
            turn_id,
            role: Role::Assistant,
            text: text.into(),
            view,
            signature: None,
            turn_type,
            covers,
        }
    }

    /// Convert to the appropriate [`LogEntry`] variant, normalizing `text` to
    /// end with `\n` so that serde_yaml emits a literal block scalar (`|`).
    fn into_log_entry(self) -> LogEntry {
        // Ensure a trailing newline so serde_yaml uses `|` block scalar style.
        let text = if self.text.ends_with('\n') {
            self.text
        } else {
            self.text + "\n"
        };
        match self.turn_type {
            TurnType::Reality => match self.role {
                Role::User | Role::System => LogEntry::UserTurn {
                    turn_id: self.turn_id,
                    text,
                    view: self.view,
                },
                Role::Assistant => LogEntry::AssistantTurn {
                    turn_id: self.turn_id,
                    text,
                    view: self.view,
                    signature: self.signature,
                },
            },
            turn_type => LogEntry::SummaryTurn {
                turn_id: self.turn_id,
                turn_type,
                text,
                covers: self.covers,
                view: self.view,
                signature: self.signature,
            },
        }
    }

    /// Build from a `UserTurn` log entry, stripping the normalizing trailing `\n`.
    fn from_user_turn(turn_id: u64, text: String, view: Vec<u64>) -> Self {
        Self {
            turn_id,
            role: Role::User,
            text: text.trim_end_matches('\n').to_owned(),
            view,
            signature: None,
            turn_type: TurnType::Reality,
            covers: Vec::new(),
        }
    }

    /// Build from an `AssistantTurn` log entry, stripping the normalizing trailing `\n`.
    fn from_assistant_turn(
        turn_id: u64,
        text: String,
        view: Vec<u64>,
        signature: Option<TurnSignature>,
    ) -> Self {
        Self {
            turn_id,
            role: Role::Assistant,
            text: text.trim_end_matches('\n').to_owned(),
            view,
            signature,
            turn_type: TurnType::Reality,
            covers: Vec::new(),
        }
    }

    /// Build from a `SummaryTurn` log entry, stripping the normalizing trailing `\n`.
    fn from_summary_turn(
        turn_id: u64,
        turn_type: TurnType,
        text: String,
        covers: Vec<u64>,
        view: Vec<u64>,
        signature: Option<TurnSignature>,
    ) -> Self {
        Self {
            turn_id,
            role: Role::Assistant,
            text: text.trim_end_matches('\n').to_owned(),
            view,
            signature,
            turn_type,
            covers,
        }
    }
}

/// A top-level document in the multi-document YAML conversation log.
///
/// The first document in every conversation file is always `Sequence`.
/// Subsequent documents are `UserTurn`, `AssistantTurn`, and `Tombstone`
/// records in append order.
///
/// **Serialization notes:**
/// - All variants are struct variants to avoid a known serde_yaml 0.9 bug
///   where newtype variants in internally-tagged enums produce a duplicate
///   `kind` field.
/// - Text fields always carry a trailing `\n` on disk so serde_yaml emits
///   the YAML literal block scalar (`|`), making logs easy to read and diff.
///   The trailing `\n` is stripped when constructing [`TurnRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogEntry {
    /// Written once as the first document when a new conversation is
    /// created.  On-disk tag is `conversation` for backwards
    /// compatibility with logs written before the variant was renamed
    /// (the Rust identifier `Sequence` is paired with the explicit
    /// rename so the YAML format stays stable).
    #[serde(rename = "conversation")]
    Sequence {
        /// Ordered segments of the system prompt template — static text blocks
        /// interleaved with named dynamic section placeholders.
        system_prompt: Vec<SystemPromptSegment>,
        /// ISO-like UTC creation timestamp (from [`crate::conversation_log::now_iso`]).
        created_at: String,
    },

    /// A user-side turn (including system messages mapped at role level).
    UserTurn {
        turn_id: u64,
        /// Text stored with trailing `\n` to force YAML literal block scalar style.
        text: String,
        view: Vec<u64>,
    },

    /// An assistant-generated turn, optionally carrying an attentional
    /// provenance fingerprint captured at end-of-decode.
    AssistantTurn {
        turn_id: u64,
        /// Text stored with trailing `\n` to force YAML literal block scalar style.
        text: String,
        view: Vec<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<TurnSignature>,
    },

    /// An async cognitive-pipeline turn: a Sleep dream, Thought flicker, or
    /// Reason plan turn.  Always assistant-generated.
    ///
    /// The `covers` field records which prior `turn_id`s were synthesized or
    /// summarized to produce this turn.  Sub-turns may themselves be Reality
    /// turns, other `SummaryTurn` entries, or a mix — supporting recursive
    /// summarization hierarchies.
    SummaryTurn {
        turn_id: u64,
        /// Cognitive layer / pipeline that produced this turn.
        turn_type: TurnType,
        /// Text stored with trailing `\n` to force YAML literal block scalar style.
        text: String,
        /// Ordered list of `turn_id`s that this turn summarizes or was derived from.
        covers: Vec<u64>,
        view: Vec<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<TurnSignature>,
    },

    /// Marks a previously-written turn as deleted.
    ///
    /// [`SubstrateStore::read_all`] silently suppresses tombstoned turns.
    Tombstone {
        /// The `turn_id` of the turn being deleted.
        turn_id: u64,
    },
}

// ── SubstrateStore ─────────────────────────────────────────────────────────

/// Append-only conversation store backed by a multi-document YAML file.
///
/// Use [`create`](Self::create) to start a new conversation (writes the header
/// document).  Use [`open`](Self::open) to continue an existing one (safe to
/// call immediately after a crash — no header is written).
///
/// Every write is flushed immediately so the file is always valid up to the
/// most recently completed entry.
#[allow(dead_code)]
pub struct SubstrateStore {
    writer: BufWriter<File>,
    path: PathBuf,
    /// Counts only [`append_turn`](Self::append_turn) calls.
    /// Tombstones and the conversation header are not counted.
    turns_written: u64,
}

#[allow(dead_code)]
impl SubstrateStore {
    /// Create a new conversation store, writing the `Sequence` header as
    /// the first YAML document.
    ///
    /// Returns an error if `path` already exists.
    pub fn create(
        path: impl AsRef<Path>,
        system_prompt: Vec<SystemPromptSegment>,
    ) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut store = Self {
            writer: BufWriter::new(file),
            path,
            turns_written: 0,
        };
        store.write_entry(&LogEntry::Sequence {
            system_prompt,
            created_at: crate::conversation_log::now_iso(),
        })?;
        Ok(store)
    }

    /// Open an existing conversation store for appending.
    ///
    /// Does **not** write a header document.  If `path` does not exist the
    /// file is created; if it exists the writer is positioned at the end.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path,
            turns_written: 0,
        })
    }

    /// Append a turn record to the log and increment `turns_written`.
    pub fn append_turn(&mut self, record: TurnRecord) -> crate::Result<()> {
        self.write_entry(&record.into_log_entry())?;
        self.turns_written += 1;
        Ok(())
    }

    /// Append a tombstone marking `turn_id` as deleted.
    ///
    /// Does **not** increment `turns_written`.  The next [`read_all`](Self::read_all)
    /// call will exclude the tombstoned turn.
    pub fn append_tombstone(&mut self, turn_id: u64) -> crate::Result<()> {
        self.write_entry(&LogEntry::Tombstone { turn_id })
    }

    /// Read and return all live (non-tombstoned) turn records in append order.
    ///
    /// Tombstones suppress their target regardless of position — a tombstone
    /// appearing after its turn suppresses that turn; a tombstone for a
    /// turn that was never written is silently ignored.
    pub fn read_all(path: impl AsRef<Path>) -> crate::Result<Vec<TurnRecord>> {
        let text = std::fs::read_to_string(path)?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut turns: Vec<TurnRecord> = Vec::new();
        let mut tombstones: HashSet<u64> = HashSet::new();

        for doc in serde_yaml::Deserializer::from_str(&text) {
            // Break on any parse/deserialization error rather than propagating
            // it.  Given the flush-after-every-write contract, the only realistic
            // failure is a truncated final document written before a crash or
            // kill signal.  All successfully flushed documents precede it.
            let entry = match LogEntry::deserialize(doc) {
                Ok(e) => e,
                Err(_) => break,
            };
            match entry {
                LogEntry::Sequence { .. } => {}
                LogEntry::UserTurn {
                    turn_id,
                    text,
                    view,
                } => {
                    turns.push(TurnRecord::from_user_turn(turn_id, text, view));
                }
                LogEntry::AssistantTurn {
                    turn_id,
                    text,
                    view,
                    signature,
                } => {
                    turns.push(TurnRecord::from_assistant_turn(
                        turn_id, text, view, signature,
                    ));
                }
                LogEntry::SummaryTurn {
                    turn_id,
                    turn_type,
                    text,
                    covers,
                    view,
                    signature,
                } => {
                    turns.push(TurnRecord::from_summary_turn(
                        turn_id, turn_type, text, covers, view, signature,
                    ));
                }
                LogEntry::Tombstone { turn_id } => {
                    tombstones.insert(turn_id);
                }
            }
        }

        if tombstones.is_empty() {
            Ok(turns)
        } else {
            Ok(turns
                .into_iter()
                .filter(|t| !tombstones.contains(&t.turn_id))
                .collect())
        }
    }

    /// Path to the store file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of turn records appended during this session.
    pub fn turns_written(&self) -> u64 {
        self.turns_written
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn write_entry(&mut self, entry: &LogEntry) -> crate::Result<()> {
        // serde_yaml 0.9 does not reliably emit a `---` document-start marker.
        // Without it, consecutive writes merge into one YAML mapping with
        // duplicate field names.  Normalize: always emit exactly one `---\n`
        // per entry, stripping any leading `---\n` that serde_yaml may have
        // added, then writing our own.
        let yaml = serde_yaml::to_string(entry)?;
        let body = yaml.strip_prefix("---\n").unwrap_or(&yaml);
        write!(self.writer, "---\n{}", body)?;
        self.writer.flush()?;
        Ok(())
    }
}
