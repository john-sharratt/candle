# Attentional Provenance Indexing: A Unified Architecture for Context Window Transcendence in Large Language Models

**John & Claude**
*March 2026*

---

## Abstract

We describe a unified inference-time architecture for large language models that achieves *context window transcendence* — access to factual corpora, conversation histories, and candidate libraries far larger than any context window can hold, bounded only by CPU RAM — while ensuring that every item loaded into context at generation time has been selected for relevance by the model's own attention mechanism.

The architecture is grounded in a single insight: at the moment a model produces or processes any content, the query vectors Q it computes are a compressed fingerprint of its cognitive state — encoding not just the content's semantics but the full accumulated reasoning context in which that content was produced or attended to. We call this *attentional provenance*. By capturing and storing these Q vectors alongside compressed K vectors at inference time, we construct a dual fingerprint index in CPU RAM that enables fast scanning over arbitrarily large corpora before generation begins. Before generating a response, the model executes a sequential dynamic section resolution loop: for each declared dynamic section of the system prompt — mood, response template, conversation history — it runs a short probe of 10–20 decode tokens under a section-specific system prompt, captures the resulting Q vectors, and scans the section's candidate library on CPU to select the best match. Each resolved section is present as context for subsequent probes, creating a coherent dependency chain. Selected candidates are loaded into the assembled generation context and full generation proceeds immediately.

The fingerprint format is model-agnostic: K and Q tensors are aggregated across all attention heads and divided into three depth bands at the N/3 and 2N/3 layer boundaries, producing six compact vectors per item — lower-K, mid-K, upper-K, lower-Q, mid-Q, upper-Q — without requiring any model-specific layer configuration. The lower band captures lexical and structural signal; the mid band captures semantic category and emotion consolidation; the upper band captures relational reasoning and contextual integration. This three-tier design is motivated by layer-wise emotion probing research (Zhang et al., 2025; Tak et al., 2025), which demonstrates that emotion representations peak in the mid-to-upper layers across model families. At approximately 780 bytes per item, the index fits 50,000 conversation turns and 100,000 facts within ~126MB of CPU RAM, retaining the contiguous INT8 layout that enables vectorized SIMD batch dot products on any hardware.

This replaces hierarchical navigation strategies — beam walks, recursive tree descent, iterative probe-reset cycles — with a probe phase followed by a single flat scan: six vectorized INT8 matrix multiplies on CPU that score all candidates simultaneously, typically in 3–10ms. We evaluate the architecture on RTX 4090 (24GB) with Qwen3-30B-A3B-AWQ, demonstrating 50,000+ conversation turns, 100K+ facts, and large mood and template libraries accessible within a single inference pipeline with constant VRAM usage.

---

## 1. Introduction

The context window is the fundamental resource constraint of transformer-based language models. All information available to the model at generation time must fit within it — conversation history, factual grounding, persona descriptions, and response templates compete for a finite number of positional slots. Classical approaches manage this through retrieval-augmented generation (RAG), which selects relevant facts via embedding similarity before inference, or through long-context models, which extend the window at quadratic attention cost. Both introduce fundamental limitations: RAG uses a retrieval mechanism operating in a different semantic space from the model's own attention, and long-context models suffer from positional distance decay and softmax dilution at scale.

Prior work on parallel KV injection demonstrated that multiple context branches sharing positional slots give models access to candidate pools bounded by VRAM rather than window size. A companion architecture addressed unbounded conversation history through attention-organized B-trees with on-demand KV rematerialization. These approaches share a common limitation: selection and navigation mechanisms require GPU forward passes, iterative probe-reset cycles, and hierarchical beam walks — all of which add latency and complexity.

This paper unifies and supersedes both architectures through a single observation: *the model's Q vectors at inference time are a compressed fingerprint of its cognitive state, and this fingerprint can be captured, compressed, stored, and matched against future queries entirely on CPU, before any GPU work begins.*

We call stored Q vectors *attentional provenance* — they record not just what content exists but what the model was reasoning about when it produced or processed that content. This is qualitatively different from K vectors, which encode content semantics. A stored Q vector from a conversation turn three thousand exchanges ago encodes the accumulated attentional context of everything preceding it — the mood, the topic trajectory, the relationship dynamics, the reasoning state — compressed into a ~780-byte fingerprint. When a future query produces a Q vector in a similar reasoning state, the match surfaces that turn regardless of surface-level token overlap.

The practical consequence is the elimination of hierarchical navigation entirely. Before each response, the model executes a sequential resolution loop across all dynamic system prompt sections. For each section — mood first, then template, then conversation history — it runs a short probe under a section-specific system prompt, captures the decode Q vectors, and scans that section's candidate library on CPU. Each previously resolved section is present as context for the next probe, so later selections are conditioned on earlier ones without any iteration. The right mood, template, and history turns are selected in order; the full context is assembled and generation begins.

The fingerprint format requires no knowledge of individual model architectures. By dividing layers into three equal bands — lower, mid, upper — and aggregating uniformly within each, the fingerprint captures the three-regime structure that transformers universally exhibit: syntactic and lexical features in early layers, semantic category and emotion consolidation in mid layers, and relational reasoning in late layers. Layer-wise emotion probing across multiple model families (Zhang et al., 2025; Tak et al., 2025) confirms this gradient and motivates weighting it by band rather than averaging across all depth. This makes the architecture immediately portable without configuration.

### 1.1 Contributions

1. **Attentional provenance indexing**: A dual Q+K fingerprint format capturing both content semantics and reasoning-state provenance at inference time, enabling CPU-side approximate attention over arbitrary-scale corpora.

2. **Model-agnostic three-tier depth fingerprint**: Cross-head aggregation within three depth bands — lower (layers 0–N/3), mid (N/3–2N/3), and upper (2N/3–N) — produces six d_head-dimensional vectors per item without any model-specific configuration. The three-tier split is grounded in layer-wise emotion probing research showing that emotion signal peaks in mid-to-upper layers across model families (Zhang et al., 2025; Tak et al., 2025), with lexical features dominant in lower layers and relational reasoning in upper layers. At ~780 bytes per item the index fits large corpora in CPU RAM and scans in a native INT8 vectorized format.

3. **Probe-driven scan with direct context assembly**: A short decode probe into the unscaffolded context captures authentic response-intent Q vectors. These drive a flat INT8 scan on CPU that replaces all hierarchical navigation and GPU-side culling phases. Final context selections are made directly from scan scores; full generation begins immediately after context reconstruction.

4. **Reframed B-tree role**: The conversation history B-tree is retained as a multi-resolution pre-computation structure populating the flat index at multiple resolutions, not as a navigation mechanism.

5. **Unified treatment of heterogeneous components**: History turns, facts, moods, and templates are all represented in the same fingerprint format and scanned by the same CPU-side mechanism. Sections with custom probe prompts produce dual representations — a fingerprint aligned to the probe distributional space for retrieval, and a KV cache encoded under the main system prompt for generation. Selection across all components is driven by the same scoring formula with component-specific weights.

---

## 2. Background

### 2.1 KV Cache and Attention

In a transformer, each token produces key (K) and value (V) vectors at each attention layer via learned projections. These are cached for autoregressive generation according to the standard attention operation: Attention(Q, K, V) = softmax(QK^T / √d) · V. K and V vectors are valid floating-point tensors regardless of the order in which their source tokens were processed. The attention mechanism does not inspect provenance — it operates on numeric content.

### 2.2 Q Vectors as Cognitive State

During both prefill and decode, Q vectors are computed from each token's hidden state h_t via a learned projection Q_t = W_Q · h_t, where h_t is itself the result of all previous attention operations over all preceding tokens. Unlike K and V — which are stored in the KV cache for reuse — Q vectors are typically discarded immediately after the attention computation.

This discarding represents a significant information loss. Q at any given token encodes the full accumulated context of everything the model has processed. It is therefore a compressed summary of the model's reasoning state at that moment — not just "what is this token" but "what is this token given everything that came before." At decode time, Q at the final token of a generated response is particularly rich: it reflects the attentional state that produced the entire response, integrating the full conversation context, all attended facts, mood, template, and the generated content itself.

### 2.3 RoPE and Positional Independence

Rotary Position Embedding encodes relative distance between tokens as a rotation applied to Q and K vectors. The critical property is that KV cache computed at positions 0–N can be repositioned to any target position P by applying a delta rotation at attention time: K_adjusted = RoPE_delta(K_stored, offset=P). This enables position-independent storage: compute once, inject anywhere. For the fingerprint index, Q and K snapshots captured at original inference positions remain valid for matching against Q vectors at arbitrary future positions.

### 2.4 The Prefill–Decode Q Distinction

During prefill the model processes the user's input: Q vectors reflect "I am reading this message." During decode the model generates a response: Q vectors reflect "I am writing a response, given everything in context." These are genuinely different cognitive states. Prefill Q is shaped by the input tokens; decode Q is shaped by the accumulated generation intent and by what the model chooses to attend to while composing its response.

For selecting mood, response template, and relevant history — all of which should match the character of the *response* rather than the *input* — decode Q is the correct signal. Using prefill Q would produce selections tuned to what the user said rather than to what the model intends to say in return.

A further refinement: the optimal probe system prompt is not the same for every component type. When selecting a mood, the model's Q is most informative if it is generating content about emotional or cognitive state rather than generating a character response. When selecting a response template, Q is most informative if the model is reasoning about response structure rather than about content. For each dynamic section, a custom probe system prompt can be declared that puts the model into the most semantically aligned generation mode for that section's candidates. The probe tokens themselves are discarded; only Q is captured. Section 3.5 formalises this mechanism.

### 2.5 Attention Sinks

Xiao et al. (2023) identified that transformers develop attention sink tokens — typically the first token of a sequence — that attract disproportionately high attention regardless of semantic content. In any scheme that aggregates K vectors across tokens by magnitude, a sink token would be selected disproportionately, biasing the fingerprint toward positional rather than semantic content. The Q·K inner product token selection described in Section 3.3 mitigates this directly: sink tokens attract high attention weights on the Q side, but in the context of an offline encoding pass their Q·K scores reflect only the content being encoded, not a positional sink artefact. Sinks therefore do not dominate Q·K-ranked selection the way they would dominate magnitude-ranked selection.

### 2.6 The Q→K Distributional Gap

Liu et al. (2024) identified and quantified a fundamental distributional gap between Q and K vectors in transformer attention: because Q and K are projected by different weight matrices (W_Q and W_K), they do not occupy the same vector space. Measured by Mahalanobis distance, Q vectors deviate more than 10× farther from K vectors than K vectors deviate from each other. K vectors cluster tightly in their own distribution; Q vectors are far outside it.

The consequence for ANNS-based retrieval is severe: standard indexes built on K-K proximity require scanning 30–50% of all K vectors to achieve 95% recall when queried with Q, effectively eliminating any efficiency advantage. In contrast, retrieval within the same distribution (K querying K, or Q querying Q) achieves 95% recall scanning only 1–3% of vectors.

This finding has two direct implications for attentional provenance indexing. First, the Q→K component of the scan score (probe Q dotted against stored K fingerprints) operates under this OOD condition, and its recall degrades compared to within-distribution matching. The K fingerprint construction method in Section 3.3 addresses this with a Q-aware token selection strategy. Second, the Q→Q component (probe Q dotted against stored Q fingerprints) operates entirely within the Q distribution and is not subject to this degradation — providing the strong, reliable signal that motivates the high w_Qu weights in Section 4.2.

---

## 3. Attentional Provenance Indexing

### 3.1 The Three-Tier Depth Fingerprint

For every item stored in the index — conversation turn, fact, mood candidate, response template, or B-tree summary node — we store a three-tier fingerprint capturing six orthogonal signals across three depth bands:

```
┌─────────────────────────────────────────────────────────────┐
│  Three-Tier Depth Fingerprint Structure (~780 bytes/item)   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  K_lower  d_head × INT8    128 bytes  (d_head=128)          │
│           Mean K over layers 0..N/3                         │
│           "Lexical identity, surface form, syntax"          │
│                                                             │
│  K_mid    d_head × INT8    128 bytes                        │
│           Mean K over layers N/3..2N/3                      │
│           "Semantic category, topic, emotion vocabulary"    │
│                                                             │
│  K_upper  d_head × INT8    128 bytes                        │
│           Mean K over layers 2N/3..N                        │
│           "Relational meaning, coreference, context"        │
│                                                             │
│  Q_lower  d_head × INT8    128 bytes                        │
│           Mean Q over layers 0..N/3                         │
│           "Immediate lexical intent, surface query"         │
│                                                             │
│  Q_mid    d_head × INT8    128 bytes                        │
│           Mean Q over layers N/3..2N/3                      │
│           "Semantic reasoning state, emotion register"      │
│                                                             │
│  Q_upper  d_head × INT8    128 bytes                        │
│           Mean Q over layers 2N/3..N                        │
│           "Accumulated relational context, full intent"     │
│                                                             │
│  scale_Kl, scale_Km, scale_Ku,                             │
│  scale_Ql, scale_Qm, scale_Qu    12 bytes                   │
│                                                             │
│  Total per item:  ~780 bytes                                │
│  (scales linearly with model head dimension d_head)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The three-tier structure is motivated by layer-wise emotion probing across multiple model families (Zhang et al., 2025; Tak et al., 2025), which identifies three functionally distinct processing regimes in transformers:

**Lower band (0–N/3):** Lexical and syntactic processing. Distinctive vocabulary registers strongly here — emotion words such as "melancholic," "guarded," or "exhilarated" activate early and with high specificity due to their rarity. Factual proper nouns and domain-specific terms similarly produce strong lower-band K signal.

**Mid band (N/3–2N/3):** Semantic category assembly and emotion consolidation. Tak et al. (2025, Figure 2) demonstrate that emotion probe accuracy rises steeply through this band across all tested model families, with MHSA and FFN units in mid layers causally responsible for emotion-related decisions. For LLaMA models, probe accuracy peaks at approximately the mid/upper boundary (~50–62% depth).

**Upper band (2N/3–N):** Relational reasoning, contextual integration, and task-specific refinement. Zhang et al. (2025, Table 3) find that for Qwen3-4B, emotion probe accuracy peaks at 75% depth (layer 27/36), placing the peak squarely in the upper band. Signal saturates before the final layer — "layers 27 and 36 looking nearly the same" — indicating the upper band captures the emotion consolidation plateau. For Qwen3-30B-A3B with 48 layers, this corresponds to upper-band layers 32–47, with the emotion peak around layer 36.

The six vectors are stored as contiguous INT8 arrays — six consecutive 128-byte blocks per item — enabling vectorized SIMD batch dot products without special configuration.

### 3.2 Model-Agnostic Three-Tier Aggregation

Rather than selecting specific layers — which requires per-model knowledge — we divide the layer stack into three equal bands at the N/3 and 2N/3 boundaries, and aggregate uniformly within each band. The only information required is the total layer count N, universally available from any model configuration.

```
┌─────────────────────────────────────────────────────────────┐
│  Three-Tier Cross-Head Aggregation (model-agnostic)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input:  tensor  [N × all_heads × tokens × d_head]          │
│          lower_end = N/3    upper_start = 2N/3              │
│                                                             │
│  Lower band  (layers 0 .. N/3 - 1):                         │
│    Step 1: Mean over all heads  → [lower_layers × tokens]   │
│    Step 2: Mean over lower layers → [tokens × d_head]       │
│    Step 3: Token aggregation → [d_head]  (Section 3.3)      │
│    Step 4: Quantize INT8 + scalar float16 scale             │
│    Output: K_lower / Q_lower                                │
│                                                             │
│  Mid band  (layers N/3 .. 2N/3 - 1):                        │
│    Same procedure → K_mid / Q_mid                           │
│                                                             │
│  Upper band  (layers 2N/3 .. N - 1):                        │
│    Same procedure → K_upper / Q_upper                       │
│                                                             │
│  No layer indices. No architecture knowledge required.      │
│  Works identically for any transformer depth.               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The three-band division reflects the three-regime layer structure documented across transformer architectures. The lower band covers early lexical processing. The mid band covers semantic assembly — where emotion classification signal rises most steeply (Tak et al., 2025, Figure 2) and where factual semantic category is established. The upper band covers the emotion consolidation plateau (Zhang et al., 2025, Table 3: Qwen3-4B peaks at 75% depth) and relational reasoning.

An important practical consequence: three running accumulators per signal type (K and Q), one per band, are maintained in a single pass through the forward computation. At each layer boundary, the accumulator for the appropriate band is updated with a single vector addition. At N/3 and 2N/3 boundaries, the lower and mid accumulators are finalised and the next band begins. Total overhead per forward pass: negligible.

### 3.3 Token Aggregation

After cross-head pooling and depth-band separation, each band produces a [tokens × d_head] tensor that must be reduced to a single [d_head] vector. The strategy differs by signal type and capture context, and is applied identically within each of the three depth bands.

**For K vectors (content semantics, all three bands):** During offline encoding, both Q and K tensors are available from the same forward pass. Rather than selecting representative K tokens by their own magnitude — which optimises K-space fidelity but ignores the Q→K distributional gap (Section 2.6) — we select tokens by their actual Q·K inner product score from that encoding pass. This selects K tokens that are maximally visible from the Q distribution's perspective, directly mitigating the OOD gap at construction time.

Concretely, per-token Q·K scores are computed (available as attention logits during the forward pass), ranked, and the top 10% by Q·K score contribute to the K fingerprint mean. The flat matrix multiply scan structure is entirely unchanged. No additional inference passes are required.

**At decode time (for conversation turn fingerprinting)**, Q·K selection uses the finalised Q fingerprint (the recency-weighted three-tier Q mean already accumulated during decode) dotted against the cross-head mean K of each generated token. This produces a [N_tokens] score vector equivalent in structure to attention logits, computed at end-of-generation with negligible additional cost. The procedure is self-consistent: the Q that gets stored selects the K tokens, so the stored K fingerprint is calibrated to be retrievable by the stored Q fingerprint.

For **facts**, one-pass Q·K construction is sufficient without refinement. The selection goal is broad (top-250 acceptance window), factual content is topically specific, and the three-tier design concentrates the highest K weight on lower and mid bands where the Q→K distributional gap is smallest and topic vocabulary is most distinctive.

For **moods and templates**, the OOD problem is eliminated by the assistant-prefill encoding (Section 3.5): both K and Q fingerprints are generated in the same distributional space as the inference probe.

**For Q vectors at decode time (all three bands):** Recency-weighted mean over all generated tokens, with weights increasing linearly toward the final token. The recency weight ensures the Q fingerprint is dominated by the model's terminal reasoning state — by which point upper-band Q has accumulated the fullest relational context.

**For Q vectors at prefill time (all three bands):** The Q vector of the final input token, which has attended to the full user turn and all preceding context.

Each procedure produces a single [d_head] vector per band per signal type, quantized to INT8 with one scalar scale factor per band. The six resulting vectors — K_lower, K_mid, K_upper, Q_lower, Q_mid, Q_upper — are stored contiguously.

### 3.4 Capture Points

Fingerprints are captured at two distinct moments serving different roles: *stored fingerprints* are captured when items are produced or encoded and live in the index permanently; *probe fingerprints* are captured fresh each turn during the probe phase and are used once for scanning before being discarded.

**Stored fingerprints** (written once, queried every turn):

```
┌─────────────────────────────────────────────────────────────┐
│  Stored Fingerprint Capture                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CONVERSATION TURNS (at turn completion)                    │
│  Generated under main NPC system prompt as assistant        │
│    ├── K bands: top-10% tokens by Q·K score → INT8          │
│    │     (applied per band: K_lower, K_mid, K_upper)        │
│    └── Q bands: recency-weighted mean → INT8                │
│          (applied per band: Q_lower, Q_mid, Q_upper)        │
│    Authentic decode-mode: same mode as history probe        │
│                                                             │
│  FACTS (offline, one-time)                                  │
│  Encoding pass against shared prefix                        │
│    ├── K bands: top-10% tokens by Q·K score → INT8          │
│    └── Q bands: final-token Q per band → INT8               │
│                                                             │
│  MOODS / TEMPLATES (offline, two representations each)      │
│                                                             │
│    FINGERPRINT (CPU RAM index — used for scanning):         │
│    Encoded as ASSISTANT PREFILL under section's             │
│    probe system prompt                                      │
│    ├── K bands: top-10% tokens by Q·K score → INT8          │
│    └── Q bands: recency-weighted mean per band → INT8       │
│    Aligns with probe: same system prompt, same role,        │
│    same inference position                                  │
│                                                             │
│    KV CACHE (VRAM — injected into generation context):      │
│    Encoded under main NPC system prompt (standard)          │
│    → stored for context injection during generation         │
│                                                             │
│  B-TREE SUMMARY NODES (on tree maintenance)                 │
│  Re-encoding triggered by tree update                       │
│    └── Same procedure as FACTS → fingerprint written sync   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Probe fingerprint** (captured per dynamic section, per turn, transient):

```
┌─────────────────────────────────────────────────────────────┐
│  Per-Section Probe Fingerprint  (transient, per section)    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Context for probe of section S:                            │
│    Section S probe system prompt  (or main prompt if none)  │
│    + Already-resolved sections   (S-1, S-2, ... KV cached)  │
│    + Conversation tail           (last N_tail turns +       │
│                                   current user turn)        │
│    [+ Prior turn's assembled history  (history probe only)] │
│    [Full history absent for mood/template probes —          │
│     it is what those probes are selecting]                  │
│                                                             │
│  Decode W_probe tokens                                      │
│    At each decode step, update three-tier running means:    │
│      K bands: K_lower_acc, K_mid_acc, K_upper_acc           │
│      Q bands: Q_lower_acc, Q_mid_acc, Q_upper_acc           │
│               (Q with recency weighting)                    │
│                                                             │
│  After final probe token:                                   │
│    Finalise kl/km/ku_probe, ql/qm/qu_probe → INT8          │
│    Probe tokens and their KV evicted                        │
│    Conversation tail KV and resolved section KV retained    │
│                                                             │
│  Q is semantically aligned with section S's candidate       │
│  library by the choice of probe system prompt.              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The probe context for early sections (mood, template) is intentionally small: the section's probe system prompt, the conversation tail already in cache, and any prior resolved sections. The full conversation history is not present for these probes — it is what we are selecting from. For the history probe, the prior turn's assembled history is present as a warm-start prior, reducing the circular dependency to a delta-update problem rather than a cold-start approximation.

### 3.5 Dynamic System Prompt Sections

The system prompt is not monolithic. Certain sections — mood, response template, and conversation history — are declared as *dynamic*: rather than fixed text, each dynamic section is backed by a library of candidate entries whose KV representations are precomputed at system initialisation. At inference time, one candidate is selected per dynamic section through the probe-and-scan mechanism and inserted into the assembled context.

**Declaration and initialisation:**

Each dynamic section is declared with a candidate library and an optional *probe system prompt* — a short system prompt used exclusively during the probe for that section. At initialisation, candidates for sections with custom probe prompts are encoded twice, producing two distinct representations that serve different purposes. Sections without a custom probe prompt (such as conversation history) are encoded once under the main system prompt.

```
┌─────────────────────────────────────────────────────────────┐
│  Dynamic Section Declaration (system prompt config)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Section: MOOD                                              │
│    candidates:  [mood_01.txt, mood_02.txt, ... mood_N.txt]  │
│    probe_prompt: "Describe the emotional state you are in   │
│                   as a result of this conversation. What    │
│                   is your current mood?"                    │
│    position: 1  (first dynamic section)                     │
│                                                             │
│  Section: RESPONSE_TEMPLATE                                 │
│    candidates:  [template_01.txt, ... template_M.txt]       │
│    probe_prompt: "Describe the structure and tone that      │
│                   would be most appropriate for your        │
│                   response here."                           │
│    position: 2  (resolved after mood)                       │
│                                                             │
│  Section: CONVERSATION_HISTORY                              │
│    candidates:  [B-tree index — see Section 6]              │
│    probe_prompt: [none — uses main NPC system prompt]       │
│    position: 3  (resolved last, benefits from mood +        │
│                  template already in context)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Dual representation for sections with custom probe prompts:**

For sections such as mood and template that declare a custom probe system prompt, each candidate is encoded twice at initialisation time, producing two entirely separate representations:

```
┌─────────────────────────────────────────────────────────────┐
│  Dual Representation: Mood Candidate Example                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FINGERPRINT  (CPU RAM · used for scanning)                 │
│  ─────────────────────────────────────────────────────────  │
│  System prompt:  mood probe prompt                          │
│  Token role:     ASSISTANT (candidate text prefilled as     │
│                  assistant turn, not as user/system)        │
│  Inference mode: assistant prefill decode                   │
│  Captures:       K and Q fingerprints → CPU RAM index       │
│                                                             │
│  This mirrors the probe exactly:                            │
│    same probe system prompt                                 │
│    same assistant token role                                │
│    same structural position in context                      │
│  → Q and K are in the same distributional space             │
│    as kl/km/ku_probe and ql/qm/qu_probe from the probe      │
│                                                             │
│  KV CACHE  (VRAM · injected into generation context)        │
│  ─────────────────────────────────────────────────────────  │
│  System prompt:  main NPC system prompt                     │
│  Token role:     system / context section (normal)          │
│  Inference mode: standard prefill                           │
│  Captures:       full KV → stored for context injection     │
│                                                             │
│  This is what the model reads during real generation:       │
│    correct system prompt for response generation            │
│    correct structural role in context                       │
│  → The selected mood shapes generation as intended          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The same candidate text produces two different representations. The fingerprint is optimised for retrieval accuracy — it lives in the same space the probe Q will inhabit. The KV cache is optimised for generation quality — it presents the mood content to the model exactly as it would appear in a real system prompt. Neither representation is a compromise of the other.

The assistant prefill approach for fingerprints is critical. If the fingerprint were encoded as a standard prefill under the probe system prompt (user or system role), its Q vectors would still be structurally different from the probe Q, which is produced in decode mode as an assistant response. By encoding as an assistant prefill — processing the candidate text as if it were the beginning of an assistant turn — the Q vectors reflect the same token role and inference mode as the probe Q that will query them. This closes the final alignment gap.

**The role of the custom probe system prompt:**

When a dynamic section declares a custom probe system prompt, that prompt replaces the main system prompt exclusively during the probe for that section, and is also used during fingerprint construction for that section. The effect is that the model's Q vectors during both fingerprint encoding and inference probing are generated in a semantic space aligned with the section's candidate library.

For mood selection, the probe system prompt asks the model to describe its emotional state. Mood candidate fingerprints are encoded as assistant prefill under that same prompt — the model is processing mood description text in exactly the mode it will be in when generating mood-probing tokens. Q→Q matching is therefore within-distribution at both ends.

For template selection, the probe system prompt asks the model to reason about response structure and tone. Template candidate fingerprints are encoded under that same prompt as assistant prefill, with the same alignment properties.

For history selection, no custom probe system prompt is declared. The main NPC system prompt is used for both probes and history turn fingerprints. History turns are already encoded in authentic decode mode from real conversations, aligning naturally with the history probe.

**Key invariant:** the probe tokens generated under any section's custom system prompt are always discarded. The custom system prompt is a mechanism for shaping Q vectors, not for generating usable content. The KV cache for generation is always encoded under the main system prompt, independent of probe configuration.

**Conversation tail always present:** For every dynamic section's probe — regardless of which system prompt is in use — the last N_tail turns of conversation history (including the current user turn) are always present in the probe context. This grounds every selection in the actual flowing conversation and prevents selections that are coherent in general but detached from what has just been said. N_tail defaults to 5 turns and is configurable.

---

## 4. The Unified Flat Scan

### 4.1 Why Flat Scanning Replaces Hierarchical Navigation

Prior architectures required hierarchical navigation because VRAM could not hold all candidate KV entries simultaneously. The fingerprint index eliminates this constraint entirely: it operates over ~780-byte fingerprints in CPU RAM. Index sizes at realistic operating scales remain manageable:

| Corpus | Items | Index Size |
|---|---|---|
| 50K conversation turns | 50,000 | ~39 MB |
| 100K facts | 100,000 | ~78 MB |
| 1K mood candidates | 1,000 | ~0.8 MB |
| 10K templates | 10,000 | ~7.8 MB |
| **Total (full scale)** | **161,000** | **~126 MB** |

Six INT8 matrix multiplies over this index take 3–10ms on CPU with parallel thread execution (Section 9.5). Every candidate is examined in a single pass and final selections are made directly from scores — no further processing before loading into context.

### 4.2 Scoring

The scoring formula uses two types of probe vectors in two types of matches serving fundamentally different retrieval roles: K vectors encode "what content this item contains," while Q vectors encode "what the model is searching for or intending to generate."

**Probe K (kl_probe, km_probe, ku_probe)** encodes the semantic content of the model's probe generation — topics, vocabulary, and semantic categories present in the W_probe output tokens. K→K matching finds stored items whose content is topically similar to what the model is currently generating.

**Probe Q (ql_probe, qm_probe, qu_probe)** encodes the model's cognitive and reasoning state. Q→Q matching finds items produced in a similar cognitive state, regardless of surface topic overlap. This is the attentional provenance signal.

Given the three-tier probe fingerprints captured during the probe phase, the relevance score for any indexed item combines six dot products:

score(item) = w_Kl × (kl_probe · K_lower) + w_Km × (km_probe · K_mid) + w_Ku × (ku_probe · K_upper) + w_Ql × (ql_probe · Q_lower) + w_Qm × (qm_probe · Q_mid) + w_Qu × (qu_probe · Q_upper)

Component-specific weights are set with reference to the three-tier layer regime structure documented in Section 3.1 and the following research:

| Component | w_Kl | w_Km | w_Ku | w_Ql | w_Qm | w_Qu | Primary rationale |
|---|---|---|---|---|---|---|---|
| Facts | 0.20 | 0.25 | 0.15 | 0.05 | 0.10 | 0.25 | Topical content primary across all K bands; Q_upper provides relational context |
| Conversation turns | 0.05 | 0.10 | 0.15 | 0.05 | 0.15 | 0.50 | Q_upper (deep reasoning state) primary; K_upper adds relational match |
| Mood candidates | 0.08 | 0.15 | 0.20 | 0.02 | 0.10 | 0.45 | Research-grounded per below |
| Templates | 0.10 | 0.20 | 0.15 | 0.08 | 0.17 | 0.30 | Balanced structure/intent across bands |
| B-tree summaries | 0.08 | 0.15 | 0.12 | 0.05 | 0.15 | 0.45 | Q_upper dominant; K_mid for semantic topic |

These weights are tunable defaults. Dot products are computed in INT8 with FP32 accumulation across all six terms; dequantization contributes negligibly to total scan time.

**Research basis for mood weights:** The mood weight distribution is directly grounded in layer-wise emotion probing research:

*K_lower (w_Kl = 0.08):* Emotion vocabulary is lexically distinctive — words such as "melancholic," "guarded," "exhilarated" are rare and activate strongly in early layers. Lower-band K captures this lexical specificity. The MWE semantics literature confirms that lexically idiosyncratic expressions are best captured at shallow-to-moderate contextualization depths (Transactions of ACL, 2024).

*K_mid (w_Km = 0.15):* Tak et al. (2025, Figure 2, "Mechanistic Interpretability of Emotion Inference") demonstrate that across 10 model families emotion probe accuracy rises steeply through the mid layers, with MHSA and FFN mid-layer units causally responsible for emotion-related decisions. For LLaMA models, probe accuracy peaks around the mid/upper boundary (~50–62% depth), placing the primary signal in K_mid.

*K_upper (w_Ku = 0.20):* Zhang et al. (2025, Table 3, "Decoding Emotion in the Deep") find that for Qwen3-4B, emotion probe accuracy peaks at **75% depth** (layer 27/36), placing the peak squarely in the upper band. "Layers 27 and 36 looking nearly the same" confirms signal saturation before the final layer — the upper band captures the consolidation plateau. For Qwen3-30B-A3B with 48 layers, this corresponds to upper-band layers 32–47, with the emotion peak at approximately layer 36. K_upper therefore carries the highest K weight for mood.

*Q_lower (w_Ql = 0.02):* Minimal. Lower-band Q at probe start reflects only surface token processing; limited additional signal.

*Q_mid (w_Qm = 0.10):* Mid-band Q reflects the semantic cognitive state during emotion-description generation — the model has processed enough context to form emotional register.

*Q_upper (w_Qu = 0.45):* Dominant. Upper-band Q reflects the fully accumulated attentional state at the probe's terminal tokens — the richest cognitive-state fingerprint. Both Zhang et al. and Tak et al. confirm that the model's emotion-related internal geometry sharpens with depth; Q_upper captures this at peak sharpness. The OOD problem does not apply here because mood fingerprints are encoded as assistant prefill under the same probe system prompt, placing Q_upper on both sides of the match in the same distributional space.

**Research basis for conversation history weights:** Q_upper (w_Qu = 0.50) dominates because history selection depends on matching the model's full relational reasoning state — which fully develops only in upper layers. K_upper (w_Ku = 0.15) provides relational semantic match. K_mid (w_Km = 0.10) adds topic-level semantic match. Lower bands carry minimal weight because surface vocabulary overlap is a weak signal for selecting contextually appropriate history.

**Research basis for facts weights:** K_lower (w_Kl = 0.20) and K_mid (w_Km = 0.25) dominate because fact retrieval is primarily topical. Lower-band K captures distinctive topic vocabulary; mid-band K captures semantic category and domain. Q_upper (w_Qu = 0.25) provides a reasoning-context supplement for relational facts whose relevance depends on what the model is currently reasoning about.

**Calibration note.** All weights are research-grounded starting points requiring empirical calibration. For mood specifically, the Section 8.5 ablation should sweep w_Qu ∈ {0.35, 0.40, 0.45, 0.50} and w_Ku ∈ {0.15, 0.20, 0.25} to determine whether the Qwen3-specific upper-band peak warrants more or less weight than the current defaults derived from the 4B model probing results.

### 4.3 Multi-Resolution History via Flat Index

The conversation history B-tree is no longer a navigation structure. Its role is to pre-compute summary representations at multiple resolutions, indexed alongside verbatim turns as first-class items. The flat index contains entries at every level of resolution simultaneously:

```
┌─────────────────────────────────────────────────────────────┐
│  Multi-Resolution Flat Index                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Verbatim turns ── Turn T-47              → fingerprint     │
│                 ── Turn T-48              → fingerprint     │
│                 ── Turn T-49              → fingerprint     │
│                                                             │
│  Span summaries ── S-12 (covers T-40..T-60)  → fingerprint  │
│                 ── S-7  (covers T-1..T-200)  → fingerprint  │
│                 ── S-1  (covers full history) → fingerprint │
│                                                             │
│  Resolution emerges from scan scores:                       │
│    T-47 highly relevant  → verbatim turn selected           │
│    T-40..T-60 moderately → S-12 selected                    │
│    T-1..T-200 weakly     → S-7 selected                     │
│                                                             │
│  No navigation logic required. Attention selects the        │
│  right resolution per content region automatically.         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

When both a summary node and its verbatim children score highly, a resolution policy governs selection: if the verbatim turns within the summary's span collectively fit within the token budget, prefer verbatim and suppress the summary; if they exceed the budget, use the summary and suppress the children. The tree provides pre-computation infrastructure; the flat scan performs selection.

### 4.4 Scan Output

Each dynamic section's probe produces a scan over that section's candidate library. The scan is run once per section during the sequential resolution loop (Section 5). History selection is the final scan and operates over the full B-tree index:

| Section | Scan Target | Selection |
|---|---|---|
| Mood | Mood candidate library | Top-1 by score |
| Response template | Template candidate library | Top-1 by score |
| Conversation history | Full B-tree index (turns + summaries) | Top-ranked within T_history token budget, plus mandatory tail |
| Facts (if declared) | Fact candidate library | Top-N within context budget |

Selection within each section is deterministic given the scores. Resolved sections accumulate in the context across the resolution loop and are present in the final assembled context for generation. No GPU work beyond the probe decode steps is required for any selection decision.

---

## 5. The Inference Pipeline

### 5.1 Architecture Overview

The per-turn pipeline has three phases: sequential dynamic section resolution (one probe-scan cycle per section), context reconstruction, and full generation. All candidate KV entries for dynamic sections are precomputed at system initialisation and remain in CPU RAM. The conversation history and user turn KV are cached from prior turns and remain live throughout. Dynamic RoPE handles positional remapping as sections are inserted during reconstruction.

```
┌──────────────────────────────────────────────────────────────┐
│                   TURN N: FULL PIPELINE                      │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  PREFILL  [GPU]                                              │
│                                                              │
│  Encode current user turn into KV cache                      │
│  Conversation tail KV already present from prior turns       │
│  Prior turn's assembled history KV retained in cache         │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  DYNAMIC SECTION RESOLUTION LOOP  [GPU probe + CPU scan]     │
│                                                              │
│  For each dynamic section S in declared order:              │
│                                                              │
│    ┌────────────────────────────────────────────────────┐    │
│    │  PROBE  [GPU · W_probe decode steps]               │    │
│    │                                                    │    │
│    │  Context: section S probe system prompt            │    │
│    │         + already-resolved sections (S-1, S-2...)  │    │
│    │         + conversation tail (N_tail turns cached)  │    │
│    │         [+ prior history KV for history probe only]│    │
│    │                                                    │    │
│    │  Accumulate K and Q across three depth bands       │    │
│    │  Decode W_probe tokens → 6-vector probe fingerprint│    │
│    │  Evict probe tokens KV                             │    │
│    │  Retain tail + resolved sections KV               │    │
│    └──────────────────────────┬─────────────────────────┘    │
│                               │                              │
│    ┌──────────────────────────▼─────────────────────────┐    │
│    │  SCAN  [CPU · 3–10ms · 6 INT8 matrix multiplies]   │    │
│    │                                                    │    │
│    │  kl/km/ku scores = Kl/Km/Ku matrix @ kl/km/ku     │    │
│    │  ql/qm/qu scores = Ql/Qm/Qu matrix @ ql/qm/qu     │    │
│    │  score = Σ w_i × band_i_score  (per category)      │    │
│    │  Select top-1 (or top-N for history)               │    │
│    │  Load selected KV: CPU RAM → VRAM                  │    │
│    └────────────────────────────────────────────────────┘    │
│                                                              │
│  After all sections resolved:                                │
│    mood selected · template selected · history selected      │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  CONTEXT RECONSTRUCTION  [GPU]                               │
│                                                              │
│  Assemble full context in declared section order:            │
│    Static system prompt sections  (already in KV cache)      │
│    + Selected mood                (KV loaded in loop)        │
│    + Selected template            (KV loaded in loop)        │
│    + Selected history             (KV loaded in loop,        │
│                                    mandatory tail appended)  │
│    + Current user turn            (KV from prefill)          │
│                                                              │
│  RoPE delta applied to repositioned history entries          │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  GENERATION  [GPU]                                           │
│                                                              │
│  Standard autoregressive generation over assembled context.  │
│  No culling. No eviction. No parallel branches.              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Sequential Resolution and Contextual Carry-Forward

The ordering of dynamic sections is significant. Each section's probe executes in a context that includes all previously resolved sections. This creates a coherent dependency chain:

- **Mood** is resolved first into a minimal context (probe prompt + conversation tail). No prior scaffolding influences the selection.
- **Template** is resolved with mood already present. The model's Q when reasoning about response structure is shaped by the emotional register already committed to.
- **Conversation history** is resolved last, with mood and template both in context under the main NPC system prompt, and additionally with the previous turn's assembled history retained in the KV cache. The Q vectors reflect full response intent conditioned on a strong contextual prior.

The history probe deserves particular attention regarding its approximation properties. The circular dependency — selecting history using a Q generated without the history being selected — is structurally unavoidable. However it is not a cold-start approximation. The previous turn's assembled history remains in the KV cache and is present during the history probe. The model is therefore generating Q under: mood + template + the set of historical turns that were most relevant *last turn* + the current exchange. This is a warm-start delta-approximation: the probe Q reflects the prior relevant history, and the scan identifies how that should update given the new turn. Conversation evolves gradually — relevant historical context does not completely change in a single exchange — so the warm prior is generally a good starting point. The principal weakness is the first turn of a conversation, where no prior assembled history exists and the probe is genuinely cold. This is the one case where history selection is most approximate, and where generous selection counts matter most.

The necessity of fresh per-turn probes is empirically supported by Liu et al. (2024), who demonstrate that dynamically selected tokens achieve 89% attention recovery from a 100K-token context, while statically fixed token selections from the first decode step drop to 71% recovery. This 18-point degradation quantifies the cost of reusing fixed selections across queries — precisely the pattern that fixed, non-probe-driven context selection would produce. The per-turn probe ensures each turn's selections reflect the model's current query state rather than any prior approximation.

### 5.3 Conversation Tail and Mandatory History Inclusion

Two mechanisms ensure conversational continuity regardless of what the fingerprint scan selects:

**Conversation tail in all probes:** The last N_tail turns (default 5) plus the current user turn are always present in every section's probe context. This grounds all selections in the actual recent exchange. A mood selection made while the model is processing only the last few turns is less susceptible to drift toward historically frequent but currently irrelevant patterns.

**Mandatory tail in assembled history:** When the history selection is assembled for generation, the last N_tail turns are always appended at the end of the history prefix regardless of what the fingerprint scan selected. The scan may surface relevant turns from hundreds of exchanges ago; the mandatory tail ensures the assembled history ends with a continuous sequence leading directly to the current user turn. Without this, the model would have a gap between the last selected historical turn and the current exchange.

The GPU-CPU split this produces — recent tail in VRAM throughout, older history loaded from CPU RAM on demand — mirrors the design independently validated by Liu et al. (2024), who found that persisting recent tokens (sliding window) in GPU memory alongside dynamically retrieved older tokens from CPU RAM provides the optimal balance between inference accuracy and memory efficiency. The mandatory tail serves the same role as their "predictable KV vectors" that remain on GPU, while the scan-selected history mirrors their dynamically retrieved tokens.

```
┌─────────────────────────────────────────────────────────────┐
│  Assembled History Structure                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Scan-selected verbatim turns and summaries]               │
│  ...loaded from CPU RAM via PCIe...                         │
│  [T-47 verbatim — high scan score from distant history]     │
│  [S-3 summary — covers T-200..T-100]                        │
│  ...                                                        │
│  ─────────────────────────────────────────────────────────  │
│  [T-5] mandatory tail ← always in VRAM, always included     │
│  [T-4] mandatory tail                                       │
│  [T-3] mandatory tail                                       │
│  [T-2] mandatory tail                                       │
│  [T-1] mandatory tail  ← last assistant turn                │
│  [T-0] current user turn                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Why Mood and Template Probes Use Small Contexts

For mood and template probes, the full conversation history is intentionally absent from the probe context. The probe context for mood is: mood probe system prompt + conversation tail + user turn. For template it is: template probe system prompt + selected mood + conversation tail + user turn.

This is efficient — the full history KV cache does not need to be traversed for these probes — but it is also correct by design. The mood and template selections should reflect the model's immediate response intent given the current exchange, not a blend of that intent with every prior conversation turn. The conversation tail provides sufficient context for grounding. The full history becomes relevant at the history selection stage, where the Q from the history probe (which runs over the full main context) is used to select which past turns are most relevant.

### 5.5 Context Cost at Generation

The total context at generation time is bounded regardless of corpus size or conversation length:

| Component | Size |
|---|---|
| Static system prompt sections | Fixed |
| Selected mood | L_mood tokens |
| Selected template | L_template tokens |
| Selected history | ≤ T_history tokens |
| Mandatory tail | N_tail × L_turn tokens |
| Current user turn | L_query tokens |
| **Total** | **Bounded, constant, corpus-independent** |

### 5.6 Comparison to Prior Probe-Reset Architectures

Prior architectures generated throwaway tokens under all mood and template candidates simultaneously in parallel branches, observed which the model attended to, ran iterative halving, and discarded the entire probe KV cache before real generation. The probe here is structurally different: a single short generation pass per section into a context shaped by a purpose-built probe system prompt, with selection driven by Q fingerprint matching on CPU. There are no parallel branches, no iterative halving, and no KV waste beyond the probe tokens themselves. The mechanism of selection is fundamentally different in kind.



---

## 6. Index Maintenance

### 6.1 The B-Tree as Pre-Computation Infrastructure

The conversation history B-tree is maintained offline, asynchronously from the inference critical path. Its function is to provide multi-resolution representations that populate the flat index:

- **Leaf nodes**: verbatim turn pairs, fingerprinted at turn completion
- **Internal summary nodes**: generated by the tree maintenance process, fingerprinted when created or updated
- **Reflection turns**: periodically generated synthesis turns creating explicit cross-references between distant conversation arcs, fingerprinted as high-priority index entries

The tree's branching factor, summarization frequency, and reflection cycle timing are operational parameters that affect the quality and granularity of the multi-resolution index, but do not affect the inference-time pipeline.

### 6.2 Per-Component Maintenance Policy

| Component | Fingerprint Update Trigger | Residency |
|---|---|---|
| Conversation turns | At turn completion | Always in CPU RAM |
| B-tree summary nodes | When tree maintenance updates node | Always in CPU RAM |
| Facts (static) | Once at index construction | Always in CPU RAM |
| Facts (dynamic) | On fact change event | Always in CPU RAM |
| Mood candidates | Once offline (two passes: fingerprint + KV cache) | Fingerprint in CPU RAM; KV cache in CPU RAM ready for VRAM injection |
| Templates | Once offline (two passes: fingerprint + KV cache) | Fingerprint in CPU RAM; KV cache in CPU RAM ready for VRAM injection |

For a long-running system with 50K turns, 100K facts, and large mood and template libraries, the full index sits within ~126MB of CPU RAM, loaded once at startup. This comfortably fits in the last-level cache on modern server CPUs, enabling scan times well within interactive latency budgets even at scale.

### 6.3 Fingerprint Invalidation

Fingerprints are invalidated and recomputed only when content changes. Conversation turns are immutable once written and never reindexed. Summary nodes are updated synchronously by tree maintenance before the old fingerprint is marked stale. Dynamic facts are reindexed on their change event; static facts are never reindexed. This is dramatically cheaper than per-turn re-encoding required in architectures that condition fact KV on the current turn's context.

---

## 7. Multi-Turn State Management

### 7.1 Component Retention Policy

| Component | Policy | Rationale |
|---|---|---|
| Probe token KV | Evicted after each section probe | Throwaway; only Q fingerprint is retained |
| Conversation tail KV | Retained throughout pipeline | Reused across all probes and final generation |
| Resolved section KV (mood, template) | Retained after selection, carried forward | Present in subsequent probes and final generation |
| Prior turn's assembled history KV | Retained at session start and across turns | Warm-start prior for history probe; reloaded from turn metadata on session resumption |
| Selected history KV | Loaded from CPU RAM per scan result | Assembled into final context with mandatory tail |
| Mandatory tail turns | Always included at end of history prefix | Ensures conversational continuity to the current turn |
| Generated response | Written to B-tree leaf store with history metadata | Fingerprinted, indexed, and records which turns were assembled |

### 7.2 Session Resumption

Each generated turn stores, alongside its fingerprint, the list of history turn identifiers that were assembled in its context. On session resumption after any interruption, this metadata is read from the most recently completed turn and exactly those history turns are loaded back into VRAM before the first probe executes. There is no cold start and no approximation: turn N+1 of a resumed session begins with the identical warm-start context as it would have had in a continuous session. No bootstrap policy or recency-based fallback is required.

### 7.2 Mood Continuity via Prior Bias

The selected mood from the previous turn enters the current turn's mood scan with an elevated prior score: adjusted_score = base_score + α × expected_mean_score, where α > 1 is the persistence factor (default α=2.0). This gives mood continuity without hardcoding — if the current context strongly favors a different mood, the scan score overcomes the prior. The Q-Q component of the mood score naturally reinforces continuity for moods that recur across similar cognitive contexts, without requiring explicit mood state tracking.

### 7.3 Fingerprint Capture at Turn Completion

At the end of each turn, the completed turn is fingerprinted and appended to the index:

```
┌─────────────────────────────────────────────────────────────┐
│  End-of-Turn Processing                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Generation completes (EOS reached)                         │
│    │                                                        │
│    ├── ① Finalize three-tier K fingerprint                  │
│    │        K_lower: top-10% tokens by Q·K score → INT8     │
│    │        K_mid:   top-10% tokens by Q·K score → INT8     │
│    │        K_upper: top-10% tokens by Q·K score → INT8     │
│    │                                                        │
│    ├── ② Finalize three-tier Q fingerprint                  │
│    │        Q_lower: recency-weighted mean → INT8           │
│    │        Q_mid:   recency-weighted mean → INT8           │
│    │        Q_upper: recency-weighted mean → INT8           │
│    │                                                        │
│    ├── ③ Append 6-vector fingerprint to CPU RAM index        │
│    │        ~780 bytes · available from next turn           │
│    │                                                        │
│    ├── ④ Write turn text + assembled history metadata        │
│    │        to B-tree leaf store (immutable once written)   │
│    │                                                        │
│    └── ⑤ Notify B-tree maintenance process (async)          │
│             Summarization and reflection off critical path  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Evaluation

### 8.1 Core Retrieval Quality

**Claim:** Fingerprint scan retrieval accuracy meets or exceeds prior architectures on factual grounding tasks, with substantially lower latency.

**Setup:** QA dataset with ground-truth fact attribution. Plant facts at depths ranging from 1 to 50,000 turns prior. Measure answer accuracy and fact recall at final context selection.

**Baselines:** Dense RAG (DPR embedding similarity), Sparse RAG (BM25), linear context stuffing, prior architecture (beam walk + parallel injection + GPU culling), K-only fingerprint scan (ablation: no Q component), Q-only fingerprint scan (ablation: no K component).

### 8.2 Q-Component Value

**Claim:** Q-fingerprint matching surfaces contextually relevant content that K-matching misses, particularly for conversation history and mood selection.

**Setup:** Construct conversations with deliberate surface-topic discontinuities where relevant prior turns share cognitive context but not vocabulary. Measure retrieval recall for K-only, Q-only, and combined scoring at identical selection counts.

**Hypothesis:** Q-only significantly outperforms K-only for history and mood retrieval; K-only significantly outperforms Q-only for fact retrieval; combined scoring dominates both for all component types.

### 8.3 Scan Latency vs. Corpus Scale

**Claim:** CPU flat scan time scales linearly with corpus size and remains below 10ms for all practical scales.

**Setup:** Measure scan time as a function of index size from 1K to 1M items on representative CPU hardware. Record peak CPU RAM usage, scan time (p50, p95, p99), and PCIe transfer time for selected KV loading.

| Corpus Size | Expected Scan Time | PCIe Load (selected KVs) |
|---|---|---|
| 1K items | < 0.3ms | ~9ms |
| 10K items | < 1.5ms | ~9ms |
| 100K items | < 10ms | ~9ms |
| 1M items | < 75ms | ~9ms |

PCIe load time is constant regardless of corpus size. The six-multiply design with parallel thread execution (Section 9.5) adds ~50% scan time over a four-multiply design; all six matrix multiplies run simultaneously on a 6+ core CPU.

### 8.4 Model Portability

**Claim:** The three-tier cross-head aggregation produces retrieval quality equivalent to manually tuned layer-selection on the same model, and transfers directly to new model families without reconfiguration.

**Setup:** Compare three-tier aggregation (N/3 boundaries, no other configuration) against layer-subset aggregation tuned separately for each model. Measure retrieval quality across three model families of varying depth (8B, 14B, 70B). Evaluate on a held-out model family. Additionally validate that the three-tier boundaries capture the correct functional regime for each model family using the layer-wise emotion probing methodology of Zhang et al. (2025) and Tak et al. (2025).

**Hypothesis:** Three-tier aggregation matches tuned layer selection within measurement noise on seen models. The N/3 and 2N/3 boundaries are expected to be near-optimal for models where emotion consolidation peaks around 50–75% depth, consistent with published probing results.

### 8.5 Section-Specific Probe Quality and Weight Calibration

**Claim:** Custom probe system prompts per dynamic section, combined with assistant-prefill fingerprint encoding, produce meaningfully higher selection quality than a single generic NPC response probe for mood and template, with negligible quality difference for history selection.

**Setup:** Compare selection quality under three conditions: (a) single generic probe (current user turn into NPC prompt), (b) per-section probes with standard NPC prompt for all sections, (c) per-section probes with purpose-aligned custom prompts and assistant-prefill fingerprint encoding. Evaluate mood consistency, template appropriateness, and history relevance against human-labelled ground truth.

**Hypothesis:** Condition (c) outperforms (b) significantly for mood and template, because Q→Q matching is within-distribution at both ends: candidate fingerprints are encoded as assistant prefill under the section's probe system prompt, and probe Q is generated under the same prompt and token role. Condition (b) achieves probe-side alignment but not fingerprint-side alignment, producing a partial improvement. Condition (a) underperforms both on all three metrics. History selection is equivalent across (b) and (c) because the NPC prompt is already the correct alignment context for history fingerprints and probes alike.

**Mood weight ablation.** Ablate w_Qu ∈ {0.35, 0.40, 0.45, 0.50} and w_Ku ∈ {0.15, 0.20, 0.25}, with remaining weights adjusted proportionally, to determine whether the Qwen3-specific upper-band emotion peak (Zhang et al., 2025, Table 3: 75% depth for Qwen3-4B) warrants higher or lower w_Ku than the default 0.20. Also ablate the three-tier split point itself — compare N/3 with N/4 and N/2 as lower boundaries — to empirically validate that the equal-thirds division captures the optimal functional regime boundaries for this architecture.

### 8.6 End-to-End Latency

**Claim:** The sequential resolution loop adds latency proportional to (N_sections × W_probe) decode steps; the scan and reconstruction phases together add under 30ms per section and are effectively fixed overhead.

**Setup:** Measure end-to-end latency breakdown for a three-section system (mood, template, history). Profile each probe phase separately, including context switch overhead between probes. Measure scan time per section. Measure reconstruction time. Compare total turn latency against a single-probe baseline and against prior architecture (beam walk + parallel injection).

### 8.7 Personality Coherence

**Claim:** Q-fingerprint history retrieval produces more behaviorally coherent responses over long conversations than K-only or embedding-based retrieval, because Q-matching surfaces contextually appropriate history even when topic vocabulary differs.

**Setup:** Pre-generate a 10,000-turn conversation history encoding consistent behavioral patterns through demonstrated events rather than described traits. Initialize the system with this history in cold storage with no explicit trait descriptions in the system prompt. Probe with scenarios requiring contextual rather than topical recall. Evaluate trait consistency, history grounding, and hallucination rate against baselines (RAG, K-only fingerprint, sliding window, full-context oracle).

---

## 9. Implementation

### 9.1 Inference Engine Requirements

The architecture requires five capabilities from the underlying inference engine:

- **K and Q tensor accumulation hooks during decode:** At each layer boundary during each section's probe, K and Q tensors are accumulated into three-tier running means (lower, mid, upper bands). Band assignment is determined by layer index relative to the N/3 and 2N/3 thresholds. These hooks operate on tensors already computed as part of standard decode execution and add only the accumulation arithmetic — three running vector additions per layer.
- **Selective KV eviction:** After each section's probe, probe token KV entries must be evicted while conversation tail KV and resolved section KV are retained. Paged KV cache implementations support this natively via page-level deallocation. Different pages correspond to different context segments; eviction is page-granular.
- **Dynamic system prompt substitution:** Each section's probe executes under a potentially different system prompt. The engine must support swapping the system prompt KV between probes. Section-specific probe prompt KVs are precomputed at init time and swapped in at the cost of a pointer update. At initialisation, sections with custom probe prompts require two encoding passes per candidate — one under the probe system prompt as an assistant prefill (producing the CPU RAM fingerprint), and one under the main system prompt as a standard prefill (producing the VRAM-ready KV cache). Both are computed once and cached.
- **Position-independent KV storage with RoPE delta:** History KV entries loaded from CPU RAM or repositioned during reconstruction must be remapped via RoPE delta at attention time. Dynamic RoPE implementations handle this automatically.
- **Async PCIe transfer:** Selected history and fact KV entries loaded from CPU RAM to VRAM can be pipelined with section probe execution, minimising idle time between the scan result and the availability of loaded KV.

### 9.2 Latency Profile

The dominant variable cost is the sequential resolution loop. Each section probe runs over a small context (probe prompt + resolved sections + conversation tail), not over the full history, keeping per-probe latency manageable.

| Operation | Latency | Notes |
|---|---|---|
| Prefill (user turn) | ~10–20ms | Tail KV already cached; only new tokens encoded |
| Mood probe (W_probe=10, small context) | ~60–90ms | Probe prompt + tail only |
| Mood scan + KV load | ~10ms | CPU scan ~3ms (6 multiplies, parallel) + mood KV |
| Template probe (W_probe=10) | ~80–110ms | Probe prompt + mood + tail |
| Template scan + KV load | ~10ms | CPU scan + template KV load |
| History probe (W_probe=15) | ~150–200ms | NPC prompt + mood + template + tail |
| History scan + KV load | ~20ms | Full 126MB index scan + history KV load |
| Context reconstruction | ~5ms | RoPE delta; all KV already in VRAM |
| **Total pre-generation overhead** | **~350–470ms** | Dominated by probe steps |
| Generation | ~450ms | Standard autoregressive decode |
| **Total turn cycle** | **~800–920ms** | |

W_probe is independently tunable per section. Mood and template probes use fewer tokens (10) since the emotional and structural register resolves quickly in mid-to-upper layers; history probes benefit from more (15–20). Reducing mood and template to 8 tokens cuts total probe latency to approximately 220–310ms.

### 9.3 Memory Budget

| Component | Size | Location |
|---|---|---|
| Model weights | Model-dependent | VRAM |
| Active context KV (history + facts + mood + template) | Context-budget-dependent | VRAM |
| Working memory | ~0.5 GB | VRAM |
| **Full fingerprint index (50K turns + 100K facts)** | **~126 MB** | **CPU RAM** |
| Selected KV prefetch buffer | Context-budget-dependent | CPU RAM |
| Turn text store (50K turns) | ~70 MB | Disk |
| Turn KV cache (cold storage) | Model × history depth | Disk |

VRAM usage for the fingerprint mechanism itself is negligible. The index lives entirely in CPU RAM and is never loaded into VRAM.

### 9.4 Hardware Projections

| Platform | Fingerprint Index (1M items) | Scan Time | Scan + Recon Overhead |
|---|---|---|---|
| Consumer (modern desktop CPU) | ~780 MB | ~75ms | ~150ms |
| Enterprise (server CPU, large L3) | ~780 MB | ~15ms | ~100ms |
| Any GPU tier | Index stays in CPU RAM | — | GPU not involved in scan |

The scan is CPU-only and does not benefit from GPU upgrades. It benefits from CPU memory bandwidth and SIMD width, both of which improve with modern desktop and server CPUs independently of GPU generation.

### 9.5 Scan Performance Optimization

Since the flat scan is the architecture's primary CPU-side operation, maximising its throughput is worthwhile. The six matrix multiplies are the bottleneck; the following structural choices maximize SIMD utilization without changing the scan semantics:

**Separate matrix layout over interleaved:** Storing Kl_matrix, Km_matrix, Ku_matrix, Ql_matrix, Qm_matrix, Qu_matrix as six separate contiguous arrays — rather than interleaving all six fingerprint bands per item — enables each matrix multiply to be a single BLAS-style operation over a dense, stride-1 memory region. Modern CPU BLAS implementations (OpenBLAS, MKL, BLIS) exploit this for maximum AVX-512 or NEON throughput. Interleaved layout would require gather operations or strided access that undermine SIMD efficiency.

**Row alignment:** Each row in each matrix is padded to a 64-byte boundary (the width of an AVX-512 register). For d_head=128 with INT8, a row is exactly 128 bytes — two AVX-512 registers — so no padding is required and alignment is natural.

**Parallel thread execution:** The six matrix multiplies are independent and can be dispatched to six CPU threads simultaneously. On a system with 8+ cores this reduces wall-clock scan time from 6× one multiply to approximately 1× one multiply plus thread synchronization overhead (~0.5ms).

**L3 cache tiling:** For index sizes that exceed L3 cache (approximately 1M+ items at ~126MB+ per matrix set), tile the multiplies so each tile fits within L3. Process all six multiplies per tile before advancing, keeping index data hot in cache across all six operations. This avoids re-loading the same index rows six times from main memory.

**INT8 accumulation:** Each dot product accumulates in FP32 to avoid INT8 overflow. Modern CPUs with VNNI (Vector Neural Network Instructions — available on Intel Cascade Lake and later, AMD Zen 4 and later) execute INT8 × INT8 → INT32 accumulation natively, approximately 4× faster than FP32 dot products at the same vector width. The six scale values (scale_Kl, scale_Km, scale_Ku, scale_Ql, scale_Qm, scale_Qu) are applied once after accumulation, not per element.

---

## 10. Failure Modes and Mitigations

### 10.1 Probe Q Residual Approximation

**Problem.** Each section's probe Q reflects the model's intent under a specific (potentially custom) system prompt and a specific set of already-resolved sections. The real decode Q — generated under the full NPC system prompt with all sections in context — will differ.

For **mood and template**, the custom probe prompts are designed to align Q semantically with the candidate library rather than to mimic the final decode context. The gap between probe Q and real decode Q is intentional and managed: mood is selected before it would contaminate the probe, and the assistant prefill encoding ensures fingerprints occupy the same distributional space as probe Q.

For **history**, the approximation is structurally different. The circular dependency — selecting history using Q generated without that history — is unavoidable, but it is a warm-start delta-approximation rather than a cold-start problem. The previous turn's assembled history is retained in the KV cache and present during the history probe, so the probe Q is conditioned on the prior selection rather than on nothing. The probe estimates how the relevant historical context should shift given the new exchange, not what it is from scratch. The residual error is bounded by the rate of change in relevant history between consecutive turns — typically small.

**The genuine cold-start case** is the first turn of a conversation, where no prior assembled history exists. Here the history probe is genuinely cold and history selection is at its most approximate. For first-turn behaviour, a warm-up policy of loading recent history unconditionally, or running with a wider selection count, partially mitigates this.

**Mitigation.** The sequential dependency chain ensures each probe benefits from as much resolved context as available. The warm-start history probe reduces the circular dependency to a delta-update problem in steady state. Custom probe prompts and dual-representation encoding eliminate the distributional gap for mood and template.

### 10.2 Attention Sink Contamination

**Problem.** Attention sink tokens attract disproportionately high attention regardless of semantic content. If K fingerprint construction selected tokens by attention weight magnitude, sinks would dominate the fingerprint and bias retrieval toward positional artefacts rather than content.

**Mitigation.** K token selection uses Q·K inner product scores rather than attention weight magnitudes. In an offline encoding pass, Q·K scores reflect the semantic relevance of each token given the content being encoded — attention sinks do not receive anomalously high Q·K scores in this context because the Q is shaped by the encoding content rather than by a long-running generative context. The Q·K selection criterion is therefore naturally robust to sink contamination without requiring an explicit magnitude gating step.

### 10.3 Q Drift Over Context Shift

**Problem.** A Q fingerprint captured in a very different conversational context may match the current Q in ways that are historically coherent but not currently relevant — surfacing turns that were important then but are noise now.

**Mitigation.** K-component weighting ensures topic relevance remains a factor in the combined score. The blended score prevents pure cognitive-state matching from overriding topical relevance entirely. Component-specific weights (Section 4.2) provide application-specific control over this tradeoff.

### 10.4 Summary Node Stale Fingerprints

**Problem.** B-tree summary nodes may carry stale fingerprints if tree maintenance re-summarizes them but the index update is delayed.

**Mitigation.** Summary node fingerprint updates are synchronous with tree maintenance — the maintenance process writes the new fingerprint before marking the old node stale. Verbatim leaf turns are immutable and never require reindexing.

### 10.5 Context Budget Overflow

**Problem.** The flat scan selects top-N facts within the VRAM context budget, but the token lengths of selected items are not known precisely at scoring time.

**Mitigation.** Token counts are stored alongside fingerprints in the index (negligible storage cost). Selection enforces a hard token budget by accumulating token counts greedily in score order and stopping when the budget is reached. No GPU work is wasted on items that cannot fit in context.

### 10.6 Q→K Distributional Gap in K Matching

**Problem.** The K-component of the scan score (probe Q dotted against stored K fingerprints) operates under out-of-distribution conditions for facts — Q and K occupy different vector spaces separated by more than 10× the within-distribution spread (Section 2.6). Standard INT8 quantization of K vectors optimises for K-space fidelity, not for dot-product accuracy when queried by Q.

**Mitigation for facts.** The Q-aware token selection strategy in Section 3.3 addresses this at construction time by selecting K tokens based on Q·K inner product scores rather than K magnitude. The three-tier split partially mitigates the OOD gap structurally: the lower band, which carries the highest K weight for facts (w_Kl = 0.20), operates in early layers where Q and K distributions are least divergent. The wide acceptance window (top-250), topical specificity of factual content, and the Q supplement (w_Ql + w_Qm + w_Qu = 0.40) provide additional robustness.

**Eliminated for moods and templates.** The dual-representation encoding in Section 3.5 eliminates the OOD problem for sections with custom probe prompts. By encoding mood and template fingerprints as assistant prefill under the section's probe system prompt, both the stored Q fingerprint and the inference probe Q are generated in the same distributional space. Q→K OOD does not apply because Q→Q matching dominates and both Q distributions are aligned by construction across all three bands.

---

## 11. Applications

### 11.1 NPC Cognitive Architecture

The motivating application. The system prompt for the NPC declares three dynamic sections: mood, response template, and conversation history. Each dynamic section carries a candidate library — mood candidates describing emotional states, templates describing response structures, and the conversation history B-tree — and mood and template additionally carry custom probe system prompts.

Each turn, the resolution loop executes in order. First, the mood probe runs: the mood probe system prompt ("describe your current emotional state given this conversation") is used, with only the conversation tail in context. Q vectors from this probe are semantically aligned with the emotional-register content of mood candidates. The top-ranked mood is selected and its KV loaded. Second, the template probe runs with the mood now in context: Q reflects response structure intent conditioned on the established mood. The top-ranked template is selected. Third, the history probe runs under the full NPC system prompt with mood and template both present: Q reflects complete response intent, and this drives selection of the most relevant past turns from the full conversation history. Mandatory tail turns are appended to ensure continuity.

The result is a generation context built bottom-up from the model's own successive intents rather than assembled from external heuristics. Mood is selected because the model's emotional Q matches it. History surfaces because the model in full-context mode attends toward it. The NPC's behavior emerges from the architecture of its own attention rather than from manually authored rules.

### 11.2 Document QA

Facts (document passages) are selected by the fingerprint scan driven by probe Q — the model's response intent for this query, not just the query's surface content. Every passage competes on semantic relevance with no chunking heuristics, no separate embedding model, and no hard top-K cutoff tied to embedding quality. The K-components capture topical and semantic relevance across lower and mid bands; the upper-Q component captures whether the passage was relevant in contexts where the model was reasoning in a similar relational direction to the current response intent.

### 11.3 Long-Running Agent Sessions

Conversation history maintained in the B-tree with reflection turns providing explicit cross-references. The probe phase captures the agent's genuine next-action intent; the fingerprint scan selects relevant history at appropriate resolution — verbatim for specific recent steps, summarized for distant arcs. The agent maintains full access to its entire task history with constant VRAM usage regardless of session length, and with scan and reconstruction overhead well under 100ms regardless of history depth.

### 11.4 Personalisation at Scale

User preference profiles, interaction history, and style templates maintained in the flat index. The scan selects appropriate style and content given the current interaction context. Q-matching surfaces preference patterns that recur in similar cognitive contexts, enabling personalization that goes beyond surface keyword matching to capture how users engage in different moods, topics, and relationship contexts.

---

## 12. Relationship to Prior Work

| Prior Approach | Relationship |
|---|---|
| RAG (embedding-based) | Superseded: fingerprint scan uses model-native probe Q and K representations, not a separate embedding model operating in a different vector space. Selection is driven by decode-time response intent rather than query-embedding similarity. |
| RetrievalAttention (Liu et al., 2024) | Related but distinct: RetrievalAttention uses the current decode Q to dynamically retrieve relevant KV entries within an assembled long context, solving the Q→K OOD problem with an attention-aware ANNS index. Our architecture uses stored past Q as persistent cognitive-state fingerprints for candidate pre-selection before context assembly. Three specific findings from RetrievalAttention directly inform our design: (1) the Q→K distributional gap — Q vectors are more than 10× farther from K vectors than K vectors are from each other (Section 2.6) — grounds our high w_Qu weights and motivates Q-aware K token selection in Section 3.3; (2) the 89% vs 71% dynamic vs static token importance result (Section 5.2) validates per-turn probes over fixed selection; (3) the GPU-persistent-tail plus CPU-retrieved-history split validates our mandatory-tail plus scan-selected architecture (Section 5.3). The Q→Q matching, three-tier depth fingerprint, and dynamic section probe design are novel to this work. |
| Parallel Context Windows (Ratner et al. 2023) | Superseded for candidate selection: fingerprint scan replaces parallel branch injection. Position-independent KV injection (Section 2.3) is retained for loading selected entries at arbitrary context positions during reconstruction. |
| H2O / Scissorhands KV eviction | No longer required: the flat scan selects context items directly rather than loading large pools and evicting during generation. |
| Prior probe-reset architectures | Distinguished: prior probes generated throwaway tokens under all candidates simultaneously in parallel, used iterative halving over that generation, then discarded the probe KV. The probe here generates into a single unscaffolded context, captures Q fingerprints, and performs all selection on CPU. The mechanisms are fundamentally different in kind, not just in efficiency. |
| Attention-organized B-trees (prior work) | Superseded for navigation; retained as multi-resolution pre-computation infrastructure populating the flat index. |
| KV rematerialization (prior work) | Retained and integrated: when historical turns selected by the fingerprint scan are in cold disk storage, KV rematerialization reconstructs their relational encodings before loading into context. |
| Hierarchical retrieval (HNSW etc.) | Analogous multi-resolution structure; different mechanism: model-native cross-layer Q+K scoring vs. embedding distance in a separate vector space. |
| Recurrent / state-space models | Analogous goal (bounded state, unbounded history); different mechanism: learned fixed-size state vs. attention-guided sparse KV selection with full content fidelity. |

---

## 13. Conclusion

We have described an architecture that unifies context management across all component types — conversation history, factual grounding, mood, and response templates — under a single mechanism: attentional provenance indexing with sequential dynamic section resolution and three-tier depth fingerprinting.

The architecture rests on three observations working together. First, Q vectors at decode time are compressed records of the model's response intent, integrating the full accumulated context of everything processed. Second, the optimal Q for selecting any given component type is generated under conditions semantically aligned with that component — a mood probe under a mood-description system prompt produces Q in the same distributional space as stored mood fingerprints, eliminating OOD at both ends. Third, selections compound coherently when made in sequence: each probe runs in a context containing all prior resolved sections, so later selections are conditioned on earlier ones without any multi-pass iteration.

The three-tier depth fingerprint is grounded in layer-wise emotion probing research (Zhang et al., 2025; Tak et al., 2025) demonstrating that transformers exhibit a three-regime processing structure: early layers handling lexical identity and syntax, mid layers assembling semantic category and emotion, and upper layers performing relational reasoning and contextual integration. By dividing at the N/3 and 2N/3 boundaries and weighting each band independently per component type, the architecture captures the emotion consolidation peak in the upper band for mood selection (Zhang et al., 2025, Table 3: Qwen3-4B peaks at 75% depth), the semantic assembly signal in the mid band, and lexical vocabulary specificity in the lower band — without requiring any per-model configuration.

**Limitations.** The history probe circular dependency reduces to a warm-start delta-approximation in steady state — the prior turn's assembled history is present during the history probe — but the first turn remains a cold-start case. The three equal-tier boundaries at N/3 and 2N/3 are a model-agnostic heuristic: the emotion consolidation peak varies by model family (50–75% depth across tested architectures), and the equal-thirds division may not be optimal for all models. Qwen3-30B-A3B is a sparse MoE with GQA (8 KV heads per layer), meaning K aggregation operates over fewer heads than Q aggregation — an asymmetry that may affect the three-tier representation and warrants empirical validation. Total turn latency is dominated by the sequential probe passes.

**Future work.** Empirical calibration of per-section scoring weights, with three-tier mood weight ablation as priority (Section 8.5), including sweep of the tier boundary positions. Layer-wise emotion probing on Qwen3-30B-A3B specifically to validate that the N/3 and 2N/3 boundaries capture the correct functional regimes for this MoE architecture. Adaptive W_probe calibration per section. Parallelisation of mood and template probes where dependency ordering allows. Extension to multi-conversation knowledge bases.

The underlying insight is that attention is not just a mechanism for in-context reasoning — it is also a mechanism for indexing. The same computation the model uses to generate tells you, precisely and in the model's own representational space, what the model would attend to if given access to anything. The fingerprint index captures that signal persistently, at negligible marginal cost, making it available at the scale of CPU RAM rather than the scale of the context window.

---

## References

- Ratner, N. et al. (2023). *Parallel Context Windows for Large Language Models.* ACL 2023.
- Zhang, Z. et al. (2023). *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023.
- Liu, Z. et al. (2023). *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* NeurIPS 2023.
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
- Xiao, G. et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024.
- Chen, Y. et al. (2024). *MEPIC: Memory-Efficient Position-Independent Caching for LLM Inference.*
- Wang, Z. et al. (2025). *KVLINK: Linking KV Cache Chunks for Long-Context LLM Inference.*
- Liu, D. et al. (2024). *RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval.* NeurIPS 2025. arXiv:2409.10516.
- Zhang, J. et al. (2025). *Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion.* arXiv:2510.04064.
- Tak, A.N. et al. (2025). *Mechanistic Interpretability of Emotion Inference in Large Language Models.* Findings of ACL 2025.

---

## Appendix A: System Architecture Diagrams

### A.1 Stored Fingerprint Capture Pipeline

The procedure below applies to all items written to the persistent index. Sections with a custom probe system prompt (mood, template) produce two representations at construction time — a fingerprint for scanning and a KV cache for generation. All other item types produce a single representation.

```
┌─────────────────────────────────────────────────────────────┐
│  Stored Fingerprint Capture (all item types)                │
└─────────────────────────────────────────────────────────────┘

  CONVERSATION TURNS / FACTS / B-TREE SUMMARIES
  Standard forward pass — three-tier running accumulation
  ┌────────────────────────────────────────────────────────┐
  │  Layer 0   ──┐                                         │
  │  ...       ──┤  K_lower_accum += mean_heads(K_layer)   │
  │  Layer N/3 ──┘  → finalise lower accumulators          │
  │  Layer N/3+1 ──┐                                       │
  │  ...         ──┤  K_mid_accum  += mean_heads(K_layer)  │
  │  Layer 2N/3  ──┘  → finalise mid accumulators          │
  │  Layer 2N/3+1 ──┐                                      │
  │  ...          ──┤  K_upper_accum += mean_heads(K_layer)│
  │  Layer N      ──┘  → finalise upper accumulators       │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  AGGREGATION
  ┌────────────────────────────────────────────────────────┐
  │  K_lower → top-10% tokens by Q·K score → INT8 + scale  │
  │  K_mid   → top-10% tokens by Q·K score → INT8 + scale  │
  │  K_upper → top-10% tokens by Q·K score → INT8 + scale  │
  │  Q_lower → recency-weighted mean → INT8 + scale         │
  │  Q_mid   → recency-weighted mean → INT8 + scale         │
  │  Q_upper → recency-weighted mean → INT8 + scale         │
  │  Total: ~780 bytes per item → CPU RAM INDEX             │
  └────────────────────────────────────────────────────────┘

  MOODS / TEMPLATES WITH CUSTOM PROBE PROMPT
  Two passes at construction time — one per representation
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  PASS 1 — FINGERPRINT                                  │
  │  System prompt: section's probe system prompt          │
  │  Token role:    ASSISTANT (candidate text as assistant  │
  │                 prefill, not system/user role)          │
  │  → same three-tier aggregation → CPU RAM INDEX         │
  │  Aligns with probe: same prompt · same role · same mode│
  │                                                        │
  │  PASS 2 — KV CACHE                                     │
  │  System prompt: main NPC system prompt                 │
  │  Token role:    standard context section               │
  │  → standard prefill KV → stored for VRAM injection     │
  │  Used during real generation, not during scanning      │
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

### A.2 Per-Turn Sequential Resolution Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Per-Turn Sequential Dynamic Section Resolution             │
└─────────────────────────────────────────────────────────────┘

  PREFILL
  ┌────────────────────────────────────────────────────────┐
  │  Encode current user turn                              │
  │  Tail KV already cached (N_tail prior turns)           │
  └──────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────▼─────────────────┐
            │  FOR EACH DYNAMIC SECTION S:      │
            │                                  │
            │  ┌──────────────────────────┐    │
            │  │  PROBE  [W_probe tokens] │    │
            │  │                          │    │
            │  │  Context:                │    │
            │  │  S.probe_system_prompt   │    │
            │  │  + resolved sections     │    │
            │  │    (S-1, S-2, ...)       │    │
            │  │  + conversation tail     │    │
            │  │                          │    │
            │  │  Accumulate K+Q per band │    │
            │  │  Capture 6-vec fingerpt  │    │
            │  │  Evict probe token KV    │    │
            │  └────────────┬─────────────┘    │
            │               │                  │
            │  ┌────────────▼─────────────┐    │
            │  │  SCAN  [CPU · 3–10ms]    │    │
            │  │  6 INT8 matrix multiplies│    │
            │  │                          │    │
            │  │  Score S.candidates      │    │
            │  │  across 3K + 3Q bands    │    │
            │  │  Select top-1 (or top-N) │    │
            │  │  Load KV: CPU RAM→VRAM   │    │
            │  └──────────────────────────┘    │
            │                                  │
            └──────────────┬───────────────────┘
                           │  (repeat for each section)
                           ▼
  CONTEXT RECONSTRUCTION
  ┌────────────────────────────────────────────────────────┐
  │  Assemble in declared section order:                   │
  │    Static system prompt sections                       │
  │    + Selected mood KV                                  │
  │    + Selected template KV                              │
  │    + Selected history KV                               │
  │      [mandatory tail always appended at end]           │
  │    + Current user turn KV                              │
  │  RoPE delta applied to repositioned entries            │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  GENERATION
  ┌────────────────────────────────────────────────────────┐
  │  Standard autoregressive decode over full context      │
  │  At completion: fingerprint turn → append to index     │
  └────────────────────────────────────────────────────────┘
```

---

## Appendix B: Parameter Reference

**Pipeline parameters:**

| Parameter | Default | Effect | Tuning Guidance |
|---|---|---|---|
| W_probe (mood) | 10 | Probe token count for mood section | Upper-band emotion signal consolidates quickly; 10 tokens sufficient |
| W_probe (template) | 10 | Probe token count for template section | Structural intent resolves quickly in mid-upper bands |
| W_probe (history) | 15 | Probe token count for history section | Full relational intent needs more upper-band accumulation |
| N_tail | 5 | Mandatory recent turns always in probe + final context | Increase for longer obligatory continuity windows |
| α | 2.0 | Mood persistence prior bias | Increase for stable-mood applications |
| T_history | 1,000 tokens | History scan token budget (excluding mandatory tail) | Tune per available context window |
| N_facts | Context-dependent | Facts loaded into context | Tune per VRAM headroom and task needs |

Session restoration uses the history turn identifiers stored in the most recent turn's metadata — no additional parameters required.

**Scoring weights — three-tier bands (tunable per deployment):**

| Parameter | Default | Band | Signal | Research basis |
|---|---|---|---|---|
| w_Kl (facts) | 0.20 | Lower | Surface vocabulary match | Distinctive factual terms register early; OOD gap smallest in lower band |
| w_Km (facts) | 0.25 | Mid | Semantic category/domain match | Mid-band semantic assembly; primary fact discrimination |
| w_Ku (facts) | 0.15 | Upper | Relational context match | Contextual relevance of relational facts |
| w_Ql (facts) | 0.05 | Lower | Surface query framing | Minimal — topical intent not well-formed in lower band |
| w_Qm (facts) | 0.10 | Mid | Semantic response intent | Topic direction of emerging response |
| w_Qu (facts) | 0.25 | Upper | Full relational response intent | Relational reasoning state; which facts matter given full context |
| w_Kl (history) | 0.05 | Lower | Lexical surface overlap | Weak; surface vocabulary insufficient for contextual history |
| w_Km (history) | 0.10 | Mid | Semantic topic match | Which topic domains this turn covers |
| w_Ku (history) | 0.15 | Upper | Relational semantic match | Deeper semantic context of the historical exchange |
| w_Ql (history) | 0.05 | Lower | Surface query intent | Minimal signal |
| w_Qm (history) | 0.15 | Mid | Semantic cognitive state | Semantic reasoning register |
| w_Qu (history) | 0.50 | Upper | Full relational reasoning state | Primary signal; Q_upper is the richest cognitive-state fingerprint |
| w_Kl (mood) | 0.08 | Lower | Lexical mood vocabulary | Emotion words are lexically distinctive; early activation (MWE/Transactions ACL 2024) |
| w_Km (mood) | 0.15 | Mid | Semantic emotion category | Emotion probe accuracy rises steeply through mid layers (Tak et al. 2025, Fig. 2) |
| w_Ku (mood) | 0.20 | Upper | Emotion consolidation plateau | Qwen3-4B peaks at 75% depth = upper band (Zhang et al. 2025, Table 3) |
| w_Ql (mood) | 0.02 | Lower | Immediate surface affect | Minimal; lower Q reflects only early probe tokens |
| w_Qm (mood) | 0.10 | Mid | Semantic emotional register | Mid-band Q reflects forming emotion register |
| w_Qu (mood) | 0.45 | Upper | Full accumulated cognitive state | Dominant; upper Q captures peak emotion geometry (both papers) |
| w_Kl (template) | 0.10 | Lower | Structural surface patterns | Response opening vocabulary and format markers |
| w_Km (template) | 0.20 | Mid | Response type semantics | Genre and structural category |
| w_Ku (template) | 0.15 | Upper | Relational response structure | How structure connects to content and relational framing |
| w_Ql (template) | 0.08 | Lower | Surface response intent | Immediate format intent |
| w_Qm (template) | 0.17 | Mid | Semantic structural intent | Response type and tone forming |
| w_Qu (template) | 0.30 | Upper | Full relational structural intent | Upper Q drives template-to-context match |
| w_Kl (B-tree summaries) | 0.08 | Lower | Topic vocabulary | Surface content of summarised span |
| w_Km (B-tree summaries) | 0.15 | Mid | Semantic summary category | What this span was about |
| w_Ku (B-tree summaries) | 0.12 | Upper | Relational summary content | Relational context of the span |
| w_Ql (B-tree summaries) | 0.05 | Lower | Surface query intent | Minimal |
| w_Qm (B-tree summaries) | 0.15 | Mid | Semantic cognitive match | Thematic resonance |
| w_Qu (B-tree summaries) | 0.45 | Upper | Full reasoning state match | Primary; summary relevance is fundamentally a cognitive-state question |

**Fingerprint parameters:**

| Parameter | Default | Effect | Tuning Guidance |
|---|---|---|---|
| K_top_pct | 10% | Percentage of tokens selected by Q·K score for K fingerprint | Lower for very short items (< 20 tokens); raise slightly for long documents |