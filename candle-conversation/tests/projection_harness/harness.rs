//! [`Harness`]: builds the projection schema and drives BDP scans.
//!
//! Responsibilities:
//!   - Parse `projection.yaml`, add one tool section per known tool into the
//!     dialogue layer's `tools` collection.
//!   - Run `BdpScanner::scan_sections` against real provenance data for a
//!     given probe scenario, returning a populated [`HarnessResolver`].
//!   - Provide scoring helpers and a projection-emission inspector.

use std::collections::HashMap;
use std::sync::OnceLock;

use candle_conversation::projection::{
    Builder, GroupId, LayerId, ProjectionTarget, Projection, SectionId,
};
use candle_conversation::provenance::{
    BdpScanner, ProvenanceFile, RawFileHeader, RawProvenanceFile, RawSigEntry, SigEntry, TokenHit,
    TokenSignature,
};

use crate::corpus::{CaseType, Manifest, RawManifest, TOOLS, projection_yaml_text, tool_stub};
use crate::resolver::HarnessResolver;

// ── SignatureStrategy ─────────────────────────────────────────────────────────

/// Which float vectors to binarise when computing `TokenSignature`s from raw
/// KVQ data.  The probe side and corpus side may differ (e.g. `QK` uses Q for
/// the probe but K for the corpus, approximating the Q·K attention score).
#[derive(Debug, Clone)]
pub enum SignatureStrategy {
    /// sign(Q[layer, head]) for both probe and corpus — mirrors production.
    QQ { layer: usize, head: usize },
    /// sign(Q) probe vs sign(K) corpus — proper Q·K alignment approximation.
    QK { layer: usize, head: usize },
    /// sign(K[layer, head]) for both sides — key-space similarity.
    KK { layer: usize, head: usize },
    /// XOR-fold sign(Q) over all KV heads for both sides.
    MultiHeadXorQQ { layer: usize },
    /// XOR-fold sign(Q) over all KV heads at two layers, combining both into one
    /// 128-bit fingerprint (8-head fold).  Captures both the smooth sequential
    /// patterns of the early layer and the sharp token-selective patterns of the
    /// later layer simultaneously.
    MultiHeadXorQQDual { layer_a: usize, layer_b: usize },
    /// XOR-fold sign(Q) probe vs XOR-fold sign(K) corpus.
    MultiHeadXorQK { layer: usize },
    /// sign(P @ Q) probe vs sign(P @ K) corpus with a fixed random ±1 projection P.
    ///
    /// Unlike direct sign(Q/K), the random projection mixes dimensions before
    /// binarising, which can better separate correlated feature directions.
    FloatSimHash { layer: usize, head: usize },
    /// sign(mean(Q over all layers in band)) for both sides.
    ///
    /// Averages Q vectors across all n_layers_per_band layers before signing,
    /// removing sensitivity to which single layer is sampled.
    BandMeanQQ { head: usize },
    /// sign(mean(Q)) probe vs sign(mean(K)) corpus, averaged over all band layers.
    BandMeanQK { head: usize },
    /// sign(mean(Q over all KV heads)) for both sides — mean-fold vs XOR-fold.
    MultiHeadMeanQQ { layer: usize },
    /// sign(mean(Q)) probe vs sign(mean(K)) corpus, mean-folded over heads.
    MultiHeadMeanQK { layer: usize },
    /// Window-mean probe: sign(mean(Q_{t-w}..Q_{t+w})) probe vs sign(K) corpus.
    ///
    /// Smooths the probe signal across a sliding token window before binarising,
    /// which reduces single-token noise and better approximates sustained
    /// attention — a run of semantically similar queries averaging together
    /// strengthens the directional signal toward the relevant corpus token.
    WindowMeanQ { window: usize, layer: usize, head: usize },
}

impl SignatureStrategy {
    pub fn name(&self) -> String {
        match self {
            Self::QQ { layer, head }         => format!("QQ_l{layer}_h{head}"),
            Self::QK { layer, head }         => format!("QK_l{layer}_h{head}"),
            Self::KK { layer, head }         => format!("KK_l{layer}_h{head}"),
            Self::MultiHeadXorQQ { layer }               => format!("MH_XOR_QQ_l{layer}"),
            Self::MultiHeadXorQQDual { layer_a, layer_b } => format!("MH_XOR_QQ_l{layer_a}xl{layer_b}"),
            Self::MultiHeadXorQK { layer }               => format!("MH_XOR_QK_l{layer}"),
            Self::FloatSimHash { layer, head }  => format!("SimHash_l{layer}_h{head}"),
            Self::BandMeanQQ { head }            => format!("BandMeanQQ_h{head}"),
            Self::BandMeanQK { head }            => format!("BandMeanQK_h{head}"),
            Self::MultiHeadMeanQQ { layer }      => format!("MH_Mean_QQ_l{layer}"),
            Self::MultiHeadMeanQK { layer }      => format!("MH_Mean_QK_l{layer}"),
            Self::WindowMeanQ { window, layer, head } => format!("WinQ_w{window}_l{layer}_h{head}"),
        }
    }
}

// ── SimHash projection matrix ─────────────────────────────────────────────────

/// Global 128×128 ±1 random projection matrix, lazily initialised.
///
/// Generated with a fixed seed (Xorshift64) for reproducibility.
/// Rows are output dimensions; columns are input dimensions.
static SIMHASH_PROJ: OnceLock<Vec<f32>> = OnceLock::new();

fn simhash_proj(n: usize) -> &'static [f32] {
    SIMHASH_PROJ.get_or_init(|| {
        let mut state: u64 = 0xdeadbeef_cafebabe;
        (0..n * n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state & 1 == 0 { 1.0f32 } else { -1.0f32 }
            })
            .collect()
    })
}

/// Apply the fixed projection: `sign(P @ v)` → `TokenSignature`.
fn apply_simhash(v: &[f32]) -> TokenSignature {
    let n = v.len().min(128);
    let proj = simhash_proj(128);
    let projected: Vec<f32> = (0..128)
        .map(|i| {
            proj[i * 128..i * 128 + n]
                .iter()
                .zip(&v[..n])
                .map(|(&p, &x)| p * x)
                .sum::<f32>()
        })
        .collect();
    TokenSignature::from_q_flat(&projected)
}

/// Fill `buf` with one K or Q vector read directly from `blob`.
///
/// Avoids the `Vec<f32>` allocation that `read_kq_vector` would require.
#[inline]
fn read_kq_from_blob(
    header: &RawFileHeader,
    blob: &[u8],
    t: usize,
    band: usize,
    layer: usize,
    head: usize,
    is_q: bool,
    buf: &mut Vec<f32>,
) {
    let hd = header.head_dim as usize;
    let off = header.entry_offset(t, band, layer, head, is_q);
    buf.clear();
    for d in 0..hd {
        let b = off + d * 4;
        buf.push(f32::from_le_bytes(blob[b..b + 4].try_into().unwrap()));
    }
}

/// Compute `TokenSignature` for one token from a pre-read entry blob.
///
/// `blob` must be the byte slice for the entry (obtained via `entry_slice` or
/// `read_entry_bytes`).  `buf` is a reusable scratch Vec — callers should pass
/// the same Vec across calls to amortise allocations.
fn sig_for_token(
    header: &RawFileHeader,
    blob: &[u8],
    strategy: &SignatureStrategy,
    is_probe: bool,
    t: usize,
    n_tok: usize,
    band: usize,
    n_heads: usize,
    buf: &mut Vec<f32>,
) -> TokenSignature {
    match strategy {
        SignatureStrategy::QQ { layer, head } => {
            read_kq_from_blob(header, blob, t, band, *layer, *head, true, buf);
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::KK { layer, head } => {
            read_kq_from_blob(header, blob, t, band, *layer, *head, false, buf);
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::QK { layer, head } => {
            read_kq_from_blob(header, blob, t, band, *layer, *head, is_probe, buf);
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::MultiHeadXorQQ { layer } => {
            let vecs: Vec<Vec<f32>> = (0..n_heads).map(|h| {
                read_kq_from_blob(header, blob, t, band, *layer, h, true, buf);
                buf.clone()
            }).collect();
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            TokenSignature::from_q_multi(&refs)
        }
        SignatureStrategy::MultiHeadXorQQDual { layer_a, layer_b } => {
            // XOR-fold all 4 heads at layer_a then all 4 heads at layer_b = 8-head fold.
            let mut vecs: Vec<Vec<f32>> = Vec::new();
            for &l in &[*layer_a, *layer_b] {
                for h in 0..n_heads {
                    read_kq_from_blob(header, blob, t, band, l, h, true, buf);
                    vecs.push(buf.clone());
                }
            }
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            TokenSignature::from_q_multi(&refs)
        }
        SignatureStrategy::MultiHeadXorQK { layer } => {
            let vecs: Vec<Vec<f32>> = (0..n_heads).map(|h| {
                read_kq_from_blob(header, blob, t, band, *layer, h, is_probe, buf);
                buf.clone()
            }).collect();
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            TokenSignature::from_q_multi(&refs)
        }
        SignatureStrategy::FloatSimHash { layer, head } => {
            read_kq_from_blob(header, blob, t, band, *layer, *head, is_probe, buf);
            apply_simhash(buf)
        }
        SignatureStrategy::BandMeanQQ { head } => {
            let nl = header.n_layers_per_band as usize;
            let hd = header.head_dim as usize;
            buf.clear();
            buf.resize(hd, 0.0);
            for l in 0..nl {
                let off = header.entry_offset(t, band, l, *head, true);
                for d in 0..hd { buf[d] += f32::from_le_bytes(blob[off + d * 4..off + d * 4 + 4].try_into().unwrap()); }
            }
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::BandMeanQK { head } => {
            let nl = header.n_layers_per_band as usize;
            let hd = header.head_dim as usize;
            buf.clear();
            buf.resize(hd, 0.0);
            for l in 0..nl {
                let off = header.entry_offset(t, band, l, *head, is_probe);
                for d in 0..hd { buf[d] += f32::from_le_bytes(blob[off + d * 4..off + d * 4 + 4].try_into().unwrap()); }
            }
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::MultiHeadMeanQQ { layer } => {
            let hd = header.head_dim as usize;
            buf.clear();
            buf.resize(hd, 0.0);
            for h in 0..n_heads {
                let off = header.entry_offset(t, band, *layer, h, true);
                for d in 0..hd { buf[d] += f32::from_le_bytes(blob[off + d * 4..off + d * 4 + 4].try_into().unwrap()); }
            }
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::MultiHeadMeanQK { layer } => {
            let hd = header.head_dim as usize;
            buf.clear();
            buf.resize(hd, 0.0);
            for h in 0..n_heads {
                let off = header.entry_offset(t, band, *layer, h, is_probe);
                for d in 0..hd { buf[d] += f32::from_le_bytes(blob[off + d * 4..off + d * 4 + 4].try_into().unwrap()); }
            }
            TokenSignature::from_q_flat(buf)
        }
        SignatureStrategy::WindowMeanQ { window, layer, head } => {
            let hd = header.head_dim as usize;
            if is_probe {
                let t_start = t.saturating_sub(*window);
                let t_end = (t + window + 1).min(n_tok);
                buf.clear();
                buf.resize(hd, 0.0);
                for ti in t_start..t_end {
                    let off = header.entry_offset(ti, band, *layer, *head, true);
                    for d in 0..hd { buf[d] += f32::from_le_bytes(blob[off + d * 4..off + d * 4 + 4].try_into().unwrap()); }
                }
                TokenSignature::from_q_flat(buf)
            } else {
                read_kq_from_blob(header, blob, t, band, *layer, *head, false, buf);
                TokenSignature::from_q_flat(buf)
            }
        }
    }
}

/// Compute per-band `TokenSignature` arrays for all tokens in one raw entry.
///
/// Uses `entry_slice` for zero-copy access when the mmap is cached (read mode),
/// falling back to a heap copy otherwise.  A single `Vec<f32>` scratch buffer
/// is reused across all token reads to avoid per-call allocation.
fn compute_sigs(
    raw_pf: &RawProvenanceFile,
    entry: RawSigEntry,
    strategy: &SignatureStrategy,
    is_probe: bool,
) -> [Vec<TokenSignature>; 3] {
    let n_tok = entry.token_count as usize;
    let header = raw_pf.header();
    let n_heads = header.n_kv_heads as usize;

    // Zero-copy slice when mmap is cached (open mode); heap copy otherwise.
    let blob_owned: Vec<u8>;
    let blob: &[u8] = match raw_pf.entry_slice(entry) {
        Some(s) => s,
        None => {
            blob_owned = raw_pf.read_entry_bytes(entry).unwrap_or_default();
            &blob_owned
        }
    };

    let mut buf = Vec::with_capacity(header.head_dim as usize);
    [0usize, 1, 2].map(|band| {
        (0..n_tok)
            .map(|t| sig_for_token(header, blob, strategy, is_probe, t, n_tok, band, n_heads, &mut buf))
            .collect()
    })
}

// ── Span scoring ──────────────────────────────────────────────────────────────

/// Score a section's hit log using run-length boosting.
///
/// For each depth, collects the unique probe token indices that achieved any
/// hit against this section, finds consecutive runs in that sorted set, and
/// scores each run of length L as `L^alpha`.  Isolated hits (L=1) score 1.0,
/// identical to Count.  Returns `[syn_score, sem_score, prag_score]`.
///
/// Power-law `L^alpha` (alpha in 1.0..2.0) models sustained attention:
/// three consecutive probe tokens all pointing to the same section is far
/// more informative than three scattered single hits, because consecutive
/// Q vectors in a live decode share semantic direction only when the model
/// is genuinely focused on that content.
pub fn span_score_section(hits: &[TokenHit], alpha: f32) -> [f32; 3] {
    let mut result = [0.0f32; 3];
    for depth in 0u8..3 {
        let mut probe_toks: Vec<u16> = hits
            .iter()
            .filter(|h| h.depth == depth)
            .map(|h| h.probe_tok)
            .collect();
        probe_toks.sort_unstable();
        probe_toks.dedup();

        let mut score = 0.0f32;
        let mut run_len = 0usize;
        let mut prev = u16::MAX;
        for &tok in &probe_toks {
            if prev == u16::MAX || tok != prev + 1 {
                score += (run_len as f32).powf(alpha);
                run_len = 1;
            } else {
                run_len += 1;
            }
            prev = tok;
        }
        score += (run_len as f32).powf(alpha);
        // Subtract the L=0 phantom run scored on first iteration when run_len starts at 0.
        score -= 0.0f32.powf(alpha); // 0^alpha = 0 for alpha > 0, so this is a no-op but clarifies intent.
        result[depth as usize] = score;
    }
    result
}

/// Mean span score across the three depths.
pub fn span_score_mean(hits: &[TokenHit], alpha: f32) -> f32 {
    let [syn, sem, prag] = span_score_section(hits, alpha);
    (syn + sem + prag) / 3.0
}

// ── RawCorpusCache ────────────────────────────────────────────────────────────

/// Precomputed corpus signatures for one `SignatureStrategy`.
///
/// Build once with `Harness::build_raw_corpus_cache`; reuse across all N probe
/// scans for that strategy to avoid redundant raw KVQ reads.
pub struct RawCorpusCache {
    /// Temporary provenance file holding all positive corpus signatures.
    pub pf: ProvenanceFile,
    /// tool_name → Vec<(scenario_id, SigEntry into `pf`)>
    pub tool_entries: HashMap<String, Vec<(String, SigEntry)>>,
}

// ── Harness ────────────────────────────────────────────────────────────────────

pub struct Harness {
    pub builder: Builder,
    /// tool name → SectionId in the dialogue `tools` collection.
    pub tool_section_ids: HashMap<String, SectionId>,
    pub dialogue_layer: LayerId,
    pub conv_group: GroupId,
}

impl Harness {
    pub fn build() -> Self {
        let yaml = projection_yaml_text();
        let mut builder = Builder::from_yaml_with_vars(&yaml, &[("workspace", "candle")])
            .expect("Builder::from_yaml_with_vars failed");

        let dialogue_layer = builder
            .id_for_layer("dialogue")
            .expect("dialogue layer not found in projection.yaml");
        let conv_group = builder
            .id_for_group("primary_conversation")
            .expect("primary_conversation group not found");
        let tools_coll = builder
            .id_for_collection_in(dialogue_layer, "tools")
            .expect("tools collection not found in dialogue layer");

        let mut tool_section_ids = HashMap::new();
        for &tool in TOOLS {
            let sid = builder
                .add_section_to_collection(
                    dialogue_layer,
                    tools_coll,
                    tool,
                    tool_stub(tool),
                    1.0,
                )
                .unwrap_or_else(|e| panic!("add_section_to_collection({tool}): {e}"));
            tool_section_ids.insert(tool.to_string(), sid);
        }

        Self { builder, tool_section_ids, dialogue_layer, conv_group }
    }

    /// Build a `ProjectionTarget` for the dialogue / primary_conversation group.
    ///
    /// No turns exist in the harness corpus so the timeline value is only used
    /// by group-scoped turn masking — allocate a fresh id to satisfy the
    /// `NonZeroU64` invariant.
    pub fn target(&self) -> ProjectionTarget {
        let timeline = candle_conversation::projection::TimelineAllocator::default().next();
        ProjectionTarget {
            layer: self.dialogue_layer,
            group: self.conv_group,
            timeline,
        }
    }

    // ── Scanning ─────────────────────────────────────────────────────────────

    /// Scan all tool sections against a named probe scenario's Q signatures.
    ///
    /// The corpus for each section is every *positive* scenario for that tool,
    /// with `probe_id` excluded from its own tool's corpus so scores reflect
    /// out-of-corpus generalisation.
    pub fn scan(
        &self,
        pf: &ProvenanceFile,
        manifest: &Manifest,
        probe_id: &str,
    ) -> HarnessResolver {
        let probe = manifest
            .scenarios
            .iter()
            .find(|s| s.id == probe_id)
            .unwrap_or_else(|| panic!("probe scenario not found: {probe_id}"));

        let (probe_syn, probe_sem, probe_prag) = pf
            .read_entry(SigEntry { byte_offset: probe.byte_offset, token_count: probe.token_count })
            .expect("read probe sigs failed");

        let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS
            .iter()
            .map(|&tool| {
                let sid = self.tool_section_ids[tool];
                let entries: Vec<SigEntry> = manifest
                    .scenarios
                    .iter()
                    .filter(|s| {
                        s.tool.as_deref() == Some(tool)
                            && s.case_type == CaseType::Positive
                            && s.id != probe_id
                    })
                    .map(|s| SigEntry { byte_offset: s.byte_offset, token_count: s.token_count })
                    .collect();
                (sid, entries)
            })
            .collect();

        let mut scanner = BdpScanner::new();
        scanner
            .scan_sections(pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
            .expect("scan_sections failed");

        let mut resolver = HarnessResolver::new();
        for (&sid, &scores) in scanner.section_scores() {
            resolver.section_scores.insert(sid, scores);
        }
        resolver
    }

    // ── Scoring helpers ───────────────────────────────────────────────────────

    /// Combined score for a tool section using equal depth weights.
    ///
    /// `use_max=true`  → `(syn.max  + sem.max  + prag.max)  / 3`
    ///     — mirrors what the projection engine computes (`score_formula: max`).
    /// `use_max=false` → `(syn.mean + sem.mean + prag.mean) / 3`
    ///     — mean-agreement; less susceptible to saturation over large corpora.
    pub fn section_score_formula(
        &self,
        resolver: &HarnessResolver,
        tool: &str,
        use_max: bool,
    ) -> f32 {
        let sid = self.tool_section_ids[tool];
        resolver
            .section_scores
            .get(&sid)
            .map(|s| if use_max {
                (s.syn.max + s.sem.max + s.prag.max) / 3.0
            } else {
                (s.syn.mean + s.sem.mean + s.prag.mean) / 3.0
            })
            .unwrap_or(0.0)
    }

    /// Production score: `section_score_formula(…, use_max=true)`.
    pub fn section_score(&self, resolver: &HarnessResolver, tool: &str) -> f32 {
        self.section_score_formula(resolver, tool, true)
    }

    // ── Projection inspector ─────────────────────────────────────────────────

    /// Names of the tool sections that survived projection, sorted.
    pub fn emitted_tools<'a>(&'a self, projection: &Projection) -> Vec<&'a str> {
        let mut out: Vec<&str> = projection
            .system_prompt
            .iter()
            .filter_map(|rs| {
                self.tool_section_ids
                    .iter()
                    .find(|(_, &s)| s == rs.id)
                    .map(|(name, _)| name.as_str())
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Same as [`Self::scan`] but returns the per-section hit log alongside
    /// the resolver.  Only pairs with agreement ≥ `hit_threshold` are recorded.
    pub fn scan_with_hits(
        &self,
        pf: &ProvenanceFile,
        manifest: &Manifest,
        probe_id: &str,
    ) -> (HarnessResolver, HashMap<SectionId, Vec<TokenHit>>) {
        let probe = manifest
            .scenarios
            .iter()
            .find(|s| s.id == probe_id)
            .unwrap_or_else(|| panic!("probe scenario not found: {probe_id}"));

        let (probe_syn, probe_sem, probe_prag) = pf
            .read_entry(SigEntry { byte_offset: probe.byte_offset, token_count: probe.token_count })
            .expect("read probe sigs failed");

        let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS
            .iter()
            .map(|&tool| {
                let sid = self.tool_section_ids[tool];
                let entries: Vec<SigEntry> = manifest
                    .scenarios
                    .iter()
                    .filter(|s| {
                        s.tool.as_deref() == Some(tool)
                            && s.case_type == CaseType::Positive
                            && s.id != probe_id
                    })
                    .map(|s| SigEntry { byte_offset: s.byte_offset, token_count: s.token_count })
                    .collect();
                (sid, entries)
            })
            .collect();

        let mut scanner = BdpScanner::new().with_record_hits(true);
        scanner
            .scan_sections(pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
            .expect("scan_sections failed");

        let mut resolver = HarnessResolver::new();
        for (&sid, &scores) in scanner.section_scores() {
            resolver.section_scores.insert(sid, scores);
        }

        let hit_log = scanner
            .section_hit_log()
            .iter()
            .map(|(&sid, hits)| (sid, hits.clone()))
            .collect();

        (resolver, hit_log)
    }

    // ── Batch helpers ─────────────────────────────────────────────────────────

    // ── Corpus cache ─────────────────────────────────────────────────────────

    /// Precompute signatures for all positive corpus scenarios under `strategy`.
    ///
    /// The returned `RawCorpusCache` holds a single `ProvenanceFile` with every
    /// positive scenario appended, plus a per-tool lookup from scenario id to
    /// `SigEntry`.  Pass it to `scan_raw_cached` / `scan_raw_with_hits_cached`
    /// so the corpus is not recomputed for each of the N probe scans.
    pub fn build_raw_corpus_cache(
        &self,
        raw_pf: &RawProvenanceFile,
        raw_manifest: &RawManifest,
        strategy: &SignatureStrategy,
    ) -> RawCorpusCache {
        let pf = ProvenanceFile::new().expect("tmp ProvenanceFile failed");
        let mut tool_entries: HashMap<String, Vec<(String, SigEntry)>> =
            TOOLS.iter().map(|&t| (t.to_string(), Vec::new())).collect();

        for scen in &raw_manifest.scenarios {
            if scen.case_type != CaseType::Positive { continue; }
            let tool = match scen.tool.as_deref() { Some(t) => t, None => continue };
            if !tool_entries.contains_key(tool) { continue; }
            let raw_entry = RawSigEntry {
                byte_offset: scen.raw_byte_offset,
                token_count: scen.raw_token_count,
            };
            let [syn, sem, prag] = compute_sigs(raw_pf, raw_entry, strategy, false);
            let sig_entry = pf.append(&syn, &sem, &prag).expect("append failed");
            tool_entries.get_mut(tool).unwrap().push((scen.id.clone(), sig_entry));
        }
        RawCorpusCache { pf, tool_entries }
    }

    /// Like `scan_raw` but uses a precomputed `RawCorpusCache` so the corpus
    /// signatures are not recomputed from raw KVQ data.
    pub fn scan_raw_cached(
        &self,
        raw_pf: &RawProvenanceFile,
        raw_manifest: &RawManifest,
        probe_id: &str,
        strategy: &SignatureStrategy,
        cache: &RawCorpusCache,
    ) -> HarnessResolver {
        let probe_scen = raw_manifest.scenarios.iter()
            .find(|s| s.id == probe_id)
            .unwrap_or_else(|| panic!("probe not found in raw manifest: {probe_id}"));
        let probe_entry = RawSigEntry {
            byte_offset: probe_scen.raw_byte_offset,
            token_count: probe_scen.raw_token_count,
        };
        let [probe_syn, probe_sem, probe_prag] =
            compute_sigs(raw_pf, probe_entry, strategy, true);

        let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS.iter().map(|&tool| {
            let sid = self.tool_section_ids[tool];
            let entries = cache.tool_entries.get(tool)
                .map(|v| v.iter()
                    .filter(|(id, _)| id.as_str() != probe_id)
                    .map(|(_, e)| *e)
                    .collect::<Vec<_>>())
                .unwrap_or_default();
            (sid, entries)
        }).collect();

        let mut scanner = BdpScanner::new();
        scanner.scan_sections(&cache.pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
            .expect("scan_sections failed");
        let mut resolver = HarnessResolver::new();
        for (&sid, &scores) in scanner.section_scores() {
            resolver.section_scores.insert(sid, scores);
        }
        resolver
    }

    /// Like `scan_raw_with_hits` but uses a precomputed `RawCorpusCache`.
    pub fn scan_raw_with_hits_cached(
        &self,
        raw_pf: &RawProvenanceFile,
        raw_manifest: &RawManifest,
        probe_id: &str,
        strategy: &SignatureStrategy,
        cache: &RawCorpusCache,
    ) -> (HarnessResolver, HashMap<SectionId, Vec<TokenHit>>) {
        let probe_scen = raw_manifest.scenarios.iter()
            .find(|s| s.id == probe_id)
            .unwrap_or_else(|| panic!("probe not found in raw manifest: {probe_id}"));
        let probe_entry = RawSigEntry {
            byte_offset: probe_scen.raw_byte_offset,
            token_count: probe_scen.raw_token_count,
        };
        let [probe_syn, probe_sem, probe_prag] =
            compute_sigs(raw_pf, probe_entry, strategy, true);

        let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS.iter().map(|&tool| {
            let sid = self.tool_section_ids[tool];
            let entries = cache.tool_entries.get(tool)
                .map(|v| v.iter()
                    .filter(|(id, _)| id.as_str() != probe_id)
                    .map(|(_, e)| *e)
                    .collect::<Vec<_>>())
                .unwrap_or_default();
            (sid, entries)
        }).collect();

        let mut scanner = BdpScanner::new().with_record_hits(true);
        scanner.scan_sections(&cache.pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
            .expect("scan_sections failed");
        let mut resolver = HarnessResolver::new();
        for (&sid, &scores) in scanner.section_scores() {
            resolver.section_scores.insert(sid, scores);
        }
        let hit_log = scanner.section_hit_log().iter()
            .map(|(&sid, hits)| (sid, hits.clone()))
            .collect();
        (resolver, hit_log)
    }

    /// Scan all tools against their `_pos_1` probe in one pass.
    ///
    /// Returns `(tool_name, HarnessResolver)` pairs in `TOOLS` order.
    pub fn scan_all_pos1(
        &self,
        pf: &ProvenanceFile,
        manifest: &Manifest,
    ) -> Vec<(&'static str, HarnessResolver)> {
        TOOLS
            .iter()
            .map(|&tool| (tool, self.scan(pf, manifest, &format!("{tool}_pos_1"))))
            .collect()
    }

    // ── Raw-strategy scanning ─────────────────────────────────────────────────

    /// Scan all tool sections against a named probe scenario using raw KVQ data
    /// and the given `SignatureStrategy`.
    #[allow(dead_code)]
    ///
    /// Signatures are computed on-the-fly from the `RawProvenanceFile`; they
    /// are written into a temporary in-memory `ProvenanceFile` so the existing
    /// `BdpScanner::scan_sections` machinery can be reused unchanged.
    pub fn scan_raw(
        &self,
        raw_pf: &RawProvenanceFile,
        raw_manifest: &RawManifest,
        probe_id: &str,
        strategy: &SignatureStrategy,
    ) -> HarnessResolver {
        let probe_scenario = raw_manifest
            .scenarios
            .iter()
            .find(|s| s.id == probe_id)
            .unwrap_or_else(|| panic!("probe scenario not found in raw manifest: {probe_id}"));

        let probe_entry = RawSigEntry {
            byte_offset: probe_scenario.raw_byte_offset,
            token_count: probe_scenario.raw_token_count,
        };

        let [probe_syn, probe_sem, probe_prag] =
            compute_sigs(raw_pf, probe_entry, strategy, true);

        let tmp_pf = ProvenanceFile::new().expect("temporary ProvenanceFile failed");

        let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS
            .iter()
            .map(|&tool| {
                let sid = self.tool_section_ids[tool];
                let entries: Vec<SigEntry> = raw_manifest
                    .scenarios
                    .iter()
                    .filter(|s| {
                        s.tool.as_deref() == Some(tool)
                            && s.case_type == CaseType::Positive
                            && s.id != probe_id
                    })
                    .map(|s| {
                        let raw_entry = RawSigEntry {
                            byte_offset: s.raw_byte_offset,
                            token_count: s.raw_token_count,
                        };
                        let [syn_sigs, sem_sigs, prag_sigs] =
                            compute_sigs(raw_pf, raw_entry, strategy, false);
                        tmp_pf
                            .append(&syn_sigs, &sem_sigs, &prag_sigs)
                            .expect("append to temp ProvenanceFile failed")
                    })
                    .collect();
                (sid, entries)
            })
            .collect();

        let mut scanner = BdpScanner::new();
        scanner
            .scan_sections(&tmp_pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
            .expect("scan_sections failed");

        let mut resolver = HarnessResolver::new();
        for (&sid, &scores) in scanner.section_scores() {
            resolver.section_scores.insert(sid, scores);
        }
        resolver
    }

    /// Like [`Self::scan_raw`] but also returns the per-section hit log for
    /// span scoring.  Each hit records the probe token index, corpus token
    /// index, agreement value, and depth, enabling run-length analysis.
    #[allow(dead_code)]
    pub fn scan_raw_with_hits(
        &self,
        raw_pf: &RawProvenanceFile,
        raw_manifest: &RawManifest,
        probe_id: &str,
        strategy: &SignatureStrategy,
    ) -> (HarnessResolver, HashMap<SectionId, Vec<TokenHit>>) {
        let probe_scenario = raw_manifest
            .scenarios
            .iter()
            .find(|s| s.id == probe_id)
            .unwrap_or_else(|| panic!("probe not found in raw manifest: {probe_id}"));

        let probe_entry = RawSigEntry {
            byte_offset: probe_scenario.raw_byte_offset,
            token_count: probe_scenario.raw_token_count,
        };

        let [probe_syn, probe_sem, probe_prag] =
            compute_sigs(raw_pf, probe_entry, strategy, true);

        let tmp_pf = ProvenanceFile::new().expect("temporary ProvenanceFile failed");

        let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS
            .iter()
            .map(|&tool| {
                let sid = self.tool_section_ids[tool];
                let entries: Vec<SigEntry> = raw_manifest
                    .scenarios
                    .iter()
                    .filter(|s| {
                        s.tool.as_deref() == Some(tool)
                            && s.case_type == CaseType::Positive
                            && s.id != probe_id
                    })
                    .map(|s| {
                        let raw_entry = RawSigEntry {
                            byte_offset: s.raw_byte_offset,
                            token_count: s.raw_token_count,
                        };
                        let [syn_sigs, sem_sigs, prag_sigs] =
                            compute_sigs(raw_pf, raw_entry, strategy, false);
                        tmp_pf
                            .append(&syn_sigs, &sem_sigs, &prag_sigs)
                            .expect("append to temp ProvenanceFile failed")
                    })
                    .collect();
                (sid, entries)
            })
            .collect();

        let mut scanner = BdpScanner::new().with_record_hits(true);
        scanner
            .scan_sections(&tmp_pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
            .expect("scan_sections failed");

        let mut resolver = HarnessResolver::new();
        for (&sid, &scores) in scanner.section_scores() {
            resolver.section_scores.insert(sid, scores);
        }
        let hit_log = scanner
            .section_hit_log()
            .iter()
            .map(|(&sid, hits)| (sid, hits.clone()))
            .collect();
        (resolver, hit_log)
    }

}
