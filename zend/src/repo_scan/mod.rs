//! `repo_map` layer ingestion.
//!
//! Walks the workspace, clusters directories under a token budget,
//! and prefills one user/assistant turn pair per cluster onto the
//! projection schema's `repo_map` layer.  The whole conversation is
//! rebuildable atomically — [`ClusterState`] tracks per-cluster
//! file-name hashes so a filesystem event triggers
//! [`refresh_repo_map`] only when something actually changed.

pub mod cluster;
pub mod types;
pub mod walk;

use std::path::Path;

use candle_conversation::projection::{self, TimelineId};
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};

use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::turn_sink::{InsertTurnSink, SequenceTurnSink};

pub use cluster::{build_clusters, Cluster};
pub use types::{FileEntry, Language, RepoMap};
pub use walk::walk_workspace;

/// Strip auto-summarization from a [`SequenceConfig`] before using
/// it to mint a utility-layer conversation (repo_map, code_reading).
///
/// The default tree config triggers summarization every 8 turns.
/// For the dialogue layer that's correct — long conversations need
/// rolling summaries to stay tractable.  For repo_map and
/// code_reading it's a fatal stall: each cluster's `insert_turn`
/// blocks inside `finalize_turn_post_done` waiting for the
/// summarizer task to complete, and the summarizer holds the
/// engine in a state that prevents the next cluster from making
/// progress.  These layers carry hundreds of small turns whose
/// content is structured (cluster listings, scope reads), not
/// dialogue — they should never be summarized.
pub(crate) fn utility_config(mut config: SequenceConfig) -> SequenceConfig {
    config.tree.summarize_every = 0;
    config.tree.segment_summarize_every = 0;
    // Utility ingests (repo_map, code_reading) are append-only cumulative
    // trunks — each turn just extends the layer. Skip the per-turn projection
    // rebuild (reset + re-project the whole trunk, which is O(n²) and serial on
    // the scheduler thread); turns still seal into the substrate. This lets the
    // parallel workers' prefills/decodes actually batch instead of serialising
    // behind reprojection.
    config.disable_reprojection = true;
    // Utility ingests are cold reference context the dialogue layer retrieves
    // rarely, so they compress their V harder than dialogue's C5 — C6 adaptive
    // selection, with the engine-wide K→Q4_KS override left on. The code-reading
    // layer inherits this same C6 level via `code_read_config`.
    config.kv_compression_level = Some(6);
    config
}

/// Per-cluster state recorded after a successful ingestion pass —
/// the canonical record of "this cluster's file-name set hashed to
/// `content_hash` and lives at `root_dir`".  The refresh path
/// compares newly-walked clusters against this list to decide
/// whether to reset+re-insert.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClusterState {
    pub clusters: Vec<ClusterRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterRecord {
    pub root_dir: String,
    pub content_hash: String,
}

impl ClusterState {
    pub fn from_clusters(clusters: &[Cluster]) -> Self {
        Self {
            clusters: clusters
                .iter()
                .map(|c| ClusterRecord {
                    root_dir: c.root_dir.clone(),
                    content_hash: c.content_hash.clone(),
                })
                .collect(),
        }
    }

    /// Quick equality check used by the refresh path — when this
    /// returns `true`, the new walk produced exactly the same
    /// cluster shape and file-name hashes as the prior ingest, so
    /// no projection-level refresh is required.
    pub fn equivalent_to(&self, clusters: &[Cluster]) -> bool {
        self.clusters.len() == clusters.len()
            && self
                .clusters
                .iter()
                .zip(clusters.iter())
                .all(|(a, b)| a.root_dir == b.root_dir && a.content_hash == b.content_hash)
    }

    /// Set of directory paths whose content hash differs between the
    /// recorded state and a freshly-walked cluster list.  Used by
    /// telemetry / logging when the refresh fires; the refresh
    /// itself is wholesale (reset + re-insert) so this is purely
    /// informational.
    pub fn changed_dirs(&self, clusters: &[Cluster]) -> Vec<String> {
        let mut out = Vec::new();
        let by_root: std::collections::HashMap<&str, &str> = self
            .clusters
            .iter()
            .map(|r| (r.root_dir.as_str(), r.content_hash.as_str()))
            .collect();
        for c in clusters {
            match by_root.get(c.root_dir.as_str()) {
                Some(h) if *h == c.content_hash.as_str() => {}
                _ => out.push(c.root_dir.clone()),
            }
        }
        // Also report directories present in the prior state but
        // absent from the new walk (deleted or merged away).
        let new_roots: std::collections::HashSet<&str> =
            clusters.iter().map(|c| c.root_dir.as_str()).collect();
        for prior in &self.clusters {
            if !new_roots.contains(prior.root_dir.as_str()) {
                out.push(prior.root_dir.clone());
            }
        }
        out
    }
}

/// Sink-driven core of the `repo_map` ingestion — walks the
/// workspace, builds the cluster list, and emits one `(user,
/// assistant)` turn pair per cluster into `sink`.  Returns the
/// walked [`RepoMap`] so the `code_reading` pass can drive its
/// per-file work without re-walking, and the [`ClusterState`] for
/// the refresh path.
pub fn ingest_repo_map_into_sink<S: InsertTurnSink>(
    sink: &mut S,
    workspace: &Path,
    progress: &LoadProgress,
) -> anyhow::Result<(RepoMap, ClusterState)> {
    let map = walk_workspace(workspace);
    let clusters = build_clusters(&map);
    progress.set_step_progress(0, clusters.len() as u64);

    tracing::info!(
        n_files = map.files.len(),
        n_clusters = clusters.len(),
        skipped_extension = map.files_skipped_extension,
        skipped_oversize = map.files_skipped_oversize,
        "repo map walk + cluster complete",
    );

    // All clusters share one repo_map conversation, so their resume-cache
    // tags accumulate into a single metadata bag written ONCE after the loop.
    // Tagging per-cluster would re-persist the whole growing bag N times
    // (O(n²) write volume) on the same conversation.
    let mut tags = std::collections::BTreeMap::new();
    tags.insert("kind".to_string(), "repo_map".to_string());
    for (i, cluster) in clusters.iter().enumerate() {
        // Restart-resume cache: skip clusters already in the substrate
        // (reloaded from the redo log with their content-hash tags).
        let rm_key = format!("rm:{}", cluster.root_dir);
        if sink.unit_cached(&rm_key, &cluster.content_hash) {
            progress.set_step_progress((i + 1) as u64, clusters.len() as u64);
            tracing::debug!(
                target: "zend::repo_scan",
                root = %cluster.root_dir,
                "skip: cluster already in substrate (resume cache hit)",
            );
            continue;
        }
        sink.insert_prefill_turn(&cluster.user_prompt, &cluster.listing)?;
        // Discrete, escaping-safe descriptive fields keyed by cluster root —
        // each value is stored verbatim by the substrate, so no JSON quoting
        // is involved. `rm:<root>` is the cache key the next run probes.
        tags.insert(rm_key, cluster.content_hash.clone());
        tags.insert(
            format!("rmdirs:{}", cluster.root_dir),
            cluster.covered_dirs.len().to_string(),
        );
        tags.insert(
            format!("rmbytes:{}", cluster.root_dir),
            cluster.listing.len().to_string(),
        );
        progress.set_step_progress((i + 1) as u64, clusters.len() as u64);
    }
    // Only the global `kind` tag means nothing changed; skip the write then.
    if tags.len() > 1 {
        sink.tag_unit(&tags);
    }

    let state = ClusterState::from_clusters(&clusters);
    tracing::info!(
        n_clusters_emitted = clusters.len(),
        total_listing_bytes = clusters.iter().map(|c| c.listing.len()).sum::<usize>(),
        "repo map prefilled into substrate",
    );
    Ok((map, state))
}

/// Top-level `repo_map` ingestion — creates the `repo_map`
/// [`Sequence`] on `engine` and runs the sink-driven ingestion
/// against it.  Returns the constructed Sequence (held by the
/// daemon so its sealed K/V stays reachable by the dialogue
/// layer's BDP retrieval), the walked map for the `code_reading`
/// pass, and the recorded [`ClusterState`] for the refresh path.
pub fn ingest_repo_map(
    engine: &ConversationEngine,
    proj_builder: projection::Builder,
    workspace: &Path,
    config: SequenceConfig,
    progress: &LoadProgress,
    skip: bool,
) -> anyhow::Result<(Sequence, RepoMap, ClusterState)> {
    let layer = proj_builder
        .id_for_layer("repo_map")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'repo_map' layer"))?;
    let group = proj_builder
        .id_for_group("structure")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'structure' group"))?;

    let system_prompt = layer_system_prompt(&proj_builder, "repo_map", &config);
    let mut sequence = engine
        .new_conversation_with_projection(
            &system_prompt,
            proj_builder,
            layer,
            group,
            utility_config(config),
        )
        .map_err(|e| anyhow::anyhow!("repo_map conv create: {e}"))?;

    // `--skip-repo-scan`: the `repo_map` layer conversation is still minted (the
    // projection schema requires it) but left empty — no walk, no cluster ingest.
    let (map, state) = if skip {
        tracing::info!("--skip-repo-scan: bypassing repo-map walk + cluster ingest");
        (RepoMap::default(), ClusterState::default())
    } else {
        let mut sink = SequenceTurnSink::new(&mut sequence);
        ingest_repo_map_into_sink(&mut sink, workspace, progress)?
    };
    Ok((sequence, map, state))
}

/// Outcome of an atomic [`refresh_repo_map`] call.
///
/// `NoOp` means the cluster hashes matched `prior` — the existing
/// `Sequence` is untouched.  `Replaced` carries a freshly minted
/// `Sequence` on a new timeline whose clusters are already
/// prefilled.  The old timeline has been tombstoned by the time
/// this is returned, so the caller's only remaining job is to swap
/// the new `Sequence` into its slot and drop the old one.
///
/// `Replaced` carries a heavy `Sequence`; `NoOp` is empty.  Only
/// one outcome is in flight at a time and the caller consumes it
/// immediately, so the size disparity is not worth boxing.
#[allow(clippy::large_enum_variant)]
pub enum RefreshOutcome {
    NoOp,
    Replaced {
        sequence: Sequence,
        state: ClusterState,
    },
}

/// Atomic refresh of the `repo_map` conversation.
///
/// Re-clusters `map` (which the caller has already walked — usually
/// once per filesystem-event burst and shared with the
/// `code_reading` refresh).  Returns `NoOp` when nothing changed.
/// Otherwise mints a new `Sequence` on a fresh timeline, prefills
/// every cluster into it, then tombstones the old timeline.  The
/// new timeline registers alongside the old; until the tombstone
/// fires the resolver picks the older one, so dialogue retrieval
/// keeps seeing the prior content throughout the refresh — stale
/// better than missing.  At the tombstone instant retrieval flips
/// atomically to the new content.  The next compaction pass drops
/// the old timeline's records from disk.
///
/// The engine mutex on `ctx` is acquired only briefly — once at the
/// start to mint the new `Sequence` and once at the end to write
/// the tombstone.  The prefill loop in between runs lock-free so
/// concurrent chat turns aren't blocked for the duration of the
/// refresh.
pub fn refresh_repo_map(
    ctx: &RefreshContext<'_>,
    map: &RepoMap,
    prior: &ClusterState,
    old_timeline: TimelineId,
    progress: &LoadProgress,
) -> anyhow::Result<RefreshOutcome> {
    let clusters = build_clusters(map);
    if prior.equivalent_to(&clusters) {
        tracing::debug!("repo map refresh: no cluster hash changed, skipping refresh");
        return Ok(RefreshOutcome::NoOp);
    }

    let changed = prior.changed_dirs(&clusters);
    tracing::info!(
        n_changed = changed.len(),
        sample_changed = ?changed.iter().take(5).collect::<Vec<_>>(),
        n_total_clusters = clusters.len(),
        "repo map refresh: minting new timeline and prefilling",
    );

    let layer = ctx
        .proj_builder
        .id_for_layer("repo_map")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'repo_map' layer"))?;
    let group = ctx
        .proj_builder
        .id_for_group("structure")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'structure' group"))?;
    let system_prompt = layer_system_prompt(&ctx.proj_builder, "repo_map", &ctx.config);

    let mut new_sequence = {
        let engine = ctx.engine.lock().unwrap();
        engine
            .new_conversation_with_projection(
                &system_prompt,
                ctx.proj_builder.clone(),
                layer,
                group,
                utility_config(ctx.config.clone()),
            )
            .map_err(|e| anyhow::anyhow!("repo_map refresh: new conv create: {e}"))?
    };

    // Lock-free prefill window — concurrent engine consumers (chat
    // commits, sidebar reads) keep running while we prefill the
    // new clusters into the freshly minted sequence.
    {
        let mut sink = SequenceTurnSink::new(&mut new_sequence);
        for (i, cluster) in clusters.iter().enumerate() {
            sink.insert_prefill_turn(&cluster.user_prompt, &cluster.listing)?;
            progress.set_step_progress((i + 1) as u64, clusters.len() as u64);
        }
    }

    {
        let engine = ctx.engine.lock().unwrap();
        engine
            .tombstone_timeline(old_timeline)
            .map_err(|e| anyhow::anyhow!("repo_map refresh: tombstone old timeline: {e}"))?;
    }

    Ok(RefreshOutcome::Replaced {
        sequence: new_sequence,
        state: ClusterState::from_clusters(&clusters),
    })
}

/// Pull the layer's system-prompt sections out of the schema and
/// wrap them with the engine's dialect markers.  The `repo_map`
/// and `code_reading` conversations only project from the layer
/// they target, so the assembly path here mirrors the dialogue
/// layer's `pre_collection_prelude` in `session.rs`.
fn layer_system_prompt(
    builder: &projection::Builder,
    layer_name: &str,
    config: &SequenceConfig,
) -> String {
    use projection::SystemPromptItem;
    let layer = builder
        .schema()
        .layers
        .iter()
        .find(|l| l.name == layer_name)
        .unwrap_or_else(|| panic!("projection schema missing '{layer_name}' layer"));

    let mut body = String::new();
    for item in &layer.system_prompt.items {
        if let SystemPromptItem::Section(s) = item {
            body.push_str(&s.content);
        }
    }
    config.dialect.format_system_prompt(&body)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_state_equivalence_detects_no_change() {
        let cs = vec![Cluster {
            root_dir: "src/".into(),
            covered_dirs: vec!["src/".into()],
            content_hash: "abc".into(),
            user_prompt: "x".into(),
            listing: "y".into(),
        }];
        let st = ClusterState::from_clusters(&cs);
        assert!(st.equivalent_to(&cs));
        assert!(st.changed_dirs(&cs).is_empty());
    }

    #[test]
    fn cluster_state_equivalence_detects_hash_change() {
        let cs_before = vec![Cluster {
            root_dir: "src/".into(),
            covered_dirs: vec!["src/".into()],
            content_hash: "abc".into(),
            user_prompt: "x".into(),
            listing: "y".into(),
        }];
        let cs_after = vec![Cluster {
            root_dir: "src/".into(),
            covered_dirs: vec!["src/".into()],
            content_hash: "def".into(),
            user_prompt: "x".into(),
            listing: "y".into(),
        }];
        let st = ClusterState::from_clusters(&cs_before);
        assert!(!st.equivalent_to(&cs_after));
        assert_eq!(st.changed_dirs(&cs_after), vec!["src/".to_string()]);
    }

    #[test]
    fn cluster_state_equivalence_detects_added_and_removed_clusters() {
        let cs_before = vec![Cluster {
            root_dir: "src/".into(),
            covered_dirs: vec!["src/".into()],
            content_hash: "abc".into(),
            user_prompt: "x".into(),
            listing: "y".into(),
        }];
        let cs_after = vec![
            Cluster {
                root_dir: "src/".into(),
                covered_dirs: vec!["src/".into()],
                content_hash: "abc".into(),
                user_prompt: "x".into(),
                listing: "y".into(),
            },
            Cluster {
                root_dir: "docs/".into(),
                covered_dirs: vec!["docs/".into()],
                content_hash: "ghi".into(),
                user_prompt: "x".into(),
                listing: "y".into(),
            },
        ];
        let st = ClusterState::from_clusters(&cs_before);
        assert!(!st.equivalent_to(&cs_after));
        assert!(st.changed_dirs(&cs_after).contains(&"docs/".to_string()));
    }
}
