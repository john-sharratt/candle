//! Tests that each card's recipe names its rung's artifact, and that the artifact's
//! header matches the engine.
//!
//! Each card gets the rung its VRAM names and a distinct artifact, and every card
//! under 64 GiB shares the floor rung's artifact. The floor rung's digest is checked
//! against the one the small cards hold, because any drift in a pin names a
//! different file and forces a rebuild. The head inventory reads the engine pack's
//! header alone, so it runs on a machine whose sources were released. A fork of a
//! staged parent holds its snapshot, not a store, until it is admitted.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle::quantized::gguf_file::Content;

    use candle::Tensor;

    use std::collections::HashSet;

    use std::fs::File;

    use super::super::tests_support::tests::{engine, pinned_shards, tokenizer};

    /// The pin table and the shard-name function describe the same six files.
    #[test]
    fn the_q8_0_pins_name_the_split() {
        for (i, (path, _, sha)) in QWEN4EXP_Q8_0_FILES.iter().enumerate() {
            assert_eq!(*path, q8_0_shard_name(i + 1));
            assert_eq!(sha.len(), 64);
        }
        for (&i, (path, _, _)) in QWEN4EXP_W4A16_EXPERT_SHARDS
            .iter()
            .zip(QWEN4EXP_W4A16_FILES.iter())
        {
            assert_eq!(*path, w4a16_shard_name(i));
        }
    }

    /// Each card in the fleet gets the rung `quant_ladder` names, the sources
    /// that rung needs, and a distinct artifact.
    #[test]
    fn each_rung_has_its_own_recipe() {
        let laptop = engine_recipe(16);
        assert_eq!(laptop.experts, GgmlDType::Q3_KO);
        assert_eq!(laptop.head_experts, GgmlDType::Q3_KO);
        assert_eq!(laptop.expert_source, ExpertSource::Requantized);
        assert_eq!(laptop.trunk, GgmlDType::Q8_0);
        assert_eq!(laptop.head_dense, GgmlDType::Q8_0);
        assert_eq!(laptop.sources_of(SourceRole::Trunk).len(), 6);
        assert_eq!(laptop.sources_of(SourceRole::DraftHead).len(), 1);
        assert!(laptop.sources_of(SourceRole::ExpertImport).is_empty());

        // Every card under 64 GiB shares the floor rung, and so its artifact.
        for gib in [24, 32] {
            assert_eq!(engine_recipe(gib), laptop, "{gib} GiB");
        }

        let blackwell = engine_recipe(72);
        assert_eq!(blackwell.experts, GgmlDType::Q4_KO);
        assert_eq!(blackwell.expert_source, ExpertSource::AwqImport);
        assert_eq!(blackwell.sources_of(SourceRole::ExpertImport).len(), 4);

        let above = engine_recipe(96);
        assert_eq!(above.experts, GgmlDType::Q8_0);
        assert_eq!(above.expert_source, ExpertSource::Verbatim);

        let names: HashSet<String> = [&laptop, &blackwell, &above]
            .iter()
            .map(|r| r.artifact_name())
            .collect();
        assert_eq!(names.len(), 3, "two rungs share an artifact name");
    }

    /// **The floor rung's recipe names the artifact the small cards hold.** The
    /// digest covers every pin and every build choice, so this is the check
    /// that the recipe here is byte-for-byte the one that artifact was built
    /// from — a drift in any pin would name a different file and force a
    /// rebuild of the whole artifact.
    #[test]
    fn the_floor_rung_names_its_built_artifact() {
        assert_eq!(
            engine_recipe(16).artifact_name(),
            "Qwen3.8-Flash-Next-Q3_KOEXP-503c2f2d6ead.gguf"
        );
    }

    /// **The engine pack's head block and top-level tensors**, from its GGUF
    /// part's header alone — no source is read, so this runs on a machine whose
    /// sources were released after the build. The routed experts are in the
    /// pack's expert section, not in this header.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     the_engine_artifact_head_inventory -- --ignored --nocapture
    #[test]
    #[ignore = "reads the engine pack's header"]
    fn the_engine_artifact_head_inventory() -> Result<()> {
        let pack = engine()?;
        let content = Content::read(&mut File::open(&pack)?)?;
        let mut names: Vec<&String> = content
            .tensor_infos
            .keys()
            .filter(|n| n.starts_with("blk.48.") || !n.starts_with("blk."))
            .collect();
        names.sort();
        for n in names {
            let i = &content.tensor_infos[n];
            println!("  {n}  {:?} {:?}", i.shape.dims(), i.ggml_dtype);
        }
        Ok(())
    }

    /// Does this checkpoint carry a NextN / MTP draft head?
    ///
    /// Speculative decode needs a drafter, and the cheapest one is the head the
    /// checkpoint ships — a single block that reuses the trunk's `lm_head`
    /// (`qwen35::mtp`). Whether Qwen3.8-Flash-Next has one is a property of the
    /// released weights, not of this engine, so it is answered by reading them
    /// rather than by reasoning about the config: this prints every metadata key
    /// and tensor name that mentions the head, so a "no" is a *checked* no.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     the_checkpoint_draft_head_inventory -- --ignored --nocapture
    #[test]
    #[ignore = "reads the engine pack's header"]
    fn the_checkpoint_draft_head_inventory() -> Result<()> {
        use crate::models::latent_moe::GgufModel;
        let pack = engine()?;
        let gguf = GgufModel::open(&[pack])?;
        let mut meta: Vec<String> = gguf
            .metadata
            .iter()
            .filter(|(k, _)| {
                let k = k.to_lowercase();
                k.contains("nextn") || k.contains("mtp") || k.contains("block_count")
            })
            .map(|(k, v)| format!("{k} = {v:?}"))
            .collect();
        meta.sort();
        println!("metadata:");
        for m in &meta {
            println!("  {m}");
        }
        let names = gguf.tensor_names();
        let mut heads: Vec<&String> = names
            .iter()
            .filter(|n| {
                let l = n.to_lowercase();
                // The head's own tensors AND its whole block: the block is what
                // says how the head wires into a hyper-connection stack, and
                // the norms' widths are what say which collapse each one is.
                l.contains("nextn") || l.contains("mtp") || n.starts_with("blk.48.")
            })
            .collect();
        heads.sort();
        println!("draft-head tensors: {}", heads.len());
        // With shapes: the head's own geometry is what decides how it wires to
        // a stack whose residual is `hc` streams wide. `hnorm` spanning
        // `hc_dim` rather than `n_embd` is what says it norms the WIDE residual
        // and the head's mixer narrows it afterwards, and `nextn.hc_head_norm`'s
        // width says the same about the output side.
        for h in heads.iter().take(24) {
            match gguf.info(h) {
                Some(i) => println!("  {h}  {:?} {:?}", i.shape.dims(), i.ggml_dtype),
                None => println!("  {h}  (no info)"),
            }
        }
        // The trunk's own block count, to see whether extra blocks are present
        // beyond the 48 the engine loads.
        let mut max_blk = 0usize;
        for n in names.iter() {
            if let Some(rest) = n.strip_prefix("blk.") {
                if let Some((idx, _)) = rest.split_once('.') {
                    if let Ok(i) = idx.parse::<usize>() {
                        max_blk = max_blk.max(i + 1);
                    }
                }
            }
        }
        println!("highest blk.N present: {max_blk} (engine loads 48)");

        // The pack is built over a CONVERTED artifact, so a head could have been
        // dropped on the way. Ask the pinned upstream Q8_0 split too — that is
        // the released checkpoint, and its answer is the one that decides
        // whether a shipped draft head exists at all.
        let src = pinned_shards()?;
        let src_gguf = GgufModel::open(&src)?;
        let src_names = src_gguf.tensor_names();
        let mut src_heads: Vec<&String> = src_names
            .iter()
            .filter(|n| {
                let n = n.to_lowercase();
                n.contains("nextn") || n.contains("mtp")
            })
            .collect();
        src_heads.sort();
        let mut src_blk = 0usize;
        for n in src_names.iter() {
            if let Some(rest) = n.strip_prefix("blk.") {
                if let Some((idx, _)) = rest.split_once('.') {
                    if let Ok(i) = idx.parse::<usize>() {
                        src_blk = src_blk.max(i + 1);
                    }
                }
            }
        }
        println!(
            "upstream Q8_0: {} tensors, draft-head tensors {}, highest blk.N {src_blk}",
            src_names.len(),
            src_heads.len()
        );
        for h in src_heads.iter().take(24) {
            println!("  {h}");
        }
        Ok(())
    }

    /// **A fork of a parent whose state is held as a snapshot holds that
    /// snapshot, not a store, until it is admitted.** The parent's state is a
    /// real prefill's, exported and staged on a second sequence — what a parked
    /// conversation is after `unpark_from_record`. Forking it builds nothing on
    /// the card; the parent keeps its snapshot for its next view; and the claim
    /// admission makes builds the child a store holding exactly the snapshot's
    /// bytes. Built at the fork instead, every ephemeral fork waiting for
    /// admission held ~288 MiB.
    #[test]
    #[ignore = "loads this card's engine artifact and needs the GPU alone. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::a_fork_of_a_staged_parent_claims_its_store_at_admission \
                -- --ignored --nocapture --test-threads=1"]
    fn a_fork_of_a_staged_parent_claims_its_store_at_admission() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::delta_net::ExportedLayerState;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use std::sync::Arc;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let model = Qwen4ExpBatched::new(Qwen4ExpGpu::load(&pack, &device)?)?;
        let tok = tokenizer()?;
        let ids: Vec<u32> = tok
            .encode(
                "The capital of France is Paris, and the capital of Spain is",
                true,
            )
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let ran = session.create_sequence()?;
        let prompt = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let n_layers = ManagedBatchedModel::num_layers(&model);
        drop(model.forward_wave(
            &mut session,
            &[],
            &[],
            &[ran],
            std::slice::from_ref(&prompt),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?);
        let (hash, layers) = ManagedBatchedModel::export_recurrent(&model, ran)?
            .ok_or_else(|| candle::Error::Msg("the prefill left no recurrent state".into()))?;
        let same = |a: &[ExportedLayerState], b: &[ExportedLayerState]| {
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| {
                    x.layer_index == y.layer_index
                        && x.state == y.state
                        && x.conv_tail == y.conv_tail
                })
        };

        let parent = session.create_sequence()?;
        model.stage_recurrent(parent, hash, Arc::from(layers.clone()))?;
        let stores = model.recurrent_len()?;
        let child = session.create_sequence()?;
        model.fork_recurrent(parent, child)?;
        assert!(
            !model.has_recurrent(child)?,
            "the fork built its store before it was admitted"
        );
        assert_eq!(
            model.recurrent_len()?,
            stores,
            "the fork put a store on the card"
        );
        assert!(
            !model.has_recurrent(parent)?,
            "forking from the parent's snapshot built the parent a store"
        );

        model.claim_carried_state(child)?;
        assert!(
            model.has_recurrent(child)?,
            "admission's claim built no store"
        );
        let (child_hash, child_layers) = ManagedBatchedModel::export_recurrent(&model, child)?
            .ok_or_else(|| candle::Error::Msg("the claimed child exports nothing".into()))?;
        assert_eq!(child_hash, hash);
        assert!(
            same(&child_layers, &layers),
            "the claimed store is not the snapshot the parent held"
        );

        // The parent still holds its snapshot: a second fork takes it too.
        let second = session.create_sequence()?;
        model.fork_recurrent(parent, second)?;
        model.claim_carried_state(second)?;
        let (_, second_layers) = ManagedBatchedModel::export_recurrent(&model, second)?
            .ok_or_else(|| candle::Error::Msg("the second fork exports nothing".into()))?;
        assert!(same(&second_layers, &layers));
        Ok(())
    }
}
