//! Shared journal sections, end to end on the hybrid model
//! (`docs/shared_journal_sections.md` §8).
//!
//! ```text
//! cargo test --release -p zend --test shared_journal_sections -- --ignored --nocapture --test-threads=1
//! ```
//!
//! One conversation seals a chain whose head is shared — two sections no
//! conversation owns, the second ending the shared run — and one section of its
//! own after it. A second conversation carries the same chain. The equivalence
//! the design promises is checked where it is made:
//!
//! - the second conversation **restores** the shared run, rather than
//!   prefilling it, so its K/V is the first's, byte for byte;
//! - the state the run's end keeps on its stream is the state the first
//!   conversation's seal handed its own section, byte for byte — so the second
//!   conversation's own section starts from exactly where the first's did.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection;
use candle_conversation::{
    Carried, Changes, ConversationEngine, Linking, SamplingConfig, SectionRef, Sequence, Sharing,
    Want,
};
use journal_core::section::JOURNAL;

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// A recurrent hybrid, so a section's seal ends on a state worth keeping.
const HYBRID: Model = Model::Qwen36_35B_A3B_Q4;

fn build_engine(workspace: PathBuf, device: &Device) -> (ConversationEngine, Sequence) {
    let dialect = HYBRID.spec().dialect.clone();
    let workspace_str = workspace.display().to_string();
    let mut proj_builder = projection::Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_YAML,
        &[
            ("workspace", workspace_str.as_str()),
            ("model", "test-model"),
        ],
        Some(&dialect),
    )
    .expect("parse projection.yaml");
    let dialogue_layer = proj_builder
        .id_for_layer("dialogue")
        .expect("dialogue layer");
    let primary_group = proj_builder
        .id_for_group("primary_conversation")
        .expect("primary group");
    let mut builder = HYBRID
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(16)
        .thinking(false);
    let conv_config = builder.conversation_config();
    let engine = builder.engine(device).expect("engine load");
    let tokenizer = engine.tokenizer().clone();
    proj_builder
        .tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })
        .expect("tokenize templates");
    let formatted_prompt = builder.format_system_prompt();
    let conv = engine
        .new_conversation_with_projection(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config,
        )
        .expect("new conv");
    (engine, conv)
}

/// The chain both conversations carry: a shared run of two, then one of the
/// conversation's own.
fn chain(own: &str) -> Vec<Want<'_>> {
    vec![
        Want {
            name: "journal/history",
            text: "Earlier in this session:",
            sharing: Sharing::Shared,
        },
        Want {
            name: "journal/1",
            text: "Entry 1 — before this session began, read the workspace's root listing.",
            sharing: Sharing::SharedEnd,
        },
        Want {
            name: "lineage/reader",
            text: own,
            sharing: Sharing::Owned,
        },
    ]
}

/// Carry `wanted` on `conv` and wait until every section has sealed.
fn carry(conv: &mut Sequence, carried: &mut Carried, wanted: &[Want<'_>]) -> Vec<String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    rt.block_on(carried.reconcile(
        conv,
        JOURNAL,
        wanted,
        Linking::Chain,
        Changes::Any,
        &[],
        Some(Duration::from_secs(120)),
        tokio::time::sleep,
    ))
}

/// The section `name` holds on `conv`.
fn section(conv: &Sequence, name: &str) -> SectionRef {
    conv.submitted_sections()
        .into_iter()
        .find(|s| s.name() == name)
        .unwrap_or_else(|| panic!("{name} is submitted"))
}

#[test]
#[ignore = "loads the hybrid model; run on its own with the daemon stopped"]
fn a_second_conversation_restores_the_shared_run_and_starts_its_own_from_the_kept_state() {
    let Ok(device @ Device::Cuda(_)) = Device::cuda_if_available(0) else {
        eprintln!("no CUDA device — skipped");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (engine, mut first) = build_engine(dir.path().to_path_buf(), &device);
    let substrate = engine.conversation();

    let mut carried = Carried::new(true);
    let sealed = carry(
        &mut first,
        &mut carried,
        &chain("This conversation is T:t/one."),
    );
    assert_eq!(sealed.len(), 3, "the first conversation's chain sealed");
    let run_end = section(&first, "journal/1");
    let handed = run_end
        .end_state()
        .expect("the run's end handed its state to the section after it");
    let end_stream = substrate
        .section_stream(run_end.section_id())
        .expect("the run's end has a stream");

    // The shared run becomes durable, and its end state kept, off the seal path.
    let deadline = Instant::now() + Duration::from_secs(60);
    let kept = loop {
        let history = substrate.section_stream(section(&first, "journal/history").section_id());
        let durable = history.is_some_and(|s| substrate.section_stream_is_persisted(s))
            && substrate.section_stream_is_persisted(end_stream);
        if let (true, Some(kept)) = (
            durable,
            substrate
                .read_section_end_state(end_stream)
                .expect("a kept end state reads"),
        ) {
            break kept;
        }
        assert!(
            Instant::now() < deadline,
            "the shared run never became durable"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(kept.schedule_hash, handed.schedule_hash);
    assert_eq!(
        *kept.layers, *handed.layers,
        "the kept state is the handed one"
    );
    assert_eq!(*kept.aux, *handed.aux);

    let before = substrate.section_loads();
    let mut second = first.fork().expect("a second conversation");
    let mut carried = Carried::new(true);
    let sealed = carry(
        &mut second,
        &mut carried,
        &chain("This conversation is T:t/two."),
    );
    assert_eq!(sealed.len(), 3, "the second conversation's chain sealed");
    let after = substrate.section_loads();
    assert_eq!(
        (
            after.restored - before.restored,
            after.prefilled - before.prefilled
        ),
        (2, 1),
        "the shared run restored, only the conversation's own line prefilled"
    );
    for name in ["journal/history", "journal/1"] {
        assert_eq!(
            substrate.section_stream(section(&second, name).section_id()),
            substrate.section_stream(section(&first, name).section_id()),
            "{name}: one stream for both conversations"
        );
    }
    assert_ne!(
        substrate.section_stream(section(&second, "lineage/reader").section_id()),
        substrate.section_stream(section(&first, "lineage/reader").section_id()),
        "each conversation's own line is its own"
    );
}
