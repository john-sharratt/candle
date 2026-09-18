//! Generate personality-vector binaries from an existing personality preset.
//!
//! The preset is read, its trigger prompts are run through the common
//! conversation engine, and the resulting anonymized N-1 activations are
//! clustered into a personality-vector artifact. The preset is never rewritten.
//!
//! ```text
//! cargo run -p candle-conversation --bin personality_vector_tool --release -- \
//!   --input candle-conversation/personality_presets/free.yaml \
//!   --output candle-conversation/personality_presets/free.vectors.bin
//! ```

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use candle::Device;
use clap::{Args as ClapArgs, Parser, Subcommand};
use candle_conversation::activation_capture::{ActivationRecord, CapturePhase};
use candle_conversation::models::Model;
use candle_conversation::personality::PersonalityPreset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Polarity {
    Down,
    Up,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabeledActivation {
    polarity: Polarity,
    relative_position: i8,
    layer: usize,
    hidden: usize,
    values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersonalityVectorFile {
    magic: String,
    version: u32,
    source_records: usize,
    downshift_records: usize,
    similarity_threshold: f32,
    default_scale: f32,
    groups: Vec<PersonalityVectorGroup>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersonalityVectorGroup {
    layer: usize,
    relative_position: i8,
    members: usize,
    scale: f32,
    hidden: usize,
    centroid: Vec<f32>,
}

#[derive(Debug, Parser)]
#[command(name = "personality_vector_tool")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate personality vectors from a personality preset.
    Generate(GenerateArgs),
    /// Test a personality preset against its trigger prompts.
    Test(TestArgs),
}

#[derive(Debug, ClapArgs)]
struct GenerateArgs {
    /// Existing personality preset YAML to read.
    #[arg(long, short)]
    input: std::path::PathBuf,
    /// Personality-vector Bincode artifact to write.
    #[arg(long, short)]
    output: std::path::PathBuf,
    /// Cosine threshold used to group personality activations.
    #[arg(long, default_value_t = 0.80)]
    threshold: f32,
}

#[derive(Debug, ClapArgs)]
struct TestArgs {
    /// Existing personality-vector Bincode artifact to load and validate.
    #[arg(long, short)]
    vectors: std::path::PathBuf,
    /// Existing personality preset YAML to read.
    #[arg(long, short)]
    personality: std::path::PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Command::Generate(args) => generate(args),
        Command::Test(args) => test_personality(args),
    }
}

fn generate(args: GenerateArgs) -> Result<(), Box<dyn std::error::Error>> {

    let preset = PersonalityPreset::from_path(&args.input)?;
    let default_scale = 0.0_f32;
    let group_scale = preset
        .personality_steers
        .iter()
        .map(|steer| steer.weight)
        .sum::<f32>()
        / preset.personality_steers.len().max(1) as f32;
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        return Err("personality_vector_tool requires CUDA".into());
    }
    let raw_path = std::env::temp_dir().join(format!(
        "personality-vector-tool-{}.raw.bin",
        std::process::id()
    ));
    let _ = fs::remove_file(&raw_path);
    let workspace = std::env::temp_dir().join(format!(
        "personality-vector-tool-workspace-{}",
        std::process::id()
    ));
    fs::create_dir_all(&workspace)?;

    let mut builder = Model::Qwen35_0_8B_Q8
        .builder()
        .max_concurrent(32)
        .max_response_tokens(128)
        .workspace_path(workspace)
        .activation_capture(&raw_path);
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();
    let started = Instant::now();
    let engine = builder.engine(&device)?;
    eprintln!("loaded personality model in {:.1}s", started.elapsed().as_secs_f64());
    let decoder = engine.token_decoder();
    let mut raw_offset = 0usize;
    let mut known = BTreeMap::new();
    let mut labeled = Vec::new();

    for prompts in preset.trigger_prompts.chunks(32) {
        let mut conversations = prompts
            .iter()
            .map(|_| engine.new_conversation(&system_prompt, config.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut handles = Vec::with_capacity(prompts.len());
        for (conversation, prompt) in conversations.iter_mut().zip(prompts) {
            handles.push(conversation.submit_turn(prompt)?);
        }
        let mut outcomes = BTreeMap::new();
        let mut finishes = Vec::with_capacity(prompts.len());
        for (offset, (conversation, handle)) in conversations
            .iter_mut()
            .zip(handles)
            .enumerate()
        {
            let response = handle.wait()?;
            let _ = decoder.decode(&response.token_ids);
            let down = preset.personality_steers.iter().any(|steer| {
                response
                    .text
                    .to_ascii_lowercase()
                    .contains(&steer.text.to_ascii_lowercase())
            });
            let sequence = conversation.id().to_string().parse::<usize>()?;
            let polarity = if down { Polarity::Down } else { Polarity::Up };
            outcomes.insert(sequence, polarity);
            finishes.push((offset, handle, response));
        }
        append_batch(&raw_path, &mut raw_offset, &outcomes, &mut known, &mut labeled)?;
        for (offset, handle, response) in finishes {
            conversations[offset].finish_turn(handle, &response)?;
        }
    }
    let _ = fs::remove_file(&raw_path);

    let downshift_records = labeled
        .iter()
        .filter(|record| matches!(record.polarity, Polarity::Down))
        .count();
    let groups = group_vectors(&labeled, args.threshold, group_scale)?;
    let file = PersonalityVectorFile {
        magic: "CANDLE-PERSONALITY-VECTORS".into(),
        version: 1,
        source_records: labeled.len(),
        downshift_records,
        similarity_threshold: args.threshold,
        default_scale,
        groups,
    };
    fs::write(&args.output, bincode::serialize(&file)?)?;
    println!("input: {}", args.input.display());
    println!("output: {}", args.output.display());
    println!("source records: {}", file.source_records);
    println!("downshift records: {}", file.downshift_records);
    println!("personality vector groups: {}", file.groups.len());
    println!("default scale: {}", file.default_scale);
    println!("group scale: {}", group_scale);
    Ok(())
}

fn test_personality(args: TestArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vector_bytes = fs::read(&args.vectors)?;
    let vector_file: PersonalityVectorFile = bincode::deserialize(&vector_bytes)?;
    if vector_file.magic != "CANDLE-PERSONALITY-VECTORS" || vector_file.version != 1 {
        return Err("unsupported personality-vector artifact".into());
    }
    let preset = PersonalityPreset::from_path(&args.personality)?;
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        return Err("personality_vector_tool requires CUDA".into());
    }
    let workspace = std::env::temp_dir().join(format!(
        "personality-vector-test-workspace-{}",
        std::process::id()
    ));
    fs::create_dir_all(&workspace)?;
    let mut builder = Model::Qwen35_0_8B_Q8
        .builder()
        .max_concurrent(32)
        .max_response_tokens(128)
        .personality_vectors_path(&args.vectors)
        .workspace_path(workspace);
    let system_prompt = builder.format_system_prompt();
    let config = builder.conversation_config();
    let engine = builder.engine(&device)?;
    let mut hits = 0usize;
    let mut misses = 0usize;
    for prompt in &preset.trigger_prompts {
        let mut conversation = engine.new_conversation(&system_prompt, config.clone())?;
        let response = conversation.send_turn(prompt)?;
        let mut negative_match = false;
        let mut positive_match = false;
        for steer in &preset.personality_steers {
            let steer_tokens = engine
                .tokenizer()
                .encode(steer.text.as_str(), false)
                .map_err(|error| error.to_string())?
                .get_ids()
                .to_vec();
            if contains_token_sequence(&response.token_ids, &steer_tokens) {
                if steer.weight > 0.0 {
                    positive_match = true;
                } else if steer.weight < 0.0 {
                    negative_match = true;
                }
            }
        }
        if negative_match {
            misses += 1;
        } else if positive_match || !negative_match {
            hits += 1;
        }
    }
    let total = hits + misses;
    println!("vectors loaded: {}", vector_file.groups.len());
    println!("trigger prompts tested: {total}");
    println!("personality hits: {hits}");
    println!("personality misses: {misses}");
    if total == 0 {
        println!("hit rate: 0.0%");
        println!("miss rate: 0.0%");
    } else {
        println!("hit rate: {:.1}%", hits as f64 * 100.0 / total as f64);
        println!("miss rate: {:.1}%", misses as f64 * 100.0 / total as f64);
    }
    Ok(())
}

fn contains_token_sequence(haystack: &[u32], needle: &[u32]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

fn append_batch(
    raw_path: &Path,
    raw_offset: &mut usize,
    outcomes: &BTreeMap<usize, Polarity>,
    known: &mut BTreeMap<usize, Polarity>,
    labeled: &mut Vec<LabeledActivation>,
) -> Result<(), Box<dyn std::error::Error>> {
    known.extend(outcomes.iter().map(|(&id, value)| (id, value.clone())));
    let mut raw = OpenOptions::new().read(true).open(raw_path)?;
    let mut bytes = Vec::new();
    raw.read_to_end(&mut bytes)?;
    let mut offset = *raw_offset;
    let mut pending = Vec::new();
    while offset < bytes.len() {
        let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
        offset += 4;
        let record: ActivationRecord = bincode::deserialize(&bytes[offset..offset + length])?;
        if let Some(polarity) = outcomes.get(&record.sequence_id).or_else(|| known.get(&record.sequence_id)) {
            pending.push((record, polarity.clone()));
        }
        offset += length;
    }
    let mut ends: BTreeMap<usize, usize> = BTreeMap::new();
    let mut first_decode: BTreeMap<usize, usize> = BTreeMap::new();
    for (record, _) in &pending {
        if matches!(record.phase, CapturePhase::Prefill) {
            ends.entry(record.sequence_id)
                .and_modify(|end| *end = (*end).max(record.position))
                .or_insert(record.position);
        } else {
            first_decode
                .entry(record.sequence_id)
                .and_modify(|position| *position = (*position).min(record.position))
                .or_insert(record.position);
        }
    }
    for (record, polarity) in pending {
        let target = ends
            .get(&record.sequence_id)
            .map(|end| end.saturating_sub(1))
            .or_else(|| first_decode.get(&record.sequence_id).copied());
        if target == Some(record.position) {
            labeled.push(LabeledActivation {
                polarity,
                relative_position: -1,
                layer: record.layer,
                hidden: record.hidden,
                values: record.values,
            });
        }
    }
    *raw_offset = offset;
    Ok(())
}

fn group_vectors(records: &[LabeledActivation], threshold: f32, scale: f32) -> Result<Vec<PersonalityVectorGroup>, Box<dyn std::error::Error>> {
    let mut by_layer: BTreeMap<usize, Vec<&LabeledActivation>> = BTreeMap::new();
    for record in records.iter().filter(|record| matches!(record.polarity, Polarity::Down)) {
        by_layer.entry(record.layer).or_default().push(record);
    }
    let mut groups = Vec::new();
    for (layer, records) in by_layer {
        let mut clusters: Vec<Vec<&LabeledActivation>> = Vec::new();
        for record in records {
            if let Some(group) = clusters.iter_mut().find(|group| group.iter().any(|item| cosine(&record.values, &item.values) >= threshold)) {
                group.push(record);
            } else {
                clusters.push(vec![record]);
            }
        }
        for cluster in clusters {
            let hidden = cluster[0].hidden;
            let mut centroid = vec![0.0; hidden];
            for record in &cluster {
                for (sum, value) in centroid.iter_mut().zip(&record.values) { *sum += *value; }
            }
            for value in &mut centroid { *value /= cluster.len() as f32; }
            normalize(&mut centroid);
            groups.push(PersonalityVectorGroup { layer, relative_position: -1, members: cluster.len(), scale, hidden, centroid });
        }
    }
    Ok(groups)
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let ln = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let rn = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (ln * rn).max(f32::MIN_POSITIVE)
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::MIN_POSITIVE { for value in values { *value /= norm; } }
}
