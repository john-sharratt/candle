//! Interactive chat binary using candle-conversation.
//!
//! Uses a pre-configured model from the `models` module for streamlined setup.
//!
//! Run with:
//! ```bash
//! cargo run --example chat -p candle-conversation --release --features "cuda,hub"
//! mradermacher/Qwen3-30B-A3B-abliterated-erotic-i1-GGUF
//! ```

use candle::Device;
use candle_conversation::{
    conversation_log::{load_resume_log, truncate_for_display},
    models::{Model, ModelBuilder},
    narrator, SequenceConfig, ConversationEngine, DecodeHealthConfig, Role, SamplingConfig,
    TokenDecoder, TurnEvent,
};
use clap::Parser;
use clap_verbosity_flag::*;
use std::{
    fmt,
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

// ─────────────────────────────────────────────────────────────────────────────
// Help text
// ─────────────────────────────────────────────────────────────────────────────

const NARRATOR_HELP: &str = "\
Commands:
  /say [--character NAME] text...    — dialogue
  /act [--character NAME] action...  — action
  /scene description...              — world event
  /cue [--character NAME] action...  — cue an NPC
  /beat description...               — narrative hint
  Free text (no /)                   — author mode (LLM converts)
  Empty line (buffer queued)         — submit to narrator
  Empty line (nothing queued)        — advance scene with a beat
  /quit                              — exit
─────────────────────────────────────────────────────────────
";

const CHAT_HELP: &str = "\
Ready to chat!

Commands:
  /quit   — exit
  /fork   — fork the conversation (creates a checkpoint)
  /stats  — show conversation stats
  /turns  — show turn history
─────────────────────────────────────────────────────────────
";

// ─────────────────────────────────────────────────────────────────────────────
// NarratorInput display
// ─────────────────────────────────────────────────────────────────────────────

struct DisplayNarratorInput<'a>(&'a narrator::NarratorInput);

impl fmt::Display for DisplayNarratorInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            narrator::NarratorInput::Say { character, text } => {
                write!(f, "say [{character}]: {text}")
            }
            narrator::NarratorInput::Act { character, action } => {
                write!(f, "act [{character}]: {action}")
            }
            narrator::NarratorInput::Scene { description } => write!(f, "scene: {description}"),
            narrator::NarratorInput::Cue { character, action } => {
                write!(f, "cue [{character}]: {action}")
            }
            narrator::NarratorInput::Beat { description } => write!(f, "beat: {description}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NarratorState
// ─────────────────────────────────────────────────────────────────────────────

/// All state owned by narrator mode. Only exists when both `--protagonist` and
/// `--persona` are provided.
struct NarratorState {
    engine: narrator::NarratorEngine,
    char_conv: candle_conversation::Sequence,
    session: narrator::SessionConfig,
    converter_config: SequenceConfig,
    protagonist: String,
    persona: String,
    buffer: Vec<narrator::NarratorInput>,
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model = if let Some(ref name) = args.model {
        parse_model(name)?
    } else {
        Model::Qwen3_14B_Q4
    };

    let min_level = verbosity_to_level(&args.verbose);
    init_file_logging(min_level)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          candle-conversation · Interactive Chat              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut builder = model
        .builder()
        .max_response_tokens(4096)
        .compression_level(0);

    if let Some(ref model_path) = args.model_dir {
        builder = apply_model_dir(builder, model_path)?;
    }

    let resume_log = if let Some(ref path) = args.resume {
        Some(load_resume_log(path)?)
    } else {
        None
    };

    // System prompt priority: explicit flag > resume log > builder default.
    if let Some(ref path) = args.system_prompt_file {
        builder = builder.system_prompt(read_file_to_string(path)?);
    } else if let Some(ref log) = resume_log {
        builder = builder.system_prompt(log.character_system_prompt.clone());
    }

    builder = apply_sampling_overrides(builder, &args);

    if let Some(ref path) = args.penalty_log {
        builder = builder.penalty_log(path);
    }

    if args.show_hidden {
        builder = builder.show_special_tokens(true);
    }

    let device = Device::cuda_if_available(0)?;
    println!("  Device: {:?}", device);

    builder = builder.health(DecodeHealthConfig::for_chat());

    let start = std::time::Instant::now();
    let engine = builder.engine(&device)?;
    let decoder = engine.token_decoder();
    println!("  Model: {}", builder);
    println!("  Loaded in {:.2}s", start.elapsed().as_secs_f64());
    print_startup_info(&args, &builder);

    // ── Mode-specific setup ───────────────────────────────────────────────

    let mode = if args.protagonist.is_some() && args.persona.is_some() {
        let protagonist = args.protagonist.as_deref().unwrap();
        let persona = args.persona.as_deref().unwrap();
        let char_system = if let Some(ref path) = args.system_prompt_file {
            read_file_to_string(path)?
        } else {
            narrator::character_system_prompt(persona, protagonist)
        };
        let narrator_state =
            setup_narrator(&engine, &builder, protagonist, persona, &char_system)?;
        Mode::Narrator(narrator_state)
    } else {
        let (conv, token_file) = setup_chat(&engine, &builder, &args, resume_log)?;
        Mode::Chat { conv, token_file }
    };

    run_loop(mode, &engine, &decoder, &args)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode enum
// ─────────────────────────────────────────────────────────────────────────────

enum Mode {
    Narrator(NarratorState),
    Chat {
        conv: candle_conversation::Sequence,
        token_file: Option<File>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup helpers
// ─────────────────────────────────────────────────────────────────────────────

fn setup_narrator(
    engine: &ConversationEngine,
    builder: &ModelBuilder,
    protagonist: &str,
    persona: &str,
    char_system: &str,
) -> anyhow::Result<NarratorState> {
    let base_config = builder.conversation_config();
    let mut converter_config = base_config.clone();
    converter_config.sampling = SamplingConfig::argmax();

    let char_conv = engine.new_conversation(char_system, base_config.clone())?;
    // initial_handle() removed: system seeding is now lazy

    let narrator_engine = narrator::NarratorEngine::new(engine, base_config)?;

    println!("  Protagonist : {protagonist}");
    println!("  Persona     : {persona}");
    println!();
    print!("{NARRATOR_HELP}");

    Ok(NarratorState {
        engine: narrator_engine,
        char_conv,
        session: narrator::SessionConfig {
            protagonist: protagonist.to_string(),
            persona: persona.to_string(),
            max_turns: narrator::DEFAULT_MAX_TURNS,
        },
        converter_config,
        protagonist: protagonist.to_string(),
        persona: persona.to_string(),
        buffer: Vec::new(),
    })
}

fn setup_chat(
    engine: &ConversationEngine,
    builder: &ModelBuilder,
    args: &Args,
    resume_log: Option<candle_conversation::conversation_log::ResumeLog>,
) -> anyhow::Result<(candle_conversation::Sequence, Option<File>)> {
    let mut c = engine.new_conversation(
        &builder.format_system_prompt(),
        builder.conversation_config(),
    )?;

    if let Some(log) = resume_log {
        println!("  Resuming from {} turns…", log.turns.len());
        for entry in &log.turns {
            c.insert_turn(&entry.user_message, &entry.character_response)?;
        }
        println!("  Done.");
        println!();
        print_resume_tail(&c);
    }

    print!("{CHAT_HELP}");

    let token_file = args
        .token_file
        .as_ref()
        .map(|path| File::create(path).expect("failed to create token file"));

    Ok((c, token_file))
}

fn print_resume_tail(c: &candle_conversation::Sequence) {
    let all_turns = c.turns();
    let pairs: Vec<_> = all_turns
        .windows(2)
        .filter(|w| w[0].role == Role::User && w[1].role == Role::Assistant)
        .collect();
    let last_n: Vec<_> = pairs.iter().rev().take(4).rev().collect();
    if !last_n.is_empty() {
        println!("─────────────────────────────────────────────────────────────");
        println!("  Last {} turn(s) from resumed conversation:", last_n.len());
        println!("─────────────────────────────────────────────────────────────");
        for pair in last_n {
            let user_text = truncate_for_display(&pair[0].text, 300);
            let asst_text = truncate_for_display(&pair[1].text, 300);
            println!("You: {user_text}");
            println!("Assistant: {asst_text}");
            println!();
        }
        println!("─────────────────────────────────────────────────────────────\n");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input loop
// ─────────────────────────────────────────────────────────────────────────────

fn run_loop(
    mut mode: Mode,
    engine: &ConversationEngine,
    decoder: &TokenDecoder,
    args: &Args,
) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        match mode {
            Mode::Narrator(_) => print!("story> "),
            Mode::Chat { .. } if !args.show_hidden => print!("You: "),
            _ => {}
        }
        io::stdout().flush()?;

        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };
        let trimmed = line.trim();

        if trimmed == "/quit" || trimmed == "/exit" {
            println!("Goodbye!");
            break;
        }

        match mode {
            Mode::Narrator(ref mut state) => {
                handle_narrator_turn(state, trimmed, engine, decoder)?;
            }
            Mode::Chat {
                ref mut conv,
                ref mut token_file,
            } => {
                handle_chat_turn(conv, token_file, trimmed, decoder, args)?;
            }
        }
    }

    if let Mode::Chat { conv, .. } = mode {
        conv.close()?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Narrator turn handler
// ─────────────────────────────────────────────────────────────────────────────

fn handle_narrator_turn(
    state: &mut NarratorState,
    trimmed: &str,
    engine: &ConversationEngine,
    decoder: &TokenDecoder,
) -> anyhow::Result<()> {
    if trimmed.is_empty() {
        run_narrator_exchange(state, engine, decoder)?;
    } else if let Some(cmd) = trimmed.strip_prefix('/') {
        match narrator::parse_turn(cmd, &state.session) {
            Ok(inputs) => state.buffer.extend(inputs),
            Err(e) => println!("  [parse error: {e}]"),
        }
    } else {
        convert_and_queue_prose(state, trimmed, engine)?;
    }
    Ok(())
}

fn run_narrator_exchange(
    state: &mut NarratorState,
    engine: &ConversationEngine,
    decoder: &TokenDecoder,
) -> anyhow::Result<()> {
    if state.buffer.is_empty() {
        state.buffer.push(narrator::NarratorInput::Beat {
            description: "continue the scene naturally, advancing the moment \
                          without resolving anything specific"
                .to_string(),
        });
    }
    let json = serde_json::to_string(&state.buffer)?;

    // ── Stage 1: narrator → prose ─────────────────────────────────────────
    println!();
    println!("─────────────────────────────────────────────────────────────");
    println!("  Narrator:");
    println!("─────────────────────────────────────────────────────────────");
    let narrator_text = match narrator_stream(&mut state.engine.conversation, &json, decoder) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  [narrator error: {e}]");
            state.buffer.clear();
            return Ok(());
        }
    };

    // ── Stage 2: character → response ─────────────────────────────────────
    println!();
    println!("─────────────────────────────────────────────────────────────");
    println!("  {}:", state.persona);
    println!("─────────────────────────────────────────────────────────────");
    let char_text = match narrator_stream(&mut state.char_conv, &narrator_text, decoder) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  [character error: {e}]");
            state.buffer.clear();
            return Ok(());
        }
    };

    println!();
    println!("─────────────────────────────────────────────────────────────");

    // ── Stage 3: feed character response back to narrator ─────────────────
    // Convert the character's response to compact waypoints, then run the
    // narrator on them so it stores its own coherent second-person prose in
    // the KV cache — not raw first-person character text.
    eprintln!("  [converting response to waypoints…]");
    if let Err(e) = state.engine.insert_character_response_streaming(
        &char_text,
        &state.persona,
        engine,
        state.converter_config.clone(),
        decoder,
    ) {
        eprintln!("  [waypoint conversion failed ({e}), narrator context not updated]");
    }

    state.buffer.clear();
    Ok(())
}

fn convert_and_queue_prose(
    state: &mut NarratorState,
    prose: &str,
    engine: &ConversationEngine,
) -> anyhow::Result<()> {
    println!("  [converting prose…]");
    match narrator::text_to_inputs(
        prose,
        narrator::ConverterMode::Author(&state.protagonist),
        3,
        engine,
        state.converter_config.clone(),
    ) {
        Ok(inputs) if !inputs.is_empty() => {
            for input in &inputs {
                println!("  [queued] {}", DisplayNarratorInput(input));
            }
            state.buffer.extend(inputs);
        }
        Ok(_) => println!("  [converter returned no inputs, skipping]"),
        Err(e) => {
            eprintln!("  [conversion failed ({e}), using scene fallback]");
            state.buffer.push(narrator::NarratorInput::Scene {
                description: prose.to_string(),
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat turn handler
// ─────────────────────────────────────────────────────────────────────────────

fn handle_chat_turn(
    c: &mut candle_conversation::Sequence,
    token_file: &mut Option<File>,
    trimmed: &str,
    decoder: &TokenDecoder,
    args: &Args,
) -> anyhow::Result<()> {
    if trimmed.is_empty() {
        return Ok(());
    }

    match trimmed {
        "/fork" => {
            match c.fork() {
                Ok(_) => println!("  [forked conversation as checkpoint]\n"),
                Err(e) => println!("  [fork failed: {e}]\n"),
            }
            return Ok(());
        }
        "/stats" => {
            println!("  Sequence ID:      {}", c.id());
            println!("  Turns:            {}", c.turn_count());
            println!("  In-flight:        {}\n", c.is_in_flight());
            return Ok(());
        }
        "/turns" => {
            for turn in c.turns() {
                let preview = if turn.text.len() > 80 {
                    format!("{}...", &turn.text[..80])
                } else {
                    turn.text.clone()
                };
                println!(
                    "  [{}] {:?}: {} tokens — {:?}",
                    turn.id,
                    turn.role,
                    turn.token_ids.len(),
                    preview
                );
            }
            println!();
            return Ok(());
        }
        _ => {}
    }

    let handle = match c.submit_turn(trimmed) {
        Ok(h) => h,
        Err(e) => {
            println!("  [error submitting turn: {e}]\n");
            return Ok(());
        }
    };

    if let Some(ref mut tf) = token_file {
        writeln!(tf, "--- turn: {} ---", trimmed).ok();
    }

    let show_special = args.show_hidden;
    let line_prefix = if show_special { "" } else { "Assistant: " };
    match stream_turn(c, handle, decoder, line_prefix, show_special, token_file.as_mut()) {
        Ok((_, boundary)) if args.show_hidden => {
            print!("{boundary}");
            io::stdout().flush().ok();
        }
        Ok(_) => {}
        Err(e) => println!("\n  [no response received: {e}]\n"),
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder helpers
// ─────────────────────────────────────────────────────────────────────────────

fn apply_model_dir(mut builder: ModelBuilder, model_path: &str) -> anyhow::Result<ModelBuilder> {
    let mut model_file = None;
    let mut tokenizer_file = None;
    for entry in Path::read_dir(Path::new(model_path))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let filename = entry.path().to_string_lossy().into_owned();
            if filename.ends_with("tokenizer.json") {
                tokenizer_file = Some(filename);
            } else if filename.ends_with(".gguf") {
                model_file = Some(filename);
            }
        }
    }
    let model_file = model_file
        .ok_or_else(|| anyhow::anyhow!("No .gguf model found in {model_path}"))?;
    let tokenizer_file = tokenizer_file
        .ok_or_else(|| anyhow::anyhow!("No tokenizer.json found in {model_path}"))?;
    println!(" Model Path: {model_file}");
    println!(" Tokenizer Path: {tokenizer_file}");
    builder = builder.model_path(model_file).tokenizer_path(tokenizer_file);
    Ok(builder)
}

fn apply_sampling_overrides(mut builder: ModelBuilder, args: &Args) -> ModelBuilder {
    if let Some(t) = args.temperature        { builder = builder.temperature(t); }
    if let Some(p) = args.top_p             { builder = builder.top_p(p); }
    if let Some(k) = args.top_k             { builder = builder.top_k(k as i32); }
    if let Some(p) = args.repeat_penalty    { builder = builder.repeat_penalty(p); }
    if let Some(p) = args.presence_penalty  { builder = builder.presence_penalty(p); }
    if let Some(s) = args.seed              { builder = builder.seed(s); }
    if args.thinking        { builder = builder.thinking(true); }
    else if args.no_thinking { builder = builder.thinking(false); }
    if let Some(ref preset) = args.sampler {
        let names = SamplingConfig::preset_names();
        if SamplingConfig::preset(preset).is_none() {
            eprintln!("Unknown sampler preset '{preset}'. Available: {}", names.join(", "));
            std::process::exit(1);
        }
        builder = builder.sampler_preset(preset);
    }
    builder
}

fn print_startup_info(args: &Args, builder: &ModelBuilder) {
    if let Some(ref preset) = args.sampler {
        println!("  Sampler: preset '{preset}'");
    }
    if builder.spec().supports_thinking {
        if args.thinking {
            println!("  Thinking: enabled (model will produce <think> blocks)");
        } else if args.no_thinking {
            println!("  Thinking: suppressed (/no_think injected, non-thinking sampling)");
        } else {
            println!("  Thinking: default (model decides)");
        }
    }
    if args.show_hidden {
        println!("  Show hidden: ON (special tokens visible in output)");
    }
    println!();
}

fn read_file_to_string(path: &str) -> anyhow::Result<String> {
    let mut text = String::new();
    File::open(path)?.read_to_string(&mut text)?;
    Ok(text)
}

fn verbosity_to_level(v: &Verbosity<InfoLevel>) -> tracing::Level {
    match v.log_level_filter() {
        clap_verbosity_flag::LevelFilter::Off | clap_verbosity_flag::LevelFilter::Error => {
            tracing::Level::ERROR
        }
        clap_verbosity_flag::LevelFilter::Warn  => tracing::Level::WARN,
        clap_verbosity_flag::LevelFilter::Info  => tracing::Level::INFO,
        clap_verbosity_flag::LevelFilter::Debug => tracing::Level::DEBUG,
        clap_verbosity_flag::LevelFilter::Trace => tracing::Level::TRACE,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Stream a submitted turn to stdout with soft word-wrap. Finishes the turn
/// on completion and returns `(response_text, boundary)`.
///
/// `line_prefix` appears at the start of the first output line.
/// `show_special` enables `decode_with_special` and the Prefill cursor rewrite.
/// `token_file` receives one token-ID line per token when `Some`.
fn stream_turn(
    conv: &mut candle_conversation::Sequence,
    handle: candle_conversation::TurnHandle,
    decoder: &TokenDecoder,
    line_prefix: &str,
    show_special: bool,
    mut token_file: Option<&mut File>,
) -> anyhow::Result<(String, String)> {
    let mut response = None;
    let mut line_tokens: Vec<u32> = Vec::new();
    let term_width = term_size::dimensions().map(|(w, _)| w).unwrap_or(80);
    let mut current_prefix = line_prefix;

    for event in handle.stream() {
        match event {
            TurnEvent::Prefill(text) => {
                if show_special {
                    print!("\x1b[1A\r\x1b[0J{text}");
                    io::stdout().flush().ok();
                }
            }
            TurnEvent::PrefillProgress { tokens_done, tokens_total } => {
                if tokens_total > 50 {
                    eprint!("\r  [prefill {tokens_done}/{tokens_total}]");
                    if tokens_done == tokens_total {
                        eprint!("\r                              \r");
                    }
                }
            }
            TurnEvent::Token(id) => {
                if let Some(tf) = &mut token_file {
                    writeln!(tf, "{id}").ok();
                }
                line_tokens.push(id);
                let decoded = decode(decoder, &line_tokens, show_special);
                if let Some(last_nl) = decoded.rfind('\n') {
                    print!("\r{current_prefix}{}", &decoded[..=last_nl]);
                    current_prefix = "";
                    line_tokens.clear();
                    if !decoded[last_nl + 1..].is_empty() {
                        print!("{}", &decoded[last_nl + 1..]);
                    }
                } else {
                    let display_width = current_prefix.len() + decoded.chars().count();
                    if display_width >= term_width {
                        line_tokens.pop();
                        let prev = decode(decoder, &line_tokens, show_special);
                        print!("\r{current_prefix}{prev}\n");
                        line_tokens.clear();
                        line_tokens.push(id);
                        current_prefix = "";
                        let fresh = decode(decoder, &line_tokens, show_special);
                        print!("{fresh}");
                    } else {
                        print!("\r{current_prefix}{decoded}");
                    }
                }
                io::stdout().flush().ok();
            }
            TurnEvent::Done(resp) => {
                response = Some(resp);
            }
            TurnEvent::Error(e) => {
                eprintln!("\n  [error: {e}]");
            }
            TurnEvent::HealthWarning(msg) => {
                eprintln!("\n\x1b[33m⚠ decode health: {msg}\x1b[0m");
            }
        }
    }

    if !line_tokens.is_empty() {
        let decoded = decode(decoder, &line_tokens, show_special);
        print!("\r{current_prefix}{decoded}");
        io::stdout().flush().ok();
    }
    println!();

    let resp = response.ok_or_else(|| anyhow::anyhow!("no response received from model"))?;
    let text = resp.text.clone();
    let boundary = conv.finish_turn(handle, &resp)?;
    Ok((text, boundary))
}

fn narrator_stream(
    conv: &mut candle_conversation::Sequence,
    msg: &str,
    decoder: &TokenDecoder,
) -> anyhow::Result<String> {
    let handle = conv.submit_turn(msg)?;
    let (text, _) = stream_turn(conv, handle, decoder, "", false, None)?;
    Ok(text)
}

#[inline]
fn decode(decoder: &TokenDecoder, tokens: &[u32], show_special: bool) -> String {
    if show_special {
        decoder.decode_with_special(tokens)
    } else {
        decoder.decode(tokens)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Args
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long)]
    pub temperature: Option<f32>,

    #[arg(long)]
    pub top_p: Option<f32>,

    #[arg(long)]
    pub top_k: Option<usize>,

    #[arg(long)]
    pub seed: Option<u64>,

    /// Protagonist name. When provided together with --persona, switches the
    /// chat into narrator mode: player input is parsed as RPE commands, the
    /// narrator narrates each beat, and the character LLM responds.
    #[arg(long)]
    pub protagonist: Option<String>,

    /// NPC persona name. Combined with --protagonist to enable narrator mode.
    #[arg(long)]
    pub persona: Option<String>,

    #[arg(long)]
    pub model_dir: Option<String>,

    #[arg(long)]
    pub repeat_penalty: Option<f32>,

    #[arg(long)]
    pub presence_penalty: Option<f32>,

    #[arg(long)]
    pub system_prompt_file: Option<String>,

    #[arg(long)]
    pub model: Option<String>,

    /// Override all sampling settings with a named preset.
    /// Available: relaxed, creative, precise, antirep.
    #[arg(long)]
    pub sampler: Option<String>,

    /// Enable thinking/reasoning mode. The model will produce <think> blocks.
    #[arg(long)]
    pub thinking: bool,

    /// Explicitly suppress thinking. Injects /no_think into the system prompt
    /// and uses non-thinking sampling parameters.
    #[arg(long, conflicts_with = "thinking")]
    pub no_thinking: bool,

    /// Show special tokens and prefill details in the console output.
    /// Useful for debugging prompt formatting and model behavior.
    #[arg(long)]
    pub show_hidden: bool,

    /// Write raw token IDs to a file for post-hoc analysis.
    /// Each token is written as a decimal ID, one per line.
    #[arg(long)]
    pub token_file: Option<String>,

    /// Write penalty information to a file during decoding.
    /// File is rewritten each decode step with current penalty state.
    /// Useful for debugging why tokens are being penalized.
    #[arg(long)]
    pub penalty_log: Option<String>,

    /// Resume from a conversation log file.
    ///
    /// All turns in the log are replayed into the KV cache at startup.
    /// The character system prompt is taken from the log unless
    /// --system-prompt-file is also given (explicit override wins).
    /// The last 4 turns are printed to console before the interactive loop.
    #[arg(long)]
    pub resume: Option<String>,

    /// Logging verbosity (-v for DEBUG, -vv for TRACE)
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

fn parse_model(name: &str) -> anyhow::Result<Model> {
    match name.to_lowercase().as_str() {
        "qwen3-8b-q4"          => Ok(Model::Qwen3_8B_Q4),
        "qwen3-8b-q6"          => Ok(Model::Qwen3_8B_Q6),
        "qwen3-14b-q4"         => Ok(Model::Qwen3_14B_Q4),
        "qwen3-14b-q5"         => Ok(Model::Qwen3_14B_Q5),
        "qwen3-14b-q6"         => Ok(Model::Qwen3_14B_Q6),
        "qwen3-30b-a3b-q4" | "qwen3-moe" => Ok(Model::Qwen3_30B_A3B_Q4),
        "qwen2-0.5b"           => Ok(Model::Qwen2_0_5B),
        "hermes3-3b-q6"        => Ok(Model::Hermes3_3B_Q6),
        "hermes3-70b-q4"       => Ok(Model::Hermes3_70B_Q4),
        _ => anyhow::bail!(
            "Unknown model '{name}'. Available: qwen3-8b-q4, qwen3-8b-q6, qwen3-14b-q4, \
             qwen3-14b-q5, qwen3-14b-q6, qwen3-30b-a3b-q4, qwen2-0.5b, hermes3-3b-q6, \
             hermes3-70b-q4"
        ),
    }
}

pub fn init_file_logging(min_level: tracing::Level) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(min_level)
        .compact()
        .init();
    Ok(())
}
