//! Sequence tree generator — life-timeline pipeline.
//!
//! Reads a Markdown life-timeline (T-N format), drives three stages to plan
//! and narrate each day, then feeds the narrated events to a character LLM
//! and logs the character's responses.
//!
//! # Pipeline
//!
//! ```text
//! 0. guide_summarize  (upfront, once per period in selection)
//!    ⇒ background (life story + cast)
//!    ⇒ appended to character system prompt + used by director
//!
//! Per day:
//! 1. guide_today     (once per day)
//!    System: GUIDE_TODAY_PROMPT + background
//!    User:   DATE, DESCRIPTION, YESTERDAY, LAST_MONTH
//!    ⇒ Vec<Waypoint>
//!
//! 2. For each waypoint:
//!    a. director      (single-shot per waypoint, fresh each time)
//!       System: DIRECTOR_PROMPT + background
//!       User:   LAST RESPONSE, WAYPOINT
//!       ⇒ scene narration (2–6 sentences, 2nd person present)
//!
//!    b. character_conv (stateful across whole run)
//!       User: scene narration
//!       ⇒ character's response
//!
//!    c. LogRecord::Turn
//! ```
//!
//! # KV cache reuse
//!
//! | Sequence       | Scope      | Rationale                                   |
//! |--------------------|------------|---------------------------------------------|
//! | `character_conv`   | Whole run  | Accumulates the character's lived history   |
//! | `guide_summarize`  | Upfront    | Runs once per period before main loop       |
//! | `guide_today`      | Whole run  | System includes background from summarize   |
//! | `director`         | Per beat   | Single-shot; fresh conversation each time   |
//!
//! # Input format
//!
//! Markdown file with `# Period Name` headers and `[T-N] description` entries.
//! T-0 = 2026-02-24. Each T-N entry = one story day.
//!
//! # Run
//!
//! ```bash
//! cargo run --example tree_gen -p candle-conversation --release \
//!   --features hub -- \
//!   --input candle-conversation/src/characters/bramble-timeline.md \
//!   --output bramble_tree.yaml \
//!   --days 3 \
//!   --skip-entries 360
//! ```

use candle::Device;
use candle_conversation::{
    conversation_log::{now_iso, LogRecord, LogWriter},
    models::{Model, ModelBuilder},
    prompts::{DIRECTOR_PROMPT, GUIDE_SUMMARIZE_PERIOD_PROMPT, GUIDE_TODAY_PROMPT},
    think_strip::strip_think_blocks,
    DecodeHealthConfig, SamplingConfig, TokenDecoder, TurnEvent, TurnHandle, TurnResponse,
};
use clap::Parser;
use clap_verbosity_flag::*;
use std::{
    fs::File,
    io::{self, Read, Write},
};

//  Main

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let min_level = match args.verbose.log_level_filter() {
        clap_verbosity_flag::LevelFilter::Off => tracing::Level::ERROR,
        clap_verbosity_flag::LevelFilter::Error => tracing::Level::ERROR,
        clap_verbosity_flag::LevelFilter::Warn => tracing::Level::WARN,
        clap_verbosity_flag::LevelFilter::Info => tracing::Level::INFO,
        clap_verbosity_flag::LevelFilter::Debug => tracing::Level::DEBUG,
        clap_verbosity_flag::LevelFilter::Trace => tracing::Level::TRACE,
    };
    init_tracing(min_level)?;

    println!();
    println!("  \x1b[1mcandle-conversation \x1b[0m Life Timeline Tree Generator");
    println!();

    //  Parse timeline
    let timeline_text = {
        let mut t = String::new();
        File::open(&args.input)?.read_to_string(&mut t)?;
        t
    };
    let periods = parse_timeline(&timeline_text);
    let total_entries: usize = periods.iter().map(|p| p.entries.len()).sum();
    println!(
        "  Timeline:      {} ({} periods, {} entries)",
        args.input,
        periods.len(),
        total_entries
    );

    //  Build a flat ordered entry list
    // Each item: (period_name, t_n, description)
    let flat: Vec<(String, u32, String)> = periods
        .iter()
        .flat_map(|p| {
            p.entries
                .iter()
                .map(move |e| (p.name.clone(), e.t_n, e.description.clone()))
        })
        .collect();

    let skip = args.skip_entries.min(flat.len());
    let take = args.days.max(1).min(3);
    let selected: Vec<&(String, u32, String)> = flat.iter().skip(skip).take(take).collect();

    if selected.is_empty() {
        anyhow::bail!("No entries to process after skipping {} entries.", skip);
    }

    println!("  Skip entries:  {}", skip);
    println!("  Days to run:   {} (max 3 enforced)", selected.len());
    println!(
        "  Entry range:   [T-{}]  [T-{}]",
        selected.first().unwrap().1,
        selected.last().unwrap().1
    );
    println!("  Output:        {}", args.output);
    println!();

    //  Build engine
    let mut builder = if args.model.is_none() && args.model_dir.is_some() {
        // Auto-detect from the GGUF file in the directory.
        ModelBuilder::from_gguf_dir(args.model_dir.as_ref().unwrap())?
            .max_response_tokens(args.max_tokens)
            .compression_level(0)
    } else {
        let model = if let Some(ref name) = args.model {
            parse_model(name)?
        } else {
            Model::Qwen3_14B_Q4
        };
        let mut b = model
            .builder()
            .max_response_tokens(args.max_tokens)
            .compression_level(0);
        if let Some(ref model_path) = args.model_dir {
            b = b.model_dir(model_path);
        }
        b
    };

    if let Some(t) = args.temperature {
        builder = builder.temperature(t);
    }
    if let Some(p) = args.top_p {
        builder = builder.top_p(p);
    }
    if let Some(k) = args.top_k {
        builder = builder.top_k(k as i32);
    }
    if let Some(p) = args.repeat_penalty {
        builder = builder.repeat_penalty(p);
    }
    if let Some(p) = args.presence_penalty {
        builder = builder.presence_penalty(p);
    }
    if let Some(s) = args.seed {
        builder = builder.seed(s);
    }
    // Default to no-think; only enable thinking if explicitly requested.
    if args.thinking {
        builder = builder.thinking(true);
    } else {
        builder = builder.thinking(false);
    }
    if let Some(ref preset) = args.sampler {
        let names = SamplingConfig::preset_names();
        if SamplingConfig::preset(preset).is_none() {
            anyhow::bail!(
                "Unknown sampler preset '{}'. Available: {}",
                preset,
                names.join(", ")
            );
        }
        builder = builder.sampler_preset(preset);
    }
    builder = builder.health(DecodeHealthConfig::for_chat());

    let device = Device::cuda_if_available(0)?;
    println!("  Device: {:?}", device);

    let start = std::time::Instant::now();
    let engine = builder.engine(&device)?;
    let decoder = engine.token_decoder();
    let conv_config = builder.conversation_config();
    println!("  Model:  {}", builder);
    println!("  Loaded in {:.2}s\n", start.elapsed().as_secs_f64());

    //  Character system prompt
    let character_system_prompt = if let Some(ref text) = args.character_prompt {
        text.clone()
    } else if let Some(ref path) = args.character_prompt_file {
        let mut text = String::new();
        File::open(path)?.read_to_string(&mut text)?;
        text.trim().to_string()
    } else {
        builder.format_system_prompt()
    };

    //  Create guide_summarize spine (accumulates timeline entries)
    let mut guide_summarize =
        engine.new_conversation(GUIDE_SUMMARIZE_PERIOD_PROMPT, conv_config.clone())?;
    // initial_handle() removed: system seeding is now lazy via ensure_system_ingested
    println!("  KV primed: guide_summarize\n");

    let term_width = term_size::dimensions().map(|(w, _)| w).unwrap_or(80);

    //  Upfront: summarize all relevant periods (prefill spine + fork to decode)
    //
    // The base conversation ("spine") accumulates all timeline entries via
    // insert_turn (prefill-only, no decode).  At each period boundary we
    // fork the spine — the fork shares all accumulated KV via CoW — and
    // decode a summary on the fork.  Each period's entries are prefilled
    // exactly once on the spine; forks are cheap.
    let last_selected_period = {
        let mut lp = String::new();
        for (period_name, _, _) in &selected {
            lp = period_name.clone();
        }
        lp
    };
    let periods_to_summarize: Vec<&str> = periods
        .iter()
        .map(|p| p.name.as_str())
        .take_while(|name| *name != last_selected_period)
        .chain(std::iter::once(last_selected_period.as_str()))
        .collect();

    // The first selected day is the "present" — the summarizer should only
    // see events that happened *before* it (higher t_n = further in the past).
    let current_t_n = selected.first().unwrap().1;

    let mut background = String::new();
    for period_name in &periods_to_summarize {
        // Check for a cached summary embedded in the timeline file.
        let cached = periods
            .iter()
            .find(|p| p.name == *period_name)
            .and_then(|p| p.cached_summary.clone());

        if let Some(ref summary) = cached {
            println!(
                "  \x1b[35m[summarize]\x1b[0m Period: {} \x1b[32m(cached)\x1b[0m",
                period_name
            );
            background = summary.clone();

            // Still prefill the entries onto the spine so subsequent
            // non-cached forks have the full timeline context.
            let period_entries: Vec<String> = periods
                .iter()
                .find(|p| p.name == *period_name)
                .map(|p| {
                    p.entries
                        .iter()
                        .filter(|e| e.t_n > current_t_n)
                        .map(|e| format!("[T-{}] {}", e.t_n, e.description))
                        .collect()
                })
                .unwrap_or_default();
            let entries_text = format!(
                "PERIOD: {}\n\nTIMELINE ENTRIES:\n{}",
                period_name,
                period_entries.join("\n")
            );
            guide_summarize.insert_turn(&entries_text, "Noted.")?;
            continue;
        }

        println!("  \x1b[35m[summarize]\x1b[0m Period: {}", period_name);

        let period_entries: Vec<String> = periods
            .iter()
            .find(|p| p.name == *period_name)
            .map(|p| {
                p.entries
                    .iter()
                    .filter(|e| e.t_n > current_t_n)
                    .map(|e| format!("[T-{}] {}", e.t_n, e.description))
                    .collect()
            })
            .unwrap_or_default();

        // Prefill this period's entries onto the spine (no decode).
        let entries_text = format!(
            "PERIOD: {}\n\nTIMELINE ENTRIES:\n{}",
            period_name,
            period_entries.join("\n")
        );
        guide_summarize.insert_turn(&entries_text, "Noted.")?;

        // Fork from the spine — shares all accumulated KV via CoW.
        let mut fork = guide_summarize.fork()?;

        // On the fork: ask for a fresh summary of everything up to this period.
        let summarize_msg = format!(
            "Now produce the Life Story and Cast summary for all periods up to and including \"{}\".",
            period_name
        );

        print!("  \x1b[35m[summarize]\x1b[0m ");
        io::stdout().flush()?;

        let sum_handle = fork.submit_turn(&summarize_msg)?;
        let sum_resp = stream_response(&sum_handle, &decoder, term_width, "    ")?;
        fork.finish_turn(sum_handle, &sum_resp)?;

        // Re-decode from raw token IDs to preserve newlines.
        // TurnResponse.text has already been through strip_think_blocks in the
        // scheduler, which collapses all whitespace.  For the file write-back
        // we need the original line structure.
        let raw_text = decoder.decode(&sum_resp.token_ids);
        background = strip_think_preserve_newlines(&raw_text);
        fork.close()?;

        // Write this summary back immediately so it's saved even if
        // a later period crashes.
        match write_summaries_to_timeline(
            &args.input,
            &[(period_name.to_string(), background.clone())],
        ) {
            Ok(()) => println!("  \x1b[32m[cached]\x1b[0m Wrote summary to {}", args.input,),
            Err(e) => eprintln!(
                "  \x1b[33m[warn]\x1b[0m Could not write summary to {}: {}",
                args.input, e,
            ),
        }
        println!();
    }
    guide_summarize.close()?;
    println!();

    //  Build enriched character prompt
    //
    // The background (life story + cast) is appended to the character's system
    // prompt so the character knows their own history and the people in it.
    // Scene rules live in the character prompt file itself.
    let enriched_character_prompt = if background.is_empty() {
        character_system_prompt.clone()
    } else {
        format!("{}\n\n---\n\n{}", character_system_prompt, background)
    };

    //  Build director system prompt
    let director_system = if background.is_empty() {
        DIRECTOR_PROMPT.to_string()
    } else {
        format!("{}\n\n---\n\nBACKGROUND:\n{}", DIRECTOR_PROMPT, background)
    };

    //  Sequence configs
    //
    // Director is capped at 512 tokens — it produces 2–4 sentences of scene
    // setup, so the full 4096-token default is wasteful.  Character and other
    // conversations use the default.
    let mut director_config = conv_config.clone();
    director_config.max_response_tokens = 512;

    //  Create long-lived conversations
    //
    // character_conv uses the enriched prompt (original + background).
    // guide_today gets background so it knows who to plan for.
    // director is single-shot per waypoint (created in the loop).

    let mut character_conv =
        engine.new_conversation(&enriched_character_prompt, conv_config.clone())?;

    //  Task observer — streams tree-internal summarization to terminal
    //
    // The conversation tree fires auto-summarization every N turns. Those
    // tasks run inside finish_turn() (blocking). The observer channel
    // forwards Token/Prefill/PrefillProgress events so we can print them
    // in real time from a dedicated thread.
    let (task_obs_tx, task_obs_rx) = crossbeam::channel::unbounded::<TurnEvent>();
    character_conv.set_task_observer(Some(task_obs_tx));

    let obs_decoder = decoder.clone();
    let obs_thread = {
        let term_w = term_width;
        std::thread::spawn(move || {
            let mut line_buf: Vec<u32> = Vec::new();
            let indent = "    ";
            let mut header_printed = false;

            for event in task_obs_rx.iter() {
                match event {
                    TurnEvent::Prefill(_) => {}
                    TurnEvent::PrefillProgress {
                        tokens_done,
                        tokens_total,
                    } => {
                        if !header_printed {
                            println!();
                            println!("  \x1b[35m[tree summarize]\x1b[0m");
                            print!("{}", indent);
                            header_printed = true;
                        }
                        if tokens_total > 50 {
                            eprint!("\r{}[prefill {}/{}]   ", indent, tokens_done, tokens_total);
                            if tokens_done == tokens_total {
                                eprint!("\r{}                              \r", indent);
                            }
                        }
                    }
                    TurnEvent::Token(id) => {
                        if !header_printed {
                            println!();
                            println!("  \x1b[35m[tree summarize]\x1b[0m");
                            print!("{}", indent);
                            header_printed = true;
                        }
                        line_buf.push(id);
                        let text = obs_decoder.decode(&line_buf);
                        if let Some(nl) = text.rfind('\n') {
                            print!("\r{}{}\n", indent, &text[..nl]);
                            line_buf.clear();
                            let after = &text[nl + 1..];
                            if !after.is_empty() {
                                print!("{}{}", indent, after);
                            }
                        } else {
                            let display_len = indent.len() + text.chars().count();
                            if display_len >= term_w {
                                line_buf.pop();
                                let prev = obs_decoder.decode(&line_buf);
                                print!("\r{}{}\n", indent, prev);
                                line_buf.clear();
                                line_buf.push(id);
                                print!("{}{}", indent, obs_decoder.decode(&line_buf));
                            } else {
                                print!("\r{}{}", indent, text);
                            }
                        }
                        io::stdout().flush().ok();
                    }
                    TurnEvent::HealthWarning(msg) => {
                        println!("\n  \x1b[33m[tree summarize health: {}]\x1b[0m", msg);
                    }
                    _ => {}
                }
            }
            // Flush any remaining partial line.
            if !line_buf.is_empty() {
                print!("\r{}{}", indent, obs_decoder.decode(&line_buf));
                io::stdout().flush().ok();
            }
            if header_printed {
                println!();
            }
        })
    };

    let today_system = format!(
        "{}\n\n---\n\nBACKGROUND:\n{}",
        GUIDE_TODAY_PROMPT, background
    );
    let mut guide_today_conv = engine.new_conversation(&today_system, conv_config.clone())?;

    // initial_handle() removed
    // initial_handle() removed
    println!("  KV primed: character, guide_today\n");

    //  Open redo log
    let mut log = LogWriter::create(&args.output)?;
    let gen_start = std::time::Instant::now();

    log.append(&LogRecord::Header {
        character_system_prompt: enriched_character_prompt.clone(),
        guide_system_prompt: format!(
            "tree_gen pipeline v3 | stages: summarize+today+director | input: {}",
            args.input
        ),
        started_at: now_iso(),
    })?;

    //  Pipeline state
    let mut yesterday_summary = String::new();
    let mut last_month_summary = String::new();
    let mut total_turns = 0usize;

    //  Main loop
    for (period_name, t_n, description) in &selected {
        let date_str = t_n_to_date(*t_n);

        println!();
        println!("  \x1b[90m{}\x1b[0m", "=".repeat(68));
        println!(
            "  \x1b[1mDay: T-{}\x1b[0m  {}  \x1b[90m[{}]\x1b[0m",
            t_n, date_str, period_name
        );
        println!("  {}", description);
        println!("  \x1b[90m{}\x1b[0m", "=".repeat(68));
        println!();

        //  Stage 2: Plan the day
        println!("  \x1b[34m[plan]\x1b[0m Generating waypoints...");

        let today_msg = format!(
            "DATE: {}\n\nDESCRIPTION: {}\n\nYESTERDAY: {}\n\nLAST MONTH: {}",
            date_str,
            description,
            if yesterday_summary.is_empty() {
                "(none  first day of selection)".to_string()
            } else {
                yesterday_summary.clone()
            },
            if last_month_summary.is_empty() {
                "(none  insufficient context)".to_string()
            } else {
                last_month_summary.clone()
            },
        );

        let today_resp = guide_today_conv.send_turn(&today_msg)?;
        let waypoints_text = strip_think_blocks(&today_resp.text);
        let waypoints = parse_waypoints(&waypoints_text);

        println!();
        println!("  \x1b[34mWaypoints ({}):\x1b[0m", waypoints.len());
        for (i, wp) in waypoints.iter().enumerate() {
            println!("    \x1b[90m{:>2}.\x1b[0m {}", i + 1, wp);
        }
        println!();

        if waypoints.is_empty() {
            println!("  \x1b[90m(no waypoints produced, skipping day)\x1b[0m\n");
            continue;
        }

        //  Waypoint loop — director + character

        let mut last_char_response = String::new();
        let mut prev_director_msg = String::new();
        let mut prev_director_scene = String::new();
        let wp_count = waypoints.len();

        for (wi, waypoint) in waypoints.iter().enumerate() {
            println!("  \x1b[90m--- waypoint {}/{} ---\x1b[0m", wi + 1, wp_count);
            println!("  \x1b[90m{}\x1b[0m", waypoint);
            println!();

            //  Director (single-shot with 1-turn history)
            let director_msg = format!(
                "LAST RESPONSE:\n{}\n\nWAYPOINT:\n{}",
                if last_char_response.is_empty() {
                    "(first beat)"
                } else {
                    &last_char_response
                },
                waypoint,
            );

            let mut director_conv =
                engine.new_conversation(&director_system, director_config.clone())?;
            // initial_handle() removed

            // Insert previous director exchange as conversation history so
            // the model sees its own second-person output in the assistant
            // role, anchoring its voice against the first-person LAST RESPONSE.
            if !prev_director_scene.is_empty() {
                director_conv.insert_turn(&prev_director_msg, &prev_director_scene)?;
            }

            println!("  \x1b[36m[director]\x1b[0m");
            print!("    ");
            io::stdout().flush()?;

            let dir_handle = director_conv.submit_turn(&director_msg)?;
            let dir_resp = stream_response(&dir_handle, &decoder, term_width, "    ")?;
            director_conv.finish_turn(dir_handle, &dir_resp)?;

            let scene_text = strip_think_blocks(&dir_resp.text);
            println!();

            // Save this exchange for next waypoint's history insert.
            prev_director_msg = director_msg;
            prev_director_scene = scene_text.clone();

            director_conv.close()?;

            if scene_text.trim().is_empty() {
                println!("    \x1b[90m(empty scene, skipping)\x1b[0m\n");
                continue;
            }

            //  Character responds
            println!("  \x1b[1m[character]\x1b[0m");
            print!("    ");
            io::stdout().flush()?;

            let char_handle = match character_conv.submit_turn(scene_text.trim()) {
                Ok(h) => h,
                Err(e) => {
                    println!("\n  [error submitting to character: {e}]");
                    break;
                }
            };
            let char_resp = stream_response(&char_handle, &decoder, term_width, "    ")?;
            character_conv.finish_turn(char_handle, &char_resp)?;

            last_char_response = strip_think_blocks(&char_resp.text);
            total_turns += 1;
            println!();

            println!(
                "    \x1b[90m({} director tok, {} character tok)\x1b[0m\n",
                dir_resp.token_ids.len(),
                char_resp.token_ids.len(),
            );

            log.append(&LogRecord::Turn {
                seq: total_turns,
                guide_message: scene_text.trim().to_string(),
                character_response: last_char_response.clone(),
                character_token_count: char_resp.token_ids.len(),
            })?;
        }

        // Advance rolling context.
        yesterday_summary = format!(
            "T-{} ({}): {}  {}",
            t_n,
            date_str,
            description,
            if last_char_response.is_empty() {
                "(no response)".to_string()
            } else {
                truncate_to_chars(&last_char_response, 200)
            }
        );
        last_month_summary = yesterday_summary.clone();
    }

    println!();
    println!("  \x1b[90m{}\x1b[0m", "=".repeat(68));
    println!(
        "  \x1b[1mDone.\x1b[0m {} turns, {:.1}s",
        total_turns,
        gen_start.elapsed().as_secs_f64()
    );
    println!("  \x1b[90m{}\x1b[0m", "=".repeat(68));
    println!();

    log.append(&LogRecord::Done {
        total_turns,
        elapsed_secs: gen_start.elapsed().as_secs_f64(),
    })?;
    println!("  Log: {}\n", args.output);

    guide_today_conv.close()?;
    // Drop the observer sender so the observer thread exits, then join.
    character_conv.set_task_observer(None);
    character_conv.close()?;
    let _ = obs_thread.join();

    Ok(())
}

//  Timeline parser

struct Period {
    name: String,
    entries: Vec<Entry>,
    /// Pre-baked summary embedded in the timeline via `<!-- summary ... -->`.
    /// When present, the summarizer skips this period entirely.
    cached_summary: Option<String>,
}
struct Entry {
    t_n: u32,
    description: String,
}

fn parse_timeline(text: &str) -> Vec<Period> {
    let mut periods: Vec<Period> = Vec::new();
    let mut current: Option<Period> = None;
    let mut in_summary = false;
    let mut summary_buf = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Detect start of a summary block: <!-- summary
        if !in_summary && trimmed == "<!-- summary" {
            in_summary = true;
            summary_buf.clear();
            continue;
        }

        // Detect end of a summary block: -->
        if in_summary {
            if trimmed == "-->" {
                in_summary = false;
                let summary = summary_buf.trim().to_string();
                if let Some(ref mut p) = current {
                    if !summary.is_empty() {
                        p.cached_summary = Some(summary);
                    }
                }
                summary_buf.clear();
            } else {
                if !summary_buf.is_empty() {
                    summary_buf.push('\n');
                }
                summary_buf.push_str(line);
            }
            continue;
        }

        if let Some(name) = trimmed.strip_prefix("# ") {
            if let Some(p) = current.take() {
                periods.push(p);
            }
            current = Some(Period {
                name: name.trim().to_string(),
                entries: Vec::new(),
                cached_summary: None,
            });
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("[T-") {
            if let Some(end) = rest.find(']') {
                if let Ok(t_n) = rest[..end].parse::<u32>() {
                    let desc = rest[end + 1..].trim().to_string();
                    if !desc.is_empty() {
                        if let Some(ref mut p) = current {
                            p.entries.push(Entry {
                                t_n,
                                description: desc,
                            });
                        }
                    }
                }
            }
        }
    }

    if let Some(p) = current {
        periods.push(p);
    }
    periods
}

/// Strip `<think>…</think>` blocks but preserve newlines in the remaining text.
///
/// Unlike `strip_think_blocks` (which collapses all whitespace to single
/// spaces), this keeps the line structure intact so summaries read correctly
/// when written back into the timeline file.
fn strip_think_preserve_newlines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let lower = text.to_ascii_lowercase();
    let mut pos = 0;

    loop {
        let remaining_lower = &lower[pos..];
        let Some(open) = remaining_lower.find("<think>") else {
            out.push_str(&text[pos..]);
            break;
        };
        out.push_str(&text[pos..pos + open]);
        let after_open = pos + open + "<think>".len();

        let remaining_lower2 = &lower[after_open..];
        match remaining_lower2.find("</think>") {
            Some(close) => {
                pos = after_open + close + "</think>".len();
            }
            None => {
                // Unterminated: strip to end
                break;
            }
        }
    }

    // Also remove stray </think> tags
    let result = out.replace("</think>", "").replace("</Think>", "");
    result.trim().to_string()
}

/// Write generated summaries back into the timeline file.
///
/// For each (period_name, summary_text) pair, inserts a `<!-- summary ... -->`
/// block right after the `# Period` heading (before any entries).
/// Existing summary blocks for a period are replaced.
fn write_summaries_to_timeline(
    timeline_path: &str,
    summaries: &[(String, String)],
) -> anyhow::Result<()> {
    if summaries.is_empty() {
        return Ok(());
    }

    let text = {
        let mut t = String::new();
        File::open(timeline_path)?.read_to_string(&mut t)?;
        t
    };

    let lookup: std::collections::HashMap<&str, &str> = summaries
        .iter()
        .map(|(name, summary)| (name.as_str(), summary.as_str()))
        .collect();

    let mut output = String::with_capacity(text.len() + summaries.len() * 1024);
    let mut current_period: Option<&str> = None;
    let mut in_summary = false;
    let mut need_insert = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip existing summary blocks for periods we're about to re-insert
        if !in_summary && trimmed == "<!-- summary" {
            if let Some(period) = current_period {
                if lookup.contains_key(period) {
                    in_summary = true;
                    continue;
                }
            }
        }
        if in_summary {
            if trimmed == "-->" {
                in_summary = false;
            }
            continue;
        }

        // Period heading — write it, then maybe insert a summary after
        if let Some(name) = trimmed.strip_prefix("# ") {
            // If we were about to insert for a prior period, do it now
            // (shouldn't happen, but safety)
            if need_insert {
                if let Some(prev) = current_period {
                    if let Some(summary) = lookup.get(prev) {
                        output.push_str("<!-- summary\n");
                        output.push_str(summary);
                        output.push_str("\n-->\n\n");
                    }
                }
                need_insert = false;
            }

            let name = name.trim();
            current_period = Some(
                // We need a &str with the same lifetime as `text`
                // Find the matching summary key
                lookup.keys().find(|&&k| k == name).copied().unwrap_or(name),
            );
            output.push_str(line);
            output.push('\n');

            if lookup.contains_key(name) {
                need_insert = true;
            }
            continue;
        }

        // First non-blank, non-summary line after a heading → insert before it
        if need_insert && !trimmed.is_empty() {
            if let Some(period) = current_period {
                if let Some(summary) = lookup.get(period) {
                    output.push_str("<!-- summary\n");
                    output.push_str(summary);
                    output.push_str("\n-->\n\n");
                }
            }
            need_insert = false;
        }

        output.push_str(line);
        output.push('\n');
    }

    // Trailing insert (last period, no entries after heading)
    if need_insert {
        if let Some(period) = current_period {
            if let Some(summary) = lookup.get(period) {
                output.push_str("<!-- summary\n");
                output.push_str(summary);
                output.push_str("\n-->\n");
            }
        }
    }

    // Write atomically: write to .tmp then rename
    let tmp_path = format!("{}.tmp", timeline_path);
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(output.as_bytes())?;
        f.flush()?;
    }
    std::fs::rename(&tmp_path, timeline_path)?;

    Ok(())
}

fn parse_waypoints(text: &str) -> Vec<String> {
    // First: try splitting on line boundaries (ideal case — model obeyed the prompt).
    let by_line: Vec<String> = text
        .lines()
        .filter_map(|line| strip_numbered_prefix(line.trim()))
        .collect();

    if by_line.len() >= 2 {
        return by_line;
    }

    // Fallback: the model may have placed all waypoints on a single line.
    // Split on inline " N. " boundaries (e.g. "...fog. 2. The front door...").
    let inline = split_inline_numbered(text);
    if inline.len() >= 2 {
        return inline;
    }

    // Last resort: non-empty lines.
    text.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Strip a leading `N. ` or `N) ` prefix from a line, returning the text after it.
fn strip_numbered_prefix(line: &str) -> Option<String> {
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let rest = &line[digits..];
    if let Some(s) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

/// Split text containing inline numbered items like "1. Foo bar. 2. Baz qux. 3. ..."
fn split_inline_numbered(text: &str) -> Vec<String> {
    // Build a flat string from all lines.
    let flat: String = text.lines().map(|l| l.trim()).collect::<Vec<_>>().join(" ");

    // Find positions of " N. " or start-of-string "N. " boundaries.
    let mut positions: Vec<usize> = Vec::new();
    let bytes = flat.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // Check for digit sequence followed by ". "
        let start = i;
        if (start == 0 || bytes[start - 1] == b' ') && bytes[start].is_ascii_digit() {
            let mut j = start;
            while j < len && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j < len - 1 && bytes[j] == b'.' && bytes[j + 1] == b' ' {
                positions.push(start);
                i = j + 2;
                continue;
            }
        }
        i += 1;
    }

    if positions.len() < 2 {
        return Vec::new();
    }

    let mut result = Vec::new();
    for (idx, &pos) in positions.iter().enumerate() {
        // Skip past the "N. " prefix.
        let after_prefix = {
            let mut p = pos;
            while p < len && bytes[p].is_ascii_digit() {
                p += 1;
            }
            p + 2 // skip ". "
        };
        let end = if idx + 1 < positions.len() {
            // Trim trailing space before next number.
            let e = positions[idx + 1];
            if e > 0 && bytes[e - 1] == b' ' {
                e - 1
            } else {
                e
            }
        } else {
            len
        };
        if after_prefix < end {
            let s = flat[after_prefix..end].trim().to_string();
            if !s.is_empty() {
                result.push(s);
            }
        }
    }
    result
}

//  Date arithmetic

fn t_n_to_date(n: u32) -> String {
    // T-0 = 2026-02-24. Subtract n days.
    let mut day = 24u32;
    let mut month = 2u32;
    let mut year = 2026i32;
    let mut rem = n as i32;

    while rem > 0 {
        if rem < day as i32 {
            day -= rem as u32;
            rem = 0;
        } else {
            rem -= day as i32;
            if month == 1 {
                month = 12;
                year -= 1;
            } else {
                month -= 1;
            }
            day = days_in_month(year, month);
        }
    }

    format!("{:04}-{:02}-{:02}", year, month, day)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

//  Streaming

fn stream_response(
    handle: &TurnHandle,
    decoder: &TokenDecoder,
    term_width: usize,
    indent: &str,
) -> anyhow::Result<TurnResponse> {
    let mut line_buf: Vec<u32> = Vec::new();
    let mut line_prefix = "";
    let mut response: Option<TurnResponse> = None;

    for event in handle.stream() {
        match event {
            TurnEvent::Prefill(_) => {}
            TurnEvent::PrefillProgress {
                tokens_done,
                tokens_total,
            } => {
                if tokens_total > 50 {
                    eprint!("\r{}[prefill {}/{}]   ", indent, tokens_done, tokens_total);
                    if tokens_done == tokens_total {
                        eprint!("\r{}                              \r", indent);
                    }
                }
            }
            TurnEvent::Token(id) => {
                line_buf.push(id);
                let text = decoder.decode(&line_buf);

                if let Some(nl) = text.rfind('\n') {
                    print!("\r{}{}{}", indent, line_prefix, &text[..=nl]);
                    let after = &text[nl + 1..];
                    line_prefix = "";
                    line_buf.clear();
                    if !after.is_empty() {
                        print!("{}{}", indent, after);
                    }
                } else {
                    let display_len = indent.len() + line_prefix.len() + text.chars().count();
                    if display_len >= term_width {
                        line_buf.pop();
                        let prev = decoder.decode(&line_buf);
                        print!("\r{}{}{}\n", indent, line_prefix, prev);
                        line_buf.clear();
                        line_buf.push(id);
                        line_prefix = "";
                        print!("{}{}", indent, decoder.decode(&line_buf));
                    } else {
                        print!("\r{}{}{}", indent, line_prefix, text);
                    }
                }
                io::stdout().flush().ok();
            }
            TurnEvent::Done(resp) => {
                response = Some(resp);
            }
            TurnEvent::Error(e) => {
                println!("\n{}[error: {}]", indent, e);
            }
            TurnEvent::HealthWarning(msg) => {
                println!("\n\x1b[33m{} decode health: {}\x1b[0m", indent, msg);
            }
        }
    }

    if !line_buf.is_empty() {
        print!("\r{}{}", indent, decoder.decode(&line_buf));
        io::stdout().flush().ok();
    }

    response.ok_or_else(|| anyhow::anyhow!("scheduler closed without Done event"))
}

//  Misc

fn truncate_to_chars(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}", s.chars().take(n).collect::<String>())
    }
}

//  CLI

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Life-timeline tree generator: 4-stage guide pipeline for character generation"
)]
struct Args {
    /// Path to the life timeline Markdown file (T-N format).
    #[arg(
        long,
        short = 'i',
        default_value = "candle-conversation/src/characters/bramble-timeline.md"
    )]
    input: String,

    /// Output YAML redo log path.
    #[arg(long, short = 'o', default_value = "bramble_tree.yaml")]
    output: String,

    /// Number of story days to process (max 3).
    #[arg(long, short = 'd', default_value_t = 3)]
    days: usize,

    /// Skip this many timeline entries before selecting days.
    /// Use to jump to interesting sections. E.g. --skip-entries 360  Northern Campaigns.
    #[arg(long, default_value_t = 0)]
    skip_entries: usize,

    /// Inline character system prompt.
    #[arg(long)]
    character_prompt: Option<String>,

    /// Path to a file containing the character system prompt.
    #[arg(long)]
    character_prompt_file: Option<String>,

    /// Named model preset (e.g. qwen3-14b-q4).
    #[arg(long)]
    model: Option<String>,

    /// Directory containing a .gguf model file and tokenizer.json.
    #[arg(long)]
    model_dir: Option<String>,

    // Sampling
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    repeat_penalty: Option<f32>,
    #[arg(long)]
    presence_penalty: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    sampler: Option<String>,

    /// Enable thinking/reasoning mode (off by default).
    #[arg(long)]
    thinking: bool,

    /// Max tokens per response (applies to all conversations).
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    #[command(flatten)]
    verbose: Verbosity<InfoLevel>,
}

//  Model lookup

fn parse_model(name: &str) -> anyhow::Result<Model> {
    match name.to_lowercase().as_str() {
        "qwen3-8b-q4" => Ok(Model::Qwen3_8B_Q4),
        "qwen3-8b-q6" => Ok(Model::Qwen3_8B_Q6),
        "qwen3-14b-q4" => Ok(Model::Qwen3_14B_Q4),
        "qwen3-14b-q5" => Ok(Model::Qwen3_14B_Q5),
        "qwen3-14b-q6" => Ok(Model::Qwen3_14B_Q6),
        "qwen3-30b-a3b-q4" | "qwen3-moe" => Ok(Model::Qwen3_30B_A3B_Q4),
        "qwen2-0.5b" => Ok(Model::Qwen2_0_5B),
        "hermes3-3b-q6" => Ok(Model::Hermes3_3B_Q6),
        "hermes3-70b-q4" => Ok(Model::Hermes3_70B_Q4),
        _ => anyhow::bail!(
            "Unknown model '{}'. Known: qwen3-8b-q4, qwen3-8b-q6, qwen3-14b-q4, \
             qwen3-14b-q5, qwen3-14b-q6, qwen3-30b-a3b-q4, qwen2-0.5b, \
             hermes3-3b-q6, hermes3-70b-q4",
            name
        ),
    }
}

fn init_tracing(min_level: tracing::Level) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(min_level)
        .compact()
        .init();
    Ok(())
}
