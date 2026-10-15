//! `zend` — the Zen Code daemon library: an OpenAI-compatible HTTP surface in
//! front of a persistent `candle_conversation` substrate, plus everything that
//! turns a workspace into that substrate's content. It runs as a tenant of the
//! `host` process (`tenant`), alone in the `zend` binary or beside npcd in
//! `motherd`.
//!
//! Grouped by concern: HTTP surface (`api`, `chatml`, `session`,
//! `projection_event`); the host contract (`tenant`); model bring-up
//! (`loading`, `config`); workspace ingestion
//! (`ingest`, `ingest_backlog`, `ingest_lease`, `ingest_report`, `ingest_worker`, `repo_scan`,
//! `code_read`, `raw_read`, `refresh_ctx`, `turn_sink`); tool
//! orchestration (`tools`, `tool_def`, `tool_summary`, `agent_exec`);
//! conversation-attached files (`conv_files`, `uploads`, `legacy_uploads`);
//! and shared wire types (`types`, `response_section`, `repo_ads`).
pub mod access;
pub mod agent_exec;
pub mod api;
pub mod branch_ingest;
pub mod chatml;
pub mod cli;
pub mod code_read;
pub mod coding_sampling;
pub mod config;
pub mod conv_branches;
pub mod conv_files;
pub mod conv_order;
pub mod conv_overlay;
pub mod fast_path;
pub mod ingest;
pub mod ingest_backlog;
pub mod ingest_cancel;
pub mod ingest_lease;
pub mod ingest_report;
pub mod ingest_worker;
pub mod legacy_uploads;
pub mod lenient_json;
pub mod live_turn;
pub mod loading;
pub mod openai_tools;
pub mod passthrough;
pub mod projection_event;
pub mod raw_read;
pub mod reasoning_split;
pub mod refresh_ctx;
pub mod repeat_guard;
pub mod repo_ads;
pub mod repo_path;
pub mod repo_scan;
pub mod response_section;
pub mod resume;
pub mod retrieval_scope;
pub mod schema_layers;
pub mod secrets;
pub mod self_check;
pub mod session;
pub mod substrate_git;
pub mod tenant;
pub mod think_budget;
pub mod think_gate;
pub mod think_progress;
pub mod tool_call_split;
pub mod tool_def;
pub mod tool_guidance;
pub mod tool_round;
pub mod tool_summary;
pub mod tools;
pub mod turn_sink;
pub mod types;
pub mod uploads;
pub mod working_set;
pub mod workspace;
