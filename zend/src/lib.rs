//! `zend` — the Zen Code daemon library: an OpenAI-compatible HTTP surface in
//! front of a persistent `candle_conversation` substrate, plus everything that
//! turns a workspace into that substrate's content.
//!
//! Grouped by concern: HTTP surface (`api`, `chatml`, `session`,
//! `log_broadcast`, `log_line`, `projection_event`); model bring-up
//! (`loading`, `model_choice`, `download`, `config`); workspace ingestion
//! (`ingest`, `repo_scan`, `code_read`, `raw_read`, `refresh_ctx`, `watcher`,
//! `turn_sink`); tool orchestration (`tools`, `tool_def`, `tool_summary`);
//! conversation-attached files (`conv_files`, `conv_file_store`); and shared
//! wire types (`types`, `response_section`).
pub mod api;
pub mod chatml;
pub mod code_read;
pub mod config;
pub mod conv_file_store;
pub mod conv_files;
pub mod download;
pub mod ingest;
pub mod ingest_report;
pub mod loading;
pub mod log_broadcast;
pub mod log_line;
pub mod model_choice;
pub mod projection_event;
pub mod raw_read;
pub mod refresh_ctx;
pub mod repo_scan;
pub mod response_section;
pub mod session;
pub mod tool_def;
pub mod tool_summary;
pub mod tools;
pub mod turn_sink;
pub mod types;
pub mod watcher;
