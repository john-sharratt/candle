//! Build script: rebuild zend when anything it embeds changes — the built-in
//! tool definitions (`tool_def`) and the web UI (`tenant::WEB`).

fn main() {
    embed_watch::watch("src/prompts/tools");
    embed_watch::watch("web");
}
