//! Build script: rebuild npcd when the console it embeds changes
//! (`tenant::CONSOLE`).

fn main() {
    embed_watch::watch("../web/content/npcd");
}
