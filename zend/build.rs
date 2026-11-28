//! Build script: rebuild zend when the web UI it embeds changes
//! (`tenant::WEB`). The tool definitions are zend-tools' own, and rebuild it.

fn main() {
    embed_watch::watch("web");
}
