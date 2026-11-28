//! Build script: rebuild zend-tools when a tool definition it embeds changes
//! (`definitions`).

fn main() {
    embed_watch::watch("definitions");
}
