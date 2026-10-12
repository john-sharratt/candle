//! Build script: rebuild the host when the shared front-end it embeds changes
//! (`ui::COMMON`).

fn main() {
    embed_watch::watch("../web/content/common");
}
