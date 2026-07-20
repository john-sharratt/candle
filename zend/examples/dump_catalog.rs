//! Print every registered tool's name + description (no GPU). Used to author
//! per-tool capture prompts for `gen_tool_cases`.

fn main() {
    let defs = zend::tool_def::all();
    println!("{} tools", defs.len());
    for d in defs {
        let desc = d.description.replace('\n', " ");
        println!("{}\t{}", d.name, desc);
    }
}
