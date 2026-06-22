//! Print every registered tool's name + description (no GPU). Used to author
//! per-tool capture prompts for `gen_tool_cases`.

fn main() {
    let tools = zend_tools::registry::all_tools();
    println!("{} tools", tools.len());
    for t in tools {
        let desc = t.description.replace('\n', " ");
        println!("{}\t{}", t.name, desc);
    }
}
