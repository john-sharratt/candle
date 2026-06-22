//! Print every projection section's SectionId + layer + name, so warnings that
//! reference `SectionId(N)` can be mapped to a concrete section.

use candle_conversation::models::Dialect;
use candle_conversation::projection::{Builder, SystemPromptItem};
use zend::tools::install_tool_catalog;

const YAML: &str = include_str!("../src/prompts/projection.yaml");

fn main() -> anyhow::Result<()> {
    let dialect = Dialect::chat_ml();
    let mut b =
        Builder::from_yaml_with_vars_and_dialect(YAML, &[("workspace", "candle")], Some(&dialect))
            .map_err(|e| anyhow::anyhow!("parse: {e}"))?;
    let dlg = b
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("no dialogue layer"))?;
    install_tool_catalog(&mut b, dlg)?;

    for layer in &b.schema().layers {
        for item in &layer.system_prompt.items {
            match item {
                SystemPromptItem::Section(s) => {
                    let kind = if s.is_template { "tmpl" } else { "sect" };
                    println!("{:?}\t{}\t{kind}\t{}", s.id, layer.name, s.name);
                }
                SystemPromptItem::Collection(c) => {
                    println!(
                        "-- collection {} ({} members) in {} --",
                        c.name,
                        c.sections.len(),
                        layer.name
                    );
                    for s in &c.sections {
                        println!("{:?}\t{}\tcoll\t{}", s.id, layer.name, s.name);
                    }
                }
            }
        }
    }
    Ok(())
}
