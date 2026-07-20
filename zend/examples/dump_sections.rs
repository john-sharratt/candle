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
    install_tool_catalog(&mut b)?;

    {
        for item in &b.schema().system_prompt.items {
            match item {
                SystemPromptItem::Section(s) => {
                    let kind = if s.is_template { "tmpl" } else { "sect" };
                    println!("{:?}\tsystem_prompt\t{kind}\t{}", s.id, s.name);
                }
                SystemPromptItem::Collection(c) => {
                    println!(
                        "-- collection {} ({} members) in {} --",
                        c.name,
                        c.sections.len(),
                        "system_prompt"
                    );
                    for s in &c.sections {
                        println!("{:?}\tsystem_prompt\tcoll\t{}", s.id, s.name);
                    }
                }
                SystemPromptItem::SectionTree(t) => {
                    for n in &t.nodes {
                        for o in &n.options {
                            for v in &o.variants {
                                println!("{:?}\tsystem_prompt\ttree\t{}:{}", v.id, n.name, o.id);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
