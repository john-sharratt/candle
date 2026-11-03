//! One guide, read from its file: YAML frontmatter between `---` lines, then
//! the method in Markdown.

use serde::Deserialize;

/// One guide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guide {
    /// What `guide_read` takes; its file's stem.
    pub name: String,
    /// When to use it, in one line.
    pub description: String,
    /// The tools it applies to, whichever backend serves them.
    pub tools: Vec<String>,
    /// The clients it is the guide for, by `mcp_server::Client::name`: served
    /// to a conversation through one of them before anything else
    /// (`docs/mcp_user_context.md` §2.4). Most guides are for none.
    pub clients: Vec<String>,
    /// The method.
    pub body: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Front {
    name: String,
    description: String,
    tools: Vec<String>,
    #[serde(default)]
    clients: Vec<String>,
}

impl Guide {
    /// The guide `text` holds, or what is wrong with it.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.replace("\r\n", "\n");
        let rest = text
            .strip_prefix("---\n")
            .ok_or("it does not open with a `---` frontmatter line")?;
        let (front, body) = rest
            .split_once("\n---\n")
            .ok_or("its frontmatter has no closing `---` line")?;
        let front: Front =
            serde_yaml::from_str(front).map_err(|e| format!("its frontmatter: {e}"))?;
        let body = body.trim().to_string();
        let guide = Self {
            name: front.name.trim().to_string(),
            description: front.description.trim().to_string(),
            tools: front.tools,
            clients: front.clients,
            body,
        };
        if guide.name.is_empty() {
            return Err("it has no `name`".into());
        }
        if guide.description.is_empty() {
            return Err("it has no `description`".into());
        }
        if guide.tools.is_empty() {
            return Err("it lists no `tools`".into());
        }
        if guide.body.is_empty() {
            return Err("it has no method below its frontmatter".into());
        }
        Ok(guide)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guide_reads_as_its_frontmatter_and_its_method() {
        let g = Guide::parse(
            "---\r\nname: research\r\ndescription: A question answered from the web.\r\n\
             tools: [repo_task_start, repo_task_wait]\r\n---\r\n# Research\r\n\r\nGather.\r\n",
        )
        .unwrap();
        assert_eq!(
            g,
            Guide {
                name: "research".into(),
                description: "A question answered from the web.".into(),
                tools: vec!["repo_task_start".into(), "repo_task_wait".into()],
                clients: Vec::new(),
                body: "# Research\n\nGather.".into(),
            }
        );
    }

    #[test]
    fn a_guide_may_name_the_clients_it_is_for() {
        let g = Guide::parse(
            "---\nname: grok\ndescription: Working with the user through Grok.\n\
             tools: [get_user_context]\nclients: [grok]\n---\n# Grok\n",
        )
        .unwrap();
        assert_eq!(g.clients, ["grok"]);
    }

    #[test]
    fn a_guide_missing_a_part_says_which() {
        assert_eq!(
            Guide::parse("# no frontmatter").unwrap_err(),
            "it does not open with a `---` frontmatter line"
        );
        assert_eq!(
            Guide::parse("---\nname: x\n").unwrap_err(),
            "its frontmatter has no closing `---` line"
        );
        assert_eq!(
            Guide::parse("---\nname: x\ndescription: y\ntools: []\n---\nbody").unwrap_err(),
            "it lists no `tools`"
        );
        assert_eq!(
            Guide::parse("---\nname: x\ndescription: y\ntools: [a]\n---\n").unwrap_err(),
            "it has no method below its frontmatter"
        );
        assert!(
            Guide::parse("---\nname: x\ndescription: y\ntools: [a]\nextra: 1\n---\nb")
                .unwrap_err()
                .starts_with("its frontmatter: ")
        );
    }
}
