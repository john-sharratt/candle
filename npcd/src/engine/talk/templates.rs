//! The response templates a conversation's words are shaped by
//! (`docs/npc_conversations.md` §6.3).
//!
//! One collection, `response`, for every loop: selected by
//! [`SelectionRule::Offered`], so each turn names the templates it may be
//! shown and provenance ranks the one it is. A talk turn offers only its
//! destination's — a reply, a parting, a journal entry, a channel post — so
//! whatever is shown always matches what the words are for. The templates
//! are the mind's, one file each under `responses/talk/<destination>/`.

use std::collections::BTreeMap;

use candle_conversation::projection::{Builder, SelectionRule, SelectionState};
use serde::Deserialize;

use crate::mind::files::MindFiles;

/// The collection the templates are installed into.
pub const COLLECTION: &str = "response";

/// Where a turn's words go, and so which templates shape them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Destination {
    /// A line to the party.
    Reply,
    /// The last line to the party: the conversation ends.
    Parting,
    /// Written down in the character's journal; nothing is sent.
    Journal,
    /// A post on a channel or group, from the mind loop.
    Channel,
}

impl Destination {
    pub const ALL: [Destination; 4] = [
        Destination::Reply,
        Destination::Parting,
        Destination::Journal,
        Destination::Channel,
    ];

    /// The folder its templates are in, under `responses/talk/`.
    pub fn folder(self) -> &'static str {
        match self {
            Destination::Reply => "reply",
            Destination::Parting => "parting",
            Destination::Journal => "journal",
            Destination::Channel => "channel",
        }
    }
}

/// A template's file.
#[derive(Debug, Deserialize)]
struct TemplateFile {
    #[serde(default)]
    template: Option<String>,
}

/// The member name a template takes in the collection.
fn member(dest: Destination, stem: &str) -> String {
    format!("{COLLECTION}/talk/{}/{stem}", dest.folder())
}

/// What was installed: each destination's member names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Installed {
    by: BTreeMap<Destination, Vec<String>>,
}

impl Installed {
    /// Offer `dest`'s templates on a turn, replacing whatever it offered.
    pub fn offer(&self, selection: &mut SelectionState, dest: Destination) {
        selection.select_all(
            COLLECTION,
            self.by.get(&dest).cloned().unwrap_or_default(),
        );
    }

    /// How many templates `dest` has.
    pub fn count(&self, dest: Destination) -> usize {
        self.by.get(&dest).map_or(0, Vec::len)
    }
}

/// Install every talk template the mind has into `builder`'s `response`
/// collection, and make it [`SelectionRule::Offered`]. A mind with none
/// installs none: the turn's own words still say what is wanted.
pub fn install(builder: &mut Builder, files: Option<&MindFiles>) -> anyhow::Result<Installed> {
    let mut installed = Installed::default();
    let Some(cid) = builder.id_for_system_collection(COLLECTION) else {
        return Ok(installed);
    };
    builder
        .set_collection_selection(
            COLLECTION,
            SelectionRule::Offered {
                selector: COLLECTION.to_string(),
                k: 1,
            },
        )
        .map_err(|e| anyhow::anyhow!("offering `{COLLECTION}` by turn: {e}"))?;
    let Some(files) = files else {
        return Ok(installed);
    };
    for dest in Destination::ALL {
        let dir = format!("responses/talk/{}", dest.folder());
        for file in files.list(&dir).into_iter().filter(|f| f.ends_with(".yaml")) {
            let rel = format!("{dir}/{file}");
            let Some(text) = files.read(&rel) else {
                tracing::warn!(%rel, "talk template unreadable, skipping");
                continue;
            };
            let template = match serde_yaml::from_str::<TemplateFile>(&text) {
                Ok(TemplateFile {
                    template: Some(t),
                }) if !t.trim().is_empty() => t,
                Ok(_) => {
                    tracing::warn!(%rel, "talk template has no `template`, skipping");
                    continue;
                }
                Err(e) => {
                    tracing::warn!(%rel, error = %e, "talk template does not parse, skipping");
                    continue;
                }
            };
            let name = member(dest, file.trim_end_matches(".yaml"));
            builder
                .add_section_to_collection(cid, &name, &template, 100.0)
                .map_err(|e| anyhow::anyhow!("installing `{name}`: {e}"))?;
            installed.by.entry(dest).or_default().push(name);
        }
    }
    Ok(installed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_templates_member_names_its_destination() {
        assert_eq!(member(Destination::Reply, "warm"), "response/talk/reply/warm");
        assert_eq!(member(Destination::Journal, "note"), "response/talk/journal/note");
    }

    /// **A turn is offered only its destination's templates.**
    #[test]
    fn a_turn_is_offered_its_destinations_templates_alone() {
        let mut installed = Installed::default();
        installed
            .by
            .insert(Destination::Reply, vec!["response/talk/reply/warm".into()]);
        installed
            .by
            .insert(Destination::Journal, vec!["response/talk/journal/note".into()]);
        let mut sel = SelectionState::new();
        installed.offer(&mut sel, Destination::Journal);
        assert_eq!(sel.members(COLLECTION), ["response/talk/journal/note"]);
        installed.offer(&mut sel, Destination::Parting);
        assert!(sel.members(COLLECTION).is_empty());
        assert_eq!(installed.count(Destination::Reply), 1);
    }
}
