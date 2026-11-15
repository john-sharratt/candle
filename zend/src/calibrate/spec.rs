//! A layer's `calibrate:` block — the prompts each of its conversations is
//! asked in turn to write questions about itself, how many each asks for, and
//! how much of each question's thinking is kept — read from the projection
//! schema it is declared in.
//!
//! ```yaml
//! - name: code_reading
//!   calibrate:
//!     think: 64
//!     prompts:
//!       - name: definitions
//!         questions: { per_tokens: 300, min: 4, max: 32 }
//!         prompt: |
//!           Write <<n>> questions about what `<<path>>` defines …
//!       - name: uses
//!         questions: { per_tokens: 400, min: 3, max: 24 }
//!         prompt: |
//!           Now write <<n>> more, about how what it defines is used …
//!     guidance:
//!       questions: { per_tokens: 250, min: 8, max: 64 }
//!       prompt: |
//!         Write <<n>> questions a worker might ask itself …
//! ```
//!
//! Nothing here names a layer: any layer whose conversations are units — a
//! file's reading, an analysis of it, a day — may declare one, and each of its
//! units is calibrated the same way ([`super::run`]).

use std::collections::HashSet;
use std::slice;

use candle_conversation::projection::{Draws, LayerSchema};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::ask::{Ask, RawAsk};
use super::count::QuestionCount;

/// The most questions one unit may be asked for, every prompt together. Each
/// is one turn of one prefill, and the calibration's K/V is held lossless
/// until it seals.
pub const MAX_QUESTIONS: usize = 64;

/// The most thinking tokens kept per question: a live probe's trailing
/// window is at most this long (`REPROJECT_MAX_PROBE_TOKENS`).
pub const MAX_THINK: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    think: usize,
    prompts: Vec<RawAsk>,
    /// What a unit a drawing group of the layer draws is asked instead.
    #[serde(default)]
    guidance: Option<RawGuidance>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGuidance {
    questions: QuestionCount,
    prompt: String,
}

/// How a layer's conversations are calibrated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrateSpec {
    /// The layer's name.
    pub layer: String,
    /// The decode budget of each question's thinking ([`super::thinking`]).
    pub think: usize,
    /// What every unit is asked, in order, each prompt seeing the answers to
    /// the ones before it in the same draft.
    pub prompts: Vec<Ask>,
    /// What a unit the layer's drawing groups draw is asked instead, and the
    /// paths they draw (`docs/guidance_layer.md` §2.2). A guide is wanted
    /// at a point in a worker's own reasoning, so its seed questions are the
    /// ones a worker asks itself mid-task, not a user's.
    pub guidance: Option<GuidancePrompt>,
}

/// The prompt for the units a layer's drawing groups draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidancePrompt {
    pub ask: Ask,
    /// What every drawing group of the layer draws.
    pub draws: Vec<Draws>,
}

impl CalibrateSpec {
    /// The layer's spec, `None` when it declares no `calibrate:` block.
    pub fn of(layer: &LayerSchema) -> anyhow::Result<Option<Self>> {
        let Some(value) = &layer.calibrate else {
            return Ok(None);
        };
        let name = &layer.name;
        let raw: Raw = serde_yaml::from_value(value.clone())
            .map_err(|e| anyhow::anyhow!("layer '{name}': its `calibrate:` block: {e}"))?;
        if raw.think == 0 || raw.think > MAX_THINK {
            anyhow::bail!(
                "layer '{name}': `calibrate.think` must be 1..={MAX_THINK}, not {}",
                raw.think
            );
        }
        if raw.prompts.is_empty() {
            anyhow::bail!("layer '{name}': `calibrate.prompts` is empty");
        }
        let prompts = raw
            .prompts
            .into_iter()
            .map(Ask::of)
            .collect::<Result<Vec<Ask>, String>>()
            .map_err(|why| anyhow::anyhow!("layer '{name}': `calibrate.prompts`: {why}"))?;
        let mut names = HashSet::new();
        if let Some(twice) = prompts.iter().find(|a| !names.insert(a.name.as_str())) {
            anyhow::bail!(
                "layer '{name}': `calibrate.prompts`: '{}' is declared twice",
                twice.name
            );
        }
        within_the_cap(name, "prompts", &prompts)?;
        let draws: Vec<Draws> = layer
            .groups
            .iter()
            .filter_map(|g| g.draws.clone())
            .collect();
        let guidance = match raw.guidance {
            None => None,
            Some(_) if draws.is_empty() => anyhow::bail!(
                "layer '{name}': `calibrate.guidance` is declared but no group of the layer \
                 draws anything"
            ),
            Some(g) => {
                let ask = Ask::of(RawAsk {
                    name: "guidance".into(),
                    questions: g.questions,
                    prompt: g.prompt,
                })
                .map_err(|why| anyhow::anyhow!("layer '{name}': `calibrate.guidance`: {why}"))?;
                within_the_cap(name, "guidance", slice::from_ref(&ask))?;
                Some(GuidancePrompt { ask, draws })
            }
        };
        Ok(Some(Self {
            layer: name.clone(),
            think: raw.think,
            prompts,
            guidance,
        }))
    }

    /// What `subject` is asked, in order: the guidance prompt for a unit a
    /// drawing group draws, the layer's prompts for every other.
    pub fn asks_for(&self, subject: &str) -> &[Ask] {
        match &self.guidance {
            Some(g) if g.draws.iter().any(|d| d.takes(subject)) => slice::from_ref(&g.ask),
            _ => &self.prompts,
        }
    }

    /// The calibration marker of the unit committed under `content_key`, whose
    /// subject is `subject`: the layer, a digest of what it is asked, and the
    /// unit's own key — so a changed prompt, its question-count rule, the set
    /// of prompts, the thinking budget or the unit's version calibrates again,
    /// and a changed guidance prompt only the units asked it. The unit's own
    /// counts are not in it: they follow from the rules and the content the key
    /// names.
    pub fn marker(&self, content_key: &str, subject: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.think.to_le_bytes());
        for ask in self.asks_for(subject) {
            ask.digest_into(&mut h);
        }
        let digest: String = h.finalize()[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("{}|{digest}|{content_key}", self.layer)
    }

    /// The content key of the unit `marker` was written for — the part a
    /// changed prompt, count or budget leaves alone.
    pub fn unit_of(marker: &str) -> Option<&str> {
        marker.splitn(3, '|').nth(2)
    }

    /// Whether `marker` is one of this layer's.
    pub fn owns(&self, marker: &str) -> bool {
        marker
            .strip_prefix(self.layer.as_str())
            .is_some_and(|rest| rest.starts_with('|'))
    }
}

/// Refuse `asks` whose budgets could together ask one unit for more than
/// [`MAX_QUESTIONS`].
fn within_the_cap(layer: &str, field: &str, asks: &[Ask]) -> anyhow::Result<()> {
    let most: usize = asks.iter().map(|a| a.questions.max).sum();
    if most > MAX_QUESTIONS {
        anyhow::bail!(
            "layer '{layer}': `calibrate.{field}` can ask a unit for {most} questions, past \
             the {MAX_QUESTIONS} one calibration holds"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::{Builder, GroupId};

    fn ask(name: &str, max: usize, prompt: &str) -> Ask {
        Ask {
            name: name.into(),
            questions: QuestionCount {
                per_tokens: 300,
                min: 3,
                max,
            },
            prompt: prompt.into(),
        }
    }

    fn spec() -> CalibrateSpec {
        CalibrateSpec {
            layer: "code_reading".into(),
            think: 64,
            prompts: vec![
                ask("definitions", 32, "<<n>> about `<<path>>` in <<repo>>"),
                ask("uses", 24, "<<n>> more on `<<path>>`"),
            ],
            guidance: None,
        }
    }

    fn with_guidance() -> CalibrateSpec {
        CalibrateSpec {
            guidance: Some(GuidancePrompt {
                ask: ask(
                    "guidance",
                    64,
                    "<<n>> a worker asks itself before `<<path>>`",
                ),
                draws: vec![Draws {
                    from: GroupId::from_raw(1).unwrap(),
                    paths: vec!["candle/mcp/src/guides/".into(), "journals/".into()],
                    except: vec!["README.md".into()],
                }],
            }),
            ..spec()
        }
    }

    /// **A drawn unit is asked the guidance prompt alone**; every other unit
    /// the layer's prompts in order, a hub file under a drawn path among them.
    #[test]
    fn a_drawn_unit_is_asked_the_guidance_prompt() {
        let s = with_guidance();
        let names = |subject: &str| {
            s.asks_for(subject)
                .iter()
                .map(|a| a.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names("candle/mcp/src/guides/deploy.md"), ["guidance"]);
        assert_eq!(
            names("journals/groups/zen/README.md"),
            ["definitions", "uses"]
        );
        assert_eq!(names("candle/docs/paper.md"), ["definitions", "uses"]);
        assert_eq!(
            s.asks_for("candle/mcp/src/guides/deploy.md")[0]
                .prompt_for("candle/mcp/src/guides/deploy.md", 3),
            "3 a worker asks itself before `mcp/src/guides/deploy.md`"
        );
    }

    /// **A changed guidance prompt calibrates only the units it asks**: a code
    /// file's marker stands.
    #[test]
    fn a_changed_guidance_prompt_moves_only_the_drawn_markers() {
        let s = with_guidance();
        let mut reworded = s.clone();
        reworded.guidance.as_mut().unwrap().ask.prompt.push('!');
        let (guide, code) = ("candle/mcp/src/guides/deploy.md", "candle/docs/paper.md");
        assert_ne!(reworded.marker("g", guide), s.marker("g", guide));
        assert_eq!(reworded.marker("c", code), s.marker("c", code));
        assert_eq!(s.marker("c", code), spec().marker("c", code));
    }

    /// The marker changes with any prompt, its rule, the set of prompts, the
    /// thinking budget and the unit, and is read back as the layer's; another
    /// layer's is not.
    #[test]
    fn a_marker_follows_what_is_asked_and_of_which_unit() {
        let s = spec();
        let at = "candle/a.rs";
        let m = s.marker("k1", at);
        assert!(m.starts_with("code_reading|") && m.ends_with("|k1"), "{m}");
        assert_eq!(s.marker("k1", at), m);
        assert_ne!(s.marker("k2", at), m);
        let mut more = s.clone();
        more.prompts[1].questions.per_tokens = 200;
        assert_ne!(more.marker("k1", at), m);
        let mut longer = s.clone();
        longer.think = 128;
        assert_ne!(longer.marker("k1", at), m);
        let mut reworded = s.clone();
        reworded.prompts[0].prompt.push('!');
        assert_ne!(reworded.marker("k1", at), m);
        let mut fewer = s.clone();
        fewer.prompts.pop();
        assert_ne!(fewer.marker("k1", at), m);
        let mut swapped = s.clone();
        swapped.prompts.swap(0, 1);
        assert_ne!(swapped.marker("k1", at), m, "the order is what is asked");
        assert_eq!(CalibrateSpec::unit_of(&m), Some("k1"));
        assert_eq!(CalibrateSpec::unit_of(&longer.marker("k1", at)), Some("k1"));
        assert_eq!(
            CalibrateSpec::unit_of("code_reading|ab|key:a|b"),
            Some("key:a|b")
        );
        assert_eq!(CalibrateSpec::unit_of("code_reading|ab"), None);
        assert!(s.owns(&m));
        assert!(!s.owns("code_reading_x|ab|k1"));
        assert!(!s.owns("repo_map|ab|k1"));
    }

    fn layer(block: &str) -> LayerSchema {
        let yaml = format!(
            r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: code_reading
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
{block}
    groups:
      - id: scopes
        selection: {{ kind: top_k, k: 2 }}
"#
        );
        Builder::from_yaml(&yaml).unwrap().schema().layers[0].clone()
    }

    /// A block with `prompts` as YAML text under `calibrate:`.
    fn block(think: &str, prompts: &str) -> LayerSchema {
        layer(&format!("    calibrate:\n{think}      prompts:\n{prompts}"))
    }

    fn prompt(name: &str, max: usize, text: &str) -> String {
        format!(
            "        - name: {name}\n          questions: {{ per_tokens: 300, min: 3, max: {max} }}\n          \
             prompt: \"{text}\"\n"
        )
    }

    #[test]
    fn a_block_is_read_and_a_bad_one_refused() {
        let two = prompt("definitions", 32, "ask <<n>>") + &prompt("uses", 24, "more <<n>>");
        let s = CalibrateSpec::of(&block("      think: 192\n", &two))
            .unwrap()
            .expect("declared");
        assert_eq!((s.layer.as_str(), s.think), ("code_reading", 192));
        assert_eq!(
            s.prompts
                .iter()
                .map(|a| (a.name.as_str(), a.questions.max))
                .collect::<Vec<_>>(),
            [("definitions", 32), ("uses", 24)]
        );
        assert!(CalibrateSpec::of(&layer("")).unwrap().is_none());
        let refused = |think: &str, prompts: &str| {
            CalibrateSpec::of(&block(think, prompts))
                .unwrap_err()
                .to_string()
        };
        let think = "      think: 8\n";
        assert_eq!(
            refused(think, &(prompt("uses", 24, "x") + &prompt("uses", 8, "y"))),
            "layer 'code_reading': `calibrate.prompts`: 'uses' is declared twice"
        );
        assert_eq!(
            refused(
                think,
                &(prompt("definitions", 40, "x") + &prompt("uses", 25, "y"))
            ),
            "layer 'code_reading': `calibrate.prompts` can ask a unit for 65 questions, past the \
             64 one calibration holds"
        );
        assert_eq!(
            refused(think, &prompt("uses", 8, " ")),
            "layer 'code_reading': `calibrate.prompts`: prompt 'uses': `prompt` is empty"
        );
        assert!(refused("      think: 0\n", &prompt("uses", 8, "x")).contains("think"));
        assert!(refused("      think: 257\n", &prompt("uses", 8, "x")).contains("think"));
        assert!(
            refused("", &prompt("uses", 8, "x")).contains("think"),
            "think is required"
        );
        let undrawn = layer(&format!(
            "    calibrate:\n      think: 8\n      prompts:\n{}      guidance:\n        \
             questions: {{ per_tokens: 250, min: 4, max: 4 }}\n        prompt: \"y\"\n",
            prompt("uses", 8, "x")
        ));
        assert_eq!(
            CalibrateSpec::of(&undrawn).unwrap_err().to_string(),
            "layer 'code_reading': `calibrate.guidance` is declared but no group of the layer \
             draws anything"
        );
    }

    /// An empty `prompts:` list asks nothing, and is refused.
    #[test]
    fn a_block_without_prompts_is_refused() {
        let empty = layer("    calibrate:\n      think: 8\n      prompts: []\n");
        assert_eq!(
            CalibrateSpec::of(&empty).unwrap_err().to_string(),
            "layer 'code_reading': `calibrate.prompts` is empty"
        );
    }

    /// **A guidance prompt takes the paths the layer's drawing groups draw.**
    #[test]
    fn a_guidance_prompt_takes_the_drawn_paths() {
        let yaml = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: code_reading
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    calibrate:
      think: 8
      prompts:
        - name: definitions
          questions: { per_tokens: 300, min: 4, max: 4 }
          prompt: "x"
      guidance:
        questions: { per_tokens: 250, min: 4, max: 4 }
        prompt: "y"
    groups:
      - id: scopes
        selection: { kind: working_set, share: remainder }
      - id: guidance
        draws: { from: scopes, paths: [candle/mcp/src/guides/, journals/], except: [README.md] }
        selection: { kind: working_set, share: guidance }
"#;
        let layer = Builder::from_yaml(yaml).unwrap().schema().layers[0].clone();
        let s = CalibrateSpec::of(&layer).unwrap().expect("declared");
        let g = s.guidance.expect("a guidance prompt");
        assert_eq!(
            (g.ask.name.as_str(), g.ask.prompt.as_str()),
            ("guidance", "y")
        );
        assert_eq!(g.draws.len(), 1);
        assert_eq!(g.draws[0].paths, ["candle/mcp/src/guides/", "journals/"]);
        assert_eq!(g.draws[0].except, ["README.md"]);
    }
}
