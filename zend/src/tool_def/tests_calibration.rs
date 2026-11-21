//! Tests that a tool's calibration cases change whenever anything the model reads
//! about the tool changes, and that its question seeds are distinct and distinctive.
//!
//! A tool's question group is calibrated as one unit, so its marker covers every
//! question. Editing, adding or reordering any question moves the marker, which stops
//! a reworded question from resuming exemplars captured from text that no longer
//! exists. Every tool carries question seeds that are really questions, no two are
//! near-identical, and each names something that distinguishes its own tool, since a
//! seed that fits every closing request would pull every closing tool toward it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_definitions::tests::def;

    /// A tool's question GROUP and its examples are distinct cases, so the
    /// resume cache cannot confuse them and editing one leaves the other alone.
    ///
    /// The group's marker covers every question at once because the group is
    /// regenerated as a unit: all of a tool's questions go in as ONE stuffed
    /// prefill, so a per-question marker could never be resumed independently.
    #[test]
    fn a_question_groups_marker_differs_from_an_examples() {
        let d = def("List the files.", serde_json::json!({}));
        assert_ne!(
            d.calibration_marker(&d.questions.join("\u{0}")),
            d.calibration_marker(&d.examples[0])
        );
    }

    /// **Editing ANY question in the group must move the group's marker.**
    /// Otherwise a reworded question resumes against exemplars captured from
    /// text that no longer exists — the same staleness the whole-definition hash
    /// prevents, one level down.
    #[test]
    fn a_question_groups_marker_moves_when_any_question_changes() {
        let mut d = def("List the files.", serde_json::json!({}));
        d.questions = vec!["what files are here".into(), "show me the directory".into()];
        let baseline = d.calibration_marker(&d.questions.join("\u{0}"));

        // Reword the SECOND question — the one a marker built from only the
        // first, or from the count, would miss.
        let mut edited = d.clone();
        edited.questions[1] = "show me this directory".into();
        assert_ne!(
            edited.calibration_marker(&edited.questions.join("\u{0}")),
            baseline,
        );

        // Adding one moves it too.
        let mut added = d.clone();
        added.questions.push("list everything".into());
        assert_ne!(
            added.calibration_marker(&added.questions.join("\u{0}")),
            baseline,
        );

        // Reordering moves it as well: order decides which block a question
        // lands in, so two orderings are genuinely different corpora.
        let mut swapped = d.clone();
        swapped.questions.swap(0, 1);
        assert_ne!(
            swapped.calibration_marker(&swapped.questions.join("\u{0}")),
            baseline,
        );
    }

    /// A recorded trajectory is its own case; a bare prompt is not one at all.
    #[test]
    fn only_an_example_with_an_answer_is_a_trajectory() {
        let mut d = def("List the files.", serde_json::json!({}));
        d.examples = vec![
            "list src<|im_end|>\n<|im_start|>assistant\n<think>x</think>".into(),
            "list the files<|im_end|>".into(),
        ];
        let t: Vec<&str> = d.trajectories("<|im_start|>assistant").collect();
        assert_eq!(
            t,
            ["list src<|im_end|>\n<|im_start|>assistant\n<think>x</think>"]
        );
    }

    /// A bare prompt joins the question group as plain text, after the
    /// authored questions and without its ChatML terminator.
    #[test]
    fn a_bare_prompt_is_calibrated_as_a_question() {
        let mut d = def("List the files.", serde_json::json!({}));
        d.examples = vec![
            "list src<|im_end|>\n<|im_start|>assistant\n<think>x</think>".into(),
            "list the files<|im_end|>".into(),
            "  what is in docs?<|im_end|>\n".into(),
        ];
        assert_eq!(
            d.calibration_questions("<|im_start|>assistant", "<|im_end|>\n"),
            ["what files are here", "list the files", "what is in docs?"]
        );
    }

    /// The calibration marker must move whenever anything the model reads about
    /// the tool moves. Hashing only the example (the earlier behaviour) let a
    /// reworded description resume its old exemplars, so the belief gallery kept
    /// answering with signatures captured against text that no longer existed.
    #[test]
    fn calibration_marker_covers_the_whole_projected_definition() {
        let params = serde_json::json!({"properties": {"prefix": {"type": "string"}}});
        let base = def("List the files.", params.clone());
        let example = &base.examples[0];
        let baseline = base.calibration_marker(example);

        // Same definition, same example → same marker (the resume cache hits).
        assert_eq!(base.calibration_marker(example), baseline);

        // Description reworded → marker moves, so the case re-runs.
        let reworded = def("List the files visible to this session.", params.clone());
        assert_ne!(reworded.calibration_marker(example), baseline);

        // A *parameter's* description is projected too — it is part of the
        // rendered schema the model reads when choosing arguments.
        let reparam = def(
            "List the files.",
            serde_json::json!({
                "properties": {"prefix": {"type": "string", "description": "project root"}}
            }),
        );
        assert_ne!(reparam.calibration_marker(example), baseline);

        // A different example under an unchanged definition is its own case.
        assert_ne!(base.calibration_marker("something else"), baseline);

        // The marker stays name-prefixed so it is greppable per tool.
        assert!(baseline.starts_with("file_list|"));
    }

    /// Two tools sharing a description and example must not collide.
    #[test]
    fn calibration_markers_are_per_tool() {
        let params = serde_json::json!({});
        let mut a = def("same text", params.clone());
        let mut b = def("same text", params);
        a.name = "file_read".into();
        b.name = "file_delete".into();
        assert_ne!(
            a.calibration_marker("ex"),
            b.calibration_marker("ex"),
            "the name must participate in the marker",
        );
    }

    /// **Every tool carries question-only seeds, and they really are questions.**
    ///
    /// A tool with none is invisible to any probe shaped like a question until
    /// its full trajectories happen to match — which is the failure the seeds
    /// exist to remove, and it is silent: the catalog still renders, the tool
    /// still executes, it just stops being *selected*. The shape assertions
    /// catch a trajectory pasted in by mistake, which would prefill an authored
    /// answer instead of a bare question.
    #[test]
    fn every_tool_seeds_the_question_gallery() {
        for def in all() {
            assert!(
                def.questions.len() >= 16,
                "tool {:?} has {} question seeds; the schema requires at least 16",
                def.name,
                def.questions.len()
            );
            for q in &def.questions {
                assert!(
                    !q.trim().is_empty(),
                    "tool {:?} has a blank question",
                    def.name
                );
                for marker in ["<|im_end|>", "<|im_start|>", "<think>", "<tool_call>"] {
                    assert!(
                        !q.contains(marker),
                        "tool {:?} question contains {marker:?} — a question is plain \
                         text; calibration adds the ChatML framing itself",
                        def.name
                    );
                }
            }
        }
    }

    /// **No two seeds may be the same or near-identical.**
    ///
    /// A seed earns its prefill only by covering phrasing the others do not. Two
    /// questions that differ by a filler word occupy two gallery slots, cost two
    /// prefills, and match the same probes — so the corpus grows without the
    /// coverage growing, which is the exact failure a bulk expansion invites.
    /// Jaccard over content tokens catches the paraphrase a diff review misses.
    #[test]
    fn question_seeds_are_distinct_from_one_another() {
        fn content(q: &str) -> HashSet<String> {
            q.to_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 1)
                .map(str::to_string)
                .collect()
        }
        let mut seen: HashSet<String> = HashSet::new();
        for def in all() {
            let sets: Vec<HashSet<String>> = def.questions.iter().map(|q| content(q)).collect();
            for (i, a) in sets.iter().enumerate() {
                assert!(
                    seen.insert(def.questions[i].to_lowercase()),
                    "tool {:?} repeats the question {:?}",
                    def.name,
                    def.questions[i]
                );
                for (j, b) in sets.iter().enumerate().skip(i + 1) {
                    if a.is_empty() || b.is_empty() {
                        continue;
                    }
                    let overlap = a.intersection(b).count() as f32;
                    let union = a.union(b).count() as f32;
                    assert!(
                        overlap / union < 0.8,
                        "tool {:?}: {:?} and {:?} are near-identical ({:.0}% token \
                         overlap) — a paraphrase costs a prefill and adds no coverage",
                        def.name,
                        def.questions[i],
                        def.questions[j],
                        100.0 * overlap / union,
                    );
                }
            }
        }
    }

    /// **A seed must name something that distinguishes its own tool.**
    ///
    /// "close it" is a valid English request and a useless exemplar: fifteen
    /// tools close something, so the seed pulls every one of them toward every
    /// closing probe. That is precisely the promiscuous gallery entry the
    /// hit-level normalizer exists to discount, and it is cheaper not to author
    /// it. The check is deliberately weak — one token that at most a quarter of
    /// the catalog uses — because the goal is to reject the genuinely contentless
    /// seed, not to police phrasing.
    #[test]
    fn every_question_seed_names_something_distinctive() {
        let defs = all();
        let mut doc_freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let words = |q: &str| -> Vec<String> {
            q.to_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 2)
                .map(str::to_string)
                .collect()
        };
        for def in defs {
            let mut here: HashSet<String> = HashSet::new();
            for q in &def.questions {
                here.extend(words(q));
            }
            for w in here {
                *doc_freq.entry(w).or_default() += 1;
            }
        }
        let cap = defs.len() / 4;
        for def in defs {
            for q in &def.questions {
                let ws = words(q);
                assert!(
                    ws.iter().any(|w| doc_freq[w] <= cap),
                    "tool {:?}: {q:?} uses only words common across the catalog — it \
                     cannot route to this tool rather than its siblings. Name the \
                     protocol, object or verb that makes this tool the answer.",
                    def.name,
                );
            }
        }
    }
}
