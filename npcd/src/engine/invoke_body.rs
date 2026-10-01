//! The `invoke` call's address and the body that address takes, as one grammar.
//!
//! # Why the body is in the grammar
//!
//! An `invoke` is two arguments that are not independent: the address says which
//! act runs, and the act says what the body must hold. Left as a free string the
//! body was the place a character failed — a field missing, a name the room does
//! not have, a number where a word was wanted — and the world's refusal arrived
//! a turn after the mistake that caused it.
//!
//! So the grammar writes the two together. Each address the character may act at
//! brings the body *its* act takes: the act's own parameters, typed and bound
//! against the room exactly as the act's compiled call is. A field the act
//! requires cannot be skipped, a world-enumerated one offers only what the room
//! has, and an address that takes nothing can only carry `{}`.
//!
//! # What an address is
//!
//! An [`Invokable`] is a verb-path and the act behind it: a station's verbs are
//! its catalogue acts (`station::verbs_at`), so the act's parameters *are* the
//! body's fields and there is no second description of them to drift.
//!
//! # Cost
//!
//! Every address with the same act has the same body, and the stencil shares
//! equal bodies ([`StencilParam::shapes`]): the tree grows with the number of
//! distinct acts reachable, not with the number of addresses. Two hundred
//! fabricators are one body and two hundred addresses.

use std::collections::{BTreeMap, BTreeSet};

use candle_conversation::stencil::{Param as StencilParam, ParamType};

use crate::engine::tools::{by_name, performable, stencil_params, Tool, Within};

/// One address an `invoke` may act at, and the act that runs there.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Invokable {
    /// The whole verb-path, `http://local/<ns>/<id>/<verb>`.
    pub url: String,
    /// The catalogue act the verb runs — what types the body.
    pub act: &'static str,
}

impl Invokable {
    pub fn new(url: impl Into<String>, act: &'static str) -> Self {
        Self {
            url: url.into(),
            act,
        }
    }
}

/// The addresses a body here could act at right now, with the act each runs.
///
/// An address whose act the body could not perform — cooling, or a required
/// argument with nothing in the room to name — is absent, for the reason the
/// act's compiled call would be ([`performable`]). First mention of an address
/// wins.
fn performable_here(within: &Within) -> Vec<(&Invokable, &'static Tool)> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    within
        .invokable
        .iter()
        .filter(|i| seen.insert(i.url.as_str()))
        .filter_map(|i| {
            let tool = by_name(i.act)?;
            performable(tool, within).then_some((i, tool))
        })
        .collect()
}

/// Every address an `invoke` may name, in the order the room lists them.
pub fn urls(within: &Within) -> Vec<String> {
    performable_here(within)
        .into_iter()
        .map(|(i, _)| i.url.clone())
        .collect()
}

/// What each address brings with it: a required `body` object holding the fields
/// of the act behind it.
pub fn shapes(within: &Within) -> Vec<(String, Vec<StencilParam>)> {
    let mut bodies: BTreeMap<&str, StencilParam> = BTreeMap::new();
    performable_here(within)
        .into_iter()
        .map(|(i, tool)| {
            let body = bodies
                .entry(tool.name)
                .or_insert_with(|| body_of(tool, within))
                .clone();
            (i.url.clone(), vec![body])
        })
        .collect()
}

fn body_of(tool: &Tool, within: &Within) -> StencilParam {
    StencilParam {
        name: "body".to_string(),
        ty: ParamType::Object,
        required: true,
        enum_values: None,
        items: None,
        min_items: 0,
        properties: Some(stencil_params(tool, within)),
        nullable: false,
        minimum: None,
        requires: Vec::new(),
        shapes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use candle_conversation::stencil::{
        compile, compile_action_loop, TestVocab, ToolCallEnvelope, ToolSpec,
    };

    use super::*;
    use crate::engine::tools::{
        estimated_paths, specs_within, Mode, ACTS_PER_TURN, CATALOG, MAX_TURN_PATHS,
    };

    const TOWER: &str = "http://local/tower/bridge-console~0/command_tower";
    const FAB: &str = "http://local/fab/fabricator~0/produce";
    const MISSION: &str = "http://local/command/order-table~0/collect_mission";

    fn tower() -> Invokable {
        Invokable::new(TOWER, "command_tower")
    }

    fn fab(n: usize) -> Invokable {
        Invokable::new(
            format!("http://local/fab/fabricator~{n}/produce"),
            "produce",
        )
    }

    fn mission() -> Invokable {
        Invokable::new(MISSION, "collect_mission")
    }

    /// A room the tower and the fabricators can both be worked from.
    fn room() -> Within {
        Within {
            tower_actions: vec!["relocate".into(), "drill down".into()],
            makeable: vec!["bolt rounds".into(), "stimpaks".into()],
            queues: vec!["1".into(), "2".into(), "3".into()],
            invokable: vec![tower(), fab(0), mission()],
            ..Within::among(&[])
        }
    }

    fn body_of_url(within: &Within, url: &str) -> StencilParam {
        shapes(within)
            .into_iter()
            .find(|(u, _)| u == url)
            .unwrap_or_else(|| panic!("{url} has no shape in {:?}", urls(within)))
            .1
            .remove(0)
    }

    fn field<'a>(body: &'a StencilParam, name: &str) -> &'a StencilParam {
        body.properties
            .as_ref()
            .expect("an object body")
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no `{name}` field in {body:?}"))
    }

    fn names(body: &StencilParam) -> Vec<&str> {
        body.properties
            .as_ref()
            .expect("an object body")
            .iter()
            .map(|p| p.name.as_str())
            .collect()
    }

    fn values(p: &StencilParam) -> Vec<&str> {
        p.enum_values
            .as_ref()
            .unwrap_or_else(|| panic!("`{}` is free", p.name))
            .iter()
            .map(String::as_str)
            .collect()
    }

    // ── what an address takes ────────────────────────────────────────────────

    #[test]
    fn an_address_takes_the_fields_of_its_act_and_no_others() {
        let body = body_of_url(&room(), TOWER);
        assert_eq!(names(&body), ["action", "target", "x", "y", "depth"]);
        assert!(field(&body, "action").required);
        for optional in ["target", "x", "y", "depth"] {
            assert!(!field(&body, optional).required, "{optional}");
        }
        let produce = body_of_url(&room(), FAB);
        assert_eq!(names(&produce), ["what", "count", "queue"]);
        assert!(field(&produce, "what").required);
        assert!(!field(&produce, "count").required);
    }

    #[test]
    fn the_body_is_a_required_object() {
        let body = body_of_url(&room(), TOWER);
        assert_eq!(body.name, "body");
        assert_eq!(body.ty, ParamType::Object);
        assert!(body.required);
    }

    #[test]
    fn an_address_that_takes_nothing_carries_the_empty_object() {
        let body = body_of_url(&room(), MISSION);
        assert_eq!(names(&body), Vec::<&str>::new());
        assert!(body.properties.is_some());
    }

    #[test]
    fn a_fields_values_come_from_the_room() {
        let within = room();
        let tower = body_of_url(&within, TOWER);
        assert_eq!(values(field(&tower, "action")), ["relocate", "drill down"]);
        let produce = body_of_url(&within, FAB);
        assert_eq!(values(field(&produce, "what")), ["bolt rounds", "stimpaks"]);
        assert_eq!(values(field(&produce, "queue")), ["1", "2", "3"]);

        let poorer = Within {
            tower_actions: vec!["relocate".into()],
            ..room()
        };
        let tower = body_of_url(&poorer, TOWER);
        assert_eq!(values(field(&tower, "action")), ["relocate"]);
    }

    /// The body an address carries is typed by the act's own parameters, so a
    /// fold written through the device cannot close without a destination
    /// either.
    #[test]
    fn a_tower_action_in_the_body_requires_the_fields_it_takes() {
        let tower = body_of_url(&room(), TOWER);
        let action = field(&tower, "action");
        let needs = |value: &str| -> Vec<&str> {
            action
                .requires
                .iter()
                .filter(|(v, _)| v == value)
                .flat_map(|(_, fields)| fields.iter().map(String::as_str))
                .collect()
        };
        assert_eq!(needs("relocate"), ["x", "y"]);
        assert_eq!(needs("drill down"), ["depth"]);
    }

    #[test]
    fn a_free_field_stays_free() {
        let tower = body_of_url(&room(), TOWER);
        assert!(field(&tower, "depth").enum_values.is_none());
        for number in ["x", "y", "depth"] {
            assert_eq!(field(&tower, number).ty, ParamType::Integer, "{number}");
        }
    }

    // ── what an address needs to be offered ──────────────────────────────────

    #[test]
    fn an_address_whose_required_field_has_nothing_to_name_is_dropped() {
        let within = Within {
            makeable: Vec::new(),
            ..room()
        };
        assert_eq!(urls(&within), [TOWER, MISSION]);
        assert!(shapes(&within).iter().all(|(u, _)| u != FAB));
    }

    #[test]
    fn an_optional_field_with_nothing_to_name_leaves_the_address() {
        let within = Within {
            queues: Vec::new(),
            ..room()
        };
        let produce = body_of_url(&within, FAB);
        assert_eq!(names(&produce), ["what", "count"]);
    }

    #[test]
    fn a_cooling_act_loses_its_addresses() {
        let within = Within {
            cooling: vec!["produce".into()],
            ..room()
        };
        assert_eq!(urls(&within), [TOWER, MISSION]);
    }

    #[test]
    fn an_address_naming_an_act_the_catalogue_lacks_is_dropped() {
        let within = Within {
            invokable: vec![Invokable::new("http://local/x/y~0/nothing", "no_such_act")],
            ..room()
        };
        assert!(urls(&within).is_empty());
    }

    #[test]
    fn an_address_listed_twice_is_offered_once() {
        let within = Within {
            invokable: vec![tower(), tower()],
            ..room()
        };
        assert_eq!(urls(&within), [TOWER]);
        assert_eq!(shapes(&within).len(), 1);
    }

    #[test]
    fn urls_and_shapes_name_the_same_addresses_in_the_same_order() {
        let within = room();
        let from_shapes: Vec<String> = shapes(&within).into_iter().map(|(u, _)| u).collect();
        assert_eq!(urls(&within), from_shapes);
        assert_eq!(from_shapes, [TOWER, FAB, MISSION]);
    }

    #[test]
    fn invoke_leaves_the_grammar_when_nothing_can_be_invoked() {
        let has = |within: &Within| {
            specs_within(Mode::Physical, within)
                .iter()
                .any(|s| s.name == "invoke")
        };
        assert!(has(&room()));
        let nothing = Within {
            invokable: Vec::new(),
            ..room()
        };
        assert!(!has(&nothing));
        let nothing_performable = Within {
            invokable: vec![fab(0)],
            makeable: Vec::new(),
            ..room()
        };
        assert!(!has(&nothing_performable));
    }

    #[test]
    fn the_invoke_spec_carries_a_shape_for_every_url_it_offers() {
        let spec = specs_within(Mode::Physical, &room())
            .into_iter()
            .find(|s| s.name == "invoke")
            .expect("invoke is offered");
        let url = spec.params.iter().find(|p| p.name == "url").expect("a url");
        let offered = url.enum_values.clone().expect("a closed set");
        let shaped: Vec<String> = url.shapes.iter().map(|(u, _)| u.clone()).collect();
        assert_eq!(offered, shaped);
    }

    // ── sharing, and what the tree costs ─────────────────────────────────────

    #[test]
    fn addresses_with_the_same_act_share_one_body() {
        let within = Within {
            invokable: (0..4).map(fab).collect(),
            ..room()
        };
        let shared = shapes(&within);
        assert_eq!(shared.len(), 4);
        let first = format!("{:?}", shared[0].1);
        assert!(shared.iter().all(|(_, body)| format!("{body:?}") == first));
    }

    fn compiled_nodes(specs: &[ToolSpec]) -> usize {
        let spec = compile_action_loop(
            specs,
            &ToolCallEnvelope::qwen3(),
            ACTS_PER_TURN,
            "<|im_end|>",
            None,
        )
        .expect("the catalog compiles");
        compile(&spec, &TestVocab::new())
            .expect("the spec lowers")
            .len()
    }

    #[test]
    fn a_fabricator_more_costs_its_address_and_not_its_body() {
        let nodes = |n: usize| {
            let within = Within {
                invokable: (0..n).map(fab).collect(),
                ..room()
            };
            compiled_nodes(&specs_within(Mode::Physical, &within))
        };
        let (few, many) = (nodes(2), nodes(40));
        let per_address = (many - few) / 38;
        assert!(
            per_address <= 400,
            "an address cost {per_address} nodes ({few} for 2, {many} for 40)"
        );
    }

    /// Every part in the catalogue, two of each, with everything a body could
    /// want to name in reach: wider than any room the maps hold.
    fn crowded() -> Within {
        let mut within = room();
        within.readable = vec!["the muster board".into()];
        within.postable = vec!["the muster board".into()];
        within.claimable = vec!["fabricator 1".into()];
        within.owed = vec!["the eastern span".into()];
        within.invokable = CATALOG
            .iter()
            .filter(|t| !t.at.is_empty())
            .flat_map(|t| {
                t.at.iter().flat_map(move |part| {
                    (0..2).map(move |n| {
                        Invokable::new(format!("http://local/{part}~{n}/{}", t.name), t.name)
                    })
                })
            })
            .collect();
        within
    }

    #[test]
    fn a_crowded_room_stays_inside_the_path_budget() {
        let within = crowded();
        assert!(!within.invokable.is_empty());
        let specs = specs_within(Mode::Physical, &within);
        let paths = estimated_paths(&specs);
        assert!(
            paths <= MAX_TURN_PATHS,
            "{} addresses admit {paths} decodes over a turn, over the {MAX_TURN_PATHS} budget",
            within.invokable.len()
        );
    }

    #[test]
    fn a_crowded_room_compiles_to_a_small_tree_quickly() {
        let within = crowded();
        let specs = specs_within(Mode::Physical, &within);
        let started = Instant::now();
        let nodes = compiled_nodes(&specs);
        let took = started.elapsed();
        let offered = urls(&within).len();
        eprintln!("{offered} addresses offered -> {nodes} nodes in {took:?}");
        assert!(
            offered > 0,
            "nothing was offered, so the tree proves nothing"
        );
        assert!(nodes < 400_000, "{nodes} nodes");
    }

    // ── the catalogue and the device agree ───────────────────────────────────

    #[test]
    fn every_station_verb_is_typed_by_its_acts_own_parameters() {
        let within = crowded();
        for (inv, tool) in performable_here(&within) {
            let body = body_of_url(&within, &inv.url);
            let declared: Vec<&str> = tool.params.iter().map(|p| p.name).collect();
            for field in body.properties.as_ref().expect("an object") {
                assert!(
                    declared.contains(&field.name.as_str()),
                    "{} invents a `{}` field",
                    inv.url,
                    field.name
                );
                let param = tool.params.iter().find(|p| p.name == field.name).unwrap();
                assert_eq!(field.required, param.required, "{}.{}", inv.url, field.name);
            }
            for param in tool.params.iter().filter(|p| p.required) {
                assert!(
                    names(&body).contains(&param.name),
                    "{} lost the required `{}`",
                    inv.url,
                    param.name
                );
            }
        }
    }
}
