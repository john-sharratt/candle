//! The acts a personality's `reaches:` offers (`docs/zen_moon_mind.md` §16),
//! each built from its tool's own definition (`zend_tools::definitions`) — its
//! name, category, description and parameters, and its examples from the
//! definition's trajectories — so a tool is described in one place.
//!
//! What the definition says is kept whole; what changes is how each parameter
//! travels through the grammar ([`super::carry`]), which its description then
//! says.

use std::sync::LazyLock;

use serde_json::{Map, Value};
use zend_tools::definitions::{self, Definition};

use super::carry::{carried, said_as, there, Carry};
use super::Group;
use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

/// One parameter of an act: what the act calls it, what its tool calls it,
/// and how it is carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carried {
    pub said: &'static str,
    pub param: &'static str,
    pub carry: Carry,
}

/// One act, and how each of its parameters is carried.
pub struct Reached {
    pub tool: Tool,
    pub definition: &'static Definition,
    pub carries: Vec<Carried>,
}

/// Every act a personality can reach, in [`Group::ALL`]'s order.
pub static REACHED: LazyLock<Vec<Reached>> = LazyLock::new(|| {
    Group::ALL
        .iter()
        .flat_map(|g| g.acts())
        .map(|name| {
            let def = definitions::find(name)
                .unwrap_or_else(|| panic!("`{name}` has no tool definition to be reached by"));
            reached(def)
        })
        .collect()
});

/// The catalogue's entries for them.
pub static BEYOND_ACTS: LazyLock<Vec<Tool>> =
    LazyLock::new(|| REACHED.iter().map(|r| r.tool).collect());

/// The act `name`, if a personality can reach it.
pub fn of(name: &str) -> Option<&'static Reached> {
    REACHED.iter().find(|r| r.tool.name == name)
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// The act `def` describes.
fn reached(def: &'static Definition) -> Reached {
    static NONE: LazyLock<Map<String, Value>> = LazyLock::new(Map::new);
    let properties = def.parameters["properties"].as_object().unwrap_or(&NONE);
    let required: Vec<&str> = def.parameters["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut params = Vec::new();
    let mut carries = Vec::new();
    for (name, schema) in properties {
        let (ty, carry) = carried(schema);
        let described = schema["description"].as_str().unwrap_or_default();
        let said = said_as(def.name.as_str(), name.as_str());
        params.push(Param {
            name: said,
            ty,
            required: required.contains(&name.as_str()),
            description: leak(format!("{described}{}", carry.note())),
        });
        carries.push(Carried {
            said,
            param: name.as_str(),
            carry,
        });
    }
    let examples: Vec<Example> = def
        .examples
        .iter()
        .filter_map(|t| example(t, &carries))
        .collect();
    Reached {
        tool: Tool {
            name: def.name.as_str(),
            at: &[],
            category: def.category.as_str(),
            plane: Plane::World,
            availability: Availability::Reached,
            description: def.description.as_str(),
            params: Box::leak(params.into_boxed_slice()),
            examples: Box::leak(examples.into_boxed_slice()),
        },
        definition: def,
        carries,
    }
}

/// The marker a calibration trajectory carries between its parts.
const MARKER: &str = "<|projection|>";

/// One of a definition's trajectories as an act's example: the question it
/// opens with, the reasoning in its thinking, and its call's arguments as the
/// act carries them. A bare question, with no call after it, is none; so is a
/// call passing an argument its tool does not declare, which would teach the
/// act a parameter it has not got.
fn example(trajectory: &str, carries: &[Carried]) -> Option<Example> {
    let (asked, answer) = trajectory.split_once("<|im_end|>")?;
    let thinking = answer.split_once("<think>")?.1.split_once("</think>")?.0;
    let call = answer
        .split_once("<tool_call>")?
        .1
        .split_once("</tool_call>")?
        .0;
    let call: Value = serde_json::from_str(call.replace(MARKER, "").trim()).ok()?;
    let given = call.get("arguments")?.as_object()?;
    let args = given
        .iter()
        .map(|(k, v)| {
            let c = carries.iter().find(|c| c.param == k)?;
            Some((c.said.to_string(), there(c.carry, v)))
        })
        .collect::<Option<Map<String, Value>>>()?;
    let because = thinking.replace(MARKER, "");
    let (situation, because) = (asked.trim(), because.trim());
    if situation.is_empty() || because.is_empty() {
        return None;
    }
    Some(Example {
        situation: leak(situation.to_string()),
        call: leak(Value::Object(args).to_string()),
        because: leak(because.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tools::LIST;

    /// **Every act a personality can reach has its tool's definition**, its
    /// name, description and parameters as the definition says them, and at
    /// least one example taken from the definition's trajectories.
    #[test]
    fn every_reached_act_is_its_tools_definition() {
        for r in REACHED.iter() {
            let def = definitions::find(r.tool.name).unwrap();
            assert_eq!(r.tool.description, def.description);
            assert_eq!(r.tool.category, def.category);
            assert_eq!(r.tool.availability, Availability::Reached);
            assert!(
                !r.tool.examples.is_empty(),
                "{} has no example",
                r.tool.name
            );
            let names: Vec<&str> = r.carries.iter().map(|c| c.param).collect();
            let declared: Vec<&str> = def.parameters["properties"]
                .as_object()
                .map(|p| p.keys().map(String::as_str).collect())
                .unwrap_or_default();
            assert_eq!(names, declared, "{}", r.tool.name);
            let said: Vec<&str> = r.tool.params.iter().map(|p| p.name).collect();
            let carried: Vec<&str> = r.carries.iter().map(|c| c.said).collect();
            assert_eq!(said, carried, "{}", r.tool.name);
        }
    }

    /// **A parameter is carried as the grammar bounds it, and says how**: a
    /// number as a string, an object of strings as lines, a list as a list.
    #[test]
    fn a_parameter_is_carried_and_says_how() {
        let param = |act: &str, name: &str| {
            *of(act)
                .unwrap()
                .tool
                .params
                .iter()
                .find(|p| p.name == name)
                .unwrap()
        };
        let value = param("unit_convert", "value");
        assert_eq!((value.ty, value.required), ("string", true));
        assert!(value
            .description
            .ends_with("Written as a number in a string: \"72.5\"."));
        let headers = param("http_request", "headers");
        assert_eq!((headers.ty, headers.required), ("string", false));
        assert!(headers
            .description
            .ends_with("Written one `name: value` to a line."));
        assert_eq!(param("random", "choices").ty, LIST);
        assert_eq!(param("weather", "forecast_days").ty, "integer");
        assert_eq!(
            param("weather", "location").description,
            definitions::find("weather").unwrap().parameters["properties"]["location"]
                ["description"]
                .as_str()
                .unwrap()
        );
    }

    /// **An example is the trajectory's question, its thinking and its call**,
    /// the call's values written as the act carries them.
    #[test]
    fn an_example_is_the_trajectorys_question_thinking_and_call() {
        let carries = [
            Carried {
                said: "value",
                param: "value",
                carry: Carry::Number,
            },
            Carried {
                said: "from",
                param: "from",
                carry: Carry::Itself,
            },
        ];
        let t = "convert 72 F<|im_end|>\n<|im_start|>assistant\n<|projection|><think>\nThe \
                 user wants a conversion.<|projection|> Use unit_convert.\n</think>\n\n\
                 <tool_call>\n<|projection|>{\"name\": \"unit_convert\", \"arguments\": \
                 {\"value\": 72, \"from\": \"fahrenheit\"}}\n</tool_call><|projection|>";
        let e = example(t, &carries).unwrap();
        assert_eq!(e.situation, "convert 72 F");
        assert_eq!(e.because, "The user wants a conversion. Use unit_convert.");
        assert_eq!(e.call, r#"{"value":"72","from":"fahrenheit"}"#);
        assert!(example("a bare question<|im_end|>", &carries).is_none());
        let stray = t.replace("\"from\"", "\"type\"");
        assert!(
            example(&stray, &carries).is_none(),
            "an undeclared argument"
        );
    }

    /// **A parameter the act's line spends is carried under its tool's first
    /// word**, and its examples call it so.
    #[test]
    fn a_spent_parameter_is_carried_under_another_name() {
        let guide = of("guide_read").unwrap();
        assert_eq!(guide.tool.params[0].name, "guide");
        assert_eq!(
            guide.carries[0],
            Carried {
                said: "guide",
                param: "name",
                carry: Carry::Itself
            }
        );
        assert_eq!(guide.tool.examples[0].call, r#"{"guide":"deploy"}"#);
    }
}
