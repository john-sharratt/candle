//! `lights_scene` and `lights_house` (§9, §14): a scene held on a group or
//! the whole track, and the house lights. Each hands its ask to the lights
//! thread and answers at once; nothing waits for a bulb.

use std::sync::mpsc;
use std::sync::Arc;

use arc_swap::ArcSwap;
use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::lights::{Ask, Scene, View};

/// The lights thread, as the tools reach it.
pub struct LightsHandle {
    pub asks: mpsc::Sender<Ask>,
    pub view: Arc<ArcSwap<View>>,
}

/// What a tool says with no bulbs mapped.
const NO_BULBS: &str = "no bulbs are mapped yet: zenj/lights.yaml is written once they are \
fitted and each has its address";

pub struct SceneTool;

impl Tool for SceneTool {
    const NAME: &'static str = "lights_scene";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Hold a lighting scene on the ceiling track, over what the set would choose, until it is let go: wash (one colour), split (two), chase (one bulb walking the track, a step a beat), pulse (bright on the beat), hit (one full-white flash), blackout, or house (warm white). `group` is a tag or an aim of the fixture map — floor, wall, decks, wash — or every bulb when left out. `auto` lets every held scene go. The lights follow the set only while rekordbox is open.",
            "inputSchema": {"type": "object", "properties": {
                "scene": {"type": "string", "enum": ["wash", "split", "chase", "pulse", "hit", "blackout", "house", "auto"]},
                "group": {"type": "string", "description": "A tag or aim of the fixture map; every bulb when left out."}
            }, "required": ["scene"]},
        })
    }
}

pub struct HouseTool;

impl Tool for HouseTool {
    const NAME: &'static str = "lights_house";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "The house lights: every bulb on the track warm white, over whatever the set wants (`on: true`), or back to the set (`on: false`).",
            "inputSchema": {"type": "object", "properties": {
                "on": {"type": "boolean"}
            }, "required": ["on"]},
        })
    }
}

fn ask(dj: &Dj, a: Ask, said: String) -> Answer {
    let Some(lights) = &dj.lights else {
        return Answer::error(NO_BULBS);
    };
    match lights.asks.send(a) {
        Ok(()) => Answer::ok(said),
        Err(_) => Answer::error("the lights are not running"),
    }
}

pub fn scene(dj: &Dj, args: &Value) -> Answer {
    let name = match text(args, "scene") {
        Ok(n) => n,
        Err(a) => return a,
    };
    let group = args
        .get("group")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .map(str::to_string);
    if name == "auto" {
        return ask(dj, Ask::Release, "the lights follow the set again".into());
    }
    let Some(scene) = Scene::parse(name) else {
        return Answer::error(
            "`scene` is wash, split, chase, pulse, hit, blackout, house, or auto",
        );
    };
    let on = group.clone().unwrap_or_else(|| "every bulb".into());
    ask(
        dj,
        Ask::Hold { scene, group },
        format!("{} held on {on}", scene.name()),
    )
}

pub fn house(dj: &Dj, args: &Value) -> Answer {
    let Some(on) = args.get("on").and_then(Value::as_bool) else {
        return Answer::error("`on` is true or false");
    };
    let said = match on {
        true => "the house lights are on",
        false => "the lights follow the set again",
    };
    ask(dj, Ask::House(on), said.into())
}

/// The lights, in a line, for `zenj_now`.
pub fn line(view: &View) -> String {
    if view.bulbs == 0 {
        return "Lights: none mapped".into();
    }
    let mut s = match view.driving {
        true => format!("Lights: {} on {} bulbs", view.scene, view.bulbs),
        false => format!(
            "Lights: the house's ({} bulbs; rekordbox is closed)",
            view.bulbs
        ),
    };
    if !view.dead.is_empty() {
        s.push_str(&format!("; not answering: {}", view.dead.join(", ")));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fixture::dj;

    /// **With no bulbs mapped the tools say so**; mapped, a scene is held on
    /// its group and the house lights are asked for.
    #[test]
    fn the_tools_hand_their_asks_to_the_lights() {
        let (mut d, _) = dj();
        assert_eq!(
            scene(&d, &json!({"scene": "wash"})),
            Answer::error(NO_BULBS)
        );
        let (tx, rx) = mpsc::channel();
        d.lights = Some(LightsHandle {
            asks: tx,
            view: Arc::new(ArcSwap::from_pointee(View::default())),
        });
        assert_eq!(
            scene(&d, &json!({"scene": "chase", "group": "wall"})),
            Answer::ok("chase held on wall")
        );
        assert_eq!(
            rx.try_recv().unwrap(),
            Ask::Hold {
                scene: Scene::Chase,
                group: Some("wall".into())
            }
        );
        assert_eq!(
            scene(&d, &json!({"scene": "auto"})),
            Answer::ok("the lights follow the set again")
        );
        assert_eq!(rx.try_recv().unwrap(), Ask::Release);
        assert!(scene(&d, &json!({"scene": "strobe"})).is_error);
        assert_eq!(
            house(&d, &json!({"on": true})),
            Answer::ok("the house lights are on")
        );
        assert_eq!(rx.try_recv().unwrap(), Ask::House(true));
        assert!(house(&d, &json!({})).is_error);
    }

    #[test]
    fn the_lights_are_said_in_a_line() {
        assert_eq!(line(&View::default()), "Lights: none mapped");
        let v = View {
            bulbs: 20,
            driving: true,
            scene: "pulse".into(),
            dead: vec!["t7".into()],
            ..View::default()
        };
        assert_eq!(line(&v), "Lights: pulse on 20 bulbs; not answering: t7");
        let idle = View {
            bulbs: 20,
            ..View::default()
        };
        assert_eq!(
            line(&idle),
            "Lights: the house's (20 bulbs; rekordbox is closed)"
        );
    }
}
