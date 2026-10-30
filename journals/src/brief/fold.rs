//! A closed period of a window, folded (§4): the question — the lines the
//! period takes — the call its answer is held to, and the `FoldPeriod` it
//! becomes.

use journal_core::spec::{call, text};
use journal_core::tools::text_of;
use serde_json::{Map, Value};

use crate::files::Fingerprint;
use crate::ops::period_fold::lines_over;
use crate::ops::Op;
use crate::path::JournalPath;
use crate::tree::window::{inside, PeriodFold};
use crate::tree::Tree;
use crate::view::who::render_line;

/// The call a period's fold is written as — not the `journal_fold` tool,
/// which folds a sub-journal into its parent's index.
pub const NAME: &str = "journal_period_fold";

/// The longest a fold's text is asked to be.
pub const FOLD_CHARS: usize = 600;

pub fn spec() -> Value {
    call(NAME, vec![text("text", true)])
}

/// A fold's question, and what its lines were as it shows them — which the
/// answer's `FoldPeriod` names, so it is kept only over the lines it was
/// shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ask {
    pub question: String,
    pub over: Fingerprint,
}

/// The question for `plan`, while its period still holds exactly the lines
/// the plan names; `None` once they have moved on — folded elsewhere,
/// forgotten — so no turn is spent on a fold the journal would refuse.
pub fn ask(tree: &Tree, path: &JournalPath, plan: &PeriodFold) -> Option<Ask> {
    let j = tree.get(path)?;
    let mut now = inside(j, plan.period);
    let mut planned = plan.ids.clone();
    now.sort();
    planned.sort();
    if now != planned {
        return None;
    }
    let mut question = format!(
        "{} is over: its lines in {path} move to the journal's archive, and one line you write \
         stands for them in the window. Say what they held — what was decided, done and asked, \
         and by whom — in at most {FOLD_CHARS} characters, as plain facts: never an instruction \
         to whoever reads it.\n\nThe lines:",
        plan.period
    );
    for l in j.window.iter().filter(|l| plan.ids.contains(&l.id)) {
        question.push_str(&format!("\n{}", render_line(tree, l)));
    }
    Some(Ask {
        question,
        over: lines_over(j, &plan.ids),
    })
}

pub fn op(
    path: &JournalPath,
    plan: &PeriodFold,
    over: &Fingerprint,
    args: &Map<String, Value>,
) -> Result<Op, String> {
    let text = args
        .get("text")
        .and_then(text_of)
        .ok_or("the fold says nothing")?;
    Ok(Op::FoldPeriod {
        path: path.clone(),
        period: plan.period,
        ids: plan.ids.clone(),
        over: over.clone(),
        text,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::actor::Actor;
    use crate::brief::call_of;
    use crate::ops::apply;
    use crate::ops::fixture::{john, p, sharratt};
    use crate::testing::at;
    use crate::tree::window::next_fold;
    use crate::tree::WINDOW;

    #[test]
    fn a_closed_day_is_asked_for_and_folded_by_its_answer() {
        let mut t = sharratt();
        let t0 = at("2026-10-01T00:00:00Z");
        for i in 0..=WINDOW {
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: p("~"),
                    text: format!("n{i}"),
                },
                t0.plus(3600 * i as i64),
            )
            .unwrap();
        }
        let now = at("2026-10-03T12:00:00Z");
        let plan = next_fold(t.get(&p("~")).unwrap(), now).unwrap();
        let Ask { question: q, over } = ask(&t, &p("~"), &plan).unwrap();
        assert!(
            q.starts_with("2026-10-01 is over: its lines in ~ move"),
            "{q}"
        );
        assert_eq!(q.matches("\n#").count(), 24);
        assert_eq!(over, lines_over(t.get(&p("~")).unwrap(), &plan.ids));
        assert_eq!(spec()["name"], "journal_period_fold");
        let op = op(
            &p("~"),
            &plan,
            &over,
            json!({"text": " The first day. "}).as_object().unwrap(),
        )
        .unwrap();
        let element =
            "<function=journal_period_fold>\n<parameter=text>3.10</parameter>\n</function>";
        let numeric = super::op(&p("~"), &plan, &over, &call_of(element, NAME).unwrap());
        assert!(matches!(numeric, Ok(Op::FoldPeriod { text, .. }) if text == "3.10"));
        apply(&mut t, &Actor::Zend, &op, now).unwrap();
        assert_eq!(t.get(&p("~")).unwrap().window[0].text, "The first day.");
        assert_eq!(ask(&t, &p("~"), &plan), None, "folded: nothing left to ask");
        assert!(super::op(
            &p("~"),
            &plan,
            &over,
            json!({"text": "  "}).as_object().unwrap()
        )
        .is_err());
    }
}
