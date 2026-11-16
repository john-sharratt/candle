//! One authored example as the turn a character would have decoded.

use candle_conversation::models::Dialect;
use host::calibration::PROJECTION_MARKER;

use crate::engine::tools::Example;

/// `example` as a calibration trajectory: the situation as what the character
/// faced, then — after the dialect's assistant header — the reasoning that
/// picks this act and the call it makes.
///
/// The reasoning is the example's `because`: why this call and not a
/// neighbouring one, which is the discriminating detail a signature is wanted
/// for. A projection fires where the reasoning opens and closes and on either
/// side of the call, the points a live decode reprojects at.
pub fn render(act: &str, example: &Example, dialect: &Dialect) -> String {
    format!(
        "{situation}{user_end}{assistant_start}\n\
         {m}<think>\n{because}{m}\n</think>\n\n\
         <tool_call>\n{m}{{\"name\": \"{act}\", \"arguments\": {call}}}\n</tool_call>{m}",
        situation = example.situation,
        user_end = dialect.user_end,
        assistant_start = dialect.assistant_start,
        m = PROJECTION_MARKER,
        because = example.because,
        call = example.call,
    )
}

#[cfg(test)]
mod tests {
    use candle_conversation::models::Dialect;

    use super::render;
    use crate::engine::tools::Example;

    #[test]
    fn an_example_reads_as_a_situation_a_reason_and_a_call() {
        let dialect = Dialect::chat_ml();
        let example = Example {
            situation: "You have not looked at the cluster since morning.",
            call: "{}",
            because: "A member that has fallen out says nothing.",
        };
        assert_eq!(
            render("cluster_status", &example, &dialect),
            format!(
                "You have not looked at the cluster since morning.{}{}\n\
                 <|projection|><think>\n\
                 A member that has fallen out says nothing.<|projection|>\n\
                 </think>\n\n\
                 <tool_call>\n\
                 <|projection|>{{\"name\": \"cluster_status\", \"arguments\": {{}}}}\n\
                 </tool_call><|projection|>",
                dialect.user_end, dialect.assistant_start
            )
        );
    }

    #[test]
    fn a_trajectory_has_exactly_one_assistant_header() {
        let dialect = Dialect::chat_ml();
        let example = Example {
            situation: "It is late.",
            call: r#"{"terminal":"the lights"}"#,
            because: "A thing is worked through its terminal.",
        };
        let t = render("terminal_connect", &example, &dialect);
        assert_eq!(t.matches(dialect.assistant_start).count(), 1);
    }
}
