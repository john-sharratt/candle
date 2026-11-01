//! Which tools cannot succeed right now, and what the model is told about them.
//!
//! A session tool with no session open can only answer `session_not_found`; a
//! worker invented session ids to try. So a tool the daemon's state already
//! rules out is taken out of the round's grammar, and the model is told why, in
//! one line beside the catalog listing that still names it: a tool that simply
//! vanished reads as one that never existed, and the model goes round it.
//!
//! **Only what the tool would refuse.** Each [`Rule`] names a state in which
//! every call to its tools fails whatever its arguments, so nothing that could
//! work is lost. The session stores are the daemon's, shared by every
//! conversation, so "open" means open anywhere: a rule can leave a tool that
//! will fail, and it never blocks one that would not.
//!
//! **The lines are fixed text, sealed at startup** beside the tool summary
//! ([`line_sections`]), one reserved section per rule, so showing a set of them
//! costs no prefill and no line ever carries a live value. A tool that comes
//! back during a turn is announced on the result that freed it
//! ([`announce_returned`]): the lines in view were projected before that round.

use std::collections::{BTreeSet, HashSet};

use candle_conversation::projection::{Reserved, SectionId};
use serde_json::{json, Value};
use zend_tools::state::OpenSessions;

use crate::tools::ToolResult;

/// What decides which tools are out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Facts {
    pub open: OpenSessions,
    pub search_key: bool,
}

/// Tools that cannot succeed in a state, and the line saying so.
struct Rule {
    tools: &'static [&'static str],
    line: &'static str,
    blocked: fn(&Facts) -> bool,
}

/// Every rule. Its index is its reserved section, so a new rule goes on the end.
const RULES: &[Rule] = &[
    Rule {
        tools: &["ssh_session_exec", "ssh_session_exec_async"],
        line: "ssh_session_exec and ssh_session_exec_async are unavailable until ssh_open has \
               opened a session.",
        blocked: |f| !f.open.ssh,
    },
    Rule {
        tools: &["ssh_session_poll"],
        line: "ssh_session_poll is unavailable until ssh_session_exec_async has started a \
               process.",
        blocked: |f| !f.open.ssh_processes,
    },
    Rule {
        tools: &["telnet_send"],
        line: "telnet_send is unavailable until telnet_session_open has opened a session.",
        blocked: |f| !f.open.telnet,
    },
    Rule {
        tools: &["http_request"],
        line: "http_request is unavailable until http_session_open has opened a session.",
        blocked: |f| !f.open.http,
    },
    Rule {
        tools: &["tcp_session_send", "tcp_session_recv"],
        line: "tcp_session_send and tcp_session_recv are unavailable until tcp_session_open has \
               opened a session.",
        blocked: |f| !f.open.tcp,
    },
    Rule {
        tools: &["udp_session_send", "udp_session_recv"],
        line: "udp_session_send and udp_session_recv are unavailable until udp_session_open has \
               opened a session.",
        blocked: |f| !f.open.udp,
    },
    Rule {
        tools: &["tls_session_send", "tls_session_recv"],
        line: "tls_session_send and tls_session_recv are unavailable until tls_session_open has \
               opened a session.",
        blocked: |f| !f.open.tls,
    },
    Rule {
        tools: &["sql_session_query"],
        line: "sql_session_query is unavailable until sql_session_open has opened a session.",
        blocked: |f| !f.open.sql,
    },
    Rule {
        tools: &[
            "remote_fs_session_stat",
            "remote_fs_session_rename",
            "remote_fs_session_put",
            "remote_fs_session_get",
            "remote_fs_session_delete",
            "remote_fs_session_list_dir",
            "remote_fs_session_mkdir",
        ],
        line: "remote_fs_session_stat, _rename, _put, _get, _delete, _list_dir and _mkdir are \
               unavailable until remote_fs_session_open has opened a session.",
        blocked: |f| !f.open.remote_fs,
    },
    Rule {
        tools: &["code_session_exec"],
        line: "code_session_exec is unavailable until code_session_open has opened a session.",
        blocked: |f| !f.open.code,
    },
    Rule {
        tools: &["web_search"],
        line: "web_search is unavailable: no search key is configured.",
        blocked: |f| !f.search_key,
    },
];

/// The rule at `index`'s reserved section.
fn section(index: usize) -> SectionId {
    SectionId::reserved(Reserved::ToolUnavailable(index as u8))
}

/// Every rule's line and its reserved section, to seal at startup.
pub fn line_sections() -> impl Iterator<Item = (SectionId, &'static str)> {
    RULES
        .iter()
        .enumerate()
        .map(|(i, rule)| (section(i), rule.line))
}

/// What is out of a round: the rules in force, by section, and their tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Unavailable {
    pub lines: Vec<SectionId>,
    pub tools: BTreeSet<&'static str>,
}

/// What `facts` rule out, among the tools in `offered`: a mode that does not
/// offer a tool has nothing to say about it.
pub fn unavailable(facts: &Facts, offered: &HashSet<String>) -> Unavailable {
    let mut out = Unavailable::default();
    for (i, rule) in RULES.iter().enumerate() {
        let tools = rule.tools.iter().filter(|t| offered.contains(**t)).copied();
        let tools: Vec<&'static str> = tools.collect();
        if (rule.blocked)(facts) && !tools.is_empty() {
            out.lines.push(section(i));
            out.tools.extend(tools);
        }
    }
    out
}

/// Tell the turn after `results` which tools came back in the round: its lines
/// were projected before the round that freed them. On the last result, which
/// stays what it was.
pub fn announce_returned(results: &mut [ToolResult], was: &Unavailable, now: &Unavailable) {
    let back: Vec<&str> = was.tools.difference(&now.tools).copied().collect();
    let (Some(last), false) = (results.last_mut(), back.is_empty()) else {
        return;
    };
    let notice = format!("Now available: {}.", back.join(", "));
    match &mut last.response {
        Value::Object(map) => {
            map.insert("tools_returned".to_string(), Value::String(notice));
        }
        other => {
            let result = other.take();
            *other = json!({"result": result, "tools_returned": notice});
        }
    }
}

#[cfg(test)]
mod tests {
    use zend_tools::registry;

    use super::*;
    use crate::tools::ToolCall;

    fn offered(names: &[&str]) -> HashSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    /// **Every tool a rule names is a real tool**, and its name is in its line,
    /// or a renamed tool would stop being withheld and go on being described.
    #[test]
    fn every_rule_names_real_tools_and_says_so() {
        for rule in RULES {
            for tool in rule.tools {
                assert!(registry::find(tool).is_some(), "{tool} is registered");
                let stem = tool.rsplit_once("session_").map_or(*tool, |(_, s)| s);
                assert!(rule.line.contains(stem), "{tool}: {}", rule.line);
            }
        }
        assert!(RULES.len() as u32 <= Reserved::TOOL_UNAVAILABLE_SLOTS);
    }

    /// **A session tool is out until its session is open**, and the tool that
    /// opens one never is: that is the way back.
    #[test]
    fn a_session_tool_is_out_until_a_session_is_open() {
        let tools = offered(&["ssh_open", "ssh_session_exec", "file_read"]);
        let closed = unavailable(&Facts::default(), &tools);
        assert_eq!(
            closed.tools,
            BTreeSet::from(["ssh_session_exec"]),
            "{closed:?}"
        );
        assert_eq!(closed.lines, [section(0)]);

        let mut facts = Facts::default();
        facts.open.ssh = true;
        let open = unavailable(&facts, &tools);
        assert!(open.tools.is_empty() && open.lines.is_empty(), "{open:?}");
    }

    /// A mode that does not offer a tool gets no line about it.
    #[test]
    fn a_tool_the_mode_does_not_offer_is_not_mentioned() {
        let none = unavailable(&Facts::default(), &offered(&["file_read"]));
        assert_eq!(none, Unavailable::default());
    }

    /// **Part of a rule's tools is enough for its line**, and only those are
    /// withheld.
    #[test]
    fn a_rule_withholds_only_what_the_mode_offers() {
        let some = unavailable(&Facts::default(), &offered(&["tcp_session_send"]));
        assert_eq!(some.tools, BTreeSet::from(["tcp_session_send"]));
        assert_eq!(some.lines.len(), 1);
    }

    /// **A tool that came back is said on the round that freed it**, the
    /// result left as it was; a round that freed nothing says nothing.
    #[test]
    fn a_tool_back_is_announced_on_the_last_result() {
        let tools = offered(&["ssh_session_exec"]);
        let was = unavailable(&Facts::default(), &tools);
        let mut facts = Facts::default();
        facts.open.ssh = true;
        let now = unavailable(&facts, &tools);
        let result = |response| ToolResult {
            call: ToolCall {
                name: "ssh_open".to_string(),
                arguments: json!({}),
            },
            response,
        };
        let mut results = vec![result(json!({"session_id": "s"}))];
        announce_returned(&mut results, &was, &now);
        assert_eq!(results[0].response["session_id"], "s");
        assert_eq!(
            results[0].response["tools_returned"],
            "Now available: ssh_session_exec."
        );

        let mut quiet = vec![result(json!({"session_id": "s"}))];
        announce_returned(&mut quiet, &now, &now);
        assert!(quiet[0].response.get("tools_returned").is_none());
    }
}
