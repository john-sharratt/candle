//! Which conversation a `get_user_context` begins or goes on with
//! (`docs/private_mode.md` §3): the ref it was given, while that is live; a
//! new one, keeping its private flag, when it has expired; a new conversation
//! when it was given none. A ref that does not open for this caller is
//! refused, never replaced — a forgery begins nothing.

use mcp_server::refs::{Ref, Refs, OPENING};

/// The conversation a `get_user_context` answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Begun {
    /// Its ref, as written.
    pub text: String,
    pub r: Ref,
    /// How it came to be.
    pub how: How,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    /// The ref asked with, still live.
    Same,
    /// A new ref for the conversation whose ref had expired.
    Renewed,
    /// A new conversation.
    New,
}

/// What a forged or malformed ref is answered.
pub const NOT_OURS: &str = "that `ref` is not one this server issued to you: call \
    get_user_context without `ref` to begin a conversation";

/// The conversation `owner`'s `get_user_context` at `now_ms` answers, given
/// the ref it named and asking private or not; `used` says when a ref was
/// last told used.
pub fn begin(
    refs: &Refs,
    owner: &str,
    named: Option<&str>,
    private: bool,
    used: impl Fn(&Ref) -> Option<u64>,
    now_ms: u64,
) -> Result<Begun, String> {
    let Some(text) = named.map(str::trim).filter(|t| !t.is_empty()) else {
        let (text, r) = refs.issue(owner, private, now_ms);
        return Ok(Begun {
            text,
            r,
            how: How::New,
        });
    };
    let r = refs.open(owner, text).map_err(|_| NOT_OURS.to_string())?;
    if r.live(used(&r), now_ms) {
        return Ok(Begun {
            text: text.to_string(),
            r,
            how: How::Same,
        });
    }
    // A private conversation never comes back normal.
    let (text, r) = refs.issue(owner, r.private || private, now_ms);
    Ok(Begun {
        text,
        r,
        how: How::Renewed,
    })
}

/// What a private conversation's answer says of it, after its ref.
pub const PRIVATE_LINE: &str = "This conversation is private: Zen's memory is read-only in it \
    to its end — nothing learned here is written to it, and every tool that would write, message, \
    run or draw is refused.";

/// The lines the answer opens with: the ref, that every call passes it, and
/// — for a private conversation — that it is one.
pub fn ref_line(b: &Begun) -> String {
    let renewed = match b.how {
        How::Renewed => " (the ref this conversation had expired; this one replaces it)",
        How::Same | How::New => "",
    };
    let line = format!(
        "{OPENING}{}`{renewed}. Pass it as `ref` on every Zen call in this conversation, \
         get_user_context included: a call without it is refused.",
        b.text
    );
    match b.r.private {
        true => format!("{line}\n{PRIVATE_LINE}"),
        false => line,
    }
}

#[cfg(test)]
mod tests {
    use mcp_server::refs::REF_IDLE_MS;

    use super::*;

    const NOW: u64 = 1_794_000_000_000;

    fn refs() -> Refs {
        Refs::new([3; 32])
    }

    fn never(_: &Ref) -> Option<u64> {
        None
    }

    #[test]
    fn no_ref_begins_a_conversation_private_or_not() {
        let b = begin(&refs(), "o", None, false, never, NOW).unwrap();
        assert_eq!(b.how, How::New);
        assert!(!b.r.private);
        assert_eq!(refs().open("o", &b.text), Ok(b.r));
        let p = begin(&refs(), "o", Some("  "), true, never, NOW).unwrap();
        assert_eq!(p.how, How::New);
        assert!(p.r.private);
    }

    /// **A live ref answers itself, whatever `private_mode` asks**: a
    /// conversation's flag is set once, when it begins.
    #[test]
    fn a_live_ref_answers_the_same_ref_and_keeps_its_flag() {
        let normal = begin(&refs(), "o", None, false, never, NOW).unwrap();
        let again = begin(&refs(), "o", Some(&normal.text), true, never, NOW + 1).unwrap();
        assert_eq!(again.how, How::Same);
        assert_eq!(again.text, normal.text);
        assert!(
            !again.r.private,
            "private_mode cannot change a conversation"
        );
        let private = begin(&refs(), "o", None, true, never, NOW).unwrap();
        let again = begin(&refs(), "o", Some(&private.text), false, never, NOW + 1).unwrap();
        assert_eq!(again.text, private.text);
        assert!(again.r.private, "nor turn private mode off");
    }

    /// **An expired ref is renewed, and a private one stays private**; one
    /// kept in use is live long past its issue.
    #[test]
    fn an_expired_ref_is_renewed_keeping_private() {
        let later = NOW + REF_IDLE_MS;
        let private = begin(&refs(), "o", None, true, never, NOW).unwrap();
        let renewed = begin(&refs(), "o", Some(&private.text), false, never, later).unwrap();
        assert_eq!(renewed.how, How::Renewed);
        assert_ne!(renewed.text, private.text);
        assert!(
            renewed.r.private,
            "an expired private ref never falls back to normal"
        );
        let normal = begin(&refs(), "o", None, false, never, NOW).unwrap();
        let asked = begin(&refs(), "o", Some(&normal.text), true, never, later).unwrap();
        assert!(asked.r.private, "a renewal may begin private mode");
        let used = |_: &Ref| Some(later - 1);
        let kept = begin(&refs(), "o", Some(&normal.text), false, used, later).unwrap();
        assert_eq!(kept.how, How::Same, "used within the four hours");
    }

    /// **Another account's ref, or one not ours, begins nothing**.
    #[test]
    fn a_ref_not_ours_is_refused() {
        let b = begin(&refs(), "o", None, false, never, NOW).unwrap();
        assert_eq!(
            begin(&refs(), "other", Some(&b.text), false, never, NOW),
            Err(NOT_OURS.into())
        );
        assert_eq!(
            begin(&refs(), "o", Some("nonsense"), false, never, NOW),
            Err(NOT_OURS.into())
        );
    }

    #[test]
    fn the_answer_opens_with_the_ref() {
        let b = Begun {
            text: "R".into(),
            r: begin(&refs(), "o", None, false, never, NOW).unwrap().r,
            how: How::New,
        };
        assert_eq!(
            ref_line(&b),
            "This conversation's ref is `R`. Pass it as `ref` on every Zen call in this \
             conversation, get_user_context included: a call without it is refused."
        );
        let renewed = Begun {
            how: How::Renewed,
            ..b.clone()
        };
        assert!(ref_line(&renewed).starts_with(
            "This conversation's ref is `R` (the ref this conversation had expired; this one \
             replaces it)."
        ));
        let private = Begun {
            r: begin(&refs(), "o", None, true, never, NOW).unwrap().r,
            ..b
        };
        assert_eq!(
            ref_line(&private),
            format!(
                "This conversation's ref is `R`. Pass it as `ref` on every Zen call in this \
                 conversation, get_user_context included: a call without it is refused.\n\
                 {PRIVATE_LINE}"
            )
        );
    }
}
