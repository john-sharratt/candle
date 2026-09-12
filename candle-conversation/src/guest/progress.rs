//! Watching a job while it runs, rather than only when it finishes.
//!
//! A guest drain is seconds long — evict, load a model across the PCIe link,
//! decode, hand the ground back — and for all of that a caller holding only a
//! [`super::GuestReceipt`] has nothing to show but a spinner. The prose it is
//! waiting for exists a token at a time well before the job ends, and a reader
//! reads at about that rate anyway.
//!
//! # Why a callback and not a channel
//!
//! The obvious shape is a `Sender` on the job. It would decide, here, which
//! channel every consumer uses: a `crossbeam` sender cannot be awaited by an
//! HTTP handler without a thread to bridge it, and a `tokio` one would put an
//! async runtime into the engine crate for the sake of a progress line.
//!
//! A callback decides nothing. The scheduler thread calls it inline; whoever
//! submitted the job wrote it, and knows whether that means a channel, a log
//! line, or a websocket. `npcd` pushes into a `tokio` sender it already had.
//!
//! # What a sink may not do
//!
//! It is called **on the scheduler thread, between decode steps, while normal
//! inference is blocked**. It must not block: a sink that waits on a full
//! channel stops the guest, and the engine behind it, for as long as it waits.
//! Push and return.

/// Something worth telling a waiting caller before its job is done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuestEvent {
    /// The engine has stopped, the ground is claimed, and the guest's weights
    /// are crossing the link. Seconds, and the only part of a drain with
    /// nothing to show — so it is worth naming rather than leaving blank.
    Loading,
    /// A decoded fragment, in order. Fragments concatenate to the final text
    /// exactly; nothing is re-sent and nothing is a correction of what came
    /// before, so a consumer may append and never rewrite.
    ///
    /// A fragment is whatever one token detokenised to, which is not a word and
    /// may not be valid UTF-8 on its own — the guest holds partial sequences
    /// back until they complete rather than emitting a replacement character.
    Token(String),
    /// Countable work, for a job whose output does not exist until it ends.
    ///
    /// Prose can show itself arriving; an image cannot — the picture is mush
    /// until the last denoise step and does not exist as pixels until the
    /// decoder runs. `done` of `total` is the only honest thing there is to say
    /// while that happens, and it is enough to draw a bar with.
    ///
    /// **`total` counts every unit the job will report, not just the loop's.**
    /// A guest whose tail is one long operation counts that operation as a unit,
    /// so the bar does not reach the end and then sit there — see the image
    /// guest, which counts its decode alongside its denoise steps.
    ///
    /// `what` names the phase the count is currently in, for a caller that shows
    /// a line as well as a bar. It is `&'static str` rather than a `String`
    /// because a phase is a fixed part of the guest's shape, and because this is
    /// emitted from the scheduler thread where an allocation per step is worth
    /// not making.
    Step {
        done: u32,
        total: u32,
        what: &'static str,
    },
}

/// Where a job's progress goes while it runs, if anywhere.
///
/// Most jobs have no watcher and carry [`GuestSink::none`], which costs one
/// `Option` check per token and nothing else.
#[derive(Default)]
pub struct GuestSink(Option<Box<dyn Fn(GuestEvent) + Send + Sync>>);

impl GuestSink {
    /// A sink nobody is watching.
    pub fn none() -> Self {
        Self(None)
    }

    pub fn new(f: impl Fn(GuestEvent) + Send + Sync + 'static) -> Self {
        Self(Some(Box::new(f)))
    }

    /// Whether anything is listening.
    ///
    /// Worth checking before *building* an event that costs something — a
    /// detokenised fragment is an allocation, and there is no point making one
    /// for nobody.
    pub fn is_watched(&self) -> bool {
        self.0.is_some()
    }

    /// Report progress. Does nothing when nobody is watching.
    pub fn emit(&self, event: GuestEvent) {
        if let Some(f) = &self.0 {
            f(event);
        }
    }
}

impl std::fmt::Debug for GuestSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GuestSink")
            .field(&self.is_watched())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn an_unwatched_sink_swallows_everything() {
        let s = GuestSink::none();
        assert!(!s.is_watched());
        s.emit(GuestEvent::Loading);
        s.emit(GuestEvent::Token("x".into()));
        s.emit(GuestEvent::Step {
            done: 1,
            total: 9,
            what: "denoising",
        });
    }

    /// **A step count never exceeds its total and never goes backwards.**
    ///
    /// Both are what a bar is drawn from directly: `done/total` past 1.0
    /// overflows the track, and a count that retreats reads as the job having
    /// lost work it already did. The image guest's shape is the one under test
    /// — steps then a decode, all against the same `total`.
    #[test]
    fn a_step_count_rises_to_its_total_and_no_further() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = Arc::clone(&seen);
            GuestSink::new(move |e| {
                if let GuestEvent::Step { done, total, what } = e {
                    seen.lock().unwrap().push((done, total, what));
                }
            })
        };

        let steps = 8u32;
        let total = steps + 1;
        for done in 1..=steps {
            sink.emit(GuestEvent::Step {
                done,
                total,
                what: "denoising",
            });
        }
        sink.emit(GuestEvent::Step {
            done: steps,
            total,
            what: "decoding",
        });

        let seen = seen.lock().unwrap();
        let mut high = 0;
        for (done, t, _) in seen.iter() {
            assert_eq!(*t, total, "the total moved mid-job");
            assert!(*done <= *t, "{done} of {t} is past the end");
            assert!(*done >= high, "the count went backwards at {done}");
            high = *done;
        }
        // The decode is announced on entry, so the last count is one short of
        // the total. The event that completes the bar is the image itself.
        assert_eq!(seen.last().map(|(d, _, w)| (*d, *w)), Some((8, "decoding")));
    }

    /// Events reach the watcher in the order they were emitted — a consumer
    /// appends fragments and never reorders them, so out-of-order delivery
    /// would be scrambled prose rather than late prose.
    #[test]
    fn a_watched_sink_receives_in_order() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = Arc::clone(&seen);
            GuestSink::new(move |e| seen.lock().unwrap().push(e))
        };
        assert!(sink.is_watched());
        sink.emit(GuestEvent::Loading);
        for t in ["Ael", "is", " Mael"] {
            sink.emit(GuestEvent::Token(t.into()));
        }
        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                GuestEvent::Loading,
                GuestEvent::Token("Ael".into()),
                GuestEvent::Token("is".into()),
                GuestEvent::Token(" Mael".into()),
            ]
        );
    }

    /// **The fragments must concatenate to the whole text.** A consumer builds
    /// the result by appending, so a guest that emitted a running prefix each
    /// time would produce "AAeAelAeli…" on the screen and the right answer in
    /// the final response — a discrepancy nobody would look for.
    #[test]
    fn fragments_concatenate_to_the_text() {
        let built = Arc::new(Mutex::new(String::new()));
        let sink = {
            let built = Arc::clone(&built);
            GuestSink::new(move |e| {
                if let GuestEvent::Token(t) = e {
                    built.lock().unwrap().push_str(&t);
                }
            })
        };
        let whole = "Aelis Maelstrom, of House Cyclone";
        for chunk in ["Ael", "is", " Maelstrom", ",", " of House", " Cyclone"] {
            sink.emit(GuestEvent::Token(chunk.into()));
        }
        assert_eq!(*built.lock().unwrap(), whole);
    }
}
