//! Who is watching a program run, for as long as a tool round lasts.
//!
//! `run_command` blocks its tool call until the program ends, which can be many
//! minutes in which a turn shows nothing but the tool's name. The turn that
//! starts a round names a watcher for the thread the round runs on, and whatever
//! runs the program reports through it as it goes. A watcher is told for its own
//! sake: nothing it hears reaches the model, whose turn stays one sealed result.

use std::cell::RefCell;
use std::sync::Arc;

/// Told how a running program stands, in a line.
pub type RunWatch = Arc<dyn Fn(String) + Send + Sync>;

thread_local! {
    static WATCH: RefCell<Option<RunWatch>> = const { RefCell::new(None) };
}

/// Run `f` with `watch` watching this thread's programs.
pub fn watching<R>(watch: RunWatch, f: impl FnOnce() -> R) -> R {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            WATCH.with(|w| *w.borrow_mut() = None);
        }
    }
    WATCH.with(|w| *w.borrow_mut() = Some(watch));
    let _reset = Reset;
    f()
}

/// The watcher of this thread's programs, if it has one.
pub fn current() -> Option<RunWatch> {
    WATCH.with(|w| w.borrow().clone())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// A watcher lasts as long as its round, on its thread alone.
    #[test]
    fn a_watcher_stands_for_its_round_on_its_thread() {
        assert!(current().is_none());
        let heard = Arc::new(Mutex::new(Vec::new()));
        let sink = heard.clone();
        let watch: RunWatch = Arc::new(move |line| sink.lock().unwrap().push(line));
        watching(watch, || {
            current().expect("watching")("12 lines".into());
            let other = std::thread::spawn(|| current().is_none());
            assert!(other.join().unwrap(), "another thread has no watcher");
        });
        assert!(current().is_none(), "gone with the round");
        assert_eq!(*heard.lock().unwrap(), ["12 lines"]);
    }
}
