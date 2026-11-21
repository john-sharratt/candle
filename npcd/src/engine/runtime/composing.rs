//! Sitting a character down to write its mission's piece in one sitting.
//!
//! A piece that is committed is not composed again, and a piece that already
//! stands is mended by its paragraphs' anchors rather than written over. The
//! basis for a sitting is the piece as the operation's store holds it, never the
//! character's own working copy, so a draft a restart left uncommitted is not
//! lost to a fresh write. A piece written back unchanged is reported as no change,
//! so the next sitting is not let through on nothing.

use super::*;

impl Runtime {
    /// Enact `compose`: the character sits down and writes its mission's piece
    /// in one sitting. See [`compose`].
    ///
    /// The mind's writing voice is put the brief, every document the mission
    /// sent the character to read, and the piece as it stands when there is
    /// one — in a conversation opened for the sitting, not on the character's
    /// own — and answers with the whole piece; that goes into the character's
    /// working set through `file_write` — the same act, held to the same checks
    /// — and the character is told to read it back, mend it and commit it.
    pub(crate) async fn compose(&self, npc_id: u64, act: &Act) -> Recorded {
        let refused = |why: String| Recorded {
            feed: format!("{} {REFUSED} {why}", act.summary()),
            answer: why,
            landed: false,
            departing: false,
        };
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return refused("There is nothing here to write at.".into());
        };
        // **A committed piece is not sat down to again.** A reviewer whose
        // write step was done and committed — told to go and report — sat down
        // and composed the piece afresh on her way out, and left the new copy
        // uncommitted in her working set.
        let committed = hosted.sim(|s| {
            let m = s.missions.active(&body)?;
            let w = m.work.as_ref()?;
            (!w.edit_optional && m.written_up()).then(|| w.writes.clone())
        });
        if let Some(writes) = committed {
            return refused(compose::already_committed(&writes));
        }
        let job = hosted.sim(|s| {
            let m = s.missions.active(&body)?;
            let w = m.work.as_ref()?;
            let root = s.bench.files()?;
            let written = written_up_to_floor(s, &body, m, &DocStep::Write(w.writes.clone()));
            Some((
                m.prompt.clone(),
                w.writes.clone(),
                w.reads.clone(),
                w.min_words,
                root,
                w.anew,
                written,
            ))
        });
        let Some((brief, writes, reads, min_words, root, anew, written)) = job else {
            return refused("You carry no piece of work to write.".into());
        };
        // **A written piece is not written over.** Back from a restart with
        // its piece composed and uncommitted, a Maker sat straight down and
        // composed it again from nothing, every time.
        if let Some(words) = written {
            return refused(compose::already_written(&writes, words));
        }
        // The piece as the operation's store has it — what the table read —
        // never the Maker's own working copy, which holds whatever it last
        // typed by hand; and nothing when the piece is to be written anew.
        let shelf = hosted.shelf_of(&body);
        let standing = match anew {
            true => None,
            false => shelf.read(&writes).map(|t| strip_calls(&t)),
        };
        let sources: Vec<(String, String)> = reads
            .iter()
            .filter(|p| **p != writes)
            .filter_map(|p| {
                let text = shelf.read(p)?;
                Some((
                    p.clone(),
                    compose::cut(&strip_calls(&text), compose::SOURCE_WORDS),
                ))
            })
            .collect();
        let told_in = match Form::of(&writes) {
            Form::LifeEvent => life_voice(&*shelf, &writes),
            _ => None,
        };
        let question = compose::question(
            &brief,
            &writes,
            min_words,
            &sources,
            standing.as_deref(),
            told_in,
        );
        let Some(minds) = self.minds.read().unwrap().clone() else {
            return refused("You cannot settle to write yet.".into());
        };
        let voice = match Config::load(&root) {
            Ok(Some(config)) => config.writing,
            Ok(None) => return refused("Nothing here says how this record is written.".into()),
            Err(e) => return refused(format!("You cannot settle to write: {e}")),
        };
        let (engine, base) = (minds.engine(), minds.base_config());
        // A piece that stands is mended by its paragraphs' anchors — `mend`
        // beside `draft`, its anchors bound in the grammar — and one to be
        // written anew is drafted (`docs/npcd_anchored_edits.md` §7).
        let standing_doc = standing
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| Doc::new(&writes, s))
            .filter(|d| !d.units().is_empty());
        let mut calls = vec![compose::spec()];
        if let Some(doc) = &standing_doc {
            calls.push(compose::mend_spec(&doc.anchors()));
        }
        // One sitting writes the whole piece: one that comes in short of its
        // floor is carried on from where it stopped — see `compose::fuller` —
        // and what comes next is put after it.
        let mut asked = question.clone();
        let mut text: Option<String> = None;
        let mut mend: Option<compose::Mended> = None;
        for _ in 0..compose::SITTINGS {
            let ask = CallAsk {
                system: &voice,
                prompt: &asked,
                calls: &calls,
                max_tokens: compose::COMPOSE_TOKENS,
                temperature: None,
                seed: resolve_seed(None),
                think: compose::THINK,
                band: WORK_BAND,
            };
            let raw = match decode_call(&engine, &base, &ask).await {
                Ok(raw) => raw,
                Err(e) if text.is_none() => {
                    return refused(format!("You sat down to write and could not: {e}"))
                }
                Err(_) => break,
            };
            if let (Some(doc), Some(edits)) = (&standing_doc, compose::read_mend(&raw, &calls)) {
                let made = compose::mended(doc, &edits);
                if made.made == 0 {
                    return refused(compose::nothing_mended(&writes, &made.refused));
                }
                text = Some(made.text.clone());
                mend = Some(made);
                break;
            }
            // Carrying a piece on is drafting it: only the first sitting may mend.
            calls.truncate(1);
            let Some(written) = compose::read(&raw, &calls) else {
                break;
            };
            let written = compose::unlabelled(&written, &writes);
            // A sitting that wrote nothing but the file's name wrote nothing.
            if written.trim().is_empty() {
                break;
            }
            let piece = match &text {
                Some(so_far) => compose::joined(so_far, &written),
                None => written,
            };
            let words = piece.split_whitespace().count();
            if words >= min_words {
                text = Some(piece);
                break;
            }
            asked = compose::fuller(&question, &piece, words, min_words);
            text = Some(piece);
        }
        let Some(text) = text else {
            return refused(
                "You sat down to write and nothing came. Read your brief and your sources again, \
                 and sit down to it again."
                    .into(),
            );
        };
        // **A piece written back as it stands is no change, and is said to be.**
        // Sent to mend a draft, a reviewer sat down and wrote the standing text
        // out again word for word; a working copy equal to the record is not a
        // change, so its commit found nothing to merge and the next sitting was
        // let through as if nothing had been written — four sittings over the
        // same 307 words.
        if standing
            .as_deref()
            .is_some_and(|s| compose::same_words(s, &text))
        {
            return refused(compose::unchanged(&writes));
        }
        let words = text.split_whitespace().count();
        let write = Act {
            tool: "file_write",
            args: json!({ "path": writes, "content": text })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        };
        let wrote = self.record_act(npc_id, &write);
        if !wrote.landed {
            return refused(wrote.answer);
        }
        if let Some(m) = mend {
            tracing::info!(
                npc_id,
                %writes,
                words,
                made = m.made,
                refused = m.refused.len(),
                "mended a piece by its anchors",
            );
            let refusals = match m.refused.is_empty() {
                true => String::new(),
                false => format!(
                    " {} of your changes could not be made: {}.",
                    m.refused.len(),
                    m.refused.join("; ")
                ),
            };
            return Recorded {
                feed: format!(
                    "{} {LANDED} mended {writes}, {} changes, {words} words",
                    act.summary(),
                    m.made
                ),
                answer: format!(
                    "You sat down with {writes} and mended it: {} changes, {words} words now, in \
                     your working copy.{refusals} Read it back with `file_read`. If it stands, \
                     `bench_commit` it; where one passage in it is still wrong, put that passage \
                     right with `file_edit` — not the rest.",
                    m.made
                ),
                landed: true,
                departing: false,
            };
        }
        tracing::info!(npc_id, %writes, words, "composed a draft");
        Recorded {
            feed: format!(
                "{} {LANDED} wrote {writes} through, {words} words",
                act.summary()
            ),
            answer: format!(
                "You sat down with your brief and your sources open and wrote {writes} through: \
                 {words} words, in your working copy. Read it back with `file_read`. If it \
                 stands, `bench_commit` it; where one passage in it is wrong, put that passage \
                 right with `file_edit` — not the rest."
            ),
            landed: true,
            departing: false,
        }
    }
}
