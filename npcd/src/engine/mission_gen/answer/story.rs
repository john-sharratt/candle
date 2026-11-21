//! Defines the call a story is answered with, and the constraints that keep it inside the world and clear of names other stories already use.
//!
//! The story's brief states the world's setting as the present and the era as the time the work happens, so a story is not set in the wrong period. Its cast is checked against the names earlier stories have invented, because a model left to choose names draws from the same handful, and invented people should be new people. A name the record's own documents use is allowed back, since those belong to the record. Names are the capitalised words of the answer, so a possessive resolves to the person it belongs to.

use super::*;

/// The call for a story.
pub const STORY: &str = "story";

/// What every brief for a piece of the record says about where it is written.
///
/// **The writers' room is not the world.** Nine of eleven pieces judged against
/// the lore had the vault in them — its rooms and machines, the Makers by
/// name, a story whose plot was somebody writing — because that was the most
/// concrete thing in front of the Maker writing it. The gate refuses it at the
/// report (`leakage`); this says it before the first word.
pub(super) const ELSEWHERE: &str =
    "\n\nYou write this from the vault, and nothing of the vault belongs in \
                         it: not its rooms or machines, not the Makers, not the writing of it. \
                         Nobody in it is writing a record. Tell it from inside its own world, in \
                         its own time, from what the record says was there.";

pub(super) fn story(field: &Fields, era: &str, corpus: &Corpus) -> Result<Answer, String> {
    let era_text = corpus.text(era).unwrap_or_default();
    let title = field.get("title");
    let slug = slug(&title);
    if slug.split('-').count() < 2 {
        return Err("`title` needs to be a real title of at least two words.".into());
    }
    let writes = format!("layers/stories/{slug}.md");
    if corpus.exists(&writes) {
        return Err(format!(
            "A story called \"{title}\" is already told; tell another."
        ));
    }
    // A year alone is a when: "2203" was refused for being one word, and the
    // generation spent on it was lost.
    let when = field.words("when", 1, WHEN_WORDS, "when it happens")?;
    let place = field.words("where", 2, WHERE_WORDS, "where it happens")?;
    let who = field
        .words("who", 1, WHO_WORDS, "who is there")?
        .replace('\n', "; ");
    let reused = reused_names(corpus, &who);
    if !reused.is_empty() {
        return Err(format!(
            "{} already {} in other stories of the record. The people this story invents are new \
             people: give them names of their own. Only somebody the record itself names — in its \
             eras, its world, or a life — is the same person again.",
            reused.join(", "),
            if reused.len() == 1 {
                "appears"
            } else {
                "appear"
            }
        ));
    }
    let turn = turns(field)?;
    let happens = field.words("happens", 25, HAPPENS_WORDS, "what happens in it")?;
    let agrees = field.get("agrees_with");
    // A list's entries are each a quote, with or without their own marks.
    let entries = agrees
        .lines()
        .map(|l| l.trim().trim_matches(['"', '“', '”']).trim().to_string());
    let quote = quotes_in(&agrees, MIN_QUOTE_CHARS)
        .into_iter()
        .chain(entries)
        .chain(std::iter::once(agrees.trim().to_string()))
        .chain(sentences_of(&agrees))
        .filter(|q| q.chars().count() >= MIN_QUOTE_CHARS)
        .find(|q| contains(&era_text, q))
        .ok_or_else(|| {
            "`agrees_with` must quote, between double quotes, the era's own words the story has \
             to agree with — and they must be its words exactly."
                .to_string()
        })?;
    let mut brief = format!(
        "Tell a story the record passes over: \"{title}\".\n\n\
         When: {when}\nWhere: {place}\nWho is there: {who}\n\n\
         What happens: {happens}\n\n\
         It turns on: {turn}\n\n\
         It must agree with the era, which says: \"{quote}\"\n\n\
         Write it as a story — past tense, six hundred to a thousand words, beginning with a \
         heading of its title. Show it as it happens: what the people in it do and say, moment by \
         moment, in the place itself. Never summarise what the era already says; the story is \
         what the era leaves out."
    );
    // **The world's own setting, so a story stays inside it.** One set in a
    // post-collapse world of towers and machines came back with quills, parchment
    // and silk weights. Said as the present, beside the era it happens in.
    let (mut world, mut words) = (String::new(), String::new());
    if let Some(e) = corpus.eras.iter().find(|e| e.path == era) {
        world = world_then(corpus, e, None).unwrap_or_default();
        words = worlds_words(
            corpus,
            &format!("{title}. {place}. {who}. {happens} {turn}"),
            e,
        );
    }
    brief.push_str(&world);
    brief.push_str(&words);
    let era_title = corpus
        .eras
        .iter()
        .find(|e| e.path == era)
        .map_or(era, |e| e.title.as_str());
    // Read up to the year the Maker works it in — the era's opening, the same
    // year its time step sets (`canon::set_in`). An era with no year is read
    // on its own.
    let read = match corpus
        .eras
        .iter()
        .find(|e| e.path == era)
        .and_then(|e| e.year)
    {
        Some(year) => research::before_writing(
            corpus,
            &writes,
            year,
            &format!("{title}. {place}. {who}. {happens}"),
        ),
        None => Vec::new(),
    };
    let reads: Vec<String> = match read.is_empty() {
        true => vec![era.to_string()],
        false => read.iter().map(|r| r.path.clone()).collect(),
    };
    let said = research::said(&read);
    brief.push_str(ELSEWHERE);
    brief.push_str(&said);
    let fields = fields([
        ("title", title.clone()),
        ("when", when),
        ("where", place),
        ("who", who),
        ("happens", happens),
        ("turns", turn),
        ("quote", quote),
        ("world-then", world.trim().to_string()),
        ("worlds-words", words.trim().to_string()),
        ("reads", said.trim().to_string()),
    ]);
    Ok(Answer::Mission(Proposal {
        objective: format!("Tell what {era_title} passes over: \"{title}\""),
        brief,
        writes,
        reads,
        min_words: STORY_MIN_WORDS,
        fields,
    }))
}

/// How many told stories a name may already be in before a new story taking it
/// is reusing a person rather than inventing one.
pub(super) const NAME_STORIES: usize = 2;

/// The names in `who` that [`NAME_STORIES`] or more stories already use and
/// the record's canon never does — the generator's own default names.
///
/// **Invented people are new people.** Left to choose, the model named its
/// invented soldiers and engineers from the same handful: three stories in one
/// afternoon had a "Kaelen", beside a "Vasko", a "Senna" and a "Jorik" each in
/// several, until the record read as one cast wandering through every era. A
/// name the eras, the world's own documents or a life use is the record's, and
/// may come back.
pub(super) fn reused_names(corpus: &Corpus, who: &str) -> Vec<String> {
    let taken = taken_names(corpus);
    let mut reused: Vec<String> = Vec::new();
    for word in names_in(who) {
        if taken.contains(&word) && !reused.contains(&word) {
            reused.push(word);
        }
    }
    reused
}

/// The names the told stories have invented and used in [`NAME_STORIES`] or
/// more of them, most used first — what a new story's cast is told to stay
/// clear of before it is refused for it.
///
/// **A name the model does not know is taken, it takes.** Refused for one
/// handful of reused names, a story's next answer swapped in another — "Thorne,
/// Voss, Aris", then "Senna, Jorik", then "Halloway" — three tries running, and
/// the target was given up. Said up front, the list is avoided.
pub(in super::super) fn taken_names(corpus: &Corpus) -> Vec<String> {
    // The whole lore — eras and every world document, where the ranks and
    // trades a cast is introduced by are named too — and the lives.
    let lore = corpus
        .mind
        .0
        .as_ref()
        .map(|m| leakage::lore(m))
        .unwrap_or_default();
    let lives: Vec<String> = corpus
        .lives
        .iter()
        .map(|l| format!("{} {}", l.name, l.anchor).to_lowercase())
        .collect();
    let stories: Vec<String> = corpus
        .stories
        .iter()
        .map(|s| s.text.to_lowercase())
        .collect();
    let mut counted: Vec<(String, usize)> = Vec::new();
    for story in &corpus.stories {
        for word in names_in(&story.text) {
            if counted.iter().any(|(w, _)| *w == word) {
                continue;
            }
            let name = word.to_lowercase();
            if has_word(&lore, &name) || lives.iter().any(|l| has_word(l, &name)) {
                continue;
            }
            let n = stories.iter().filter(|s| has_word(s, &name)).count();
            counted.push((word, n));
        }
    }
    counted.retain(|(_, n)| *n >= NAME_STORIES);
    counted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counted.into_iter().map(|(w, _)| w).collect()
}

/// The capitalised words of `text` three letters or longer — the candidates
/// for a name — each without a possessive "'s", so "Korr's" is Korr.
pub(super) fn names_in(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’')
        .map(|w| {
            w.strip_suffix("'s")
                .or_else(|| w.strip_suffix("’s"))
                .unwrap_or(w)
                .trim_matches(['\'', '’'])
        })
        .filter(|w| w.chars().count() >= 3 && w.chars().next().is_some_and(char::is_uppercase))
        .map(str::to_string)
        .collect()
}

/// Whether `word` stands in `text` as a whole word.
pub(super) fn has_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
    })
}

/// A title as a story's file name: lower case, words joined by hyphens.
pub(super) fn slug(title: &str) -> String {
    title
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}
