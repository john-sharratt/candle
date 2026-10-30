//! Groups (§3.5): making one, who is in it and as what, and which accounts
//! are a person (§3.4). Every person in a group has their own journal in it,
//! `people/<person>`, opened when they join.

use crate::error::{Error, Result};
use crate::files::{
    Account, AccountKey, Email, GroupFile, Member, Meta, PersonFile, Role, Who, MAX_GROUPS,
};
use crate::ops::fold::fold_child;
use crate::ops::{Applied, OwnerFile};
use crate::path::{Below, JournalPath};
use crate::scope::Section;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{skeleton, Group, Journal, Person, Tree, MAX_CHILDREN};

fn group_file<'a>(tree: &'a mut Tree, group: &Slug) -> Result<&'a mut GroupFile> {
    let g = tree
        .groups
        .get_mut(group)
        .ok_or_else(|| Error::not_found(format!("there is no group @{group}")))?;
    g.file.as_mut().map_err(|p| {
        Error::refused(format!(
            "@{group} cannot change until group.yaml is fixed: {p}"
        ))
    })
}

fn person_file<'a>(tree: &'a mut Tree, person: &Slug) -> Result<Option<&'a mut PersonFile>> {
    match tree.persons.get_mut(person) {
        None => Ok(None),
        Some(p) => p.file.as_mut().map(Some).map_err(|e| {
            Error::refused(format!(
                "{person} cannot change until person.yaml is fixed: {e}"
            ))
        }),
    }
}

/// An email no other person lists.
fn free_email(tree: &Tree, person: &Slug, email: &Email) -> Result<()> {
    for (slug, p) in &tree.persons {
        if slug == person {
            continue;
        }
        if p.file
            .as_ref()
            .is_ok_and(|f| f.accounts.iter().any(|a| &a.email == email))
        {
            return Err(Error::refused(format!(
                "{email} is {slug}'s: an account is one person's"
            )));
        }
    }
    Ok(())
}

fn one_owner_left(file: &GroupFile, person: &Slug) -> bool {
    file.owners().all(|o| o == person)
}

/// `group` joins `file`'s groups, and becomes its default when it is their
/// first. A person with other groups and no default chooses one themselves:
/// an owner adding them somewhere decides nothing about where they write.
fn join(file: &mut PersonFile, group: &Slug) -> Result<()> {
    if file.groups.len() >= MAX_GROUPS {
        return Err(Error::refused(format!(
            "{} belongs to {MAX_GROUPS} groups already: at most {MAX_GROUPS}",
            file.person
        )));
    }
    if file.groups.is_empty() {
        file.default_group = Some(group.clone());
    }
    file.groups.push(group.clone());
    Ok(())
}

/// What opening a member's own journal touched.
#[derive(Default)]
struct Opened {
    /// Their journal, when this opened it.
    journal: Option<JournalPath>,
    /// The journal folded to make room for it.
    folded: Option<JournalPath>,
}

impl Opened {
    /// Into `applied`: each journal whose files changed, and the fold.
    fn record(self, applied: &mut Applied) {
        if let Some(f) = self.folded {
            applied
                .touched
                .push(f.parent().expect("a fold is below its section"));
            applied.touched.push(f.clone());
            applied.folded.push(f);
        }
        applied.touched.extend(self.journal);
    }
}

/// `person`'s own journal in `group`, `people/<person>`, opened if it is not
/// there yet. `people` takes it as it takes any journal — written, and not
/// over a reference's name — except that a full `people` folds the idlest
/// journal of someone who is not a member, so every member has theirs
/// (§2.3).
fn open_person(
    tree: &mut Tree,
    group: &Slug,
    person: &Slug,
    name: &str,
    now: Stamp,
) -> Result<Opened> {
    let people = JournalPath::of_section(group, Section::People);
    let path = people.child(person.clone())?;
    let members: Vec<Slug> = tree
        .group_file(group)
        .map(|f| f.persons().map(|(p, _)| p.clone()).collect())
        .unwrap_or_default();
    let section = tree.need_mut(&people)?;
    if section.children.contains_key(person) {
        return Ok(Opened::default());
    }
    section.writable(&people.show())?;
    if let Some(r) = section
        .meta
        .iter()
        .flat_map(|m| &m.refs)
        .find(|r| &r.name == person)
    {
        return Err(Error::refused(format!(
            "{path} is a reference already, to {}: restructuring {people} is a coding task on \
             the journals repository",
            r.to
        )));
    }
    let mut folded = None;
    if section.live_count() >= MAX_CHILDREN {
        let victim = section
            .live()
            .filter(|(n, _)| !members.contains(n))
            .min_by_key(|(n, j)| (j.last_active(), (*n).clone()))
            .map(|(n, _)| n.clone())
            .ok_or_else(|| {
                Error::refused(format!(
                    "{people} holds {MAX_CHILDREN} members' journals, so {person} has no room \
                     for theirs; restructuring it is a coding task on the journals repository"
                ))
            })?;
        folded = Some(fold_child(section, &people, &victim, now)?);
    }
    let cue = format!("open for {name}: who they are, what they want, what they are working on");
    section.children.insert(
        person.clone(),
        Journal::new(
            path.place()?,
            Some(Meta::new(person.as_str(), Some(cue), now)),
        ),
    );
    Ok(Opened {
        journal: Some(path),
        folded,
    })
}

/// A new group, `creator` its first owner, with their own journal in it.
pub fn create_group(
    tree: &mut Tree,
    creator: &Slug,
    group: &Slug,
    name: &str,
    about: Option<&str>,
    now: Stamp,
) -> Result<Applied> {
    if tree.groups.contains_key(group) {
        return Err(Error::refused(format!(
            "@{group} exists already: pick another name"
        )));
    }
    let file = GroupFile {
        group: group.clone(),
        name: name.trim().to_string(),
        about: about.map(str::trim).map(String::from),
        members: vec![Member {
            who: Who::Person(creator.clone()),
            role: Role::Owner,
            joined: now,
            reads: vec![],
        }],
    };
    file.check().map_err(Error::Invalid)?;
    let person = person_file(tree, creator)?
        .ok_or_else(|| Error::not_found(format!("there is no person {creator}")))?;
    join(person, group)?;
    // A group they make is one they chose: it is their default if they have
    // none.
    if person.default_group.is_none() {
        person.default_group = Some(group.clone());
    }
    person.check().map_err(Error::Invalid)?;
    let creator_name = person.name.clone();
    tree.groups.insert(
        group.clone(),
        Group {
            file: Ok(file),
            root: skeleton(),
            problems: Vec::new(),
        },
    );
    let mut applied = Applied {
        subject: format!("Create @{group}"),
        owners: vec![
            OwnerFile::Group(group.clone()),
            OwnerFile::Person(creator.clone()),
        ],
        ..Applied::default()
    };
    open_person(tree, group, creator, &creator_name, now)?.record(&mut applied);
    Ok(applied)
}

#[allow(clippy::too_many_arguments)]
pub fn add_member(
    tree: &mut Tree,
    group: &Slug,
    person: &Slug,
    name: Option<&str>,
    emails: &[Email],
    role: Role,
    now: Stamp,
) -> Result<Applied> {
    group_file(tree, group)?;
    for e in emails {
        free_email(tree, person, e)?;
    }
    let mut changed = false;
    let their_name = match person_file(tree, person)? {
        None => {
            let name = name
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| Error::invalid(format!("{person} is new: give their name")))?;
            let mut accounts: Vec<Account> = Vec::new();
            for e in emails {
                if !accounts.iter().any(|a| &a.email == e) {
                    accounts.push(Account {
                        email: e.clone(),
                        key: None,
                    });
                }
            }
            let file = PersonFile {
                person: person.clone(),
                name: name.to_string(),
                accounts,
                groups: vec![group.clone()],
                default_group: Some(group.clone()),
            };
            file.check().map_err(Error::Invalid)?;
            tree.persons.insert(
                person.clone(),
                Person {
                    file: Ok(file),
                    problems: Vec::new(),
                },
            );
            changed = true;
            name.to_string()
        }
        Some(file) => {
            // Their first account makes them someone who can sign in, and the
            // admin who lists it decides where they start: whoever made the
            // person before then — any owner may make a newcomer without an
            // account — chose nothing on their behalf.
            let first_account = file.accounts.is_empty() && !emails.is_empty();
            if let Some(n) = name
                .map(str::trim)
                .filter(|n| !n.is_empty() && *n != file.name)
            {
                file.name = n.to_string();
                changed = true;
            }
            for e in emails {
                if !file.accounts.iter().any(|a| &a.email == e) {
                    file.accounts.push(Account {
                        email: e.clone(),
                        key: None,
                    });
                    changed = true;
                }
            }
            if !file.groups.contains(group) {
                join(file, group)?;
                changed = true;
            }
            if first_account {
                file.default_group = Some(group.clone());
            }
            file.check().map_err(Error::Invalid)?;
            file.name.clone()
        }
    };
    let g = group_file(tree, group)?;
    let at = g
        .members
        .iter()
        .position(|m| matches!(&m.who, Who::Person(p) if p == person));
    let subject = match at {
        Some(i) if g.members[i].role == role => format!("{person} in @{group}"),
        Some(i) => {
            if g.members[i].role == Role::Owner && one_owner_left(g, person) {
                return Err(Error::refused(format!("{person} is @{group}'s last owner")));
            }
            g.members[i].role = role;
            changed = true;
            format!("{person} is now @{group}'s {}", role.name())
        }
        None => {
            g.members.push(Member {
                who: Who::Person(person.clone()),
                role,
                joined: now,
                reads: vec![],
            });
            changed = true;
            format!("Add {person} to @{group} as {}", role.name())
        }
    };
    if !changed {
        return Err(Error::refused(format!(
            "{person} is in @{group} as {} already",
            role.name()
        )));
    }
    g.check().map_err(Error::Invalid)?;
    let mut applied = Applied {
        subject,
        owners: vec![
            OwnerFile::Group(group.clone()),
            OwnerFile::Person(person.clone()),
        ],
        ..Applied::default()
    };
    open_person(tree, group, person, &their_name, now)?.record(&mut applied);
    Ok(applied)
}

/// `group`'s `about`: what it is for, as its README and its frames say.
pub fn describe_group(tree: &mut Tree, group: &Slug, about: &str) -> Result<Applied> {
    let g = group_file(tree, group)?;
    let about = about.trim();
    if about.is_empty() {
        return Err(Error::invalid(
            "say what the group is for: `about` is empty",
        ));
    }
    if g.about.as_deref() == Some(about) {
        return Err(Error::refused(format!("@{group} says so already")));
    }
    g.about = Some(about.to_string());
    g.check().map_err(Error::Invalid)?;
    Ok(Applied {
        subject: format!("Describe @{group}"),
        owners: vec![OwnerFile::Group(group.clone())],
        ..Applied::default()
    })
}

/// `bot` joins `group` as a reader of the subtrees `reads` grants it.
pub fn add_bot(
    tree: &mut Tree,
    group: &Slug,
    bot: &Slug,
    reads: &[Below],
    now: Stamp,
) -> Result<Applied> {
    for r in reads {
        r.at(group)?;
    }
    let g = group_file(tree, group)?;
    if g.bot(bot).is_some() {
        return Err(Error::refused(format!("bot {bot} is in @{group} already")));
    }
    g.members.push(Member {
        who: Who::Bot(bot.clone()),
        role: Role::Reader,
        joined: now,
        reads: reads.to_vec(),
    });
    g.check().map_err(Error::Invalid)?;
    let granted: Vec<String> = reads.iter().map(Below::to_string).collect();
    Ok(Applied {
        subject: format!(
            "Add bot {bot} to @{group}, reading {}",
            if granted.is_empty() {
                "nothing".to_string()
            } else {
                granted.join(", ")
            }
        ),
        owners: vec![OwnerFile::Group(group.clone())],
        ..Applied::default()
    })
}

/// `who` out of `group`. A person's own journal there stays, with their
/// lines, as everything they wrote does; they read the group no longer.
pub fn remove_member(tree: &mut Tree, group: &Slug, who: &Who) -> Result<Applied> {
    let g = group_file(tree, group)?;
    let i = g
        .members
        .iter()
        .position(|m| &m.who == who)
        .ok_or_else(|| Error::not_found(format!("{who} is not in @{group}")))?;
    if let Who::Person(person) = who {
        if g.members[i].role == Role::Owner && one_owner_left(g, person) {
            return Err(Error::refused(format!("{person} is @{group}'s last owner")));
        }
    }
    g.members.remove(i);
    let mut owners = vec![OwnerFile::Group(group.clone())];
    if let Who::Person(person) = who {
        if let Some(file) = person_file(tree, person)? {
            file.groups.retain(|x| x != group);
            // Out of their default, they have none: any other group they are
            // in may be one an owner added them to, and would put their
            // writes where that owner chose.
            if file.default_group.as_ref() == Some(group) {
                file.default_group = None;
            }
            owners.push(OwnerFile::Person(person.clone()));
        }
    }
    Ok(Applied {
        subject: format!("Remove {who} from @{group}"),
        owners,
        ..Applied::default()
    })
}

/// `person`'s choice of the group their conversations open in.
pub fn choose_default(tree: &mut Tree, person: &Slug, group: &Slug) -> Result<Applied> {
    let file = person_file(tree, person)?
        .ok_or_else(|| Error::not_found(format!("there is no person {person}")))?;
    if file.default_group.as_ref() == Some(group) {
        return Err(Error::refused(format!(
            "{person}'s conversations open in @{group} already"
        )));
    }
    file.default_group = Some(group.clone());
    file.check().map_err(Error::Invalid)?;
    Ok(Applied {
        subject: format!("{person}'s conversations open in @{group}"),
        owners: vec![OwnerFile::Person(person.clone())],
        ..Applied::default()
    })
}

pub fn bind(tree: &mut Tree, person: &Slug, email: &Email, key: &AccountKey) -> Result<Applied> {
    let elsewhere = tree.persons.iter().any(|(_, p)| {
        p.file
            .as_ref()
            .is_ok_and(|f| f.accounts.iter().any(|a| a.key.as_ref() == Some(key)))
    });
    if elsewhere {
        return Err(Error::conflict(format!(
            "the account {key} is bound already"
        )));
    }
    let file = person_file(tree, person)?
        .ok_or_else(|| Error::not_found(format!("there is no person {person}")))?;
    let account = file
        .accounts
        .iter_mut()
        .find(|a| &a.email == email)
        .ok_or_else(|| Error::not_found(format!("{person} lists no {email}")))?;
    if account.key.is_some() {
        return Err(Error::conflict(format!("{email} is bound already")));
    }
    account.key = Some(key.clone());
    Ok(Applied {
        subject: format!("Bind {email} to {person}"),
        owners: vec![OwnerFile::Person(person.clone())],
        ..Applied::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{bot_may_read, home_group, may_read, Actor};
    use crate::ops::fixture::{john, now, p, stranger, yen, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::slug;
    use crate::tree::check::check;

    fn add(person: &str, name: Option<&str>, emails: &[&str], role: Role) -> Op {
        Op::AddMember {
            group: slug("zen"),
            person: slug(person),
            name: name.map(Into::into),
            emails: emails.iter().map(|e| Email::parse(e).unwrap()).collect(),
            role,
        }
    }

    fn remove(person: &str) -> Op {
        Op::RemoveMember {
            group: slug("zen"),
            who: Who::Person(slug(person)),
        }
    }

    fn make(group: &str) -> Op {
        Op::CreateGroup {
            group: slug(group),
            name: group.into(),
            about: None,
        }
    }

    /// The tree checked afresh holds no problem anywhere.
    fn clean(t: &Tree) {
        let mut c = t.clone();
        for p in c.persons.values_mut() {
            p.problems.clear();
        }
        for g in c.groups.values_mut() {
            g.problems.clear();
        }
        c.problems.clear();
        check(&mut c);
        assert!(
            c.problems.is_empty()
                && c.persons.values().all(|p| p.problems.is_empty())
                && c.groups.values().all(|g| g.problems.is_empty())
                && c.journals().iter().all(|(_, j)| j.problems.is_empty()),
            "{:?} {:?}",
            c.problems,
            c.groups.values().map(|g| &g.problems).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_new_group_is_made_with_its_creator_as_owner_and_their_journal() {
        let mut t = zen();
        let op = Op::CreateGroup {
            group: slug("battle-cities"),
            name: "Battle Cities".into(),
            about: Some("The game's world.".into()),
        };
        let a = apply(&mut t, &john(), &op, now()).unwrap();
        assert_eq!(a.subject, "Create @battle-cities");
        assert_eq!(
            a.owners,
            vec![
                OwnerFile::Group(slug("battle-cities")),
                OwnerFile::Person(slug("john"))
            ]
        );
        assert_eq!(a.touched, vec![p("@battle-cities/people/john")]);
        let g = t.group_file(&slug("battle-cities")).unwrap();
        assert_eq!(g.about.as_deref(), Some("The game's world."));
        assert_eq!(g.owners().collect::<Vec<_>>(), [&slug("john")]);
        let pf = t.person_file(&slug("john")).unwrap();
        assert_eq!(
            pf.groups,
            [slug("zen"), slug("guest"), slug("battle-cities")]
        );
        assert_eq!(pf.default_group, Some(slug("zen")), "the default stays");
        assert!(may_read(&t, &slug("john"), &p("@battle-cities")));
        assert_eq!(
            t.get(&p("@battle-cities/people/john"))
                .unwrap()
                .cue()
                .as_deref(),
            Some("open for john: who they are, what they want, what they are working on")
        );
        clean(&t);
        assert_eq!(
            apply(&mut t, &yen(), &op, now()).unwrap_err().to_string(),
            "@battle-cities exists already: pick another name"
        );
        // A person in no group makes their first, and it is their default.
        apply(&mut t, &stranger(), &make("solo-team"), now()).unwrap();
        assert_eq!(
            t.person_file(&slug("stranger")).unwrap().default_group,
            Some(slug("solo-team"))
        );
        clean(&t);
    }

    #[test]
    fn a_new_person_is_made_joined_and_given_their_journal_in_one_change() {
        let mut t = zen();
        let a = apply(
            &mut t,
            &john(),
            &add(
                "yen-nhi",
                Some("Yen Nhi"),
                &["yen.nhi@example.com"],
                Role::Member,
            ),
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "Add yen-nhi to @zen as member");
        assert_eq!(
            a.owners,
            vec![
                OwnerFile::Group(slug("zen")),
                OwnerFile::Person(slug("yen-nhi"))
            ]
        );
        assert_eq!(a.touched, vec![p("people/yen-nhi")]);
        let pf = t.person_file(&slug("yen-nhi")).unwrap();
        assert_eq!(pf.groups, vec![slug("zen")]);
        assert_eq!(pf.default_group, Some(slug("zen")));
        assert_eq!(pf.accounts[0].email.as_str(), "yen.nhi@example.com");
        assert_eq!(
            t.group_file(&slug("zen"))
                .unwrap()
                .member(&slug("yen-nhi"))
                .unwrap()
                .joined,
            now()
        );
        assert_eq!(
            t.get(&p("people/yen-nhi")).unwrap().cue().as_deref(),
            Some("open for Yen Nhi: who they are, what they want, what they are working on")
        );
        clean(&t);
    }

    #[test]
    fn a_new_person_needs_a_name_and_free_emails() {
        let mut t = zen();
        assert_eq!(
            apply(&mut t, &john(), &add("x", None, &[], Role::Member), now())
                .unwrap_err()
                .to_string(),
            "x is new: give their name"
        );
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &add("x", Some("X"), &["yen@example.com"], Role::Member),
                now()
            )
            .unwrap_err()
            .to_string(),
            "yen@example.com is yen's: an account is one person's"
        );
    }

    #[test]
    fn a_second_account_is_added_to_an_existing_person() {
        let mut t = zen();
        let a = apply(
            &mut t,
            &john(),
            &add("yen", None, &["yen.other@example.net"], Role::Owner),
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "yen in @zen");
        assert_eq!(t.person_file(&slug("yen")).unwrap().accounts.len(), 2);
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &add("yen", None, &["yen.other@example.net"], Role::Owner),
                now()
            )
            .unwrap_err()
            .to_string(),
            "yen is in @zen as owner already"
        );
    }

    #[test]
    fn roles_change_but_the_last_owner_stays() {
        let mut t = zen();
        let a = apply(
            &mut t,
            &john(),
            &add("thanhlong", None, &[], Role::Owner),
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "thanhlong is now @zen's owner");
        apply(&mut t, &john(), &add("yen", None, &[], Role::Member), now()).unwrap();
        apply(
            &mut t,
            &john(),
            &add("thanhlong", None, &[], Role::Reader),
            now(),
        )
        .unwrap();
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &add("john", None, &[], Role::Member),
                now()
            )
            .unwrap_err()
            .to_string(),
            "john is @zen's last owner"
        );
        assert_eq!(
            apply(&mut t, &john(), &remove("john"), now())
                .unwrap_err()
                .to_string(),
            "john is @zen's last owner"
        );
    }

    /// **A removed member reads the group no longer**, and leaving their
    /// default leaves them none; their journal and lines stay.
    #[test]
    fn a_removed_member_reads_the_group_no_longer() {
        let mut t = zen();
        let a = apply(&mut t, &yen(), &remove("thanhlong"), now()).unwrap();
        assert_eq!(a.subject, "Remove thanhlong from @zen");
        let pf = t.person_file(&slug("thanhlong")).unwrap();
        assert!(pf.groups.is_empty() && pf.default_group.is_none());
        assert!(!may_read(&t, &slug("thanhlong"), &p("@zen")));
        assert_eq!(
            apply(&mut t, &yen(), &remove("thanhlong"), now())
                .unwrap_err()
                .to_string(),
            "thanhlong is not in @zen"
        );
        clean(&t);
    }

    /// **A member always has their journal**: joining a full `people` folds
    /// the idlest journal of someone who is not a member, never a member's.
    #[test]
    fn a_full_people_makes_room_for_a_new_member() {
        let mut t = zen();
        let people = p("people");
        for i in 0.. {
            let made = apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: people.clone(),
                    name: slug(&format!("friend{i}")),
                    cue: "open for a friend".into(),
                    on_taken: OnTaken::Refuse,
                },
                now().plus(i),
            );
            if made.is_err() {
                break;
            }
        }
        assert_eq!(t.get(&people).unwrap().live_count(), MAX_CHILDREN);
        let a = apply(
            &mut t,
            &john(),
            &Op::AddMember {
                group: slug("zen"),
                person: slug("newcomer"),
                name: Some("Newcomer".into()),
                emails: vec![],
                role: Role::Member,
            },
            now().plus(100),
        )
        .unwrap();
        assert_eq!(a.folded.len(), 1);
        let gone = a.folded[0].last().unwrap().as_str().to_string();
        assert!(gone.starts_with("friend"), "a non-member's: {gone}");
        assert!(t.get(&p("people/newcomer")).is_some_and(|j| !j.is_folded()));
        assert_eq!(t.get(&people).unwrap().live_count(), MAX_CHILDREN);
        clean(&t);
    }

    /// **An owner taken out of their default by another owner is left with
    /// no default**, never handed the group that owner left them in — Yen
    /// adds John to a group of hers, then takes him out of @zen.
    #[test]
    fn an_owner_cannot_steer_where_another_writes() {
        let mut t = zen();
        apply(&mut t, &yen(), &make("evil"), now()).unwrap();
        let add_john = Op::AddMember {
            group: slug("evil"),
            person: slug("john"),
            name: None,
            emails: vec![],
            role: Role::Member,
        };
        apply(&mut t, &yen(), &add_john, now()).unwrap();
        assert_eq!(
            t.person_file(&slug("john")).unwrap().default_group,
            Some(slug("zen")),
            "joining decides nothing about the default"
        );
        apply(&mut t, &yen(), &remove("john"), now()).unwrap();
        let pf = t.person_file(&slug("john")).unwrap();
        assert_eq!(pf.groups, [slug("guest"), slug("evil")]);
        assert_eq!(pf.default_group, None);
        assert_eq!(
            home_group(&t, &slug("john")),
            None,
            "two groups, none chosen"
        );
        // Left in @evil alone, he is still not opened there.
        let mut alone = t.clone();
        let guest = alone.groups.get_mut(&slug("guest")).unwrap();
        guest
            .file
            .as_mut()
            .unwrap()
            .members
            .retain(|m| m.who != Who::Person(slug("john")));
        assert_eq!(home_group(&alone, &slug("john")), None, "alone in @evil");
        clean(&t);
        // A group he makes himself is his own choice, and becomes it.
        apply(&mut t, &john(), &make("johns"), now()).unwrap();
        assert_eq!(
            t.person_file(&slug("john")).unwrap().default_group,
            Some(slug("johns"))
        );
        clean(&t);
    }

    /// **Whoever made a person before their first account chose nothing for
    /// them**: Stranger makes "mum" in a group of theirs; the admin listing
    /// her account through @zen makes @zen her default.
    #[test]
    fn a_first_account_starts_them_where_the_admin_lists_it() {
        let mut t = zen();
        apply(&mut t, &stranger(), &make("evil"), now()).unwrap();
        let squat = Op::AddMember {
            group: slug("evil"),
            person: slug("mum"),
            name: Some("Mum".into()),
            emails: vec![],
            role: Role::Member,
        };
        apply(&mut t, &stranger(), &squat, now()).unwrap();
        assert_eq!(
            t.person_file(&slug("mum")).unwrap().default_group,
            Some(slug("evil"))
        );
        apply(
            &mut t,
            &john(),
            &add("mum", None, &["mum@example.com"], Role::Member),
            now(),
        )
        .unwrap();
        let pf = t.person_file(&slug("mum")).unwrap();
        assert_eq!(pf.groups, [slug("evil"), slug("zen")]);
        assert_eq!(pf.default_group, Some(slug("zen")));
        assert!(t.get(&p("people/mum")).is_some(), "her journal in @zen");
        clean(&t);
    }

    /// **A person chooses where their conversations open**, among their
    /// groups.
    #[test]
    fn a_person_chooses_their_default_group() {
        let mut t = zen();
        apply(&mut t, &john(), &make("work"), now()).unwrap();
        let choose = |g: &str| Op::ChooseDefault { group: slug(g) };
        let a = apply(&mut t, &john(), &choose("work"), now()).unwrap();
        assert_eq!(a.subject, "john's conversations open in @work");
        assert_eq!(a.owners, vec![OwnerFile::Person(slug("john"))]);
        assert_eq!(home_group(&t, &slug("john")), Some(slug("work")));
        assert_eq!(
            apply(&mut t, &john(), &choose("work"), now())
                .unwrap_err()
                .to_string(),
            "john's conversations open in @work already"
        );
        assert_eq!(
            apply(&mut t, &john(), &choose("nowhere"), now())
                .unwrap_err()
                .to_string(),
            "`default_group` is nowhere, which is not one of their groups"
        );
        clean(&t);
    }

    /// **A bot joins as a reader of what it is granted and leaves alone**:
    /// no person.yaml is touched either way.
    #[test]
    fn a_bot_joins_as_a_reader_of_its_grants_and_leaves() {
        let mut t = zen();
        let add_bot = |reads: &[&str]| Op::AddBot {
            group: slug("zen"),
            bot: slug("d00d"),
            reads: reads.iter().map(|r| Below::parse(r).unwrap()).collect(),
        };
        let a = apply(&mut t, &john(), &add_bot(&["topics/game"]), now()).unwrap();
        assert_eq!(a.subject, "Add bot d00d to @zen, reading topics/game");
        assert_eq!(a.owners, vec![OwnerFile::Group(slug("zen"))]);
        assert!(bot_may_read(&t, &slug("d00d"), &p("topics/game/canon")));
        assert!(!bot_may_read(&t, &slug("d00d"), &p("people")));
        assert_eq!(
            apply(&mut t, &john(), &add_bot(&[]), now())
                .unwrap_err()
                .to_string(),
            "bot d00d is in @zen already"
        );
        let gone = apply(
            &mut t,
            &john(),
            &Op::RemoveMember {
                group: slug("zen"),
                who: Who::Bot(slug("d00d")),
            },
            now(),
        )
        .unwrap();
        assert_eq!(gone.subject, "Remove bot d00d from @zen");
        assert_eq!(gone.owners, vec![OwnerFile::Group(slug("zen"))]);
        assert!(!bot_may_read(&t, &slug("d00d"), &p("topics/game")));
        assert!(apply(&mut t, &john(), &add_bot(&["channels/grok"]), now())
            .unwrap_err()
            .to_string()
            .contains("`channels` is not a section"));
        clean(&t);
    }

    #[test]
    fn an_eleventh_group_is_refused() {
        let mut t = zen();
        // John is in @zen and @guest already.
        for i in 0..8 {
            apply(&mut t, &john(), &make(&format!("g{i}")), now()).unwrap();
        }
        assert_eq!(
            apply(&mut t, &john(), &make("g8"), now())
                .unwrap_err()
                .to_string(),
            "john belongs to 10 groups already: at most 10"
        );
        assert!(
            !t.groups.contains_key(&slug("g8")),
            "a refusal makes nothing"
        );
    }

    #[test]
    fn an_account_binds_once_to_its_email() {
        let mut t = zen();
        let key = AccountKey::parse("154797f34c5a6ca6").unwrap();
        let email = Email::parse("john@example.com").unwrap();
        let op = Op::Bind {
            person: slug("john"),
            email: email.clone(),
            key: key.clone(),
        };
        let a = apply(&mut t, &Actor::System, &op, now()).unwrap();
        assert_eq!(a.subject, "Bind john@example.com to john");
        assert_eq!(
            t.person_file(&slug("john")).unwrap().accounts[0].key,
            Some(key.clone())
        );
        assert!(matches!(
            apply(&mut t, &Actor::System, &op, now()),
            Err(Error::Conflict(_))
        ));
        let other = Op::Bind {
            person: slug("yen"),
            email: Email::parse("yen@example.com").unwrap(),
            key,
        };
        assert_eq!(
            apply(&mut t, &Actor::System, &other, now())
                .unwrap_err()
                .to_string(),
            "the account 154797f34c5a6ca6 is bound already"
        );
        let missing = Op::Bind {
            person: slug("yen"),
            email,
            key: AccountKey::parse("2222222222222222").unwrap(),
        };
        assert!(matches!(
            apply(&mut t, &Actor::System, &missing, now()),
            Err(Error::NotFound(_))
        ));
    }
}
