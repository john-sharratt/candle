//! Groups (§3.5): making one, who is in it and as what, and which accounts
//! are a person (§3.4).

use crate::error::{Error, Result};
use crate::files::{
    Account, AccountKey, Email, GroupFile, Kind, Member, PersonFile, Role, Who, MAX_FRAMED,
    MAX_GROUPS,
};
use crate::ops::Applied;
use crate::path::Owner;
use crate::scope::Scope;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{skeleton, Group, Person, Tree};

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

/// A new group, `creator` its first owner. What it is for is said after,
/// [`describe_group`].
pub fn create_group(
    tree: &mut Tree,
    creator: &Slug,
    group: &Slug,
    name: &str,
    kind: Kind,
    now: Stamp,
) -> Result<Applied> {
    if tree.groups.contains_key(group) {
        return Err(Error::refused(format!(
            "@{group} exists already: pick another name"
        )));
    }
    let name = name.trim();
    let file = GroupFile {
        group: group.clone(),
        name: name.to_string(),
        kind,
        about: None,
        members: vec![Member {
            who: Who::Person(creator.clone()),
            role: Role::Owner,
            joined: now,
        }],
    };
    file.check().map_err(Error::Invalid)?;
    let person = person_file(tree, creator)?
        .ok_or_else(|| Error::not_found(format!("there is no person {creator}")))?;
    join(person, group)?;
    // A group they make is one they chose: it is their default if they have
    // none, and room is left for one beside those they frame.
    if person.default_group.is_none() && person.framed.len() < MAX_FRAMED {
        person.default_group = Some(group.clone());
    }
    person.check().map_err(Error::Invalid)?;
    tree.groups.insert(
        group.clone(),
        Group {
            file: Ok(file),
            root: skeleton(Scope::Group),
            problems: Vec::new(),
        },
    );
    Ok(Applied {
        subject: format!("Create @{group}, a {kind}"),
        owners: vec![Owner::Group(group.clone()), Owner::Person(creator.clone())],
        ..Applied::default()
    })
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
    match person_file(tree, person)? {
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
                framed: vec![],
            };
            file.check().map_err(Error::Invalid)?;
            tree.persons.insert(
                person.clone(),
                Person {
                    file: Ok(file),
                    root: skeleton(Scope::Personal),
                    problems: Vec::new(),
                },
            );
            changed = true;
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
                file.framed.clear();
            }
            file.check().map_err(Error::Invalid)?;
        }
    }
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
    Ok(Applied {
        subject,
        owners: vec![Owner::Group(group.clone()), Owner::Person(person.clone())],
        ..Applied::default()
    })
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
        owners: vec![Owner::Group(group.clone())],
        ..Applied::default()
    })
}

/// `bot` joins `group` as a reader of its journals.
pub fn add_bot(tree: &mut Tree, group: &Slug, bot: &Slug, now: Stamp) -> Result<Applied> {
    let g = group_file(tree, group)?;
    if g.bot(bot).is_some() {
        return Err(Error::refused(format!("bot {bot} reads @{group} already")));
    }
    g.members.push(Member {
        who: Who::Bot(bot.clone()),
        role: Role::Reader,
        joined: now,
    });
    g.check().map_err(Error::Invalid)?;
    Ok(Applied {
        subject: format!("Add bot {bot} to @{group} as reader"),
        owners: vec![Owner::Group(group.clone())],
        ..Applied::default()
    })
}

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
    let mut owners = vec![Owner::Group(group.clone())];
    if let Who::Person(person) = who {
        if let Some(file) = person_file(tree, person)? {
            file.groups.retain(|x| x != group);
            file.framed.retain(|x| x != group);
            if file.default_group.as_ref() == Some(group) {
                // A group they framed is one they chose to have in front of
                // them, so it takes the default's place; failing that, none
                // does. Any other group they are in may be one an owner added
                // them to, and would put their writes where that owner chose.
                file.default_group = file.framed.first().cloned();
                if let Some(d) = &file.default_group {
                    let d = d.clone();
                    file.framed.retain(|x| x != &d);
                }
            }
            owners.push(Owner::Person(person.clone()));
        }
    }
    Ok(Applied {
        subject: format!("Remove {who} from @{group}"),
        owners,
        ..Applied::default()
    })
}

/// `person`'s choice of default group, when given, and of the others framed.
pub fn choose_groups(
    tree: &mut Tree,
    person: &Slug,
    default: Option<&Slug>,
    framed: &[Slug],
) -> Result<Applied> {
    let file = person_file(tree, person)?
        .ok_or_else(|| Error::not_found(format!("there is no person {person}")))?;
    let before = (file.default_group.clone(), file.framed.clone());
    if let Some(d) = default {
        file.default_group = Some(d.clone());
    }
    file.framed = framed.to_vec();
    file.check().map_err(Error::Invalid)?;
    if (file.default_group.clone(), file.framed.clone()) == before {
        return Err(Error::refused(format!(
            "{person}'s groups open as chosen already"
        )));
    }
    let shown: Vec<String> = file
        .framed_groups()
        .iter()
        .map(|g| format!("@{g}"))
        .collect();
    Ok(Applied {
        subject: format!("{person} opens with {}", shown.join(", ")),
        owners: vec![Owner::Person(person.clone())],
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
        owners: vec![Owner::Person(person.clone())],
        ..Applied::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{bot_may_read, may_read, viewer, Actor};
    use crate::ops::fixture::{john, now, sharratt, stranger, yen};
    use crate::ops::{apply, Op};
    use crate::testing::slug;
    use crate::tree::check::check;

    fn add(person: &str, name: Option<&str>, emails: &[&str], role: Role) -> Op {
        Op::AddMember {
            group: slug("sharratt"),
            person: slug(person),
            name: name.map(Into::into),
            emails: emails.iter().map(|e| Email::parse(e).unwrap()).collect(),
            role,
        }
    }

    fn remove(person: &str) -> Op {
        Op::RemoveMember {
            group: slug("sharratt"),
            who: Who::Person(slug(person)),
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
                && c.groups.values().all(|g| g.problems.is_empty()),
            "{:?} {:?}",
            c.problems,
            c.groups.values().map(|g| &g.problems).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_new_group_is_made_with_its_creator_as_owner() {
        let mut t = sharratt();
        let op = Op::CreateGroup {
            group: slug("battle-cities"),
            name: "Battle Cities".into(),
            kind: Kind::World,
        };
        let a = apply(&mut t, &john(), &op, now()).unwrap();
        assert_eq!(a.subject, "Create @battle-cities, a world");
        assert_eq!(
            a.owners,
            vec![
                Owner::Group(slug("battle-cities")),
                Owner::Person(slug("john"))
            ]
        );
        let g = t.group_file(&slug("battle-cities")).unwrap();
        assert_eq!(g.kind, Kind::World);
        assert_eq!(g.owners().collect::<Vec<_>>(), [&slug("john")]);
        let p = t.person_file(&slug("john")).unwrap();
        assert_eq!(p.groups, [slug("sharratt"), slug("battle-cities")]);
        assert_eq!(p.default_group, Some(slug("sharratt")), "the default stays");
        assert!(may_read(
            &t,
            &slug("john"),
            &Owner::Group(slug("battle-cities"))
        ));
        clean(&t);
        assert_eq!(
            apply(&mut t, &yen(), &op, now()).unwrap_err().to_string(),
            "@battle-cities exists already: pick another name"
        );
        // A person in no group makes their first, and it is their default.
        let mine = Op::CreateGroup {
            group: slug("solo-team"),
            name: "Solo".into(),
            kind: Kind::Team,
        };
        apply(&mut t, &stranger(), &mine, now()).unwrap();
        assert_eq!(
            t.person_file(&slug("stranger")).unwrap().default_group,
            Some(slug("solo-team"))
        );
        clean(&t);
    }

    #[test]
    fn a_new_person_is_made_and_joined_in_one_change() {
        let mut t = sharratt();
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
        assert_eq!(a.subject, "Add yen-nhi to @sharratt as member");
        assert_eq!(
            a.owners,
            vec![
                Owner::Group(slug("sharratt")),
                Owner::Person(slug("yen-nhi"))
            ]
        );
        let p = t.person_file(&slug("yen-nhi")).unwrap();
        assert_eq!(p.groups, vec![slug("sharratt")]);
        assert_eq!(p.default_group, Some(slug("sharratt")));
        assert_eq!(p.accounts[0].email.as_str(), "yen.nhi@example.com");
        assert_eq!(
            t.group_file(&slug("sharratt"))
                .unwrap()
                .member(&slug("yen-nhi"))
                .unwrap()
                .joined,
            now()
        );
        assert_eq!(
            t.persons[&slug("yen-nhi")].root.children.len(),
            2,
            "with their fixed sections"
        );
        clean(&t);
    }

    #[test]
    fn a_new_person_needs_a_name_and_free_emails() {
        let mut t = sharratt();
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
        let mut t = sharratt();
        let a = apply(
            &mut t,
            &john(),
            &add("yen", None, &["yen.other@example.net"], Role::Owner),
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "yen in @sharratt");
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
            "yen is in @sharratt as owner already"
        );
    }

    #[test]
    fn roles_change_but_the_last_owner_stays() {
        let mut t = sharratt();
        let a = apply(
            &mut t,
            &john(),
            &add("thanhlong", None, &[], Role::Owner),
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "thanhlong is now @sharratt's owner");
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
            "john is @sharratt's last owner"
        );
        assert_eq!(
            apply(&mut t, &john(), &remove("john"), now())
                .unwrap_err()
                .to_string(),
            "john is @sharratt's last owner"
        );
    }

    /// **A removed member reads the group no longer**, and their default
    /// moves to the next group they are in.
    #[test]
    fn a_removed_member_reads_the_group_no_longer() {
        let mut t = sharratt();
        let a = apply(&mut t, &yen(), &remove("thanhlong"), now()).unwrap();
        assert_eq!(a.subject, "Remove thanhlong from @sharratt");
        let p = t.person_file(&slug("thanhlong")).unwrap();
        assert!(p.groups.is_empty() && p.default_group.is_none());
        assert!(!may_read(
            &t,
            &slug("thanhlong"),
            &Owner::Group(slug("sharratt"))
        ));
        assert_eq!(
            apply(&mut t, &yen(), &remove("thanhlong"), now())
                .unwrap_err()
                .to_string(),
            "thanhlong is not in @sharratt"
        );
        clean(&t);
        let mut t = sharratt();
        for g in ["work", "club"] {
            let op = Op::CreateGroup {
                group: slug(g),
                name: g.into(),
                kind: Kind::Team,
            };
            apply(&mut t, &yen(), &op, now()).unwrap();
        }
        t.persons
            .get_mut(&slug("yen"))
            .unwrap()
            .file
            .as_mut()
            .unwrap()
            .framed = vec![slug("club")];
        apply(&mut t, &john(), &remove("yen"), now()).unwrap();
        let p = t.person_file(&slug("yen")).unwrap();
        assert_eq!(
            p.default_group,
            Some(slug("club")),
            "the group they framed, before the first they joined"
        );
        assert!(p.framed.is_empty(), "the new default is framed by being it");
        clean(&t);
    }

    /// **An owner taken out of their default by another owner is left with
    /// no default**, never handed the group that owner left them in — Yen
    /// adds John to a group of hers, then takes him out of @sharratt.
    #[test]
    fn an_owner_cannot_steer_where_another_writes() {
        let mut t = sharratt();
        let evil = Op::CreateGroup {
            group: slug("evil"),
            name: "Evil".into(),
            kind: Kind::Team,
        };
        apply(&mut t, &yen(), &evil, now()).unwrap();
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
            Some(slug("sharratt")),
            "joining decides nothing about the default"
        );
        apply(&mut t, &yen(), &remove("john"), now()).unwrap();
        let p = t.person_file(&slug("john")).unwrap();
        assert_eq!(p.groups, [slug("evil")]);
        assert_eq!(p.default_group, None);
        assert_eq!(viewer(&t, &slug("john")).default_group, None);
        clean(&t);
        // Leaving on his own, the same: @evil is Yen's pick, not his.
        let mut t = sharratt();
        apply(&mut t, &yen(), &evil, now()).unwrap();
        apply(&mut t, &yen(), &add_john, now()).unwrap();
        let leave = Op::RemoveMember {
            group: slug("sharratt"),
            who: Who::Person(slug("john")),
        };
        apply(&mut t, &john(), &leave, now()).unwrap();
        assert_eq!(t.person_file(&slug("john")).unwrap().default_group, None);
        // A group he makes himself is his own choice, and becomes it.
        let mine = Op::CreateGroup {
            group: slug("johns"),
            name: "John's".into(),
            kind: Kind::Team,
        };
        apply(&mut t, &john(), &mine, now()).unwrap();
        assert_eq!(
            t.person_file(&slug("john")).unwrap().default_group,
            Some(slug("johns"))
        );
        clean(&t);
        // Framing three already, no default fits beside them: the group is
        // made, and the default left for him to choose.
        let f = t
            .persons
            .get_mut(&slug("john"))
            .unwrap()
            .file
            .as_mut()
            .unwrap();
        f.framed = vec![slug("evil"), slug("johns")];
        f.default_group = None;
        for g in ["third", "fourth"] {
            let op = Op::CreateGroup {
                group: slug(g),
                name: g.into(),
                kind: Kind::Team,
            };
            apply(&mut t, &john(), &op, now()).unwrap();
        }
        let f = t
            .persons
            .get_mut(&slug("john"))
            .unwrap()
            .file
            .as_mut()
            .unwrap();
        assert_eq!(f.default_group, Some(slug("third")));
        f.default_group = None;
        f.framed = vec![slug("evil"), slug("johns"), slug("third")];
        let fifth = Op::CreateGroup {
            group: slug("fifth"),
            name: "fifth".into(),
            kind: Kind::Team,
        };
        apply(&mut t, &john(), &fifth, now()).unwrap();
        assert_eq!(t.person_file(&slug("john")).unwrap().default_group, None);
        clean(&t);
    }

    /// **Whoever made a person before their first account chose nothing for
    /// them**: Stranger makes "mum" in a group of theirs; the admin listing
    /// her account through @sharratt makes @sharratt her default.
    #[test]
    fn a_first_account_starts_them_where_the_admin_lists_it() {
        let mut t = sharratt();
        let evil = Op::CreateGroup {
            group: slug("evil"),
            name: "Evil".into(),
            kind: Kind::Team,
        };
        apply(&mut t, &stranger(), &evil, now()).unwrap();
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
        let p = t.person_file(&slug("mum")).unwrap();
        assert_eq!(p.groups, [slug("evil"), slug("sharratt")]);
        assert_eq!(p.default_group, Some(slug("sharratt")));
        assert!(p.framed.is_empty());
        clean(&t);
    }

    /// **A person chooses their own default and framed groups**, within the
    /// rules of person.yaml.
    #[test]
    fn a_person_chooses_which_groups_open_their_conversations() {
        let mut t = sharratt();
        for g in ["work", "club"] {
            let op = Op::CreateGroup {
                group: slug(g),
                name: g.into(),
                kind: Kind::Team,
            };
            apply(&mut t, &john(), &op, now()).unwrap();
        }
        let a = apply(
            &mut t,
            &john(),
            &Op::ChooseGroups {
                default: Some(slug("work")),
                framed: vec![slug("sharratt")],
            },
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "john opens with @work, @sharratt");
        assert_eq!(a.owners, vec![Owner::Person(slug("john"))]);
        let p = t.person_file(&slug("john")).unwrap();
        assert_eq!(p.default_group, Some(slug("work")));
        assert_eq!(p.framed, [slug("sharratt")]);
        let keep = Op::ChooseGroups {
            default: None,
            framed: vec![],
        };
        assert_eq!(
            apply(&mut t, &john(), &keep, now()).unwrap().subject,
            "john opens with @work"
        );
        assert_eq!(
            apply(&mut t, &john(), &keep, now())
                .unwrap_err()
                .to_string(),
            "john's groups open as chosen already"
        );
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &Op::ChooseGroups {
                    default: Some(slug("nowhere")),
                    framed: vec![],
                },
                now()
            )
            .unwrap_err()
            .to_string(),
            "`default_group` is nowhere, which is not one of their groups"
        );
        clean(&t);
    }

    /// **A bot joins as a reader and leaves alone**: no person.yaml is
    /// touched either way.
    #[test]
    fn a_bot_joins_as_a_reader_and_leaves() {
        let mut t = sharratt();
        let a = apply(
            &mut t,
            &john(),
            &Op::AddBot {
                group: slug("sharratt"),
                bot: slug("d00d"),
            },
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "Add bot d00d to @sharratt as reader");
        assert_eq!(a.owners, vec![Owner::Group(slug("sharratt"))]);
        assert!(bot_may_read(
            &t,
            &slug("d00d"),
            &Owner::Group(slug("sharratt"))
        ));
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &Op::AddBot {
                    group: slug("sharratt"),
                    bot: slug("d00d"),
                },
                now()
            )
            .unwrap_err()
            .to_string(),
            "bot d00d reads @sharratt already"
        );
        let gone = apply(
            &mut t,
            &john(),
            &Op::RemoveMember {
                group: slug("sharratt"),
                who: Who::Bot(slug("d00d")),
            },
            now(),
        )
        .unwrap();
        assert_eq!(gone.subject, "Remove bot d00d from @sharratt");
        assert_eq!(gone.owners, vec![Owner::Group(slug("sharratt"))]);
        assert!(!bot_may_read(
            &t,
            &slug("d00d"),
            &Owner::Group(slug("sharratt"))
        ));
        clean(&t);
    }

    #[test]
    fn an_eleventh_group_is_refused() {
        let mut t = sharratt();
        for i in 0..9 {
            let op = Op::CreateGroup {
                group: slug(&format!("g{i}")),
                name: format!("G{i}"),
                kind: Kind::Team,
            };
            apply(&mut t, &john(), &op, now()).unwrap();
        }
        let op = Op::CreateGroup {
            group: slug("g9"),
            name: "G9".into(),
            kind: Kind::Team,
        };
        assert_eq!(
            apply(&mut t, &john(), &op, now()).unwrap_err().to_string(),
            "john belongs to 10 groups already: at most 10"
        );
        assert!(
            !t.groups.contains_key(&slug("g9")),
            "a refusal makes nothing"
        );
    }

    #[test]
    fn an_account_binds_once_to_its_email() {
        let mut t = sharratt();
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
