//! Who is in a family (§3.5), and which accounts are a person (§3.4).

use crate::error::{Error, Result};
use crate::files::{
    Account, AccountKey, Email, FamilyFile, Member, PersonFile, Role, MAX_FAMILIES,
};
use crate::ops::Applied;
use crate::path::Owner;
use crate::scope::Scope;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{skeleton, Person, Tree};

fn family_file<'a>(tree: &'a mut Tree, family: &Slug) -> Result<&'a mut FamilyFile> {
    let f = tree
        .families
        .get_mut(family)
        .ok_or_else(|| Error::not_found(format!("there is no family @{family}")))?;
    f.file.as_mut().map_err(|p| {
        Error::refused(format!(
            "@{family} cannot change until family.yaml is fixed: {p}"
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

fn one_owner_left(file: &FamilyFile, person: &Slug) -> bool {
    file.owners().all(|m| &m.person == person)
}

#[allow(clippy::too_many_arguments)]
pub fn add_member(
    tree: &mut Tree,
    family: &Slug,
    person: &Slug,
    name: Option<&str>,
    emails: &[Email],
    role: Role,
    now: Stamp,
) -> Result<Applied> {
    family_file(tree, family)?;
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
                families: vec![family.clone()],
                default_family: Some(family.clone()),
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
            if !file.families.contains(family) {
                if file.families.len() >= MAX_FAMILIES {
                    return Err(Error::refused(format!(
                        "{person} belongs to {MAX_FAMILIES} families already: at most {MAX_FAMILIES}"
                    )));
                }
                file.families.push(family.clone());
                file.default_family.get_or_insert_with(|| family.clone());
                changed = true;
            }
            file.check().map_err(Error::Invalid)?;
        }
    }
    let fam = family_file(tree, family)?;
    let at = fam.members.iter().position(|m| &m.person == person);
    let subject = match at {
        Some(i) if fam.members[i].role == role => format!("{person} in @{family}"),
        Some(i) => {
            if fam.members[i].role == Role::Owner && one_owner_left(fam, person) {
                return Err(Error::refused(format!(
                    "{person} is @{family}'s last owner"
                )));
            }
            fam.members[i].role = role;
            changed = true;
            format!("{person} is now @{family}'s {}", role.name())
        }
        None => {
            fam.members.push(Member {
                person: person.clone(),
                role,
                joined: now,
            });
            changed = true;
            format!("Add {person} to @{family} as {}", role.name())
        }
    };
    if !changed {
        return Err(Error::refused(format!(
            "{person} is in @{family} as {} already",
            role.name()
        )));
    }
    fam.check().map_err(Error::Invalid)?;
    Ok(Applied {
        subject,
        owners: vec![Owner::Family(family.clone()), Owner::Person(person.clone())],
        ..Applied::default()
    })
}

pub fn remove_member(tree: &mut Tree, family: &Slug, person: &Slug) -> Result<Applied> {
    let fam = family_file(tree, family)?;
    let i = fam
        .members
        .iter()
        .position(|m| &m.person == person)
        .ok_or_else(|| Error::not_found(format!("{person} is not in @{family}")))?;
    if fam.members[i].role == Role::Owner && one_owner_left(fam, person) {
        return Err(Error::refused(format!(
            "{person} is @{family}'s last owner"
        )));
    }
    fam.members.remove(i);
    let mut owners = vec![Owner::Family(family.clone())];
    if let Some(file) = person_file(tree, person)? {
        file.families.retain(|f| f != family);
        if file.default_family.as_ref() == Some(family) {
            file.default_family = file.families.first().cloned();
        }
        owners.push(Owner::Person(person.clone()));
    }
    Ok(Applied {
        subject: format!("Remove {person} from @{family}"),
        owners,
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
    use crate::actor::{may_read, Actor};
    use crate::ops::fixture::{john, now, sharratt, yen};
    use crate::ops::{apply, Op};
    use crate::testing::slug;
    use crate::tree::check::check;
    use crate::tree::Family;

    fn add(person: &str, name: Option<&str>, emails: &[&str], role: Role) -> Op {
        Op::AddMember {
            family: slug("sharratt"),
            person: slug(person),
            name: name.map(Into::into),
            emails: emails.iter().map(|e| Email::parse(e).unwrap()).collect(),
            role,
        }
    }

    fn clean(t: &Tree) {
        let mut c = t.clone();
        for p in c.persons.values_mut() {
            p.problems.clear();
        }
        for f in c.families.values_mut() {
            f.problems.clear();
        }
        c.problems.clear();
        check(&mut c);
        assert!(
            c.problems.is_empty()
                && c.persons.values().all(|p| p.problems.is_empty())
                && c.families.values().all(|f| f.problems.is_empty())
        );
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
                Owner::Family(slug("sharratt")),
                Owner::Person(slug("yen-nhi"))
            ]
        );
        let p = t.person_file(&slug("yen-nhi")).unwrap();
        assert_eq!(p.families, vec![slug("sharratt")]);
        assert_eq!(p.default_family, Some(slug("sharratt")));
        assert_eq!(p.accounts[0].email.as_str(), "yen.nhi@example.com");
        assert_eq!(
            t.family_file(&slug("sharratt"))
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
        assert!(
            apply(&mut t, &john(), &add("x", None, &[], Role::Member), now())
                .unwrap_err()
                .to_string()
                .contains("give their name")
        );
        assert!(apply(
            &mut t,
            &john(),
            &add("x", Some("X"), &["yen@example.com"], Role::Member),
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("is yen's"));
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
        assert!(apply(
            &mut t,
            &john(),
            &add("yen", None, &["yen.other@example.net"], Role::Owner),
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("already"));
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
            &add("thanhlong", None, &[], Role::Member),
            now(),
        )
        .unwrap();
        assert!(apply(
            &mut t,
            &john(),
            &add("john", None, &[], Role::Member),
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("last owner"));
        assert!(apply(
            &mut t,
            &john(),
            &Op::RemoveMember {
                family: slug("sharratt"),
                person: slug("john")
            },
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("last owner"));
    }

    #[test]
    fn a_removed_member_reads_the_family_no_longer() {
        let mut t = sharratt();
        let a = apply(
            &mut t,
            &yen(),
            &Op::RemoveMember {
                family: slug("sharratt"),
                person: slug("thanhlong"),
            },
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "Remove thanhlong from @sharratt");
        let p = t.person_file(&slug("thanhlong")).unwrap();
        assert!(p.families.is_empty() && p.default_family.is_none());
        assert!(!may_read(
            &t,
            &slug("thanhlong"),
            &Owner::Family(slug("sharratt"))
        ));
        assert!(apply(
            &mut t,
            &yen(),
            &Op::RemoveMember {
                family: slug("sharratt"),
                person: slug("thanhlong")
            },
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("is not in"));
        clean(&t);
    }

    #[test]
    fn a_fourth_family_is_refused() {
        let mut t = sharratt();
        for f in ["a", "b"] {
            let file = FamilyFile {
                family: slug(f),
                name: f.into(),
                members: vec![Member {
                    person: slug("john"),
                    role: Role::Owner,
                    joined: now(),
                }],
            };
            t.families.insert(
                slug(f),
                Family {
                    file: Ok(file),
                    root: skeleton(Scope::Family),
                    problems: vec![],
                },
            );
            let pf = t
                .persons
                .get_mut(&slug("john"))
                .unwrap()
                .file
                .as_mut()
                .unwrap();
            pf.families.push(slug(f));
        }
        let file = FamilyFile {
            family: slug("c"),
            name: "c".into(),
            members: vec![Member {
                person: slug("yen"),
                role: Role::Owner,
                joined: now(),
            }],
        };
        t.families.insert(
            slug("c"),
            Family {
                file: Ok(file),
                root: skeleton(Scope::Family),
                problems: vec![],
            },
        );
        let op = Op::AddMember {
            family: slug("c"),
            person: slug("john"),
            name: None,
            emails: vec![],
            role: Role::Member,
        };
        assert!(apply(&mut t, &yen(), &op, now())
            .unwrap_err()
            .to_string()
            .contains("3 families already"));
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
        assert!(apply(&mut t, &Actor::System, &other, now())
            .unwrap_err()
            .to_string()
            .contains("bound already"));
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
