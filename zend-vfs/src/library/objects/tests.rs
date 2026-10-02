use super::*;
use crate::testing::TestRepo;
use crate::types::GitTime;

fn who(name: &str, seconds: i64) -> Signature {
    Signature::new(
        name,
        "a@example.com",
        GitTime {
            seconds,
            offset_minutes: 90,
        },
    )
    .unwrap()
}

/// `hello\n` is the well-known blob `ce0136…`, and is stored.
#[test]
fn a_blob_is_stored_unconverted_under_its_standard_id() {
    let t = TestRepo::init();
    let repo = t.repo();
    let lib = repo.library().unwrap();
    let id = hash_raw(&lib, b"hello\n").unwrap();
    assert_eq!(id.as_str(), "ce013625030ba8dba906f756967f9e9ca394464a");
    assert_eq!(blob_bytes(&lib, &id).unwrap(), Some(b"hello\n".to_vec()));
    let absent = Oid::parse(&"1".repeat(40)).unwrap();
    assert_eq!(blob_bytes(&lib, &absent).unwrap(), None);
    assert_eq!(t.git(&["cat-file", "blob", id.as_str()]), "hello\n");
}

/// **The tree is the one `git` makes of the same entries**, whatever
/// order they are given in.
#[test]
fn a_tree_from_entries_is_the_tree_git_writes() {
    let t = TestRepo::init();
    t.write("b.txt", b"b\n");
    t.write("a/z.txt", b"z\n");
    t.write("a.txt", b"a\n");
    t.git(&["add", "."]);
    let by_git = t.git(&["write-tree"]).trim().to_string();
    let repo = t.repo();
    let lib = repo.library().unwrap();
    let blob = |path: &str| {
        let sha = t.git(&["rev-parse", &format!(":{path}")]);
        Oid::parse(sha.trim()).unwrap()
    };
    let entries = vec![
        (FileMode::Regular, blob("b.txt"), "b.txt".to_string()),
        (FileMode::Regular, blob("a/z.txt"), "a/z.txt".to_string()),
        (FileMode::Regular, blob("a.txt"), "a.txt".to_string()),
    ];
    assert_eq!(tree_from_entries(&lib, &entries).unwrap().as_str(), by_git);
    assert_eq!(index_tree(&lib).unwrap().unwrap().as_str(), by_git);
}

/// **The commit is byte for byte `commit-tree`'s**: the same id for the
/// same tree, parents, message, identities and times.
#[test]
fn a_commit_has_the_id_commit_tree_gives_it() {
    let t = TestRepo::init();
    t.write("a.txt", b"a\n");
    let parent = t.commit_all("first");
    let tree = Oid::parse(t.git(&["rev-parse", "HEAD^{tree}"]).trim()).unwrap();
    let (author, committer) = (who("Ann", 1_700_000_000), who("Bo", 1_700_000_500));
    let repo = t.repo();
    let message = "subject\n\nbody\n";
    let by_library = repo
        .commit_tree(&tree, &[&parent], message, &author, &committer)
        .unwrap();
    let by_git = repo
        .without_library()
        .commit_tree(&tree, &[&parent], message, &author, &committer)
        .unwrap();
    assert_eq!(by_library, by_git);
}

/// The listing carries every entry's mode and the two flags git is told to
/// leave an entry alone by.
#[test]
fn the_index_listing_carries_modes_and_the_leave_alone_flags() {
    let t = TestRepo::init();
    t.write("plain.txt", b"p\n");
    t.write("hidden.txt", b"h\n");
    t.write("same.txt", b"s\n");
    t.write("both.txt", b"b\n");
    t.commit_all("base");
    t.git(&["update-index", "--skip-worktree", "hidden.txt", "both.txt"]);
    t.git(&["update-index", "--assume-unchanged", "same.txt", "both.txt"]);
    let repo = t.repo();
    let lib = repo.library().unwrap();
    let listing: Vec<(String, FileMode, bool, bool)> = index_listing(&lib)
        .unwrap()
        .into_iter()
        .map(|e| (e.path, e.mode, e.assume_unchanged, e.skip_worktree))
        .collect();
    let regular = FileMode::Regular;
    assert_eq!(
        listing,
        vec![
            ("both.txt".to_string(), regular, true, true),
            ("hidden.txt".to_string(), regular, false, true),
            ("plain.txt".to_string(), regular, false, false),
            ("same.txt".to_string(), regular, true, false),
        ]
    );
}

/// An index holding an intent-to-add entry writes the tree without it, as
/// `write-tree` does.
#[test]
fn an_intent_to_add_entry_is_left_out_of_the_tree() {
    let t = TestRepo::init();
    t.write("kept.txt", b"k\n");
    t.commit_all("base");
    t.write("later.txt", b"l\n");
    t.git(&["add", "--intent-to-add", "later.txt"]);
    let by_git = t.git(&["write-tree"]).trim().to_string();
    let repo = t.repo();
    let lib = repo.library().unwrap();
    assert_eq!(index_tree(&lib).unwrap().unwrap().as_str(), by_git);
}
