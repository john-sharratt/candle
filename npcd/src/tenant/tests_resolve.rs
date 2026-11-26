//! Tests how npcd resolves its settings before anything is opened or stood up.
//!
//! A mind repository is read from the substrate rather than a folder, and nothing
//! is written to disk at resolve time. A relative data directory is made absolute
//! so the substrate's volume and mirror paths do not depend on the working
//! directory. A mind that arrives after start replaces the empty libraries and
//! registers in place, instead of restarting the daemon. npcd needs the same
//! model markers and dialect that zend needs, and npcd's model provides them.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use clap::Parser;

    use super::super::tests_flags::tests::{parse, Cli};

    /// **A mind repository is read from the substrate**: no folder, its
    /// schema not read at resolve — the substrate is not open yet — and
    /// nothing on disk made (`docs/substrate_git.md` §8); `mind_repo:` reads
    /// the same from `motherd.yaml`.
    #[test]
    fn a_mind_repository_is_read_from_the_substrate() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().to_str().unwrap();
        let t = parse(&["--mind-repo", "mind", "--data", data])
            .resolve()
            .unwrap();
        let repo = t.repo.clone().unwrap();
        assert_eq!(repo.name.as_str(), "mind");
        assert_eq!(repo.branch.as_str(), "main");
        assert_eq!(t.schema.dir, None, "a mind in the substrate has no folder");
        assert!(t.schema.files.is_none(), "read once the substrate opens");
        assert_eq!(t.data, dir.path());

        let branch = parse(&[
            "--mind-repo",
            "mind",
            "--data",
            data,
            "--mind-branch",
            "record",
        ])
        .resolve()
        .unwrap();
        assert_eq!(branch.repo.unwrap().branch.as_str(), "record");

        assert!(
            Cli::try_parse_from(["npcd", "--mind", "../mind"]).is_err(),
            "a folder is no mind"
        );
        let no_data = parse(&["--mind-repo", "mind"]).resolve();
        assert!(no_data
            .err()
            .unwrap()
            .to_string()
            .contains("--mind-repo keeps its store under --data"));
        let yaml: NpcdArgs =
            serde_yaml::from_str(&format!("mind_repo: mind\ndata: '{data}'\n")).unwrap();
        assert_eq!(yaml, parse(&["--mind-repo", "mind", "--data", data]));
    }

    /// **A relative `--data` is made absolute at resolve**, as `motherd.yaml`'s
    /// `data: ../npcd-data` is: the substrate's volume is measured from its
    /// path. Resolving touches nothing on disk.
    #[test]
    fn a_relative_data_directory_resolves_absolute() {
        let here = current_dir().unwrap();
        let t = parse(&["--mind-repo", "mind", "--data", "unmade-npcd-data"])
            .resolve()
            .unwrap();
        assert_eq!(t.data, here.join("unmade-npcd-data"));
        let plain = parse(&["--data", "unmade-npcd-data"]).resolve().unwrap();
        assert_eq!(plain.data, here.join("unmade-npcd-data"));
        assert!(!here.join("unmade-npcd-data").exists());
    }

    /// **A mind that arrives after start is read again in place**, not
    /// restarted onto: the libraries and registers read from no mind are
    /// empty, and those read from the one that arrived take their place in
    /// the state the routes answer from.
    #[test]
    fn libraries_read_from_a_mind_that_arrived_replace_the_empty_ones() {
        let dir = tempfile::tempdir().unwrap();
        let mind = dir.path().join("mind");
        std::fs::create_dir_all(mind.join("moods")).unwrap();
        std::fs::write(mind.join(projection::SCHEMA_FILE), "layers: []\n").unwrap();
        std::fs::write(mind.join("moods").join("calm.yaml"), "template: Calm.\n").unwrap();

        let (none, felt) = read_libraries(&projection::Source::bundled());
        assert_eq!(none.moods.len(), 0);
        assert!(felt.is_empty(), "no mind, no registers");

        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let npcs = Npcs::load(&data).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &data.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &data.join("images")).unwrap();
        let authored = Authored::new(
            Registry::load("world", data.join(WORLDS)).unwrap(),
            Registry::load("personality", data.join(PERSONALITIES)).unwrap(),
            accounts,
            npcs,
            Roles::default(),
            none,
            Mind::new(None),
            images,
            Default::default(),
        );
        assert_eq!(authored.libraries().moods.len(), 0);

        let files = Arc::new(MindFiles::folder(&mind));
        let schema = projection::Source::from_files(files, "arrived".to_string()).unwrap();
        let (libraries, feelings) = read_libraries(&schema);
        assert_eq!(feelings, ["calm"]);
        authored.set_libraries(libraries);
        let now = authored.libraries();
        assert_eq!(now.moods.len(), 1);
        assert_eq!(now.moods.sections[0].id, "calm");
    }

    /// npcd needs what zend needs of the model, and the prescribed one has it.
    #[test]
    fn npcd_needs_the_three_markers_and_the_qwen35_dialect() {
        let r = requires();
        assert_eq!(r.tokens, ["<tool_call>", "<think>", "</think>"]);
        assert_eq!(r.dialect, DialectType::Qwen35);
        assert_eq!(MODEL.spec().chat_format, DialectType::Qwen35);
    }
}
