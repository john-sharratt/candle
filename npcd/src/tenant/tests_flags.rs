//! Pins the command-line and `motherd.yaml` settings that control what npcd runs,
//! and that the console polls only through its visibility helper.
//!
//! Each behaviour-changing setting is off unless asked for, and the flag and the
//! YAML key read the same setting with the same default. A restart must not
//! forget anything by accident, and a wipe is never implied by a narrower flag.
//! The one-shot flags are real flags of the binary and cannot come from YAML, so
//! a relaunch that strips them does not repeat their action. The console test
//! walks every file under its directory and fails if an interval starts anywhere
//! other than the visibility helper or the mock backend.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use clap::Parser;

    use include_dir::File;

    /// Every file under `dir`, at any depth.
    pub(in super::super) fn every_file<'a>(dir: &'a Dir<'a>, out: &mut Vec<&'a File<'a>>) {
        out.extend(dir.files());
        for sub in dir.dirs() {
            every_file(sub, out);
        }
    }

    /// **The console polls only while it is seen**, so a hidden tab does not
    /// load the server: every timed poll goes through `/common/lib/visible.js`, so the only
    /// files that start an interval of their own are the landing page's
    /// animation and the mock backend.
    #[test]
    fn the_console_polls_only_through_the_visibility_helper() {
        let mut files = Vec::new();
        every_file(&CONSOLE, &mut files);
        let mut timed: Vec<String> = files
            .into_iter()
            .filter(|f| {
                f.contents_utf8()
                    .is_some_and(|text| text.contains("setInterval("))
            })
            .map(|f| f.path().to_string_lossy().replace('\\', "/"))
            .collect();
        timed.sort();
        assert_eq!(timed, ["lib/api.mock.js", "pages/landing.js"]);
    }

    /// **What npcd asks without loading is what its load asks** — its prefill
    /// pass — and no guests: a host running zend alone seats this in npcd's
    /// place, and nothing there would call a guest.
    #[test]
    fn the_engine_requests_carry_npcds_prefill_pass_and_no_guests() {
        let asks = NpcdArgs::default().engine_requests();
        assert_eq!(asks.prefill_pass_tokens, Some(8192));
        assert!(asks.guests.is_empty());
    }

    #[derive(Parser)]
    pub(in super::super) struct Cli {
        #[command(flatten)]
        pub(in super::super) npcd: NpcdArgs,
    }

    pub(in super::super) fn parse(args: &[&str]) -> NpcdArgs {
        let mut all = vec!["npcd"];
        all.extend_from_slice(args);
        Cli::try_parse_from(all).unwrap().npcd
    }

    /// **Off unless asked for**, each flag its own: a restart must not forget
    /// anything by accident, and a wipe is never implied by a narrower flag.
    #[test]
    fn forgetting_is_only_ever_asked_for() {
        let none = parse(&[]);
        assert!(!none.forget_conversations && !none.forget_dreams);
        assert!(!none.forget_journal && !none.wipe_conversations);
        let both = parse(&["--forget-conversations", "--forget-dreams"]);
        assert!(both.forget_conversations && both.forget_dreams && !both.wipe_conversations);
        let journal = parse(&["--forget-journal"]);
        assert!(journal.forget_journal && !journal.forget_conversations);
        assert!(parse(&["--wipe-conversations"]).wipe_conversations);
    }

    /// The flags and `motherd.yaml`'s `npcd:` section read the same settings,
    /// with the same defaults; there is no `--content` and no embedded console.
    #[test]
    fn flags_and_yaml_agree() {
        assert_eq!(parse(&[]), NpcdArgs::default());
        assert_eq!(
            serde_yaml::from_str::<NpcdArgs>("{}").unwrap(),
            NpcdArgs::default()
        );
        let flags = parse(&["--mind-repo", "battle-mind", "--log-identity"]);
        let yaml: NpcdArgs =
            serde_yaml::from_str("mind_repo: battle-mind\nlog_identity: true\n").unwrap();
        assert_eq!(flags, yaml);
        assert!(Cli::try_parse_from(["npcd", "--content", "web/content/npcd"]).is_err());
        assert!(serde_yaml::from_str::<NpcdArgs>("content: x").is_err());
    }

    /// **A member runs characters unless told not to**, by flag or by
    /// `motherd.yaml`, the two reading the same setting.
    #[test]
    fn characters_run_unless_turned_off() {
        assert!(parse(&[]).characters);
        assert!(!parse(&["--no-characters"]).characters);
        let yaml: NpcdArgs = serde_yaml::from_str("characters: false\n").unwrap();
        assert_eq!(yaml, parse(&["--no-characters"]));
    }

    /// **A member writes the mind unless told not to**, by flag or by
    /// `motherd.yaml`, the two reading the same setting.
    #[test]
    fn the_mind_is_ingested_unless_turned_off() {
        assert!(parse(&[]).ingest);
        assert!(!parse(&["--no-ingest"]).ingest);
        let yaml: NpcdArgs = serde_yaml::from_str("ingest: false\n").unwrap();
        assert_eq!(yaml, parse(&["--no-ingest"]));
    }

    /// **A member hosts worlds without characters only when told to**, by flag
    /// or by `motherd.yaml`, the two reading the same setting.
    #[test]
    fn worlds_are_hosted_without_characters_only_when_turned_on() {
        assert!(!parse(&[]).host_worlds);
        let flags = parse(&["--no-characters", "--host-worlds"]);
        assert!(!flags.characters && flags.host_worlds);
        let yaml: NpcdArgs =
            serde_yaml::from_str("characters: false\nhost_worlds: true\n").unwrap();
        assert_eq!(yaml, flags);
    }

    /// Every one-shot flag is a real flag of the binary — a relaunch strips
    /// what the shell passed — and none can be set from `motherd.yaml`.
    #[test]
    fn the_one_shot_flags_are_flags_and_never_yaml() {
        for o in NpcdArgs::ONE_SHOT {
            assert!(Cli::try_parse_from(["npcd", o.flag]).is_ok(), "{}", o.flag);
            let key = o.flag.trim_start_matches("--").replace('-', "_");
            assert!(
                serde_yaml::from_str::<NpcdArgs>(&format!("{key}: true")).is_err(),
                "{key} must not be settable from motherd.yaml"
            );
        }
    }
}
