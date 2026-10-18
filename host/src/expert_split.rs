//! The CPU expert split's settings (`docs/cpu_expert_compute.md` §5.3.1): the
//! `expert_split:` section under `host:` in `motherd.yaml`, and the same
//! settings as `zend`/`npcd` flags. Deployment config — whether this machine
//! computes some routed experts on its CPU, and the knobs the split starts
//! from — applied once, before the engine loads. `POST /v1/expert_split`
//! changes the knobs live.

use clap::Args;
use serde::Deserialize;

use candle_conversation::expert_split::{shared, Knobs, Promote};

/// The split's settings. Defaults are the engine's: on, and neutral knobs.
#[derive(Args, Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ExpertSplitArgs {
    /// Compute some routed experts on the host CPU: the pinned misses the
    /// device judges cheaper there than over the link. The pool calibrates
    /// at load, and the split stays all-GPU until it has. On by default;
    /// `--expert-split false` (`enabled: false`) keeps every expert on the GPU.
    #[arg(
        long = "expert-split",
        id = "expert_split_enabled",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    pub enabled: bool,

    /// Most CPU assignments one layer may take. 0 keeps every expert on the
    /// GPU. Sizes the pool's mapped rings at load; a live change can lower it
    /// but not raise it past that.
    #[arg(long = "expert-split-max-assignments", default_value_t = Knobs::default().max_cpu_assignments)]
    pub max_cpu_assignments: u32,

    /// Multiplies every CPU cost; above 1 biases the split to the GPU.
    #[arg(long = "expert-split-cpu-scale", default_value_t = Knobs::default().cpu_scale)]
    pub cpu_scale: f32,

    /// Multiplies the cost of copying an expert over the link.
    #[arg(long = "expert-split-gpu-scale", default_value_t = Knobs::default().gpu_scale)]
    pub gpu_scale: f32,

    /// Nanoseconds added to the measured handoff.
    #[arg(long = "expert-split-handoff-extra-ns", default_value_t = Knobs::default().handoff_extra_ns)]
    pub handoff_extra_ns: f32,

    /// An expert stays on the GPU when its CPU compute exceeds this times
    /// what the GPU would spend on it.
    #[arg(long = "expert-split-cut-scale", default_value_t = Knobs::default().cut_scale)]
    pub cut_scale: f32,

    /// Nanoseconds the CPU path must save over all-GPU before it is taken.
    #[arg(long = "expert-split-min-gain-ns", default_value_t = Knobs::default().min_gain_ns)]
    pub min_gain_ns: f32,

    /// Nanoseconds the device polls tightly for the pool's results before it
    /// backs off.
    #[arg(long = "expert-split-wait-tight-ns", default_value_t = Knobs::default().wait_tight_ns)]
    pub wait_tight_ns: u64,

    /// Invocations the pool may go without work before one explores, sending
    /// the cheapest miss when it loses by at most `explore_ns`. 0 never
    /// explores.
    #[arg(long = "expert-split-explore-period", default_value_t = Knobs::default().explore_period)]
    pub explore_period: u32,

    /// Nanoseconds an exploring invocation's CPU expert may lose by.
    #[arg(long = "expert-split-explore-ns", default_value_t = Knobs::default().explore_ns)]
    pub explore_ns: f32,

    /// Milliseconds: the time constant at which the measured CPU costs follow
    /// what the pool actually takes.
    #[arg(long = "expert-split-online-tau-ms", default_value_t = Knobs::default().online_tau_ms)]
    pub online_tau_ms: f32,

    /// Residency: the weight of the pack read in what evicting an NVMe-only
    /// expert costs. Above 1 keeps NVMe-backed experts resident ahead of
    /// RAM-backed ones of equal heat. Applies whether or not the split runs.
    #[arg(long = "expert-split-cold-bias", default_value_t = Knobs::default().cold_bias)]
    pub cold_bias: f32,

    /// Which experts the CPU served the next layer copies into VRAM behind
    /// it: `none`, `cold` (those read from the pack for the CPU) or `all`.
    #[arg(long = "expert-split-promote", default_value_t = Knobs::default().promote)]
    pub promote: Promote,
}

impl Default for ExpertSplitArgs {
    fn default() -> Self {
        let k = Knobs::default();
        Self {
            enabled: true,
            max_cpu_assignments: k.max_cpu_assignments,
            cpu_scale: k.cpu_scale,
            gpu_scale: k.gpu_scale,
            handoff_extra_ns: k.handoff_extra_ns,
            cut_scale: k.cut_scale,
            min_gain_ns: k.min_gain_ns,
            wait_tight_ns: k.wait_tight_ns,
            explore_period: k.explore_period,
            explore_ns: k.explore_ns,
            online_tau_ms: k.online_tau_ms,
            cold_bias: k.cold_bias,
            promote: k.promote,
        }
    }
}

impl ExpertSplitArgs {
    pub fn knobs(&self) -> Knobs {
        Knobs {
            cpu_scale: self.cpu_scale,
            gpu_scale: self.gpu_scale,
            handoff_extra_ns: self.handoff_extra_ns,
            cut_scale: self.cut_scale,
            min_gain_ns: self.min_gain_ns,
            max_cpu_assignments: self.max_cpu_assignments,
            wait_tight_ns: self.wait_tight_ns,
            explore_period: self.explore_period,
            explore_ns: self.explore_ns,
            online_tau_ms: self.online_tau_ms,
            cold_bias: self.cold_bias,
            promote: self.promote,
        }
    }

    /// Writes these settings into the process's split. Called before the
    /// engine loads, which is when the pool reads whether to start.
    pub fn apply(&self) {
        let params = shared();
        let mut p = params.lock().unwrap_or_else(|e| e.into_inner());
        p.enabled = self.enabled;
        p.knobs = self.knobs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        split: ExpertSplitArgs,
    }

    /// The flags and the YAML read the same settings, and both default to the
    /// engine's own knobs, on; either turns it off.
    #[test]
    fn flags_and_yaml_agree_and_default_to_the_engines_knobs() {
        let flags = Cli::try_parse_from([
            "x",
            "--expert-split",
            "--expert-split-max-assignments",
            "256",
            "--expert-split-cpu-scale",
            "1.25",
            "--expert-split-wait-tight-ns",
            "50000",
            "--expert-split-promote",
            "all",
        ])
        .unwrap()
        .split;
        let yaml: ExpertSplitArgs = serde_yaml::from_str(
            "enabled: true\nmax_cpu_assignments: 256\ncpu_scale: 1.25\nwait_tight_ns: 50000\n\
             promote: all\n",
        )
        .unwrap();
        assert_eq!(flags, yaml);
        assert!(flags.enabled);
        assert_eq!(flags.knobs().max_cpu_assignments, 256);
        assert_eq!(flags.knobs().promote, Promote::All);
        assert!(Cli::try_parse_from(["x", "--expert-split-promote", "some"]).is_err());
        assert_eq!(flags.knobs().gpu_scale, 1.0);

        let none = Cli::try_parse_from(["x"]).unwrap().split;
        assert_eq!(none, ExpertSplitArgs::default());
        assert!(none.enabled);
        assert_eq!(none.knobs(), Knobs::default());
        let off = Cli::try_parse_from(["x", "--expert-split", "false"])
            .unwrap()
            .split;
        assert!(!off.enabled);
        assert_eq!(
            off,
            serde_yaml::from_str::<ExpertSplitArgs>("enabled: false\n").unwrap()
        );
        assert_eq!(
            serde_yaml::from_str::<ExpertSplitArgs>("{}").unwrap(),
            ExpertSplitArgs::default()
        );
        assert!(serde_yaml::from_str::<ExpertSplitArgs>("cpu_sclae: 2\n").is_err());
    }
}
