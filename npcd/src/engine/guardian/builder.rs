//! Assembling a [`Guardian`] from modules, either piece by piece or from the
//! configuration file.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};

use crate::engine::guardian::config::{GuardianConfig, ModuleSpec};
use crate::engine::guardian::ladder::{Ladder, Rung};
use crate::engine::guardian::log::GuardianLog;
use crate::engine::guardian::module::Module;
use crate::engine::guardian::modules::{Drift, Journal, Looping, Stall, StepTracker};
use crate::engine::guardian::runner::Guardian;

#[derive(Default)]
pub struct GuardianBuilder {
    modules: Vec<Box<dyn Module>>,
    rungs: Vec<Rung>,
    scan_every: Duration,
    cooldown: Duration,
    settle: Duration,
}

fn module_of(spec: &ModuleSpec) -> Box<dyn Module> {
    match spec {
        ModuleSpec::Drift => Box::new(Drift),
        ModuleSpec::Looping => Box::new(Looping),
        ModuleSpec::Stall { no_progress_secs } => {
            Box::new(Stall::new(Duration::from_secs(*no_progress_secs)))
        }
        ModuleSpec::StepTracker { confirmations } => Box::new(StepTracker::new(*confirmations)),
        ModuleSpec::Journal => Box::new(Journal),
    }
}

impl GuardianBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder holding everything a configuration file says.
    pub fn from_config(config: &GuardianConfig) -> Self {
        Self {
            modules: config.modules.iter().map(module_of).collect(),
            rungs: config.escalation.clone(),
            scan_every: Duration::from_secs(config.scan_every_secs),
            cooldown: Duration::from_secs(config.cooldown_secs),
            settle: Duration::from_secs(config.settle_secs),
        }
    }

    pub fn module(mut self, module: Box<dyn Module>) -> Self {
        self.modules.push(module);
        self
    }

    pub fn escalation(mut self, rungs: Vec<Rung>) -> Self {
        self.rungs = rungs;
        self
    }

    pub fn scan_every(mut self, every: Duration) -> Self {
        self.scan_every = every;
        self
    }

    pub fn cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    pub fn settle(mut self, settle: Duration) -> Self {
        self.settle = settle;
        self
    }

    pub fn build(self) -> Result<Guardian> {
        if self.modules.is_empty() {
            bail!("a guardian needs at least one module");
        }
        if self.rungs.is_empty() {
            bail!("a guardian needs at least one rung of escalation");
        }
        if self.scan_every.is_zero() {
            bail!("scan_every must be more than zero");
        }
        if self.settle > self.cooldown {
            bail!("settle must not be longer than cooldown: a nudge is judged before the next may be given");
        }
        let ladder = Ladder::new(self.rungs, self.cooldown, self.settle);
        Ok(Guardian::new(
            self.modules,
            ladder,
            self.scan_every,
            Arc::new(GuardianLog::new()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    fn complete() -> GuardianBuilder {
        GuardianBuilder::new()
            .module(Box::new(Drift))
            .escalation(vec![Rung::Nudge])
            .scan_every(secs(10))
            .cooldown(secs(60))
            .settle(secs(30))
    }

    #[test]
    fn a_complete_builder_builds() {
        assert!(complete().build().is_ok());
    }

    #[test]
    fn a_guardian_with_no_modules_is_refused() {
        let b = GuardianBuilder::new()
            .escalation(vec![Rung::Nudge])
            .scan_every(secs(10));
        assert!(b.build().is_err());
    }

    #[test]
    fn a_guardian_with_no_rungs_is_refused() {
        let b = GuardianBuilder::new()
            .module(Box::new(Drift))
            .scan_every(secs(10));
        assert!(b.build().is_err());
    }

    #[test]
    fn a_zero_scan_interval_is_refused() {
        assert!(complete().scan_every(Duration::ZERO).build().is_err());
    }

    #[test]
    fn a_settle_window_longer_than_the_cooldown_is_refused() {
        assert!(complete().settle(secs(61)).build().is_err());
    }

    #[test]
    fn a_config_becomes_the_modules_it_names() {
        let config = GuardianConfig::parse(
            "
enabled: true
scan_every_secs: 45
cooldown_secs: 180
settle_secs: 60
modules:
  - kind: drift
  - kind: stall
    no_progress_secs: 300
escalation: [nudge, flag]
",
        )
        .unwrap();
        let guardian = GuardianBuilder::from_config(&config).build().unwrap();
        let names: Vec<_> = guardian.modules.iter().map(|m| m.name()).collect();
        assert_eq!(names, vec!["drift", "stall"]);
        assert_eq!(guardian.scan_every, secs(45));
    }
}
