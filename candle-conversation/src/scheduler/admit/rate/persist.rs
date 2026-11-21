//! Saves what the model has learned about a card as text, and restores it at the
//! next start.
//!
//! The export is the model rather than the figures it last published, so a restarted
//! member prices its state afresh and does not keep advertising a loaded card's step
//! while it idles. Import refuses a model learned on another geometry and leaves the
//! fresh model as it was. The link rate is this run's own, although a forward that
//! copies faster than the probe still raises it, because the probe can read low at
//! startup.

use super::*;

impl WaveRate {
    /// What this model has learned about the card, as text a restart can read
    /// back ([`Self::import_learned`]): the copy rate, the layer time, the hit
    /// exponent, the forward and decode fits and the depths they were taught
    /// at — what it would take a run of forwards to learn again, and not what
    /// one pass judged.
    ///
    /// **Kept as the model, not as its figures.** The figures a member
    /// published last were priced at the state it stood in then — a loaded
    /// card, wide and deep — and kept as figures they are what a restarted
    /// member advertises until it runs a forward. An idle one never does: on
    /// WhiteWorld (ed4d8c781, 2026-10-20) its kept 208 ms step and 961 ms burst
    /// stood for a whole run, priced it out of all 87 turns, and so nothing
    /// came to correct them. Kept as the model, the figures are priced afresh
    /// for the state the member is in.
    pub fn export_learned(&self) -> String {
        let g = &self.geometry;
        let m = &self.model;
        let mut out = format!(
            "geometry {} {} {}\nbw {}\nlayer {}\nhit_exponent {}\nmodel {} {} {} {}\n\
             context_per_row {}\ndecode_context {}\nsamples {} {} {}\n",
            g.moe_layers,
            g.experts_per_layer,
            g.slot_bytes,
            self.bw_bytes_per_s,
            self.layer_secs,
            self.decode.hit_exponent,
            m.fixed_secs,
            m.overhead_secs,
            m.compute_secs_per_token,
            m.attend_secs_per_pair,
            self.context_per_row,
            self.decode_context,
            self.samples,
            self.decode_samples,
            self.hit_samples,
        );
        let row = |v: &[f64]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        };
        out.push_str(&format!(
            "fit {} {} {}\n",
            self.fit.n,
            row(&self.fit.gram.concat()),
            row(&self.fit.rhs)
        ));
        let d = &self.decode_fit;
        out.push_str(&format!(
            "decode_fit {} {} {} {}\n",
            d.n,
            d.tt,
            row(&d.xx.concat()),
            row(&d.xt)
        ));
        if let Some((lo, hi)) = d.miss_range {
            out.push_str(&format!("miss_range {lo} {hi}\n"));
        }
        out
    }

    /// Take back what [`Self::export_learned`] wrote — `false`, and nothing
    /// changed, when it is unreadable or was learned on another model's
    /// geometry. The link's measured rate is this run's, and stays.
    pub fn import_learned(&mut self, text: &str) -> bool {
        let mut fields: std::collections::HashMap<&str, Vec<&str>> = Default::default();
        for line in text.lines() {
            let mut words = line.split_whitespace();
            if let Some(key) = words.next() {
                fields.insert(key, words.collect());
            }
        }
        let nums = |key: &str| -> Option<Vec<f64>> {
            fields
                .get(key)?
                .iter()
                .map(|w| w.parse::<f64>().ok())
                .collect()
        };
        let Some(g) = nums("geometry") else {
            return false;
        };
        let geo = &self.geometry;
        if g != [
            geo.moe_layers as f64,
            geo.experts_per_layer as f64,
            geo.slot_bytes as f64,
        ] {
            return false;
        }
        let one = |key: &str| nums(key).and_then(|v| v.first().copied());
        let (Some(bw), Some(layer), Some(hit), Some(model), Some(ctx), Some(dctx), Some(samples)) = (
            one("bw"),
            one("layer"),
            one("hit_exponent"),
            nums("model"),
            one("context_per_row"),
            one("decode_context"),
            nums("samples"),
        ) else {
            return false;
        };
        let (Some(fit), Some(dfit)) = (nums("fit"), nums("decode_fit")) else {
            return false;
        };
        if model.len() != 4
            || samples.len() != 3
            || fit.len() != 1 + 16 + 4
            || dfit.len() != 2 + 25 + 5
        {
            return false;
        }
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(bw) || !positive(layer) || !positive(hit) {
            return false;
        }
        // A rate the card's own forwards reached is a link it has: it corrects
        // a probe that read low at this start, as a forward faster than the
        // probe does in a run. WhiteWorld's probe read 6.2 GB/s on one start
        // and 12.7 on the one before (502c311db, 2026-10-20), the card and
        // its bus unchanged.
        self.link_bytes_per_s = self.link_bytes_per_s.max(bw);
        self.bw_bytes_per_s = bw;
        self.layer_secs = layer;
        self.decode.hit_exponent = hit;
        self.model.fixed_secs = model[0];
        self.model.overhead_secs = model[1];
        self.model.compute_secs_per_token = model[2];
        self.model.attend_secs_per_pair = model[3];
        self.context_per_row = ctx;
        self.decode_context = dctx;
        self.samples = samples[0] as u64;
        self.decode_samples = samples[1] as u64;
        self.hit_samples = samples[2] as u64;
        self.fit.n = fit[0] as u64;
        for i in 0..4 {
            for j in 0..4 {
                self.fit.gram[i][j] = fit[1 + i * 4 + j];
            }
            self.fit.rhs[i] = fit[17 + i];
        }
        let d = &mut self.decode_fit;
        d.n = dfit[0] as u64;
        d.tt = dfit[1];
        for i in 0..5 {
            for j in 0..5 {
                d.xx[i][j] = dfit[2 + i * 5 + j];
            }
            d.xt[i] = dfit[27 + i];
        }
        d.miss_range = nums("miss_range")
            .filter(|v| v.len() == 2 && v.iter().all(|x| x.is_finite()))
            .map(|v| (v[0], v[1]));
        d.cached = Some(d.compute());
        true
    }
}
