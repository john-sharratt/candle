//! Resolving a model pack and reading its header before any of it is loaded.
//!
//! A pack's header records its sources' digests and its tokenizer, which is enough
//! to compute the model's identity with no weight loaded. A pack without a tokenizer
//! refuses to start, because the key a substrate is checked against hashes the
//! vocabulary. The tokenizer is checked against the checkpoint's own token table at
//! the ids the model reserves meaning for, not against the entry count, because
//! added tokens sit at arbitrary high ids. A local checkpoint is packed into the
//! model cache, never beside the file, and is never deleted. A preset's pack is
//! pinned by every coordinate, revisions included, so its weights cannot move under
//! it.

use super::*;

/// A model pack [`ModelBuilder::prepare`] resolved and read: its path, its
/// tokenizer, and the vocabulary size its header states.
pub(super) struct Prepared {
    pub(super) pack: PathBuf,
    pub(super) tokenizer: tokenizers::Tokenizer,
    pub(super) gguf_vocab_size: Option<usize>,
}

// Read-back & terminal methods

impl ModelBuilder {
    /// Every file this model's weights are loaded from: its model pack, which
    /// already holds any tensor overrides and gate donor, and — for the dense
    /// lineage — its adapters. Resolved once and handed to both
    /// [`Self::model_identity`] and [`Self::load_model`], so the key names
    /// exactly what was loaded.
    pub fn weight_files(&self, pack: &Path) -> crate::Result<WeightFiles> {
        let adapters = if matches!(self.spec.arch, ModelArch::Qwen35Dense) {
            self.resolve_loras()?
        } else {
            Vec::new()
        };
        Ok(WeightFiles {
            pack: pack.to_path_buf(),
            adapters,
        })
    }

    /// The identity of the model loaded from `weights` — what a substrate is
    /// bound to. Reads the pack's header, which records its sources' digests and
    /// carries its `tokenizer.json`; no weight is loaded. A pack whose sources
    /// or tokenizer cannot be read refuses the start: a model whose key cannot
    /// be computed cannot be checked against a substrate.
    pub fn model_identity(&self, weights: &WeightFiles) -> crate::Result<ModelIdentity> {
        let tokenizer_json = ModelPack::read_header(&weights.pack)
            .and_then(|h| {
                h.metadata
                    .get(TOKENIZER_JSON)
                    .ok_or_else(|| candle::Error::Msg(format!("no {TOKENIZER_JSON}")))?
                    .to_string()
                    .map(|t| t.as_bytes().to_vec())
            })
            .map_err(|e| {
                ConversationError::Tokenizer(format!(
                    "read the tokenizer of {} to identify the model: {e}",
                    weights.pack.display()
                ))
            })?;
        // A prepared model is the release its artifact is built from, at
        // whichever rung this card built it: every rung of it is one model.
        let digest = if self.spec.prepared_from_source {
            engine_model_digest()
        } else {
            weights.digest().map_err(|e| {
                ConversationError::Other(format!(
                    "identify the weights at {}: {e}",
                    weights.pack.display()
                ))
            })?
        };
        ModelIdentity::new(
            format!("{:?}", self.spec.arch),
            self.spec.model_repo.clone(),
            self.spec.model_filename.clone(),
            digest,
            tokenizer_json,
            rope_key(&self.spec.rope),
        )
        .map_err(ConversationError::Other)
    }

    /// Resolve the model pack for `device`, and take what the model states —
    /// architecture, context, sampling, thinking — into this builder.
    pub(super) fn prepare(&mut self, device: &Device) -> crate::Result<Prepared> {
        let resolved = self.resolve_model(device)?;
        let model_path = resolved.pack.clone();
        // From here on this builder names the pack itself, so whatever reads the
        // model file afterwards (`engine_config`'s vocabulary width) reads it.
        self.model_path = Some(model_path.clone());

        // ── Read GGUF metadata ──────────────────────────────────────────
        // Always read the GGUF header for arch, vocab_size, context_length,
        // and thinking detection.  Only override sampling when the user
        // hasn't explicitly set it via --sampler or individual args.
        let mut gguf_vocab_size = None;
        let mut gguf_has_thinking = None;
        match Self::detect_sampling_from_gguf(&model_path) {
            Ok(info) => {
                gguf_vocab_size = info.vocab_size;
                gguf_has_thinking = Some(info.has_thinking);

                // Override model display name from GGUF `general.name`.
                self.spec.model_filename = format!("{}.gguf", info.name);

                // Override ModelArch from GGUF — ensures the correct weight
                // loader is used even when --model-dir points to a different
                // architecture than the --model preset.
                if let Some(arch) = info.arch {
                    if arch != self.spec.arch {
                        tracing::info!(
                            "Overriding ModelArch: preset={:?} → GGUF={:?}",
                            self.spec.arch,
                            arch
                        );
                        self.spec.arch = arch;
                        // The preset's RoPE schedule belonged to the checkpoint
                        // it named; this file is another, so it runs what it
                        // states (or its lineage's own).
                        self.spec.rope = arch.file_rope();
                    }
                }

                // Override max_seq_len from GGUF context_length.
                if let Some(ctx) = info.context_length {
                    if ctx != self.max_seq_len {
                        tracing::info!(
                            "Overriding max_seq_len: preset={} → GGUF={}",
                            self.max_seq_len,
                            ctx
                        );
                        self.max_seq_len = ctx;
                    }
                }

                if !self.sampling_user_set {
                    tracing::info!(
                        "Auto-detected sampling from GGUF: temp={}, top_k={}, top_p={}, repeat_penalty={}",
                        info.sampling.temperature, info.sampling.top_k, info.sampling.top_p, info.sampling.repeat_penalty
                    );
                    self.sampling = info.sampling;
                    if let Some(ref nt) = info.non_thinking {
                        tracing::info!(
                            "Auto-detected non-thinking sampling: temp={}, top_k={}, top_p={}",
                            nt.temperature,
                            nt.top_k,
                            nt.top_p
                        );
                    }
                    self.spec.non_thinking_sampling = info.non_thinking;
                }
            }
            Err(e) => {
                tracing::warn!("Could not read GGUF metadata: {e}, using preset defaults");
            }
        }

        let tokenizer = resolved.tokenizer()?;

        // Before anything reads a token id: the checkpoint is the authority on
        // what its own ids mean, so hold the tokenizer against it here rather
        // than trusting the path it was resolved from.
        Self::verify_tokenizer_matches_checkpoint(&tokenizer, &model_path)?;

        // ── Auto-detect thinking support from chat_template ───────────
        // The GGUF chat_template is the authoritative signal.  Many models
        // (e.g. non-thinking Qwen3 finetunes) have the <think> token in
        // the tokenizer but never use it — those are NOT thinking models.
        // Fall back to tokenizer check only when GGUF detection was skipped.
        let detected_thinking =
            gguf_has_thinking.unwrap_or_else(|| tokenizer.token_to_id("<think>").is_some());
        if detected_thinking != self.spec.supports_thinking {
            tracing::info!(
                "Thinking support overridden: preset={}, detected={}",
                self.spec.supports_thinking,
                detected_thinking
            );
            self.spec.supports_thinking = detected_thinking;
        }
        if self.spec.supports_thinking {
            let mode = match self.thinking {
                Some(true) => "enabled (--thinking)",
                Some(false) => "suppressed (--no-thinking)",
                None => "default (model decides)",
            };
            tracing::info!("Model supports thinking: {}", mode);
        }

        // ── Resolve <think>/<\u200b/think> token IDs from tokenizer ───────────
        // Patch the thinking token IDs into all sampling configs that have
        // EOT boost configured.  This replaces the placeholder -1 values.
        self.sampling.resolve_thinking_tokens(&tokenizer);
        if let Some(ref mut nt) = self.spec.non_thinking_sampling {
            nt.resolve_thinking_tokens(&tokenizer);
        }
        self.health_config.resolve_structural_tokens(&tokenizer);
        Ok(Prepared {
            pack: model_path,
            tokenizer,
            gguf_vocab_size,
        })
    }

    /// Check the loaded tokenizer against the checkpoint's **own** token table.
    ///
    /// `tokenizer.ggml.tokens` is the vocabulary the embedding and `lm_head`
    /// rows were built against, so it — not any repo, pin or filename — is what
    /// makes a `tokenizer.json` the right one. The GGUF ships the token list but
    /// no merges or pretokenizer, which is why a separate `tokenizer.json` is
    /// loaded at all; that copy has until now been trusted on the strength of a
    /// pinned repo constant and a comment asserting the two agree. Nothing
    /// re-established it at runtime, and zend does not pin revisions, so a
    /// wrong-but-loadable tokenizer was accepted in silence: ids stayed in
    /// range, decoded to real pieces, and the model was fed fluent-looking
    /// nonsense while every sealed turn recorded it.
    ///
    /// The comparison is against ids the model actually reserves meaning for —
    /// the whole special-token band plus a stride across the ordinary range —
    /// rather than the entire vocabulary, which would cost a full string
    /// compare of 248k entries on every boot to catch what any of these catch.
    pub(super) fn verify_tokenizer_matches_checkpoint(
        tokenizer: &tokenizers::Tokenizer,
        model_path: &Path,
    ) -> crate::Result<()> {
        let ct = gguf_header(model_path).map_err(ConversationError::Model)?;
        let Some(tokens) = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
        else {
            // Not every conversion carries the table. Say so rather than
            // implying the pair was checked.
            tracing::warn!(
                "checkpoint carries no `tokenizer.ggml.tokens`; the tokenizer cannot be \
                 verified against it and is trusted on its resolved path alone"
            );
            return Ok(());
        };

        // The GGUF's table is padded out to the output projection's width
        // (Qwen3.6-35B: 248,320 entries against the tokenizer's 248,070 real
        // tokens), so the two lengths are NOT expected to be equal — only
        // compatible. A tokenizer with MORE entries than the model has rows is
        // the unambiguous error; short of that, the agreement of the ids
        // themselves is what settles it.
        let n = tokens.len();
        let vocab = tokenizer.get_vocab(true);
        // **The id space, not the entry count.** `get_vocab().len()` counts
        // entries; added tokens sit at arbitrary high ids, so a tokenizer with
        // few entries can still address an id past the checkpoint's last row.
        // Bounding on the count would let exactly that through.
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let tok_vocab = max_id + 1;
        if tok_vocab > n {
            return Err(ConversationError::Tokenizer(format!(
                "tokenizer does not match the checkpoint: the tokenizer at {} defines \
                 {tok_vocab} tokens but the checkpoint's table has only {n} — ids past the end \
                 of the model's own vocabulary. This tokenizer was built for a different model.",
                model_path.display(),
            )));
        }

        // **The specials by id, not by position.** A mismatch on one of these
        // is the difference between ending a turn and emitting a word, and they
        // do not live at the bottom of the id space: Qwen puts `<|endoftext|>`
        // at 151643 on one model and 248044 on another, which is precisely the
        // pair this check exists to tell apart. A `0..1024` prefix — what this
        // used to probe while its comment claimed "specials first" — reaches
        // neither, and a 512-sample stride hits a given high id only by luck.
        //
        // `get_added_tokens_decoder` is the authoritative set: every special and
        // every added token, keyed by the id it actually occupies.
        let specials: Vec<usize> = tokenizer
            .get_added_tokens_decoder()
            .keys()
            .map(|&id| id as usize)
            .collect();
        let stride = (tok_vocab / 512).max(1);
        let probes = specials
            .into_iter()
            .chain(0..tok_vocab.min(1024))
            .chain((0..tok_vocab).step_by(stride))
            .filter(|&id| id < n);
        for id in probes {
            let Ok(want) = tokens[id].to_string() else {
                continue;
            };
            let got = tokenizer.id_to_token(id as u32);
            if got.as_deref() != Some(want.as_str()) {
                return Err(ConversationError::Tokenizer(format!(
                    "tokenizer does not match the checkpoint: at id {id} the GGUF's token table \
                     has {want:?} but the tokenizer at {} has {got:?}. The two disagree about \
                     what an id means, which no length check can see.",
                    model_path.display(),
                )));
            }
        }

        tracing::info!(
            vocab = n,
            "tokenizer verified against the checkpoint's own token table"
        );
        Ok(())
    }

    // ── Resolution ─────────────────────────────────────────────────────

    /// The model pack this builder loads, with its tokenizer — fetching and
    /// building through `fetch` when the model cache does not hold it.
    ///
    /// - A [`Self::model_path`] that is a pack is taken as it is.
    /// - One that is a checkpoint is packed with its [`Self::tokenizer_path`]
    ///   into the model cache, never beside it, and is never deleted.
    /// - Otherwise the spec's pack: every coordinate pinned, the revisions
    ///   included — a pin nothing consults reads as protection while the
    ///   weights move under it.
    pub fn resolve_model_with(
        &self,
        device: &Device,
        fetch: &dyn SourceFetch,
    ) -> crate::Result<ResolvedModel> {
        match (&self.model_path, &self.tokenizer_path) {
            (Some(p), _) if is_pack(p) => ResolvedModel::open(p.clone()),
            (Some(m), Some(t)) => resolve_local(&self.spec, m, t, device, fetch),
            (Some(m), None) => Err(ConversationError::Download(format!(
                "{} is a checkpoint, not a model pack, and packing it needs its tokenizer.json \
                 — set tokenizer_path too",
                m.display()
            ))),
            (None, Some(_)) => Err(ConversationError::Download(
                "a tokenizer_path is the tokenizer of a local checkpoint — set model_path too, \
                 or neither to load the spec's model"
                    .into(),
            )),
            (None, None) => resolve_spec(&self.spec, device, fetch),
        }
    }

    /// [`Self::resolve_model_with`] over the HuggingFace hub, cache first.
    #[cfg(feature = "hub")]
    pub fn resolve_model(&self, device: &Device) -> crate::Result<ResolvedModel> {
        self.resolve_model_with(device, &HubFetch)
    }

    /// [`Self::resolve_model_with`] with nowhere to fetch from: everything the
    /// pack needs must be on disk already.
    #[cfg(not(feature = "hub"))]
    pub fn resolve_model(&self, device: &Device) -> crate::Result<ResolvedModel> {
        self.resolve_model_with(device, &NoFetch)
    }

    /// Every adapter this build should load, as `(name, directory)`.
    ///
    /// Two sources, and they mean different things. The spec's
    /// [`loras`](ModelSpec::loras) are the ones the model *comes with* — the
    /// preset's, or whatever `models.override.yaml` replaced them with — and are
    /// fetched from HuggingFace here. [`Self::lora`]'s are the ones this
    /// particular build was told about, already local. Both end up in the same
    /// registry, addressed by name.
    ///
    /// A name collision is refused rather than resolved. The registry is a map,
    /// so a duplicate would silently keep whichever was inserted last, and a
    /// conversation asking for that name would get an adapter nobody chose.
    pub(super) fn resolve_loras(&self) -> crate::Result<Vec<(String, PathBuf)>> {
        let mut out: Vec<(String, PathBuf)> = Vec::new();

        for l in &self.spec.loras {
            let dir = self.resolve_lora_repo(&l.repo, &l.revision)?;
            out.push((l.name.clone(), dir));
        }
        for (name, dir) in &self.loras {
            out.push((name.clone(), dir.clone()));
        }

        let mut seen = std::collections::HashSet::new();
        for (name, _) in &out {
            if !seen.insert(name.clone()) {
                return Err(ConversationError::Model(candle::Error::Msg(format!(
                    "two LoRA adapters are both called {name:?} — the name is how a \
                     conversation selects one, so a duplicate silently serves whichever \
                     loaded last"
                ))));
            }
        }
        Ok(out)
    }

    /// Fetch a PEFT adapter's two files and return the directory holding them.
    ///
    /// PEFT writes `adapter_config.json` and `adapter_model.safetensors` side by
    /// side and the loader reads the pair, so this fetches both and hands back
    /// their common parent — which is the one way a LoRA's coordinates differ
    /// from a GGUF's, where a single filename is the whole answer.
    #[cfg(feature = "hub")]
    pub(super) fn resolve_lora_repo(&self, repo: &str, revision: &str) -> crate::Result<PathBuf> {
        use candle_transformers::models::hub_download::repo_file;
        use hf_hub::{Repo, RepoType};

        let r = Repo::with_revision(repo.to_owned(), RepoType::Model, revision.to_owned());
        let mut dirs = Vec::new();
        for f in ["adapter_config.json", "adapter_model.safetensors"] {
            let path = repo_file(&r, f).map_err(|e| ConversationError::Download(e.to_string()))?;
            let dir = path
                .parent()
                .ok_or_else(|| {
                    ConversationError::Download(format!("{path:?} has no parent directory"))
                })?
                .to_path_buf();
            dirs.push(dir);
        }
        // Both files share a revision directory in the cache. Asserted rather
        // than assumed: a loader pointed at a directory holding only one of the
        // two fails in a way that reads as a corrupt adapter.
        if dirs[0] != dirs[1] {
            return Err(ConversationError::Download(format!(
                "LoRA {repo}@{revision}: its two files landed in different directories, \
                 {:?} and {:?}",
                dirs[0], dirs[1]
            )));
        }
        Ok(dirs.remove(0))
    }

    #[cfg(not(feature = "hub"))]
    pub(super) fn resolve_lora_repo(&self, repo: &str, _revision: &str) -> crate::Result<PathBuf> {
        Err(ConversationError::Download(format!(
            "this model declares the LoRA adapter {repo:?}, which has to be downloaded — \
             enable the 'hub' feature, or use ModelBuilder::lora with a local directory"
        )))
    }
}
