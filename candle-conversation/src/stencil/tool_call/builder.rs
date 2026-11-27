//! Builds the argument grammar for one tool's parameters, writing the structure
//! the model should not have to choose.
//!
//! Each value is written by the grammar where it can be: a closed set, a bounded
//! integer, or an object or array whose structure the grammar can name. Where
//! the grammar cannot begin a value, the value is left free. Required fields come
//! in schema order and optional ones are offered as gates, so a call cannot close
//! before its required fields. A value's successor is folded into its arm, so the
//! separator after a value and the value's close are one choice in the model's
//! own tokens. Trees are shared per parameter list, so an alias costs a name arm
//! rather than a copy of its tool's argument grammar.

use super::*;

pub(super) struct ToolTreeBuilder<'a> {
    pub(super) spec: TreeSpec,
    pub(super) env: &'a ToolCallEnvelope,
    /// How many guided arrays enclose the value being built. An array inside
    /// one decodes free: unrolled, it would repeat per enclosing element and
    /// the tree would grow as the product of the bounds.
    pub(super) array_depth: u32,
    /// `(param name, its pre-compiled body sub-stencil)` for the one parameter
    /// whose value is spliced in from a resource's schema rather than free-typed
    /// (effector design §11). Set per-tool by [`compile_action_loop_with_body`],
    /// `None` everywhere else. JSON styles only.
    pub(super) body_override: Option<(&'a str, &'a TreeSpec)>,
}

/// One object's gate and value memo, so the gate graph stays linear instead of
/// exploding over subsets — and never leaks between objects, which have
/// different optional lists.
#[derive(Default)]
pub(super) struct FieldMemo {
    /// `(optional index, emitted_any, owed) -> gate entry`.
    pub(super) gates: HashMap<(usize, bool, Owed), SpecId>,
    /// `(optional index, owed before it) -> (lead-in, value entry)`. Every gate
    /// that offers an optional from the same standing leads its value to the
    /// same successor gate, so the value is the same sub-tree whichever gate it
    /// was reached from.
    pub(super) values: HashMap<(usize, Owed), (String, SpecId)>,
}

/// The values of `p` when they decide the rest of its object — an enum that
/// requires or brings fields per value ([`Param::requires`], [`Param::shapes`])
/// and cannot be `null`.
fn decider(p: &Param) -> Option<&[String]> {
    let decides = !p.requires.is_empty() || !p.shapes.is_empty();
    match (decides, &p.enum_values, p.nullable) {
        (true, Some(values), false) => Some(values),
        _ => None,
    }
}

/// Whether the grammar can begin a value of this schema — offer every way it
/// can start as an arm. Only such elements can be guided inside an array: the
/// arms carry the separator and the opening together (`, {`, `, true`), and an
/// element the model opens itself (a number) has nothing for an arm to hold.
pub(super) fn guided_element(p: &Param) -> bool {
    p.enum_values.is_some()
        || matches!(p.ty, ParamType::String | ParamType::Boolean)
        || (p.ty == ParamType::Object && p.properties.is_some())
}

impl<'a> ToolTreeBuilder<'a> {
    pub(super) fn new(env: &'a ToolCallEnvelope) -> Self {
        ToolTreeBuilder {
            spec: TreeSpec::new(TOOL_CALL_TREE_LABEL),
            env,
            array_depth: 0,
            body_override: None,
        }
    }

    /// What a tool's name arm leads to: the arguments' opening, then its
    /// argument object through the envelope close, ending at `end`.
    ///
    /// **One node per distinct parameter list.** A tool and each alias it
    /// answers to take the same arguments, so what follows their names is one
    /// grammar, and the arms lead to one node: the compiler lowers a node once
    /// however many arms reach it, so an alias costs a name arm instead of a
    /// copy of its tool's whole argument grammar. The catalog's aliases outnumber
    /// its tools five to one, and a copy each made the tree — and the seconds it
    /// took to compile at every daemon start — five times what it needed to be.
    /// `shared` is the memo for one level, whose `end` is fixed; a tool with a
    /// typed body spliced in is its own.
    pub(super) fn tool_arm_target(
        &mut self,
        tool: &ToolSpec,
        end: SpecId,
        shared: &mut HashMap<String, SpecId>,
    ) -> Result<SpecId, BuildError> {
        // A required argument with one value is written into its key, no
        // branch choosing it, so the call records it by its tool instead.
        let fixed: Vec<(Arc<str>, Arc<str>)> = tool
            .params
            .iter()
            .filter(|p| p.required && !p.nullable)
            .filter_map(|p| match p.enum_values.as_deref() {
                Some([only]) => Some((Arc::from(p.name.as_str()), Arc::from(only.as_str()))),
                _ => None,
            })
            .collect();
        if !fixed.is_empty() {
            self.spec.fixed.insert(Arc::from(tool.name.as_str()), fixed);
        }
        let key = format!("{:?}", tool.params);
        if self.body_override.is_none() {
            if let Some(&target) = shared.get(&key) {
                return Ok(target);
            }
        }
        let close = self.env.close.clone();
        let args_entry = self.build_fields(&tool.params, &close, end)?;
        let target = self.spec.push(NodeSpec::Static {
            text: self.env.args_open.clone(),
            next: args_entry,
        });
        if self.body_override.is_none() {
            shared.insert(key, target);
        }
        Ok(target)
    }

    /// An object's field sequence, written up to and including `close` — the
    /// envelope close for a call's arguments, `}` for an object value — and
    /// ending at `end`.  Returns the entry node.
    pub(super) fn build_fields(
        &mut self,
        params: &[Param],
        close: &str,
        end: SpecId,
    ) -> Result<SpecId, BuildError> {
        let required: Vec<&Param> = params.iter().filter(|p| p.required).collect();
        let optional: Vec<&Param> = params.iter().filter(|p| !p.required).collect();
        let owed = Owed::initial(&optional);
        self.build_sequence(&required, &optional, true, owed, close, end)
    }

    /// `required` in order, then the gates over `optional` from standing
    /// `owed`, through `close` to `end`. `first` says no field precedes them.
    pub(super) fn build_sequence(
        &mut self,
        required: &[&Param],
        optional: &[&Param],
        first: bool,
        owed: Owed,
        close: &str,
        end: SpecId,
    ) -> Result<SpecId, BuildError> {
        let Some((&p, rest)) = required.split_first() else {
            // Optional gates start with emitted_any = (a required field precedes them).
            let mut memo = FieldMemo::default();
            return self.opt_gates(optional, 0, !first, owed, close, end, &mut memo);
        };
        if let Some(values) = decider(p) {
            return self.build_discriminated(p, values, rest, optional, first, close, end);
        }
        let tail = self.build_sequence(rest, optional, false, owed, close, end)?;
        let (leadin, value) = self.build_value(p, tail)?;
        Ok(self.spec.push(NodeSpec::Static {
            text: self.key(p, first, &leadin),
            next: value,
        }))
    }

    /// A required enum field `p` whose values decide the rest of the object
    /// ([`Param::requires`], [`Param::shapes`]): each value goes on to the rest
    /// of the object with the fields it requires moved from `optional` to
    /// required, and the fields it brings added, after `rest`.
    ///
    /// **One tail per distinct continuation.** A value's tail is fixed by the
    /// fields it promotes and the fields it brings, so values that agree on
    /// both point at the same sub-tree: a thousand endpoints that accept the
    /// same body cost that body once.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_discriminated(
        &mut self,
        p: &Param,
        values: &[String],
        rest: &[&Param],
        optional: &[&Param],
        first: bool,
        close: &str,
        end: SpecId,
    ) -> Result<SpecId, BuildError> {
        let arms = self.value_tails(p, values, rest, optional, None, close, end)?;
        let branch = self.decided_branch(p, arms);
        Ok(self.spec.push(NodeSpec::Static {
            text: self.key(p, first, ""),
            next: branch,
        }))
    }

    /// Each of `values` with the tail it goes on to: `rest`, then the fields
    /// the value requires moved from `optional` to required and the fields it
    /// brings, then the gates over what is left of `optional` — from `owed`
    /// when the field is itself one of the optionals already standing, or
    /// afresh over what is left.
    #[allow(clippy::too_many_arguments)]
    fn value_tails(
        &mut self,
        p: &Param,
        values: &[String],
        rest: &[&Param],
        optional: &[&Param],
        owed: Option<Owed>,
        close: &str,
        end: SpecId,
    ) -> Result<Vec<(String, SpecId)>, BuildError> {
        let mut tails: HashMap<String, SpecId> = HashMap::new();
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(values.len());
        for value in values {
            // Every rule naming the value counts.
            let needs: Vec<&String> = p
                .requires
                .iter()
                .filter(|(v, _)| v == value)
                .flat_map(|(_, fields)| fields)
                .collect();
            let brought: Vec<Param> = p
                .shapes
                .iter()
                .filter(|(v, _)| v == value)
                .flat_map(|(_, fields)| fields)
                .map(|f| Param {
                    required: true,
                    ..f.clone()
                })
                .collect();
            let brings = |o: &&Param| brought.iter().any(|b| b.name == o.name);
            let wanted = |o: &&Param| needs.contains(&&o.name);
            let promoted: Vec<Param> = optional
                .iter()
                .copied()
                .filter(|o| wanted(o) && !brings(o))
                .map(|o| Param {
                    required: true,
                    ..o.clone()
                })
                .collect();
            let key = format!("{promoted:?}{brought:?}");
            let tail = match tails.get(&key) {
                Some(&tail) => tail,
                None => {
                    let mut now_required: Vec<&Param> =
                        rest.iter().copied().filter(|r| !brings(r)).collect();
                    now_required.extend(promoted.iter());
                    now_required.extend(brought.iter());
                    let still_optional: Vec<&Param> = optional
                        .iter()
                        .copied()
                        .filter(|o| !wanted(o) && !brings(o))
                        .collect();
                    let standing = owed.unwrap_or_else(|| Owed::initial(&still_optional));
                    let tail = self.build_sequence(
                        &now_required,
                        &still_optional,
                        false,
                        standing,
                        close,
                        end,
                    )?;
                    tails.insert(key, tail);
                    tail
                }
            };
            arms.push((value.clone(), tail));
        }
        Ok(arms)
    }

    /// The branch over a deciding field's values, each arm going on to its own
    /// tail ([`Self::value_tails`]), recorded as `p`'s choice.
    fn decided_branch(&mut self, p: &Param, arms: Vec<(String, SpecId)>) -> SpecId {
        let (branch, values) = match self.env.style {
            // The end marker rides on each arm, as in [`Self::build_value`].
            CallStyle::FunctionBlock => {
                let marker = self.env.param_close;
                let values = arms
                    .iter()
                    .map(|(v, _)| Some(Arc::from(v.as_str())))
                    .collect();
                let arms = arms
                    .into_iter()
                    .map(|(v, next)| (format!("{v}{marker}"), next))
                    .collect();
                (self.spec.push(NodeSpec::Branch { arms }), values)
            }
            // Quoted, each carrying the first step of its own tail — so the
            // value and the key it requires next are one choice — and its
            // space, the key ending at the colon.
            CallStyle::JsonBlock | CallStyle::Lines => {
                let quoted = arms
                    .into_iter()
                    .map(|(v, next)| {
                        let value = Some(Arc::from(v.as_str()));
                        (Value::String(v).to_string(), next, value)
                    })
                    .collect();
                let (arms, values) = self.with_continuations(quoted);
                let arms = arms
                    .into_iter()
                    .map(|(t, next)| (format!(" {t}"), next))
                    .collect();
                (self.spec.push(NodeSpec::Branch { arms }), values)
            }
        };
        self.chosen(branch, p, values);
        branch
    }

    /// What opens one argument, up to the point its value begins.
    ///
    /// `first` says whether a separator is needed before it — which only the
    /// JSON shapes have. A function block's arguments are self-delimiting
    /// elements, so there is nothing between them and the flag is ignored.
    pub(super) fn key(&self, p: &Param, first: bool, leadin: &str) -> String {
        match self.env.style {
            CallStyle::FunctionBlock => format!(
                "{}{}{}",
                self.env.param_open, p.name, self.env.param_name_close
            ),
            // Through the lead-in when there is one, else up to the colon: the
            // space before a value is the first byte of the model's own token
            // (` [`, ` 5`, ` true`), and a key ending on it leaves the model
            // mid-token. See the module docs.
            CallStyle::JsonBlock | CallStyle::Lines => {
                let sep = if first { "" } else { ", " };
                match leadin {
                    "" => format!("{sep}\"{}\":", p.name),
                    _ => format!("{sep}\"{}\": {leadin}", p.name),
                }
            }
        }
    }

    /// The gate over optionals `idx..`, given whether a field was already
    /// emitted and where the object stands on its choice: the optionals
    /// [`Owed::offers`] and, when it [`Owed::may_close`], the close.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn opt_gates(
        &mut self,
        opts: &[&Param],
        idx: usize,
        emitted_any: bool,
        owed: Owed,
        close: &str,
        end: SpecId,
        memo: &mut FieldMemo,
    ) -> Result<SpecId, BuildError> {
        if let Some(&id) = memo.gates.get(&(idx, emitted_any, owed)) {
            return Ok(id);
        }
        // No more optionals: emit the close and finish.
        if idx == opts.len() {
            let id = self.spec.push(NodeSpec::Static {
                text: close.to_string(),
                next: end,
            });
            memo.gates.insert((idx, emitted_any, owed), id);
            return Ok(id);
        }
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(opts.len() - idx + 1);
        for (j, p) in opts.iter().enumerate().skip(idx) {
            if !owed.offers(opts, j) {
                continue;
            }
            let (leadin, value) = match memo.values.get(&(j, owed)) {
                Some(built) => built.clone(),
                None => {
                    // Include optional j: a field is emitted, so everything
                    // after has emitted_any = true.
                    let settled = owed.after(p.pick);
                    let built = match decider(p) {
                        // **An optional field whose values decide the rest
                        // brings their fields too.** Only a required one did,
                        // so `order_post`'s optional `job` brought nothing:
                        // Zen chose `coding-task` and wrote a `brief`, never
                        // `what` or `repository`, and the post was refused.
                        Some(values) => {
                            let later: Vec<&Param> = opts[j + 1..].to_vec();
                            let arms = self.value_tails(
                                p,
                                values,
                                &[],
                                &later,
                                Some(settled),
                                close,
                                end,
                            )?;
                            (String::new(), self.decided_branch(p, arms))
                        }
                        None => {
                            let after =
                                self.opt_gates(opts, j + 1, true, settled, close, end, memo)?;
                            self.build_value(p, after)?
                        }
                    };
                    memo.values.insert((j, owed), built.clone());
                    built
                }
            };
            arms.push((self.key(p, !emitted_any, &leadin), value));
        }
        // The "stop" arm: close the object, once it may.
        if owed.may_close() {
            arms.push((close.to_string(), end));
        }
        let id = match <[(String, SpecId); 1]>::try_from(arms) {
            // One way on is no choice.
            Ok([(text, next)]) => self.spec.push(NodeSpec::Static { text, next }),
            Err(arms) => self.spec.push(NodeSpec::Branch { arms }),
        };
        memo.gates.insert((idx, emitted_any, owed), id);
        Ok(id)
    }

    /// Record that `span` writes `p`'s value: what a turn's held values are
    /// looked up by (`docs/stencil_tree.md` §9.2).
    pub(super) fn keyed(&mut self, span: SpecId, p: &Param) -> SpecId {
        self.spec.keys.insert(span, Arc::from(p.name.as_str()));
        span
    }

    /// A value sub-tree for `p`, transitioning to `next` after the value.
    /// Returns a `lead-in` string that must be appended to the preceding key/arm
    /// static, plus the value's entry node.  Folding the lead-in (a string/enum
    /// value's opening `"`) into the key keeps structural merges like ` "`
    /// internal to one static, rather than leaving a lone `"` after a branch arm
    /// that merges backward into the committed arm (an unrepresentable retract).
    pub(super) fn build_value(
        &mut self,
        p: &Param,
        next: SpecId,
    ) -> Result<(String, SpecId), BuildError> {
        // **A resource's typed invoke body, spliced in** from its own pre-compiled
        // sub-stencil (effector design §11). The key ends at its colon and the
        // body opens with `{`, so the value has no lead-in. JSON styles only — a
        // function block's raw value keeps its free span below.
        if let Some((pname, spec)) = self.body_override {
            if pname == p.name && self.env.style != CallStyle::FunctionBlock {
                let entry = splice(&mut self.spec, spec, next);
                return Ok((String::new(), entry));
            }
        }
        // A nested JSON object refined into typed fields (`object | recurse`).
        // Function blocks keep the raw free value below. The opening brace
        // rides on the lead-in, same as `value_arms`'s object arm. A
        // *nullable* object still needs the null-vs-object choice `value_arms`
        // builds below — forcing the object here would deny the model `null`.
        if self.env.style != CallStyle::FunctionBlock && !p.nullable {
            if let Some(fields) = &p.properties {
                let entry = self.build_fields(fields, "}", next)?;
                return Ok(("{".to_string(), entry));
            }
        }
        // **A function block's values are raw**, so the closing delimiter is the
        // element's own end marker rather than a quote, and nothing is escaped.
        // Handled before the JSON cases because both the enum branch and the
        // free span differ, not just one of them.
        if self.env.style == CallStyle::FunctionBlock {
            let close = self.env.param_close;
            if let Some(values) = &p.enum_values {
                // The end marker rides on each arm for the reason the quote
                // does below: a value that prefixes another stays
                // distinguishable in the trie.
                let branch = self.spec.push(NodeSpec::Branch {
                    arms: values
                        .iter()
                        .map(|v| (format!("{v}{close}"), next))
                        .collect(),
                });
                let chosen = values.iter().map(|v| Some(Arc::from(v.as_str()))).collect();
                self.chosen(branch, p, chosen);
                return Ok((String::new(), branch));
            }
            // Every type is raw text here — there is no JSON to be structurally
            // valid against, and the act that receives the call parses its own
            // scalars exactly as it does on the other styles.
            return Ok((String::new(), self.element_value(p, next)));
        }
        // **A plain string's opening quote is the model's.** The key stops at
        // the colon, because which token opens a string depends on the value:
        // Qwen writes an empty string as the single token ` ""` and a non-empty
        // one as ` "` then content, so a prefilled ` "` has chosen "non-empty"
        // before the model has chosen anything. See `Terminator::JsonStringValue`.
        //
        // A nullable string cannot be a choice between ` "` and ` null` for the
        // same reason — the ` "` arm is that prefill — so it is a free JSON
        // value instead, where ` ""`, ` "src"` and ` null` are each the model's
        // own tokens and a skipped value is written `null`. That is the common
        // case, not an edge: an `Option<String>` argument such as `file_list`'s
        // `prefix` is nullable in its schema. An array element's quote stays
        // the separator arm's (`, "`) — see [`Self::value_arms`].
        if p.ty == ParamType::String && p.enum_values.is_none() {
            if p.nullable {
                return Ok((String::new(), self.free_value(next)));
            }
            if let Some(pattern) = &p.pattern {
                let span = self.spec.push(NodeSpec::Value {
                    pattern: pattern.clone(),
                    opened: false,
                    limits: self.env.string_limits(p),
                    next,
                });
                return Ok((String::new(), self.keyed(span, p)));
            }
            let span = self.spec.push(NodeSpec::FreeText {
                term: Terminator::JsonStringValue,
                eos_ends: false,
                limits: self.env.string_limits(p),
                close_token: None,
                suppress_close: false,
                next,
            });
            return Ok((String::new(), self.keyed(span, p)));
        }
        // An enum field's arms carry the first step of what follows the field
        // — the next key, or the close — so the value's closing quote and the
        // separator after it are one arm's text. See [`Self::continuations`].
        // Each enum arm writes its value, in the order `value_arms` gives them
        // — the values, then a nullable field's `null`, which writes none.
        let (arms, values) = match (self.value_arms(p, next)?, &p.enum_values) {
            (Some(arms), Some(enum_values)) => {
                let valued = arms
                    .into_iter()
                    .enumerate()
                    .map(|(i, (text, n))| {
                        (text, n, enum_values.get(i).map(|v| Arc::from(v.as_str())))
                    })
                    .collect();
                let (arms, values) = self.with_continuations(valued);
                (Some(arms), Some(values))
            }
            (arms, _) => (arms, None),
        };
        match arms {
            // One way to begin — an object's `{`, an array's `[`, a one-value
            // enum: it is the lead-in, folded into the key (`"filter": {`). A
            // one-value enum's value is its tool's to record (`TreeSpec::fixed`).
            Some(mut arms) if arms.len() == 1 => Ok(arms.remove(0)),
            // A choice — `true`/`false`, an enum, a value or `null`. The key
            // ends at the colon, so each arm carries its space.
            Some(arms) => {
                let branch = self.spec.push(NodeSpec::Branch {
                    arms: arms
                        .into_iter()
                        .map(|(t, n)| (format!(" {t}"), n))
                        .collect(),
                });
                if let Some(values) = values {
                    self.chosen(branch, p, values);
                }
                Ok((String::new(), branch))
            }
            None => Ok((String::new(), self.free_value(next))),
        }
    }

    /// Every way a value of `p` can begin, as arm text with no leading space
    /// and where each leads — or `None` when the grammar cannot begin it and
    /// the model writes the value whole ([`Self::free_value`]).
    ///
    /// **A closed set is a choice, never free text.** An enum, `true`/`false`
    /// and `null` are each a complete value the grammar can write, so the model
    /// chooses among them under the mask: it cannot spell `True`, and a
    /// nullable field can actually be `null`. A string, a guided object and a
    /// guided array begin with their opening byte, and `null` beside it when
    /// the schema allows it.
    pub(super) fn value_arms(
        &mut self,
        p: &Param,
        next: SpecId,
    ) -> Result<Option<Vec<(String, SpecId)>>, BuildError> {
        let mut arms: Vec<(String, SpecId)> = match (&p.enum_values, p.ty) {
            // The closing quote rides on each arm, so a value that prefixes
            // another stays distinguishable in the trie.
            (Some(values), _) => values
                .iter()
                .map(|v| (Value::String(v.clone()).to_string(), next))
                .collect(),
            (None, ParamType::Boolean) => vec![("true".into(), next), ("false".into(), next)],
            (None, ParamType::String) => {
                let limits = self.env.string_limits(p);
                let span = self.spec.push(match &p.pattern {
                    Some(pattern) => NodeSpec::Value {
                        pattern: pattern.clone(),
                        opened: true,
                        limits,
                        next,
                    },
                    None => NodeSpec::FreeText {
                        term: Terminator::JsonString,
                        eos_ends: false,
                        limits,
                        close_token: None,
                        suppress_close: false,
                        next,
                    },
                });
                vec![("\"".into(), self.keyed(span, p))]
            }
            // **An object with a schema is written by the grammar**: its brace,
            // its keys, its separators — the model decodes only the values.
            (None, ParamType::Object) => match &p.properties {
                Some(fields) => vec![("{".into(), self.build_fields(fields, "}", next)?)],
                None => return Ok(None),
            },
            // **So is an array whose elements the grammar can begin** — see
            // [`Self::build_array`]. Anything else stays a free value.
            (None, ParamType::Array) => match p.items.as_deref() {
                Some(item) if self.array_depth == 0 && guided_element(item) => {
                    self.array_depth += 1;
                    let built = self.build_array(item, p.min_items, p.max_items, next);
                    self.array_depth -= 1;
                    vec![("[".into(), built?)]
                }
                _ => return Ok(None),
            },
            (None, ParamType::Integer) => match self.integer_arms(p, next) {
                Some(arms) => arms,
                None => return Ok(None),
            },
            (None, ParamType::Number) => return Ok(None),
        };
        if p.nullable {
            arms.push(("null".into(), next));
        }
        Ok(Some(arms))
    }

    /// **An integer is written digit by digit under the mask**, so nothing but
    /// an integer can be decoded there.
    ///
    /// ```text
    ///   ─┬─ "0" ──────────────── <end>
    ///    ├─ "1".."9" ─ D₁₈ ─┬─ "0".."9" ─ D₁₇ ─ … ─ D₀ ─ <end>
    ///    └─ "-" ─ "1".."9" ─┘  └─ <end>
    /// ```
    ///
    /// `<end>` is what follows the value — the next key, the close — taken as
    /// each arm's text ([`Self::continuations`]), so the digit the model does
    /// not write and the separator it does are one choice. `-` is offered only
    /// when the schema's `minimum` allows a negative, `0` only when it allows
    /// zero, and a leading `0` ends the number, as JSON requires.
    ///
    /// It was a free JSON value, which guarantees structure and not type: live,
    /// a `git_show` call decoded `"page": "7eb876659c1b…"` — a commit id in an
    /// integer field — and was refused a layer later. Under the mask the `"`
    /// that opens it is never a candidate.
    ///
    /// `None` when what follows the value cannot be written as arm text — its
    /// first step is free text, not a key or a close — and the value stays a
    /// free JSON value.
    pub(super) fn integer_arms(
        &mut self,
        p: &Param,
        next: SpecId,
    ) -> Option<Vec<(String, SpecId)>> {
        const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
        let ends = self.continuations(next);
        if ends.iter().any(|(text, _)| text.is_empty()) {
            return None;
        }
        let finished = self.choice(ends.clone());
        // Built back to front: after the last admitted digit only the end
        // remains, and each level before it offers one more digit or the end.
        // A schema's `maximum` bounds the digits, so a value too long for it
        // cannot be written; one of the right length past it is the tool's.
        let digits = p
            .max_tokens
            .map_or(MAX_INTEGER_DIGITS, |d| (d as usize).min(MAX_INTEGER_DIGITS));
        let mut more = finished;
        for _ in 1..digits {
            let mut arms: Vec<(String, SpecId)> =
                DIGITS.iter().map(|d| (d.to_string(), more)).collect();
            arms.extend(ends.iter().cloned());
            more = self.spec.push(NodeSpec::Branch { arms });
        }
        // A one-digit value's own maximum is its last digit.
        let top = match p.maximum {
            Some(m) if digits == 1 && (1.0..10.0).contains(&m) => m as usize,
            _ => 9,
        };
        let leading: Vec<(String, SpecId)> = DIGITS[1..=top]
            .iter()
            .map(|d| (d.to_string(), more))
            .collect();
        // `0` is a whole value of its own, offered only where the schema's
        // `minimum` admits it: a one-based page has no page 0 to write.
        let mut first = Vec::with_capacity(DIGITS.len() + 1);
        if p.minimum.is_none_or(|m| m <= 0.0) {
            first.push(("0".to_string(), finished));
        }
        first.extend(leading.iter().cloned());
        if p.minimum.is_none_or(|m| m < 0.0) {
            let negative = self.spec.push(NodeSpec::Branch { arms: leading });
            first.push(("-".to_string(), negative));
        }
        Some(first)
    }

    /// A node offering `arms`: a static when there is only one way on, which is
    /// no choice at all.
    pub(super) fn choice(&mut self, mut arms: Vec<(String, SpecId)>) -> SpecId {
        match arms.len() {
            1 => {
                let (text, next) = arms.remove(0);
                self.spec.push(NodeSpec::Static { text, next })
            }
            _ => self.spec.push(NodeSpec::Branch { arms }),
        }
    }

    /// The first step of `next`, as `(text, successor)` arms an enum value's
    /// arm can carry on its end: a static's text, each arm of a branch, or —
    /// for anything the grammar does not write itself — nothing.
    ///
    /// **Why an enum value takes its successor's first step with it.** Its
    /// closing quote is the grammar's, and the next thing written is a
    /// separator or a close the model chooses (`, "path":` against `}`). A
    /// tokenizer writes those two together — Qwen's `",` and `"}` are single
    /// tokens — so a quote committed on its own leaves the model to choose from
    /// a boundary its encoding never produces, holding a lone `,` it is never
    /// trained to write after `"`. Measured live: a `file_list` whose `repo` is
    /// an enum closed its arguments after the repository three times running,
    /// the model's reasoning each time saying it meant to pass `path`. With
    /// the separator on the arm, the value and what follows it are one choice
    /// the model writes in its own tokens.
    ///
    /// Object fields only. An array element's successor is the next element's
    /// own separator branch, so folding there compounds across every element
    /// of the array; a field's successor is a key or the close, one step deep.
    pub(super) fn continuations(&self, next: SpecId) -> Vec<(String, SpecId)> {
        match self.spec.node(next) {
            NodeSpec::Static { text, next } => vec![(text.clone(), *next)],
            NodeSpec::Branch { arms } => arms.clone(),
            NodeSpec::FreeText { .. }
            | NodeSpec::Value { .. }
            | NodeSpec::Element { .. }
            | NodeSpec::End => vec![(String::new(), next)],
        }
    }

    /// A function block's raw value for `p`, going on to `next`.
    ///
    /// **It closes where structure follows `</parameter>`**
    /// (`element_close`), so a value holding the tag keeps it. The close is
    /// the marker, a separator and the next element's tag, all the model's —
    /// so what follows is `next` with that tag already written, one fork per
    /// tag `next` can open with ([`Self::after_written`]): the tags `next`
    /// cannot open with are not offered, and a close the tree would refuse
    /// never fires. A value cut short still goes to `next`, which writes the
    /// whole continuation after the completed `</parameter>`.
    pub(super) fn element_value(&mut self, p: &Param, next: SpecId) -> SpecId {
        let marker = self.env.param_close;
        let element = self.after_written(next, &self.env.param_open.clone());
        let close = self.after_written(next, &format!("\n{FUNCTION_TAG}"));
        let follows = match (element, close) {
            (Some(_), Some(_)) => Some(Follows::Either),
            (Some(_), None) => Some(Follows::Element),
            (None, Some(_)) => Some(Follows::Close),
            (None, None) => None,
        };
        let limits = self.env.string_limits(p);
        let Some(follows) = follows.filter(|_| marker == PARAM_CLOSE) else {
            // A layout with no tag this rule knows: the marker alone ends it.
            let span = self.spec.push(NodeSpec::FreeText {
                term: Terminator::Until { marker, tails: &[] },
                eos_ends: false,
                limits,
                close_token: None,
                suppress_close: false,
                next,
            });
            return self.keyed(span, p);
        };
        let forks = (0..follows.tails().len())
            .map(|k| match follows.tag(k) {
                PARAM_TAG => element,
                _ => close,
            })
            .map(|fork| fork.expect("a tail is offered only for a tag `next` opens with"))
            .collect();
        let span = self.spec.push(NodeSpec::Element {
            marker,
            tails: follows.tails(),
            limits,
            forks,
            next,
        });
        self.keyed(span, p)
    }

    /// What follows `node` once `written` — the start of its text — has been
    /// written by the model rather than by the grammar. `None` when `node`
    /// cannot begin with `written`.
    ///
    /// A static loses `written` from its front; a branch keeps the arms that
    /// begin with it, each without it. A static shorter than `written` hands
    /// the rest on to its successor.
    pub(super) fn after_written(&mut self, node: SpecId, written: &str) -> Option<SpecId> {
        match self.spec.node(node).clone() {
            NodeSpec::Static { text, next } => {
                if let Some(rest) = text.strip_prefix(written) {
                    return Some(match rest.is_empty() {
                        true => next,
                        false => self.spec.push(NodeSpec::Static {
                            text: rest.to_string(),
                            next,
                        }),
                    });
                }
                let more = written.strip_prefix(text.as_str())?;
                self.after_written(next, more)
            }
            NodeSpec::Branch { arms } => {
                let kept: Vec<(String, SpecId)> = arms
                    .iter()
                    .filter_map(|(text, n)| Some((text.strip_prefix(written)?.to_string(), *n)))
                    .collect();
                if kept.is_empty() || kept.iter().any(|(text, _)| text.is_empty()) {
                    return None;
                }
                Some(match <[(String, SpecId); 1]>::try_from(kept) {
                    Ok([(text, next)]) => self.spec.push(NodeSpec::Static { text, next }),
                    Err(arms) => self.spec.push(NodeSpec::Branch { arms }),
                })
            }
            _ => None,
        }
    }

    /// `arms` with each one's successor's first step appended
    /// ([`Self::continuations`]), each carrying the value its arm writes — an
    /// arm becomes one per continuation, every one writing the same value.
    pub(super) fn with_continuations(
        &self,
        arms: Vec<ValuedArm>,
    ) -> (Vec<(String, SpecId)>, Vec<ArmValue>) {
        let mut out = Vec::with_capacity(arms.len());
        let mut values = Vec::with_capacity(arms.len());
        for (text, next, value) in arms {
            for (tail, after) in self.continuations(next) {
                out.push((format!("{text}{tail}"), after));
                values.push(value.clone());
            }
        }
        (out, values)
    }

    /// Mark `branch` as choosing `p`'s value, `values` the value of each of its
    /// arms in arm order.
    pub(super) fn chosen(&mut self, branch: SpecId, p: &Param, values: Vec<ArmValue>) {
        self.spec.chosen.insert(
            branch,
            ChosenArms {
                field: Arc::from(p.name.as_str()),
                values,
            },
        );
    }

    /// Any structurally-valid JSON value, lookahead-terminated at the enclosing
    /// `,`, `}` or `]` (the session pushes that delimiter back, or drops it when
    /// the successor does not continue with it).  This guarantees valid JSON
    /// structure; it does not strictly enforce the scalar type. The value's
    /// leading space is the model's — see [`Self::key`].
    pub(super) fn free_value(&mut self, next: SpecId) -> SpecId {
        self.spec.push(NodeSpec::FreeText {
            term: Terminator::JsonValue,
            eos_ends: false,
            limits: FreeTextLimits::json_value(),
            close_token: None,
            suppress_close: false,
            next,
        })
    }

    /// The elements of an array value, after its `[` (the caller's lead-in),
    /// through its `]`, ending at `next`.
    ///
    /// ```text
    ///   [ ─┬─ <open> element₁ ─┬─ ", <open>" element₂ ─ … ─ element₆₄ ─ "]"
    ///      └─ "]"              └─ "]"
    /// ```
    ///
    /// `<open>` is every way the element can begin ([`Self::value_arms`]): one
    /// arm for a string or object, one per value for a closed set — an array
    /// of booleans offers `true`, `false` and `]` at each element.
    ///
    /// **The grammar writes every bracket and separator**, so the model cannot
    /// close the array with the wrong bracket or leave an element open. Live,
    /// with the array a free value, a Cline `read_files` call came out
    /// `{"path": "…", "start_line": 3380, "end_line": 3420]}}}` — the element's
    /// `}` never written, a `]` in its place, not JSON, so no call. A free span
    /// counts one depth for `[` and `{` alike and could not see it.
    ///
    /// **The first `min_items` levels offer no `]`**, so the schema's
    /// `minItems` holds under the mask. Live, `git_push` — whose `pushes`
    /// needs at least one — was decoded `"pushes": []` three turns running
    /// while the model's reasoning named the branch it meant to push: offered
    /// the close beside the first element, it took the close.
    ///
    /// **After `max_items` elements only the close is offered**, so the
    /// schema's `maxItems` holds too. A list the model went round in — a
    /// table's faults, the same two passages again and again — ran to the
    /// unrolled bound, sixty-four entries and three thousand words, where its
    /// schema wanted eight.
    ///
    /// Unrolled to `max_items`, or [`MAX_ARRAY_ELEMENTS`] when the schema sets
    /// none or more, and built back to front, as [`compile_action_loop`] builds
    /// its levels: each level's continuation points at the next, so the deepest
    /// exists first.
    pub(super) fn build_array(
        &mut self,
        item: &Param,
        min_items: usize,
        max_items: Option<usize>,
        next: SpecId,
    ) -> Result<SpecId, BuildError> {
        let levels = max_items.map_or(MAX_ARRAY_ELEMENTS, |m| m.min(MAX_ARRAY_ELEMENTS));
        // After the last admitted element, only the close remains.
        let mut after = self.spec.push(NodeSpec::Static {
            text: "]".to_string(),
            next,
        });
        for level in (0..levels).rev() {
            let opens = self
                .value_arms(item, after)?
                .expect("`build_array` is reached only for a guided element");
            let sep = if level == 0 { "" } else { ", " };
            let mut arms: Vec<(String, SpecId)> = opens
                .into_iter()
                .map(|(open, element)| (format!("{sep}{open}"), element))
                .collect();
            if level >= min_items {
                arms.push(("]".to_string(), next));
            }
            after = self.choice(arms);
        }
        Ok(after)
    }
}
