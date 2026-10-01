# TODO

- [ ] **GUI shows an error on fast-path tool calls.** A served call renders as
  `[error: repeated_call]` while other tool calls in the same conversation show
  `in_context`. Find where `repeated_call` is raised for a served call and make
  the GUI show the `in_context` reply.
- [ ] **Remove the thinking blocks from `repo_map` and `code_reading` ingestion.**
- [ ] **Implement the journal tool.** It matters for anchoring the model as a
  session runs.
- [ ] **Tune decode performance.** Inference decode runs at about half the speed
  it should.
- [ ] **Turns don't finish while the substrate is compacting.** A turn finishes
  decoding but does not complete until compaction ends — probably its persist
  blocking on a lock compaction holds. Find the lock and stop compaction from
  holding it across the whole pass.
- [ ] **Test from machines other than the one running zend.** Ask the user
  about the setup before writing any code for this.
