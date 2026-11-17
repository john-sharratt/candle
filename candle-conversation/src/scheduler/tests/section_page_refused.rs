//! **A section whose index page the model refuses goes in as a gap, and is
//! prefilled fresh next time.**
//!
//! A projection borrows a sealed section's K/V and pushes its index page beside
//! it. When the model will not read the page — one written in a format it no
//! longer takes — the K/V is in the slot regardless, so the index must still be
//! advanced over it, or every later piece is placed short of its K/V and every
//! select past the section refuses. On WhiteWorld a 108-token section's pre-Q8
//! page did exactly that to every turn that borrowed it. These run the real
//! assembler on the CPU double.

use super::*;
use crate::projection::{ProjectionSegment, ResolvedSection, SealedKind};

/// A section of `tokens` sealed from a scratch slot, registered against
/// `stream` on a fresh conversation.
fn sealed_section(
    scheduler: &mut Scheduler,
    section: SectionId,
    stream: StreamId,
    tokens: usize,
) -> Conversation {
    let src = scheduler.session.create_sequence().expect("scratch slot");
    scheduler
        .session
        .ensure_capacity(&[src], tokens)
        .expect("capacity");
    for b in scheduler.session.backings() {
        b.set_len(src, tokens);
    }
    scheduler
        .session
        .advance_sequence(src, tokens)
        .expect("advance");
    let sealed = scheduler
        .session
        .snapshot_sequence_per_layer(src)
        .expect("snapshot");
    let conversation = Conversation::new();
    conversation.update(|substrate| {
        substrate.restore_section(
            section,
            stream,
            tokens,
            sealed.clone(),
            Vec::new(),
            Arc::new(vec![0; tokens]),
        )
    });
    conversation
}

fn project_section(scheduler: &mut Scheduler, slot: SequenceId, section: SectionId) {
    let segments = [ProjectionSegment::Sealed(SealedKind::Section(
        ResolvedSection::new(section),
    ))];
    scheduler
        .apply_projection(slot, BlockCount(0), &segments)
        .expect("projection applies");
}

#[test]
fn a_refused_section_page_is_a_gap_and_ends_the_sections_generation() {
    let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
    let (section, stream) = (SectionId::new(2), StreamId(12));
    let conversation = sealed_section(&mut scheduler, section, stream, 108);
    // A page in a format the double does not read — the way the model refuses
    // a page written before the index was Q8_0.
    scheduler.section_positional.insert(
        stream,
        &conversation,
        Arc::new(StoredPages::new(b"\x03\x00\x00\x00f32 index".to_vec())),
    );
    let slot = create_scratch_slot(&mut scheduler, &conversation);

    project_section(&mut scheduler, slot, section);

    assert_eq!(
        probe
            .pushed
            .lock()
            .unwrap()
            .get(&slot.0)
            .cloned()
            .unwrap_or_default(),
        vec![Pushed::Gap(108)],
        "the section's tokens are advanced over, exactly"
    );
    assert!(
        conversation.read().is_section_tombstoned(stream),
        "the section's generation ended, so the next request prefills it fresh"
    );
}

/// The page the model reads is pushed, and the section stands.
#[test]
fn a_readable_section_page_is_pushed_and_the_section_stands() {
    let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
    let (section, stream) = (SectionId::new(3), StreamId(13));
    let conversation = sealed_section(&mut scheduler, section, stream, 40);
    let page = [TOY_PAGE, b"-section"].concat();
    scheduler.section_positional.insert(
        stream,
        &conversation,
        Arc::new(StoredPages::new(page.clone())),
    );
    let slot = create_scratch_slot(&mut scheduler, &conversation);

    project_section(&mut scheduler, slot, section);

    assert_eq!(
        probe
            .pushed
            .lock()
            .unwrap()
            .get(&slot.0)
            .cloned()
            .unwrap_or_default(),
        vec![Pushed::Page(page)],
    );
    assert!(!conversation.read().is_section_tombstoned(stream));
}
