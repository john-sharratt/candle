//! Tests the character record's byte layout and its compatibility with records
//! already on disk.
//!
//! Byte pins catch a renamed field, which a round trip against itself would miss.
//! Records written before the later fields existed still decode as unwritten, and
//! an id above 2^53 survives exactly. The record's tag is pinned, and it is shared
//! with the branch checkpoint's tag, so a collision cannot happen unnoticed.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    pub(in super::super) fn npc_fixture() -> NpcPayload {
        NpcPayload {
            npc_id: 10_237_749_914_772_934_281,
            owner_id: "u_8812".to_string(),
            revision: 3,
            created_ms: 1_740_200_000_000,
            updated_ms: 1_740_300_112_340,
            state: "active".to_string(),
            name: "Varek".to_string(),
            world_id: "battle-cities".to_string(),
            personality_id: "commander".to_string(),
            hidden: false,
            heartbeat_ms: 30_000,
            salience_gate: 0.5,
            tags: vec!["campaign-2".to_string(), "north".to_string()],
            persona_description: "Fifty-three, a former staff sergeant.".to_string(),
            persona_origin: "generated".to_string(),
            portrait_image_id: Some("img_4471".to_string()),
            portrait_origin: Some("uploaded".to_string()),
            // Absent, deliberately: this fixture backs the byte-pinned encoding
            // test, and a character that has never been embodied is what those
            // bytes were pinned against. `where_a_character_stood_round_trips`
            // covers the field being present.
            at: None,
            mood: None,
            suspended_from: None,
            beliefs: vec![AuthoredBelief {
                belief_id: "hess_word".to_string(),
                statement: "Hess keeps his word.".to_string(),
                confidence: 0.72,
                threshold: 0.6,
            }],
            relationships: vec![AuthoredRelationship {
                entity_id: "p_17".to_string(),
                display: "Ilse".to_string(),
                trust: 0.4,
                affect: 0.1,
                familiarity: 0.8,
                notes: "Met at the crossing.".to_string(),
            }],
            agency: vec![AuthoredStrategy {
                strategy_id: "hold_ridge".to_string(),
                statement: "Hold the eastern ridge.".to_string(),
                parent_id: None,
                state: "active".to_string(),
            }],
            modulation: Modulation {
                affect: -0.2,
                threat: 0.66,
                curiosity: 0.3,
            },
        }
    }

    /// The exact bytes, not a round trip. A field silently renamed by an edit
    /// to the struct round-trips perfectly against itself while making every
    /// record already on disk undecodable — asserting the encoding against
    /// itself cannot catch that, and this is the format's compatibility
    /// boundary.
    /// **A record written before the authoring layers existed still decodes.**
    ///
    /// This is what a real substrate holds. Without the defaults, the walker
    /// skips every one of them and a store of thirteen characters reports a cast
    /// of zero — with a warning per record that reads as log corruption rather
    /// than as a reader that has learned new required questions.
    ///
    /// A character nobody wrote beliefs for holds none, so an absent field and
    /// an empty one denote the same character.
    #[test]
    fn a_record_without_the_authoring_layers_decodes_as_an_unwritten_one() {
        let full = npc_fixture();
        let mut v = serde_json::to_value(&full).expect("encodes");
        let obj = v.as_object_mut().expect("an object");
        for layer in ["beliefs", "relationships", "agency", "modulation"] {
            assert!(obj.remove(layer).is_some(), "{layer} was not in the record");
        }

        let back: NpcPayload = serde_json::from_value(v).expect("an older record still decodes");
        assert!(back.beliefs.is_empty());
        assert!(back.relationships.is_empty());
        assert!(back.agency.is_empty());
        // And the fields that were always there are untouched.
        assert_eq!(back.npc_id, full.npc_id);
        assert_eq!(back.name, full.name);
        assert_eq!(back.world_id, full.world_id);
    }

    /// **Where a character stood round-trips, and its absence is not a place.**
    ///
    /// The world holds no persistence of its own — who is standing where lives
    /// in the daemon's RAM and goes with the process — so this field is the only
    /// thing a restart has to rebuild the world from. A record written before it
    /// existed decodes as a character that has never been embodied, which is the
    /// truth about it: it arrives at the way in, exactly as it did before.
    #[test]
    fn where_a_character_stood_round_trips() {
        let mut placed = npc_fixture();
        placed.at = Some("vault-casting/green-room".to_string());
        let back = NpcPayload::decode(&placed.encode()).unwrap();
        assert_eq!(back.at.as_deref(), Some("vault-casting/green-room"));
        assert_eq!(back, placed);

        // A record from before the field existed.
        let mut v = serde_json::to_value(npc_fixture()).unwrap();
        assert!(
            v.as_object_mut().unwrap().remove("at").is_none(),
            "an unplaced character must not write the key at all — it is a \
             per-move field on a record every compaction carries forward"
        );
        let older: NpcPayload = serde_json::from_value(v).unwrap();
        assert_eq!(older.at, None);
    }

    /// **A character a suspension of the whole cast holds carries the state it
    /// found in one key, and any other carries none.** The key sits between
    /// `portrait_origin` and the authoring plane, where the struct declares it;
    /// a record without it decodes as one no such suspension holds.
    #[test]
    fn the_state_a_cast_suspension_found_is_one_key_and_absent_otherwise() {
        let mut held = npc_fixture();
        held.state = "suspended".to_string();
        held.suspended_from = Some("active".to_string());
        let text = String::from_utf8(held.encode()).unwrap();
        assert!(
            text.contains(r#""portrait_origin":"uploaded","suspended_from":"active","beliefs":["#),
            "{text}"
        );
        assert!(text.contains(r#""state":"suspended","#), "{text}");
        assert_eq!(NpcPayload::decode(held.encode().as_slice()).unwrap(), held);

        let free = String::from_utf8(npc_fixture().encode()).unwrap();
        assert!(!free.contains("suspended_from"), "{free}");
        let older = NpcPayload::decode(free.as_bytes()).unwrap();
        assert_eq!(older.suspended_from, None);
    }

    #[test]
    fn npc_payload_encodes_exact_bytes() {
        let bytes = npc_fixture().encode();
        let expected = concat!(
            r#"{"npc_id":10237749914772934281,"owner_id":"u_8812","revision":3,"#,
            r#""created_ms":1740200000000,"updated_ms":1740300112340,"#,
            r#""state":"active","name":"Varek","world_id":"battle-cities","#,
            r#""personality_id":"commander","#,
            r#""hidden":false,"heartbeat_ms":30000,"#,
            r#""salience_gate":0.5,"tags":["campaign-2","north"],"#,
            r#""persona_description":"Fifty-three, a former staff sergeant.","#,
            r#""persona_origin":"generated","portrait_image_id":"img_4471","#,
            r#""portrait_origin":"uploaded","#,
            r#""beliefs":[{"belief_id":"hess_word","statement":"Hess keeps his word.","#,
            r#""confidence":0.72,"threshold":0.6}],"#,
            r#""relationships":[{"entity_id":"p_17","display":"Ilse","trust":0.4,"#,
            r#""affect":0.1,"familiarity":0.8,"notes":"Met at the crossing."}],"#,
            r#""agency":[{"strategy_id":"hold_ridge","statement":"Hold the eastern ridge.","#,
            r#""parent_id":null,"state":"active"}],"#,
            r#""modulation":{"affect":-0.2,"threat":0.66,"curiosity":0.3}}"#,
        );
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            expected,
            "the on-disk field names or their order changed"
        );
    }

    #[test]
    fn npc_payload_round_trips_through_a_record_frame() {
        let p = npc_fixture();
        let bytes = p.encode();
        let header = RecordHeader {
            record_type: RecordType::Npc,
            format: 0,
            payload_len: bytes.len() as u64,
            crc: crc32(&bytes),
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let frame = encode_record(&header, &bytes);
        let (hdr, payload, _total) = decode_record(&frame).unwrap();
        assert_eq!(hdr.record_type, RecordType::Npc);
        assert_eq!(NpcPayload::decode(payload).unwrap(), p);
    }

    /// An id above 2^53 must survive exactly. These are u64 on the wire as
    /// decimal strings precisely because a browser would round them; the same
    /// value must not be mangled on the way to disk either.
    #[test]
    fn a_large_npc_id_survives_encoding() {
        let mut p = npc_fixture();
        p.npc_id = u64::MAX;
        let back = NpcPayload::decode(&p.encode()).unwrap();
        assert_eq!(back.npc_id, u64::MAX);
        assert_eq!(back.owner_id, "u_8812");
    }

    #[test]
    fn a_character_without_a_portrait_omits_neither_field_nor_meaning() {
        let mut p = npc_fixture();
        p.portrait_image_id = None;
        p.portrait_origin = None;
        let bytes = p.encode();
        // Present as explicit nulls: absent-by-omission and absent-by-null
        // decode the same here, but a reader diffing two records should see
        // the field disappear only when the schema changes.
        assert!(std::str::from_utf8(&bytes)
            .unwrap()
            .contains(r#""portrait_image_id":null"#));
        assert_eq!(NpcPayload::decode(&bytes).unwrap(), p);
    }

    #[test]
    fn tombstoned_is_recognised_from_the_state_field() {
        let mut p = npc_fixture();
        assert!(!p.is_tombstoned());
        p.state = "tombstoned".to_string();
        assert!(p.is_tombstoned());
        // Deletion is a superseding record, not a removal: the payload still
        // carries the name and owner, so the id stays taken.
        let back = NpcPayload::decode(&p.encode()).unwrap();
        assert!(back.is_tombstoned());
        assert_eq!(back.name, "Varek");
    }

    /// The wire tag is the compatibility contract for `HeaderIndex` digests:
    /// change it and every index entry already written points at the wrong
    /// type.
    #[test]
    fn the_npc_record_tag_is_pinned() {
        assert_eq!(RecordType::Npc.tag(), 22);
        assert_eq!(RecordType::from_tag(22), RecordType::Npc);
        // 21 belongs to `BranchCheckpoint`, which claimed it first. Asserted
        // here as well as in that type's own test because the two were authored
        // on separate branches and both reached for the same free tag.
        assert_eq!(RecordType::from_tag(21), RecordType::BranchCheckpoint);
        // And an unrecognised tag still degrades rather than erroring, which is
        // what lets an older build read a log containing NPC records.
        assert_eq!(RecordType::from_tag(200), RecordType::Unknown);
    }

    #[test]
    fn the_npc_header_variant_round_trips_as_json() {
        let h = RecordHeader {
            record_type: RecordType::Npc,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let js = serde_json::to_string(&h).unwrap();
        // The header serialises the discriminant as `type`, not `record_type`.
        assert!(js.contains(r#""type":"npc""#), "{js}");
        let back: RecordHeader = serde_json::from_str(&js).unwrap();
        assert_eq!(back.record_type, RecordType::Npc);
    }
}
