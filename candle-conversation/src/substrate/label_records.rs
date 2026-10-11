//! A timeline's label as the records that hold it.
//!
//! Each field of a label — `conv_id`, `label`, every `custom` key — keeps the
//! value of the greatest record that set it (§6.5), so a timeline's label is
//! the union of several records' winning fields. A rewrite carries it as
//! exactly those records: one `Label` per winning identity, holding the fields
//! that identity won and nothing else, so on replay every field lands on the
//! same value with the same identity, whatever order the records come back in.
//!
//! The least record that named a `conv_id` is carried too, because the
//! conversation's creation order is its clock reading (§6.5): dropping it
//! would move the conversation in the sidebar the first time a compaction ran.

use std::collections::BTreeMap;

use super::TimelineEntry;
use crate::persistence::identity::{RecordIdentity, VersionStamp};
use crate::persistence::manifest::ConvMeta;

type ByRecord = BTreeMap<(VersionStamp, u64), (ConvMeta, RecordIdentity)>;

/// The fields record `id` is carrying, started empty.
fn fields_of(by_record: &mut ByRecord, id: RecordIdentity) -> &mut ConvMeta {
    &mut by_record
        .entry((id.stamp(), id.oseq))
        .or_insert_with(|| (ConvMeta::default(), id))
        .0
}

/// `entry`'s label as the records that hold it, least stamp first.
pub fn label_records(entry: &TimelineEntry) -> Vec<(ConvMeta, RecordIdentity)> {
    let mut by_record = ByRecord::new();
    if let Some(conv_id) = &entry.conv_id {
        fields_of(&mut by_record, entry.ids.conv_id).conv_id = conv_id.clone();
    }
    if let Some(label) = &entry.label {
        fields_of(&mut by_record, entry.ids.label).label = label.clone();
    }
    for (k, v) in &entry.custom {
        let id = entry.ids.custom.get(k).copied().unwrap_or_default();
        fields_of(&mut by_record, id)
            .custom
            .insert(k.clone(), v.clone());
    }
    if let Some(named) = entry.ids.named {
        let meta = fields_of(&mut by_record, named);
        if meta.conv_id.is_empty() {
            meta.conv_id = entry.ids.named_conv_id.clone();
        }
    }
    by_record.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId};

    fn id(hlc: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 4,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        }
    }

    /// Fields won by different records come back as one record each, holding
    /// only what that record won, plus the record that first named the
    /// conversation.
    #[test]
    fn each_winning_record_carries_its_own_fields() {
        let mut entry =
            TimelineEntry::new(LayerId::from_raw(1).unwrap(), GroupId::from_raw(1).unwrap());
        entry.conv_id = Some("c-2".into());
        entry.ids.conv_id = id(20);
        entry.label = Some("title".into());
        entry.ids.label = id(30);
        entry.custom.insert("k".into(), "v".into());
        entry.ids.custom.insert("k".into(), id(20));
        entry.ids.named = Some(id(10));
        entry.ids.named_conv_id = "c-1".into();

        let got: Vec<(u64, String, String, Vec<String>)> = label_records(&entry)
            .iter()
            .map(|(m, rid)| {
                (
                    rid.hlc,
                    m.conv_id.clone(),
                    m.label.clone(),
                    m.custom.keys().cloned().collect(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (10, "c-1".into(), String::new(), vec![]),
                (20, "c-2".into(), String::new(), vec!["k".to_string()]),
                (30, String::new(), "title".into(), vec![]),
            ]
        );
    }

    /// A label written before identities existed comes back as the one record
    /// it always was.
    #[test]
    fn an_unstamped_label_is_one_record() {
        let mut entry =
            TimelineEntry::new(LayerId::from_raw(1).unwrap(), GroupId::from_raw(1).unwrap());
        entry.conv_id = Some("c".into());
        entry.label = Some("t".into());
        entry.custom.insert("k".into(), "v".into());
        entry.ids.named = Some(RecordIdentity::NONE);
        entry.ids.named_conv_id = "c".into();
        let records = label_records(&entry);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0.conv_id, "c");
        assert_eq!(records[0].0.label, "t");
        assert_eq!(records[0].1, RecordIdentity::NONE);
    }
}
