//! A collection's `range:` (`docs/placed_section_layout.md` §3.1, §3.7b).

use super::*;

/// The range a collection declares, checked: a size no member could stand in
/// — zero, or a small-context size past the full one — is refused by name.
pub(super) fn parse_reserved_range(
    label: &str,
    yaml: Option<&YamlReservedRange>,
) -> Result<Option<ReservedRange>, ConstructionError> {
    let Some(r) = yaml else {
        return Ok(None);
    };
    let small_context = r.small_context.unwrap_or(r.tokens);
    if r.tokens == 0 || small_context == 0 {
        return Err(ConstructionError::InvalidRange(format!(
            "{label}: a range of {} positions ({small_context} on a small context) holds nothing",
            r.tokens
        )));
    }
    if small_context > r.tokens {
        return Err(ConstructionError::InvalidRange(format!(
            "{label}: the small-context range ({small_context}) is past the full one ({})",
            r.tokens
        )));
    }
    let fill = match r.fill.as_deref() {
        None | Some("top") => RangeFill::Top,
        Some("bottom") => RangeFill::Bottom,
        Some(other) => {
            return Err(ConstructionError::InvalidRange(format!(
                "{label}: a range fills from `top` or `bottom`, not {other:?}"
            )))
        }
    };
    Ok(Some(ReservedRange {
        tokens: r.tokens,
        small_context,
        fill,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(tokens: usize, small_context: Option<usize>) -> YamlReservedRange {
        YamlReservedRange {
            tokens,
            small_context,
            fill: None,
        }
    }

    /// **A range with no small size takes the full one on every context**, and
    /// fills from the top unless it says otherwise.
    #[test]
    fn a_range_with_no_small_size_takes_the_full_one() {
        assert_eq!(
            parse_reserved_range("p/tools", Some(&yaml(8_192, None))).unwrap(),
            Some(ReservedRange {
                tokens: 8_192,
                small_context: 8_192,
                fill: RangeFill::Top,
            })
        );
        assert_eq!(parse_reserved_range("p/tools", None).unwrap(), None);
        let bottom = YamlReservedRange {
            fill: Some("bottom".into()),
            ..yaml(8_192, Some(2_048))
        };
        assert_eq!(
            parse_reserved_range("p/journal", Some(&bottom))
                .unwrap()
                .map(|r| r.fill),
            Some(RangeFill::Bottom)
        );
        let sideways = YamlReservedRange {
            fill: Some("middle".into()),
            ..yaml(8_192, None)
        };
        assert!(parse_reserved_range("p/journal", Some(&sideways)).is_err());
    }

    /// **An empty range, or a small size past the full one, is refused.**
    #[test]
    fn an_empty_or_inverted_range_is_refused() {
        assert!(parse_reserved_range("p/a", Some(&yaml(0, None))).is_err());
        assert!(parse_reserved_range("p/a", Some(&yaml(8, Some(0)))).is_err());
        assert!(parse_reserved_range("p/a", Some(&yaml(8, Some(9)))).is_err());
        assert!(parse_reserved_range("p/a", Some(&yaml(8, Some(8)))).is_ok());
    }
}
