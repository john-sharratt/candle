//! `unit_convert` tool — convert between physical units.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// Numeric value to convert, expressed in the `from` unit. Required.
    pub value: f64,
    /// Source unit name or alias (e.g. "kg", "celsius", "GiB", "mi"). Case
    /// -insensitive. Required; must be a recognised unit. Must share a dimension
    /// with `to`.
    #[validate(length(min = 1))]
    pub from: String,
    /// Target unit name or alias (e.g. "lb", "fahrenheit", "GB", "km"). Case
    /// -insensitive. Required; must be a recognised unit. Must share a dimension
    /// with `from`.
    #[validate(length(min = 1))]
    pub to: String,
}

#[derive(Serialize)]
pub struct Response {
    pub value: f64,
    pub from: String,
    pub to: String,
    pub result: f64,
}

#[derive(Debug, Error)]
pub enum UnitError {
    #[error("unknown unit: {0}")]
    UnknownUnit(String),
    #[error("dimension mismatch: {0} and {1} are incompatible")]
    DimensionMismatch(String, String),
}

impl ToolError for UnitError {
    fn code(&self) -> &'static str {
        match self {
            UnitError::UnknownUnit(_) => "unknown_unit",
            UnitError::DimensionMismatch(_, _) => "dimension_mismatch",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum Dimension {
    Length,
    Mass,
    Volume,
    Temperature,
    Time,
    Data,
    Area,
    Speed,
    Pressure,
    Energy,
    Power,
    Frequency,
    Angle,
}

/// (dimension, factor_to_base, offset_to_base)
/// base units: meter, kg, liter, kelvin, second, byte
fn lookup_unit(unit: &str) -> Option<(Dimension, f64, f64)> {
    // Normalize the way the model naturally writes multi-word units: lowercase
    // and collapse spaces/hyphens to underscores, so "fluid ounce", "Fluid-Ounce",
    // and "fluid_ounce" all resolve identically.
    let u = unit.to_lowercase().replace([' ', '-'], "_");
    Some(match u.as_str() {
        // Length (base: meter)
        "m" | "meter" | "meters" | "metre" | "metres" => (Dimension::Length, 1.0, 0.0),
        "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => {
            (Dimension::Length, 1000.0, 0.0)
        }
        "cm" | "centimeter" | "centimeters" => (Dimension::Length, 0.01, 0.0),
        "mm" | "millimeter" | "millimeters" => (Dimension::Length, 0.001, 0.0),
        "um" | "micrometer" | "micrometers" | "micron" | "microns" => {
            (Dimension::Length, 1e-6, 0.0)
        }
        "nm" | "nanometer" | "nanometers" => (Dimension::Length, 1e-9, 0.0),
        "in" | "inch" | "inches" => (Dimension::Length, 0.0254, 0.0),
        "ft" | "foot" | "feet" => (Dimension::Length, 0.3048, 0.0),
        "yd" | "yard" | "yards" => (Dimension::Length, 0.9144, 0.0),
        "mi" | "mile" | "miles" => (Dimension::Length, 1609.344, 0.0),
        "nmi" | "nautical_mile" | "nautical_miles" => (Dimension::Length, 1852.0, 0.0),
        "ly" | "light_year" | "light_years" => (Dimension::Length, 9.461e15, 0.0),
        "au" | "astronomical_unit" => (Dimension::Length, 1.496e11, 0.0),

        // Mass (base: kg)
        "kg" | "kilogram" | "kilograms" => (Dimension::Mass, 1.0, 0.0),
        "g" | "gram" | "grams" => (Dimension::Mass, 0.001, 0.0),
        "mg" | "milligram" | "milligrams" => (Dimension::Mass, 1e-6, 0.0),
        "ug" | "microgram" | "micrograms" => (Dimension::Mass, 1e-9, 0.0),
        "t" | "tonne" | "tonnes" | "metric_ton" | "metric_tons" => (Dimension::Mass, 1000.0, 0.0),
        "lb" | "lbs" | "pound" | "pounds" => (Dimension::Mass, 0.453592, 0.0),
        "oz" | "ounce" | "ounces" => (Dimension::Mass, 0.0283495, 0.0),
        "st" | "stone" | "stones" => (Dimension::Mass, 6.35029, 0.0),
        "ton" | "short_ton" | "short_tons" => (Dimension::Mass, 907.185, 0.0),
        "long_ton" | "long_tons" => (Dimension::Mass, 1016.05, 0.0),

        // Volume (base: liter)
        "l" | "liter" | "liters" | "litre" | "litres" => (Dimension::Volume, 1.0, 0.0),
        "ml" | "milliliter" | "milliliters" => (Dimension::Volume, 0.001, 0.0),
        "cl" | "centiliter" | "centiliters" => (Dimension::Volume, 0.01, 0.0),
        "dl" | "deciliter" | "deciliters" => (Dimension::Volume, 0.1, 0.0),
        "kl" | "kiloliter" | "kiloliters" => (Dimension::Volume, 1000.0, 0.0),
        "m3" | "cubic_meter" | "cubic_meters" => (Dimension::Volume, 1000.0, 0.0),
        "cm3" | "cubic_centimeter" | "cubic_centimeters" | "cc" => (Dimension::Volume, 0.001, 0.0),
        "mm3" | "cubic_millimeter" | "cubic_millimeters" => (Dimension::Volume, 1e-6, 0.0),
        "gal" | "gallon" | "gallons" | "us_gal" => (Dimension::Volume, 3.78541, 0.0),
        "imp_gal" | "imperial_gallon" | "imperial_gallons" => (Dimension::Volume, 4.54609, 0.0),
        "qt" | "quart" | "quarts" => (Dimension::Volume, 0.946353, 0.0),
        "pt" | "pint" | "pints" => (Dimension::Volume, 0.473176, 0.0),
        "cup" | "cups" => (Dimension::Volume, 0.236588, 0.0),
        "fl_oz" | "fluid_ounce" | "fluid_ounces" | "us_fl_oz" | "us_fluid_ounce"
        | "us_fluid_ounces" => (Dimension::Volume, 0.0295735, 0.0),
        "tbsp" | "tablespoon" | "tablespoons" => (Dimension::Volume, 0.0147868, 0.0),
        "tsp" | "teaspoon" | "teaspoons" => (Dimension::Volume, 0.00492892, 0.0),

        // Temperature — affine: value_in_base = value * factor + offset
        // base: Kelvin
        "k" | "kelvin" => (Dimension::Temperature, 1.0, 0.0),
        "c" | "celsius" | "degc" | "°c" => (Dimension::Temperature, 1.0, 273.15),
        "f" | "fahrenheit" | "degf" | "°f" => (Dimension::Temperature, 5.0 / 9.0, 255.372222),
        "r" | "rankine" => (Dimension::Temperature, 5.0 / 9.0, 0.0),

        // Time (base: second)
        "s" | "sec" | "second" | "seconds" => (Dimension::Time, 1.0, 0.0),
        "ms" | "millisecond" | "milliseconds" => (Dimension::Time, 0.001, 0.0),
        "us" | "microsecond" | "microseconds" => (Dimension::Time, 1e-6, 0.0),
        "ns" | "nanosecond" | "nanoseconds" => (Dimension::Time, 1e-9, 0.0),
        "min" | "minute" | "minutes" => (Dimension::Time, 60.0, 0.0),
        "h" | "hr" | "hour" | "hours" => (Dimension::Time, 3600.0, 0.0),
        "d" | "day" | "days" => (Dimension::Time, 86400.0, 0.0),
        "wk" | "week" | "weeks" => (Dimension::Time, 604800.0, 0.0),
        "mo" | "month" | "months" => (Dimension::Time, 2629800.0, 0.0), // avg
        "yr" | "year" | "years" => (Dimension::Time, 31557600.0, 0.0),  // Julian

        // Data (base: byte)
        "b" | "byte" | "bytes" => (Dimension::Data, 1.0, 0.0),
        "kb" | "kilobyte" | "kilobytes" => (Dimension::Data, 1000.0, 0.0),
        "mb" | "megabyte" | "megabytes" => (Dimension::Data, 1e6, 0.0),
        "gb" | "gigabyte" | "gigabytes" => (Dimension::Data, 1e9, 0.0),
        "tb" | "terabyte" | "terabytes" => (Dimension::Data, 1e12, 0.0),
        "pb" | "petabyte" | "petabytes" => (Dimension::Data, 1e15, 0.0),
        "kib" | "kibibyte" | "kibibytes" => (Dimension::Data, 1024.0, 0.0),
        "mib" | "mebibyte" | "mebibytes" => (Dimension::Data, 1048576.0, 0.0),
        "gib" | "gibibyte" | "gibibytes" => (Dimension::Data, 1073741824.0, 0.0),
        "tib" | "tebibyte" | "tebibytes" => (Dimension::Data, 1099511627776.0, 0.0),
        "bit" | "bits" => (Dimension::Data, 0.125, 0.0),
        "kbit" | "kilobit" | "kilobits" => (Dimension::Data, 125.0, 0.0),
        "mbit" | "megabit" | "megabits" => (Dimension::Data, 125000.0, 0.0),
        "gbit" | "gigabit" | "gigabits" => (Dimension::Data, 125000000.0, 0.0),

        _ => return None,
    })
}

fn convert_temp(
    value: f64,
    from_factor: f64,
    from_offset: f64,
    to_factor: f64,
    to_offset: f64,
) -> f64 {
    // from -> kelvin -> to
    let kelvin = value * from_factor + from_offset;
    (kelvin - to_offset) / to_factor
}

pub struct UnitConvertTool;

impl Tool for UnitConvertTool {
    const NAME: &'static str = "unit_convert";
    const DESCRIPTION: &'static str =
        "Convert a numeric value between units of the same physical dimension — length, mass, \
         volume, temperature, time, or data size. Use for: converting kilograms to pounds, \
         miles to kilometres, GB to GiB, Celsius to Fahrenheit, hours to seconds. Triggered \
         by \"convert X to Y\", \"how many [unit] in\", \"what's [value] in [other unit]\", \
         \"X in metric/imperial\", \"express in\". Recognises common aliases (celsius/C/°C) and \
         the binary-vs-decimal distinction for data sizes (GB vs GiB). Returns the converted \
         value, input value, from/to unit names. Cannot cross dimensions (metres to kilograms \
         returns dimension_mismatch). For raw arithmetic without units use calculator.";

    type Request = Request;
    type Response = Response;
    type Error = UnitError;

    fn run(_ctx: &ToolContext, req: Request) -> Result<Response, UnitError> {
        let (from_dim, from_factor, from_offset) =
            lookup_unit(&req.from).ok_or_else(|| UnitError::UnknownUnit(req.from.clone()))?;
        let (to_dim, to_factor, to_offset) =
            lookup_unit(&req.to).ok_or_else(|| UnitError::UnknownUnit(req.to.clone()))?;

        if from_dim != to_dim {
            return Err(UnitError::DimensionMismatch(
                req.from.clone(),
                req.to.clone(),
            ));
        }

        let result = if from_dim == Dimension::Temperature {
            convert_temp(req.value, from_factor, from_offset, to_factor, to_offset)
        } else {
            // linear: value * from_factor / to_factor
            req.value * from_factor / to_factor
        };

        Ok(Response {
            value: req.value,
            from: req.from,
            to: req.to,
            result,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<UnitConvertTool>();
