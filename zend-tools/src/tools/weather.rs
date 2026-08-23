//! `weather` tool — current weather and forecast via Open-Meteo.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// City name or location to look up; resolved via geocoding.
    #[validate(length(min = 1))]
    pub location: String,
    /// Number of forecast days to return (0-7); 0 returns current conditions only. Default: 0.
    #[validate(range(max = 7))]
    pub forecast_days: Option<u8>,
    /// Unit system: "metric" (Celsius, km/h) or "imperial" (Fahrenheit, mph). Default: "metric".
    pub units: Option<String>,
}

#[derive(Serialize)]
pub struct Coords {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Serialize)]
pub struct CurrentWeather {
    pub temperature: f64,
    pub feels_like: f64,
    pub humidity: f64,
    pub wind_kph: f64,
    pub wind_direction_deg: f64,
    pub conditions: String,
    pub weather_code: i64,
}

#[derive(Serialize)]
pub struct DayForecast {
    pub date: String,
    pub high: f64,
    pub low: f64,
    pub conditions: String,
    pub precipitation_mm: f64,
}

#[derive(Serialize)]
pub struct Response {
    pub location: String,
    pub coords: Coords,
    pub timezone: String,
    pub units: String,
    pub current: CurrentWeather,
    pub forecast: Vec<DayForecast>,
}

#[derive(Debug, Error)]
pub enum WeatherError {
    #[error("location not found: {0}")]
    LocationNotFound(String),
    #[error("weather unavailable: {0}")]
    WeatherUnavailable(String),
}

impl ToolError for WeatherError {
    fn code(&self) -> &'static str {
        match self {
            WeatherError::LocationNotFound(_) => "location_not_found",
            WeatherError::WeatherUnavailable(_) => "weather_unavailable",
        }
    }
}

fn wmo_description(code: i64) -> &'static str {
    match code {
        0 => "Clear sky",
        1 => "Mainly clear",
        2 => "Partly cloudy",
        3 => "Overcast",
        45 | 48 => "Foggy",
        51 | 53 | 55 => "Drizzle",
        56 | 57 => "Freezing drizzle",
        61 | 63 | 65 => "Rain",
        66 | 67 => "Freezing rain",
        71 | 73 | 75 => "Snow",
        77 => "Snow grains",
        80..=82 => "Rain showers",
        85 | 86 => "Snow showers",
        95 => "Thunderstorm",
        96 | 99 => "Thunderstorm with hail",
        _ => "Unknown",
    }
}

pub struct WeatherTool;

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";
    const DESCRIPTION: &'static str =
        "Get current weather conditions and a short-term forecast for a city or location. \
         Use for: \"is it raining\", \"what's the weather like\", \"do I need a jacket\", \
         \"will it rain tomorrow\", \"temperature in [city]\", \"weather forecast for the weekend\", \
         \"how hot is it\", \"is there snow expected\". Returns current temperature, feels_like, \
         humidity, wind_kph, conditions text, and optional daily forecast (high/low/conditions/\
         precipitation) for up to 7 days. Resolves city names automatically. Use web_search \
         for weather articles, not live data. Backed by Open-Meteo (no API key needed).";

    type Request = Request;
    type Response = Response;
    type Error = WeatherError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, WeatherError> {
        let units = req.units.as_deref().unwrap_or("metric");
        let forecast_days = req.forecast_days.unwrap_or(0);

        // Geocode
        let geo_url = format!(
            "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=en&format=json",
            urlencoding::encode(&req.location)
        );
        let geo_resp = ctx
            .http_client
            .get(&geo_url)
            .send()
            .map_err(|e| WeatherError::WeatherUnavailable(e.to_string()))?;
        let geo_json: serde_json::Value = geo_resp
            .json()
            .map_err(|e| WeatherError::WeatherUnavailable(e.to_string()))?;

        let results = geo_json["results"]
            .as_array()
            .and_then(|a| a.first())
            .ok_or_else(|| WeatherError::LocationNotFound(req.location.clone()))?;

        let lat = results["latitude"].as_f64().unwrap_or(0.0);
        let lon = results["longitude"].as_f64().unwrap_or(0.0);
        let location_name = results["name"]
            .as_str()
            .unwrap_or(&req.location)
            .to_string();

        let wind_unit = if units == "imperial" { "mph" } else { "kmh" };
        let temp_unit = if units == "imperial" {
            "fahrenheit"
        } else {
            "celsius"
        };
        let actual_days = forecast_days.max(1);

        let weather_url = format!(
            "https://api.open-meteo.com/v1/forecast?\
             latitude={lat}&longitude={lon}\
             &current=temperature_2m,apparent_temperature,relative_humidity_2m,\
             wind_speed_10m,wind_direction_10m,weather_code\
             &daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum\
             &timezone=auto\
             &forecast_days={actual_days}\
             &wind_speed_unit={wind_unit}\
             &temperature_unit={temp_unit}"
        );

        let w_resp = ctx
            .http_client
            .get(&weather_url)
            .send()
            .map_err(|e| WeatherError::WeatherUnavailable(e.to_string()))?;
        let w_json: serde_json::Value = w_resp
            .json()
            .map_err(|e| WeatherError::WeatherUnavailable(e.to_string()))?;

        let cur = &w_json["current"];
        let wcode = cur["weather_code"].as_i64().unwrap_or(0);
        let current = CurrentWeather {
            temperature: cur["temperature_2m"].as_f64().unwrap_or(0.0),
            feels_like: cur["apparent_temperature"].as_f64().unwrap_or(0.0),
            humidity: cur["relative_humidity_2m"].as_f64().unwrap_or(0.0),
            wind_kph: cur["wind_speed_10m"].as_f64().unwrap_or(0.0),
            wind_direction_deg: cur["wind_direction_10m"].as_f64().unwrap_or(0.0),
            conditions: wmo_description(wcode).to_string(),
            weather_code: wcode,
        };

        let daily = &w_json["daily"];
        let dates = daily["time"].as_array().cloned().unwrap_or_default();
        let tmax = daily["temperature_2m_max"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let tmin = daily["temperature_2m_min"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let wcodes = daily["weather_code"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let precip = daily["precipitation_sum"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let forecast: Vec<DayForecast> = if forecast_days == 0 {
            vec![]
        } else {
            (0..forecast_days as usize)
                .filter_map(|i| {
                    Some(DayForecast {
                        date: dates.get(i)?.as_str()?.to_string(),
                        high: tmax.get(i)?.as_f64()?,
                        low: tmin.get(i)?.as_f64()?,
                        conditions: wmo_description(wcodes.get(i)?.as_i64().unwrap_or(0))
                            .to_string(),
                        precipitation_mm: precip.get(i)?.as_f64().unwrap_or(0.0),
                    })
                })
                .collect()
        };

        Ok(Response {
            location: location_name,
            coords: Coords { lat, lon },
            timezone: w_json["timezone"].as_str().unwrap_or("UTC").to_string(),
            units: units.to_string(),
            current,
            forecast,
        })
    }
}

// Need urlencoding helper
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<WeatherTool>();
