using System;
using System.Collections.Generic;
using System.Reflection.Metadata.Ecma335;
using System.Text;

namespace ConsoleApp1
{
    public class WeatherAPITime
    { 
        public string FullDate { set {
                DateTime date = DateTime.Parse(value);
                DayOfWeek = date.DayOfWeek.ToString();
                Full = date.ToString("dd/MM/yyyy");
                Hour = date.ToString("HH:mm");
                Month = date.ToString("MMMM");
                Year = date.Year.ToString();
                Date = date.Day.ToString();
        } }
        public string DayOfWeek { get; set; }
        public string Full { get; set; }
        public string Hour { get; set; }
        public string Month { get; set; }
        public string Year { get; set; }
        public string Date { get; set; }
    }
    #region [Forecast]
    public class WeatherAPIForecastDayAstro
    {
        public string sunrise { get; set; }
        public string sunset { get; set; }
        public string moonrise { get; set; }
        public string moonset { get; set; }
        public string moon_phase { get; set; }
        public int moon_illumination { get; set; }
        public bool is_moon_up { get; set; }
        public bool is_sun_up { get; set; }
    }
    public class WeatherAPIForecastDaySingle
    { 
        public float maxtemp_c { get; set; }
        public float maxtemp_f { get; set; }
        public float mintemp_c { get; set; }
        public float mintemp_f { get; set; }
        public float avgtemp_c { get; set; }
        public float avgtemp_f { get; set; }
        public float maxwind_mph { get; set; }
        public float maxwind_kph { get; set; }
        public float totalprecip_mm { get; set; }
        public float totalprecip_in { get; set; }
        public float totalsnow_cm { get; set; }
        public float avgvis_km { get; set; }
        public float avgvis_miles { get; set; }
        public int avghumidity { get; set; }
        public bool daily_will_it_rain { get; set; }
        public bool daily_chance_of_rain { get; set; }
        public bool daily_will_it_snow { get; set; }
        public bool daily_chance_of_snow { get; set; }
        public float uv { get; set; }
        public WeatherAPICondition condition { get; set; }
    }
    public class WeatherAPIForecastDay
    { 
        public string date { get; set; }
        public long date_epoch { get; set; }
        public WeatherAPIForecastDaySingle day { get; set; }
        public WeatherAPIForecastDayAstro astro { get; set; }
        public WeatherAPICurrent[] hour { get; set; }
    }
    public class WeatherAPIForecast 
    {
        public WeatherAPIForecastDay[] forecastday { get; set; }
    }
    #endregion
    public class WeatherAPIAirQuality
    {
        public float co { get; set; }
        public float no2 { get; set; }
        public float o3 { get; set; }
        public float so2 { get; set; }
        public float pm2_5 { get; set; }
        public float pm10 { get; set; }
    }
    public class WeatherAPICondition
    {
        public string text { get; set; }
        public string icon { get; set; }
        public int code { get; set; }
    }
    public class WeatherAPICurrent
    {
        public WeatherAPITime TimeInfo = new WeatherAPITime();
        public string time { set { TimeInfo.FullDate = value; } }
        public string hour { get; set; }
        public long last_updated_epoch { get; set; }
        public string last_updated { get; set; }
        public float temp_c { get; set; }
        public float temp_f { get; set; }
        public bool is_day { get; set; }
        public WeatherAPICondition condition { get; set; }
        public float wind_mph { get; set; }
        public float wind_kph { get; set; }
        public float wind_degree { get; set; }
        public string wind_dir { get; set; }
        public float pressure_mb { get; set; }
        public float pressure_in { get; set; }
        public float precip_mm { get; set; }
        public float precip_in { get; set; }
        public int humidity { get; set; }
        public int cloud { get; set; }
        public float feelslike_c { get; set; }
        public float feelslike_f { get; set; }
        public float windchill_c { get; set; }
        public float windchill_f { get; set; }
        public float dewpoint_c { get; set; }
        public float dewpoint_f { get; set; }
        public float vis_km { get; set; }
        public float vis_miles { get; set; }
        public float uv { get; set; }
        public float gust_mph { get; set; }
        public float gust_kph { get; set; }
        public WeatherAPIAirQuality air_quality { get; set; }
    }
    public class WeatherAPILocation
    {
        public WeatherAPITime TimeInfo = new WeatherAPITime();
        public string name { get; set; }
        public string region { get; set; }
        public string country { get; set; }
        public float lat { get; set; }
        public float lon { get; set; }
        public string tz_id { get; set; }
        public long localtime_epoch { get; set; }
        public string localtime { set { TimeInfo.FullDate = value; } }
    }
    public class WeatherAPI
    {
        public WeatherAPILocation location { get; set; }
        public WeatherAPICurrent current { get; set; }
        public WeatherAPIForecast forecast { get; set; }
    }
}
