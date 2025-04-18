using System;
using System.Collections.Generic;
using System.Text;

namespace ConsoleApp1
{
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
        public string name { get; set; }
        public string region { get; set; }
        public string country { get; set; }
        public float lat { get; set; }
        public float lon { get; set; }
        public string tz_id { get; set; }
        public long localtime_epoch { get; set; }
        public string localtime { get; set; }
    }
    public class WeatherAPI
    {
        public WeatherAPILocation location { get; set; }
        public WeatherAPICurrent current { get; set; }
    }
}
