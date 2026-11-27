export type NavPage = 'dashboard' | 'map' | 'simulator' | 'analytics' | 'rubric';

export interface ActiveLocation {
  lat: number;
  lon: number;
  alt: number;
  city: string;
  country: string;
  source: string;
}

export interface DayForecast {
  date: string;
  day: string;
  temp_max: number;
  temp_min: number;
  category: string;
  desc: string;
}

export interface DashboardData {
  location: {
    city: string;
    country: string;
    lat: number;
    lon: number;
    alt: number;
    region: string;
    season: string;
  };
  current: {
    temp: number;
    feels_like: number;
    dewpoint: number;
    humidity: number;
    pressure: number;
    wind_speed: number;
    wind_gusts: number;
    wind_dir: number;
    uv: number;
    visibility: number;
    precip_prob: number;
    cloud: string;
    cloud_pct: number;
    wmo_desc: string;
    wmo_cat: string;
    sunrise: string;
    sunset: string;
    timestamp: string;
  };
  ai: {
    predicted: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  diurnal: number[];
  wind_series: number[];
  daily: DayForecast[];
}

export interface GlobalStation {
  id: string;
  name: string;
  country: string;
  lat: number;
  lon: number;
  alt: number;
  region: string;
  loc: string;
  temp: number;
  feels_like: number;
  humidity: number;
  pressure: number;
  wind_speed: number;
  wind_gusts: number;
  precip_prob: number;
  visibility: number;
  wmo_desc: string;
  predicted: string;
  confidence: number;
  is_live: boolean;
}

export interface TimesFMPoint {
  time: string;
  timesfm: number;
  low: number;
  high: number;
  nwp: number;
  pressure: number;
}

export interface TimesFMResult {
  model_id: string;
  device: string;
  context_hours: number;
  mae_vs_nwp: number;
  series: TimesFMPoint[];
}

export interface AnalyticsData {
  comparison: Record<string, string | number>[];
  leaderboard: Record<string, string | number>[];
  figures: string[];
}
