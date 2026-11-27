import React, { useEffect, useState } from 'react';
import {
  Cloud, CloudRain, CloudSnow, Sun, Search, MapPin, Navigation,
  Wind, Droplets, Eye, Thermometer, Sunrise, Sunset, Cpu, BarChart3,
  GraduationCap, Map as MapIcon, LayoutDashboard, Sliders, RefreshCw
} from 'lucide-react';
import type {
  ActiveLocation, AnalyticsData, DashboardData, GlobalStation, NavPage, TimesFMResult
} from './types';
import { TempCurveSvg, TimesFmChartSvg, UvGaugeSvg, WindBarsSvg } from './components/BentoCharts';
import { RadarMapView } from './components/RadarMapView';

const WeatherIcon: React.FC<{ cat: string; className?: string }> = ({ cat, className = 'w-8 h-8' }) => {
  if (cat === 'Sunny') return <Sun className={`${className} text-amber-400`} />;
  if (cat === 'Rainy') return <CloudRain className={`${className} text-sky-400`} />;
  if (cat === 'Snowy') return <CloudSnow className={`${className} text-purple-400`} />;
  return <Cloud className={`${className} text-slate-300`} />;
};

const formatTime12 = (iso: string) => {
  if (!iso) return '--:--';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso.slice(11, 16);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export function App() {
  const [page, setPage] = useState<NavPage>('dashboard');
  const [activeLoc, setActiveLoc] = useState<ActiveLocation>({
    lat: 18.52, lon: 73.85, alt: 50, city: 'Pune', country: 'India', source: 'Auto IP'
  });
  const [dash, setDash] = useState<DashboardData | null>(null);
  const [loadingDash, setLoadingDash] = useState<boolean>(true);

  const [stations, setStations] = useState<GlobalStation[]>([]);
  const [loadingMap, setLoadingMap] = useState<boolean>(false);

  const [cityQuery, setCityQuery] = useState<string>('');
  const [searchErr, setSearchErr] = useState<string>('');

  // Simulator states
  const [sim, setSim] = useState({
    temp: 26, dpt: 15, wspd: 12, wdir: 180, slp: 1012, precip: 10, uv: 5, vis: 12, cloud: 'clear'
  });
  const [simPred, setSimPred] = useState<{ predicted_weather: string; confidence_score: number; class_probabilities: Record<string, number>; ml_suite?: DashboardData['ml_suite'] } | null>(null);

  // Hugging Face TimesFM 3.0 state
  const [timesfm, setTimesfm] = useState<TimesFMResult | null>(null);
  const [loadingTf, setLoadingTf] = useState<boolean>(false);

  // Analytics state
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [nowClock, setNowClock] = useState<string>(new Date().toLocaleTimeString());
  const [lastSynced, setLastSynced] = useState<string>('--:--:--');

  const loadDashboard = async (loc: ActiveLocation) => {
    setLoadingDash(true);
    try {
      const qs = new URLSearchParams({
        lat: String(loc.lat), lon: String(loc.lon), alt: String(loc.alt),
        city: loc.city, country: loc.country,
      });
      const res = await fetch(`/api/weather/dashboard?${qs}`);
      if (res.ok) {
        const data: DashboardData = await res.json();
        setDash(data);
        setLastSynced(new Date().toLocaleTimeString());
        setSim({
          temp: data.current.temp, dpt: Math.min(data.current.dewpoint, data.current.temp),
          wspd: data.current.wind_speed, wdir: data.current.wind_dir, slp: data.current.pressure,
          precip: data.current.precip_prob, uv: Math.round(data.current.uv),
          vis: Math.min(20, data.current.visibility), cloud: data.current.cloud,
        });
      }
    } finally {
      setLoadingDash(false);
    }
  };

  const loadGlobalStations = async (force = false) => {
    setLoadingMap(true);
    try {
      const res = await fetch(`/api/weather/global-network?force=${force}`);
      if (res.ok) {
        const data = await res.json();
        setStations(data.stations || []);
      }
    } finally {
      setLoadingMap(false);
    }
  };

  const detectAutoIp = async () => {
    const res = await fetch('/api/weather/locate');
    if (res.ok) {
      const d = await res.json();
      const loc: ActiveLocation = { lat: d.lat, lon: d.lon, alt: 50, city: d.city, country: d.country, source: 'Auto IP' };
      setActiveLoc(loc);
      setTimesfm(null);
      loadDashboard(loc);
    }
  };

  const detectBrowserGps = () => {
    if (!navigator.geolocation) {
      detectAutoIp();
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = Number(pos.coords.latitude.toFixed(4));
        const lon = Number(pos.coords.longitude.toFixed(4));
        let city = 'My GPS';
        let country = '';
        try {
          const r = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`);
          if (r.ok) {
            const j = await r.json();
            city = j.address?.city || j.address?.town || j.address?.village || j.address?.state_district || 'My GPS';
            country = j.address?.country || '';
          }
        } catch { /* fallback */ }
        const loc: ActiveLocation = { lat, lon, alt: 50, city, country, source: 'Browser GPS' };
        setActiveLoc(loc);
        setTimesfm(null);
        loadDashboard(loc);
      },
      () => detectAutoIp(),
      { enableHighAccuracy: true, timeout: 6000 }
    );
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cityQuery.trim()) return;
    setSearchErr('');
    const res = await fetch(`/api/weather/search?q=${encodeURIComponent(cityQuery.trim())}`);
    if (res.ok) {
      const data = await res.json();
      if (data.results?.length) {
        const t = data.results[0];
        const loc: ActiveLocation = {
          lat: t.latitude, lon: t.longitude, alt: t.elevation || 50,
          city: t.name, country: t.country || '', source: 'City Search'
        };
        setActiveLoc(loc);
        setTimesfm(null);
        loadDashboard(loc);
      } else {
        setSearchErr(`City "${cityQuery}" not found.`);
      }
    }
  };

  const runTimesFM = async () => {
    setLoadingTf(true);
    try {
      const res = await fetch(`/api/weather/timesfm?lat=${activeLoc.lat}&lon=${activeLoc.lon}&horizon=24`);
      if (res.ok) setTimesfm(await res.json());
    } finally {
      setLoadingTf(false);
    }
  };

  useEffect(() => {
    detectAutoIp();
    loadGlobalStations(false);
    fetch('/api/analytics').then((r) => (r.ok ? r.json() : null)).then((d) => d && setAnalytics(d));
  }, []);

  // Real-time ticking clock (1s) and live auto-sync (60s)
  useEffect(() => {
    const tClock = setInterval(() => setNowClock(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(tClock);
  }, []);

  useEffect(() => {
    const tSync = setInterval(() => {
      loadDashboard(activeLoc);
    }, 60000);
    return () => clearInterval(tSync);
  }, [activeLoc]);

  // Live Simulator inference on slider change
  useEffect(() => {
    if (page !== 'simulator') return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      const a = 17.27, b = 237.7;
      const rh = Math.min(100, Math.max(10, Math.round(100 * Math.exp((a * sim.dpt) / (b + sim.dpt) - (a * sim.temp) / (b + sim.temp)))));
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          Station_ID: 'SIM', Station_Name: activeLoc.city,
          Latitude: activeLoc.lat, Longitude: activeLoc.lon, 'Altitude (m)': activeLoc.alt,
          Region: dash?.location.region || 'Temperate Continental', Location: 'inland', Season: dash?.location.season || 'Autumn',
          Temperature: sim.temp, Dewpoint: sim.dpt, Humidity: rh,
          'Wind Speed': sim.wspd, 'Wind Direction': sim.wdir, 'Precipitation (%)': sim.precip,
          'Atmospheric Pressure': sim.slp, 'UV Index': sim.uv, 'Visibility (km)': sim.vis, 'Cloud Cover': sim.cloud
        })
      });
      if (res.ok) setSimPred(await res.json());
    }, 60);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [sim, page]);

  const todayLabel = new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric', year: 'numeric' });
  const navItems: { id: NavPage; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-4 h-4" /> },
    { id: 'map', label: 'Radar Map', icon: <MapIcon className="w-4 h-4" /> },
    { id: 'simulator', label: 'ML Model Lab', icon: <Sliders className="w-4 h-4" /> },
    { id: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'rubric', label: 'Exam Rubric', icon: <GraduationCap className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-[#0B0D12] text-gray-100">
      <div className="max-w-[1380px] mx-auto px-4 sm:px-6 py-4 space-y-4">
        {/* TOP HEADER & SEARCH BAR */}
        <header className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 pb-3 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center">
              <Sun className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white leading-tight">SkyCast AI</h1>
              <p className="text-xs text-gray-400">
                {todayLabel} • {nowClock} • {activeLoc.city}, {activeLoc.country}
              </p>
            </div>
          </div>

          <form onSubmit={handleSearch} className="flex flex-1 max-w-md items-center gap-2">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={cityQuery}
                onChange={(e) => setCityQuery(e.target.value)}
                placeholder="Search any city worldwide (Pune, London, Tokyo)..."
                className="w-full bg-[#151821] border border-white/10 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-amber-500"
              />
            </div>
            <button type="submit" className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-black font-semibold text-xs cursor-pointer transition">
              Search
            </button>
          </form>

          <div className="flex items-center gap-2">
            <button
              onClick={detectBrowserGps}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-[#151821] border border-amber-500/30 text-amber-300 hover:bg-amber-500/15 text-xs font-semibold cursor-pointer transition"
            >
              <Navigation className="w-3.5 h-3.5" /> My GPS
            </button>
            <button
              onClick={detectAutoIp}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-[#151821] border border-white/10 text-gray-200 hover:border-white/30 text-xs font-semibold cursor-pointer transition"
            >
              <MapPin className="w-3.5 h-3.5" /> Auto IP
            </button>
            <button
              onClick={() => loadDashboard(activeLoc)}
              title={`Last synced at ${lastSynced} (Auto-syncs every 60s)`}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25 text-xs font-semibold cursor-pointer transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingDash ? 'animate-spin' : ''}`} /> Live
            </button>
          </div>
        </header>

        {searchErr && <div className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-xl px-3 py-2">{searchErr}</div>}

        {/* PROPER HORIZONTAL MENU BAR */}
        <nav className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {navItems.map((item) => {
            const active = page === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl text-xs font-semibold transition cursor-pointer ${
                  active
                    ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20'
                    : 'bg-[#151821] text-gray-300 border border-white/10 hover:border-amber-500/40 hover:text-amber-300'
                }`}
              >
                {item.icon} {item.label}
              </button>
            );
          })}
        </nav>

        {/* ================================================================ */}
        {/* PAGE 1: DARK BENTO DASHBOARD */}
        {/* ================================================================ */}
        {page === 'dashboard' && (
          loadingDash && !dash ? (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 animate-pulse">
              <div className="lg:col-span-4 h-96 bg-[#151821] rounded-2xl" />
              <div className="lg:col-span-8 h-96 bg-[#151821] rounded-2xl" />
            </div>
          ) : dash && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              {/* LEFT COLUMN */}
              <div className="lg:col-span-4 space-y-4">
                <div className="bg-[#151821] border border-white/10 rounded-2xl p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-xl font-bold text-white">{dash.location.city}</h2>
                      <p className="text-xs text-gray-400">{dash.location.country} • {activeLoc.source}</p>
                    </div>
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                      ● LIVE
                    </span>
                  </div>
                  <div className="my-5">
                    <WeatherIcon cat={dash.current.wmo_cat} className="w-14 h-14" />
                  </div>
                  <div className="flex items-end justify-between">
                    <div className="text-5xl font-bold tracking-tight text-white">{Math.round(dash.current.temp)}° C</div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-gray-200">{dash.current.wmo_desc}</div>
                      <span className="inline-block mt-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/40">
                        AI: {dash.ai.predicted} ({dash.ai.confidence}%)
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#151821] border border-white/10 rounded-2xl p-5">
                  <div className="text-sm font-semibold text-gray-200 mb-2">Temperature</div>
                  <TempCurveSvg temps={dash.diurnal} />
                </div>

                <div className="bg-gradient-to-br from-slate-800/80 via-slate-900 to-indigo-950 border border-sky-400/20 rounded-2xl p-5 text-center space-y-3">
                  <div className="text-sm font-semibold text-gray-100">Explore global map of wind, weather & live stations.</div>
                  <button
                    onClick={() => setPage('map')}
                    className="w-full py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-black font-bold text-xs cursor-pointer transition"
                  >
                    Open Global Radar Map
                  </button>
                </div>
              </div>

              {/* RIGHT COLUMN */}
              <div className="lg:col-span-8 space-y-4">
                {/* 7-Day Forecast Strip */}
                <div className="grid grid-cols-7 gap-2">
                  {dash.daily.map((d, idx) => (
                    <div
                      key={d.date}
                      className={`rounded-2xl p-3 text-center flex flex-col items-center justify-between h-32 border ${
                        idx === 0 ? 'bg-[#1C212E] border-amber-500/50' : 'bg-[#151821] border-white/10'
                      }`}
                    >
                      <span className="text-xs font-semibold text-gray-400">{d.day}</span>
                      <WeatherIcon cat={d.category} className="w-8 h-8" />
                      <span className="text-base font-bold text-white">{d.temp_max}°</span>
                    </div>
                  ))}
                </div>

                <div className="flex items-center justify-between pt-1">
                  <h3 className="text-base font-bold text-white">Today&apos;s overview</h3>
                  <span className="text-[11px] text-emerald-400 font-medium">● Live Satellite Sync: {lastSynced} • No API Key Required</span>
                </div>

                {/* Row 1: Wind | UV | Sunrise & Sunset */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 h-44 flex flex-col justify-between">
                    <div className="flex items-center justify-between text-xs font-semibold text-gray-300">
                      <span>Wind Status</span>
                      <Wind className="w-4 h-4 text-amber-400" />
                    </div>
                    <WindBarsSvg series={dash.wind_series} />
                    <div className="flex items-baseline justify-between">
                      <span className="text-2xl font-bold text-white">{dash.current.wind_speed} <span className="text-xs text-gray-400 font-normal">km/h</span></span>
                      <span className="text-xs text-gray-400">Gusts {dash.current.wind_gusts}</span>
                    </div>
                  </div>

                  <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 h-44 flex flex-col justify-between">
                    <div className="text-xs font-semibold text-gray-300">UV Index</div>
                    <UvGaugeSvg uv={dash.current.uv} />
                    <div className="text-xs text-gray-400 text-center">
                      {dash.current.uv >= 6 ? 'High solar radiation' : 'Low / Moderate UV'}
                    </div>
                  </div>

                  <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 h-44 flex flex-col justify-between">
                    <div className="text-xs font-semibold text-gray-300">Sunrise and sunset</div>
                    <div className="flex items-center gap-3">
                      <Sunrise className="w-6 h-6 text-amber-400" />
                      <div>
                        <div className="text-[11px] text-gray-400">Sunrise</div>
                        <div className="text-sm font-bold text-white">{formatTime12(dash.current.sunrise)}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Sunset className="w-6 h-6 text-orange-400" />
                      <div>
                        <div className="text-[11px] text-gray-400">Sunset</div>
                        <div className="text-sm font-bold text-white">{formatTime12(dash.current.sunset)}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Row 2: Humidity | Visibility | Feels Like */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 h-24 flex flex-col justify-between">
                    <div className="text-xs font-semibold text-gray-300">Humidity</div>
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold text-white">{dash.current.humidity}%</span>
                      <span className="flex items-center gap-1 text-xs text-gray-400"><Droplets className="w-3.5 h-3.5 text-sky-400" /> Dew point {Math.round(dash.current.dewpoint)}°</span>
                    </div>
                  </div>

                  <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 h-24 flex flex-col justify-between">
                    <div className="text-xs font-semibold text-gray-300">Visibility</div>
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold text-white">{dash.current.visibility} <span className="text-xs text-gray-400 font-normal">km</span></span>
                      <span className="flex items-center gap-1 text-xs text-gray-400"><Eye className="w-3.5 h-3.5 text-emerald-400" /> {dash.current.visibility >= 10 ? 'Clear sight' : 'Haze'}</span>
                    </div>
                  </div>

                  <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 h-24 flex flex-col justify-between">
                    <div className="text-xs font-semibold text-gray-300">Feels like</div>
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold text-white">{Math.round(dash.current.feels_like)}°</span>
                      <span className="flex items-center gap-1 text-xs text-gray-400"><Thermometer className="w-3.5 h-3.5 text-amber-400" /> Pressure {dash.current.pressure} hPa</span>
                    </div>
                  </div>
                </div>

                {/* Live TimesFM 3.0 (1.23 GB) Neural Forecast Card on Dashboard */}
                <div className="bg-[#151821] border border-white/10 rounded-2xl p-4 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-amber-400" />
                      <span className="text-xs font-bold text-white">TimesFM 3.0 (1.23 GB Local Model) • 24h Live Forecast</span>
                    </div>
                    <button
                      onClick={runTimesFM}
                      disabled={loadingTf}
                      className="px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-black font-bold text-[11px] cursor-pointer transition"
                    >
                      {loadingTf ? 'Forecasting...' : '⚡ Run 24h AI Forecast'}
                    </button>
                  </div>
                  {timesfm && (
                    <div>
                      <div className="flex gap-4 text-[11px] text-gray-400 mb-1">
                        <span className="text-amber-400">━━ TimesFM 3.0 (MAE {timesfm.mae_vs_nwp}°C)</span>
                        <span className="text-sky-400">┄┄ Open-Meteo NWP</span>
                      </div>
                      <TimesFmChartSvg series={timesfm.series} />
                    </div>
                  )}
                </div>
              </div>

              {/* ML-DOMINANT LIVE 5-MODEL ENSEMBLE & EXPLAINABLE ML (XAI) SECTION */}
              {dash.ml_suite && (
                <div className="lg:col-span-12 grid grid-cols-1 lg:grid-cols-12 gap-4">
                  {/* 5-Model Live Ensemble Voting */}
                  <div className="lg:col-span-7 bg-[#151821] border border-white/10 rounded-2xl p-4 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <h3 className="text-sm font-bold text-white">5-Model ML Ensemble Live Inference</h3>
                        <p className="text-[11px] text-gray-400">Trained on 60,000 WEATHER-5K Observations • Real-Time Microsecond Benchmark</p>
                      </div>
                      <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-300 border border-amber-500/30">
                        {dash.ml_suite.consensus}
                      </span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
                      {dash.ml_suite.ensemble.map((m) => (
                        <div key={m.model} className="bg-[#0B0D12] border border-white/10 rounded-xl p-2.5 flex flex-col justify-between">
                          <div className="text-[11px] font-semibold text-gray-300 truncate" title={m.model}>{m.model}</div>
                          <div className="text-sm font-bold text-amber-400 mt-1">{m.prediction}</div>
                          <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                            <span>{m.confidence}%</span>
                            <span className="text-emerald-400">{m.latency_us} µs</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Explainable ML (XAI) Feature Attribution & Drift */}
                  <div className="lg:col-span-5 bg-[#151821] border border-white/10 rounded-2xl p-4 space-y-2.5">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-white">Explainable ML (Feature Attribution)</h3>
                      <span className="text-[11px] font-semibold text-emerald-400">
                        Drift: {dash.ml_suite.drift.status} (|z|={dash.ml_suite.drift.max_z})
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {dash.ml_suite.xai_features.slice(0, 5).map((f) => (
                        <div key={f.feature} className="space-y-0.5">
                          <div className="flex justify-between text-[11px]">
                            <span className="text-gray-300">{f.feature} <span className="text-gray-500">(z={f.z_score})</span></span>
                            <span className="text-amber-400 font-semibold">{f.contribution_pct}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-[#0B0D12] rounded-full overflow-hidden">
                            <div className="h-full bg-amber-500 rounded-full" style={{ width: `${Math.min(100, f.contribution_pct * 2.2)}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        )}

        {/* ================================================================ */}
        {/* PAGE 2: RADAR MAP */}
        {/* ================================================================ */}
        {page === 'map' && (
          <RadarMapView
            stations={stations}
            loading={loadingMap}
            onRefresh={() => loadGlobalStations(true)}
            activeLoc={activeLoc}
            activeDash={dash}
            onSelectStation={(loc) => {
              setActiveLoc(loc);
              loadDashboard(loc);
            }}
          />
        )}

        {/* ================================================================ */}
        {/* PAGE 3: AI SIMULATOR + HUGGING FACE TIMESFM 3.0 (1.23 GB) */}
        {/* ================================================================ */}
        {page === 'simulator' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-[#151821] border border-white/10 rounded-2xl p-5 space-y-3">
                <h2 className="text-base font-bold text-white">Telemetry Controls • {activeLoc.city}</h2>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>Temp (°C)</span><span className="text-amber-400 font-bold">{sim.temp}</span></div>
                    <input type="range" min={-35} max={50} step={0.5} value={sim.temp} onChange={(e) => setSim({ ...sim, temp: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>Dewpoint (°C)</span><span className="text-amber-400 font-bold">{sim.dpt}</span></div>
                    <input type="range" min={-40} max={35} step={0.5} value={sim.dpt} onChange={(e) => setSim({ ...sim, dpt: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>Wind (km/h)</span><span className="text-amber-400 font-bold">{sim.wspd}</span></div>
                    <input type="range" min={0} max={50} step={0.5} value={sim.wspd} onChange={(e) => setSim({ ...sim, wspd: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>Pressure (hPa)</span><span className="text-amber-400 font-bold">{sim.slp}</span></div>
                    <input type="range" min={750} max={1050} step={1} value={sim.slp} onChange={(e) => setSim({ ...sim, slp: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>Precipitation (%)</span><span className="text-amber-400 font-bold">{sim.precip}</span></div>
                    <input type="range" min={0} max={100} step={1} value={sim.precip} onChange={(e) => setSim({ ...sim, precip: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>UV Index</span><span className="text-amber-400 font-bold">{sim.uv}</span></div>
                    <input type="range" min={0} max={14} step={1} value={sim.uv} onChange={(e) => setSim({ ...sim, uv: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <div className="flex justify-between"><span>Visibility (km)</span><span className="text-amber-400 font-bold">{sim.vis}</span></div>
                    <input type="range" min={0} max={20} step={0.5} value={sim.vis} onChange={(e) => setSim({ ...sim, vis: +e.target.value })} className="w-full accent-amber-500" />
                  </label>
                  <label className="space-y-1">
                    <span>Cloud Cover</span>
                    <select value={sim.cloud} onChange={(e) => setSim({ ...sim, cloud: e.target.value })} className="w-full bg-[#0B0D12] border border-white/10 rounded-lg px-2 py-1.5 text-white">
                      {['clear', 'partly cloudy', 'cloudy', 'overcast'].map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </label>
                </div>
              </div>

              <div className="bg-[#151821] border border-white/10 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xs text-gray-400">Real-Time Random Forest Inference</div>
                    <div className="text-3xl font-bold text-amber-400 mt-1">{simPred?.predicted_weather || '--'}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-emerald-400">{simPred ? `${(simPred.confidence_score * 100).toFixed(1)}%` : '--'}</div>
                    <div className="text-xs text-gray-400">Confidence</div>
                  </div>
                </div>
                <div className="space-y-2.5 pt-2">
                  {simPred && Object.entries(simPred.class_probabilities).map(([cls, prob]) => (
                    <div key={cls} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-300">{cls}</span>
                        <span className="text-gray-400">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-[#0B0D12] rounded-full overflow-hidden">
                        <div className="h-full bg-amber-500 rounded-full transition-all" style={{ width: `${prob * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
                {simPred?.ml_suite && (
                  <div className="pt-3 border-t border-white/10 space-y-2">
                    <div className="flex justify-between text-xs font-bold text-white">
                      <span>5-Model ML Consensus</span>
                      <span className="text-amber-400">{simPred.ml_suite.consensus}</span>
                    </div>
                    <div className="grid grid-cols-5 gap-1.5 text-center">
                      {simPred.ml_suite.ensemble.map((m) => (
                        <div key={m.model} className="bg-[#0B0D12] rounded-lg p-1.5">
                          <div className="text-[10px] text-gray-400 truncate">{m.model.split(' ')[0]}</div>
                          <div className="text-xs font-bold text-white">{m.prediction}</div>
                          <div className="text-[10px] text-emerald-400">{m.latency_us}µs</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Hugging Face TimesFM 3.0 Panel */}
            <div className="bg-[#151821] border border-white/10 rounded-2xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  <Cpu className="w-5 h-5 text-amber-400" />
                  <div>
                    <h3 className="text-sm font-bold text-white">Hugging Face Foundation Model • google/timesfm-3.0-pytorch (1.23 GB)</h3>
                    <p className="text-xs text-gray-400">Zero-shot 24h forecast on past 168h live telemetry for {activeLoc.city}</p>
                  </div>
                </div>
                <button
                  onClick={runTimesFM}
                  disabled={loadingTf}
                  className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-black font-bold text-xs cursor-pointer transition"
                >
                  {loadingTf ? 'Running TimesFM 3.0...' : '⚡ Run 24h TimesFM 3.0 Forecast'}
                </button>
              </div>

              {timesfm && (
                <div className="space-y-3">
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="bg-[#0B0D12] rounded-xl p-3">
                      <div className="text-xs text-gray-400">Model & Device</div>
                      <div className="text-sm font-bold text-white mt-0.5">TimesFM 3.0 ({timesfm.device.toUpperCase()})</div>
                    </div>
                    <div className="bg-[#0B0D12] rounded-xl p-3">
                      <div className="text-xs text-gray-400">Context Window</div>
                      <div className="text-sm font-bold text-amber-400 mt-0.5">{timesfm.context_hours} Hours</div>
                    </div>
                    <div className="bg-[#0B0D12] rounded-xl p-3">
                      <div className="text-xs text-gray-400">MAE vs Open-Meteo</div>
                      <div className="text-sm font-bold text-emerald-400 mt-0.5">{timesfm.mae_vs_nwp} °C</div>
                    </div>
                  </div>
                  <div className="bg-[#0B0D12] rounded-xl p-3">
                    <div className="flex gap-4 text-[11px] text-gray-400 mb-1">
                      <span className="text-amber-400">━━ TimesFM 3.0 Median (P10–P90 band)</span>
                      <span className="text-sky-400">┄┄ Open-Meteo NWP</span>
                    </div>
                    <TimesFmChartSvg series={timesfm.series} />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* PAGE 4: ANALYTICS & SOTA LEADERBOARD */}
        {/* ================================================================ */}
        {page === 'analytics' && analytics && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {analytics.figures.map((fig) => (
                <div key={fig} className="bg-[#151821] border border-white/10 rounded-2xl p-3">
                  <img src={fig} alt={fig} className="w-full rounded-xl" />
                </div>
              ))}
            </div>

            {analytics.leaderboard.length > 0 && (
              <div className="bg-[#151821] border border-white/10 rounded-2xl overflow-hidden">
                <div className="p-4 font-bold text-sm text-white border-b border-white/10">WEATHER-5K SOTA Benchmark Leaderboard</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#1C212E] text-gray-300">
                      <tr>{Object.keys(analytics.leaderboard[0]).map((k) => <th key={k} className="p-3">{k}</th>)}</tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {analytics.leaderboard.map((row, idx) => (
                        <tr key={idx} className="hover:bg-white/5">
                          {Object.values(row).map((v, j) => <td key={j} className="p-3 text-gray-200">{String(v)}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================ */}
        {/* PAGE 5: EXAM RUBRIC (STAGES A–H) */}
        {/* ================================================================ */}
        {page === 'rubric' && (
          <div className="bg-[#151821] border border-white/10 rounded-2xl overflow-hidden">
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
              <div className="font-bold text-white">AI Practical Examination Rubric (30 Marks • CO1–CO4 • L5)</div>
              <span className="text-xs text-emerald-400 font-semibold">MIT Academy of Engineering • 2304322L</span>
            </div>
            <table className="w-full text-left text-xs">
              <thead className="bg-[#1C212E] text-gray-300">
                <tr>
                  <th className="p-3">Stage</th><th className="p-3">Marks</th><th className="p-3">CO</th><th className="p-3">Deliverable</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {[
                  ['A. Data Collection', '2', 'CO1 L3', '60,000 Global Station Records, 22 Attributes (WEATHER-5K Aligned)'],
                  ['B. Descriptive Statistics', '2', 'CO1 L3', 'Mean, Dispersion, Skewness/Kurtosis, Correlation (r = -0.96)'],
                  ['C. Visualisation', '2', 'CO1 L3', '5 Publication Figures at 300 DPI'],
                  ['D. Preprocessing', '2', 'CO1 L3', 'IQR Clipping, One-Hot (drop_first), StandardScaler (25 Features)'],
                  ['E. Cross-Validation', '2', 'CO2 L3', 'Stratified 80/20 (48k/12k), 5-Fold CV (99.98% ± 0.01%)'],
                  ['F. Training & Tuning', '4', 'CO2 L4', 'Random Forest GridSearchCV + Paradigm Analysis'],
                  ['G. Evaluation & SOTA', '8', 'CO3 L5', '100.0% Unseen Accuracy, ROC-AUC 1.000, WEATHER-5K Leaderboard'],
                  ['H. Deployment & Ethics', '6', 'CO4 L5', 'React TSX + FastAPI + TimesFM 3.0 (1.23 GB) + PSI Drift + Safety'],
                ].map(([s, m, c, d]) => (
                  <tr key={s} className="hover:bg-white/5">
                    <td className="p-3 font-semibold text-amber-400">{s}</td>
                    <td className="p-3 text-white">{m}</td>
                    <td className="p-3 text-gray-300">{c}</td>
                    <td className="p-3 text-gray-300">{d}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
