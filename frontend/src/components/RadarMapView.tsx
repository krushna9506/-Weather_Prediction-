import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import type { GlobalStation, DashboardData, ActiveLocation } from '../types';
import { RefreshCw, Crosshair } from 'lucide-react';

interface Props {
  stations: GlobalStation[];
  loading: boolean;
  onRefresh: () => void;
  activeLoc: ActiveLocation;
  activeDash: DashboardData | null;
  onSelectStation: (loc: ActiveLocation) => void;
}

const COLOR_BY_WEATHER: Record<string, string> = {
  Sunny: '#F59E0B',
  Rainy: '#0EA5E9',
  Cloudy: '#94A3B8',
  Snowy: '#A855F7',
};

const TILES: Record<string, string> = {
  'Dark Radar': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  'Voyager Terrain': 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
  'Light Atlas': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
};

export const RadarMapView: React.FC<Props> = ({
  stations,
  loading,
  onRefresh,
  activeLoc,
  activeDash,
  onSelectStation,
}) => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const leafletMap = useRef<L.Map | null>(null);
  const tileLayer = useRef<L.TileLayer | null>(null);
  const markerLayer = useRef<L.LayerGroup | null>(null);

  const [theme, setTheme] = useState<string>('Dark Radar');
  const [filter, setFilter] = useState<string>('All');

  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return;
    const map = L.map(mapRef.current, { center: [22, 15], zoom: 2, minZoom: 2, worldCopyJump: true });
    tileLayer.current = L.tileLayer(TILES[theme], { attribution: '&copy; OpenStreetMap &copy; CARTO' }).addTo(map);
    markerLayer.current = L.layerGroup().addTo(map);
    leafletMap.current = map;
    return () => {
      map.remove();
      leafletMap.current = null;
    };
  }, []);

  useEffect(() => {
    if (tileLayer.current) {
      tileLayer.current.setUrl(TILES[theme]);
    }
  }, [theme]);

  useEffect(() => {
    const group = markerLayer.current;
    if (!group) return;
    group.clearLayers();

    const filtered = filter === 'All' ? stations : stations.filter((s) => s.predicted === filter);

    filtered.forEach((st) => {
      const col = COLOR_BY_WEATHER[st.predicted] || '#F59E0B';
      // Outer precipitation halo
      L.circleMarker([st.lat, st.lon], {
        radius: 14 + Math.min(12, st.precip_prob / 10),
        color: col,
        weight: 0,
        fillColor: col,
        fillOpacity: 0.18,
      }).addTo(group);

      // Core station pin
      const pin = L.circleMarker([st.lat, st.lon], {
        radius: 7.5,
        color: '#ffffff',
        weight: 1.8,
        fillColor: col,
        fillOpacity: 0.92,
      }).addTo(group);

      pin.bindPopup(`
        <div style="min-width:210px; font-family:Inter,sans-serif; font-size:12px;">
          <div style="font-size:14px; font-weight:700; color:#FBBF24; margin-bottom:2px;">${st.name}, ${st.country}</div>
          <div style="color:#9CA3AF; margin-bottom:6px;">Elev: ${st.alt}m • ${st.region}</div>
          <div style="margin-bottom:4px;"><b>AI:</b> ${st.predicted} (${st.confidence}%) • <b>WMO:</b> ${st.wmo_desc}</div>
          <div>🌡️ ${st.temp}°C (Feels ${st.feels_like}°C) | 💧 ${st.humidity}%</div>
          <div>⏲️ ${st.pressure} hPa | 💨 ${st.wind_speed} km/h | 🌧️ ${st.precip_prob}%</div>
        </div>
      `);

      pin.on('click', () => {
        onSelectStation({ lat: st.lat, lon: st.lon, alt: st.alt, city: st.name, country: st.country, source: 'Radar Pin' });
      });
    });

    // Active GPS/Searched Pin
    if (activeDash) {
      const col = COLOR_BY_WEATHER[activeDash.ai.predicted] || '#06B6D4';
      L.circleMarker([activeLoc.lat, activeLoc.lon], {
        radius: 20,
        color: '#06B6D4',
        weight: 2,
        fillColor: '#06B6D4',
        fillOpacity: 0.22,
      }).addTo(group);

      L.circleMarker([activeLoc.lat, activeLoc.lon], {
        radius: 9,
        color: '#06B6D4',
        weight: 2.5,
        fillColor: col,
        fillOpacity: 1,
      })
        .bindPopup(`
          <div style="min-width:210px; font-family:Inter,sans-serif; font-size:12px;">
            <div style="font-size:14px; font-weight:700; color:#22D3EE;">📍 ${activeLoc.city} (${activeLoc.source})</div>
            <div><b>AI:</b> ${activeDash.ai.predicted} (${activeDash.ai.confidence}%) • ${activeDash.current.wmo_desc}</div>
            <div>🌡️ ${activeDash.current.temp}°C | 💧 ${activeDash.current.humidity}% | 💨 ${activeDash.current.wind_speed} km/h</div>
          </div>
        `)
        .addTo(group);
    }
  }, [stations, filter, activeLoc, activeDash]);

  const flyToActive = () => {
    leafletMap.current?.flyTo([activeLoc.lat, activeLoc.lon], 6, { duration: 1.0 });
  };

  const resetGlobal = () => {
    leafletMap.current?.flyTo([22, 15], 2, { duration: 1.0 });
  };

  const avgTemp = stations.length ? (stations.reduce((a, b) => a + b.temp, 0) / stations.length).toFixed(1) : '--';
  const avgHum = stations.length ? Math.round(stations.reduce((a, b) => a + b.humidity, 0) / stations.length) : '--';
  const peakWind = stations.length ? stations.reduce((a, b) => (b.wind_speed > a.wind_speed ? b : a), stations[0]) : null;
  const filteredRows = filter === 'All' ? stations : stations.filter((s) => s.predicted === filter);

  return (
    <div className="space-y-4">
      {/* KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-[#151821] border border-white/10 rounded-2xl p-4">
          <div className="text-xs text-gray-400">Live Observatories</div>
          <div className="text-2xl font-bold text-white mt-1">{stations.length} Stations</div>
        </div>
        <div className="bg-[#151821] border border-white/10 rounded-2xl p-4">
          <div className="text-xs text-gray-400">Global Avg Temp</div>
          <div className="text-2xl font-bold text-amber-400 mt-1">{avgTemp}° C</div>
        </div>
        <div className="bg-[#151821] border border-white/10 rounded-2xl p-4">
          <div className="text-xs text-gray-400">Global Avg Humidity</div>
          <div className="text-2xl font-bold text-sky-400 mt-1">{avgHum}%</div>
        </div>
        <div className="bg-[#151821] border border-white/10 rounded-2xl p-4">
          <div className="text-xs text-gray-400">Peak Wind Station</div>
          <div className="text-xl font-bold text-emerald-400 mt-1">
            {peakWind ? `${peakWind.wind_speed} km/h (${peakWind.name})` : '--'}
          </div>
        </div>
      </div>

      {/* Map Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-[#151821] border border-white/10 rounded-2xl p-3">
        <div className="flex flex-wrap gap-1.5">
          {['All', 'Sunny', 'Rainy', 'Cloudy', 'Snowy'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
                filter === f ? 'bg-amber-500 text-black' : 'bg-[#0B0D12] text-gray-300 hover:text-amber-400'
              }`}
            >
              {f}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            className="bg-[#0B0D12] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200"
          >
            {Object.keys(TILES).map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <button
            onClick={flyToActive}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 cursor-pointer"
          >
            <Crosshair className="w-3.5 h-3.5" /> Focus {activeLoc.city}
          </button>
          <button
            onClick={resetGlobal}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-[#0B0D12] text-gray-300 border border-white/10 hover:text-white cursor-pointer"
          >
            Global View
          </button>
          <button
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30 hover:bg-amber-500/25 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Sync
          </button>
        </div>
      </div>

      {/* Interactive Leaflet Map */}
      <div ref={mapRef} className="w-full h-[460px] border border-white/10 rounded-2xl overflow-hidden shadow-2xl" />

      {/* Station Telemetry Table */}
      <div className="bg-[#151821] border border-white/10 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto max-h-72">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#1C212E] text-gray-300 sticky top-0">
              <tr>
                <th className="p-3">Station</th>
                <th className="p-3">Country</th>
                <th className="p-3">Temp</th>
                <th className="p-3">Feels</th>
                <th className="p-3">RH</th>
                <th className="p-3">Pressure</th>
                <th className="p-3">Wind</th>
                <th className="p-3">WMO Observed</th>
                <th className="p-3">AI Prediction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filteredRows.map((s) => (
                <tr
                  key={s.id}
                  onClick={() => onSelectStation({ lat: s.lat, lon: s.lon, alt: s.alt, city: s.name, country: s.country, source: 'Station' })}
                  className="hover:bg-white/5 cursor-pointer transition"
                >
                  <td className="p-3 font-semibold text-white">{s.name}</td>
                  <td className="p-3 text-gray-400">{s.country}</td>
                  <td className="p-3 font-semibold text-amber-400">{s.temp}°C</td>
                  <td className="p-3 text-gray-300">{s.feels_like}°C</td>
                  <td className="p-3 text-sky-400">{s.humidity}%</td>
                  <td className="p-3 text-gray-300">{s.pressure} hPa</td>
                  <td className="p-3 text-gray-300">{s.wind_speed} km/h</td>
                  <td className="p-3 text-gray-300">{s.wmo_desc}</td>
                  <td className="p-3">
                    <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 text-amber-300">
                      {s.predicted} ({s.confidence}%)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
