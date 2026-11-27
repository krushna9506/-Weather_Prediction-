import React from 'react';
import type { TimesFMPoint } from '../types';

export const TempCurveSvg: React.FC<{ temps: number[] }> = ({ temps }) => {
  const vals = temps.length >= 4 ? temps.slice(0, 4) : [20, 25, 22, 18];
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = Math.max(max - min, 2);
  const ys = vals.map((t) => 46 - ((t - min) / span) * 26);
  const xs = [30, 105, 180, 255];
  const labels = ['Morning', 'Afternoon', 'Evening', 'Night'];

  const pathD = `M ${xs[0]},${ys[0]} C 65,${ys[0]} 70,${ys[1]} ${xs[1]},${ys[1]} C 140,${ys[1]} 145,${ys[2]} ${xs[2]},${ys[2]} C 215,${ys[2]} 220,${ys[3]} ${xs[3]},${ys[3]}`;

  return (
    <svg viewBox="0 0 285 90" className="w-full h-24">
      <defs>
        <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${pathD} L ${xs[3]},62 L ${xs[0]},62 Z`} fill="url(#tempGrad)" />
      <path d={pathD} fill="none" stroke="#F59E0B" strokeWidth="2.4" strokeLinecap="round" />
      {xs.map((x, i) => (
        <g key={labels[i]}>
          <circle cx={x} cy={ys[i]} r="3.6" fill="#FBBF24" stroke="#151821" strokeWidth="1.5" />
          <text x={x} y="74" fill="#9CA3AF" fontSize="9.5" textAnchor="middle">{labels[i]}</text>
          <text x={x} y="87" fill="#F3F4F6" fontSize="10.5" fontWeight="600" textAnchor="middle">
            {Math.round(vals[i])}°
          </text>
        </g>
      ))}
    </svg>
  );
};

export const WindBarsSvg: React.FC<{ series: number[] }> = ({ series }) => {
  const vals = series.length >= 16 ? series.slice(0, 18) : [8, 11, 14, 12, 16, 19, 15, 13, 11, 9, 12, 15, 17, 14, 11, 10, 12, 9];
  const wMax = Math.max(...vals, 5);
  return (
    <svg viewBox="0 0 260 54" className="w-full h-14">
      {vals.map((v, idx) => {
        const h = Math.max(4, (v / wMax) * 44);
        const x = 6 + idx * 14;
        const y = 50 - h;
        const opacity = v >= wMax * 0.7 ? 0.95 : 0.55;
        return <rect key={idx} x={x} y={y} width="5" height={h} rx="2.5" fill="#F59E0B" opacity={opacity} />;
      })}
    </svg>
  );
};

export const UvGaugeSvg: React.FC<{ uv: number }> = ({ uv }) => {
  const clamped = Math.min(Math.max(uv, 0), 12);
  const dash = (clamped / 12) * 157;
  return (
    <svg viewBox="0 0 140 76" className="w-full h-20">
      <path d="M 20,66 A 50,50 0 0,1 120,66" fill="none" stroke="#262B38" strokeWidth="11" strokeLinecap="round" />
      <path
        d="M 20,66 A 50,50 0 0,1 120,66"
        fill="none"
        stroke="#F59E0B"
        strokeWidth="11"
        strokeLinecap="round"
        strokeDasharray={`${dash.toFixed(1)} 157.0`}
      />
      <text x="70" y="63" fill="#F9FAFB" fontSize="16" fontWeight="700" textAnchor="middle">
        {uv.toFixed(1)} <tspan fontSize="10" fill="#9CA3AF">UV</tspan>
      </text>
    </svg>
  );
};

export const TimesFmChartSvg: React.FC<{ series: TimesFMPoint[] }> = ({ series }) => {
  if (!series.length) return null;
  const allVals = series.flatMap((s) => [s.low, s.high, s.nwp]);
  const min = Math.min(...allVals) - 1;
  const max = Math.max(...allVals) + 1;
  const span = Math.max(max - min, 2);
  const W = 680;
  const H = 170;
  const padX = 36;
  const padY = 16;
  const xAt = (i: number) => padX + (i / Math.max(series.length - 1, 1)) * (W - padX * 2);
  const yAt = (v: number) => H - padY - ((v - min) / span) * (H - padY * 2);

  const tfPath = series.map((s, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)},${yAt(s.timesfm).toFixed(1)}`).join(' ');
  const nwpPath = series.map((s, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)},${yAt(s.nwp).toFixed(1)}`).join(' ');
  const bandTop = series.map((s, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)},${yAt(s.high).toFixed(1)}`).join(' ');
  const bandBot = [...series].reverse().map((s, i) => `L ${xAt(series.length - 1 - i).toFixed(1)},${yAt(s.low).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H + 22}`} className="w-full h-52">
      <path d={`${bandTop} ${bandBot} Z`} fill="#F59E0B" opacity="0.16" />
      <path d={nwpPath} fill="none" stroke="#38BDF8" strokeWidth="1.8" strokeDasharray="4 3" />
      <path d={tfPath} fill="none" stroke="#F59E0B" strokeWidth="2.5" />
      {[0, 6, 12, 18, series.length - 1].map((idx) => (
        series[idx] ? (
          <text key={idx} x={xAt(idx)} y={H + 14} fill="#9CA3AF" fontSize="10" textAnchor="middle">
            {series[idx].time}
          </text>
        ) : null
      ))}
      <text x="6" y={yAt(max - 1) + 4} fill="#9CA3AF" fontSize="10">{Math.round(max - 1)}°</text>
      <text x="6" y={yAt(min + 1) + 4} fill="#9CA3AF" fontSize="10">{Math.round(min + 1)}°</text>
    </svg>
  );
};
