const { useState, useMemo, useRef, useEffect } = React;

const clamp = (v,a,b) => Math.max(a,Math.min(b,v));

// ── REAL DATA ────────────────────────────────────────────────────────────────
const AGES = [0,10,20,30,40,50,60,70,80];
const PLAYERS    = window.__DATA__.players;
const MIN_QUALIFYING_SHIFT_SECONDS = window.__DATA__.min_qualifying_shift_seconds ?? 10;
const MIN_QUALIFYING_SHIFTS = window.__DATA__.min_qualifying_shifts ?? 200;
const CHARTED    = PLAYERS.filter(p => p.eligible_for_graphs);
const LINES      = window.__DATA__.lines;
const LEAGUE_P25 = window.__DATA__.league_p25;
const LEAGUE_P75 = window.__DATA__.league_p75;
const LEAGUE_MED = window.__DATA__.league_med;
const TEAMS      = [...new Set(PLAYERS.map(p=>p.team))].sort();
const SEASONS    = window.__DATA__.seasons;
const RAPM_READY = window.__DATA__.rapm_ready !== false;

// ── UTILS ────────────────────────────────────────────────────────────────────
const normalizeSearchText = (value='') => String(value)
  .toLowerCase()
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .replace(/&/g, ' and ')
  .replace(/[^a-z0-9]+/g, ' ')
  .trim();

const compactSearchText = (value='') => normalizeSearchText(value).replace(/\s+/g, '');

const TEAM_ALIASES = {
  ANA: ['ANA', 'Anaheim', 'Ducks', 'Anaheim Ducks'],
  BOS: ['BOS', 'Boston', 'Bruins', 'Boston Bruins'],
  BUF: ['BUF', 'Buffalo', 'Sabres', 'Sabre', 'Buffalo Sabres'],
  CAR: ['CAR', 'Carolina', 'Hurricanes', 'Canes', 'Carolina Hurricanes'],
  CBJ: ['CBJ', 'Columbus', 'Blue Jackets', 'Jackets', 'Columbus Blue Jackets'],
  CGY: ['CGY', 'Calgary', 'Flames', 'Calgary Flames'],
  CHI: ['CHI', 'Chicago', 'Blackhawks', 'Black Hawks', 'Hawks', 'Chicago Blackhawks'],
  COL: ['COL', 'Colorado', 'Avalanche', 'Avs', 'Colorado Avalanche'],
  DAL: ['DAL', 'Dallas', 'Stars', 'Dallas Stars'],
  DET: ['DET', 'Detroit', 'Red Wings', 'Wings', 'Detroit Red Wings'],
  EDM: ['EDM', 'Edmonton', 'Oilers', 'Edmonton Oilers'],
  FLA: ['FLA', 'Florida', 'Panthers', 'Cats', 'Florida Panthers'],
  LAK: ['LAK', 'LA', 'Los Angeles', 'Kings', 'LA Kings', 'Los Angeles Kings'],
  MIN: ['MIN', 'Minnesota', 'Wild', 'Minnesota Wild'],
  MTL: ['MTL', 'Montreal', 'Montréal', 'Canadiens', 'Habs', 'Montreal Canadiens'],
  NJD: ['NJD', 'New Jersey', 'Devils', 'New Jersey Devils', 'Jersey'],
  NSH: ['NSH', 'Nashville', 'Predators', 'Preds', 'Nashville Predators'],
  NYI: ['NYI', 'New York Islanders', 'Islanders', 'Isles'],
  NYR: ['NYR', 'New York Rangers', 'Rangers', 'Blueshirts', 'Blue Shirts'],
  OTT: ['OTT', 'Ottawa', 'Senators', 'Sens', 'Ottawa Senators'],
  PHI: ['PHI', 'Philadelphia', 'Flyers', 'Philadelphia Flyers'],
  PIT: ['PIT', 'Pittsburgh', 'Penguins', 'Pens', 'Pittsburgh Penguins'],
  SEA: ['SEA', 'Seattle', 'Kraken', 'Seattle Kraken'],
  SJS: ['SJS', 'SJ', 'San Jose', 'Sharks', 'San Jose Sharks'],
  STL: ['STL', 'St Louis', 'St. Louis', 'Blues', 'St Louis Blues', 'St. Louis Blues'],
  TBL: ['TBL', 'Tampa', 'Tampa Bay', 'Lightning', 'Bolts', 'Tampa Bay Lightning'],
  TOR: ['TOR', 'Toronto', 'Leafs', 'Maple Leafs', 'Toronto Maple Leafs'],
  UTA: ['UTA', 'Utah', 'Mammoth', 'Utah Mammoth', 'Hockey Club', 'Utah Hockey Club', 'ARI', 'Arizona', 'Coyotes', 'Phoenix'],
  VAN: ['VAN', 'Vancouver', 'Canucks', 'Vancouver Canucks'],
  VGK: ['VGK', 'Vegas', 'Golden Knights', 'Knights', 'Vegas Golden Knights'],
  WPG: ['WPG', 'Winnipeg', 'Jets', 'Winnipeg Jets'],
  WSH: ['WSH', 'Washington', 'Capitals', 'Caps', 'Washington Capitals'],
};

const TEAM_SEARCH_TOKENS = Object.fromEntries(
  Object.entries(TEAM_ALIASES).map(([code, aliases]) => {
    const tokens = new Set([normalizeSearchText(code), compactSearchText(code)]);
    aliases.forEach(alias => {
      const normalized = normalizeSearchText(alias);
      const compact = compactSearchText(alias);
      if (normalized) tokens.add(normalized);
      if (compact) tokens.add(compact);
    });
    return [code, [...tokens]];
  })
);

const lineSearchBlob = (line) => {
  const teamTokens = TEAM_SEARCH_TOKENS[line.team] || [normalizeSearchText(line.team), compactSearchText(line.team)];
  return {
    normalized: [
      normalizeSearchText(line.players),
      normalizeSearchText(line.team),
      ...teamTokens,
    ].join(' | '),
    compact: [
      compactSearchText(line.players),
      compactSearchText(line.team),
      ...teamTokens.map(compactSearchText),
    ].join('|'),
  };
};

const playerSearchBlob = (player) => {
  const teamTokens = TEAM_SEARCH_TOKENS[player.team] || [normalizeSearchText(player.team), compactSearchText(player.team)];
  return {
    normalized: [
      normalizeSearchText(player.name),
      normalizeSearchText(player.team),
      ...teamTokens,
    ].join(' | '),
    compact: [
      compactSearchText(player.name),
      compactSearchText(player.team),
      ...teamTokens.map(compactSearchText),
    ].join('|'),
  };
};

function rankField(arr, field, hi=true) {
  const vals = arr.map(p=>p[field]).sort((a,b)=>a-b);
  return arr.map(p => {
    const idx = vals.indexOf(p[field]);
    const pct = idx/(vals.length-1)*100;
    return hi ? pct : 100-pct;
  });
}

// ── OLS regression ───────────────────────────────────────────────────────────
function ols(xs, ys) {
  const n = xs.length;
  if (n < 2) return {slope:0, intercept:0, r2:0, r:0};
  const mX = xs.reduce((a,b)=>a+b,0)/n;
  const mY = ys.reduce((a,b)=>a+b,0)/n;
  const ssXX = xs.reduce((s,x)=>s+(x-mX)**2, 0);
  const ssXY = xs.reduce((s,x,i)=>s+(x-mX)*(ys[i]-mY), 0);
  const ssYY = ys.reduce((s,y)=>s+(y-mY)**2, 0);
  const slope = ssXX > 0 ? ssXY/ssXX : 0;
  const intercept = mY - slope*mX;
  const r2 = (ssXX > 0 && ssYY > 0) ? ssXY**2/(ssXX*ssYY) : 0;
  const r  = (ssXX > 0 && ssYY > 0) ? ssXY/Math.sqrt(ssXX*ssYY) : 0;
  return {slope, intercept, r2, r};
}

const lerp = (a,b,t) => a+(b-a)*t;
function pctBg(pct) {
  const t = clamp(pct,0,100)/100;
  if (t<0.5) { const s=t*2; return `rgb(${Math.round(lerp(214,255,s))},${Math.round(lerp(40,255,s))},${Math.round(lerp(40,255,s))})`; }
  const s=(t-0.5)*2;
  return `rgb(${Math.round(lerp(255,29,s))},${Math.round(lerp(255,100,s))},${Math.round(lerp(255,220,s))})`;
}
function pctFg(pct) { return (pct<18||pct>82)?'#fff':'#222'; }
const fmt = (v,d=2) => v==null?'N/A':(v>0?'+':'')+Number(v).toFixed(d);
const overuseCount = CHARTED.filter(p=>p.flagged).length;
const avgDurability = CHARTED.length ? (CHARTED.reduce((s,p)=>s+p.drop,0)/CHARTED.length) : 0;
const avgOverallXgd = CHARTED.length ? (CHARTED.reduce((s,p)=>s+p.overall_xgd,0)/CHARTED.length) : 0;

// ── SHARED COMPONENTS ────────────────────────────────────────────────────────
const KPICard = ({label, value, sub, accent}) => (
  <div style={{padding:'20px 24px', background:'#fff', border:'1px solid #e8e8e8', borderRadius:6, borderTop: accent?`3px solid ${accent}`:'3px solid transparent'}}>
    <div style={{fontSize:11,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.07em',color:'#888',marginBottom:8}}>{label}</div>
    <div style={{fontSize:28,fontWeight:700,letterSpacing:'-0.02em',fontFamily:'IBM Plex Mono'}}>{value}</div>
    {sub && <div style={{fontSize:11,color:'#999',marginTop:4}}>{sub}</div>}
  </div>
);

const PctCell = ({pct, value, bold}) => {
  const p = Math.max(1, pct);
  return <td style={{background:pctBg(p), color:pctFg(p), fontFamily:'IBM Plex Mono', fontSize:12, fontWeight:bold?600:400}}>{value}</td>;
};

const PctBadge = ({pct, label}) => {
  const p = Math.max(1, pct);
  return (
    <div style={{textAlign:'center'}}>
      <div style={{display:'inline-block', background:pctBg(p), color:pctFg(p), borderRadius:4, padding:'4px 10px', fontFamily:'IBM Plex Mono', fontSize:13, fontWeight:600, minWidth:52}}>{p}<span style={{fontSize:10,fontWeight:400}}>th</span></div>
      <div style={{fontSize:11,color:'#888',marginTop:4,fontWeight:500}}>{label}</div>
    </div>
  );
};

const teamLogoUrl = (code) => `https://assets.nhle.com/logos/nhl/svg/${code}_light.svg`;
const SEASON = window.__DATA__.current_season;
const headshotUrl = (id, team) => `https://assets.nhle.com/mugs/nhl/${SEASON}/${team}/${id}.png`;

const PlayerHeadshot = ({id, name, team, size=80}) => {
  const [failed, setFailed] = React.useState(false);
  if (failed) {
    return <div style={{width:size,height:size,borderRadius:'50%',background:'#e8e8e8',border:'2px solid #ddd',
      display:'flex',alignItems:'center',justifyContent:'center',color:'#bbb',fontSize:size*0.4,flexShrink:0}}>?</div>;
  }
  return <img src={headshotUrl(id, team)} alt={name} width={size} height={size}
    onError={()=>setFailed(true)}
    style={{borderRadius:'50%',objectFit:'cover',background:'#f0f0f0',border:'2px solid #e8e8e8',flexShrink:0,display:'block'}}/>;
};

const TeamLogo = ({code, size=28}) => (
  <img src={teamLogoUrl(code)} alt={code} width={size} height={size}
    style={{objectFit:'contain',flexShrink:0}} onError={e=>e.currentTarget.style.display='none'}/>
);

const InlinePlayerList = ({parts = [], onSelect}) => {
  if (!parts.length) return null;
  return (
    <>
      {parts.map((part, idx) => (
        <React.Fragment key={part.id || `${part.name}-${idx}`}>
          {idx > 0 && <span style={{color:'#9ca3af'}}> / </span>}
          <button
            onClick={() => onSelect?.(PLAYERS.find(p => p.id === part.id) || null)}
            disabled={!PLAYERS.some(p => p.id === part.id)}
            style={{
              color:'#1d4ed8',
              fontWeight:700,
              fontSize:'inherit',
              lineHeight:'inherit',
              textAlign:'left',
              background:'none',
              border:'none',
              padding:0,
              cursor:PLAYERS.some(p => p.id === part.id) ? 'pointer' : 'default',
            }}
          >
            {part.name}
          </button>
        </React.Fragment>
      ))}
    </>
  );
};

const Tag = ({children, color='#666', bg='#f0f0f0'}) => (
  <span style={{fontSize:11,fontWeight:600,padding:'2px 7px',background:bg,color,borderRadius:3,letterSpacing:'0.04em'}}>{children}</span>
);

const InfoTooltip = ({text}) => {
  const [open, setOpen] = React.useState(false);
  return (
    <span
      style={{position:'relative',display:'inline-flex',alignItems:'center',marginLeft:8,verticalAlign:'middle'}}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label="Show methodology"
        style={{
          width:18,height:18,borderRadius:'50%',border:'1px solid #cbd5e1',background:'#fff',color:'#64748b',
          fontSize:11,fontWeight:700,lineHeight:1,cursor:'help',display:'inline-flex',alignItems:'center',
          justifyContent:'center',padding:0
        }}
      >
        i
      </button>
      {open && (
        <div style={{
          position:'absolute',top:24,left:0,width:320,background:'#111827',color:'#e5e7eb',borderRadius:6,
          padding:'10px 12px',fontSize:11,lineHeight:1.5,boxShadow:'0 10px 24px rgba(0,0,0,0.2)',zIndex:20
        }}>
          {text}
        </div>
      )}
    </span>
  );
};

const SectionHeader = ({title, sub}) => (
  <div style={{marginBottom:16}}>
    <div style={{fontSize:16,fontWeight:700,letterSpacing:'-0.01em'}}>{title}</div>
    {sub && <div style={{fontSize:12,color:'#888',marginTop:3,fontStyle:'italic'}}>{sub}</div>}
  </div>
);

// ── NAV ──────────────────────────────────────────────────────────────────────
const NAV_PAGES = ['overview','players','lines','team'];
const NAV_LABELS = {overview:'Overview',players:'Players',lines:'Lines',team:'Team Report'};
const APP_PAGES = new Set([...NAV_PAGES, 'profile']);

function seasonLabel(s) {
  // "20232024" -> "2023-24"
  if (s && s.length === 8) return s.slice(0,4) + '–' + s.slice(6);
  return s;
}

function parseRouteState() {
  const params = new URLSearchParams(window.location.search);
  const rawPlayerId = params.get('player');
  return {
    season: params.get('season') || window.__DATA__.current_season,
    page: params.get('page') || 'overview',
    playerId: rawPlayerId && /^\d+$/.test(rawPlayerId) ? Number(rawPlayerId) : null,
    team: (params.get('team') || '').toUpperCase(),
  };
}

function normalizeRouteState(raw) {
  const season = raw.season || window.__DATA__.current_season;
  let page = APP_PAGES.has(raw.page) ? raw.page : 'overview';
  const playerId = PLAYERS.some(p => p.id === raw.playerId) ? raw.playerId : null;
  const team = TEAMS.includes(raw.team) ? raw.team : (TEAMS[0] || '');

  if (page === 'profile' && !playerId) page = 'players';

  return {season, page, playerId, team};
}

function buildRouteUrl(route) {
  const params = new URLSearchParams();
  params.set('season', route.season || window.__DATA__.current_season);
  if (route.page && route.page !== 'overview') params.set('page', route.page);
  if (route.playerId && route.page === 'profile') params.set('player', String(route.playerId));
  if (route.team) params.set('team', route.team);
  return `/?${params.toString()}`;
}

const TopNav = ({page,setPage,season,onSeasonChange,breadcrumb}) => (
  <div style={{background:'#fff',borderBottom:'1px solid #e5e5e5',position:'sticky',top:0,zIndex:100}}>
    <div style={{maxWidth:1200,margin:'0 auto',padding:'0 24px',display:'flex',alignItems:'center',gap:0,height:52}}>
      <div style={{fontWeight:700,fontSize:14,letterSpacing:'-0.01em',marginRight:32,color:'#111',whiteSpace:'nowrap'}}>
        <span style={{color:'#1d4ed8'}}>⬡</span> Shift Fade
      </div>
      {NAV_PAGES.map(p=>(
        <button key={p} onClick={()=>setPage(p)} style={{
          padding:'0 16px',height:52,fontSize:13,fontWeight:600,
          color:page===p?'#1d4ed8':'#555',
          borderBottom:page===p?'2px solid #1d4ed8':'2px solid transparent',
          transition:'all .15s',letterSpacing:'0.01em'
        }}>{NAV_LABELS[p]}</button>
      ))}
      <div style={{flex:1}}/>
      <select value={season} onChange={e=>onSeasonChange(e.target.value)} style={{fontSize:12}}>
        {SEASONS.map(s=><option key={s} value={s}>{seasonLabel(s)}</option>)}
      </select>
    </div>
    {breadcrumb && (
      <div style={{maxWidth:1200,margin:'0 auto',padding:'6px 24px',background:'#fafaf9',borderTop:'1px solid #f0f0f0',fontSize:12,color:'#888',display:'flex',alignItems:'center',gap:6}}>
        {breadcrumb.map((b,i)=>(
          <React.Fragment key={i}>
            {i>0&&<span style={{color:'#ccc'}}>/</span>}
            {b.onClick
              ? <button onClick={b.onClick} style={{color:'#1d4ed8',fontWeight:500,fontSize:12}}>{b.label}</button>
              : <span style={{color:i===breadcrumb.length-1?'#333':'#888',fontWeight:i===breadcrumb.length-1?600:400}}>{b.label}</span>}
          </React.Fragment>
        ))}
      </div>
    )}
  </div>
);

// ── PLAYER SHIFT TREND CHART ─────────────────────────────────────────────────
// Show change from fresh-shift performance so the viewer can see actual decay,
// not just absolute quality. TOI bars make sample strength obvious by bucket.
const PlayerShiftTrendChart = ({
  buckets,
  deltaValues,
  rawValues,
  toiSec,
  leagueBuckets,
  leagueDeltaValues,
  width = 680,
  height = 280,
}) => {
  const m = {top: 18, right: 20, bottom: 62, left: 56};
  const W = width - m.left - m.right;
  const H = height - m.top - m.bottom;
  const barBand = 46;
  const lineH = H - barBand;

  const pts = (buckets || []).map((bucket, i) => ({
    bucket,
    delta: deltaValues?.[i],
    raw: rawValues?.[i],
    toi: toiSec?.[i] || 0,
  })).filter(d => d.delta != null && isFinite(d.delta));

  const lg = (leagueBuckets || []).map((bucket, i) => ({
    bucket,
    delta: leagueDeltaValues?.[i],
  })).filter(d => d.delta != null && isFinite(d.delta));
  const leagueMap = new Map(lg.map(d => [d.bucket, d.delta]));
  const displayPts = pts.sort((a,b) => a.bucket - b.bucket);
  const tickBuckets = [];
  const maxBucket = Math.max(...displayPts.map(d => d.bucket), 0);
  for (let t = 0; t <= maxBucket; t += 10) tickBuckets.push(t);
  const displayLeague = tickBuckets
    .map(bucket => ({bucket, delta: leagueMap.get(bucket)}))
    .filter(d => d.delta != null && isFinite(d.delta));
  const count = Math.max(tickBuckets.length, 2);

  const allY = [...displayPts.map(d => d.delta), ...displayLeague.map(d => d.delta), 0].filter(v => isFinite(v));
  const rawMin = allY.length ? Math.min(...allY) : -1;
  const rawMax = allY.length ? Math.max(...allY) : 1;
  const span = Math.max(rawMax - rawMin, 1.5);
  const pad = span * 0.22;
  const targetTickCount = 7;
  const roughStep = (span + pad * 2) / targetTickCount;
  const tickSteps = [0.5, 1, 2, 2.5, 5];
  const tickStep = tickSteps.find(step => roughStep <= step) || 5;
  const yMin = Math.floor((rawMin - pad) / tickStep) * tickStep;
  const yMax = Math.ceil((rawMax + pad) / tickStep) * tickStep;
  const yTicks = [];
  for (let v = yMin; v <= yMax + 0.001; v += tickStep) yTicks.push(Number(v.toFixed(2)));

  const maxToi = Math.max(...displayPts.map(d => d.toi), 1);
  const slotW = count > 1 ? W / (count - 1) : W;
  const xS = i => count === 1 ? W / 2 : i * slotW;
  const yS = v => lineH - ((v - yMin) / (yMax - yMin)) * lineH;
  const barH = v => (v / maxToi) * (barBand - 10);
  const barW = Math.max(18, Math.min(34, slotW * 0.45));
  const tickIndex = new Map(tickBuckets.map((bucket, i) => [bucket, i]));

  const playerPath = displayPts.map((d, i) => `${i ? 'L' : 'M'} ${xS(tickIndex.get(d.bucket) ?? i)} ${yS(d.delta)}`).join(' ');
  const leaguePath = displayLeague.map((d, i) => `${i ? 'L' : 'M'} ${xS(tickIndex.get(d.bucket) ?? i)} ${yS(d.delta)}`).join(' ');
  const avgShiftIndex = tickBuckets.findIndex(bucket => bucket >= 40);
  const avgShiftX = avgShiftIndex >= 0 ? xS(avgShiftIndex) : null;

  return (
    <div>
      <svg width={width} height={height} style={{display:'block', overflow:'visible'}}>
        <g transform={`translate(${m.left},${m.top})`}>
          {yTicks.map(v => (
            <g key={v}>
              <line x1={0} y1={yS(v)} x2={W} y2={yS(v)} stroke={v === 0 ? '#d1d5db' : '#f1f5f9'} strokeWidth={v === 0 ? 1.5 : 1}/>
              <text x={-8} y={yS(v)+4} textAnchor="end" fontSize="11" fill={v === 0 ? '#64748b' : '#94a3b8'} fontFamily="IBM Plex Mono">
                {v > 0 ? `+${Number(v.toFixed(2)).toString()}` : Number(v.toFixed(2)).toString()}
              </text>
            </g>
          ))}

          {avgShiftX != null && (
            <>
              <line x1={avgShiftX} y1={0} x2={avgShiftX} y2={lineH} stroke="#e5e7eb" strokeDasharray="3 3" strokeWidth="1"/>
              <text x={avgShiftX+4} y={12} fontSize="9" fill="#9ca3af" fontFamily="IBM Plex Mono">avg shift</text>
            </>
          )}

          {displayPts.map((d) => (
            <g key={`bar-${d.bucket}`}>
              <rect
                x={xS(tickIndex.get(d.bucket) ?? 0) - barW/2}
                y={H - barH(d.toi)}
                width={barW}
                height={barH(d.toi)}
                rx={3}
                fill="#dbeafe"
                stroke="#bfdbfe"
              />
            </g>
          ))}

          {leaguePath && (
            <path d={leaguePath} stroke="#0f766e" strokeWidth="2" strokeDasharray="6 4" fill="none"/>
          )}
          {playerPath && (
            <path d={playerPath} stroke="#1d4ed8" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          )}

          {tickBuckets.map((bucket, i) => (
            <text key={`tick-${bucket}`} x={xS(i)} y={H + 18} textAnchor="middle" fontSize="11" fill="#94a3b8" fontFamily="IBM Plex Mono">
              {bucket}
            </text>
          ))}

          {displayPts.map((d) => (
            <g key={`pt-${d.bucket}`}>
              <circle cx={xS(tickIndex.get(d.bucket) ?? 0)} cy={yS(d.delta)} r="4.5" fill="#1d4ed8" stroke="#fff" strokeWidth="1.5"/>
              <text x={xS(tickIndex.get(d.bucket) ?? 0)} y={H + 33} textAnchor="middle" fontSize="10" fill="#cbd5e1" fontFamily="IBM Plex Mono">
                {Math.round(d.toi / 60)}m
              </text>
            </g>
          ))}

          <text x={-38} y={lineH/2} textAnchor="middle" fontSize="12" fill="#64748b" transform={`rotate(-90,-38,${lineH/2})`}>
            Change in xGD/60 vs fresh shift
          </text>
          <text x={W/2} y={H + 50} textAnchor="middle" fontSize="12" fill="#64748b">
            Shift age bucket (seconds) • blue bars show TOI in each bucket
          </text>
        </g>
      </svg>

      <div style={{display:'flex',gap:16,alignItems:'center',paddingLeft:m.left,marginTop:4,fontSize:10,color:'#94a3b8',flexWrap:'wrap'}}>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="#1d4ed8" strokeWidth="3"/></svg>
          player change from 0–10s bucket
        </span>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke="#0f766e" strokeWidth="2" strokeDasharray="6 4"/></svg>
          league-average change
        </span>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <svg width="20" height="10"><rect x="3" y="1" width="14" height="8" rx="2" fill="#dbeafe" stroke="#bfdbfe"/></svg>
          player TOI by bucket
        </span>
      </div>
    </div>
  );
};

// ── HISTOGRAM (durability distribution) ──────────────────────────────────────
const Histogram = ({players, width=360, height=190}) => {
  const m = {top:12,right:12,bottom:50,left:48};
  const W = width-m.left-m.right, H = height-m.top-m.bottom;
  const vals = players.map(p=>p.drop);
  const min = Math.min(...vals, -4);
  const max = Math.max(...vals, 2);
  const avg = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0;
  const bins=14;
  const step=(max-min)/bins;
  const counts = Array(bins).fill(0);
  vals.forEach(v=>{ const i=Math.min(bins-1,Math.floor((v-min)/step)); if(i>=0) counts[i]++; });
  const maxC=Math.max(...counts);
  const xS=i=>(i/bins)*W, yS=c=>H-(c/maxC)*H;
  const barW=W/bins-1;
  const pcts = counts.map((_,i)=>((min+(i+0.5)*step-min)/(max-min)*100));
  return (
    <svg width={width} height={height}>
      <g transform={`translate(${m.left},${m.top})`}>
        <line x1={0} y1={H} x2={W} y2={H} stroke="#ddd"/>
        {counts.map((c,i)=>(
          <rect key={i} x={xS(i)+0.5} y={yS(c)} width={barW} height={H-yS(c)} fill={pctBg(pcts[i])} opacity="0.9"/>
        ))}
        {/* average line */}
        <line x1={((avg-min)/(max-min))*W} y1={0} x2={((avg-min)/(max-min))*W} y2={H+4} stroke="#c00" strokeWidth="1.5" strokeDasharray="4 2"/>
        <text x={((avg-min)/(max-min))*W} y={H+16} textAnchor="middle" fontSize="10" fill="#c00" fontFamily="IBM Plex Mono">avg</text>
        {[min, (min+max)/2, max].map((v,i)=>(
          <text key={i} x={((v-min)/(max-min))*W} y={H+28} textAnchor="middle" fontSize="10" fill="#999" fontFamily="IBM Plex Mono">{v.toFixed(1)}</text>
        ))}
        {[0,Math.round(maxC/2),maxC].map((c,i)=>(
          <text key={i} x={-6} y={yS(c)+4} textAnchor="end" fontSize="10" fill="#aaa" fontFamily="IBM Plex Mono">{c}</text>
        ))}
        {/* axis labels */}
        <text x={W/2} y={H+44} textAnchor="middle" fontSize="11" fill="#666" fontFamily="IBM Plex Sans, sans-serif">Durability (late xGD/60 minus early xGD/60)</text>
        <text x={-36} y={H/2} textAnchor="middle" fontSize="11" fill="#666" fontFamily="IBM Plex Sans, sans-serif" transform={`rotate(-90,-36,${H/2})`}>Players</text>
      </g>
    </svg>
  );
};

// ── SCATTER (Overall xGD vs Durability) ──────────────────────────────────────
const ScatterPlot = ({players, onSelect, reg, width=360, height=190}) => {
  const [tip, setTip] = React.useState(null);
  const m = {top:12,right:12,bottom:50,left:52};
  const W=width-m.left-m.right, H=height-m.top-m.bottom;
  const xs=players.map(p=>p.drop), ys=players.map(p=>p.overall_xgd);
  const xMin=Math.min(...xs)-0.2, xMax=Math.max(...xs)+0.2;
  const yMin=Math.min(...ys)-0.2, yMax=Math.max(...ys)+0.2;
  const xS=v=>((v-xMin)/(xMax-xMin))*W;
  const yS=v=>H-((v-yMin)/(yMax-yMin))*H;
  const rl_y1 = reg.slope*xMin + reg.intercept;
  const rl_y2 = reg.slope*xMax + reg.intercept;

  const handleMove = (e, p) => {
    const rect = e.currentTarget.closest('svg').getBoundingClientRect();
    setTip({p, x: e.clientX - rect.left, y: e.clientY - rect.top});
  };

  return (
    <div style={{position:'relative',display:'inline-block'}}>
      <svg width={width} height={height} style={{cursor:'crosshair'}} onMouseLeave={()=>setTip(null)}>
        <g transform={`translate(${m.left},${m.top})`}>
          <line x1={0} y1={yS(0)} x2={W} y2={yS(0)} stroke="#eee"/>
          <line x1={xS(0)} y1={0} x2={xS(0)} y2={H} stroke="#ddd" strokeDasharray="4 2"/>
          <line x1={xS(xMin)} y1={yS(rl_y1)} x2={xS(xMax)} y2={yS(rl_y2)}
            stroke="#8b5cf6" strokeWidth="1.8" strokeDasharray="6 3" opacity="0.85"/>
          {players.map(p=>(
            <circle key={p.id} cx={xS(p.drop)} cy={yS(p.overall_xgd)} r="5"
              fill={p.flagged?'#dc2626':'#1d4ed8'} opacity="0.7" stroke="#fff" strokeWidth="1"
              onClick={()=>onSelect(p)} style={{cursor:'pointer'}}
              onMouseEnter={e=>handleMove(e,p)}
              onMouseMove={e=>handleMove(e,p)}
              onMouseLeave={()=>setTip(null)}/>
          ))}
          {[xMin+(xMax-xMin)*0.25, xMin+(xMax-xMin)*0.5, xMin+(xMax-xMin)*0.75].map((v,i)=>(
            <text key={i} x={xS(v)} y={H+20} textAnchor="middle" fontSize="10" fill="#aaa" fontFamily="IBM Plex Mono">{v.toFixed(1)}</text>
          ))}
          <text x={W/2} y={H+44} textAnchor="middle" fontSize="11" fill="#666" fontFamily="IBM Plex Sans, sans-serif">Durability (late xGD/60 minus early xGD/60)</text>
          <text x={-40} y={H/2} textAnchor="middle" fontSize="11" fill="#666" fontFamily="IBM Plex Sans, sans-serif" transform={`rotate(-90,-40,${H/2})`}>Overall xGD/60</text>
        </g>
      </svg>
      {/* Legend row outside SVG so it never overlaps data */}
      <div style={{display:'flex',alignItems:'center',gap:16,marginTop:6,paddingLeft:m.left,fontSize:11,color:'#666',fontFamily:'IBM Plex Sans, sans-serif'}}>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="#dc2626" opacity="0.8"/></svg> Flagged
        </span>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="#1d4ed8" opacity="0.7"/></svg> Normal
        </span>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <svg width="16" height="10"><line x1="0" y1="5" x2="16" y2="5" stroke="#8b5cf6" strokeWidth="1.8" strokeDasharray="4 2"/></svg> OLS trend
        </span>
        <span style={{marginLeft:'auto',fontFamily:'IBM Plex Mono',fontWeight:600,color:'#8b5cf6',paddingRight:m.right}}>
          r={reg.r>=0?'+':''}{reg.r.toFixed(3)} · R²={reg.r2.toFixed(3)}
        </span>
      </div>
      {tip && (
        <div style={{
          position:'absolute',
          left: tip.x + 12,
          top:  tip.y - 10,
          background:'#1a1a2e',
          color:'#fff',
          borderRadius:6,
          padding:'8px 11px',
          fontSize:12,
          fontFamily:'IBM Plex Sans, sans-serif',
          pointerEvents:'none',
          whiteSpace:'nowrap',
          boxShadow:'0 4px 16px rgba(0,0,0,0.25)',
          zIndex:999,
        }}>
          <div style={{fontWeight:600,marginBottom:3}}>{tip.p.name} <span style={{opacity:0.6,fontWeight:400}}>({tip.p.team})</span></div>
          <div style={{fontFamily:'IBM Plex Mono',fontSize:11,color:'#a5b4fc'}}>
            Overall xGD/60: <span style={{color:'#fff'}}>{tip.p.overall_xgd.toFixed(2)}</span>
          </div>
          <div style={{fontFamily:'IBM Plex Mono',fontSize:11,color:'#a5b4fc'}}>
            Durability: <span style={{color:'#fff'}}>{tip.p.drop.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

// ── OVERVIEW PAGE ─────────────────────────────────────────────────────────────
const OverviewPage = ({openPlayer}) => {
  const sorted_decay = [...CHARTED].sort((a,b)=>a.drop-b.drop).slice(0,12);
  const sorted_overall  = [...CHARTED].sort((a,b)=>b.overall_xgd-a.overall_xgd).slice(0,12);

  return (
    <div style={{maxWidth:1200,margin:'0 auto',padding:'32px 24px'}}>
      <div style={{marginBottom:28}}>
        <h1 style={{fontSize:26,fontWeight:700,letterSpacing:'-0.02em',marginBottom:4}}>League Overview</h1>
        <p style={{fontSize:13,color:'#888'}}>A beginner-friendly look at which players help their team create better chances at 5v5, and whether that impact holds up as a shift gets longer.</p>
      </div>
      <div style={{background:'#f8f9ff',border:'1px solid #dde4f5',borderRadius:6,padding:'18px 20px',marginBottom:36}}>
        <div style={{fontSize:12,fontWeight:700,color:'#1d4ed8',textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:10}}>Start Here</div>
        <div style={{display:'grid',gridTemplateColumns:'1.2fr 1fr 1.3fr',gap:18,fontSize:12,color:'#374151',lineHeight:1.7}}>
          <div>
            <span style={{fontWeight:700,color:'#111'}}>What xGD/60 means:</span> It is a simple way to estimate whether a player's team is getting the better scoring chances while he is on the ice at 5v5.
            Above zero is good. Below zero is bad.
          </div>
          <div>
            <span style={{fontWeight:700,color:'#111'}}>What durability means:</span> We compare a player's early-shift results to his late-shift results.
            If the number is very negative, he fades as the shift gets older. If it is near zero or positive, he holds up well.
          </div>
          <div>
            <span style={{fontWeight:700,color:'#111'}}>How we calculate it:</span> We add up expected goals for and against during tracked 5v5 shift segments, subtract against from for, divide by ice time, and scale to 60 minutes.
            Early, middle, and late numbers use that same calculation inside different parts of the shift.
          </div>
        </div>
      </div>

      <div style={{display:'flex',flexDirection:'column',gap:20,marginBottom:36}}>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px 24px 16px'}}>
          <SectionHeader title="How Much Players Fade Over A Shift" sub="This shows how common each durability result is across the league. Most players get a little worse as their shift goes on."/>
          <div style={{display:'flex',gap:16,alignItems:'flex-start',marginTop:4,marginBottom:10}}>
            <div style={{fontSize:11,color:'#888',lineHeight:1.5,maxWidth:480}}>
              <span style={{fontWeight:600,color:'#555'}}>How to read it:</span> Each bar counts how many players fall in that durability range.<br/>
              <span style={{fontWeight:600,color:'#555'}}>Left side:</span> Players whose results get worse later in the shift.<br/>
              <span style={{fontWeight:600,color:'#555'}}>Right side:</span> Players who stay level or improve late in the shift.<br/>
              <span style={{fontWeight:600,color:'#555'}}>Color:</span> Red = more fade, blue = better durability.
            </div>
          </div>
          <Histogram players={CHARTED} width={1100} height={220}/>
        </div>

        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px 24px 16px'}}>
          <SectionHeader title="Who Is Good, And Who Stays Good Late?" sub="Each dot is one player. Higher dots had better overall 5v5 results. Dots farther right held up better as the shift got longer. Click any dot to open that player."/>
          <div style={{fontSize:11,color:'#888',lineHeight:1.5,maxWidth:640,marginTop:4,marginBottom:10}}>
            <span style={{fontWeight:600,color:'#555'}}>Up:</span> Better overall 5v5 impact.<br/>
            <span style={{fontWeight:600,color:'#555'}}>Right:</span> Better late-shift durability.<br/>
            <span style={{fontWeight:600,color:'#555'}}>Purple line:</span> The general league trend.<br/>
            <span style={{color:'#dc2626',fontWeight:600}}>Red dots</span> = players whose results drop off especially hard late in shifts.
          </div>
          {(() => {
            const xs = CHARTED.map(p=>p.drop);
            const ys = CHARTED.map(p=>p.overall_xgd);
            const reg = ols(xs, ys);
            return <>
              <ScatterPlot players={CHARTED} onSelect={openPlayer} reg={reg} width={1100} height={260}/>
              <div style={{marginTop:10,padding:'8px 12px',background:'#faf9ff',border:'1px solid #e9d5ff',borderRadius:4,fontSize:11,color:'#6b21a8',display:'flex',alignItems:'center',gap:10,flexWrap:'wrap'}}>
                <span style={{fontFamily:'IBM Plex Mono',fontWeight:600}}>r = {reg.r>=0?'+':''}{reg.r.toFixed(3)}</span>
                <span style={{color:'#c4b5fd'}}>·</span>
                <span style={{fontFamily:'IBM Plex Mono',fontWeight:600}}>R² = {reg.r2.toFixed(3)}</span>
                <span style={{color:'#c4b5fd'}}>·</span>
                <span>{reg.r > 0 ? 'In this sample, better players also tend to hold up a bit better as the shift gets longer.' : 'In this sample, being a strong player overall does not automatically mean holding up better late in a shift.'}</span>
              </div>
            </>;
          })()}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:24}}>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
          <SectionHeader title="Players Whose Results Fall Off The Most" sub="These players have the biggest late-shift drop after clearing a meaningful ice-time minimum. Click a name to view the full profile."/>
          <div style={{fontSize:11,color:'#888',lineHeight:1.5,maxWidth:560,marginTop:4,marginBottom:10}}>
            If you are new to this page, start with the <span style={{fontWeight:600,color:'#555'}}>Durability</span> column.
            More negative numbers mean the player's results got much worse later in the shift.
          </div>
          <table>
            <thead><tr>
              <th style={{textAlign:'left'}}>Player</th><th>Team</th><th>Overall</th><th>Durability</th><th>TOI</th><th>Late</th>
            </tr></thead>
            <tbody>{sorted_decay.map(p=>(
              <tr key={p.id}>
                <td><button onClick={()=>openPlayer(p)} style={{color:'#1d4ed8',fontWeight:600,fontSize:13,textAlign:'left'}}>{p.name}</button></td>
                <td style={{textAlign:'center'}}><Tag>{p.team}</Tag></td>
                <PctCell pct={p.pct_overall} value={fmt(p.overall_xgd,2)}/>
                <PctCell pct={p.pct_durability} value={fmt(p.drop,2)}/>
                <td style={{fontFamily:'IBM Plex Mono',fontSize:12}}>{p.toi_min}</td>
                <PctCell pct={p.pct_late} value={fmt(p.late,2)}/>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
          <SectionHeader title="Best Overall 5v5 Results" sub="These players drive the strongest net chance quality overall, regardless of where in the shift it happens."/>
          <div style={{fontSize:11,color:'#888',lineHeight:1.5,maxWidth:560,marginTop:4,marginBottom:10}}>
            Start with the <span style={{fontWeight:600,color:'#555'}}>Overall</span> column here.
            Bigger positive numbers mean the player's team generated better chances than it allowed while he was on the ice.
          </div>
          <table>
            <thead><tr>
              <th style={{textAlign:'left'}}>Player</th><th>Pos</th><th>Overall</th><th>Early</th><th>Late</th><th>TOI</th>
            </tr></thead>
            <tbody>{sorted_overall.map(p=>(
              <tr key={p.id}>
                <td><button onClick={()=>openPlayer(p)} style={{color:'#1d4ed8',fontWeight:600,fontSize:13}}>{p.name}</button></td>
                <td style={{textAlign:'center',color:'#888',fontSize:12}}>{p.pos}</td>
                <PctCell pct={p.pct_overall} value={fmt(p.overall_xgd,2)}/>
                <PctCell pct={p.pct_early} value={fmt(p.early)}/>
                <PctCell pct={p.pct_late} value={fmt(p.late)}/>
                <td style={{fontFamily:'IBM Plex Mono',fontSize:12}}>{p.toi_min}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// ── PLAYERS PAGE ──────────────────────────────────────────────────────────────
const PlayersPage = ({openPlayer}) => {
  const [search, setSearch] = useState('');
  const query = normalizeSearchText(search);
  const compactQuery = compactSearchText(search);

  const searchResults = query
    ? [...PLAYERS]
        .filter(p => {
          const blob = playerSearchBlob(p);
          return blob.normalized.includes(query) || (compactQuery && blob.compact.includes(compactQuery));
        })
        .sort((a,b) => b.toi_min - a.toi_min)
    : null;

  const boards = [
    {
      title: 'Best Early Impact',
      desc: 'Highest xGD/60 in the first 30 seconds on ice. Who dominates immediately off a fresh change.',
      players: [...CHARTED].sort((a,b)=>b.early-a.early).slice(0,10),
      stat: p => fmt(p.early,2), label: 'Early', pct: p => p.pct_early,
    },
    {
      title: 'Best Middle Impact',
      desc: 'Highest xGD/60 from 30 to 45 seconds on ice. Who stays dangerous through the middle of a shift.',
      players: [...CHARTED].sort((a,b)=>b.mid-a.mid).slice(0,10),
      stat: p => fmt(p.mid,2), label: 'Mid', pct: p => p.pct_mid,
    },
    {
      title: 'Best Late Impact',
      desc: 'Highest xGD/60 after 45+ seconds on ice. Who holds up as their shift stales, a rare and valuable trait.',
      players: [...CHARTED].sort((a,b)=>b.late-a.late).slice(0,10),
      stat: p => fmt(p.late,2), label: 'Late', pct: p => p.pct_late,
    },
    {
      title: 'Top Overall xGD/60',
      desc: 'Best observed 5v5 impact across all qualifying shift segments.',
      players: [...CHARTED].sort((a,b)=>b.overall_xgd-a.overall_xgd).slice(0,10),
      stat: p => fmt(p.overall_xgd,2), label: 'xGD', pct: p => p.pct_overall,
    },
    {
      title: 'Most Durable',
      desc: 'Smallest early→late drop (late xGD − early xGD). Values near 0 or positive mean the player sustains impact across a long shift. Min 30 min TOI.',
      players: [...CHARTED].filter(p=>p.toi_min>=30).sort((a,b)=>b.drop-a.drop).slice(0,10),
      stat: p => fmt(p.drop,2), label: 'Drop', pct: p => p.pct_drop,
    },
    {
      title: 'Least Durable',
      desc: 'Most negative durability scores. These players lose observed impact fastest as their shift ages. Min 30 min TOI.',
      players: [...CHARTED].filter(p=>p.toi_min>=30).sort((a,b)=>a.drop-b.drop).slice(0,10),
      stat: p => fmt(p.drop,2), label: 'Drop', pct: p => p.pct_durability,
    },
  ];

  return (
    <div style={{maxWidth:1200,margin:'0 auto',padding:'32px 24px'}}>
      <div style={{marginBottom:24}}>
        <h1 style={{fontSize:26,fontWeight:700,letterSpacing:'-0.02em',marginBottom:4}}>Player Stat Leaders</h1>
        <p style={{fontSize:13,color:'#888'}}>5v5 shift-age analysis · Minimum {MIN_QUALIFYING_SHIFTS} shifts longer than {MIN_QUALIFYING_SHIFT_SECONDS}s · Click any player to open their full profile</p>
      </div>

      {/* Search bar */}
      <div style={{position:'relative',marginBottom:24}}>
        <span style={{position:'absolute',left:14,top:'50%',transform:'translateY(-50%)',color:'#bbb',fontSize:17,pointerEvents:'none'}}>⌕</span>
        <input
          type="text"
          placeholder="Search by player name or team…"
          value={search}
          onChange={e=>setSearch(e.target.value)}
          style={{width:'100%',padding:'11px 40px',fontSize:14,border:'1px solid #d1d5db',borderRadius:6,outline:'none',
            fontFamily:'IBM Plex Sans, sans-serif',boxSizing:'border-box',background:'#fff'}}
        />
        {search && (
          <button onClick={()=>setSearch('')}
            style={{position:'absolute',right:12,top:'50%',transform:'translateY(-50%)',background:'none',border:'none',color:'#bbb',fontSize:16,cursor:'pointer',lineHeight:1}}>
            ✕
          </button>
        )}
      </div>

      {/* Search results */}
      {searchResults && (
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'16px 20px',marginBottom:28}}>
          <div style={{fontSize:12,fontWeight:600,color:'#888',textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:12}}>
            {searchResults.length} player{searchResults.length!==1?'s':''} matching "{search}"
          </div>
          {searchResults.length===0
            ? <div style={{color:'#aaa',fontSize:13,padding:'12px 0'}}>No players found.</div>
            : <table>
                <thead><tr>
                  <th style={{textAlign:'left'}}>Player</th>
                  <th>Team</th><th>Pos</th>
                  <th title="Observed impact across all qualifying shift segments">Overall xGD</th>
                  <th title="xGD/60 in first 30s of shift">Early xGD</th>
                  <th title="xGD/60 after 45s on ice">Late xGD</th>
                  <th title="Late minus Early; negative means fades">Drop</th>
                  <th>TOI (min)</th><th>Sample</th><th>Flag</th>
                </tr></thead>
                <tbody>{searchResults.map(p=>(
                  <tr key={p.id} onClick={()=>openPlayer(p)} style={{cursor:'pointer'}}>
                    <td style={{fontWeight:600,color:'#1d4ed8',whiteSpace:'nowrap'}}>{p.name}</td>
                    <td>{p.team}</td><td>{p.pos}</td>
                    <PctCell pct={p.pct_overall}  value={fmt(p.overall_xgd,2)}/>
                    <PctCell pct={p.pct_early} value={fmt(p.early,2)}/>
                    <PctCell pct={p.pct_late}  value={fmt(p.late,2)}/>
                    <PctCell pct={p.pct_drop}  value={fmt(p.drop,2)}/>
                    <td style={{fontFamily:'IBM Plex Mono',fontSize:12}}>{p.toi_min}</td>
                    <td style={{textAlign:'center'}}>
                      {p.eligible_for_graphs
                        ? <span style={{color:'#166534',fontWeight:600,fontSize:12}}>Qualified</span>
                        : <span style={{color:'#a16207',fontWeight:600,fontSize:12}}>Low sample</span>}
                    </td>
                    <td style={{textAlign:'center'}}>{p.flagged?<span style={{color:'#dc2626',fontWeight:700}}>⚠</span>:<span style={{color:'#ccc'}}>N/A</span>}</td>
                  </tr>
                ))}</tbody>
              </table>
          }
        </div>
      )}

      {/* Stat key */}
      {!searchResults && (
        <div style={{background:'#f8f9ff',border:'1px solid #dde4f5',borderRadius:6,padding:'13px 18px',marginBottom:24,fontSize:12,color:'#374151',lineHeight:1.8,display:'flex',flexWrap:'wrap',gap:'4px 24px'}}>
          <span><span style={{fontWeight:700,color:'#1d4ed8'}}>Overall xGD/60</span>: observed 5v5 xG differential per 60 across all qualifying shift segments</span>
          <span><span style={{fontWeight:700,color:'#1d4ed8'}}>Early xGD/60</span>: avg xG differential per 60 in first 30s on ice</span>
          <span><span style={{fontWeight:700,color:'#1d4ed8'}}>Late xGD/60</span>: avg xG differential per 60 after 45s on ice</span>
          <span><span style={{fontWeight:700,color:'#1d4ed8'}}>Drop</span>: Late minus Early (negative = fades with shift age)</span>
          <span><span style={{fontWeight:700,color:'#1d4ed8'}}>Durability</span>: Late minus Early xGD/60 (more negative = faster fade)</span>
          <span>Cell color: <span style={{background:'#1d4ed8',color:'#fff',borderRadius:3,padding:'0 5px',fontSize:11}}>blue</span> = elite percentile · <span style={{background:'#dc2626',color:'#fff',borderRadius:3,padding:'0 5px',fontSize:11}}>red</span> = bottom</span>
        </div>
      )}

      {/* Leaderboard grid */}
      {!searchResults && (
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:18}}>
          {boards.map(board=>(
            <div key={board.title} style={{background:board.headBg||'#fff',border:`1px solid ${board.accent?'#fecaca':'#e8e8e8'}`,borderRadius:6,overflow:'hidden'}}>
              <div style={{padding:'14px 16px 10px',borderBottom:'1px solid #f0f0f0',background:board.headBg||'#fff',minHeight:96,boxSizing:'border-box'}}>
                <div style={{fontSize:13,fontWeight:700,color:board.accent||'#111',marginBottom:3}}>{board.title}</div>
                <div style={{fontSize:11,color:'#888',lineHeight:1.5}}>{board.desc}</div>
              </div>
              <table style={{width:'100%',borderCollapse:'collapse'}}>
                <thead>
                  <tr style={{background:'#fafafa'}}>
                    <th style={{textAlign:'left',padding:'6px 12px',fontSize:10,color:'#aaa',fontWeight:600,textTransform:'uppercase',letterSpacing:'0.05em',width:20}}>#</th>
                    <th style={{textAlign:'left',padding:'6px 4px',fontSize:10,color:'#aaa',fontWeight:600,textTransform:'uppercase',letterSpacing:'0.05em'}}>Player</th>
                    <th style={{padding:'6px 8px',fontSize:10,color:'#aaa',fontWeight:600,textTransform:'uppercase',letterSpacing:'0.05em'}}>Team</th>
                    <th style={{padding:'6px 10px',fontSize:10,color:'#aaa',fontWeight:600,textTransform:'uppercase',letterSpacing:'0.05em'}}>{board.label}</th>
                  </tr>
                </thead>
                <tbody>
                  {board.players.map((p,i)=>(
                    <tr key={p.id} onClick={()=>openPlayer(p)} style={{cursor:'pointer',borderTop:'1px solid #f5f5f5'}}
                      onMouseEnter={e=>e.currentTarget.style.background='#f8faff'}
                      onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
                      <td style={{padding:'7px 12px',fontFamily:'IBM Plex Mono',fontSize:10,color:'#ccc',width:20}}>{i+1}</td>
                      <td style={{padding:'5px 4px'}}>
                        <div style={{display:'flex',alignItems:'center',gap:7}}>
                          <PlayerHeadshot id={p.id} name={p.name} team={p.team} size={28}/>
                          <div>
                            <div style={{fontSize:12,fontWeight:600,color:'#1d4ed8'}}>{p.name}</div>
                            <div style={{fontSize:10,color:'#aaa'}}>{p.pos}</div>
                          </div>
                        </div>
                      </td>
                      <td style={{padding:'7px 8px',textAlign:'center'}}><TeamLogo code={p.team} size={20}/></td>
                      <PctCell pct={board.pct(p)} value={board.stat(p)}/>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── STINTS TABLE ─────────────────────────────────────────────────────────────
const ZONE_LABELS = {O:'Off. zone', N:'Neutral', D:'Def. zone'};
const ZONE_COLORS = {O:'#dcfce7', N:'#f3f4f6', D:'#fee2e2'};
const ZONE_TEXT   = {O:'#166534', N:'#555', D:'#991b1b'};

function fmtPeriod(p) { return p > 3 ? 'OT' : `P${p}`; }
function fmtTime(sec) {
  const s = sec % 1200;
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
}
function fmtScore(n) {
  if (n === 0) return <span style={{color:'#888'}}>Tied</span>;
  return n > 0
    ? <span style={{color:'#166534'}}>Up {n > 1 ? n : ''}</span>
    : <span style={{color:'#991b1b'}}>Down {n < -1 ? Math.abs(n) : ''}</span>;
}

const StintsTable = ({player, season}) => {
  const [stints, setStints]   = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [page, setPage]       = React.useState(0);
  const [sortCol, setSortCol] = React.useState('game_id');
  const [sortDir, setSortDir] = React.useState('desc');
  const [zoneFilter, setZoneFilter] = React.useState('');
  const [expanded, setExpanded] = React.useState(null);
  const [minDur, setMinDur]     = React.useState(10);
  const PAGE = 50;

  const load = () => {
    setLoading(true);
    fetch(`/player/${player.id}/stints?season=${season}`)
      .then(r => r.json())
      .then(d => { setStints(d.stints || []); setLoading(false); })
      .catch(() => { setStints([]); setLoading(false); });
  };

  const toggleSort = col => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
    setPage(0);
  };

  if (!stints) {
    return (
      <div style={{textAlign:'center',padding:'24px 0'}}>
        <button onClick={load} disabled={loading} style={{
          background:'#1d4ed8', color:'#fff', border:'none', borderRadius:5,
          padding:'10px 24px', fontSize:13, fontWeight:600, cursor:'pointer',
          opacity: loading ? 0.6 : 1,
        }}>{loading ? 'Loading…' : `Load shifts`}</button>
        <div style={{fontSize:11,color:'#aaa',marginTop:6}}>Individual 5v5 player shifts (consecutive on-ice sequences grouped together)</div>
      </div>
    );
  }

  const filtered = stints.filter(s => s.duration >= minDur && (!zoneFilter || s.zone === zoneFilter));
  const cmpDir   = sortDir === 'asc' ? 1 : -1;
  const sorted   = [...filtered].sort((a,b) => {
    const v = a[sortCol] < b[sortCol] ? -1 : a[sortCol] > b[sortCol] ? 1 : 0;
    return v * cmpDir;
  });
  const pageData  = sorted.slice(page * PAGE, (page+1) * PAGE);
  const totalPages = Math.ceil(sorted.length / PAGE);

  // zone summary
  const zoneCounts = {O:0,N:0,D:0};
  stints.forEach(s => { if (zoneCounts[s.zone] != null) zoneCounts[s.zone]++; });
  const total = stints.length;

  const Th = ({col, label, right}) => {
    const active = sortCol === col;
    return (
      <th onClick={() => toggleSort(col)} style={{cursor:'pointer',textAlign:right?'right':'left',
        color:active?'#1d4ed8':'#666', userSelect:'none'}}>
        {label}{active ? (sortDir==='asc' ? ' ↑' : ' ↓') : ''}
      </th>
    );
  };

  return (
    <div>
      <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:12,flexWrap:'wrap'}}>
        <div style={{fontSize:13,color:'#666',fontWeight:500}}>{sorted.length.toLocaleString()} shifts</div>
        <div style={{display:'flex',gap:4}}>
          {[0,10,20,30].map(d => (
            <button key={d} onClick={() => { setMinDur(d); setPage(0); }} style={{
              padding:'3px 9px', fontSize:11, fontWeight:600, borderRadius:4, border:'1px solid',
              borderColor: minDur===d ? '#374151' : '#ddd',
              background: minDur===d ? '#f3f4f6' : '#fff',
              color: minDur===d ? '#111' : '#888', cursor:'pointer',
            }}>{d===0?'All':`≥${d}s`}</button>
          ))}
        </div>
        <div style={{width:1,height:16,background:'#e5e5e5'}}/>
        <div style={{display:'flex',gap:6}}>
          {['','O','N','D'].map(z => (
            <button key={z} onClick={() => { setZoneFilter(z); setPage(0); }} style={{
              padding:'3px 10px', fontSize:11, fontWeight:600, borderRadius:4, border:'1px solid',
              borderColor: zoneFilter===z ? '#1d4ed8' : '#ddd',
              background: zoneFilter===z ? '#eff6ff' : '#fff',
              color: zoneFilter===z ? '#1d4ed8' : '#555', cursor:'pointer',
            }}>{z===''?'All zones':ZONE_LABELS[z]}</button>
          ))}
        </div>
        {total > 0 && (
          <div style={{marginLeft:'auto',fontSize:11,color:'#aaa',display:'flex',gap:12}}>
            {Object.entries(zoneCounts).map(([z,n]) => (
              <span key={z} style={{background:ZONE_COLORS[z],color:ZONE_TEXT[z],borderRadius:3,padding:'1px 7px',fontWeight:600}}>
                {ZONE_LABELS[z]} {Math.round(n/total*100)}%
              </span>
            ))}
          </div>
        )}
      </div>

      <div style={{border:'1px solid #e8e8e8',borderRadius:6,overflow:'hidden'}}>
        <table style={{fontSize:12}}>
          <thead><tr style={{background:'#fafafa'}}>
            <Th col="game_id" label="Game"/>
            <Th col="period" label="Per" right/>
            <th style={{textAlign:'right',color:'#666'}}>Time</th>
            <Th col="duration" label="Length" right/>
            <Th col="zone" label="Zone"/>
            <Th col="score" label="Score" right/>
            <Th col="xgd" label="xGD" right/>
            <th style={{textAlign:'right',color:'#ccc',fontSize:10,fontWeight:400}}>segs</th>
          </tr></thead>
          <tbody>
            {pageData.map((s,i) => {
              const isExp = expanded === `${s.game_id}-${s.period}-${s.start_sec}`;
              const xgdColor = s.xgd > 0.005 ? '#166534' : s.xgd < -0.005 ? '#991b1b' : '#555';
              return (
                <tr key={i} onClick={() => setExpanded(isExp ? null : `${s.game_id}-${s.period}-${s.start_sec}`)}
                  style={{cursor:'pointer', background: isExp ? '#f0f4ff' : 'transparent',
                    borderTop:'1px solid #f5f5f5'}}>
                  <td style={{color:'#888',fontSize:11}} title={`Game ID ${s.game_id}`}>{s.game_label || s.game_id}</td>
                  <td style={{textAlign:'right',fontFamily:'IBM Plex Mono'}}>{fmtPeriod(s.period)}</td>
                  <td style={{textAlign:'right',fontFamily:'IBM Plex Mono',color:'#aaa'}}>{fmtTime(s.start_sec)}</td>
                  <td style={{textAlign:'right',fontFamily:'IBM Plex Mono',fontWeight:600}}>{s.duration}s</td>
                  <td>
                    <span style={{background:ZONE_COLORS[s.zone]||'#f3f4f6',color:ZONE_TEXT[s.zone]||'#555',
                      borderRadius:3,padding:'1px 6px',fontWeight:600,fontSize:10}}>
                      {ZONE_LABELS[s.zone]||s.zone}
                    </span>
                  </td>
                  <td style={{textAlign:'right'}}>{fmtScore(s.score)}</td>
                  <td style={{textAlign:'right',fontFamily:'IBM Plex Mono',fontWeight:600,color:xgdColor}}>
                    {s.xgd >= 0 ? '+' : ''}{s.xgd.toFixed(4)}
                  </td>
                  <td style={{textAlign:'right',fontFamily:'IBM Plex Mono',fontSize:11,color:'#ccc'}}>{s.n_segs||1}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div style={{display:'flex',justifyContent:'center',gap:6,marginTop:12,alignItems:'center'}}>
          <button onClick={() => setPage(p => Math.max(0,p-1))} disabled={page===0}
            style={{padding:'4px 12px',borderRadius:4,border:'1px solid #ddd',cursor:'pointer',fontSize:12,background:'#fff'}}>
            ‹ Prev
          </button>
          <span style={{fontSize:12,color:'#888'}}>Page {page+1} of {totalPages} ({sorted.length.toLocaleString()} stints)</span>
          <button onClick={() => setPage(p => Math.min(totalPages-1,p+1))} disabled={page===totalPages-1}
            style={{padding:'4px 12px',borderRadius:4,border:'1px solid #ddd',cursor:'pointer',fontSize:12,background:'#fff'}}>
            Next ›
          </button>
        </div>
      )}
    </div>
  );
};

// ── PLAYER PROFILE ────────────────────────────────────────────────────────────
const PlayerProfile = ({player, setPage}) => {
  const [empData, setEmpData] = React.useState(null);
  const season = window.__DATA__.current_season;
  const hasEnoughShiftData = !!player?.eligible_for_graphs;

  React.useEffect(() => {
    if (!player) return;
    if (!player.eligible_for_graphs) {
      setEmpData({buckets: [], values: [], insufficient: true});
      return;
    }
    setEmpData(null);
    fetch(`/player/${player.id}/decay?season=${season}`)
      .then(r => r.json())
      .then(d => setEmpData(d))
      .catch(() => setEmpData({buckets:[], values:[], se:[]}));
  }, [player?.id, season]);

  if (!player) return <div style={{padding:40,color:'#888'}}>No player selected.</div>;

  return (
    <div style={{maxWidth:1140,margin:'0 auto',padding:'32px 24px'}}>
      {/* Header */}
      <div style={{display:'flex',alignItems:'center',gap:20,marginBottom:24}}>
        <PlayerHeadshot id={player.id} name={player.name} team={player.team} size={88}/>
        <div>
          <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}>
            <TeamLogo code={player.team} size={32}/>
            <h1 style={{fontSize:32,fontWeight:700,letterSpacing:'-0.025em',lineHeight:1}}>{player.name}</h1>
          </div>
          <div style={{fontSize:13,color:'#888',display:'flex',gap:10,alignItems:'center'}}>
            <Tag bg="#f0f4ff" color="#1d4ed8">{player.team}</Tag>
            <span>{player.pos}</span><span>·</span><span>5v5</span>
            {player.flagged && <Tag bg="#fee2e2" color="#dc2626">⚠ Overuse Flagged</Tag>}
          </div>
        </div>
      </div>

      {/* Top row: chart + shift breakdown */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 210px',gap:16,marginBottom:16}}>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
          <SectionHeader
            title={<span>Observed Shift Fade <InfoTooltip text="This chart uses tracked 5v5 on-ice stint data, split into 10-second shift-age buckets. We calculate xGD/60 separately for each bucket, then rebase the chart so the player's 0-10s bucket equals zero. That means every later point shows how much better or worse the player performed relative to his own fresh-shift baseline, not whether he was good in absolute terms. The teal dashed line is the league-average change over the same buckets. Blue bars show how much player ice time exists in each bucket, so very small bars mean the late-shift points are less trustworthy." /></span>}
            sub="How hard does this player fall off the cliff as his shift gets longer? The blue line shows change from his own fresh-shift level, so downward movement means a real drop-off. Teal dashed = league-average drop-off. Blue bars = sample size."/>
          {empData === null
            ? <div style={{height:280,display:'flex',alignItems:'center',justifyContent:'center',color:'#bbb',fontSize:13}}>Loading…</div>
            : !hasEnoughShiftData
              ? <div style={{height:280,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',color:'#999',fontSize:13,gap:6,textAlign:'center'}}>
                  <div style={{fontSize:18,fontWeight:700,color:'#777'}}>Not enough data</div>
                  <div style={{fontSize:12,color:'#aaa'}}>{player.qualifying_shifts_10s || 0} qualifying shifts over {MIN_QUALIFYING_SHIFT_SECONDS}s · minimum {MIN_QUALIFYING_SHIFTS} required</div>
                </div>
            : empData.buckets.length < 4
              ? <div style={{height:280,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',color:'#999',fontSize:13,gap:6,textAlign:'center'}}>
                  <div style={{fontSize:14,fontWeight:600,color:'#777'}}>Not enough data for a reliable curve</div>
                  <div style={{fontSize:12,color:'#aaa'}}>{player.toi_min} min 5v5 TOI · insufficient shift-age coverage</div>
                </div>
              : <PlayerShiftTrendChart
                  buckets={empData.buckets}
                  deltaValues={empData.delta_values}
                  rawValues={empData.values}
                  toiSec={empData.toi_sec}
                  leagueBuckets={empData.league?.buckets || []}
                  leagueDeltaValues={empData.league?.delta_values || []}
                  width={680}
                  height={280}
                />
          }
        </div>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
          <div style={{fontSize:11,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.06em',color:'#888',marginBottom:14}}>Shift Breakdown</div>
          {[
            ['Fresh (0–30s)', fmt(player.early), player.pct_early],
            ['Mid (30–45s)',  fmt(player.mid), null],
            ['Late (45s+)',   fmt(player.late),  player.pct_late],
            ['Durability',    fmt(player.drop),  player.pct_drop],
          ].map(([l,v,pct])=>(
            <div key={l} style={{marginBottom:12}}>
              <div style={{fontSize:10,color:'#aaa',marginBottom:1,display:'flex',justifyContent:'space-between'}}>
                <span>{l}</span>
                {pct != null && <span style={{background:pctBg(Math.max(1,pct)),color:pctFg(Math.max(1,pct)),
                  borderRadius:3,padding:'0 5px',fontSize:10,fontWeight:600}}>{Math.max(1,pct)}th</span>}
              </div>
              <div style={{fontSize:20,fontWeight:700,fontFamily:'IBM Plex Mono',letterSpacing:'-0.02em'}}>{v}</div>
            </div>
          ))}
          <hr style={{border:'none',borderTop:'1px solid #f0f0f0',margin:'10px 0'}}/>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
            {[['TOI',`${player.toi_min}m`],['Stints',player.stints.toLocaleString()]].map(([l,v])=>(
              <div key={l}>
                <div style={{fontSize:10,color:'#aaa'}}>{l}</div>
                <div style={{fontSize:16,fontWeight:700,fontFamily:'IBM Plex Mono'}}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Context row: league percentiles + observed impact */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16,marginBottom:16}}>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'16px 20px'}}>
          <div style={{fontSize:11,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.06em',color:'#888',marginBottom:12}}>Context vs. League</div>
          <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
            <PctBadge pct={player.pct_early} label="Early xGD/60"/>
            <PctBadge pct={player.pct_late}  label="Late xGD/60"/>
            <PctBadge pct={player.pct_drop}  label="Durability"/>
            <PctBadge pct={player.pct_overall}  label="Overall xGD/60"/>
            <PctBadge pct={player.pct_toi}   label="TOI"/>
          </div>
        </div>
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'16px 20px'}}>
          <div style={{fontSize:11,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.06em',color:'#888',marginBottom:12}}>Observed Impact</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12}}>
            {[['Overall xGD/60',fmt(player.overall_xgd,2)],['Durability',fmt(player.drop,2)],['Qualifying shifts',player.qualifying_shifts_10s]].map(([l,v])=>(
              <div key={l}><div style={{fontSize:10,color:'#aaa',marginBottom:2}}>{l}</div>
                <div style={{fontSize:17,fontWeight:700,fontFamily:'IBM Plex Mono'}}>{v}</div></div>
            ))}
          </div>
          <div style={{fontSize:10,color:'#bbb',marginTop:8,fontStyle:'italic'}}>Empirical 5v5 xGD/60 from qualifying shift segments.</div>
        </div>
      </div>


      {/* Stint list */}
      <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
        <SectionHeader title="Individual Shifts" sub="Each row = one player shift. Clock-adjacent sub-stints (same player on ice across a line change) are merged into one. 'Segs' = how many sub-stints were merged. Stoppages (face-offs, icings) always break a shift."/>
        <StintsTable player={player} season={season}/>
      </div>
    </div>
  );
};

// ── LINES PAGE ────────────────────────────────────────────────────────────────

// Visual bar showing a value relative to zero: left=negative, right=positive
const ShiftBar = ({value, maxAbs=6, width=120, height=18, label}) => {
  const clamp = v => Math.max(-maxAbs, Math.min(maxAbs, v));
  const v = clamp(value);
  const mid = width / 2;
  const scale = (width / 2) / maxAbs;
  const barW = Math.abs(v) * scale;
  const barX = v >= 0 ? mid : mid - barW;
  const fill = v >= 0 ? '#16a34a' : '#dc2626';
  return (
    <svg width={width} height={height} style={{display:'block',overflow:'visible'}}>
      <line x1={mid} y1={2} x2={mid} y2={height-2} stroke="#e0e0e0" strokeWidth={1}/>
      {barW > 0 && <rect x={barX} y={4} width={barW} height={height-8} fill={fill} opacity={0.75} rx={2}/>}
      {label && <text x={v >= 0 ? mid + barW + 4 : mid - barW - 4} y={height/2+4}
        textAnchor={v >= 0 ? 'start' : 'end'} fontSize={10} fill={fill} fontFamily="IBM Plex Mono">
        {value > 0 ? '+' : ''}{value.toFixed(1)}
      </text>}
    </svg>
  );
};

// Early→Late arrow showing shift fade or durability
const EarlyLateArrow = ({early, late, width=220}) => {
  const maxAbs = 8;
  const clamp = v => Math.max(-maxAbs, Math.min(maxAbs, v));
  const scale = v => ((clamp(v) + maxAbs) / (2 * maxAbs)) * width;
  const earlyX = scale(early);
  const lateX  = scale(late);
  const decay  = late - early;
  const arrowColor = decay >= 0 ? '#16a34a' : decay > -2 ? '#f59e0b' : '#dc2626';
  const H = 28;
  const mid = ((0 + maxAbs) / (2 * maxAbs)) * width;
  return (
    <svg width={width} height={H} style={{display:'block',overflow:'visible'}}>
      <line x1={0} y1={H/2} x2={width} y2={H/2} stroke="#f0f0f0" strokeWidth={1}/>
      <line x1={mid} y1={4} x2={mid} y2={H-4} stroke="#d0d0d0" strokeWidth={1} strokeDasharray="2 2"/>
      {/* early marker */}
      <circle cx={earlyX} cy={H/2} r={5} fill={early >= 0 ? '#1d4ed8' : '#dc2626'} opacity={0.9}/>
      {/* arrow shaft */}
      <line x1={earlyX} y1={H/2} x2={lateX} y2={H/2} stroke={arrowColor} strokeWidth={2.5} opacity={0.8}/>
      {/* arrowhead */}
      {Math.abs(lateX - earlyX) > 4 && (() => {
        const dir = lateX > earlyX ? 1 : -1;
        return <polygon points={`${lateX},${H/2} ${lateX - dir*7},${H/2-4} ${lateX - dir*7},${H/2+4}`}
          fill={arrowColor} opacity={0.8}/>;
      })()}
      {/* late marker */}
      <circle cx={lateX} cy={H/2} r={4} fill={late >= 0 ? '#16a34a' : '#dc2626'} opacity={0.9} stroke="#fff" strokeWidth={1}/>
    </svg>
  );
};

const LineCard = ({line, rank, openPlayer}) => {
  const decay = line.decay;
  const decayColor = decay >= 0 ? '#16a34a' : decay > -2 ? '#f59e0b' : '#dc2626';
  const decayLabel = decay >= 1 ? 'Improves late' : decay >= -1 ? 'Holds level' : decay >= -3 ? 'Moderate fade' : 'Sharp fade';
  return (
    <div style={{
      background: '#fff', border: '1px solid', borderColor: line.flagged ? '#fca5a5' : '#e8e8e8',
      borderRadius: 6, padding: '14px 18px', display: 'grid',
      gridTemplateColumns: '32px 1fr auto', gap: '0 16px', alignItems: 'center',
    }}>
      <div style={{fontSize: 13, fontWeight: 700, color: '#bbb', textAlign: 'center'}}>{rank}</div>
      <div>
        <div style={{display:'flex', alignItems:'center', gap:8, marginBottom:5}}>
          <TeamLogo code={line.team} size={20}/>
          <span style={{fontSize:13, fontWeight:700, color:'#111', lineHeight:1.3}}>
            <InlinePlayerList parts={line.player_parts || []} onSelect={openPlayer}/>
          </span>
          {line.flagged && <span style={{fontSize:10,background:'#fee2e2',color:'#dc2626',borderRadius:3,padding:'1px 5px',fontWeight:700}}>FADE</span>}
        </div>
        <div style={{display:'flex', alignItems:'center', gap:12, flexWrap:'wrap'}}>
          <EarlyLateArrow early={line.early_xgd} late={line.late_xgd} width={180}/>
          <div style={{fontSize:11, color:'#666', display:'flex', gap:10}}>
            <span>Early <span style={{fontFamily:'IBM Plex Mono', fontWeight:600, color: line.early_xgd >= 0 ? '#1d4ed8' : '#dc2626'}}>{line.early_xgd > 0 ? '+' : ''}{line.early_xgd.toFixed(1)}</span></span>
            <span style={{color:'#ccc'}}>→</span>
            <span>Late <span style={{fontFamily:'IBM Plex Mono', fontWeight:600, color: line.late_xgd >= 0 ? '#16a34a' : '#dc2626'}}>{line.late_xgd > 0 ? '+' : ''}{line.late_xgd.toFixed(1)}</span></span>
            <span style={{color:'#ccc'}}>·</span>
            <span style={{color: decayColor, fontWeight:600}}>{decayLabel}</span>
          </div>
        </div>
      </div>
      <div style={{textAlign:'right', minWidth:120}}>
        <div style={{fontSize:20, fontWeight:700, fontFamily:'IBM Plex Mono', color: decay >= 0 ? '#16a34a' : '#dc2626', letterSpacing:'-0.03em'}}>
          {decay > 0 ? '+' : ''}{decay.toFixed(2)}
        </div>
        <div style={{fontSize:10, color:'#999', marginBottom:4}}>Fade</div>
        <div style={{fontSize:11, color:'#aaa', fontFamily:'IBM Plex Mono'}}>
          {line.toi_min} min · {line.stints} stints
        </div>
      </div>
    </div>
  );
};

const SpotlightCard = ({title, sub, line, accent, openPlayer}) => {
  if (!line) return null;
  const decay = line.decay;
  return (
    <div style={{background:'#fff', border:'1px solid #e8e8e8', borderTop:`3px solid ${accent}`, borderRadius:6, padding:'16px 18px'}}>
      <div style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.08em', color:accent, marginBottom:6}}>{title}</div>
      <div style={{fontSize:13, fontWeight:700, lineHeight:1.35, marginBottom:2, color:'#111'}}>
        <InlinePlayerList parts={line.player_parts || []} onSelect={openPlayer}/>
      </div>
      <div style={{fontSize:11, color:'#888', marginBottom:10}}><Tag>{line.team}</Tag> · {line.toi_min} min TOI</div>
      <div style={{display:'flex', gap:16}}>
        <div>
          <div style={{fontSize:10, color:'#aaa', textTransform:'uppercase', letterSpacing:'0.05em'}}>Overall</div>
          <div style={{fontSize:18, fontWeight:700, fontFamily:'IBM Plex Mono', color: line.xgd60 >= 0 ? '#111' : '#dc2626'}}>
            {line.xgd60 > 0 ? '+' : ''}{line.xgd60.toFixed(2)}
          </div>
        </div>
        <div>
          <div style={{fontSize:10, color:'#aaa', textTransform:'uppercase', letterSpacing:'0.05em'}}>Late</div>
          <div style={{fontSize:18, fontWeight:700, fontFamily:'IBM Plex Mono', color: line.late_xgd >= 0 ? '#16a34a' : '#dc2626'}}>
            {line.late_xgd > 0 ? '+' : ''}{line.late_xgd.toFixed(2)}
          </div>
        </div>
        <div>
          <div style={{fontSize:10, color:'#aaa', textTransform:'uppercase', letterSpacing:'0.05em'}}>Decay</div>
          <div style={{fontSize:18, fontWeight:700, fontFamily:'IBM Plex Mono', color: decay >= 0 ? '#16a34a' : '#dc2626'}}>
            {decay > 0 ? '+' : ''}{decay.toFixed(2)}
          </div>
        </div>
      </div>
      <div style={{fontSize:11, color:'#999', marginTop:8}}>{sub}</div>
    </div>
  );
};

const LinesPage = ({openPlayer}) => {
  const [teamFilter, setTeamFilter] = useState('All');
  const [search, setSearch] = useState('');
  const [minToi, setMinToi] = useState(20);
  const [sortBy, setSortBy] = useState('overall');
  const [showCount, setShowCount] = useState(40);
  const query = normalizeSearchText(search);
  const compactQuery = compactSearchText(search);
  const lineTeams = [...new Set(LINES.map(l=>l.team))].filter(Boolean).sort();

  const scoped = LINES.filter(l =>
    (teamFilter === 'All' || l.team === teamFilter) &&
    l.toi_min >= minToi &&
    (() => {
      if (!query) return true;
      const blob = lineSearchBlob(l);
      return blob.normalized.includes(query) || (compactQuery && blob.compact.includes(compactQuery));
    })()
  );

  const sorted = [...scoped].sort((a,b) => {
    if (sortBy === 'overall') return b.xgd60 - a.xgd60;
    if (sortBy === 'late')    return b.late_xgd - a.late_xgd;
    if (sortBy === 'durable') return b.decay - a.decay;
    if (sortBy === 'fade')    return a.decay - b.decay;
    return b.xgd60 - a.xgd60;
  });

  const visible = sorted.slice(0, showCount);
  const flaggedCount = scoped.filter(l => l.flagged).length;

  const bestOverall  = [...LINES].filter(l=>l.toi_min>=20).sort((a,b)=>b.xgd60-a.xgd60)[0];
  const mostDurable  = [...LINES].filter(l=>l.toi_min>=20).sort((a,b)=>b.decay-a.decay)[0];
  const bestLate     = [...LINES].filter(l=>l.toi_min>=20).sort((a,b)=>b.late_xgd-a.late_xgd)[0];

  const SORTS = [
    {k:'overall', label:'Best Overall'},
    {k:'late',    label:'Best Late'},
    {k:'durable', label:'Most Durable'},
    {k:'fade',    label:'Sharpest Fade'},
  ];

  return (
    <div style={{maxWidth:1100,margin:'0 auto',padding:'32px 24px'}}>
      <div style={{marginBottom:24}}>
        <h1 style={{fontSize:26,fontWeight:700,letterSpacing:'-0.02em',marginBottom:6}}>Line Combination Analysis</h1>
        <p style={{fontSize:13,color:'#777',maxWidth:680,lineHeight:1.6}}>
          Each row is one 3-forward line at 5v5, independent of which defense pair was behind it. We compare that trio's results in the
          first 30 seconds of a shift with its results after 45 seconds, so you can see which forward lines hold up and which ones fade late.
        </p>
      </div>

      {/* Spotlight row */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:14,marginBottom:28}}>
        <SpotlightCard title="Most Dominant Line" sub="Best overall chance results among lines with 20+ minutes together." line={bestOverall} accent="#1d4ed8" openPlayer={openPlayer}/>
        <SpotlightCard title="Most Durable Line" sub="Smallest fade from early shift to late shift." line={mostDurable} accent="#16a34a" openPlayer={openPlayer}/>
        <SpotlightCard title="Best Late-Shift Line" sub="Best chance results after the shift gets old." line={bestLate} accent="#0891b2" openPlayer={openPlayer}/>
      </div>

      {/* Filters */}
      <div style={{display:'flex',gap:10,marginBottom:18,alignItems:'center',flexWrap:'wrap'}}>
        <select value={teamFilter} onChange={e=>setTeamFilter(e.target.value)} style={{minWidth:130}}>
          <option value="All">All teams</option>
          {lineTeams.map(t=><option key={t} value={t}>{t}</option>)}
        </select>
        <div style={{display:'flex',gap:4}}>
          {[10,20,50,100].map(v=>(
            <button key={v} onClick={()=>setMinToi(v)} style={{
              padding:'7px 11px',fontSize:12,fontWeight:600,border:'1px solid',
              borderColor:minToi===v?'#1d4ed8':'#ddd',borderRadius:5,
              background:minToi===v?'#eff6ff':'#fff',color:minToi===v?'#1d4ed8':'#666'
            }}>{v}+ min</button>
          ))}
        </div>
        <input type="text" placeholder="Search player or team..." value={search} onChange={e=>setSearch(e.target.value)}
          style={{flex:'1 1 200px',padding:'8px 12px',fontSize:13,border:'1px solid #d1d5db',borderRadius:5}}/>
        <div style={{marginLeft:'auto',fontSize:12,color:'#999'}}>{scoped.length} units · {flaggedCount} flagged</div>
      </div>

      {/* Sort tabs */}
      <div style={{display:'flex',gap:4,marginBottom:16}}>
        <span style={{fontSize:12,color:'#aaa',alignSelf:'center',marginRight:4}}>Sort:</span>
        {SORTS.map(({k,label})=>(
          <button key={k} onClick={()=>setSortBy(k)} style={{
            padding:'6px 12px',fontSize:12,fontWeight:600,border:'1px solid',
            borderColor:sortBy===k?'#1d4ed8':'#e5e5e5',borderRadius:5,
            background:sortBy===k?'#1d4ed8':'#fff',color:sortBy===k?'#fff':'#555'
          }}>{label}</button>
        ))}
      </div>

      {/* Legend */}
      <div style={{background:'#f8f9ff',border:'1px solid #e0e7ff',borderRadius:5,padding:'10px 14px',marginBottom:16,
        fontSize:12,color:'#4b5563',display:'flex',gap:24,flexWrap:'wrap'}}>
        <span><span style={{fontFamily:'IBM Plex Mono',fontWeight:700,color:'#1d4ed8'}}>●</span> Blue dot = how the line performed early in the shift</span>
        <span><span style={{fontFamily:'IBM Plex Mono',fontWeight:700,color:'#16a34a'}}>●→</span> Arrow end = how the line performed late in the shift</span>
        <span><span style={{fontFamily:'IBM Plex Mono',fontWeight:700,color:'#dc2626'}}>Fade</span> = late result minus early result, so more negative means a harder drop-off</span>
      </div>

      {/* Unit list */}
      <div style={{display:'flex',flexDirection:'column',gap:8,marginBottom:20}}>
        {visible.length === 0
          ? <div style={{padding:40,textAlign:'center',color:'#bbb',fontSize:14}}>No units match the current filters.</div>
          : visible.map((l,i) => <LineCard key={l.players+l.team} line={l} rank={i+1} openPlayer={openPlayer}/>)
        }
      </div>

      {sorted.length > showCount && (
        <div style={{textAlign:'center',marginBottom:32}}>
          <button onClick={()=>setShowCount(c=>c+40)} style={{
            padding:'10px 24px',fontSize:13,fontWeight:600,border:'1px solid #ddd',
            borderRadius:6,background:'#fff',color:'#555',cursor:'pointer'
          }}>Show more ({sorted.length - showCount} remaining)</button>
        </div>
      )}
    </div>
  );
};

// ── TEAM REPORT ───────────────────────────────────────────────────────────────
const TeamReportPage = ({selectedTeam, onTeamChange, openPlayer}) => {
  const team = selectedTeam || (TEAMS[0] || '');
  const [tip, setTip] = useState(null);
  const [sortKey, setSortKey] = useState('toi_min');
  const [sortDir, setSortDir] = useState('desc');
  const roster = CHARTED.filter(p=>p.team===team).sort((a,b)=>b.toi_min-a.toi_min);
  const flags = roster.filter(p=>p.flagged).length;
  const avgDurabilityValue = roster.length ? roster.reduce((s,p)=>s+p.drop,0)/roster.length : 0;
  const avgD  = roster.length ? avgDurabilityValue.toFixed(2) : 'N/A';
  const avgOverall  = roster.length ? (roster.reduce((s,p)=>s+p.overall_xgd,0)/roster.length).toFixed(2) : 'N/A';
  const qualityThreshold = avgOverallXgd;
  const durabilityThreshold = avgDurability;

  // ── Quality vs. Durability scatter ──────────────────────────────────────────
  const scW=480, scH=280;
  const sm={top:20,right:20,bottom:48,left:56};
  const sW=scW-sm.left-sm.right, sH=scH-sm.top-sm.bottom;
  const xs=roster.map(p=>p.drop), ys=roster.map(p=>p.overall_xgd);
  const xPad=0.2, yPad=0.2;
  const xDataMin = xs.length ? Math.min(...xs) : durabilityThreshold - 1;
  const xDataMax = xs.length ? Math.max(...xs) : durabilityThreshold + 1;
  const yDataMin = ys.length ? Math.min(...ys) : qualityThreshold - 1;
  const yDataMax = ys.length ? Math.max(...ys) : qualityThreshold + 1;
  const xMin = Math.min(xDataMin - xPad, durabilityThreshold - xPad);
  const xMax = Math.max(xDataMax + xPad, durabilityThreshold + xPad);
  const yMin = Math.min(yDataMin - yPad, qualityThreshold - yPad);
  const yMax = Math.max(yDataMax + yPad, qualityThreshold + yPad);
  const sxS=v=>((v-xMin)/(xMax-xMin))*sW;
  const syS=v=>sH-((v-yMin)/(yMax-yMin))*sH;
  // quadrant labels
  const quadrants = [
    {x:sW-12,y:18,   label:'Elite and Durable',color:'#15803d',anchor:'end', bg:'rgba(255,255,255,0.94)'},
    {x:12,y:18,      label:'Elite but Fades',  color:'#b45309',anchor:'start', bg:'rgba(255,255,255,0.94)'},
    {x:sW-12,y:sH-12,label:'Safe but Low Output', color:'#6b7280',anchor:'end', bg:'rgba(255,255,255,0.94)'},
    {x:12,y:sH-12,   label:'Avoid',             color:'#dc2626',anchor:'start', bg:'rgba(255,255,255,0.94)'},
  ];

  const onSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir(key === 'name' || key === 'pos' ? 'asc' : 'desc');
    }
  };

  const sortedRoster = [...roster].sort((a, b) => {
    const dir = sortDir === 'asc' ? 1 : -1;
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === 'string' || typeof bv === 'string') {
      return String(av).localeCompare(String(bv)) * dir;
    }
    return ((Number(av) || 0) - (Number(bv) || 0)) * dir;
  });

  const sortArrow = (key) => sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '';

  const handleScatterMove = (e, p) => {
    const rect = e.currentTarget.closest('svg').getBoundingClientRect();
    setTip({p, x: e.clientX - rect.left, y: e.clientY - rect.top});
  };

  // ── Deployment guide groups ──────────────────────────────────────────────────
  const deployGroups = [
    {
      label: 'Elite and Durable', color: '#15803d', bg: '#f0fdf4',
      advice: 'Long shifts OK. Play in all situations.',
      players: roster.filter(p => p.overall_xgd > qualityThreshold && p.drop >= durabilityThreshold),
    },
    {
      label: 'Elite but Fades', color: '#b45309', bg: '#fffbeb',
      advice: 'Keep shifts short. Swap out before 40s.',
      players: roster.filter(p => p.overall_xgd > qualityThreshold && p.drop < durabilityThreshold),
    },
    {
      label: 'Safe but Low Output', color: '#6b7280', bg: '#f9fafb',
      advice: 'Depth minutes. Shift length flexible.',
      players: roster.filter(p => p.overall_xgd <= qualityThreshold && p.drop >= durabilityThreshold),
    },
    {
      label: 'Avoid', color: '#dc2626', bg: '#fff5f5',
      advice: 'Minimize deployment. Below league-average impact and fading.',
      players: roster.filter(p => p.overall_xgd <= qualityThreshold && p.drop < durabilityThreshold),
    },
  ].filter(g => g.players.length > 0);

  return (
    <div style={{maxWidth:1200,margin:'0 auto',padding:'32px 24px'}}>
      {/* Header */}
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:24}}>
        <div style={{display:'flex',alignItems:'center',gap:16}}>
          <TeamLogo code={team} size={52}/>
          <div>
            <h1 style={{fontSize:26,fontWeight:700,letterSpacing:'-0.02em',marginBottom:4}}>Team Deployment Report</h1>
            <p style={{fontSize:13,color:'#888'}}>Quality and durability profiles for each skater · 5v5</p>
          </div>
        </div>
        <select value={team} onChange={e=>onTeamChange(e.target.value)} style={{minWidth:120}}>
          {TEAMS.map(t=><option key={t} value={t}>{t}</option>)}
        </select>
      </div>

      {/* KPI row */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12,marginBottom:24}}>
        <KPICard label="Players in sample" value={roster.length} accent="#1d4ed8" sub={`Minimum ${MIN_QUALIFYING_SHIFTS} shifts over ${MIN_QUALIFYING_SHIFT_SECONDS}s`}/>
        <KPICard label="Overuse-risk skaters" value={flags} sub="Bottom 15% durability among high-TOI players" accent={flags>0?'#dc2626':'#16a34a'}/>
        <KPICard label="Team avg durability" value={avgD} accent="#8b5cf6"/>
        <KPICard label="Team avg xGD/60" value={avgOverall} accent="#0891b2"/>
      </div>

      {/* Main two-column section */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:20,marginBottom:20}}>

        {/* Quality vs Durability scatter */}
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px',position:'relative'}}>
          <SectionHeader title="Quality vs. Durability" sub="Each dot = one player. Horizontal split = league average xGD/60. Vertical split = league average durability. Hover to identify, click to open profile."/>
          <div style={{fontSize:11,color:'#888',marginBottom:10,display:'flex',gap:16}}>
            <span><b style={{color:'#1d4ed8'}}>X →</b> Durability (right = above league average)</span>
            <span><b style={{color:'#1d4ed8'}}>Y ↑</b> Overall xGD/60 (up = above league average)</span>
          </div>
          <div style={{fontSize:10,color:'#94a3b8',fontFamily:'IBM Plex Mono',marginBottom:8,display:'flex',gap:16,flexWrap:'wrap'}}>
            <span>league avg {qualityThreshold.toFixed(2)}</span>
            <span>league dur {durabilityThreshold.toFixed(2)}</span>
          </div>
          <div style={{position:'relative',display:'inline-block'}}>
            <svg width={scW} height={scH} style={{cursor:'crosshair'}} onMouseLeave={()=>setTip(null)}>
              <g transform={`translate(${sm.left},${sm.top})`}>
                {/* quadrant shading */}
                <rect x={sxS(durabilityThreshold)} y={0} width={sW-sxS(durabilityThreshold)} height={syS(qualityThreshold)} fill="#f0fdf4" opacity="0.6"/>
                <rect x={0} y={0} width={sxS(durabilityThreshold)} height={syS(qualityThreshold)} fill="#fffbeb" opacity="0.5"/>
                <rect x={sxS(durabilityThreshold)} y={syS(qualityThreshold)} width={sW-sxS(durabilityThreshold)} height={sH-syS(qualityThreshold)} fill="#f9fafb" opacity="0.4"/>
                <rect x={0} y={syS(qualityThreshold)} width={sxS(durabilityThreshold)} height={sH-syS(qualityThreshold)} fill="#fff5f5" opacity="0.5"/>
                {/* split lines */}
                <line x1={0} y1={syS(qualityThreshold)} x2={sW} y2={syS(qualityThreshold)} stroke="#94a3b8" strokeWidth="1.2" strokeDasharray="6 4"/>
                <line x1={sxS(durabilityThreshold)} y1={0} x2={sxS(durabilityThreshold)} y2={sH} stroke="#94a3b8" strokeWidth="1.2" strokeDasharray="6 4"/>
                {/* quadrant labels */}
                {quadrants.map(q=>(
                  <g key={q.label}>
                    <rect
                      x={q.anchor === 'end' ? q.x - 118 : q.x - 4}
                      y={q.y - 11}
                      width={122}
                      height={16}
                      rx={4}
                      fill={q.bg}
                    />
                    <text x={q.x} y={q.y} fontSize="9" fill={q.color} opacity="0.85"
                      fontFamily="IBM Plex Sans, sans-serif" fontWeight="600" textAnchor={q.anchor}>{q.label}</text>
                  </g>
                ))}
                {/* dots */}
                {roster.map(p=>(
                  <circle key={p.id}
                    cx={sxS(p.drop)} cy={syS(p.overall_xgd)} r={5}
                    fill={p.flagged?'#dc2626':'#1d4ed8'} opacity="0.75" stroke="#fff" strokeWidth="1.2"
                    style={{cursor:'pointer'}}
                    onClick={()=>openPlayer(p)}
                    onMouseEnter={e=>handleScatterMove(e,p)}
                    onMouseMove={e=>handleScatterMove(e,p)}
                    onMouseLeave={()=>setTip(null)}/>
                ))}
                {/* axes */}
                {[xMin+(xMax-xMin)*0.25, xMin+(xMax-xMin)*0.5, xMin+(xMax-xMin)*0.75].map((v,i)=>(
                  <text key={i} x={sxS(v)} y={sH+18} textAnchor="middle" fontSize="9" fill="#bbb" fontFamily="IBM Plex Mono">{v.toFixed(4)}</text>
                ))}
                <text x={sW/2} y={sH+36} textAnchor="middle" fontSize="11" fill="#888">Durability</text>
                <text x={-42} y={sH/2} textAnchor="middle" fontSize="11" fill="#888" transform={`rotate(-90,-42,${sH/2})`}>Overall xGD/60</text>
                {[yMin+(yMax-yMin)*0.25, qualityThreshold, yMin+(yMax-yMin)*0.75].map((v,i)=>(
                  <text key={i} x={-6} y={syS(v)+4} textAnchor="end" fontSize="9" fill="#bbb" fontFamily="IBM Plex Mono">{v.toFixed(3)}</text>
                ))}
              </g>
            </svg>
            {tip && (
              <div style={{position:'absolute',left:tip.x+12,top:tip.y-10,background:'#1a1a2e',color:'#fff',
                borderRadius:6,padding:'8px 11px',fontSize:12,pointerEvents:'none',whiteSpace:'nowrap',
                boxShadow:'0 4px 16px rgba(0,0,0,0.25)',zIndex:999}}>
                <div style={{fontWeight:600,marginBottom:2}}>{tip.p.name} <span style={{opacity:0.5,fontWeight:400}}>({tip.p.pos})</span></div>
                <div style={{fontFamily:'IBM Plex Mono',fontSize:11,color:'#a5b4fc'}}>xGD <span style={{color:'#fff'}}>{tip.p.overall_xgd.toFixed(2)}</span>  Durability <span style={{color:'#fff'}}>{tip.p.drop.toFixed(2)}</span></div>
                <div style={{fontSize:10,color:'#94a3b8',marginTop:2}}>TOI {tip.p.toi_min} min · Click to open profile</div>
              </div>
            )}
          </div>
          <div style={{fontSize:10,color:'#bbb',marginTop:6}}><span style={{color:'#dc2626'}}>Red</span> = overuse-flagged.</div>
        </div>

        {/* Deployment guide */}
        <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
          <SectionHeader title="Deployment Guide" sub="Players grouped by shift-fade quadrant. Click any name to open profile."/>
          <div style={{marginTop:8,display:'flex',flexDirection:'column',gap:12,maxHeight:320,overflowY:'auto'}}>
            {deployGroups.map(g=>(
              <div key={g.label} style={{background:g.bg,borderRadius:6,padding:'10px 12px',border:`1px solid ${g.color}22`}}>
                <div style={{display:'flex',alignItems:'baseline',gap:8,marginBottom:4}}>
                  <span style={{fontSize:11,fontWeight:700,color:g.color,textTransform:'uppercase',letterSpacing:'0.06em'}}>{g.label}</span>
                  <span style={{fontSize:11,color:'#888'}}>{g.advice}</span>
                </div>
                <div style={{display:'flex',flexWrap:'wrap',gap:'4px 10px'}}>
                  {g.players.sort((a,b)=>b.overall_xgd-a.overall_xgd).map(p=>(
                    <button key={p.id} onClick={()=>openPlayer(p)}
                      style={{fontSize:12,fontWeight:600,color:'#1d4ed8',background:'none',border:'none',padding:0,cursor:'pointer'}}>
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Roster table */}
      <div style={{background:'#fff',border:'1px solid #e8e8e8',borderRadius:6,padding:'20px'}}>
        <SectionHeader title="Roster Breakdown" sub="Click a player name to open their full profile."/>
        <table>
          <thead><tr>
            <th style={{textAlign:'left'}}><button onClick={()=>onSort('name')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Player{sortArrow('name')}</button></th>
            <th><button onClick={()=>onSort('pos')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Pos{sortArrow('pos')}</button></th>
            <th><button onClick={()=>onSort('overall_xgd')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Overall xGD{sortArrow('overall_xgd')}</button></th>
            <th><button onClick={()=>onSort('drop')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Durability{sortArrow('drop')}</button></th>
            <th><button onClick={()=>onSort('early')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Early xGD{sortArrow('early')}</button></th>
            <th><button onClick={()=>onSort('late')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Late xGD{sortArrow('late')}</button></th>
            <th><button onClick={()=>onSort('toi_min')} style={{fontSize:12,fontWeight:600,color:'#666'}}>TOI (min){sortArrow('toi_min')}</button></th>
            <th><button onClick={()=>onSort('stints')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Shifts{sortArrow('stints')}</button></th>
            <th><button onClick={()=>onSort('flagged')} style={{fontSize:12,fontWeight:600,color:'#666'}}>Flagged{sortArrow('flagged')}</button></th>
          </tr></thead>
          <tbody>{sortedRoster.map(p=>(
            <tr key={p.id}>
              <td><button onClick={()=>openPlayer(p)} style={{color:'#1d4ed8',fontWeight:600,fontSize:13}}>{p.name}</button></td>
              <td style={{textAlign:'center',color:'#888',fontSize:12}}>{p.pos}</td>
              <PctCell pct={p.pct_overall}  value={fmt(p.overall_xgd,2)}/>
              <PctCell pct={p.pct_durability} value={fmt(p.drop,2)}/>
              <PctCell pct={p.pct_early} value={fmt(p.early)}/>
              <PctCell pct={p.pct_late}  value={fmt(p.late)}/>
              <td style={{fontFamily:'IBM Plex Mono',fontSize:12}}>{p.toi_min}</td>
              <td style={{fontFamily:'IBM Plex Mono',fontSize:12}}>{p.stints}</td>
              <td style={{textAlign:'center'}}>{p.flagged?<Tag bg="#fee2e2" color="#dc2626">⚠ Yes</Tag>:<span style={{color:'#aaa',fontSize:12}}>N/A</span>}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
};

// ── APP ────────────────────────────────────────────────────────────────────────
const App = () => {
  const [route, setRoute] = useState(() => normalizeRouteState(parseRouteState()));
  const selectedPlayer = useMemo(
    () => PLAYERS.find(p => p.id === route.playerId) || null,
    [route.playerId]
  );

  useEffect(() => {
    const onPopState = () => setRoute(normalizeRouteState(parseRouteState()));
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    window.history.replaceState(route, '', buildRouteUrl(route));
  }, []);

  const commitRoute = (nextRoute, {replace=false, reload=false} = {}) => {
    const normalized = normalizeRouteState(nextRoute);
    if (reload || normalized.season !== window.__DATA__.current_season) {
      window.location.href = buildRouteUrl(normalized);
      return;
    }
    setRoute(normalized);
    const method = replace ? 'replaceState' : 'pushState';
    window.history[method](normalized, '', buildRouteUrl(normalized));
  };

  const onSeasonChange = (season) => {
    commitRoute({...route, season}, {reload:true});
  };

  const goPage = (page) => {
    commitRoute({
      ...route,
      page,
      playerId: page === 'profile' ? route.playerId : null,
    });
  };

  const openPlayer = (player) => {
    if (!player) return;
    commitRoute({
      ...route,
      page: 'profile',
      playerId: player.id,
      team: player.team || route.team,
    });
  };

  const setTeam = (team) => {
    commitRoute({
      ...route,
      page: 'team',
      team,
      playerId: null,
    });
  };

  const breadcrumbs = {
    overview: [{label:'League Overview'}],
    players:  [{label:'League Overview', onClick:()=>goPage('overview')},{label:'Players'}],
    profile:  [{label:'League Overview',onClick:()=>goPage('overview')},{label:'Players',onClick:()=>goPage('players')},{label:selectedPlayer?.name||'Profile'}],
    lines:    [{label:'League Overview',onClick:()=>goPage('overview')},{label:'Lines'}],
    team:     [{label:'League Overview',onClick:()=>goPage('overview')},{label:'Team Report'}],
  };

  return (
    <div style={{minHeight:'100vh',background:'#f8f8f7'}}>
      <TopNav page={route.page==='profile'?'players':route.page} setPage={goPage} season={route.season} onSeasonChange={onSeasonChange} breadcrumb={breadcrumbs[route.page]}/>
      {route.page==='overview' && <OverviewPage openPlayer={openPlayer}/>}
      {route.page==='players'  && <PlayersPage openPlayer={openPlayer}/>}
      {route.page==='profile'  && <PlayerProfile player={selectedPlayer} setPage={goPage}/>}
      {route.page==='lines'    && <LinesPage openPlayer={openPlayer}/>}
      {route.page==='team'     && <TeamReportPage selectedTeam={route.team} onTeamChange={setTeam} openPlayer={openPlayer}/>}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
