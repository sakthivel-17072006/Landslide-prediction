"""MILESTONE 3 - GIS landslide-susceptibility zone map (like national landslide-class maps).
Run: python m3_gis_map.py  ->  output/risk_map.html (open in browser) + output/risk_zones.png
Uses the REAL Mizoram boundary (data/mizoram.geojson). Terrain/rainfall fields are SYNTHETIC until you add real rasters."""
import json, zlib, struct, base64, joblib
from scipy.ndimage import gaussian_filter, zoom
from common import *

best = joblib.load(ROOT / "models/landslide_model.joblib"); best_name = (ROOT / "models/best_name.txt").read_text()
gj = json.loads((ROOT / "data/mizoram.geojson").read_text()); geom = gj["features"][0]["geometry"]
rings = [np.array(p[0]) for p in ([geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"])]
lon0, lat0 = np.vstack(rings).min(0); lon1, lat1 = np.vstack(rings).max(0)

def inside(x, y):                                   # point-in-polygon (ray casting), vectorised
    res = np.zeros(x.shape, bool)
    for r in rings:
        ins = np.zeros(x.shape, bool)
        for i in range(len(r)):
            (x1, y1), (x2, y2) = r[i], r[(i + 1) % len(r)]
            ins ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        res |= ins
    return res

NX, NY, UP = 110, 160, 4                            # model grid and display upscale
LON, LAT = np.meshgrid(np.linspace(lon0, lon1, NX), np.linspace(lat1, lat0, NY))   # row 0 = north
mask = inside(LON, LAT)

# ---- synthetic but spatially-coherent inputs (heavy-rain scenario) ----
rng = np.random.default_rng(11)
def field(sy, sx):
    z = gaussian_filter(rng.normal(size=(NY, NX)), (sy, sx)); return (z - z.mean()) / z.std()
elev = (900 + 420 * field(9, 4) + 120 * field(3, 3)).clip(100, 2150)           # N-S ridges
cy, cx = (lat1 - lat0) * 111000 / NY, (lon1 - lon0) * 111000 * np.cos(np.radians(23.3)) / NX
gy, gx = np.gradient(elev, cy, cx); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
slope = (slope * 22 / slope[mask].mean()).clip(0, 65)                           # rescaled to Mizoram-like mean slope
rain = (200 + 55 * field(12, 10)).clip(40, 420)
moist = (25 + .14 * rain + 5 * field(4, 4)).clip(5, 100)
ndvi = (.62 + .10 * field(5, 5) - .004 * (slope - 22)).clip(.15, .9)
prev = (field(3, 3) > 1.0).astype(int)

X = pd.DataFrame({"rainfall": rain[mask], "soil_moisture": moist[mask], "slope": slope[mask],
                  "elevation": elev[mask], "ndvi": ndvi[mask], "prev_landslide": prev[mask]})[FEATURES]
P = np.full(mask.shape, np.nan); P[mask] = best.predict_proba(X)[:, 1]

# ---- smooth high-resolution zone image ----
Pf = np.where(mask, P, np.nanmean(P))
Phi = gaussian_filter(zoom(Pf, UP, order=1), UP / 2)
LONh, LATh = np.meshgrid(np.linspace(lon0, lon1, NX * UP), np.linspace(lat1, lat0, NY * UP))
mh = inside(LONh, LATh)
COL = {"VERY LOW": "#2f7fc1", "LOW": "#a8dba0", "MEDIUM": "#ffd400", "HIGH": "#f4a259", "VERY HIGH": "#d62828"}
rgb = np.array([[int(COL[c][i:i + 2], 16) for i in (1, 3, 5)] for c in CLASSES], np.uint8)
img = np.zeros(Phi.shape + (4,), np.uint8)
img[..., :3] = rgb[np.digitize(Phi, [.2, .4, .6, .8])]; img[..., 3] = np.where(mh, 235, 0)
def png(a):
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(a.shape[0]))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 8, 6, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(raw, 9)) + ch(b"IEND", b""))
png_bytes = png(img); (ROOT / "output/risk_zones.png").write_bytes(png_bytes)

# ---- tables / stats ----
cls = np.digitize(P[mask], [.2, .4, .6, .8]); share = np.bincount(cls, minlength=5) / len(cls)
badge = lambda l: f'<span class="b" style="background:{COL[l]};color:{"#222" if l in ("LOW","MEDIUM") else "#fff"}">{l}</span>'
cards = "".join(f'<div class="c" style="border-top:4px solid {COL[l]}"><b>{share[i]:.0%}</b><span>{l}</span></div>' for i, l in enumerate(CLASSES))
cells = pd.DataFrame({"lat": LAT[mask], "lon": LON[mask], "p": P[mask]})
cells["town"] = [nearest_town(a, b) for a, b in zip(cells.lat, cells.lon)]
picked = []
for r in cells.sort_values("p", ascending=False).itertuples():
    if all((r.lat - q.lat) ** 2 + (r.lon - q.lon) ** 2 > .12 ** 2 for q in picked): picked.append(r)
    if len(picked) == 10: break
top_rows = "".join(f"<tr><td>{i+1}</td><td>{r.town}</td><td>{r.lat:.2f}, {r.lon:.2f}</td><td>{r.p:.0%}</td><td>{badge(risk_level(r.p))}</td></tr>" for i, r in enumerate(picked))
tw = cells.groupby("town").p.mean().sort_values(ascending=False)
town_rows = "".join(f"<tr><td>{t}</td><td>{v:.0%}</td><td>{badge(risk_level(v))}</td></tr>" for t, v in tw.items())
q = lambda a: [(-1 if np.isnan(v) else int(round(v))) for v in a.ravel()]
data = dict(nx=NX, ny=NY, b=[lon0, lat0, lon1, lat1], p=q(P * 100), r=q(np.where(mask, rain, np.nan)),
            s=q(np.where(mask, slope, np.nan)), e=q(np.where(mask, elev, np.nan)))

html = """<!doctype html><html><head><meta charset=utf-8><title>Mizoram Landslide Susceptibility</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<link rel=stylesheet href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;background:#f3f5f8;color:#1c2430}
header{background:#1f3a5f;color:#fff;padding:18px 24px}h1{margin:0 0 4px;font-size:22px}header p{margin:2px 0;opacity:.9}
.w{padding:16px 24px;display:grid;gap:16px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.c{background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 3px #0002}.c b{display:block;font-size:24px}.c span{font-size:12px}
.main{display:grid;grid-template-columns:2fr 1fr;gap:16px}#m{height:680px;border-radius:8px;background:#fff}
.p{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px #0002;margin-bottom:16px;overflow-x:auto}
h2{margin:0 0 8px;font-size:15px}table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:5px 6px;border-bottom:1px solid #e5e9ef;text-align:left}.b{padding:2px 7px;border-radius:10px;font-size:11px;white-space:nowrap}
.lg{background:#fff;padding:8px 12px;border-radius:6px;font-size:12px;box-shadow:0 1px 4px #0003}.lg i{display:inline-block;width:14px;height:12px;margin-right:6px}
small{color:#5b6675}@media(max-width:850px){.main{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}</style></head><body>
<header><h1>Landslide Susceptibility Map &ndash; Mizoram, North Eastern India</h1>
<p><b>Area predicted:</b> Mizoram state (__BBOX__), ~__NCELL__ grid cells inside the state boundary</p>
<p><b>Scenario:</b> heavy rainfall &nbsp;|&nbsp; <b>Model:</b> __MODEL__ &nbsp;|&nbsp; <b>Inputs:</b> rainfall, soil moisture, slope, elevation, NDVI, past landslides</p></header>
<div class="w"><div><small>Share of Mizoram area in each landslide class:</small><div class="cards" style="margin-top:6px">__CARDS__</div></div>
<div class="main"><div id="m"></div><div>
<div class="p"><h2>Top 10 highest-risk locations</h2><table><tr><th>#</th><th>Nearest town</th><th>Lat, Lon</th><th>Risk</th><th>Class</th></tr>__TOP__</table></div>
<div class="p"><h2>Average risk by nearest town</h2><table><tr><th>Town</th><th>Avg risk</th><th>Class</th></tr>__TOWNS__</table></div></div></div>
<small>Boundary is real; terrain, rainfall and landslide labels are SYNTHETIC demo data. Risk = predicted landslide probability: Very Low &lt;20%, Low 20&ndash;40%, Medium 40&ndash;60%, High 60&ndash;80%, Very High &gt;80%. Click the map for values at that spot.</small></div>
<script>const D=__DATA__,C=__COL__,T=__TOWNS_JS__,CL=__CLASSES__,B=[[D.b[1],D.b[0]],[D.b[3],D.b[2]]];
const m=L.map('m').setView([23.3,92.85],8);
const streets=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap',maxZoom:19}).addTo(m);
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{attribution:'Tiles &copy; Esri',maxZoom:18});
const terr=L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenTopoMap',maxZoom:17});
const plain=L.layerGroup();
const zones=L.imageOverlay('data:image/png;base64,__PNG__',B,{opacity:.88,interactive:false}).addTo(m);
const edge=L.geoJSON(__GEO__,{style:{color:'#222',weight:1.6,fill:false,interactive:false}}).addTo(m);
const towns=L.layerGroup(Object.entries(T).map(([n,c])=>L.circleMarker(c,{radius:4,color:'#000',fillColor:'#fff',fillOpacity:1,weight:1.5}).bindTooltip(n,{permanent:true,direction:'right',offset:[6,0]}))).addTo(m);
L.control.layers({'Streets':streets,'Satellite':sat,'Terrain':terr,'Plain (white)':plain},{'Risk zones':zones,'State boundary':edge,'Towns':towns},{position:'topright'}).addTo(m);
L.control.scale().addTo(m);m.fitBounds(B);
const cls=p=>p<20?0:p<40?1:p<60?2:p<80?3:4;
m.on('click',e=>{const x=Math.floor((e.latlng.lng-D.b[0])/(D.b[2]-D.b[0])*(D.nx-1)+.5),y=Math.floor((D.b[3]-e.latlng.lat)/(D.b[3]-D.b[1])*(D.ny-1)+.5);
 if(x<0||y<0||x>=D.nx||y>=D.ny)return;const i=y*D.nx+x;if(D.p[i]<0)return;
 L.popup().setLatLng(e.latlng).setContent(`<b style="color:${C[CL[cls(D.p[i])]]}">${CL[cls(D.p[i])]}</b> (${D.p[i]}%)<br>Rainfall: ${D.r[i]} mm<br>Slope: ${D.s[i]}&deg;<br>Elevation: ${D.e[i]} m`).openOn(m)});
const lg=L.control({position:'bottomright'});lg.onAdd=()=>{const d=L.DomUtil.create('div','lg');
d.innerHTML='<b>Landslide Class</b><br>'+[...CL].reverse().map(k=>`<i style="background:${C[k]}"></i>${k}`).join('<br>');return d};lg.addTo(m);
</script></body></html>"""
for k, v in {"__BBOX__": f"{lat0:.2f}&deg;N&ndash;{lat1:.2f}&deg;N, {lon0:.2f}&deg;E&ndash;{lon1:.2f}&deg;E", "__NCELL__": f"{int(mask.sum()):,}",
             "__MODEL__": best_name, "__CARDS__": cards, "__TOP__": top_rows, "__TOWNS__": town_rows, "__DATA__": json.dumps(data),
             "__COL__": json.dumps(COL), "__TOWNS_JS__": json.dumps({k: list(v) for k, v in TOWNS.items()}),
             "__CLASSES__": json.dumps(CLASSES), "__PNG__": base64.b64encode(png_bytes).decode(), "__GEO__": json.dumps(gj)}.items():
    html = html.replace(k, v)
(ROOT / "output/risk_map.html").write_text(html, encoding="utf-8")
print("Class share:", {c: f"{share[i]:.0%}" for i, c in enumerate(CLASSES)})
print("Map written: output/risk_map.html")
